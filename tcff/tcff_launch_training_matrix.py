"""Launch explicitly assigned independent single-GPU training runs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import threading
import time


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding='utf-8')
    tmp.replace(path)


def parse_assignments(text):
    jobs = []
    for item in text.split(','):
        gpu, spec = item.strip().split('=', 1)
        variant, seed = spec.split(':', 1)
        jobs.append({'gpu': gpu, 'variant': variant, 'seed': int(seed)})
    return jobs


def main(args):
    root = Path(args.project_root).resolve()
    results = Path(args.results_root).resolve()
    jobs = parse_assignments(args.assignments)
    state_dir = results / '02_manifests'
    state_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / f'{args.manifest_prefix}_{args.node_tag}.json'
    state = {'node_tag': args.node_tag, 'jobs': jobs, 'completed': [],
             'failed': [], 'started_at_epoch': time.time()}
    lock = threading.Lock()
    atomic_json(state_path, state)

    def run_job(job):
        run_id = f"{args.run_prefix}{job['variant']}_seed{job['seed']}"
        run_dir = results / '03_raw_runs' / args.experiment_id / run_id
        if run_dir.exists():
            with lock:
                state['failed'].append({**job, 'reason': 'run_dir_exists'})
                atomic_json(state_path, state)
            return
        command = [
            args.python, str(root / 'tcff_train_wrapper.py'),
            '--project_root', str(root),
            '--results_root', str(results),
            '--experiment_id', args.experiment_id,
            '--run_id', run_id,
            '--trust_variant', job['variant'],
            '--seed', str(job['seed']),
            '--epochs', str(args.epochs),
            '--batch_size', str(args.batch_size),
            '--num_workers', str(args.num_workers),
        ]
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = job['gpu']
        env['PYTHONPATH'] = f"{root / 'SAM-Med3D'}{os.pathsep}{root}"
        env['NUMEXPR_MAX_THREADS'] = '64'
        env['OMP_NUM_THREADS'] = '8'
        conda_lib = '/root/miniconda3/envs/cu122py39/lib'
        env['LD_LIBRARY_PATH'] = conda_lib + ':' + env.get('LD_LIBRARY_PATH', '')
        launcher_log = results / '10_logs' / f"train_launcher_{run_id}.log"
        launcher_log.parent.mkdir(parents=True, exist_ok=True)
        with open(launcher_log, 'w', encoding='utf-8', buffering=1) as log:
            process = subprocess.run(command, cwd=root, env=env,
                                     stdout=log, stderr=subprocess.STDOUT,
                                     check=False)
        record = {**job, 'run_id': run_id, 'exit_code': process.returncode,
                  'finished_at_epoch': time.time()}
        with lock:
            target = state['completed'] if process.returncode == 0 else state['failed']
            target.append(record)
            atomic_json(state_path, state)
            print(json.dumps(record, sort_keys=True), flush=True)

    threads = [threading.Thread(target=run_job, args=(job,), daemon=False)
               for job in jobs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    state['finished_at_epoch'] = time.time()
    atomic_json(state_path, state)
    print(json.dumps({'completed': len(state['completed']),
                      'failed': len(state['failed'])}, sort_keys=True))
    return 0 if not state['failed'] else 2


if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_root', default=str(here))
    parser.add_argument('--results_root', default=str(here / 'results_tcff_rebuttal'))
    parser.add_argument('--python', default='/root/autodl-tmp/tcff-venv/bin/python')
    parser.add_argument('--node_tag', required=True)
    parser.add_argument('--assignments', required=True,
                        help='comma list, e.g. 0=full:0,1=u_only:0')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--experiment_id', default='P0_D_TRAIN_VARIANTS')
    parser.add_argument('--run_prefix', default='')
    parser.add_argument('--manifest_prefix', default='p0_training_matrix')
    raise SystemExit(main(parser.parse_args()))
