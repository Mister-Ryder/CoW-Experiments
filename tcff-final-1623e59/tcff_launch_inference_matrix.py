"""Launch the preregistered fixed-checkpoint P0 matrix on independent GPUs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time


DATASETS = (
    ('crown', 'TopCoW_val', 'crown_processed'),
    ('lausanne', 'Lausanne', 'lausanne_processed'),
    ('ixi', 'IXI_HH', 'ixi_processed'),
)

CORRUPTIONS = (
    ('P0_C1_GAUSSIAN', 'gaussian', (0.05, 0.10, 0.20, 0.40), 'channel_std_multiplier'),
    ('P0_C1_CHANNEL_DROPOUT', 'channel_dropout', (0.10, 0.25, 0.50, 0.75), 'channel_fraction'),
    ('P0_C2_TRANSLATION', 'translation', (1.0, 2.0, 4.0, 8.0), 'feature_voxel'),
    ('P0_C3_DONOR_MIX', 'donor_mix', (0.10, 0.25, 0.50, 0.75), 'mixing_weight'),
)

FIXED_VARIANTS = ('no_aux', 'residual', 'u_only', 'g_only', 'additive')


def severity_tag(value):
    return f'{value:g}'.replace('.', 'p')


def build_jobs(root):
    jobs = []
    for tag, label, directory in DATASETS:
        jobs.append({
            'experiment_id': 'P0_AB_CLEAN_MECHANISM',
            'run_id': f'full_clean_{tag}',
            'dataset_tag': tag,
            'dataset_name': label,
            'dataset_dir': str(root / directory),
            'variant': 'full',
            'corruption': 'none',
            'severity': 0.0,
            'unit': 'none',
            'realization': 0,
            'save_predictions': True,
        })
        for variant in FIXED_VARIANTS:
            jobs.append({
                'experiment_id': 'P0_D_FIXED_CKPT_INTERVENTIONS',
                'run_id': f'{variant}_clean_{tag}',
                'dataset_tag': tag,
                'dataset_name': label,
                'dataset_dir': str(root / directory),
                'variant': variant,
                'corruption': 'none',
                'severity': 0.0,
                'unit': 'none',
                'realization': 0,
                'save_predictions': False,
            })
        for experiment_id, corruption, severities, unit in CORRUPTIONS:
            for severity in severities:
                for realization in range(3):
                    jobs.append({
                        'experiment_id': experiment_id,
                        'run_id': (f'full_{corruption}_{severity_tag(severity)}_'
                                   f'r{realization}_{tag}'),
                        'dataset_tag': tag,
                        'dataset_name': label,
                        'dataset_dir': str(root / directory),
                        'variant': 'full',
                        'corruption': corruption,
                        'severity': severity,
                        'unit': unit,
                        'realization': realization,
                        'save_predictions': False,
                    })
    return jobs


def build_product_jobs(root):
    """Matched fixed-checkpoint intervention with G(1-U) truly executed."""
    jobs = []
    for tag, label, directory in DATASETS:
        jobs.append({
            'experiment_id': 'P0_D_PRODUCT_FIXED_CLEAN',
            'run_id': f'product_clean_{tag}',
            'dataset_tag': tag,
            'dataset_name': label,
            'dataset_dir': str(root / directory),
            'variant': 'product',
            'corruption': 'none',
            'severity': 0.0,
            'unit': 'none',
            'realization': 0,
            'save_predictions': False,
        })
        for _, corruption, severities, unit in CORRUPTIONS:
            for severity in severities:
                for realization in range(3):
                    jobs.append({
                        'experiment_id': 'P0_D_PRODUCT_FIXED_CORRUPTION',
                        'run_id': (f'product_{corruption}_{severity_tag(severity)}_'
                                   f'r{realization}_{tag}'),
                        'dataset_tag': tag,
                        'dataset_name': label,
                        'dataset_dir': str(root / directory),
                        'variant': 'product',
                        'corruption': corruption,
                        'severity': severity,
                        'unit': unit,
                        'realization': realization,
                        'save_predictions': False,
                    })
    return jobs


def atomic_json(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True), encoding='utf-8')
    tmp.replace(path)


def main(args):
    root = Path(args.project_root).resolve()
    results = Path(args.results_root).resolve()
    matrix_name = ('p0_inference_matrix' if args.matrix == 'core'
                   else 'p0_product_intervention_matrix')
    logs = results / '10_logs' / 'inference_matrix' / args.matrix
    manifests = results / '02_manifests'
    logs.mkdir(parents=True, exist_ok=True)
    manifests.mkdir(parents=True, exist_ok=True)
    jobs = build_jobs(root) if args.matrix == 'core' else build_product_jobs(root)
    if args.limit is not None:
        jobs = jobs[:args.limit]
    plan_path = manifests / f'{matrix_name}_plan.json'
    status_path = manifests / f'{matrix_name}_status.json'
    atomic_json(plan_path, {'n_jobs': len(jobs), 'jobs': jobs})

    gpu_ids = [item.strip() for item in args.gpus.split(',') if item.strip()]
    work = queue.Queue()
    for job in jobs:
        run_dir = (results / '03_raw_runs' / job['experiment_id'] / job['run_id'])
        status_file = run_dir / 'status.json'
        if status_file.exists():
            status = json.loads(status_file.read_text()).get('status')
            if status == 'success':
                continue
            # Never overwrite a failed or partial run; preserve it for audit.
            print(f"SKIP existing non-success run (manual retry required): {run_dir}")
            continue
        work.put(job)

    state = {
        'planned': len(jobs),
        'queued': work.qsize(),
        'completed': [],
        'failed': [],
        'started_at_epoch': time.time(),
    }
    lock = threading.Lock()
    atomic_json(status_path, state)

    def worker(gpu):
        while True:
            try:
                job = work.get_nowait()
            except queue.Empty:
                return
            run_dir = results / '03_raw_runs' / job['experiment_id'] / job['run_id']
            log_path = logs / f"{job['experiment_id']}__{job['run_id']}.log"
            command = [
                args.python, str(root / 'tcff_rebuttal_runner.py'),
                '--project_root', str(root),
                '--results_root', str(results),
                '--experiment_id', job['experiment_id'],
                '--run_id', job['run_id'],
                '--exp_dir', str(root / 'Runs_foundation' / 'sam_tgf_v3'),
                '--dataset_dir', job['dataset_dir'],
                '--dataset_name', job['dataset_name'],
                '--model_variant', job['variant'],
                '--corruption_type', job['corruption'],
                '--severity', str(job['severity']),
                '--severity_unit', job['unit'],
                '--realization', str(job['realization']),
                '--seed', '0',
                '--device', 'cuda:0',
            ]
            if job['save_predictions']:
                command.append('--save_predictions')
            env = os.environ.copy()
            env['CUDA_VISIBLE_DEVICES'] = gpu
            env['PYTHONPATH'] = f"{root / 'SAM-Med3D'}{os.pathsep}{root}"
            env['NUMEXPR_MAX_THREADS'] = '64'
            env['OMP_NUM_THREADS'] = '8'
            conda_lib = '/root/miniconda3/envs/cu122py39/lib'
            env['LD_LIBRARY_PATH'] = conda_lib + ':' + env.get('LD_LIBRARY_PATH', '')
            with open(log_path, 'w', encoding='utf-8', buffering=1) as log:
                process = subprocess.run(command, cwd=root, env=env,
                                         stdout=log, stderr=subprocess.STDOUT,
                                         check=False)
            if run_dir.exists():
                (run_dir / 'stdout.log').write_bytes(log_path.read_bytes())
                (run_dir / 'stderr.log').touch(exist_ok=True)
            item = dict(job)
            item.update({'gpu': gpu, 'exit_code': process.returncode,
                         'finished_at_epoch': time.time()})
            with lock:
                if process.returncode == 0:
                    state['completed'].append(item)
                else:
                    state['failed'].append(item)
                state['remaining'] = work.qsize()
                atomic_json(status_path, state)
                print(f"GPU {gpu} exit={process.returncode} {job['run_id']}", flush=True)
            work.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,), daemon=False)
               for gpu in gpu_ids]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    state['finished_at_epoch'] = time.time()
    atomic_json(status_path, state)
    print(json.dumps({
        'planned': state['planned'],
        'queued_this_launch': state['queued'],
        'completed': len(state['completed']),
        'failed': len(state['failed']),
    }, sort_keys=True))
    return 0 if not state['failed'] else 2


if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_root', default=str(here))
    parser.add_argument('--results_root', default=str(here / 'results_tcff_rebuttal'))
    parser.add_argument('--python', default='/root/autodl-tmp/tcff-venv/bin/python')
    parser.add_argument('--gpus', default='0,1')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--matrix', choices=['core', 'product'], default='core')
    raise SystemExit(main(parser.parse_args()))
