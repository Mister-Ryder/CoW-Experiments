"""Audited wrapper for one independent TCFF training-variant run."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch
import yaml

from tcff_rebuttal_runner import (
    canonical_json,
    git_commit,
    sha256_bytes,
    sha256_file,
    split_hash,
    utc_now,
    write_json,
)
from topcow_foundation_model import TRUST_VARIANTS


def run(args):
    project_root = Path(args.project_root).resolve()
    results_root = Path(args.results_root).resolve()
    os.chdir(project_root)
    run_id = args.run_id or (
        f"{args.trust_variant}_seed{args.seed}_"
        f"{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}")
    run_dir = results_root / '03_raw_runs' / args.experiment_id / run_id
    if run_dir.exists():
        raise FileExistsError(run_dir)
    run_dir.mkdir(parents=True)
    train_parent = run_dir / 'training_output'
    train_parent.mkdir()

    foundation_ckpt = Path(args.foundation_ckpt).resolve()
    train_dir = Path(args.train_dir).resolve()
    test_dir = Path(args.test_dir).resolve()
    config = vars(args).copy()
    config.update({
        'run_id': run_id,
        'foundation_ckpt': str(foundation_ckpt),
        'train_dir': str(train_dir),
        'test_dir': str(test_dir),
        'training_protocol': 'independent_single_gpu_fixed_schedule',
    })
    config_hash = sha256_bytes(canonical_json(config).encode())
    with open(run_dir / 'run_config.yaml', 'w', encoding='utf-8') as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=True)
    manifest = {
        'experiment_id': args.experiment_id,
        'run_id': run_id,
        'git_commit': git_commit(project_root),
        'config_hash': config_hash,
        'checkpoint_hash': sha256_file(foundation_ckpt),
        'checkpoint_role': 'frozen_foundation_initialization',
        'data_split_hash': split_hash(train_dir),
        'dataset': 'TopCoW_train+TopCoW_val',
        'model_variant': args.trust_variant,
        'corruption': {'type': 'none', 'severity': 0, 'unit': 'none'},
        'seed': args.seed,
        'started_at': utc_now(),
        'finished_at': None,
        'status': 'running',
        'failure_reason': None,
    }
    write_json(run_dir / 'run_manifest.json', manifest)
    write_json(run_dir / 'status.json', {'status': 'running', 'started_at': utc_now()})

    command = [
        sys.executable, str(project_root / 'topcow_train_foundation.py'),
        '--exp_name', 'model',
        '--foundation_type', 'sam3d_turbo',
        '--fusion_type', 'tgf',
        '--trust_variant', args.trust_variant,
        '--foundation_ckpt', str(foundation_ckpt),
        '--seed', str(args.seed),
        '--train_dir', str(train_dir),
        '--test_dir', str(test_dir),
        '--output_dir', str(train_parent),
        '--epochs', str(args.epochs),
        '--batch_size', str(args.batch_size),
        '--lr', str(args.lr),
        '--num_workers', str(args.num_workers),
        '--no_save_last',
    ]
    write_json(run_dir / 'command.json', command)
    started = time.perf_counter()
    exit_code = None
    error = None
    with open(run_dir / 'stdout.log', 'w', encoding='utf-8', buffering=1) as stdout, \
            open(run_dir / 'stderr.log', 'w', encoding='utf-8', buffering=1) as stderr:
        try:
            process = subprocess.run(command, cwd=project_root,
                                     stdout=stdout, stderr=stderr, check=False)
            exit_code = process.returncode
        except Exception as exc:
            error = f'{type(exc).__name__}: {exc}'
            stderr.write(traceback.format_exc())

    elapsed = time.perf_counter() - started
    model_dir = train_parent / 'model'
    best = model_dir / 'best_model.pth'
    success = exit_code == 0 and best.exists()
    final_hash = sha256_file(best) if best.exists() else None
    runtime = {
        'wall_time_sec': elapsed,
        'exit_code': exit_code,
        'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'final_checkpoint': str(best),
        'final_checkpoint_hash': final_hash,
    }
    write_json(run_dir / 'runtime.json', runtime)
    status = 'success' if success else 'failed'
    reason = None if success else (error or f'exit_code={exit_code}, best_exists={best.exists()}')
    manifest.update({
        'finished_at': utc_now(),
        'status': status,
        'failure_reason': reason,
        'trained_checkpoint_hash': final_hash,
    })
    write_json(run_dir / 'run_manifest.json', manifest)
    write_json(run_dir / 'status.json', {
        'status': status,
        'finished_at': manifest['finished_at'],
        'failure_reason': reason,
    })
    print(f'RUN_DIR={run_dir}')
    print(f'STATUS={status} elapsed_sec={elapsed:.1f}')
    return 0 if success else 1


def parser():
    here = Path(__file__).resolve().parent
    p = argparse.ArgumentParser()
    p.add_argument('--project_root', default=str(here))
    p.add_argument('--results_root', default=str(here / 'results_tcff_rebuttal'))
    p.add_argument('--experiment_id', default='P0_D_TRAIN_VARIANTS')
    p.add_argument('--run_id')
    p.add_argument('--trust_variant', required=True, choices=TRUST_VARIANTS)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--foundation_ckpt', default=str(here / 'sam_med3d_turbo.pth'))
    p.add_argument('--train_dir', default=str(here / 'TopCoW_processed'))
    p.add_argument('--test_dir', default=str(here / 'crown_processed'))
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--num_workers', type=int, default=4)
    return p


if __name__ == '__main__':
    raise SystemExit(run(parser().parse_args()))
