"""Audited fixed-checkpoint TCFF rebuttal experiment runner.

The runner never edits input data, submitted checkpoints, or historical metrics.
Every invocation writes a self-contained run under results_tcff_rebuttal.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback

import numpy as np
from scipy import ndimage
from skimage.morphology import skeletonize_3d
import torch
import torch.nn.functional as F
import pandas as pd
import yaml

from topcow_dataset import TopCoWTestDataset
from topcow_eval_official_all import (
    GROUP2_LABELS,
    NUM_CLASSES,
    VESSEL_NAMES,
    aggregate_results,
    evaluate_single_case,
)
from topcow_foundation_model import (
    TRUST_VARIANTS,
    TopCoWFoundationModel,
    build_foundation_encoder,
)


SMALL_VESSEL_CLASSES = {8, 9, 10, 13}
LEVEL_ORDER = ('shallow', 'mid', 'deep')


def utc_now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def canonical_json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'),
                      ensure_ascii=False, default=str)


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    h = hashlib.sha256()
    with open(path, 'rb') as handle:
        while True:
            block = handle.read(chunk_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def stable_seed(*parts):
    payload = '|'.join(map(str, parts)).encode('utf-8')
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], 'little')


def git_commit(project_root):
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=project_root,
            text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        tracked = sorted(Path(project_root).glob('*.py'))
        h = hashlib.sha256()
        for path in tracked:
            h.update(path.name.encode())
            h.update(path.read_bytes())
        return 'snapshot-' + h.hexdigest()


def find_split_file(dataset_dir):
    files = sorted(Path(dataset_dir).glob('*_split.pkl'))
    return files[0] if files else None


def split_hash(dataset_dir):
    split = find_split_file(dataset_dir)
    if split is not None:
        return sha256_file(split)
    files = sorted((Path(dataset_dir) / 'image').glob('*.npy'))
    return sha256_bytes('\n'.join(p.name for p in files).encode())


def dataset_label(dataset_dir, explicit=None):
    if explicit:
        return explicit
    name = Path(dataset_dir).name.lower()
    if 'crown' in name:
        return 'TopCoW_val'
    if 'lausanne' in name:
        return 'Lausanne'
    if 'ixi' in name:
        return 'IXI_HH'
    return Path(dataset_dir).name


def safe_number(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def binary_dice(pred, gt):
    pred = np.asarray(pred, dtype=bool)
    gt = np.asarray(gt, dtype=bool)
    denom = pred.sum() + gt.sum()
    return float((2 * np.logical_and(pred, gt).sum() + 1e-7) /
                 (denom + 1e-7))


def beta0_error(pred, gt):
    _, pn = ndimage.label(np.asarray(pred, dtype=np.uint8))
    _, gn = ndimage.label(np.asarray(gt, dtype=np.uint8))
    return abs(int(pn) - int(gn))


def per_case_detection_f1(detection):
    tp = fp = fn = 0
    for cls in GROUP2_LABELS:
        item = detection.get(cls)
        if item is None:
            continue
        tp += item['tp']; fp += item['fp']; fn += item['fn']
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return (2 * precision * recall / (precision + recall)
            if precision + recall else 0.0)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True,
                              ensure_ascii=False, default=str), encoding='utf-8')
    tmp.replace(path)


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text('', encoding='utf-8')
        return
    fields = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fields.append(key); seen.add(key)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with open(tmp, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    tmp.replace(path)


def write_parquet(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    tmp = path.with_suffix(path.suffix + '.tmp')
    frame.to_parquet(tmp, index=False)
    tmp.replace(path)


def load_foundation_model(exp_dir, device, trust_variant='full'):
    exp_dir = Path(exp_dir)
    cfg = json.loads((exp_dir / 'config.json').read_text(encoding='utf-8'))
    checkpoint = exp_dir / 'best_model.pth'
    encoder = build_foundation_encoder(
        cfg.get('foundation_type', 'sam3d_turbo'),
        cfg.get('foundation_ckpt'))
    model = TopCoWFoundationModel(
        encoder,
        fusion_type=cfg.get('fusion_type', 'tgf'),
        trust_variant=trust_variant).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(state.get('model_state_dict', state), strict=True)
    model.eval()
    model.set_epoch(999999)
    return model, cfg, checkpoint


@torch.no_grad()
def model_forward(model, x, variant, corruption, instrument=False,
                  donor_features=None, gate_alpha=None,
                  return_probability_signals=False):
    model.set_rebuttal_runtime(
        trust_variant=variant,
        corruption=corruption,
        instrumentation=instrument,
        donor_features=donor_features,
        gate_alpha=gate_alpha)
    torch.cuda.synchronize(x.device)
    torch.cuda.reset_peak_memory_stats(x.device)
    started = time.perf_counter()
    seg, _, _ = model(x)
    torch.cuda.synchronize(x.device)
    elapsed = time.perf_counter() - started
    peak_mb = torch.cuda.max_memory_allocated(x.device) / (1024 ** 2)
    pred = seg[0].argmax(dim=0).cpu().numpy().astype(np.int16)
    signals = model.get_rebuttal_signals() if instrument else {}
    if return_probability_signals:
        probability = seg[0].float().softmax(dim=0)
        confidence = probability.max(dim=0, keepdim=True).values[None]
        entropy = -(probability.clamp_min(1e-7) *
                    probability.clamp_min(1e-7).log()).sum(dim=0, keepdim=True)
        entropy = entropy / math.log(probability.shape[0])
        probability_signals = {
            'confidence': confidence.detach(),
            'entropy': entropy[None].detach(),
        }
        return pred, signals, elapsed, peak_mb, probability_signals
    return pred, signals, elapsed, peak_mb


def level_summary(signals):
    rows = {}
    for level, sig in signals.items():
        a_impl = sig.get('A_impl', sig['T'])
        t_formula = sig.get('T_formula', sig['G'] * (1.0 - sig['U']))
        rows[level] = {
            'U': float(sig['U'].float().mean().cpu()),
            'G': float(sig['G'].float().mean().cpu()),
            'T_formula': float(t_formula.float().mean().cpu()),
            'A_impl': float(a_impl.float().mean().cpu()),
            # Raw acceptance alias retained for old consumers.
            'T': float(sig['T'].float().mean().cpu()),
            'FDD': float(sig['FDD'].float().mean().cpu()),
            'delta_l1': float(sig['delta'].float().abs().mean().cpu()),
            'accepted_delta_l1': float(
                sig['accepted_delta'].float().abs().mean().cpu()),
            'accepted_delta_l2': float(
                sig['accepted_delta'].float().pow(2).sum(1).sqrt().mean().cpu()),
            'channel_weight': float(sig['channel_weight'].float().mean().cpu()),
        }
    return rows


def signal_mean_in_mask(signal, mask):
    spatial = signal.shape[2:]
    mask_t = torch.from_numpy(mask.astype(np.float32))[None, None].to(signal.device)
    if tuple(mask_t.shape[2:]) != tuple(spatial):
        mask_t = F.interpolate(mask_t, size=spatial, mode='nearest')
    mask_t = mask_t > 0.5
    if not bool(mask_t.any()):
        return None
    if signal.shape[1] != 1:
        signal = signal.float().mean(dim=1, keepdim=True)
    return float(signal.float()[mask_t.expand_as(signal)].mean().cpu())


def make_regions(gt, pred_noaux, pred_residual, pred_target):
    regions = []
    classes = sorted(set(np.unique(gt).tolist()) |
                     set(np.unique(pred_noaux).tolist()) |
                     set(np.unique(pred_residual).tolist()) |
                     set(np.unique(pred_target).tolist()))
    for cls in classes:
        if cls <= 0 or cls >= NUM_CLASSES:
            continue
        gt_mask = gt == cls
        support = gt_mask
        source = 'gt'
        if not support.any():
            support = ((pred_noaux == cls) | (pred_residual == cls) |
                       (pred_target == cls))
            source = 'prediction_union_for_gt_absent_class'
        if not support.any():
            continue
        component = ndimage.binary_dilation(support, iterations=2)
        regions.append({
            'region_id': f'class_{cls:02d}_component',
            'region_type': 'vessel_component_roi',
            'mask': component,
            'class': int(cls),
            'source': source,
        })

        if gt_mask.any():
            centerline = skeletonize_3d(gt_mask.astype(np.uint8)).astype(bool)
            if centerline.any():
                regions.append({
                    'region_id': f'class_{cls:02d}_centerline',
                    'region_type': 'centerline_segment_roi',
                    'mask': centerline,
                    'class': int(cls),
                    'source': 'gt_centerline',
                })

        coords = np.argwhere(support)
        center = np.round(coords.mean(axis=0)).astype(int)
        half = np.array([4, 8, 8])
        lo = np.maximum(center - half, 0)
        hi = np.minimum(center + half, np.array(gt.shape))
        patch = np.zeros_like(gt_mask, dtype=bool)
        patch[tuple(slice(int(a), int(b)) for a, b in zip(lo, hi))] = True
        regions.append({
            'region_id': f'class_{cls:02d}_fixed_patch',
            'region_type': 'fixed_feature_aligned_patch',
            'mask': patch,
            'class': int(cls),
            'source': source,
        })
    return regions


def region_utility(region, gt, pred_noaux, pred_residual, pred_target):
    mask = region['mask']
    cls = region['class']
    gt_cls = gt == cls
    no_cls = pred_noaux == cls
    res_cls = pred_residual == cls
    target_cls = pred_target == cls

    no_correct = (pred_noaux == gt)[mask].mean() if mask.any() else np.nan
    res_correct = (pred_residual == gt)[mask].mean() if mask.any() else np.nan
    target_correct = (pred_target == gt)[mask].mean() if mask.any() else np.nan

    q_component = binary_dice(res_cls, gt_cls) - binary_dice(no_cls, gt_cls)
    q_target_component = binary_dice(target_cls, gt_cls) - binary_dice(no_cls, gt_cls)
    q_vox = float(res_correct - no_correct)
    q_target_vox = float(target_correct - no_correct)

    skeleton = skeletonize_3d(gt_cls.astype(np.uint8)).astype(bool)
    if skeleton.any():
        q_centerline = float((pred_residual == gt)[skeleton].mean() -
                             (pred_noaux == gt)[skeleton].mean())
        q_target_centerline = float((pred_target == gt)[skeleton].mean() -
                                    (pred_noaux == gt)[skeleton].mean())
    else:
        q_centerline = None
        q_target_centerline = None

    q_topology = float(beta0_error(no_cls, gt_cls) -
                       beta0_error(res_cls, gt_cls))
    q_target_topology = float(beta0_error(no_cls, gt_cls) -
                              beta0_error(target_cls, gt_cls))
    return {
        'Q_vox': q_vox,
        'Q_centerline': q_centerline,
        'Q_component': q_component,
        'Q_topology': q_topology,
        'Q_target_vox': q_target_vox,
        'Q_target_centerline': q_target_centerline,
        'Q_target_component': q_target_component,
        'Q_target_topology': q_target_topology,
        'B_vox': int(q_vox > 0),
        'B_centerline': None if q_centerline is None else int(q_centerline > 0),
        'B_component': int(q_component > 0),
    }


def region_rows(base, signals, regions, gt, pred_noaux, pred_residual,
                pred_target, clean_summary=None, probability_signals=None):
    rows = []
    for region in regions:
        util = region_utility(
            region, gt, pred_noaux, pred_residual, pred_target)
        size = int((gt == region['class']).sum())
        size_bin = 'small' if size < 100 else ('medium' if size < 1000 else 'large')
        for level in LEVEL_ORDER:
            if level not in signals:
                continue
            sig = signals[level]
            mask = region['mask']
            u = float(sig['U'].float().mean().cpu())
            g = signal_mean_in_mask(sig['G'], mask)
            a_impl_tensor = sig.get('A_impl', sig['T'])
            formula_tensor = sig.get(
                'T_formula', sig['G'] * (1.0 - sig['U']))
            a_impl = signal_mean_in_mask(a_impl_tensor, mask)
            t_formula = signal_mean_in_mask(formula_tensor, mask)
            row = dict(base)
            row.update({
                'region_id': region['region_id'],
                'region_type': region['region_type'],
                'region_source': region['source'],
                'feature_level': level,
                'vessel_class': region['class'],
                'vessel_name': VESSEL_NAMES.get(region['class'], str(region['class'])),
                'vessel_group': ('communicating_or_variant'
                                 if region['class'] in SMALL_VESSEL_CLASSES else 'major'),
                'size_bin': size_bin,
                'centerline_or_boundary': ('centerline' if 'centerline' in region['region_type']
                                           else 'region'),
                'U': u,
                'G': g,
                'T_formula': t_formula,
                'A_impl': a_impl,
                # Historical raw name retained only for provenance.
                'T': a_impl,
                'FDD': signal_mean_in_mask(sig['FDD'], mask),
                'delta_l1': signal_mean_in_mask(sig['delta'].abs(), mask),
                'accepted_delta_l1': signal_mean_in_mask(
                    sig['accepted_delta'].abs(), mask),
                'channel_weight': float(sig['channel_weight'].float().mean().cpu()),
                'trust_bin': None if a_impl is None else min(9, int(a_impl * 10)),
                'A_impl_bin': None if a_impl is None else min(9, int(a_impl * 10)),
                'T_formula_bin': (None if t_formula is None else
                                  min(9, int(t_formula * 10))),
            })
            row.update(util)
            if probability_signals:
                row.update({
                    'confidence': signal_mean_in_mask(
                        probability_signals['confidence'], mask),
                    'entropy': signal_mean_in_mask(
                        probability_signals['entropy'], mask),
                    'prediction_error_rate': float(
                        (pred_target != gt)[mask].mean()) if mask.any() else None,
                })
            if clean_summary and level in clean_summary:
                row.update({
                    'clean_U': clean_summary[level]['U'],
                    'clean_G': clean_summary[level]['G'],
                    'clean_T_formula': clean_summary[level]['T_formula'],
                    'clean_A_impl': clean_summary[level]['A_impl'],
                    'clean_T': clean_summary[level]['A_impl'],
                    'delta_U_from_clean': u - clean_summary[level]['U'],
                    'delta_G_from_clean': (None if g is None else
                                           g - clean_summary[level]['G']),
                    'delta_T_formula_from_clean': (
                        None if t_formula is None else
                        t_formula - clean_summary[level]['T_formula']),
                    'delta_A_impl_from_clean': (
                        None if a_impl is None else
                        a_impl - clean_summary[level]['A_impl']),
                    'delta_T_from_clean': (
                        None if a_impl is None else
                        a_impl - clean_summary[level]['A_impl']),
                })
            beneficial = util['B_component'] == 1
            accepted = a_impl is not None and a_impl >= 0.5
            row['utility_quadrant_at_0p5'] = (
                'correct_acceptance' if beneficial and accepted else
                'beneficial_rejection' if beneficial else
                'harmful_acceptance' if accepted else
                'correct_suppression')
            rows.append(row)
    return rows


def save_signal_maps(path, signals, probability_signals=None):
    arrays = {}
    for level, signal in signals.items():
        arrays[f'{level}_U'] = signal['U'].float().cpu().numpy()
        for name in ('G', 'T', 'FDD'):
            arrays[f'{level}_{name}'] = signal[name].float().cpu().numpy()
        a_impl = signal.get('A_impl', signal['T'])
        t_formula = signal.get(
            'T_formula', signal['G'] * (1.0 - signal['U']))
        arrays[f'{level}_A_impl'] = a_impl.float().cpu().numpy()
        arrays[f'{level}_T_formula'] = t_formula.float().cpu().numpy()
        arrays[f'{level}_accepted_delta_l1'] = (
            signal['accepted_delta'].float().abs().mean(dim=1, keepdim=True)
            .cpu().numpy())
    if probability_signals:
        arrays['confidence'] = probability_signals['confidence'].float().cpu().numpy()
        arrays['entropy'] = probability_signals['entropy'].float().cpu().numpy()
    np.savez_compressed(path, **arrays)


def class_rows(base, gt, pred_noaux, pred_residual, pred_target):
    rows = []
    for cls in range(1, NUM_CLASSES):
        masks = {
            'gt': gt == cls,
            'no_aux': pred_noaux == cls,
            'residual': pred_residual == cls,
            'target': pred_target == cls,
        }
        if not any(mask.any() for mask in masks.values()):
            continue
        row = dict(base)
        row.update({
            'vessel_class': cls,
            'vessel_name': VESSEL_NAMES.get(cls, str(cls)),
            'gt_voxels': int(masks['gt'].sum()),
            'no_aux_dice': binary_dice(masks['no_aux'], masks['gt']),
            'residual_dice': binary_dice(masks['residual'], masks['gt']),
            'target_dice': binary_dice(masks['target'], masks['gt']),
            'no_aux_beta0_error': beta0_error(masks['no_aux'], masks['gt']),
            'residual_beta0_error': beta0_error(masks['residual'], masks['gt']),
            'target_beta0_error': beta0_error(masks['target'], masks['gt']),
        })
        rows.append(row)
    return rows


def persist_partial(run_dir, case_metrics, class_metrics, internal_rows,
                    target_official, failures):
    write_csv(run_dir / 'case_metrics.csv', case_metrics)
    write_csv(run_dir / 'class_metrics.csv', class_metrics)
    write_parquet(run_dir / 'internal_signals.parquet', internal_rows)
    write_json(run_dir / 'case_failures.json', failures)
    if target_official:
        write_json(run_dir / 'aggregate_metrics.json',
                   aggregate_results(target_official))


def run(args):
    project_root = Path(args.project_root).resolve()
    os.chdir(project_root)
    results_root = Path(args.results_root).resolve()
    run_id = args.run_id or (
        f"{args.model_variant}_{args.corruption_type}_s{args.severity:g}_r{args.realization}_"
        f"{dt.datetime.now().strftime('%Y%m%dT%H%M%S')}")
    run_dir = results_root / '03_raw_runs' / args.experiment_id / run_id
    if run_dir.exists():
        raise FileExistsError(f'Run directory already exists: {run_dir}')
    run_dir.mkdir(parents=True)
    (run_dir / 'predictions').mkdir()
    (run_dir / 'maps').mkdir()

    config = vars(args).copy()
    config['run_id'] = run_id
    config['protocol_notes'] = {
        'tta': False,
        'main_path': 'clean',
        'corruption_scope': 'projected_auxiliary_features_only',
        'utility_reference': 'uncalibrated_residual_minus_no_auxiliary',
        'roi_selection': 'predefined_gt_component_centerline_and_fixed_patch',
        'probability_signals': ('persisted_region_means'
                                if args.save_probability_signals else 'not_requested'),
    }
    config_hash = sha256_bytes(canonical_json(config).encode())
    with open(run_dir / 'run_config.yaml', 'w', encoding='utf-8') as handle:
        yaml.safe_dump(config, handle, allow_unicode=True, sort_keys=True)

    dataset_dir = Path(args.dataset_dir).resolve()
    exp_dir = Path(args.exp_dir).resolve()
    checkpoint = exp_dir / 'best_model.pth'
    manifest = {
        'experiment_id': args.experiment_id,
        'run_id': run_id,
        'git_commit': git_commit(project_root),
        'config_hash': config_hash,
        'checkpoint_hash': sha256_file(checkpoint),
        'data_split_hash': split_hash(dataset_dir),
        'dataset': dataset_label(dataset_dir, args.dataset_name),
        'model_variant': args.model_variant,
        'corruption': {
            'type': args.corruption_type,
            'severity': args.severity,
            'unit': args.severity_unit,
            'realization': args.realization,
        },
        'seed': args.seed,
        'started_at': utc_now(),
        'finished_at': None,
        'status': 'running',
        'failure_reason': None,
    }
    write_json(run_dir / 'run_manifest.json', manifest)
    write_json(run_dir / 'status.json', {'status': 'running', 'started_at': utc_now()})

    device = torch.device(args.device)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model, model_cfg, checkpoint = load_foundation_model(
        exp_dir, device, trust_variant=args.model_variant)
    dataset = TopCoWTestDataset(str(dataset_dir), split='test')
    selected = list(range(len(dataset)))
    if args.case_ids:
        wanted = set(args.case_ids.split(','))
        selected = [i for i in selected if dataset.samples[i]['case_name'] in wanted]
    if args.max_cases is not None:
        selected = selected[:args.max_cases]
    if not selected:
        raise RuntimeError('No cases selected')
    write_json(run_dir / 'selected_cases.json', [
        dataset.samples[i]['case_name'] for i in selected])

    case_metrics = []
    class_metrics = []
    internal_rows = []
    target_official = []
    failures = []
    started_all = time.perf_counter()

    for position, idx in enumerate(selected):
        sample = dataset[idx]
        case_id = sample['case_name']
        print(f'[{position + 1}/{len(selected)}] {case_id}', flush=True)
        try:
            x = sample['image'].unsqueeze(0).float().to(device)
            gt = sample['multiclass'].numpy().astype(np.int16)
            case_seed = stable_seed(args.seed, args.realization, case_id)
            corruption = {
                'type': args.corruption_type,
                'severity': args.severity,
                'seed': case_seed,
            }
            if args.target_level:
                corruption['target_level'] = args.target_level

            donor_features = None
            donor_case = None
            if args.corruption_type == 'donor_mix' and args.severity != 0:
                donor_idx = selected[(position + 1) % len(selected)]
                donor = dataset[donor_idx]
                donor_case = donor['case_name']
                donor_x = donor['image'].unsqueeze(0).float().to(device)
                _, donor_signals, _, _ = model_forward(
                    model, donor_x, 'full',
                    {'type': 'none', 'severity': 0.0, 'seed': case_seed},
                    instrument=True)
                donor_features = {
                    level: sig['A_clean'] for level, sig in donor_signals.items()}

            clean_summary = None
            if args.corruption_type not in ('none', 'clean') and args.severity != 0:
                _, clean_signals, _, _ = model_forward(
                    model, x, 'full',
                    {'type': 'none', 'severity': 0.0, 'seed': case_seed},
                    instrument=True)
                clean_summary = level_summary(clean_signals)

            pred_noaux, _, noaux_s, _ = model_forward(
                model, x, 'no_aux', corruption, instrument=False,
                donor_features=donor_features)
            pred_residual, _, residual_s, _ = model_forward(
                model, x, 'residual', corruption, instrument=False,
                donor_features=donor_features)
            probability_signals = None
            if args.save_probability_signals or args.save_signal_maps:
                (pred_target, signals, target_s, peak_mb,
                 probability_signals) = model_forward(
                    model, x, args.model_variant, corruption, instrument=True,
                    donor_features=donor_features,
                    return_probability_signals=True)
            else:
                pred_target, signals, target_s, peak_mb = model_forward(
                    model, x, args.model_variant, corruption, instrument=True,
                    donor_features=donor_features)

            official = evaluate_single_case(pred_target, gt)
            official['case'] = case_id
            target_official.append(official)
            summaries = level_summary(signals)
            mean = lambda key: float(np.mean([v[key] for v in summaries.values()]))
            base = {
                'experiment_id': args.experiment_id,
                'run_id': run_id,
                'dataset': manifest['dataset'],
                'case_id': case_id,
                'seed': args.seed,
                'realization': args.realization,
                'model_variant': args.model_variant,
                'corruption_type': args.corruption_type,
                'severity': args.severity,
                'donor_case_id': donor_case,
            }
            row = dict(base)
            row.update({
                'dice': official['binary_dice'],
                'cldice': official['binary_cldice'],
                'hd95': safe_number(official['class_avg_hd95']),
                'beta0_error': official['binary_b0'],
                'clsavg': official['class_avg_dice'],
                'cls_beta0': official['class_avg_b0'],
                'det_f1': per_case_detection_f1(official['detection']),
                'mean_U': mean('U'),
                'mean_G': mean('G'),
                'mean_T_formula': mean('T_formula'),
                'mean_A_impl': mean('A_impl'),
                'mean_T': mean('A_impl'),
                'mean_FDD': mean('FDD'),
                'accepted_delta_l1': mean('accepted_delta_l1'),
                'accepted_delta_l2': mean('accepted_delta_l2'),
                'runtime_sec': target_s,
                'reference_runtime_sec': noaux_s + residual_s,
                'peak_gpu_mem_mb': peak_mb,
                'prediction_sha256': sha256_bytes(pred_target.tobytes()),
            })
            if probability_signals:
                row.update({
                    'mean_confidence': float(
                        probability_signals['confidence'].float().mean().cpu()),
                    'mean_entropy': float(
                        probability_signals['entropy'].float().mean().cpu()),
                    'prediction_error_rate': float((pred_target != gt).mean()),
                })
            case_metrics.append(row)
            class_metrics.extend(class_rows(
                base, gt, pred_noaux, pred_residual, pred_target))
            regions = make_regions(
                gt, pred_noaux, pred_residual, pred_target)
            internal_rows.extend(region_rows(
                base, signals, regions, gt, pred_noaux, pred_residual,
                pred_target, clean_summary=clean_summary,
                probability_signals=probability_signals))
            if args.save_predictions:
                np.savez_compressed(
                    run_dir / 'predictions' / f'{case_id}.npz',
                    target=pred_target, no_aux=pred_noaux,
                    residual=pred_residual, gt=gt)
            if args.save_signal_maps:
                save_signal_maps(run_dir / 'maps' / f'{case_id}.npz', signals,
                                 probability_signals=probability_signals)
            persist_partial(run_dir, case_metrics, class_metrics,
                            internal_rows, target_official, failures)
            del x, signals, donor_features
            torch.cuda.empty_cache()
        except Exception as exc:
            failure = {
                'case_id': case_id,
                'error_type': type(exc).__name__,
                'error': str(exc),
                'traceback': traceback.format_exc(),
            }
            failures.append(failure)
            print(f"FAILED {case_id}: {exc}", file=sys.stderr, flush=True)
            persist_partial(run_dir, case_metrics, class_metrics,
                            internal_rows, target_official, failures)
            if args.fail_fast:
                raise

    elapsed_all = time.perf_counter() - started_all
    status = ('success' if not failures else
              'partial' if case_metrics else 'failed')
    runtime = {
        'wall_time_sec': elapsed_all,
        'n_selected': len(selected),
        'n_success': len(case_metrics),
        'n_failed': len(failures),
        'device': str(device),
        'gpu_name': torch.cuda.get_device_name(device),
        'torch': torch.__version__,
        'cuda': torch.version.cuda,
        'model_config': model_cfg,
    }
    write_json(run_dir / 'runtime.json', runtime)
    manifest.update({
        'finished_at': utc_now(),
        'status': status,
        'failure_reason': None if not failures else f'{len(failures)} case failures',
    })
    write_json(run_dir / 'run_manifest.json', manifest)
    write_json(run_dir / 'status.json', {
        'status': status,
        'finished_at': manifest['finished_at'],
        'n_success': len(case_metrics),
        'n_failed': len(failures),
    })
    print(f'RUN_DIR={run_dir}')
    print(f'STATUS={status} success={len(case_metrics)} failed={len(failures)}')
    return 0 if status == 'success' else 2


def build_parser():
    parser = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    parser.add_argument('--project_root', default=str(here))
    parser.add_argument('--results_root', default=str(here / 'results_tcff_rebuttal'))
    parser.add_argument('--experiment_id', required=True)
    parser.add_argument('--run_id')
    parser.add_argument('--exp_dir', default=str(here / 'Runs_foundation' / 'sam_tgf_v3'))
    parser.add_argument('--dataset_dir', required=True)
    parser.add_argument('--dataset_name')
    parser.add_argument('--model_variant', choices=TRUST_VARIANTS, default='full')
    parser.add_argument('--corruption_type', default='none', choices=[
        'none', 'gaussian', 'translation', 'channel_dropout',
        'attenuation', 'donor_mix', 'missing_level'])
    parser.add_argument('--severity', type=float, default=0.0)
    parser.add_argument('--severity_unit', default='none')
    parser.add_argument('--target_level', choices=['', 'shallow', 'mid', 'deep'], default='')
    parser.add_argument('--realization', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--case_ids', default='')
    parser.add_argument('--max_cases', type=int)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--save_predictions', action='store_true')
    parser.add_argument('--save_probability_signals', action='store_true')
    parser.add_argument('--save_signal_maps', action='store_true')
    parser.add_argument('--fail_fast', action='store_true')
    return parser


if __name__ == '__main__':
    try:
        raise SystemExit(run(build_parser().parse_args()))
    except Exception:
        traceback.print_exc()
        raise
