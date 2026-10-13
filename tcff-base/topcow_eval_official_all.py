"""
topcow_eval_official.py (v3)
=============================
完整对齐官方 TopCoW 2024 评估指标

官方 7 项指标 (Task 1 Segmentation):
  体素级:
    1. Class-average Dice (mean±std across cases)
    2. clDice on merged binary (mean±std)
    3. Class-average β0 error (mean±std)
    4. Class-average HD95 (mean±std)
  超越分割:
    5. Group 2 detection F1
    6. Variant-balanced graph classification  ← 需要 edge-list, 未实现
    7. Topology match rate                    ← 需要 edge-list, 未实现

v3 vs v2 改动:
  - 保存文件名改为 official_metrics_{dataset}.json
    dataset 自动从 --test_dir basename 推断, 也可用 --eval_tag 手动指定
  - --summarize 同时读取新格式 official_metrics_*.json 和旧格式 official_metrics.json
    以及 results.json, 向后兼容历史结果
  - 其余评估逻辑与 v2 完全一致

用法:
  # 单个实验评估 (dataset tag 自动推断为 "crown_processed")
  python topcow_eval_official.py --eval \
      --exp_dir ./runs_topcow/tgf_clean \
      --test_dir ./crown_processed

  # 换数据集, 不影响已有结果
  python topcow_eval_official.py --eval \
      --exp_dir ./runs_topcow/tgf_clean \
      --test_dir ./ixi_processed

  # 手动指定后缀
  python topcow_eval_official.py --eval \
      --exp_dir ./runs_topcow/tgf_clean \
      --test_dir ./ixi_processed \
      --eval_tag ixi

  # 所有实验跑一遍
  python topcow_eval_official.py --eval_all \
      --output_dir ./runs_topcow \
      --test_dir ./ixi_processed

  # 汇总 (读取全部数据集的结果)
  python topcow_eval_official.py --summarize --output_dir ./runs_topcow

依赖:
  pip install scikit-image scipy numpy torch
"""

import os
import sys
import glob
import json
import argparse
import time
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize_3d
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SAM_MED3D_PATH = '/mnt/e/00-Data/Brain-TopCoW/24/topcow-TGF/SAM-Med3D'
sys.path.insert(0, SAM_MED3D_PATH)

NUM_CLASSES = 14
VESSEL_NAMES = {
    1: "BA",    2: "R-PCA",  3: "L-PCA",  4: "R-ICA",  5: "R-MCA",
    6: "L-ICA", 7: "L-MCA",  8: "R-Pcom", 9: "L-Pcom", 10: "Acom",
    11: "R-ACA", 12: "L-ACA", 13: "3rd-A2"
}

# Group 2: 变异性最大的血管组件
GROUP2_LABELS = [8, 9, 10, 13]  # R-Pcom, L-Pcom, Acom, 3rd-A2

ALL_MODEL_TYPES = ['tgf', 'tgf_nogate', 'tgf_pretrain', 'unet',
                   'concat', 'crossattn', 'bagf', 'pug', 'dsff']


# ============================================================
# 工具: 从 test_dir 推断 dataset tag
# ============================================================

def infer_dataset_tag(test_dir, eval_tag=None):
    """返回用于文件名的 tag 字符串"""
    if eval_tag:
        return eval_tag
    return os.path.basename(test_dir.rstrip('/\\'))


def metrics_filename(dataset_tag):
    """新格式文件名"""
    return f'official_metrics_{dataset_tag}.json'


# ============================================================
# 指标计算
# ============================================================

def compute_dice(p, g):
    s = 1e-7
    i = (p * g).sum()
    return float((2 * i + s) / (p.sum() + g.sum() + s))


def compute_cldice(p, g):
    p, g = p.astype(np.uint8), g.astype(np.uint8)
    if p.sum() == 0 and g.sum() == 0:
        return 1.0
    if p.sum() == 0 or g.sum() == 0:
        return 0.0
    sp = skeletonize_3d(p).astype(bool)
    sg = skeletonize_3d(g).astype(bool)
    if sp.sum() == 0 or sg.sum() == 0:
        return 0.0
    tp = (sp & g.astype(bool)).sum() / sp.sum()
    ts = (sg & p.astype(bool)).sum() / sg.sum()
    return float(2 * tp * ts / (tp + ts)) if tp + ts > 0 else 0.0


def compute_b0(p, g):
    if p.sum() == 0 and g.sum() == 0:
        return 0
    _, pn = ndimage.label(p.astype(np.int32))
    _, gn = ndimage.label(g.astype(np.int32))
    return abs(pn - gn)


def compute_hd95(pred_bin, gt_bin, voxel_spacing=None):
    """Hausdorff Distance 95th percentile"""
    pred_bin = pred_bin.astype(bool)
    gt_bin   = gt_bin.astype(bool)
    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return 0.0
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return float('inf')
    spacing = voxel_spacing if voxel_spacing is not None else (1.0, 1.0, 1.0)
    dt_gt   = distance_transform_edt(~gt_bin,   sampling=spacing)
    dt_pred = distance_transform_edt(~pred_bin, sampling=spacing)
    dist_p2g = dt_gt[pred_bin]
    dist_g2p = dt_pred[gt_bin]
    return float(np.percentile(np.concatenate([dist_p2g, dist_g2p]), 95))


def compute_detection_iou(pred_c, gt_c):
    inter = (pred_c & gt_c).sum()
    union = (pred_c | gt_c).sum()
    return float(inter / union) if union > 0 else 0.0


# ============================================================
# 单 Case 评估
# ============================================================

def evaluate_single_case(pred, gt, voxel_spacing=None):
    r = {}

    # Binary
    pb = (pred > 0).astype(np.uint8)
    gb = (gt   > 0).astype(np.uint8)
    r['binary_dice']   = compute_dice(pb, gb)
    r['binary_cldice'] = compute_cldice(pb, gb)
    r['binary_b0']     = compute_b0(pb, gb)

    # Per-class
    pc = {}
    for c in range(1, NUM_CLASSES):
        p_c = (pred == c).astype(np.uint8)
        g_c = (gt   == c).astype(np.uint8)
        if g_c.sum() == 0 and p_c.sum() == 0:
            pc[c] = None
            continue
        pc[c] = {
            'dice': compute_dice(p_c, g_c),
            'b0':   compute_b0(p_c, g_c),
            'hd95': compute_hd95(p_c, g_c, voxel_spacing),
        }
    r['per_class'] = pc

    # Class-average
    valid    = [v for v in pc.values() if v is not None]
    valid_hd = [v['hd95'] for v in valid if np.isfinite(v['hd95'])]
    r['class_avg_dice'] = float(np.mean([v['dice'] for v in valid])) if valid else 0.0
    r['class_avg_b0']   = float(np.mean([v['b0']   for v in valid])) if valid else 0.0
    r['class_avg_hd95'] = float(np.mean(valid_hd)) if valid_hd else float('inf')

    # Group 2 Detection
    det = {}
    for c in GROUP2_LABELS:
        p_c = (pred == c).astype(bool)
        g_c = (gt   == c).astype(bool)
        gt_ex   = g_c.sum() > 0
        pred_ex = p_c.sum() > 0
        if gt_ex and pred_ex:
            iou = compute_detection_iou(p_c, g_c)
            tp  = 1 if iou >= 0.25 else 0
            det[c] = {'tp': tp, 'fp': 1 - tp, 'fn': 0, 'iou': iou}
        elif gt_ex and not pred_ex:
            det[c] = {'tp': 0, 'fp': 0, 'fn': 1, 'iou': 0.0}
        elif not gt_ex and pred_ex:
            det[c] = {'tp': 0, 'fp': 1, 'fn': 0, 'iou': 0.0}
        else:
            det[c] = None
    r['detection'] = det

    return r


# ============================================================
# 批量汇总
# ============================================================

def aggregate_results(all_results):
    if not all_results:
        return {}
    s = {'n': len(all_results)}

    for k in ['binary_dice', 'binary_cldice', 'binary_b0']:
        vals = [r[k] for r in all_results]
        s[k]              = float(np.mean(vals))
        s[f'{k}_std']     = float(np.std(vals))

    for k in ['class_avg_dice', 'class_avg_b0', 'class_avg_hd95']:
        vals = [r[k] for r in all_results if np.isfinite(r.get(k, float('inf')))]
        s[k]          = float(np.mean(vals)) if vals else float('inf')
        s[f'{k}_std'] = float(np.std(vals))  if vals else 0.0

    per_class_summary = {}
    for c in range(1, NUM_CLASSES):
        cases = [r['per_class'].get(c) for r in all_results
                 if r['per_class'].get(c) is not None]
        if not cases:
            per_class_summary[c] = None
            continue
        pcs = {'n': len(cases)}
        for metric in ['dice', 'b0', 'hd95']:
            vals = [v[metric] for v in cases if np.isfinite(v.get(metric, float('inf')))]
            if vals:
                pcs[f'{metric}_mean'] = float(np.mean(vals))
                pcs[f'{metric}_std']  = float(np.std(vals))
            else:
                pcs[f'{metric}_mean'] = float('inf') if metric == 'hd95' else 0.0
                pcs[f'{metric}_std']  = 0.0
        per_class_summary[c] = pcs
    s['per_class'] = per_class_summary

    # Group 2 Detection F1
    tp = fp = fn = 0
    for r in all_results:
        for c in GROUP2_LABELS:
            d = r.get('detection', {}).get(c)
            if d is None:
                continue
            tp += d['tp']; fp += d['fp']; fn += d['fn']
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    s['grp2_detection_f1'] = float(f1)
    s['grp2_precision']    = float(precision)
    s['grp2_recall']       = float(recall)
    s['per_case']          = all_results

    return s


# ============================================================
# 模型加载
# ============================================================

def load_model(exp_dir, model_type, device, ckpt='best_model.pth'):
    path = os.path.join(exp_dir, ckpt)
    if not os.path.exists(path):
        print(f"  [SKIP] Not found: {path}")
        return None

    config_path = os.path.join(exp_dir, 'config.json')
    foundation_type = fusion_type = None
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
        foundation_type = cfg.get('foundation_type')
        fusion_type     = cfg.get('fusion_type')
        if model_type is None:
            model_type = cfg.get('model_type', 'tgf')

    state = torch.load(path, map_location=device, weights_only=False)
    sd    = state.get('model_state_dict', state)

    # Foundation 模型
    try:
        from topcow_foundation_model import (
            TopCoWFoundationModel, build_foundation_encoder, FOUNDATION_TYPES
        )
        if foundation_type and foundation_type in FOUNDATION_TYPES:
            encoder = build_foundation_encoder(
                foundation_type, cfg.get('foundation_ckpt', None))
            m = TopCoWFoundationModel(
                encoder, fusion_type=fusion_type or 'tgf',
                trust_variant=cfg.get('trust_variant', 'full'))
            m.to(device)
            m.load_state_dict(sd, strict=True)
            m.eval()
            if hasattr(m, 'set_epoch'):
                m.set_epoch(999999)
            print(f"  Loaded foundation model: {foundation_type}+{fusion_type}")
            return m
    except ImportError:
        pass

    # Baseline 模型
    from topcow_model import TopCoWTGFModel, TopCoWUNet
    from topcow_model_baselines import TopCoWBaselineFusionModel

    if model_type == 'unet':
        m = TopCoWUNet()
    elif model_type in ('tgf', 'tgf_pretrain'):
        m = TopCoWTGFModel(disable_gating=False)
    elif model_type == 'tgf_nogate':
        m = TopCoWTGFModel(disable_gating=True)
    elif model_type in ('concat', 'crossattn', 'bagf', 'pug', 'dsff'):
        m = TopCoWBaselineFusionModel(fusion_type=model_type)
    else:
        print(f"  [SKIP] Unknown model_type: {model_type}")
        return None

    m.load_state_dict(sd, strict=False)
    m.to(device).eval()
    if hasattr(m, 'set_epoch'):
        m.set_epoch(999999)
    print(f"  Loaded: {model_type}")
    return m


# ============================================================
# 推理
# ============================================================

def tta_infer(model, x, device):
    preds = []
    for fz in (False, True):
        for fy in (False, True):
            a = x.clone()
            if fz: a = a.flip(2)
            if fy: a = a.flip(3)
            s, _, _ = model(a)
            if fy: s = s.flip(3)
            if fz: s = s.flip(2)
            preds.append(s)
    return torch.stack(preds).mean(0)


@torch.no_grad()
def eval_model(model, test_dir, device, use_tta=True):
    from topcow_dataset import TopCoWTestDataset
    ds = TopCoWTestDataset(test_dir, split='test')
    dl = DataLoader(ds, batch_size=1, shuffle=False,
                    num_workers=2, pin_memory=True)
    results = []
    for batch in dl:
        x  = batch['image'].float().to(device)
        mc = batch.get('multiclass')
        if mc is None:
            continue
        seg  = tta_infer(model, x, device) if use_tta else model(x)[0]
        pred = torch.argmax(seg[0], dim=0).cpu().numpy()
        gt   = mc[0].numpy()
        t0   = time.time()
        m    = evaluate_single_case(pred, gt)
        name = batch['case_name']
        name = name[0] if isinstance(name, (list, tuple)) else name
        m['case'] = name
        results.append(m)
        hd_str = f"{m['class_avg_hd95']:.1f}" if np.isfinite(m['class_avg_hd95']) else "inf"
        print(f"  {name}: Dice={m['binary_dice']:.4f} "
              f"clDice={m['binary_cldice']:.4f} "
              f"b0={m['binary_b0']} "
              f"HD95={hd_str} "
              f"ClsAvg={m['class_avg_dice']:.4f} "
              f"({time.time()-t0:.1f}s)")
    return aggregate_results(results)


# ============================================================
# 打印报表
# ============================================================

def print_report(s, title=""):
    if not s:
        return
    if title:
        print(f"\n{'='*70}")
        print(f"  {title}")
        print(f"{'='*70}")

    print(f"\n  Binary 指标 (N={s['n']}):")
    print(f"    Dice:   {s['binary_dice']:.4f} ± {s['binary_dice_std']:.4f}")
    print(f"    clDice: {s['binary_cldice']:.4f} ± {s['binary_cldice_std']:.4f}")
    print(f"    β0:     {s['binary_b0']:.2f} ± {s['binary_b0_std']:.2f}")

    print(f"\n  Class-Average 指标:")
    print(f"    Dice:   {s['class_avg_dice']:.4f} ± {s.get('class_avg_dice_std', 0):.4f}")
    print(f"    β0:     {s['class_avg_b0']:.2f} ± {s.get('class_avg_b0_std', 0):.2f}")
    hd   = s.get('class_avg_hd95', float('inf'))
    hd_s = f"{hd:.2f}" if np.isfinite(hd) else "inf"
    print(f"    HD95:   {hd_s} ± {s.get('class_avg_hd95_std', 0):.2f}")

    print(f"\n  Group 2 Detection:")
    print(f"    F1:        {s.get('grp2_detection_f1', 0):.4f}")
    print(f"    Precision: {s.get('grp2_precision', 0):.4f}")
    print(f"    Recall:    {s.get('grp2_recall', 0):.4f}")

    pc = s.get('per_class', {})
    print(f"\n  Per-class Dice (mean±std):")
    print(f"    {'Class':>8}  {'Dice':>12}  {'β0':>10}  {'HD95':>14}  {'N':>3}")
    print(f"    {'-'*55}")
    for c in range(1, NUM_CLASSES):
        info = pc.get(c, pc.get(str(c)))
        if info is None:
            continue
        d_m  = info.get('dice_mean', 0)
        d_s  = info.get('dice_std', 0)
        b_m  = info.get('b0_mean', 0)
        b_s  = info.get('b0_std', 0)
        h_m  = info.get('hd95_mean', float('inf'))
        h_s  = info.get('hd95_std', 0)
        h_str = f"{h_m:.1f}±{h_s:.1f}" if np.isfinite(h_m) else "inf"
        print(f"    {VESSEL_NAMES.get(c,'?'):>8}  {d_m:.4f}±{d_s:.4f}  "
              f"{b_m:.1f}±{b_s:.1f}  {h_str:>14}  {info.get('n',0):>3}")


def print_comparison_table(all_summaries):
    """多实验 × 多数据集 对比表"""
    sorted_items = sorted(all_summaries.items(),
                          key=lambda x: x[1].get('class_avg_dice', 0), reverse=True)

    W = 130
    print(f"\n{'='*W}")
    print("综合对比表 (均值±标准差)")
    print(f"{'='*W}")
    print(f"{'Exp':<28} {'Dice↑':>14} {'clDice↑':>14} {'β0↓':>10} "
          f"{'HD95↓':>12} {'ClsAvg↑':>14} {'Cls_β0↓':>10} {'Det_F1↑':>8}")
    print(f"{'-'*W}")

    for key, s in sorted_items:
        bd   = s.get('binary_dice', 0);       bd_s = s.get('binary_dice_std', 0)
        cl   = s.get('binary_cldice', 0);     cl_s = s.get('binary_cldice_std', 0)
        b0   = s.get('binary_b0', 0);         b0_s = s.get('binary_b0_std', 0)
        ca   = s.get('class_avg_dice', 0);    ca_s = s.get('class_avg_dice_std', 0)
        cb   = s.get('class_avg_b0', 0);      cb_s = s.get('class_avg_b0_std', 0)
        ch   = s.get('class_avg_hd95', float('inf'))
        ch_s = s.get('class_avg_hd95_std', 0)
        f1   = s.get('grp2_detection_f1', 0)
        ch_str = f"{ch:.1f}±{ch_s:.1f}" if np.isfinite(ch) else "inf"
        print(f"{key:<28} "
              f"{bd:>6.4f}±{bd_s:.3f}  "
              f"{cl:>6.4f}±{cl_s:.3f}  "
              f"{b0:>4.1f}±{b0_s:.1f}  "
              f"{ch_str:>12}  "
              f"{ca:>6.4f}±{ca_s:.3f}  "
              f"{cb:>4.1f}±{cb_s:.1f}  "
              f"{f1:>8.3f}")

    # Per-class Dice 表
    classes = list(range(1, NUM_CLASSES))
    print(f"\n{'='*150}")
    print("Per-class Dice (mean±std)")
    print(f"{'='*150}")
    hdr = f"{'Exp':<28} " + " ".join(f"{VESSEL_NAMES[c]:>11}" for c in classes)
    print(hdr)
    print(f"{'-'*len(hdr)}")
    for key, s in sorted_items:
        pc   = s.get('per_class', {})
        vals = []
        for c in classes:
            info = pc.get(c, pc.get(str(c)))
            if isinstance(info, dict):
                dm = info.get('dice_mean', info.get('dice', 0))
                ds = info.get('dice_std', 0)
                vals.append(f"{dm:.3f}±{ds:.2f}")
            else:
                vals.append("—")
        print(f"{key:<28} " + " ".join(f"{v:>11}" for v in vals))


# ============================================================
# 读取结果文件 (向后兼容)
# ============================================================

def load_all_results(output_dir):
    """
    读取 output_dir 下所有实验的结果文件.

    优先级 (每个实验目录):
      1. official_metrics_*.json  (v3 新格式, 一个文件对应一个数据集)
      2. official_metrics.json    (v1/v2 旧格式)
      3. results.json             (训练时保存的初步结果)

    key 格式:
      新格式: "{exp_name}__{dataset_tag}"   e.g. "tgf_clean__crown_processed"
      旧格式: "{exp_name}"                  e.g. "tgf_clean"
      results.json: "{exp_name}__train_results"
    """
    all_summaries = {}

    for exp_name in sorted(os.listdir(output_dir)):
        ed = os.path.join(output_dir, exp_name)
        if not os.path.isdir(ed):
            continue

        found_any = False

        # 1. 新格式: official_metrics_*.json
        new_files = sorted(glob.glob(os.path.join(ed, 'official_metrics_*.json')))
        for fp in new_files:
            tag = os.path.basename(fp) \
                    .replace('official_metrics_', '') \
                    .replace('.json', '')
            key = f'{exp_name}__{tag}'
            with open(fp) as f:
                r = json.load(f)
            r['exp_name'] = exp_name
            r['dataset_tag'] = tag
            all_summaries[key] = r
            found_any = True

        # 2. 旧格式: official_metrics.json (没有 tag 的历史文件)
        old_fp = os.path.join(ed, 'official_metrics.json')
        if os.path.exists(old_fp):
            key = exp_name  # 旧格式 key 不带数据集后缀
            with open(old_fp) as f:
                r = json.load(f)
            r['exp_name'] = exp_name
            r.setdefault('dataset_tag', 'legacy')
            all_summaries[key] = r
            found_any = True

        # 3. 兜底: results.json (训练时产出)
        if not found_any:
            res_fp = os.path.join(ed, 'results.json')
            if os.path.exists(res_fp):
                key = f'{exp_name}__train_results'
                with open(res_fp) as f:
                    r = json.load(f)
                r['exp_name'] = exp_name
                r.setdefault('dataset_tag', 'train_results')
                all_summaries[key] = r

    return all_summaries


# ============================================================
# 命令行动作
# ============================================================

def run_eval(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tag    = infer_dataset_tag(args.test_dir, args.eval_tag)

    # 推断 model_type
    mt = args.model_type
    if mt is None:
        cp = os.path.join(args.exp_dir, 'config.json')
        if os.path.exists(cp):
            with open(cp) as f:
                mt = json.load(f).get('model_type', 'tgf')
        else:
            mt = 'tgf'

    model = load_model(args.exp_dir, mt, device)
    if model is None:
        return

    print(f"\nEval: {args.exp_dir}")
    print(f"  Dataset : {args.test_dir}  (tag={tag})")
    print(f"  TTA     : {not args.no_tta}")

    r = eval_model(model, args.test_dir, device, use_tta=not args.no_tta)
    if not r:
        print("No results.")
        return

    print_report(r, title=f"{args.exp_dir}  [{tag}]")

    save_path = os.path.join(args.exp_dir, metrics_filename(tag))
    with open(save_path, 'w') as f:
        json.dump(r, f, indent=2, default=str)
    print(f"\nSaved: {save_path}")


def run_eval_all(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    tag    = infer_dataset_tag(args.test_dir, args.eval_tag)
    all_summaries = {}

    for exp_name in sorted(os.listdir(args.output_dir)):
        ed = os.path.join(args.output_dir, exp_name)
        if not os.path.isdir(ed):
            continue
        ckpt = os.path.join(ed, 'best_model.pth')
        if not os.path.exists(ckpt):
            continue

        mt = args.model_type
        cp = os.path.join(ed, 'config.json')
        if mt is None and os.path.exists(cp):
            with open(cp) as f:
                mt = json.load(f).get('model_type', 'tgf')

        print(f"\n{'='*60}")
        print(f"Eval: {exp_name}  dataset={tag}")
        print(f"{'='*60}")

        model = load_model(ed, mt, device)
        if model is None:
            continue

        r = eval_model(model, args.test_dir, device, use_tta=not args.no_tta)
        if r:
            save_path = os.path.join(ed, metrics_filename(tag))
            with open(save_path, 'w') as f:
                json.dump(r, f, indent=2, default=str)
            print(f"  Saved: {save_path}")
            all_summaries[f'{exp_name}__{tag}'] = r

        del model
        torch.cuda.empty_cache()

    if all_summaries:
        print_comparison_table(all_summaries)


def run_summarize(args):
    all_summaries = load_all_results(args.output_dir)
    if not all_summaries:
        print("No results found.")
        return
    print(f"Found {len(all_summaries)} result entries.")
    print_comparison_table(all_summaries)
    plot_curves(args.output_dir)


# ============================================================
# 可选: 训练曲线图
# ============================================================

def plot_curves(output_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        return

    exps = {}
    for d in sorted(os.listdir(output_dir)):
        hp = os.path.join(output_dir, d, 'history.json')
        if os.path.exists(hp):
            with open(hp) as f:
                exps[d] = json.load(f)
    if not exps:
        return

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for name, h in exps.items():
        ep = range(1, len(h['train_loss']) + 1)
        axes[0, 0].plot(ep, h['train_loss'], label=name, alpha=0.8)
        if 'train_dice' in h:
            td = h['train_dice']
            axes[0, 1].plot(range(1, len(td)+1), td, label=name, alpha=0.8)
        for k, ax in [('test_binary', axes[1, 0]), ('test_classavg', axes[1, 1])]:
            vs  = h.get(k, [])
            pts = [(e, v) for e, v in zip(ep, vs) if v is not None]
            if pts:
                ax.plot(*zip(*pts), 'o-', label=name, alpha=0.8)
    for ax, t in zip(axes.flat, ['Loss', 'Train Dice', 'Test Binary', 'Test ClassAvg']):
        ax.set_title(t); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    sp = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(sp, dpi=150)
    print(f"Saved: {sp}")
    plt.close()


# ============================================================
# 入口
# ============================================================

def main():
    p = argparse.ArgumentParser(description='TopCoW official evaluation (v3)')
    p.add_argument('--eval',      action='store_true', help='评估单个实验')
    p.add_argument('--eval_all',  action='store_true', help='评估 output_dir 下所有实验')
    p.add_argument('--summarize', action='store_true', help='汇总已有结果')
    p.add_argument('--plot',      action='store_true', help='画训练曲线 (含于 summarize)')

    p.add_argument('--exp_dir',    type=str, help='单个实验目录 (--eval 时使用)')
    p.add_argument('--output_dir', type=str, default='./runs_topcow',
                   help='所有实验的父目录 (--eval_all / --summarize 时使用)')
    p.add_argument('--test_dir',   type=str, default='./crown_processed',
                   help='测试集目录')
    p.add_argument('--model_type', type=str, default=None, choices=ALL_MODEL_TYPES + [None],
                   help='模型类型 (默认从 config.json 读取)')
    p.add_argument('--no_tta',    action='store_true', help='关闭 TTA')
    p.add_argument('--eval_tag',  type=str, default=None,
                   help='结果文件后缀, 默认从 test_dir basename 自动推断')

    args = p.parse_args()

    if args.eval:
        if not args.exp_dir:
            p.error('--eval 需要 --exp_dir')
        run_eval(args)
    elif args.eval_all:
        run_eval_all(args)
    elif args.summarize or args.plot:
        run_summarize(args)
    else:
        p.print_help()


if __name__ == '__main__':
    main()
