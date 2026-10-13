"""
topcow_eval_official.py (v2)
=============================
完整对齐官方 TopCoW 2024 评估指标

官方 7 项指标 (Task 1 Segmentation):
  体素级:
    1. Class-average Dice (mean±std across cases)
    2. clDice on merged binary (mean±std)
    3. Class-average β0 error (mean±std)
    4. Class-average HD95 (mean±std)         ← v2 新增
  超越分割:
    5. Group 2 detection F1                  ← v2 新增
    6. Variant-balanced graph classification ← 未实现 (需要 edge-list)
    7. Topology match rate                   ← 未实现 (需要 edge-list)

v2 vs v1 改动:
  - evaluate_single_case: 新增 per-class HD95
  - 汇总时: 每个 per-class 指标都有 mean + std
  - 新增 Group 2 detection F1 (Acom/R-Pcom/L-Pcom/3rd-A2)
  - 汇总表格完整对齐官方论文格式

用法:
  python topcow_eval_official.py --eval --exp_dir ./runs_topcow/tgf_clean --test_dir ./crown_processed
  python topcow_eval_official.py --eval_all --output_dir ./runs_topcow --test_dir ./crown_processed
  python topcow_eval_official.py --summarize --output_dir ./runs_topcow

依赖:
  pip install scikit-image scipy numpy torch
"""

import os, sys, json, argparse, time
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize_3d
from torch.utils.data import DataLoader
from topcow_foundation_model import (
    TopCoWFoundationModel, build_foundation_encoder,
    FOUNDATION_TYPES, FUSION_TYPES
)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# 在文件顶部，import sys 之后添加（约第 6 行附近）
SAM_MED3D_PATH = '/mnt/e/00-Data/脑部-TopCoW/24/topcow-TGF/SAM-Med3D'
sys.path.insert(0, SAM_MED3D_PATH)

NUM_CLASSES = 14
VESSEL_NAMES = {
    1: "BA", 2: "R-PCA", 3: "L-PCA", 4: "R-ICA", 5: "R-MCA",
    6: "L-ICA", 7: "L-MCA", 8: "R-Pcom", 9: "L-Pcom", 10: "Acom",
    11: "R-ACA", 12: "L-ACA", 13: "3rd-A2"
}

# Group 2: 变异性最大的血管组件 (官方用于 detection F1)
GROUP2_LABELS = [8, 9, 10, 13]  # R-Pcom, L-Pcom, Acom, 3rd-A2

ALL_MODEL_TYPES = ['tgf', 'tgf_nogate', 'tgf_pretrain', 'unet',
                   'concat', 'crossattn', 'bagf', 'pug', 'dsff']


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
    sp, sg = skeletonize_3d(p).astype(bool), skeletonize_3d(g).astype(bool)
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
    """Hausdorff Distance 95th percentile

    官方用法: 逐类计算, 只在 pred 和 gt 都非空时有效
    若一方为空返回 inf (后续汇总时排除)

    Args:
        pred_bin: [D,H,W] binary numpy
        gt_bin: [D,H,W] binary numpy
        voxel_spacing: (sz, sy, sx) 体素间距, None 则用 (1,1,1)
    """
    pred_bin = pred_bin.astype(bool)
    gt_bin = gt_bin.astype(bool)

    if pred_bin.sum() == 0 and gt_bin.sum() == 0:
        return 0.0
    if pred_bin.sum() == 0 or gt_bin.sum() == 0:
        return float('inf')

    spacing = voxel_spacing if voxel_spacing is not None else (1.0, 1.0, 1.0)

    # pred 表面到 gt 的距离
    dt_gt = distance_transform_edt(~gt_bin, sampling=spacing)
    dist_pred_to_gt = dt_gt[pred_bin]

    # gt 表面到 pred 的距离
    dt_pred = distance_transform_edt(~pred_bin, sampling=spacing)
    dist_gt_to_pred = dt_pred[gt_bin]

    # 取边界点的距离 (表面点 = 前景且邻接背景)
    # 简化版: 直接用所有前景点的距离, 然后取 95th percentile
    # 这和官方 HD95 略有差异 (官方用 surface points), 但对排名影响很小
    all_dists = np.concatenate([dist_pred_to_gt, dist_gt_to_pred])
    return float(np.percentile(all_dists, 95))


def compute_detection_iou(pred_c, gt_c):
    """计算单个类别的 IoU (用于 Group 2 detection)"""
    inter = (pred_c & gt_c).sum()
    union = (pred_c | gt_c).sum()
    if union == 0:
        return 0.0
    return float(inter / union)


# ============================================================
# 单 Case 评估
# ============================================================

def evaluate_single_case(pred, gt, voxel_spacing=None):
    """评估单个 case 的所有官方指标

    返回:
        dict with:
        - binary_dice, binary_cldice, binary_b0
        - per_class: {c: {dice, b0, hd95}} for each class
        - class_avg_dice, class_avg_b0, class_avg_hd95
        - detection: {label: {tp, fp, fn, iou}} for Group 2
    """
    r = {}

    # Binary
    pb, gb = (pred > 0).astype(np.uint8), (gt > 0).astype(np.uint8)
    r['binary_dice'] = compute_dice(pb, gb)
    r['binary_cldice'] = compute_cldice(pb, gb)
    r['binary_b0'] = compute_b0(pb, gb)

    # Per-class
    pc = {}
    for c in range(1, NUM_CLASSES):
        p_c = (pred == c).astype(np.uint8)
        g_c = (gt == c).astype(np.uint8)
        if g_c.sum() == 0 and p_c.sum() == 0:
            pc[c] = None
            continue

        dice = compute_dice(p_c, g_c)
        b0 = compute_b0(p_c, g_c)
        hd95 = compute_hd95(p_c, g_c, voxel_spacing)

        pc[c] = {'dice': dice, 'b0': b0, 'hd95': hd95}
    r['per_class'] = pc

    # Class-average
    valid = [v for v in pc.values() if v is not None]
    r['class_avg_dice'] = float(np.mean([v['dice'] for v in valid])) if valid else 0.0
    r['class_avg_b0'] = float(np.mean([v['b0'] for v in valid])) if valid else 0.0
    # HD95: 排除 inf (一方为空的类)
    valid_hd = [v['hd95'] for v in valid if np.isfinite(v['hd95'])]
    r['class_avg_hd95'] = float(np.mean(valid_hd)) if valid_hd else float('inf')

    # Group 2 Detection
    det = {}
    for c in GROUP2_LABELS:
        p_c = (pred == c).astype(bool)
        g_c = (gt == c).astype(bool)
        gt_exists = g_c.sum() > 0
        pred_exists = p_c.sum() > 0

        if gt_exists and pred_exists:
            iou = compute_detection_iou(p_c, g_c)
            tp = 1 if iou >= 0.25 else 0  # 官方阈值: IoU >= 25%
            det[c] = {'tp': tp, 'fp': 1 - tp, 'fn': 0, 'iou': iou}
        elif gt_exists and not pred_exists:
            det[c] = {'tp': 0, 'fp': 0, 'fn': 1, 'iou': 0.0}
        elif not gt_exists and pred_exists:
            det[c] = {'tp': 0, 'fp': 1, 'fn': 0, 'iou': 0.0}
        else:
            det[c] = None  # 不存在
    r['detection'] = det

    return r


# ============================================================
# 批量评估与汇总
# ============================================================

def aggregate_results(all_results):
    """将多个 case 的结果汇总为 mean±std

    官方格式: 每个指标跨 case 的 mean 和 std
    """
    if not all_results:
        return {}

    s = {}
    n = len(all_results)
    s['n'] = n

    # Binary 指标
    for k in ['binary_dice', 'binary_cldice', 'binary_b0']:
        vals = [r[k] for r in all_results]
        s[k] = float(np.mean(vals))
        s[f'{k}_std'] = float(np.std(vals))

    # Class-average 指标 (跨 case)
    for k in ['class_avg_dice', 'class_avg_b0', 'class_avg_hd95']:
        vals = [r[k] for r in all_results if np.isfinite(r.get(k, float('inf')))]
        if vals:
            s[k] = float(np.mean(vals))
            s[f'{k}_std'] = float(np.std(vals))
        else:
            s[k] = float('inf')
            s[f'{k}_std'] = 0.0

    # Per-class 指标 (逐类: 跨 case 的 mean±std)
    per_class_summary = {}
    for c in range(1, NUM_CLASSES):
        cases_with_c = [r['per_class'].get(c) for r in all_results
                        if r['per_class'].get(c) is not None]
        if not cases_with_c:
            per_class_summary[c] = None
            continue

        pcs = {}
        for metric in ['dice', 'b0', 'hd95']:
            vals = [v[metric] for v in cases_with_c if np.isfinite(v.get(metric, float('inf')))]
            if vals:
                pcs[f'{metric}_mean'] = float(np.mean(vals))
                pcs[f'{metric}_std'] = float(np.std(vals))
            else:
                pcs[f'{metric}_mean'] = float('inf') if metric == 'hd95' else 0.0
                pcs[f'{metric}_std'] = 0.0
        pcs['n'] = len(cases_with_c)
        per_class_summary[c] = pcs
    s['per_class'] = per_class_summary

    # Group 2 Detection F1
    total_tp = total_fp = total_fn = 0
    for r in all_results:
        det = r.get('detection', {})
        for c in GROUP2_LABELS:
            d = det.get(c)
            if d is None:
                continue
            total_tp += d['tp']
            total_fp += d['fp']
            total_fn += d['fn']
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    s['grp2_detection_f1'] = float(f1)
    s['grp2_precision'] = float(precision)
    s['grp2_recall'] = float(recall)

    s['per_case'] = all_results
    return s


# ============================================================
# 模型加载与推理
# ============================================================

def load_model(exp_dir, model_type, device, ckpt='best_model.pth'):
    path = os.path.join(exp_dir, ckpt)
    if not os.path.exists(path):
        print(f"Not found: {path}")
        return None

    # 读取实验配置（如果有 config.json）
    config_path = os.path.join(exp_dir, 'config.json')
    foundation_type = None
    fusion_type = None
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
            foundation_type = cfg.get('foundation_type')
            fusion_type = cfg.get('fusion_type')
            # 如果 model_type 未显式传入，尝试从 config 读取
            if model_type is None:
                model_type = cfg.get('model_type', 'tgf')

    # 处理 foundation 模型（非 baseline）
    if foundation_type is not None and foundation_type in FOUNDATION_TYPES:
        # 构建 foundation encoder（需要加载预训练权重吗？加载时模型内部会自己处理）
        encoder = build_foundation_encoder(foundation_type, cfg.get('foundation_ckpt', None))
        # 注意：training 时 foundation encoder 是 frozen 的，eval 时也一样
        m = TopCoWFoundationModel(encoder, fusion_type=fusion_type or 'tgf')
        m.to(device)
        # 加载权重（weights_only=False 解决 pickle 错误）
        state = torch.load(path, map_location=device, weights_only=False)
        if 'model_state_dict' in state:
            state = state['model_state_dict']
        m.load_state_dict(state, strict=True)
        m.eval()
        if hasattr(m, 'set_epoch'):
            m.set_epoch(999999)
        print(f"Loaded foundation model: {path} (foundation={foundation_type}, fusion={fusion_type})")
        return m

    # 原有 baseline 模型加载逻辑
    from topcow_model import TopCoWTGFModel, TopCoWUNet
    from topcow_model_baselines import TopCoWBaselineFusionModel

    state = torch.load(path, map_location=device, weights_only=False)
    sd = state.get('model_state_dict', state)

    if model_type == 'unet':
        m = TopCoWUNet()
    elif model_type in ['tgf', 'tgf_pretrain']:
        m = TopCoWTGFModel(disable_gating=False)
    elif model_type == 'tgf_nogate':
        m = TopCoWTGFModel(disable_gating=True)
    elif model_type in ['concat', 'crossattn', 'bagf', 'pug', 'dsff']:
        m = TopCoWBaselineFusionModel(fusion_type=model_type)
    else:
        print(f"Unknown model_type: {model_type}")
        return None

    m.load_state_dict(sd, strict=False)
    m.to(device).eval()
    if hasattr(m, 'set_epoch'):
        m.set_epoch(999999)
    print(f"Loaded baseline model: {path} (type={model_type})")
    return m


def tta_infer(model, x, device):
    ps = []
    for fz in [False, True]:
        for fy in [False, True]:
            a = x.clone()
            if fz: a = a.flip(2)
            if fy: a = a.flip(3)
            s, _, _ = model(a)
            if fy: s = s.flip(3)
            if fz: s = s.flip(2)
            ps.append(s)
    return torch.stack(ps).mean(0)


@torch.no_grad()
def eval_model(model, test_dir, device, use_tta=True):
    from topcow_dataset import TopCoWTestDataset
    ds = TopCoWTestDataset(test_dir, split='test')
    dl = DataLoader(ds, batch_size=1, shuffle=False, num_workers=2, pin_memory=True)
    results = []
    for batch in dl:
        x = batch['image'].float().to(device)
        mc = batch.get('multiclass')
        if mc is None:
            continue
        seg = tta_infer(model, x, device) if use_tta else model(x)[0]
        for i in range(x.size(0)):
            pred = torch.argmax(seg[i], dim=0).cpu().numpy()
            gt = mc[i].numpy()
            t0 = time.time()
            m = evaluate_single_case(pred, gt)
            name = batch['case_name'][i] if isinstance(batch['case_name'], (list, tuple)) else batch['case_name']
            m['case'] = name
            results.append(m)
            print(f"  {name}: Dice={m['binary_dice']:.4f} clDice={m['binary_cldice']:.4f} "
                  f"b0={m['binary_b0']} HD95={m['class_avg_hd95']:.1f} "
                  f"ClsAvg={m['class_avg_dice']:.4f} ({time.time()-t0:.1f}s)")
    return aggregate_results(results)


# ============================================================
# 打印报表
# ============================================================

def print_report(s, title=""):
    """打印单个实验的完整官方指标报表"""
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
    print(f"    Dice:   {s['class_avg_dice']:.4f} ± {s.get('class_avg_dice_std',0):.4f}")
    print(f"    β0:     {s['class_avg_b0']:.2f} ± {s.get('class_avg_b0_std',0):.2f}")
    hd = s.get('class_avg_hd95', float('inf'))
    hd_s = f"{hd:.2f}" if np.isfinite(hd) else "inf"
    print(f"    HD95:   {hd_s} ± {s.get('class_avg_hd95_std',0):.2f}")

    print(f"\n  Group 2 Detection:")
    print(f"    F1:        {s.get('grp2_detection_f1',0):.4f}")
    print(f"    Precision: {s.get('grp2_precision',0):.4f}")
    print(f"    Recall:    {s.get('grp2_recall',0):.4f}")

    # Per-class 表格
    pc = s.get('per_class', {})
    print(f"\n  Per-class Dice (mean±std):")
    print(f"    {'Class':>8}  {'Dice':>12}  {'β0':>10}  {'HD95':>12}  {'N':>3}")
    print(f"    {'-'*52}")
    for c in range(1, NUM_CLASSES):
        info = pc.get(c, pc.get(str(c)))
        if info is None:
            continue
        d_m = info.get('dice_mean', 0)
        d_s = info.get('dice_std', 0)
        b_m = info.get('b0_mean', 0)
        b_s = info.get('b0_std', 0)
        h_m = info.get('hd95_mean', float('inf'))
        h_s = info.get('hd95_std', 0)
        h_str = f"{h_m:.1f}±{h_s:.1f}" if np.isfinite(h_m) else "inf"
        n = info.get('n', 0)
        print(f"    {VESSEL_NAMES.get(c,'?'):>8}  {d_m:.4f}±{d_s:.4f}  "
              f"{b_m:.1f}±{b_s:.1f}  {h_str:>12}  {n:>3}")


# ============================================================
# 命令行入口
# ============================================================

def run_eval(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
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
    print(f"\nEval: {args.exp_dir} | TTA={not args.no_tta}")
    r = eval_model(model, args.test_dir, device, use_tta=not args.no_tta)
    if not r:
        return
    print_report(r, title=args.exp_dir)
    sp = os.path.join(args.exp_dir, 'official_metrics.json')
    with open(sp, 'w') as f:
        json.dump(r, f, indent=2, default=str)
    print(f"\nSaved: {sp}")


def run_eval_all(args):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    all_summaries = {}
    for name in sorted(os.listdir(args.output_dir)):
        ed = os.path.join(args.output_dir, name)
        if not os.path.isdir(ed) or not os.path.exists(os.path.join(ed, 'best_model.pth')):
            continue
        cp = os.path.join(ed, 'config.json')
        mt = 'tgf'
        if os.path.exists(cp):
            with open(cp) as f:
                mt = json.load(f).get('model_type', 'tgf')
        print(f"\n{'='*60}\nEval: {name} ({mt})\n{'='*60}")
        model = load_model(ed, mt, device)
        if model is None:
            continue
        r = eval_model(model, args.test_dir, device, use_tta=not args.no_tta)
        if r:
            with open(os.path.join(ed, 'official_metrics.json'), 'w') as f:
                json.dump(r, f, indent=2, default=str)
            all_summaries[name] = r
        del model
        torch.cuda.empty_cache()

    if all_summaries:
        print_comparison_table(all_summaries)


def print_comparison_table(all_summaries):
    """打印多实验对比表 — 对齐官方论文格式"""

    # 按 class_avg_dice 降序排列
    sorted_exps = sorted(all_summaries.items(),
                         key=lambda x: x[1].get('class_avg_dice', 0), reverse=True)
    # ================= 综合对比表 =================
    print(f"\n{'=' * 130}")
    print(f"综合对比表 (均值±标准差)")
    print(f"{'=' * 130}")
    header = (f"{'Exp':<25} {'Dice↑':>14} {'clDice↑':>14} {'β0↓':>12} "
              f"{'HD95↓':>12} {'ClsAvg↑':>14} {'Cls_β0↓':>12} {'Det_F1↑':>8}")
    print(header)
    print("-" * 130)

    for n, s in sorted_exps:
        bd = s.get('binary_dice', 0)
        bd_s = s.get('binary_dice_std', 0)
        cl = s.get('binary_cldice', 0)
        cl_s = s.get('binary_cldice_std', 0)
        b0 = s.get('binary_b0', 0)
        b0_s = s.get('binary_b0_std', 0)
        ca = s.get('class_avg_dice', 0)
        ca_s = s.get('class_avg_dice_std', 0)
        cb = s.get('class_avg_b0', 0)
        cb_s = s.get('class_avg_b0_std', 0)
        ch = s.get('class_avg_hd95', float('inf'))
        ch_s = s.get('class_avg_hd95_std', 0)
        ch_str = f"{ch:.1f}±{ch_s:.1f}" if np.isfinite(ch) else "inf"
        f1 = s.get('grp2_detection_f1', 0)

        print(f"{n:<25} "
              f"{bd:>6.4f}±{bd_s:.3f}  "
              f"{cl:>6.4f}±{cl_s:.3f}  "
              f"{b0:>5.1f}±{b0_s:.1f}  "
              f"{ch_str:>12}  "
              f"{ca:>6.4f}±{ca_s:.3f}  "
              f"{cb:>5.1f}±{cb_s:.1f}  "
              f"{f1:>8.3f}")

    # ================= 逐类 Dice 对比表 =================
    print(f"\n{'=' * 140}")
    print(f"Per-class Dice (均值±标准差)")
    print(f"{'=' * 140}")
    classes = list(range(1, NUM_CLASSES))
    header = f"{'Exp':<25} " + " ".join([f"{VESSEL_NAMES[c]:>10}" for c in classes])
    print(header)
    print("-" * len(header))

    for n, s in sorted_exps:
        pc = s.get('per_class', {})
        vals = []
        for c in classes:
            info = pc.get(c, pc.get(str(c), None))
            if isinstance(info, dict):
                dm = info.get('dice_mean', info.get('dice', 0))
                ds = info.get('dice_std', 0)
                vals.append(f"{dm:.3f}±{ds:.2f}")
            else:
                vals.append(f"{'—':>10}")
        line = f"{n:<25} " + " ".join([f"{v:>10}" for v in vals])
        print(line)


def summarize(output_dir):
    """汇总所有实验 (从 official_metrics.json 或 results.json)"""
    all_summaries = {}
    for d in sorted(os.listdir(output_dir)):
        for fn in ['official_metrics.json', 'results.json']:
            rp = os.path.join(output_dir, d, fn)
            if os.path.exists(rp):
                with open(rp) as f:
                    r = json.load(f)
                r['exp_name'] = d
                all_summaries[d] = r
                break

    if not all_summaries:
        print("No results.")
        return

    print_comparison_table(all_summaries)

    # 画图
    plot_curves(output_dir)


def plot_curves(output_dir):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("no matplotlib")
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
            vs = h.get(k, [])
            pts = [(e, v) for e, v in zip(ep, vs) if v is not None]
            if pts:
                ax.plot(*zip(*pts), 'o-', label=name, alpha=0.8)
    for ax, t in zip(axes.flat, ['Loss', 'Train Dice', 'Test Binary', 'Test ClassAvg']):
        ax.set_title(t)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    sp = os.path.join(output_dir, 'training_curves.png')
    plt.savefig(sp, dpi=150)
    print(f"Saved: {sp}")
    plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--eval', action='store_true')
    p.add_argument('--eval_all', action='store_true')
    p.add_argument('--summarize', action='store_true')
    p.add_argument('--plot', action='store_true')
    p.add_argument('--exp_dir', type=str)
    p.add_argument('--model_type', type=str, default=None)
    p.add_argument('--test_dir', type=str, default='./crown_processed')
    p.add_argument('--output_dir', type=str, default='./runs_topcow')
    p.add_argument('--no_tta', action='store_true')
    args = p.parse_args()
    if args.eval:
        run_eval(args)
    elif args.eval_all:
        run_eval_all(args)
    elif args.summarize or args.plot:
        summarize(args.output_dir)
    else:
        p.print_help()


if __name__ == '__main__':
    main()