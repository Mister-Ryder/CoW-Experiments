"""
topcow_train_baselines.py (v4)
====================
TopCoW TGF 训练 — 14类多分类 + 逐类评估

Loss: multiclass CE + per-class Dice + clDice + 辅助头CE + 深监督
Eval: per-class Dice, class-avg Dice, binary Dice, clDice
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from topcow_dataset import TopCoWDataset, TopCoWTestDataset
from topcow_model import TopCoWTGFModel, TopCoWUNet, NUM_SEG_CLASSES

VESSEL_NAMES = {
    1: "BA", 2: "R-PCA", 3: "L-PCA", 4: "R-ICA", 5: "R-MCA",
    6: "L-ICA", 7: "L-MCA", 8: "R-Pcom", 9: "L-Pcom", 10: "Acom",
    11: "R-ACA", 12: "L-ACA", 13: "3rd-A2"
}


# ============================================================
# Loss
# ============================================================

def multiclass_dice_loss(pred_softmax, target, num_classes, smooth=1.0):
    """Per-class Dice loss, averaged over present classes"""
    dice_sum = 0.0
    count = 0
    for c in range(1, num_classes):  # skip background
        pred_c = pred_softmax[:, c]
        target_c = (target == c).float()
        if target_c.sum() == 0:
            continue
        inter = (pred_c * target_c).sum()
        dice = (2 * inter + smooth) / (pred_c.sum() + target_c.sum() + smooth)
        dice_sum += (1 - dice)
        count += 1
    return dice_sum / max(count, 1)


def soft_cldice_loss(pred_binary, target_binary, iters=3, smooth=1.0):
    """clDice on merged binary mask"""
    def soft_skel(x, iters):
        s = x.clone()
        for _ in range(iters):
            eroded = -F.max_pool3d(-s, 3, stride=1, padding=1)
            dilated = F.max_pool3d(eroded, 3, stride=1, padding=1)
            s = F.relu(s - dilated)
        return s

    sp = soft_skel(pred_binary, iters)
    st = soft_skel(target_binary, iters)
    tprec = ((sp * target_binary).sum(dim=(2, 3, 4))) / (sp.sum(dim=(2, 3, 4)) + smooth)
    tsens = ((st * pred_binary).sum(dim=(2, 3, 4))) / (st.sum(dim=(2, 3, 4)) + smooth)
    cl = 2.0 * tprec * tsens / (tprec + tsens + smooth)
    return 1.0 - cl.mean()


def compute_loss(seg, aux_output, seg_logits, batch, model_type,
                 seg_w=1.0, aux_w=0.3, cldice_w=0.3, ds_w=0.2):
    mc_target = batch['multiclass'].long().cuda()  # [B, D, H, W]

    # 1. 主分割 loss: CE + per-class Dice
    l_ce = F.cross_entropy(seg_logits, mc_target)
    l_dice = multiclass_dice_loss(seg, mc_target, NUM_SEG_CLASSES)
    l_seg = l_ce + l_dice

    # 2. clDice on binary
    pred_binary = (1 - seg[:, 0:1]).clamp(0, 1)  # 1 - background prob
    target_binary = (mc_target > 0).float().unsqueeze(1)
    l_cldice = soft_cldice_loss(pred_binary, target_binary)

    total = seg_w * l_seg + cldice_w * l_cldice

    # 3. 辅助头 loss
    l_aux = torch.tensor(0.0)
    if model_type in ['tgf', 'tgf_nogate']:
        ce = nn.CrossEntropyLoss()
        pairs = [
            ('circuit_logits', 'circuit_gt'),
            ('caliber_logits', 'caliber_gt'),
            ('laterality_logits', 'laterality_gt'),
        ]
        n = 0
        for lk, gk in pairs:
            logits = aux_output.get(lk)
            if logits is None or gk not in batch:
                continue
            gt = batch[gk].long().cuda()
            if logits.shape[2:] != gt.shape[1:]:
                logits = F.interpolate(logits, size=gt.shape[1:],
                                       mode='trilinear', align_corners=False)
            l_aux = l_aux + ce(logits, gt)
            n += 1
        if n > 0:
            l_aux = l_aux / n
            total = total + aux_w * l_aux

    # 4. 深监督 loss (14类)
    l_ds = torch.tensor(0.0)
    ds_logits = aux_output.get('ds_logits')
    if ds_logits is not None:
        for i, dl in enumerate(ds_logits):
            if dl.shape[2:] != mc_target.shape[1:]:
                dl = F.interpolate(dl, size=mc_target.shape[1:],
                                   mode='trilinear', align_corners=False)
            w = 1.0 / (2 ** i)
            l_ds = l_ds + w * F.cross_entropy(dl, mc_target)
        total = total + ds_w * l_ds

    return total, {
        'total': total.item(), 'ce': l_ce.item(), 'dice': l_dice.item(),
        'cldice': l_cldice.item(), 'aux': l_aux.item(), 'ds': l_ds.item(),
    }


# ============================================================
# 评估
# ============================================================

def compute_per_class_dice(pred_cls, target, num_classes=NUM_SEG_CLASSES, smooth=1.0):
    """计算逐类 Dice"""
    results = {}
    for c in range(1, num_classes):
        pred_c = (pred_cls == c).float()
        target_c = (target == c).float()
        inter = (pred_c * target_c).sum()
        union = pred_c.sum() + target_c.sum()
        if union == 0 and inter == 0:
            results[c] = None  # 类别不存在
        else:
            results[c] = ((2 * inter + smooth) / (union + smooth)).item()
    return results


def compute_binary_dice(pred_cls, target, smooth=1.0):
    pred_bin = (pred_cls > 0).float()
    target_bin = (target > 0).float()
    inter = (pred_bin * target_bin).sum()
    return ((2 * inter + smooth) / (pred_bin.sum() + target_bin.sum() + smooth)).item()


@torch.no_grad()
def evaluate(model, loader, device, use_tta=False):
    model.eval()
    all_results = []

    for batch in loader:
        x = batch['image'].float().to(device)
        mc_gt = batch.get('multiclass')
        if mc_gt is None:
            continue
        mc_gt = mc_gt.to(device)

        if use_tta:
            seg = tta_inference(model, x, device)
        else:
            seg, _, _ = model(x)

        for i in range(x.size(0)):
            pred_cls = torch.argmax(seg[i], dim=0)  # [D, H, W]
            gt = mc_gt[i]

            per_class = compute_per_class_dice(pred_cls, gt)
            binary_dice = compute_binary_dice(pred_cls, gt)

            # class-average (只算 GT 中存在的类)
            valid_dices = [v for v in per_class.values() if v is not None]
            class_avg = np.mean(valid_dices) if valid_dices else 0.0

            name = batch['case_name'][i] if isinstance(batch['case_name'], (list, tuple)) else batch['case_name']

            all_results.append({
                'case': name,
                'binary_dice': binary_dice,
                'class_avg_dice': class_avg,
                'per_class': per_class,
            })

    if not all_results:
        return {'binary_dice': 0, 'class_avg_dice': 0, 'std': 0, 'n': 0, 'per_case': {}}

    bd = [r['binary_dice'] for r in all_results]
    cd = [r['class_avg_dice'] for r in all_results]

    # 逐类汇总
    class_summary = {}
    for c in range(1, NUM_SEG_CLASSES):
        vals = [r['per_class'].get(c) for r in all_results if r['per_class'].get(c) is not None]
        if vals:
            class_summary[c] = {'mean': np.mean(vals), 'std': np.std(vals), 'n': len(vals)}

    per_case = {r['case']: {'binary': r['binary_dice'], 'class_avg': r['class_avg_dice']}
                for r in all_results}

    return {
        'binary_dice': np.mean(bd),
        'binary_std': np.std(bd),
        'class_avg_dice': np.mean(cd),
        'class_avg_std': np.std(cd),
        'n': len(all_results),
        'per_class': class_summary,
        'per_case': per_case,
    }


def tta_inference(model, x, device):
    """Z+Y 轴翻转 TTA (不翻转 X 避免 L/R 语义错乱)"""
    model.eval()
    preds = []
    for fz in [False, True]:
        for fy in [False, True]:
            x_aug = x.clone()
            if fz: x_aug = x_aug.flip(2)
            if fy: x_aug = x_aug.flip(3)
            seg, _, _ = model(x_aug)
            if fy: seg = seg.flip(3)
            if fz: seg = seg.flip(2)
            preds.append(seg)
    return torch.stack(preds).mean(dim=0)


# ============================================================
# 训练
# ============================================================

def train(args):
    # ================================================================
    # 严格随机种子固定（可复现）
    seed = 42
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # ================================================================

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    exp_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    config = vars(args)
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Exp: {args.exp_name} | Model: {args.model_type} | Noise: {args.noise_rate}")
    print(f"{'='*60}")

    # 数据
    train_ds = TopCoWDataset(args.train_dir, split='train',
                             noise_rate=args.noise_rate, augment=True)
    test_ds = TopCoWTestDataset(args.test_dir, split='test')

    # 固定 worker 种子，保证每次数据增强结果一致
    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=2, pin_memory=True,
                             worker_init_fn=seed_worker)

    print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")

    # 模型
    if args.model_type == 'unet':
        model = TopCoWUNet()
    elif args.model_type == 'tgf':
        model = TopCoWTGFModel(disable_gating=False)
    elif args.model_type == 'tgf_nogate':
        model = TopCoWTGFModel(disable_gating=True)
    else:
        raise ValueError(f"Unknown: {args.model_type}")

    model = model.to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    best_dice = 0.0
    history = {'train_loss': [], 'train_dice': [], 'test_binary': [],
               'test_classavg': [], 'lr': []}

    n_batches = len(train_loader)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        losses, dices_tr = [], []

        for batch_idx, batch in enumerate(train_loader):
            x = batch['image'].float().to(device)
            mc = batch['multiclass'].long().to(device)

            optimizer.zero_grad()
            seg, aux_out, seg_logits = model(x)
            total, ld = compute_loss(seg, aux_out, seg_logits, batch, args.model_type)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(ld['total'])
            with torch.no_grad():
                pred = torch.argmax(seg, dim=1)
                bd = compute_binary_dice(pred[0], mc[0])
                dices_tr.append(bd)

            if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
                print(f"  E{epoch} [{batch_idx+1}/{n_batches}] "
                      f"loss={ld['total']:.4f} dice_b={dices_tr[-1]:.4f} "
                      f"ce={ld['ce']:.4f} cldice={ld['cldice']:.4f}", flush=True)

        scheduler.step()
        avg_loss = np.mean(losses)
        avg_tr = np.mean(dices_tr)
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        history['train_loss'].append(avg_loss)
        history['train_dice'].append(avg_tr)
        history['lr'].append(lr)

        # 评估
        eval_every = max(1, args.epochs // 10)
        do_eval = (epoch % eval_every == 0) or (epoch == args.epochs) or (epoch <= 3)

        if do_eval:
            res = evaluate(model, test_loader, device)
            bd = res['binary_dice']
            cd = res['class_avg_dice']
            history['test_binary'].append(bd)
            history['test_classavg'].append(cd)

            if cd > best_dice:
                best_dice = cd
                torch.save({
                    'epoch': epoch, 'model_state_dict': model.state_dict(),
                    'best_dice': best_dice, 'config': config,
                }, os.path.join(exp_dir, 'best_model.pth'))

            # 打印逐类结果
            cls_str = " ".join([f"{VESSEL_NAMES.get(c,'?')}:{v['mean']:.2f}"
                                for c, v in sorted(res.get('per_class', {}).items())
                                if v['n'] > 0][:6])

            print(f"E{epoch:3d}/{args.epochs} | L={avg_loss:.4f} TrD={avg_tr:.4f} | "
                  f"Binary={bd:.4f} ClassAvg={cd:.4f}±{res['class_avg_std']:.4f} "
                  f"Best={best_dice:.4f} | {cls_str} | {elapsed:.0f}s")
        else:
            history['test_binary'].append(None)
            history['test_classavg'].append(None)
            if epoch % 5 == 0:
                print(f"E{epoch:3d}/{args.epochs} | L={avg_loss:.4f} TrD={avg_tr:.4f} "
                      f"LR={lr:.1e} {elapsed:.0f}s")

    # 最终评估
    ckpt = torch.load(os.path.join(exp_dir, 'best_model.pth'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    final = evaluate(model, test_loader, device, use_tta=False)
    final_tta = evaluate(model, test_loader, device, use_tta=True)

    print(f"\n{'='*60}")
    print(f"FINAL:     Binary={final['binary_dice']:.4f} ClassAvg={final['class_avg_dice']:.4f}")
    print(f"FINAL+TTA: Binary={final_tta['binary_dice']:.4f} ClassAvg={final_tta['class_avg_dice']:.4f}")
    print(f"{'='*60}")

    # 逐类详情
    print("\nPer-class Dice:")
    for c in range(1, NUM_SEG_CLASSES):
        info = final_tta['per_class'].get(c, {})
        if info:
            print(f"  {VESSEL_NAMES.get(c, f'cls{c}'):>8s}: {info['mean']:.4f}±{info['std']:.4f} (n={info['n']})")

    result = {
        'exp_name': args.exp_name, 'model_type': args.model_type,
        'noise_rate': args.noise_rate, 'best_epoch': ckpt['epoch'],
        'final_binary': final['binary_dice'], 'final_binary_std': final['binary_std'],
        'final_classavg': final['class_avg_dice'], 'final_classavg_std': final['class_avg_std'],
        'tta_binary': final_tta['binary_dice'], 'tta_classavg': final_tta['class_avg_dice'],
        'n_test': final['n'], 'params': params,
        'per_class': final_tta['per_class'],
        'per_case': final_tta['per_case'],
    }

    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)
    with open(os.path.join(exp_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2, default=str)
    torch.save({'epoch': args.epochs, 'model_state_dict': model.state_dict()},
               os.path.join(exp_dir, 'last_model.pth'))

    print(f"Saved to: {exp_dir}")
    return result


def summarize(output_dir):
    results = []
    for d in sorted(os.listdir(output_dir)):
        rp = os.path.join(output_dir, d, 'results.json')
        if os.path.exists(rp):
            with open(rp) as f:
                results.append(json.load(f))
    if not results:
        print("No results."); return

    print(f"\n{'Experiment':<22} {'Model':<12} {'Noise':<6} {'Binary':<10} {'ClassAvg':<12} {'TTA_Avg':<10}")
    print("-" * 80)
    for r in results:
        print(f"{r['exp_name']:<22} {r['model_type']:<12} {r['noise_rate']:<6.1f} "
              f"{r['final_binary']:<10.4f} {r['final_classavg']:<12.4f} {r.get('tta_classavg',0):<10.4f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_name', type=str, required=True)
    p.add_argument('--model_type', type=str, default='tgf',
                   choices=['tgf', 'tgf_nogate', 'unet'])
    p.add_argument('--noise_rate', type=float, default=0.0)
    p.add_argument('--train_dir', type=str, default='./TopCoW_processed')
    p.add_argument('--test_dir', type=str, default='./crown_processed')
    p.add_argument('--output_dir', type=str, default='./runs_topcow')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--summarize', action='store_true')
    args = p.parse_args()

    if args.summarize:
        summarize(args.output_dir); return

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    args.gpu = 0
    train(args)


if __name__ == '__main__':
    main()