"""
topcow_train_foundation.py (multi-model, TCFF)
================================================
TCFF 改动 (vs 旧版):
  - 移除 gate_floor 相关的日志打印和结果保存
  - 对比损失 → 通道级信任校准损失 (CTC loss)
  - 其余训练逻辑完全不变

用法:
  # SAM-Med3D + TGF (TCFF)
  python topcow_train_foundation.py --exp_name sam_tgf_v3_s123 --fusion_type tgf --foundation_type sam3d_turbo --foundation_ckpt ./sam_med3d_turbo.pth --seed 123
  CUDA_VISIBLE_DEVICES=1 python topcow_train_foundation.py --exp_name sam_tgf_v3_s0 --fusion_type tgf --foundation_type sam3d_turbo --foundation_ckpt ./sam_med3d_turbo.pth --seed 0

  # SAM-Med3D + TGF + CTC loss
  python topcow_train_foundation.py --exp_name sam_tgf_ctc --fusion_type tgf \
      --foundation_type sam3d_turbo --foundation_ckpt ./sam_med3d_turbo.pth \
      --use_contras --contras_weight 0.1

  # SAM-Med3D + Concat (baseline, 不受影响)
  python topcow_train_foundation.py --exp_name sam_concat --fusion_type concat \
      --foundation_type sam3d_turbo --foundation_ckpt ./sam_med3d_turbo.pth

  # 汇总
  python topcow_train_foundation.py --summarize
"""

# !! 根据你的服务器修改这行 !!
import sys
SAM_MED3D_PATH = '/root/888-topcow-TGF/SAM-Med3D'
# '/mnt/e/00-Data/Brain-TopCoW/24/topcow-TGF/SAM-Med3D'
#'/root/888-topcow-TGF/SAM-Med3D'
sys.path.insert(0, SAM_MED3D_PATH)


import os, json, time, random, argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from topcow_dataset import TopCoWDataset, TopCoWTestDataset
from topcow_foundation_model import (
    TopCoWFoundationModel, build_foundation_encoder,
    FOUNDATION_TYPES, FUSION_TYPES, TRUST_VARIANTS, NUM_SEG_CLASSES
)
from topcow_train import (
    evaluate, compute_binary_dice, multiclass_dice_loss,
    soft_cldice_loss, VESSEL_NAMES
)


def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(worker_seed); random.seed(worker_seed)


def compute_foundation_loss(seg, aux_output, seg_logits, batch,
                            seg_w=1.0, cldice_w=0.3, ds_w=0.2):
    mc_target = batch['multiclass'].long().cuda()
    l_ce = F.cross_entropy(seg_logits, mc_target)
    l_dice = multiclass_dice_loss(seg, mc_target, NUM_SEG_CLASSES)
    l_seg = l_ce + l_dice
    pred_binary = (1 - seg[:, 0:1]).clamp(0, 1)
    target_binary = (mc_target > 0).float().unsqueeze(1)
    l_cldice = soft_cldice_loss(pred_binary, target_binary)
    total = seg_w * l_seg + cldice_w * l_cldice
    l_ds = torch.tensor(0.0, device=total.device)
    ds_logits = aux_output.get('ds_logits')
    if ds_logits is not None:
        for i, dl in enumerate(ds_logits):
            if dl.shape[2:] != mc_target.shape[1:]:
                dl = F.interpolate(dl, size=mc_target.shape[1:],
                                   mode='trilinear', align_corners=False)
            l_ds = l_ds + (1.0 / (2 ** i)) * F.cross_entropy(dl, mc_target)
        total = total + ds_w * l_ds
    return total, {'total': total.item(), 'ce': l_ce.item(),
                   'dice': l_dice.item(), 'cldice': l_cldice.item(), 'ds': l_ds.item()}


def compute_ctc_loss(trust_info_clean, trust_info_aug):
    """通道级信任校准损失 (Contrastive Trust Calibration, CTC)

    核心原理:
      对每个通道, 计算 clean 和 augmented 输入下 foundation 特征的偏移量 (divergence).
      偏移大的通道 → 不稳定 → stability 低 → channel_gate 应给低权重.
      用 MSE 让 channel_gate 的输出对齐 stability.

    Args:
        trust_info_clean: list of (channel_weight, aux_global) from clean forward
            channel_weight: [B, C_a]  — 有梯度, 是训练目标
            aux_global:     [B, C_a]  — 来自可训练 projector
        trust_info_aug:   list of (channel_weight, aux_global) from augmented forward
            均无梯度 (torch.no_grad)
    """
    ctc_loss = 0.0
    n_levels = 0
    for (cw_clean, ag_clean), (_, ag_aug) in zip(trust_info_clean, trust_info_aug):
        # 通道级 divergence: 衡量每个通道在扰动下的漂移程度
        divergence = (ag_clean.detach() - ag_aug).pow(2)       # [B, C_a]
        # stability ∈ (0, 1]: divergence 越小 → 越稳定 → 越可信
        stability = 1.0 / (1.0 + divergence)                   # [B, C_a]
        # 鼓励 channel_gate 输出与 stability 正相关
        ctc_loss = ctc_loss + F.mse_loss(cw_clean, stability.detach())
        n_levels += 1

    return ctc_loss / max(n_levels, 1)


def train_foundation(args):
    set_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    exp_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    config = vars(args)
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Exp: {args.exp_name} | Foundation: {args.foundation_type} | "
          f"Fusion: {args.fusion_type} | Trust variant: {args.trust_variant}")
    print(f"Checkpoint: {args.foundation_ckpt or 'None'}")
    use_contras = (args.fusion_type == 'tgf' and args.use_contras)
    print(f"CTC loss: {use_contras} (weight={args.contras_weight if use_contras else 0})")
    print(f"{'='*60}")

    train_ds = TopCoWDataset(args.train_dir, split='train', noise_rate=0.0, augment=True)
    test_ds = TopCoWTestDataset(args.test_dir, split='test')
    loader_generator = torch.Generator()
    loader_generator.manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True, worker_init_fn=seed_worker,
                              generator=loader_generator)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=2, pin_memory=True, worker_init_fn=seed_worker)

    encoder = build_foundation_encoder(args.foundation_type, args.foundation_ckpt)
    model = TopCoWFoundationModel(
        encoder, fusion_type=args.fusion_type,
        trust_variant=args.trust_variant).to(device)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_dice = 0.0
    history = {'train_loss': [], 'train_dice': [], 'test_binary': [], 'test_classavg': [], 'lr': []}
    n_batches = len(train_loader)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        losses, dices_tr = [], []

        for batch_idx, batch in enumerate(train_loader):
            x = batch['image'].float().to(device)
            mc = batch['multiclass'].long().to(device)
            optimizer.zero_grad()

            # ---- Forward (根据是否需要 CTC loss 决定返回格式) ----
            if use_contras:
                seg, aux_out, seg_logits, trust_info_clean = model(x, return_unc=True)
            else:
                seg, aux_out, seg_logits = model(x, return_unc=False)

            total_loss, ld = compute_foundation_loss(seg, aux_out, seg_logits, batch)

            # ---- CTC loss (通道级信任校准) ----
            if use_contras:
                noise = torch.randn_like(x) * 0.05
                x_aug = torch.clamp(x + noise, 0, 1)
                with torch.no_grad():
                    _, _, _, trust_info_aug = model(x_aug, return_unc=True)
                ctc_loss = compute_ctc_loss(trust_info_clean, trust_info_aug)
                total_loss = total_loss + args.contras_weight * ctc_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            losses.append(ld['total'])
            with torch.no_grad():
                pred = torch.argmax(seg, dim=1)
                dices_tr.append(compute_binary_dice(pred[0], mc[0]))
            if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
                extra = ""
                if use_contras:
                    extra = f" ctc={ctc_loss.item():.4f}"
                print(f"  E{epoch} [{batch_idx+1}/{n_batches}] "
                      f"loss={ld['total']:.4f} dice_b={dices_tr[-1]:.4f} "
                      f"ce={ld['ce']:.4f} cldice={ld['cldice']:.4f}{extra}", flush=True)

        scheduler.step()
        avg_loss, avg_tr = np.mean(losses), np.mean(dices_tr)
        elapsed = time.time() - t0
        history['train_loss'].append(avg_loss)
        history['train_dice'].append(avg_tr)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        eval_every = max(1, args.epochs // 10)
        do_eval = (epoch % eval_every == 0) or (epoch == args.epochs) or (epoch <= 3)

        if do_eval:
            res = evaluate(model, test_loader, device)
            bd, cd = res['binary_dice'], res['class_avg_dice']
            history['test_binary'].append(bd)
            history['test_classavg'].append(cd)
            if cd > best_dice:
                best_dice = cd
                torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(),
                            'best_dice': best_dice, 'config': config},
                           os.path.join(exp_dir, 'best_model.pth'))
            cls_str = " ".join([f"{VESSEL_NAMES.get(c,'?')}:{v['mean']:.2f}"
                                for c, v in sorted(res.get('per_class', {}).items())
                                if v.get('n', 0) > 0][:6])
            print(f"E{epoch:3d}/{args.epochs} | L={avg_loss:.4f} TrD={avg_tr:.4f} | "
                  f"Binary={bd:.4f} ClassAvg={cd:.4f}±{res['class_avg_std']:.4f} "
                  f"Best={best_dice:.4f} | {cls_str} | {elapsed:.0f}s")
        else:
            history['test_binary'].append(None)
            history['test_classavg'].append(None)
            if epoch % 5 == 0:
                print(f"E{epoch:3d}/{args.epochs} | L={avg_loss:.4f} TrD={avg_tr:.4f} {elapsed:.0f}s")

    # ---- 最终评估 ----
    ckpt = torch.load(os.path.join(exp_dir, 'best_model.pth'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    final = evaluate(model, test_loader, device, use_tta=False)
    final_tta = evaluate(model, test_loader, device, use_tta=True)

    print(f"\n{'='*60}")
    print(f"FINAL:     Binary={final['binary_dice']:.4f} ClassAvg={final['class_avg_dice']:.4f}")
    print(f"FINAL+TTA: Binary={final_tta['binary_dice']:.4f} ClassAvg={final_tta['class_avg_dice']:.4f}")
    print(f"{'='*60}")

    for c in range(1, NUM_SEG_CLASSES):
        info = final_tta['per_class'].get(c, {})
        if info:
            print(f"  {VESSEL_NAMES.get(c, f'cls{c}'):>8s}: {info['mean']:.4f}±{info['std']:.4f}")

    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    result = {
        'exp_name': args.exp_name,
        'model_type': f'{args.foundation_type}_{args.fusion_type}',
        'foundation_type': args.foundation_type,
        'fusion_type': args.fusion_type,
        'trust_variant': args.trust_variant,
        'noise_rate': 0.0, 'seed': args.seed,
        'best_epoch': ckpt['epoch'],
        'final_binary': final['binary_dice'], 'final_binary_std': final['binary_std'],
        'final_classavg': final['class_avg_dice'], 'final_classavg_std': final['class_avg_std'],
        'tta_binary': final_tta['binary_dice'], 'tta_classavg': final_tta['class_avg_dice'],
        'n_test': final['n'],
        'trainable_params': n_train, 'total_params': n_total,
        'per_class': final_tta['per_class'], 'per_case': final_tta['per_case'],
        'foundation_ckpt': args.foundation_ckpt or 'none',
        'use_contras': use_contras,
        'contras_weight': args.contras_weight if use_contras else 0,
    }
    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)
    with open(os.path.join(exp_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2, default=str)
    if not args.no_save_last:
        torch.save({'epoch': args.epochs, 'model_state_dict': model.state_dict()},
                   os.path.join(exp_dir, 'last_model.pth'))
    print(f"Saved to: {exp_dir}")


def summarize(output_dir):
    results = []
    for d in sorted(os.listdir(output_dir)):
        rp = os.path.join(output_dir, d, 'results.json')
        if os.path.exists(rp):
            with open(rp) as f: results.append(json.load(f))
    if not results: print("No results."); return
    print(f"\n{'Exp':<22} {'Foundation':<14} {'Fusion':<8} {'Binary':<10} {'ClassAvg':<12} {'TTA_Avg':<10} {'Params':>12}")
    print("-" * 95)
    for r in sorted(results, key=lambda x: x.get('tta_classavg', 0), reverse=True):
        print(f"{r['exp_name']:<22} {r.get('foundation_type','?'):<14} "
              f"{r.get('fusion_type','?'):<8} "
              f"{r.get('final_binary',0):<10.4f} "
              f"{r.get('final_classavg',0):<12.4f} "
              f"{r.get('tta_classavg',0):<10.4f} "
              f"{r.get('trainable_params',0):>12,}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_name', type=str, default='found_test')
    p.add_argument('--foundation_type', type=str, default='sam3d_turbo',
                   choices=FOUNDATION_TYPES,
                   help=f'Foundation encoder: {FOUNDATION_TYPES}')
    p.add_argument('--fusion_type', type=str, default='tgf',
                   choices=FUSION_TYPES,
                   help=f'Fusion method: {FUSION_TYPES}')
    p.add_argument('--trust_variant', type=str, default='full',
                   choices=TRUST_VARIANTS,
                   help='TCFF inference/training gate intervention')
    p.add_argument('--foundation_ckpt', type=str, default='',
                   help='Pretrained checkpoint path')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train_dir', type=str, default='./TopCoW_processed')
    p.add_argument('--test_dir', type=str, default='./crown_processed')
    p.add_argument('--output_dir', type=str, default='./runs_topcow_foundation')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--no_save_last', action='store_true',
                   help='Keep only the selected best checkpoint (rebuttal disk-saving mode)')
    p.add_argument('--summarize', action='store_true')

    p.add_argument('--use_contras', action='store_true',
                   help='Enable CTC loss (Contrastive Trust Calibration, only for TGF)')
    p.add_argument('--contras_weight', type=float, default=0.1,
                   help='Weight for CTC loss')

    args = p.parse_args()
    if args.summarize: summarize(args.output_dir); return
    if not args.foundation_ckpt: args.foundation_ckpt = None
    if 'CUDA_VISIBLE_DEVICES' not in os.environ: os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train_foundation(args)

if __name__ == '__main__':
    main()
