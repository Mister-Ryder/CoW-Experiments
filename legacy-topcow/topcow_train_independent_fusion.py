"""
topcow_train_independent_fusion.py
==================================
独立分类器特征融合实验训练脚本

步骤:
  1. 训练独立解剖分类器 (或加载预训练权重)
  2. 冻结分类器, 训练融合模型 (Concat 或 TGF)

使用:
  # 训练分类器
  CUDA_VISIBLE_DEVICES=0 python topcow_train_independent_fusion.py --exp_name indcls_classifier --phase train_classifier

  # 训练 TGF 融合模型
CUDA_VISIBLE_DEVICES=0 python topcow_train_independent_fusion.py --exp_name indcls_tgf --fusion_type tgf --phase train_fusion --classifier_ckpt ./runs_indcls/indcls_classifier/best_model.pth > logs/cl_tgf.log 2>&1 &
CUDA_VISIBLE_DEVICES=1 python topcow_train_independent_fusion.py --exp_name indcls_concat --fusion_type concat --phase train_fusion --classifier_ckpt ./runs_indcls/indcls_classifier/best_model.pth > logs/cl_concat.log 2>&1 &
wait
echo "All experiments launched. Use 'tail -f logs/*.log' to monitor."
  # 训练 Concat 融合模型
  CUDA_VISIBLE_DEVICES=1 python topcow_train_independent_fusion.py --exp_name indcls_concat --fusion_type concat --phase train_fusion --classifier_ckpt ./runs_indcls/indcls_classifier/best_model.pth
"""

import os, sys, json, time, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from topcow_dataset import TopCoWDataset, TopCoWTestDataset
from topcow_anatomical_classifier import TopCoWAnatomicalClassifier
from topcow_model_independent_fusion import TopCoWIndependentClassifierFusionModel
from topcow_train_baselines import (
    set_seed, evaluate, compute_binary_dice,
    multiclass_dice_loss, soft_cldice_loss, VESSEL_NAMES
)

NUM_CLASSES = 14


def train_classifier(args):
    """第一阶段: 训练独立分类器"""
    set_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    exp_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # ==================== 实验头信息 ====================
    print(f"\n{'='*60}")
    print(f"Exp: {args.exp_name} | Phase: classifier | Noise: 0.0")
    print(f"{'='*60}")

    # ==================== 数据加载 ====================
    train_ds = TopCoWDataset(args.train_dir, split='train', augment=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True)

    print(f"[TopCoWDataset] train: {len(train_ds)}, noise=0.0, augment=True")
    print(f"Train samples: {len(train_ds)}")

    # ==================== 模型 ====================
    model = TopCoWAnatomicalClassifier().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Params: {params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    ce = nn.CrossEntropyLoss()
    best_loss = float('inf')
    n_batches = len(train_loader)

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss = 0.0

        for batch_idx, batch in enumerate(train_loader):
            x = batch['image'].float().to(device)
            ci = batch['circuit_gt'].long().to(device)
            ca = batch['caliber_gt'].long().to(device)
            la = batch['laterality_gt'].long().to(device)

            out = model(x)
            circuit_logits = F.interpolate(out['circuit_logits'], size=ci.shape[1:],
                                           mode='trilinear', align_corners=False)
            caliber_logits = F.interpolate(out['caliber_logits'], size=ca.shape[1:],
                                           mode='trilinear', align_corners=False)
            laterality_logits = F.interpolate(out['laterality_logits'], size=la.shape[1:],
                                              mode='trilinear', align_corners=False)

            loss = (ce(circuit_logits, ci) + ce(caliber_logits, ca) + ce(laterality_logits, la)) / 3

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # 每 10 个 batch 打印一次进度（与 v4 训练脚本一致）
            if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
                avg_loss_sofar = total_loss / (batch_idx + 1)
                print(f"  E{epoch} [{batch_idx+1}/{n_batches}] "
                      f"loss={loss.item():.4f} avg_loss={avg_loss_sofar:.4f}", flush=True)

        scheduler.step()
        avg_loss = total_loss / n_batches
        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        print(f"E{epoch:3d}/{args.epochs} | L={avg_loss:.4f} "
              f"LR={lr:.1e} {elapsed:.0f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(exp_dir, 'best_model.pth'))

    print(f"Classifier training finished. Best loss={best_loss:.4f}")


def train_fusion(args):
    """第二阶段: 训练融合模型 (分类器冻结)"""
    set_seed(args.seed)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    exp_dir = os.path.join(args.output_dir, args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    # 保存配置 (与v4一致)
    config = vars(args)
    with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)

    # 实验头信息 (与v4一致)
    print(f"\n{'='*60}")
    print(f"Exp: {args.exp_name} | Fusion: {args.fusion_type} | Noise: 0.0 | Seed: {args.seed}")
    print(f"{'='*60}")

    # 加载冻结分类器
    classifier = TopCoWAnatomicalClassifier()
    classifier.load_state_dict(torch.load(args.classifier_ckpt, map_location=device))
    classifier.to(device).eval()
    for p in classifier.parameters():
        p.requires_grad = False
    print(f"Loaded frozen classifier from {args.classifier_ckpt}")

    # 数据加载 (与v4一致)
    train_ds = TopCoWDataset(args.train_dir, split='train', augment=True)
    test_ds = TopCoWTestDataset(args.test_dir, split='test')
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, pin_memory=True, drop_last=True,
                              worker_init_fn=seed_worker)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                             num_workers=2, pin_memory=True, worker_init_fn=seed_worker)
    print(f"Train: {len(train_ds)} | Test: {len(test_ds)}")

    # 模型 (与v4一致)
    model = TopCoWIndependentClassifierFusionModel(classifier, fusion_type=args.fusion_type,disable_gating=args.disable_gating).to(device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable params: {params:,} | Total: {total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

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

            # 损失计算 (复用 v4 的损失函数)
            # 注意：这里我们需要使用 compute_loss 的类似格式，但没有辅助头 loss，
            # 所以我们手动构建一个兼容的 loss 字典
            l_ce = F.cross_entropy(seg_logits, mc)
            l_dice = multiclass_dice_loss(seg, mc, NUM_CLASSES)
            l_seg = l_ce + l_dice

            pred_binary = (1 - seg[:, 0:1]).clamp(0, 1)
            target_binary = (mc > 0).float().unsqueeze(1)
            l_cldice = soft_cldice_loss(pred_binary, target_binary)

            total = l_seg + 0.3 * l_cldice
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(total.item())
            with torch.no_grad():
                pred = torch.argmax(seg, dim=1)
                dices_tr.append(compute_binary_dice(pred[0], mc[0]))

            # 打印 batch 进度 (与v4完全一致的格式)
            if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
                print(f"  E{epoch} [{batch_idx+1}/{n_batches}] "
                      f"loss={total.item():.4f} dice_b={dices_tr[-1]:.4f} "
                      f"ce={l_ce.item():.4f} cldice={l_cldice.item():.4f}", flush=True)

        scheduler.step()
        avg_loss = np.mean(losses)
        avg_tr = np.mean(dices_tr)
        lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - t0

        history['train_loss'].append(avg_loss)
        history['train_dice'].append(avg_tr)
        history['lr'].append(lr)

        # 评估周期 (与v4一致)
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
                print(f"E{epoch:3d}/{args.epochs} | L={avg_loss:.4f} TrD={avg_tr:.4f} "
                      f"LR={lr:.1e} {elapsed:.0f}s")

    # 最终评估 (与v4一致)
    ckpt = torch.load(os.path.join(exp_dir, 'best_model.pth'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])
    final = evaluate(model, test_loader, device, use_tta=False)
    final_tta = evaluate(model, test_loader, device, use_tta=True)

    print(f"\n{'='*60}")
    print(f"FINAL:     Binary={final['binary_dice']:.4f} ClassAvg={final['class_avg_dice']:.4f}")
    print(f"FINAL+TTA: Binary={final_tta['binary_dice']:.4f} ClassAvg={final_tta['class_avg_dice']:.4f}")
    print(f"{'='*60}")

    for c in range(1, NUM_CLASSES):
        info = final_tta['per_class'].get(c, {})
        if info:
            print(f"  {VESSEL_NAMES.get(c, f'cls{c}'):>8s}: {info['mean']:.4f}±{info['std']:.4f} (n={info['n']})")

    # 保存结果 (与v4完全一致)
    result = {
        'exp_name': args.exp_name, 'fusion_type': args.fusion_type,
        'noise_rate': 0.0, 'seed': args.seed,
        'best_epoch': ckpt['epoch'],
        'final_binary': final['binary_dice'], 'final_binary_std': final['binary_std'],
        'final_classavg': final['class_avg_dice'], 'final_classavg_std': final['class_avg_std'],
        'tta_binary': final_tta['binary_dice'], 'tta_classavg': final_tta['class_avg_dice'],
        'n_test': final['n'], 'trainable_params': params,
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
    print("Fusion training finished.")

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_name', type=str, required=True)
    p.add_argument('--phase', type=str, choices=['train_classifier', 'train_fusion'], required=True)
    # 扩展 fusion_type 选择，包含所有已实现的融合方法
    p.add_argument('--fusion_type', type=str, default='tgf',
                   choices=['tgf', 'concat', 'crossattn', 'bagf', 'dsff'])
    p.add_argument('--classifier_ckpt', type=str, default='./runs_indcls/indcls_classifier/best_model.pth')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train_dir', type=str, default='./TopCoW_processed')
    p.add_argument('--test_dir', type=str, default='./crown_processed')
    p.add_argument('--output_dir', type=str, default='./runs_indcls')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--gpu', type=int, default=0)

    p.add_argument('--disable_gating', action='store_true',
                   help='Ablation: keep conditional routing but disable uncertainty gating')

    args = p.parse_args()

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    args.gpu = 0

    if args.phase == 'train_classifier':
        train_classifier(args)
    elif args.phase == 'train_fusion':
        if not args.classifier_ckpt:
            raise ValueError("Must provide --classifier_ckpt for fusion training")
        train_fusion(args)


if __name__ == '__main__':
    main()