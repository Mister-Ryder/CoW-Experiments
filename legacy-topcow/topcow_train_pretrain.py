"""
topcow_train_pretrain.py
========================
两阶段预训练策略 (对应气道版的预训练分类器方案)

Stage 1: 只训练辅助 decoder + 三个头 (冻结 encoder 和 main decoder)
         → 辅助头学会可靠的 prediction 和 uncertainty
Stage 2: 冻结辅助 decoder, 训练 encoder + main decoder + 融合算子
         → 融合算子基于可靠的辅助信号工作

用法:
  CUDA_VISIBLE_DEVICES=0 python topcow_train_pretrain.py \
      --exp_name v4_pretrain_clean --noise_rate 0.0 \
      --stage1_epochs 100 --stage2_epochs 200 \
      --train_dir ./TopCoW_processed --test_dir ./crown_processed

python topcow_train_pretrain.py \
    --exp_name v4_pretrain_clean \
    --skip_stage1 --stage1_path ./runs_topcow_v4/v4_pretrain_clean/stage1_model.pth \
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
from topcow_model import TopCoWTGFModel, NUM_SEG_CLASSES
from topcow_train import (compute_loss, evaluate, compute_binary_dice, VESSEL_NAMES)


def train_pretrain(args):
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
    print(f"Exp: {args.exp_name} | TWO-STAGE PRETRAIN | Noise: {args.noise_rate}")
    print(f"Stage1: {args.stage1_epochs} ep (aux only) | Stage2: {args.stage2_epochs} ep (full)")
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

    # 模型
    model = TopCoWTGFModel(disable_gating=False).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {params:,}")

    history = {'stage': [], 'train_loss': [], 'test_binary': [],
               'test_classavg': [], 'lr': []}
    n_batches = len(train_loader)

    if not args.skip_stage1:
        # ================================================================
        # Stage 1: 只训练辅助 decoder + 三个头
        # ================================================================
        print(f"\n{'='*60}")
        print(f"STAGE 1: Pretrain auxiliary decoder ({args.stage1_epochs} epochs)")
        print(f"{'='*60}")

        # 冻结 encoder + main decoder + 融合算子
        for name, param in model.named_parameters():
            if any(k in name for k in ['aux_decoder', 'circuit_head', 'caliber_head',
                                        'laterality_head', 'temperature']):
                param.requires_grad = True
            else:
                param.requires_grad = False

        trainable_s1 = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Stage1 trainable: {trainable_s1:,} / {params:,}")

        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.lr, weight_decay=1e-4
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.stage1_epochs, eta_min=1e-6)

        for epoch in range(1, args.stage1_epochs + 1):
            t0 = time.time()
            model.train()
            losses = []

            for batch_idx, batch in enumerate(train_loader):
                x = batch['image'].float().to(device)

                optimizer.zero_grad()

                # Forward (只跑 encoder + aux decoder, 跳过 main decoder)
                seg, aux_out, seg_logits = model(x, aux_only=True)

                # 只算辅助头 loss
                ce = nn.CrossEntropyLoss()
                l_aux = torch.tensor(0.0, device=device)
                for lk, gk in [('circuit_logits', 'circuit_gt'),
                                ('caliber_logits', 'caliber_gt'),
                                ('laterality_logits', 'laterality_gt')]:
                    logits = aux_out[lk]
                    gt = batch[gk].long().to(device)
                    if logits.shape[2:] != gt.shape[1:]:
                        logits = F.interpolate(logits, size=gt.shape[1:],
                                               mode='trilinear', align_corners=False)
                    l_aux = l_aux + ce(logits, gt)
                l_aux = l_aux / 3.0

                l_aux.backward()
                optimizer.step()
                losses.append(l_aux.item())

                if (batch_idx + 1) % 20 == 0 or batch_idx == n_batches - 1:
                    print(f"  S1 E{epoch} [{batch_idx+1}/{n_batches}] "
                          f"aux_loss={losses[-1]:.4f}", flush=True)

            scheduler.step()
            avg_loss = np.mean(losses)
            history['stage'].append(1)
            history['train_loss'].append(avg_loss)
            history['lr'].append(optimizer.param_groups[0]['lr'])

            if epoch % 20 == 0 or epoch == args.stage1_epochs:
                print(f"S1 E{epoch}/{args.stage1_epochs} | AuxLoss={avg_loss:.4f} | "
                      f"{time.time()-t0:.0f}s")

            history['test_binary'].append(None)
            history['test_classavg'].append(None)

        # 保存 Stage 1 权重
        torch.save(model.state_dict(), os.path.join(exp_dir, 'stage1_model.pth'))
        print(f"\nStage 1 complete. Aux decoder pretrained.")
        print(f"Temperature: {model.temperature.item():.4f}")
    else :
        # 加载 stage1 权重
        stage1_path = args.stage1_path or os.path.join(exp_dir, 'stage1_model.pth')
        state = torch.load(stage1_path, map_location=device)
        if isinstance(state, dict) and 'model_state_dict' in state:
            model.load_state_dict(state['model_state_dict'])
        else:
            model.load_state_dict(state)
        # 重新设置梯度冻结状态（与原 Stage 2 开始前的设置完全一样）
        for name, param in model.named_parameters():
            if any(k in name for k in ['aux_decoder', 'circuit_head', 'caliber_head',
                                       'laterality_head']):
                param.requires_grad = False
            else:
                param.requires_grad = True
        model.temperature.requires_grad = True

    # --- 新增：清空 CUDA 缓存 ---
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    print("CUDA cache cleared between stages.")
    # ----------------------------

    # # ================================================================
    # # Stage 2: 冻结辅助 decoder, 训练其余部分
    # # ================================================================
    # print(f"\n{'='*60}")
    # print(f"STAGE 2: Train main decoder + fusion ({args.stage2_epochs} epochs)")
    # print(f"Auxiliary decoder FROZEN")
    # print(f"{'='*60}")
    #
    # # 冻结辅助 decoder, 解冻其余
    # for name, param in model.named_parameters():
    #     if any(k in name for k in ['aux_decoder', 'circuit_head', 'caliber_head',
    #                                 'laterality_head']):
    #         param.requires_grad = False
    #     else:
    #         param.requires_grad = True
    #
    # # temperature 保持可训练
    # model.temperature.requires_grad = True
    #
    # trainable_s2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"Stage2 trainable: {trainable_s2:,} / {params:,}")
    #
    # optimizer = torch.optim.AdamW(
    #     [p for p in model.parameters() if p.requires_grad],
    #     lr=args.lr, weight_decay=1e-4
    # )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    #     optimizer, T_max=args.stage2_epochs, eta_min=1e-6)
    #
    # best_dice = 0.0
    #
    # for epoch in range(1, args.stage2_epochs + 1):
    #     t0 = time.time()
    #     model.train()
    #     losses, dices_tr = [], []
    #
    #     for batch_idx, batch in enumerate(train_loader):
    #         x = batch['image'].float().to(device)
    #         mc = batch['multiclass'].long().to(device)
    #
    #         optimizer.zero_grad()
    #         seg, aux_out, seg_logits = model(x)
    #
    #         # 主分割 loss (不含辅助头 loss, 因为已冻结)
    #         total, ld = compute_loss(seg, aux_out, seg_logits, batch,
    #                                  model_type='tgf', aux_w=0.0)  # aux_w=0
    #         total.backward()
    #         torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    #         optimizer.step()
    #
    #         losses.append(ld['total'])
    #         with torch.no_grad():
    #             pred = torch.argmax(seg, dim=1)
    #             dices_tr.append(compute_binary_dice(pred[0], mc[0]))
    #
    #         if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
    #             print(f"  S2 E{epoch} [{batch_idx+1}/{n_batches}] "
    #                   f"loss={ld['total']:.4f} dice_b={dices_tr[-1]:.4f}", flush=True)
    #
    #     scheduler.step()
    #     avg_loss = np.mean(losses)
    #     avg_tr = np.mean(dices_tr)
    #     elapsed = time.time() - t0
    #
    #     history['stage'].append(2)
    #     history['train_loss'].append(avg_loss)
    #     history['train_dice'] = history.get('train_dice', [])
    #     history['train_dice'].append(avg_tr)
    #     history['lr'].append(optimizer.param_groups[0]['lr'])
    #
    #     eval_every = max(1, args.stage2_epochs // 10)
    #     do_eval = (epoch % eval_every == 0) or (epoch == args.stage2_epochs) or (epoch <= 3)
    #
    #     if do_eval:
    #         res = evaluate(model, test_loader, device)
    #         bd = res['binary_dice']
    #         cd = res['class_avg_dice']
    #         history['test_binary'].append(bd)
    #         history['test_classavg'].append(cd)
    #
    #         if cd > best_dice:
    #             best_dice = cd
    #             torch.save({
    #                 'epoch': epoch, 'model_state_dict': model.state_dict(),
    #                 'best_dice': best_dice, 'config': config,
    #             }, os.path.join(exp_dir, 'best_model.pth'))
    #
    #         print(f"S2 E{epoch:3d}/{args.stage2_epochs} | L={avg_loss:.4f} TrD={avg_tr:.4f} | "
    #               f"Binary={bd:.4f} ClassAvg={cd:.4f} Best={best_dice:.4f} | "
    #               f"Temp={model.temperature.item():.3f} {elapsed:.0f}s")
    #     else:
    #         history['test_binary'].append(None)
    #         history['test_classavg'].append(None)
    #         if epoch % 5 == 0:
    #             print(f"S2 E{epoch:3d}/{args.stage2_epochs} | L={avg_loss:.4f} TrD={avg_tr:.4f} "
    #                   f"{elapsed:.0f}s")
    # ================================================================
    # Stage 2: 联调 — 极低学习率更新 aux decoder + 正常训练主网络
    # ================================================================
    print(f"\n{'='*60}")
    print(f"STAGE 2: Joint training with slow aux update ({args.stage2_epochs} epochs)")
    print(f"Aux decoder lr = {args.lr * args.aux_lr_ratio:.2e} (ratio={args.aux_lr_ratio})")
    print(f"{'='*60}")

    # 所有参数均可训练，但通过优化器分组控制学习率
    aux_param_names = ['aux_decoder', 'circuit_head', 'caliber_head', 'laterality_head']

    aux_params = []
    main_params = []
    for name, param in model.named_parameters():
        if any(k in name for k in aux_param_names):
            param.requires_grad = True
            aux_params.append(param)
        else:
            param.requires_grad = True
            main_params.append(param)

    # temperature 也加入主网络组（或者独立，这里简单放入main）
    model.temperature.requires_grad = True
    if model.temperature not in main_params:
        main_params.append(model.temperature)

    trainable_s2 = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Stage2 trainable: {trainable_s2:,} / {params:,}")

    param_groups = [
        {'params': main_params, 'lr': args.lr},
        {'params': aux_params, 'lr': args.lr * args.aux_lr_ratio},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.stage2_epochs, eta_min=1e-6)

    best_dice = 0.0

    for epoch in range(1, args.stage2_epochs + 1):
        t0 = time.time()
        model.train()
        losses, dices_tr = [], []

        for batch_idx, batch in enumerate(train_loader):
            x = batch['image'].float().to(device)
            mc = batch['multiclass'].long().to(device)

            optimizer.zero_grad()
            seg, aux_out, seg_logits = model(x)

            # 使用完整的 loss，包括辅助头 loss（aux_w=0.3）
            total, ld = compute_loss(seg, aux_out, seg_logits, batch,
                                     model_type='tgf', aux_w=0.3)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            losses.append(ld['total'])
            with torch.no_grad():
                pred = torch.argmax(seg, dim=1)
                dices_tr.append(compute_binary_dice(pred[0], mc[0]))

            if (batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1:
                print(f"  S2 E{epoch} [{batch_idx+1}/{n_batches}] "
                      f"loss={ld['total']:.4f} dice_b={dices_tr[-1]:.4f}", flush=True)

        scheduler.step()
        avg_loss = np.mean(losses)
        avg_tr = np.mean(dices_tr)
        elapsed = time.time() - t0

        history['stage'].append(2)
        history['train_loss'].append(avg_loss)
        history['train_dice'] = history.get('train_dice', [])
        history['train_dice'].append(avg_tr)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        eval_every = max(1, args.stage2_epochs // 10)
        do_eval = (epoch % eval_every == 0) or (epoch == args.stage2_epochs) or (epoch <= 3)

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

            print(f"S2 E{epoch:3d}/{args.stage2_epochs} | L={avg_loss:.4f} TrD={avg_tr:.4f} | "
                  f"Binary={bd:.4f} ClassAvg={cd:.4f} Best={best_dice:.4f} | "
                  f"Temp={model.temperature.item():.3f} {elapsed:.0f}s")
        else:
            history['test_binary'].append(None)
            history['test_classavg'].append(None)
            if epoch % 5 == 0:
                print(f"S2 E{epoch:3d}/{args.stage2_epochs} | L={avg_loss:.4f} TrD={avg_tr:.4f} "
                      f"{elapsed:.0f}s")

    # 最终评估
    ckpt = torch.load(os.path.join(exp_dir, 'best_model.pth'), map_location=device)
    model.load_state_dict(ckpt['model_state_dict'])

    final = evaluate(model, test_loader, device, use_tta=False)
    final_tta = evaluate(model, test_loader, device, use_tta=True)

    print(f"\n{'='*60}")
    print(f"FINAL:     Binary={final['binary_dice']:.4f} ClassAvg={final['class_avg_dice']:.4f}")
    print(f"FINAL+TTA: Binary={final_tta['binary_dice']:.4f} ClassAvg={final_tta['class_avg_dice']:.4f}")
    print(f"{'='*60}")

    # 逐类
    for c in range(1, NUM_SEG_CLASSES):
        info = final_tta.get('per_class', {}).get(c, {})
        if info:
            print(f"  {VESSEL_NAMES.get(c,'?'):>8s}: {info['mean']:.4f}±{info['std']:.4f}")

    result = {
        'exp_name': args.exp_name, 'model_type': 'tgf_pretrain',
        'noise_rate': args.noise_rate, 'best_epoch': ckpt['epoch'],
        'stage1_epochs': args.stage1_epochs, 'stage2_epochs': args.stage2_epochs,
        'final_binary': final['binary_dice'], 'final_binary_std': final['binary_std'],
        'final_classavg': final['class_avg_dice'], 'final_classavg_std': final['class_avg_std'],
        'tta_binary': final_tta['binary_dice'], 'tta_classavg': final_tta['class_avg_dice'],
        'n_test': final['n'], 'params': params,
        'per_class': final_tta.get('per_class', {}),
        'per_case': final_tta.get('per_case', {}),
    }

    with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
        json.dump(result, f, indent=2, default=str)
    with open(os.path.join(exp_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2, default=str)

    print(f"Saved to: {exp_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--exp_name', type=str, required=True)
    p.add_argument('--noise_rate', type=float, default=0.0)
    p.add_argument('--train_dir', type=str, default='./TopCoW_processed')
    p.add_argument('--test_dir', type=str, default='./crown_processed')
    p.add_argument('--output_dir', type=str, default='./runs_topcow_v4')
    p.add_argument('--stage1_epochs', type=int, default=100)
    p.add_argument('--stage2_epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=2)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--num_workers', type=int, default=4)

    p.add_argument('--skip_stage1', action='store_true',
                   help='Skip Stage 1, load stage1_model.pth and run Stage 2 only')
    p.add_argument('--stage1_path', type=str, default='',
                   help='Path to stage1_model.pth (default: <exp_dir>/stage1_model.pth)')

    p.add_argument('--aux_lr_ratio', type=float, default=0.01,
                   help='Stage 2 learning rate ratio for aux decoder (default 0.01)')

    args = p.parse_args()

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    train_pretrain(args)


if __name__ == '__main__':
    main()