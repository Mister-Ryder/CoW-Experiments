"""
topcow_train_foundation.py (DDP multi-GPU version, v3/TCFF compatible)
======================================================================
Only training script is modified. Supports single GPU and torchrun DDP.
Compatible with v3 FoundationTGF/TCFF: return_unc -> trust_info and CTC loss.

Single GPU:
  CUDA_VISIBLE_DEVICES=0 python topcow_train_foundation.py --exp_name sam_tgf ...

2 GPUs, global batch_size=2 -> per-GPU batch=1:
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 topcow_train_foundation.py \
      --exp_name sam_crossattn --foundation_type sam3d_turbo --fusion_type crossattn \
      --foundation_ckpt ./sam_med3d_turbo.pth --epochs 200 --batch_size 2
"""

# !! 根据你的服务器修改这行 !!
import sys
# SAM_MED3D_PATH = '/mnt/e/00-Data/脑部-TopCoW/24/topcow-TGF/SAM-Med3D'
SAM_MED3D_PATH = '/root/888-topcow-TGF/SAM-Med3D'
sys.path.insert(0, SAM_MED3D_PATH)

import os, json, time, random, argparse
import numpy as np
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from topcow_dataset import TopCoWDataset, TopCoWTestDataset
from topcow_foundation_model import (
    TopCoWFoundationModel, build_foundation_encoder,
    FOUNDATION_TYPES, FUSION_TYPES, NUM_SEG_CLASSES
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
    # DDP 下不同 rank 的 DataLoader worker 会各自继承 rank seed；这里保持可复现即可
    np.random.seed(42 + worker_id); random.seed(42 + worker_id)


def setup_distributed(args):
    """torchrun 启动时自动进入 DDP；普通 python 启动时保持单卡逻辑。"""
    args.world_size = int(os.environ.get('WORLD_SIZE', '1'))
    args.rank = int(os.environ.get('RANK', '0'))
    args.local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    args.distributed = args.world_size > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        dist.init_process_group(backend='nccl', init_method='env://')
        device = torch.device(f'cuda:{args.local_rank}')
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return device


def cleanup_distributed(args):
    if getattr(args, 'distributed', False) and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def is_main_process(args):
    return (not getattr(args, 'distributed', False)) or args.rank == 0


def print_main(args, *msg, **kwargs):
    if is_main_process(args):
        print(*msg, **kwargs)


def unwrap_model(model):
    return model.module if isinstance(model, DDP) else model


def reduce_mean_scalar(value, args, device):
    """把各 rank 的 epoch 均值再平均，便于主进程打印。"""
    t = torch.tensor(float(value), device=device)
    if getattr(args, 'distributed', False):
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= args.world_size
    return t.item()


def compute_foundation_loss(seg, aux_output, seg_logits, batch,
                            seg_w=1.0, cldice_w=0.3, ds_w=0.2):
    # 关键修改：不要硬编码 .cuda()，否则 DDP 的 local_rank != 0 时可能把 target 放错卡
    device = seg_logits.device
    mc_target = batch['multiclass'].long().to(device, non_blocking=True)

    l_ce = F.cross_entropy(seg_logits, mc_target)
    l_dice = multiclass_dice_loss(seg, mc_target, NUM_SEG_CLASSES)
    l_seg = l_ce + l_dice

    pred_binary = (1 - seg[:, 0:1]).clamp(0, 1)
    target_binary = (mc_target > 0).float().unsqueeze(1)
    l_cldice = soft_cldice_loss(pred_binary, target_binary)

    total = seg_w * l_seg + cldice_w * l_cldice
    l_ds = torch.tensor(0.0, device=device)
    ds_logits = aux_output.get('ds_logits') if aux_output is not None else None
    if ds_logits is not None:
        for i, dl in enumerate(ds_logits):
            if dl.shape[2:] != mc_target.shape[1:]:
                dl = F.interpolate(dl, size=mc_target.shape[1:],
                                   mode='trilinear', align_corners=False)
            l_ds = l_ds + (1.0 / (2 ** i)) * F.cross_entropy(dl, mc_target)
        total = total + ds_w * l_ds

    return total, {
        'total': float(total.detach().item()),
        'ce': float(l_ce.detach().item()),
        'dice': float(l_dice.detach().item()),
        'cldice': float(l_cldice.detach().item()),
        'ds': float(l_ds.detach().item()),
    }


def compute_ctc_loss(trust_info_clean, trust_info_aug):
    """Channel-level Trust Calibration loss for v3/TCFF TGF.

    v3 FoundationTGF returns, for each fusion level:
      (channel_weight, aux_global)
    where channel_weight is trainable and aux_global is the channel descriptor.
    We use augmentation-induced channel drift to construct a stability target.
    """
    ctc_loss = torch.tensor(0.0, device=trust_info_clean[0][0].device)
    n_levels = 0
    for (cw_clean, ag_clean), (_, ag_aug) in zip(trust_info_clean, trust_info_aug):
        divergence = (ag_clean.detach() - ag_aug).pow(2)
        stability = 1.0 / (1.0 + divergence)
        ctc_loss = ctc_loss + F.mse_loss(cw_clean, stability.detach())
        n_levels += 1
    return ctc_loss / max(n_levels, 1)


def train_foundation(args):
    device = setup_distributed(args)
    set_seed(args.seed + getattr(args, 'rank', 0))

    exp_dir = os.path.join(args.output_dir, args.exp_name)
    if is_main_process(args):
        os.makedirs(exp_dir, exist_ok=True)
        config = vars(args).copy()
        with open(os.path.join(exp_dir, 'config.json'), 'w') as f:
            json.dump(config, f, indent=2)
    else:
        config = vars(args).copy()

    if getattr(args, 'distributed', False):
        dist.barrier()

    print_main(args, f"\n{'='*60}")
    print_main(args, f"Exp: {args.exp_name} | Foundation: {args.foundation_type} | Fusion: {args.fusion_type}")
    print_main(args, f"Checkpoint: {args.foundation_ckpt or 'None'}")
    print_main(args, f"DDP: {args.distributed} | world_size={args.world_size} | global_batch={args.batch_size}")
    print_main(args, f"CTC loss: {args.use_contras and args.fusion_type == 'tgf'} "
                     f"(weight={args.contras_weight if args.use_contras else 0})")
    print_main(args, f"{'='*60}")

    train_ds = TopCoWDataset(args.train_dir, split='train', noise_rate=0.0, augment=True)

    # 这里把 --batch_size 解释为“全局 batch size”。双卡 batch_size=2 时，每卡 batch=1。
    if args.distributed:
        per_gpu_batch = max(1, args.batch_size // args.world_size)
        if args.batch_size % args.world_size != 0 and is_main_process(args):
            print(f"[Warn] global batch_size={args.batch_size} 不能被 world_size={args.world_size} 整除，"
                  f"将使用 per_gpu_batch={per_gpu_batch}，实际 global={per_gpu_batch * args.world_size}")
        train_sampler = DistributedSampler(train_ds, num_replicas=args.world_size,
                                           rank=args.rank, shuffle=True, drop_last=True)
        shuffle = False
    else:
        per_gpu_batch = args.batch_size
        train_sampler = None
        shuffle = True

    # DDP 下每个进程都有自己的 worker，num_workers 不宜过大；双卡可先用 2。
    train_loader = DataLoader(train_ds, batch_size=per_gpu_batch, shuffle=shuffle,
                              sampler=train_sampler,
                              num_workers=args.num_workers, pin_memory=True,
                              drop_last=True, worker_init_fn=seed_worker)

    if is_main_process(args):
        test_ds = TopCoWTestDataset(args.test_dir, split='test')
        test_loader = DataLoader(test_ds, batch_size=1, shuffle=False,
                                 num_workers=2, pin_memory=True, worker_init_fn=seed_worker)
    else:
        test_loader = None

    encoder = build_foundation_encoder(args.foundation_type, args.foundation_ckpt)
    model = TopCoWFoundationModel(encoder, fusion_type=args.fusion_type).to(device)

    if args.distributed:
        # find_unused_parameters=True 对 TGF + return_unc 分支更稳；其它融合也可正常运行。
        model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank,
                    find_unused_parameters=True)

    use_contras = (args.fusion_type == 'tgf' and args.use_contras)

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6)

    scaler = torch.cuda.amp.GradScaler(enabled=args.amp)

    best_dice = 0.0
    history = {'train_loss': [], 'train_dice': [], 'test_binary': [], 'test_classavg': [], 'lr': []}
    n_batches = len(train_loader)
    best_path = os.path.join(exp_dir, 'best_model.pth')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        model.train()
        losses, dices_tr = [], []

        for batch_idx, batch in enumerate(train_loader):
            x = batch['image'].float().to(device, non_blocking=True)
            mc = batch['multiclass'].long().to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=args.amp):
                if use_contras:
                    seg, aux_out, seg_logits, trust_info_clean = model(x, return_unc=True)
                else:
                    seg, aux_out, seg_logits = model(x, return_unc=False)

                total_loss, ld = compute_foundation_loss(seg, aux_out, seg_logits, batch)

                if use_contras:
                    noise = torch.randn_like(x) * 0.05
                    x_aug = torch.clamp(x + noise, 0, 1)
                    with torch.no_grad():
                        _, _, _, trust_info_aug = model(x_aug, return_unc=True)
                    ctc_loss = compute_ctc_loss(trust_info_clean, trust_info_aug)
                    total_loss = total_loss + args.contras_weight * ctc_loss

            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()

            losses.append(ld['total'])
            with torch.no_grad():
                pred = torch.argmax(seg, dim=1)
                dices_tr.append(compute_binary_dice(pred[0], mc[0]))

            if is_main_process(args) and ((batch_idx + 1) % 10 == 0 or batch_idx == n_batches - 1):
                extra = f" ctc={ctc_loss.item():.4f}" if use_contras else ""
                print(f"  E{epoch} [{batch_idx+1}/{n_batches}] "
                      f"loss={ld['total']:.4f} dice_b={dices_tr[-1]:.4f} "
                      f"ce={ld['ce']:.4f} cldice={ld['cldice']:.4f}{extra}", flush=True)

        scheduler.step()
        local_avg_loss = float(np.mean(losses)) if losses else 0.0
        local_avg_tr = float(np.mean(dices_tr)) if dices_tr else 0.0
        avg_loss = reduce_mean_scalar(local_avg_loss, args, device)
        avg_tr = reduce_mean_scalar(local_avg_tr, args, device)
        elapsed = time.time() - t0

        if is_main_process(args):
            history['train_loss'].append(avg_loss)
            history['train_dice'].append(avg_tr)
            history['lr'].append(optimizer.param_groups[0]['lr'])

        eval_every = max(1, args.epochs // 10)
        do_eval = (epoch % eval_every == 0) or (epoch == args.epochs) or (epoch <= 3)

        if do_eval:
            if is_main_process(args):
                raw_model = unwrap_model(model)
                res = evaluate(raw_model, test_loader, device)
                bd, cd = res['binary_dice'], res['class_avg_dice']
                history['test_binary'].append(bd)
                history['test_classavg'].append(cd)

                if cd > best_dice or not os.path.exists(best_path):
                    best_dice = cd
                    torch.save({'epoch': epoch,
                                'model_state_dict': raw_model.state_dict(),
                                'best_dice': best_dice,
                                'config': config}, best_path)

                cls_str = " ".join([f"{VESSEL_NAMES.get(c,'?')}:{v['mean']:.2f}"
                                    for c, v in sorted(res.get('per_class', {}).items())
                                    if v.get('n', 0) > 0][:6])
                print(f"E{epoch:3d}/{args.epochs} | L={avg_loss:.4f} TrD={avg_tr:.4f} | "
                      f"Binary={bd:.4f} ClassAvg={cd:.4f}±{res['class_avg_std']:.4f} "
                      f"Best={best_dice:.4f} | {cls_str} | {elapsed:.0f}s")
            if args.distributed:
                dist.barrier()
        else:
            if is_main_process(args):
                history['test_binary'].append(None)
                history['test_classavg'].append(None)
                if epoch % 5 == 0:
                    print(f"E{epoch:3d}/{args.epochs} | L={avg_loss:.4f} TrD={avg_tr:.4f} {elapsed:.0f}s")

    if args.distributed:
        dist.barrier()

    # 最终评估、写结果、保存 last 只在主进程做，避免多进程写同一文件。
    if is_main_process(args):
        raw_model = unwrap_model(model)
        ckpt = torch.load(best_path, map_location=device)
        raw_model.load_state_dict(ckpt['model_state_dict'])
        final = evaluate(raw_model, test_loader, device, use_tta=False)
        final_tta = evaluate(raw_model, test_loader, device, use_tta=True)

        print(f"\n{'='*60}")
        print(f"FINAL:     Binary={final['binary_dice']:.4f} ClassAvg={final['class_avg_dice']:.4f}")
        print(f"FINAL+TTA: Binary={final_tta['binary_dice']:.4f} ClassAvg={final_tta['class_avg_dice']:.4f}")
        print(f"{'='*60}")

        for c in range(1, NUM_SEG_CLASSES):
            info = final_tta['per_class'].get(c, {})
            if info:
                print(f"  {VESSEL_NAMES.get(c, f'cls{c}'):>8s}: {info['mean']:.4f}±{info['std']:.4f}")

        n_train = sum(p.numel() for p in raw_model.parameters() if p.requires_grad)
        n_total = sum(p.numel() for p in raw_model.parameters())
        result = {
            'exp_name': args.exp_name,
            'model_type': f'{args.foundation_type}_{args.fusion_type}',
            'foundation_type': args.foundation_type,
            'fusion_type': args.fusion_type,
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
            'distributed': args.distributed,
            'world_size': args.world_size,
            'per_gpu_batch': per_gpu_batch,
            'global_batch_effective': per_gpu_batch * args.world_size,
            'amp': args.amp,
        }
        with open(os.path.join(exp_dir, 'results.json'), 'w') as f:
            json.dump(result, f, indent=2, default=str)
        with open(os.path.join(exp_dir, 'history.json'), 'w') as f:
            json.dump(history, f, indent=2, default=str)
        torch.save({'epoch': args.epochs, 'model_state_dict': raw_model.state_dict()},
                   os.path.join(exp_dir, 'last_model.pth'))
        print(f"Saved to: {exp_dir}")

    cleanup_distributed(args)


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
    p.add_argument('--foundation_ckpt', type=str, default='',
                   help='Pretrained checkpoint path')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--train_dir', type=str, default='./TopCoW_processed')
    p.add_argument('--test_dir', type=str, default='./crown_processed')
    p.add_argument('--output_dir', type=str, default='./runs_topcow_foundation')
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch_size', type=int, default=2,
                   help='Global batch size. In DDP, per-GPU batch = batch_size // world_size.')
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--num_workers', type=int, default=4)
    p.add_argument('--summarize', action='store_true')
    p.add_argument('--amp', action='store_true',
                   help='Use mixed precision to reduce memory. It cannot fix N^2 attention by itself.')

    p.add_argument('--use_contras', action='store_true', help='Enable CTC loss (only for v3/TCFF TGF)')
    p.add_argument('--contras_weight', type=float, default=0.1, help='Weight for CTC loss')

    args = p.parse_args()
    if args.summarize:
        summarize(args.output_dir); return
    if not args.foundation_ckpt:
        args.foundation_ckpt = None

    # 普通 python 启动时保留单卡默认；torchrun 启动时不要覆盖 CUDA_VISIBLE_DEVICES。
    if 'WORLD_SIZE' not in os.environ and 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    try:
        train_foundation(args)
    except Exception:
        # 出错时确保 DDP 进程组被释放，避免僵尸进程。
        if hasattr(args, 'distributed') and args.distributed and dist.is_initialized():
            dist.destroy_process_group()
        raise


if __name__ == '__main__':
    main()
"""
cp /mnt/data/topcow_train_foundation_ddp.py /root/888-topcow-TGF/topcow_train_foundation.py

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 topcow_train_foundation.py \
  --exp_name sam_crossattn \
  --foundation_type sam3d_turbo \
  --fusion_type crossattn \
  --foundation_ckpt ./sam_med3d_turbo.pth \
  --epochs 200 \
  --batch_size 2 \
  --num_workers 2 \
  --amp > logs_foundation/sam_crossattn_ddp.log 2>&1
  
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 topcow_train_foundation.py \
  --exp_name sam_crossattn \
  --foundation_type sam3d_turbo \
  --fusion_type crossattn \
  --foundation_ckpt ./sam_med3d_turbo.pth \
  --epochs 200 \
  --batch_size 4 \
  --num_workers 2 \
  --amp > logs_foundation/sam_crossattn_ddp4.log 2>&1
  
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -m torch.distributed.run \
  --standalone \
  --nproc_per_node=6 \
  topcow_train_foundation_ddp_v3_tcff.py \
  --exp_name sam_tgf_v3_ddp \
  --foundation_type sam3d_turbo \
  --fusion_type tgf \
  --foundation_ckpt ./sam_med3d_turbo.pth \
  --epochs 400 \
  --batch_size 18 \
  --num_workers 2 \
  --seed 0
  --amp > logs_foundation/sam_tgf_v3_ddp6.log 2>&1

  
  CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc_per_node=2 \
  topcow_train_foundation_ddp_v3_tcff.py \
  --exp_name sam_tgf_v3_ddp \
  --foundation_type sam3d_turbo \
  --fusion_type tgf \
  --foundation_ckpt ./sam_med3d_turbo.pth \
  --epochs 200 \
  --batch_size 2 \
  --num_workers 2 \
  --amp
"""
