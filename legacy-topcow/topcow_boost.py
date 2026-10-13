# -*- coding: utf-8 -*-
# @Time    : 2026/4/24 9:42
# @Author  : JIA
# @FileName: topcow_boost.py
# @Software: PyCharm
# @Blog    ：
"""
topcow_boost.py
===============
性能提升模块 — 不修改融合框架，只增强训练和推理

包含:
1. clDice loss (血管连通性)
2. 主 decoder 深监督
3. TTA (测试时增强)
4. 增强修复 (X轴翻转时同步交换 laterality 标签)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import distance_transform_edt


# ============================================================
# 1. clDice Loss (简化版, GPU-friendly)
# ============================================================

def soft_skeletonize(x, iters=3):
    """可微骨架化: 迭代腐蚀后取差集的近似"""
    # 使用 max pool 做膨胀, -max_pool(-x) 做腐蚀
    skeleton = x.clone()
    for _ in range(iters):
        # 腐蚀: min pool = -max_pool(-x)
        eroded = -F.max_pool3d(-skeleton, kernel_size=3, stride=1, padding=1)
        # 骨架 = 原始 - 腐蚀后再膨胀
        dilated = F.max_pool3d(eroded, kernel_size=3, stride=1, padding=1)
        skeleton = skeleton - dilated
        skeleton = F.relu(skeleton)  # 保留正值
    return skeleton


def soft_cldice_loss(pred, target, iters=3, smooth=1.0):
    """clDice loss: 惩罚血管断裂

    pred: [B, 1, D, H, W] sigmoid 后的概率
    target: [B, 1, D, H, W] binary label
    """
    # 骨架化
    skel_pred = soft_skeletonize(pred, iters)
    skel_target = soft_skeletonize(target, iters)

    # Topology precision: 预测骨架覆盖了多少 GT
    tprec_num = (skel_pred * target).sum(dim=(2, 3, 4))
    tprec_den = skel_pred.sum(dim=(2, 3, 4)) + smooth

    # Topology sensitivity: GT 骨架被预测覆盖了多少
    tsens_num = (skel_target * pred).sum(dim=(2, 3, 4))
    tsens_den = skel_target.sum(dim=(2, 3, 4)) + smooth

    tprec = tprec_num / tprec_den
    tsens = tsens_num / tsens_den

    cl_dice = 2.0 * (tprec * tsens) / (tprec + tsens + smooth)
    return 1.0 - cl_dice.mean()


# ============================================================
# 2. 深监督头
# ============================================================

class DeepSupervisionHeads(nn.Module):
    """主 decoder 各 level 的深监督预测头

    在 main_decoder 的每个 level 加一个 1x1 conv → binary prediction
    """

    def __init__(self, decoder_channels=[128, 64, 32]):
        super().__init__()
        self.heads = nn.ModuleList([
            nn.Conv3d(ch, 1, 1) for ch in decoder_channels
        ])

    def forward(self, main_feats):
        """
        Args:
            main_feats: list of [B, C, D, H, W] from each decoder level
        Returns:
            list of [B, 1, D, H, W] sigmoid predictions
        """
        preds = []
        for feat, head in zip(main_feats, self.heads):
            preds.append(torch.sigmoid(head(feat)))
        return preds


def deep_supervision_loss(ds_preds, target, weights=[0.5, 0.3, 0.2]):
    """深监督 loss: 各 level 的 Dice+BCE, 越深权重越高

    ds_preds: list of [B, 1, D, H, W]
    target: [B, 1, D, H, W]
    """
    total = 0.0
    for pred, w in zip(ds_preds, weights):
        if pred.shape[2:] != target.shape[2:]:
            pred = F.interpolate(pred, size=target.shape[2:],
                                 mode='trilinear', align_corners=False)
        # Dice
        smooth = 1.0
        inter = (pred * target).sum()
        dice = (2 * inter + smooth) / (pred.sum() + target.sum() + smooth)
        l_dice = 1 - dice

        # BCE
        l_bce = F.binary_cross_entropy(pred.clamp(1e-7, 1 - 1e-7),
                                        target, reduction='mean')
        total += w * (l_dice + l_bce)
    return total


# ============================================================
# 3. TTA (测试时增强)
# ============================================================

@torch.no_grad()
def tta_inference(model, x, device):
    """3轴翻转 TTA (不翻转 X 轴, 避免 L/R 语义错乱)

    只用 Z 和 Y 轴翻转 → 4 种组合
    比 8 种组合安全, 且 CoW 左右对称不适合翻转

    x: [1, 1, D, H, W]
    returns: [1, 2, D, H, W] 平均后的 segmentation
    """
    model.eval()
    preds = []

    # 4 种组合: (flip_z, flip_y)
    for fz in [False, True]:
        for fy in [False, True]:
            x_aug = x.clone()
            if fz:
                x_aug = x_aug.flip(2)  # Z轴
            if fy:
                x_aug = x_aug.flip(3)  # Y轴

            seg, _, _ = model(x_aug)

            # 反翻转
            if fy:
                seg = seg.flip(3)
            if fz:
                seg = seg.flip(2)

            preds.append(seg)

    return torch.stack(preds).mean(dim=0)


# ============================================================
# 4. 增强修复: X 轴翻转时同步交换 laterality 标签
# ============================================================

def flip_laterality_gt(laterality_gt):
    """X 轴翻转后, left(0) ↔ right(1), midline(2) 和 bg(3) 不变"""
    flipped = laterality_gt.copy()
    left_mask = laterality_gt == 0
    right_mask = laterality_gt == 1
    flipped[left_mask] = 1
    flipped[right_mask] = 0
    return flipped


def flip_circuit_gt(circuit_gt):
    """X 轴翻转不影响 circuit (anterior/posterior/communicating 无左右)"""
    return circuit_gt.copy()


def flip_caliber_gt(caliber_gt):
    """X 轴翻转不影响 caliber (large/medium/small 无左右)"""
    return caliber_gt.copy()


def safe_augment(image, label, mc_mask, circuit_gt, caliber_gt, laterality_gt):
    """安全的数据增强: X 轴翻转时同步修正标签

    Args:
        image: [D, H, W]
        label: [D, H, W]
        mc_mask: [D, H, W] int64
        circuit_gt: [D, H, W] int64
        caliber_gt: [D, H, W] int64
        laterality_gt: [D, H, W] int64

    Returns:
        所有输入的增强版本
    """
    import random

    # Z 轴翻转 (无语义影响)
    if random.random() > 0.5:
        image = np.flip(image, axis=0).copy()
        label = np.flip(label, axis=0).copy()
        mc_mask = np.flip(mc_mask, axis=0).copy()
        circuit_gt = np.flip(circuit_gt, axis=0).copy()
        caliber_gt = np.flip(caliber_gt, axis=0).copy()
        laterality_gt = np.flip(laterality_gt, axis=0).copy()

    # Y 轴翻转 (无语义影响)
    if random.random() > 0.5:
        image = np.flip(image, axis=1).copy()
        label = np.flip(label, axis=1).copy()
        mc_mask = np.flip(mc_mask, axis=1).copy()
        circuit_gt = np.flip(circuit_gt, axis=1).copy()
        caliber_gt = np.flip(caliber_gt, axis=1).copy()
        laterality_gt = np.flip(laterality_gt, axis=1).copy()

    # X 轴翻转 (需要交换 laterality 的 left/right)
    if random.random() > 0.5:
        image = np.flip(image, axis=2).copy()
        label = np.flip(label, axis=2).copy()
        mc_mask = np.flip(mc_mask, axis=2).copy()
        circuit_gt = flip_circuit_gt(np.flip(circuit_gt, axis=2).copy())
        caliber_gt = flip_caliber_gt(np.flip(caliber_gt, axis=2).copy())
        laterality_gt = flip_laterality_gt(np.flip(laterality_gt, axis=2).copy())

    # 强度扰动
    if random.random() > 0.5:
        sigma = random.uniform(0.01, 0.05)
        image = image + np.random.normal(0, sigma, image.shape).astype(np.float32)
        image = np.clip(image, 0, 1)

    # Gamma 变换
    if random.random() > 0.3:
        gamma = random.uniform(0.8, 1.2)
        image = np.power(image.clip(1e-8, 1), gamma).astype(np.float32)

    return image, label, mc_mask, circuit_gt, caliber_gt, laterality_gt


# ============================================================
# 5. 组合 loss 函数 (替换 topcow_train.py 中的 compute_loss)
# ============================================================

def compute_loss_boosted(seg, aux_output, gau, batch, model_type,
                         ds_preds=None,
                         seg_w=1.0, aux_w=0.3, gau_w=0.3,
                         cldice_w=0.5, ds_w=0.3):
    """增强版 loss: seg + clDice + 3个辅助头 + 深监督 + GAU"""
    from func.loss_func import dice_loss

    labels = batch['label'].float().cuda()
    seg_pred = seg[:, 1:2]

    # 1. 基础分割 loss
    l_dice = dice_loss(seg_pred, labels)
    l_bce = F.binary_cross_entropy(seg_pred.clamp(1e-7, 1 - 1e-7),
                                    labels, reduction='mean')
    l_seg = l_dice + l_bce

    # 2. clDice loss
    l_cldice = soft_cldice_loss(seg_pred, labels)

    total = seg_w * l_seg + cldice_w * l_cldice

    # 3. 三个辅助头 loss
    l_aux = torch.tensor(0.0)
    if model_type in ['tgf', 'tgf_nogate']:
        ce = nn.CrossEntropyLoss(ignore_index=-1)
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

    # 4. 深监督 loss
    l_ds = torch.tensor(0.0)
    if ds_preds is not None:
        l_ds = deep_supervision_loss(ds_preds, labels)
        total = total + ds_w * l_ds

    # 5. GAU loss
    l_gau = torch.tensor(0.0)
    if gau is not None:
        l_gau = F.mse_loss(gau, labels)
        total = total + gau_w * l_gau

    return total, {
        'total': total.item(), 'seg': l_seg.item(),
        'dice': l_dice.item(), 'cldice': l_cldice.item(),
        'aux': l_aux.item(), 'ds': l_ds.item(), 'gau': l_gau.item(),
    }