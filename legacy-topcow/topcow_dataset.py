# -*- coding: utf-8 -*-
# @Time    : 2026/4/24 9:19
# @Author  : JIA
# @FileName: topcow_dataset.py.py
# @Software: PyCharm
# @Blog    ：
"""
topcow_dataset.py (v4)
======================
TopCoW Dataset — 多分类版本

返回:
- image: [1,D,H,W] 归一化 MRA
- multiclass: [D,H,W] 14类GT (主decoder监督)
- circuit_gt: [D,H,W] 4类 (辅助头监督)
- caliber_gt: [D,H,W] 4类
- laterality_gt: [D,H,W] 4类
"""

import os
import pickle
import random
import numpy as np
import torch
from torch.utils.data import Dataset

CIRCUIT_LUT = np.array([3, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0], dtype=np.int64)
CALIBER_LUT = np.array([3, 0, 1, 1, 0, 1, 0, 1, 2, 2, 2, 1, 1, 1], dtype=np.int64)
LATERALITY_LUT = np.array([3, 2, 1, 0, 1, 1, 0, 0, 1, 0, 2, 1, 0, 2], dtype=np.int64)

# X轴翻转时 multiclass 标签交换对
LR_SWAP_PAIRS = [(2, 3), (4, 6), (5, 7), (8, 9), (11, 12)]


class TopCoWDataset(Dataset):
    def __init__(self, data_dir, split='train', noise_rate=0.0,
                 augment=True, num_classes=14):
        self.noise_rate = noise_rate
        self.augment = augment and (split == 'train')
        self.num_classes = num_classes

        meta_path = self._find_pkl(data_dir, '_metadata.pkl')
        with open(meta_path, 'rb') as f:
            all_meta = pickle.load(f)

        split_path = self._find_pkl(data_dir, '_split.pkl')
        target = None
        if split_path:
            with open(split_path, 'rb') as f:
                si = pickle.load(f)
            key = f'{split}_cases'
            if key in si:
                target = set(si[key])

        self.samples = []
        for name, m in all_meta.items():
            if target and m['case_id'] not in target:
                continue
            mc = m.get('multiclass_mask')
            if mc and os.path.exists(mc):
                self.samples.append({
                    'case_name': name,
                    'image_path': m['image'],
                    'label_path': m['label'],
                    'multiclass_path': mc,
                })

        assert self.samples, f"No samples for split={split}"
        print(f"[TopCoWDataset] {split}: {len(self.samples)}, "
              f"noise={noise_rate}, augment={self.augment}")

    @staticmethod
    def _find_pkl(d, suffix):
        cs = [f for f in os.listdir(d) if f.endswith(suffix)]
        return os.path.join(d, sorted(cs)[0]) if cs else None

    def __len__(self):
        return len(self.samples)

    def _inject_noise(self, mc):
        """逐体素噪声: 随机翻转前景体素的 multiclass 类别"""
        noisy = mc.copy()
        fg = noisy > 0
        if fg.sum() == 0 or self.noise_rate <= 0:
            return noisy
        flip = (np.random.random(mc.shape) < self.noise_rate) & fg
        rand = np.random.randint(1, self.num_classes, size=mc.shape).astype(mc.dtype)
        same = rand == noisy
        rand[same] = ((rand[same].astype(int) % (self.num_classes - 1)) + 1).astype(mc.dtype)
        noisy[flip] = rand[flip]
        return noisy

    def _augment(self, image, mc):
        # Z轴翻转
        if random.random() > 0.5:
            image = np.flip(image, axis=0).copy()
            mc = np.flip(mc, axis=0).copy()

        # Y轴翻转
        if random.random() > 0.5:
            image = np.flip(image, axis=1).copy()
            mc = np.flip(mc, axis=1).copy()

        # X轴翻转 + 交换左右类别
        if random.random() > 0.5:
            image = np.flip(image, axis=2).copy()
            mc = np.flip(mc, axis=2).copy()
            mc_swap = mc.copy()
            for l, r in LR_SWAP_PAIRS:
                mc_swap[mc == l] = r
                mc_swap[mc == r] = l
            mc = mc_swap

        # 强度扰动
        if random.random() > 0.5:
            sigma = random.uniform(0.01, 0.05)
            image = image + np.random.normal(0, sigma, image.shape).astype(np.float32)
            image = np.clip(image, 0, 1)

        # Gamma变换
        if random.random() > 0.3:
            gamma = random.uniform(0.8, 1.2)
            image = np.power(image.clip(1e-8, 1), gamma).astype(np.float32)

        return image, mc

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = np.load(s['image_path']).astype(np.float32)
        mc = np.load(s['multiclass_path']).astype(np.int64)

        if self.augment:
            image, mc = self._augment(image, mc)
        if self.noise_rate > 0:
            mc = self._inject_noise(mc)

        # 派生标签 (从可能带噪声的 mc 派生 → 噪声自然传播到辅助头)
        mc_clipped = np.clip(mc, 0, len(CIRCUIT_LUT) - 1)

        return {
            'image': torch.from_numpy(image).unsqueeze(0).float(),
            'multiclass': torch.from_numpy(mc_clipped.copy()).long(),
            'circuit_gt': torch.from_numpy(CIRCUIT_LUT[mc_clipped].copy()).long(),
            'caliber_gt': torch.from_numpy(CALIBER_LUT[mc_clipped].copy()).long(),
            'laterality_gt': torch.from_numpy(LATERALITY_LUT[mc_clipped].copy()).long(),
            'case_name': s['case_name'],
        }


class TopCoWTestDataset(Dataset):
    def __init__(self, data_dir, split='test'):
        meta_path = TopCoWDataset._find_pkl(data_dir, '_metadata.pkl')
        if meta_path:
            with open(meta_path, 'rb') as f:
                all_meta = pickle.load(f)
            split_path = TopCoWDataset._find_pkl(data_dir, '_split.pkl')
            target = None
            if split_path:
                with open(split_path, 'rb') as f:
                    si = pickle.load(f)
                target = set(si.get(f'{split}_cases', []))
            self.samples = []
            for name, m in all_meta.items():
                if target and m['case_id'] not in target:
                    continue
                self.samples.append({
                    'case_name': name,
                    'image_path': m['image'],
                    'label_path': m['label'],
                    'multiclass_path': m.get('multiclass_mask'),
                })
        else:
            self.samples = self._scan(data_dir)
        print(f"[TopCoWTestDataset] {split}: {len(self.samples)}")

    @staticmethod
    def _scan(d):
        samples = []
        img_dir = os.path.join(d, 'image')
        if not os.path.isdir(img_dir):
            return samples
        for f in sorted(os.listdir(img_dir)):
            if f.endswith('.npy'):
                n = f.replace('.npy', '')
                samples.append({
                    'case_name': n,
                    'image_path': os.path.join(img_dir, f),
                    'label_path': os.path.join(d, 'binary_label', f),
                    'multiclass_path': os.path.join(d, 'multiclass_mask', f),
                })
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        image = np.load(s['image_path']).astype(np.float32)
        result = {
            'image': torch.from_numpy(image).unsqueeze(0).float(),
            'case_name': s['case_name'],
        }

        mc_path = s.get('multiclass_path')
        if mc_path and os.path.exists(mc_path):
            mc = np.load(mc_path).astype(np.int64)
            mc_clipped = np.clip(mc, 0, len(CIRCUIT_LUT) - 1)
            result['multiclass'] = torch.from_numpy(mc_clipped).long()

        # 同时提供 binary label (用于 binary Dice / clDice 计算)
        label_path = s.get('label_path')
        if label_path and os.path.exists(label_path):
            label = np.load(label_path).astype(np.float32)
            result['label'] = torch.from_numpy(label).unsqueeze(0).float()

        return result