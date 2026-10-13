# -*- coding: utf-8 -*-
# @Time    : 2026/4/24 10:29
# @Author  : JIA
# @FileName: topcow_preprocess_crown.py
# @Software: PyCharm
# @Blog    ：
"""
topcow_preprocess_crown.py
==========================
Crown 20 MRA 测试集预处理

Crown 的特殊性:
- 已提供 ROI 裁切版 (crown_mr_roi_*) 和全图版 (crown_mr_whole_*)
- 目录名不同: mul_labelsTr (非 cow_seg_labelsTr), roi_size_loc (非 roi_loc_labelsTr)
- 文件名格式: crown_mr_roi_001_0000.nii.gz
- 标签值 0-12 (无 label 15/3rd-A2)

策略:
- 直接使用 ROI 版本 (已裁切), 跳过 ROI 裁切步骤
- 只做: 归一化 + pad 到 target_size + 生成分类标签
- 和训练集保持一致的 target_size 和归一化参数

用法:
    python topcow_preprocess_crown.py --data_root ../crown_20_mr_01062023 --output_dir ./crown_processed --target_size 64 160 160
"""

import os
import glob
import argparse
import pickle
import numpy as np
import SimpleITK as sitk
from collections import OrderedDict


# 和训练集完全一致的标签映射
LABEL_REMAP = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
    8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 15: 13
}


def remap_labels(mask):
    remapped = np.zeros_like(mask, dtype=np.uint8)
    for orig, new in LABEL_REMAP.items():
        remapped[mask == orig] = new
    return remapped


def normalize_mra(image, p_low=0.5, p_high=99.5):
    image = image.astype(np.float32)
    lo = np.percentile(image, p_low)
    hi = np.percentile(image, p_high)
    if hi - lo < 1e-6:
        return np.zeros_like(image)
    return np.clip((image - lo) / (hi - lo), 0, 1)


def pad_to_target(array, target_size, pad_value=0):
    current = np.array(array.shape)
    target = np.array(target_size)

    # 超出 target 的维度居中裁切
    for i in range(3):
        if current[i] > target[i]:
            start = (current[i] - target[i]) // 2
            slices = [slice(None)] * 3
            slices[i] = slice(start, start + target[i])
            array = array[tuple(slices)]

    current = np.array(array.shape)
    pad_total = target - current
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    padding = [(pad_before[i], pad_after[i]) for i in range(3)]
    padded = np.pad(array, padding, mode='constant', constant_values=pad_value)
    return padded, pad_before


def main():
    parser = argparse.ArgumentParser(description='Crown MRA Test Set Preprocessing')
    parser.add_argument('--data_root', type=str, required=True,
                        help='crown_20_mr_01062023 目录路径')
    parser.add_argument('--output_dir', type=str, default='./crown_processed')
    parser.add_argument('--target_size', type=int, nargs=3,
                        default=[64, 160, 160])
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    for subdir in ['image', 'binary_label', 'multiclass_mask']:
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)

    # 查找 ROI 版本的图像文件
    img_dir = os.path.join(args.data_root, 'imagesTr')
    roi_files = sorted(glob.glob(os.path.join(img_dir, 'crown_mr_roi_*_0000.nii.gz')))

    if not roi_files:
        print(f"未找到 crown_mr_roi_* 文件, 尝试查找所有文件...")
        roi_files = sorted(glob.glob(os.path.join(img_dir, '*_0000.nii.gz')))

    # 只用 ROI 版本, 过滤掉 whole 版本
    roi_files = [f for f in roi_files if 'whole' not in os.path.basename(f)]

    print(f"Found {len(roi_files)} ROI cases")
    print(f"Target size: {args.target_size}")

    # 确定标签目录
    mul_dir = os.path.join(args.data_root, 'mul_labelsTr')
    if not os.path.isdir(mul_dir):
        mul_dir = os.path.join(args.data_root, 'cow_seg_labelsTr')
    print(f"Multiclass label dir: {mul_dir}")

    all_metadata = OrderedDict()
    crop_sizes = []

    for img_path in roi_files:
        basename = os.path.basename(img_path)
        # 提取 case_id: crown_mr_roi_001_0000.nii.gz → 001
        parts = basename.replace('_0000.nii.gz', '').split('_')
        case_id = parts[-1]  # 最后一个部分是数字
        case_prefix = '_'.join(parts)  # crown_mr_roi_001

        print(f"\n[{case_id}] Processing {basename}...")

        # 读取图像
        img_sitk = sitk.ReadImage(img_path)
        img_array = sitk.GetArrayFromImage(img_sitk)  # [Z, Y, X]
        spacing = list(img_sitk.GetSpacing())

        # 查找对应的多类标签
        mask_path = os.path.join(mul_dir, f'{case_prefix}.nii.gz')
        if not os.path.exists(mask_path):
            # 尝试其他命名格式
            candidates = glob.glob(os.path.join(mul_dir, f'*roi*{case_id}*'))
            if candidates:
                mask_path = candidates[0]
            else:
                print(f"  [SKIP] Mask not found for {case_prefix}")
                continue

        mask_sitk = sitk.ReadImage(mask_path)
        mask_array = sitk.GetArrayFromImage(mask_sitk)

        crop_shape = img_array.shape
        crop_sizes.append(crop_shape)

        # 重映射标签
        mask_remapped = remap_labels(mask_array)

        # 二值标签
        binary_label = (mask_remapped > 0).astype(np.uint8)

        # 归一化
        img_norm = normalize_mra(img_array)

        # Pad 到目标尺寸
        img_padded, pad_before = pad_to_target(img_norm, args.target_size)
        label_padded, _ = pad_to_target(binary_label, args.target_size)
        mask_padded, _ = pad_to_target(mask_remapped, args.target_size)

        # 保存
        case_name = f'crown_mr_{case_id}'
        np.save(os.path.join(args.output_dir, 'image', f'{case_name}.npy'), img_padded)
        np.save(os.path.join(args.output_dir, 'binary_label', f'{case_name}.npy'), label_padded)
        np.save(os.path.join(args.output_dir, 'multiclass_mask', f'{case_name}.npy'), mask_padded)

        fg_ratio = float(binary_label.sum()) / binary_label.size
        unique_vessels = np.unique(mask_remapped[mask_remapped > 0])

        all_metadata[case_name] = {
            'image': os.path.join(args.output_dir, 'image', f'{case_name}.npy'),
            'label': os.path.join(args.output_dir, 'binary_label', f'{case_name}.npy'),
            'multiclass_mask': os.path.join(args.output_dir, 'multiclass_mask', f'{case_name}.npy'),
            'case_id': case_id,
            'split': 'test',
            'original_shape': list(crop_shape),
            'padded_shape': list(img_padded.shape),
            'spacing': spacing,
            'foreground_ratio': fg_ratio,
            'num_vessel_classes_present': len(unique_vessels),
        }

        print(f"  Shape: {crop_shape} → Pad: {list(img_padded.shape)}")
        print(f"  Spacing: [{spacing[0]:.3f}, {spacing[1]:.3f}, {spacing[2]:.3f}] mm")
        print(f"  FG ratio: {fg_ratio:.4f}, Vessels: {len(unique_vessels)}")

    # 保存 metadata
    pkl_path = os.path.join(args.output_dir, 'crown_mr_metadata.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(all_metadata, f)

    # 保存 split (全部作为 test)
    split_info = {
        'test_cases': [m['case_id'] for m in all_metadata.values()],
        'train_cases': [],
    }
    split_path = os.path.join(args.output_dir, 'crown_mr_split.pkl')
    with open(split_path, 'wb') as f:
        pickle.dump(split_info, f)

    # 统计
    print(f"\n{'='*60}")
    print(f"Crown 预处理完成: {len(all_metadata)} cases")
    if crop_sizes:
        arr = np.array(crop_sizes)
        print(f"ROI 尺寸 [Z,Y,X]: Min={arr.min(axis=0)}, Max={arr.max(axis=0)}, Mean={arr.mean(axis=0).astype(int)}")
        oversize = (arr > np.array(args.target_size)).any(axis=1)
        if oversize.sum() > 0:
            print(f"  ⚠️  {oversize.sum()} cases 超出 target_size, 已居中裁切")
        else:
            print(f"  ✓ 所有 case 均在 target_size 内")
    print(f"输出: {args.output_dir}")
    print(f"  Metadata: {pkl_path}")
    print(f"  Split: {split_path}")


if __name__ == '__main__':
    main()