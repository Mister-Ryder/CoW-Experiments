# -*- coding: utf-8 -*-
# @Time    : 2026/4/23 15:56
# @Author  : JIA
# @FileName: topcow_preprocess_v2.py
# @Software: PyCharm
# @Blog    ：
"""
topcow_preprocess.py (v2 - 修复版)
====================================
TopCoW MRA 数据预处理脚本

关键修复:
- 修复 ROI txt 解析器以适配实际文件格式
- 添加 spacing 信息记录
- 优化归一化策略

预处理流程 (参考获奖方案):
1. 读取 MRA NIfTI 图像 + multiclass mask + ROI 坐标
2. 裁切到 ROI（带 margin）— 所有 top teams 都用 ROI 裁切
3. Percentile-based 归一化 — 标准 MRA 预处理
4. 生成 binary label + 提取辅助分类标签
5. Pad 到统一尺寸 → 保存为 NPY

注: 不做 spacing resampling (保持原始各向异性)
    不做 skull stripping (官方数据已处理)
    不做 N4 bias correction (获奖方案均不做)

用法:
    python topcow_preprocess_v2.py --data_root ../TopCoW2024_Data_Release --output_dir ./TopCoW_processed
        # --modality mr
        # --margin 10 \
        # --target_size 64 160 160
"""

import os
import re
import glob
import argparse
import pickle
import numpy as np
import SimpleITK as sitk
from collections import OrderedDict

# ============================================================
# TopCoW 标签映射
# ============================================================

LABEL_REMAP = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
    8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 15: 13
}
NUM_VESSEL_CLASSES = 14

VESSEL_NAMES = {
    0: "Background", 1: "BA", 2: "R-PCA", 3: "L-PCA",
    4: "R-ICA", 5: "R-MCA", 6: "L-ICA", 7: "L-MCA",
    8: "R-Pcom", 9: "L-Pcom", 10: "Acom",
    11: "R-ACA", 12: "L-ACA", 13: "3rd-A2"
}

# 循环系统归属: 0=anterior, 1=posterior, 2=communicating, 3=background
CIRCULATION_MAP = {
    0: 3, 1: 1, 2: 1, 3: 1, 4: 0, 5: 0, 6: 0, 7: 0,
    8: 2, 9: 2, 10: 2, 11: 0, 12: 0, 13: 0,
}
NUM_CIRCULATION_CLASSES = 4

# 侧别: 0=left, 1=right, 2=midline, 3=background
LATERALITY_MAP = {
    0: 3, 1: 2, 2: 1, 3: 0, 4: 1, 5: 1, 6: 0, 7: 0,
    8: 1, 9: 0, 10: 2, 11: 1, 12: 0, 13: 2,
}
NUM_LATERALITY_CLASSES = 4


# ============================================================
# ROI 解析 (修复版 — 兼容多种格式)
# ============================================================

def read_roi_txt(roi_path):
    """读取 TopCoW 的 ROI bounding box txt 文件

    支持两种格式:
    格式 A (2024):
        --- ROI Meta Data ---
        Size (Voxels): 143 121 29
        Location (Voxels): 174 146 96

    格式 B (纯数字):
        143 121 29
        174 146 96
        或
        143 121 29 174 146 96
    """
    with open(roi_path, 'r') as f:
        content = f.read()

    # 尝试格式 A: 有 "Size" 和 "Location" 关键词
    size_match = re.search(r'Size\s*\(Voxels\)\s*:\s*(\d+)\s+(\d+)\s+(\d+)', content)
    loc_match = re.search(r'Location\s*\(Voxels\)\s*:\s*(\d+)\s+(\d+)\s+(\d+)', content)

    if size_match and loc_match:
        size = [int(size_match.group(i)) for i in range(1, 4)]
        origin = [int(loc_match.group(i)) for i in range(1, 4)]
    else:
        # 格式 B: 提取所有数字
        numbers = [int(x) for x in re.findall(r'\d+', content)]
        if len(numbers) == 6:
            size = numbers[:3]
            origin = numbers[3:]
        else:
            raise ValueError(f"Cannot parse ROI file {roi_path}. "
                             f"Found {len(numbers)} numbers: {numbers}")

    return {
        'size': size,  # [size_x, size_y, size_z]
        'origin': origin,  # [x_min, y_min, z_min]
        'x_min': origin[0], 'y_min': origin[1], 'z_min': origin[2],
        'x_max': origin[0] + size[0],
        'y_max': origin[1] + size[1],
        'z_max': origin[2] + size[2],
    }


# ============================================================
# 核心处理函数
# ============================================================

def remap_labels(mask_array):
    """将原始标签（含跳号 15）映射为连续编号 0-13"""
    remapped = np.zeros_like(mask_array, dtype=np.uint8)
    for orig, new in LABEL_REMAP.items():
        remapped[mask_array == orig] = new
    return remapped


def extract_classification_labels(multiclass_patch):
    """从 multiclass patch 提取三种分类标签"""
    foreground = multiclass_patch[multiclass_patch > 0]
    if len(foreground) == 0:
        return 0, 3, 3  # background
    unique, counts = np.unique(foreground, return_counts=True)
    dominant = int(unique[np.argmax(counts)])
    return dominant, CIRCULATION_MAP.get(dominant, 3), LATERALITY_MAP.get(dominant, 3)


def normalize_mra(image, p_low=0.5, p_high=99.5):
    """MRA 图像归一化 (percentile-based, 标准做法)"""
    image = image.astype(np.float32)
    lo = np.percentile(image, p_low)
    hi = np.percentile(image, p_high)
    if hi - lo < 1e-6:
        return np.zeros_like(image)
    return np.clip((image - lo) / (hi - lo), 0, 1)


def pad_to_target(array, target_size, pad_value=0):
    """将 3D array pad 到 target_size，居中对齐
    如果某维度超过 target，居中裁切
    """
    current = np.array(array.shape)
    target = np.array(target_size)

    # 超过 target 的维度先裁切
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


def process_single_case(case_id, data_root, modality, margin, target_size):
    """处理单个 case"""

    # 构建文件路径
    img_path = os.path.join(data_root, "imagesTr",
                            f"topcow_{modality}_{case_id}_0000.nii.gz")
    mask_path = os.path.join(data_root, "cow_seg_labelsTr",
                             f"topcow_{modality}_{case_id}.nii.gz")
    roi_path = os.path.join(data_root, "roi_loc_labelsTr",
                            f"topcow_{modality}_{case_id}.txt")

    # 检查文件存在
    for p, name in [(img_path, "image"), (mask_path, "mask"), (roi_path, "roi")]:
        if not os.path.exists(p):
            print(f"  [SKIP] {name} not found: {p}")
            return None

    # 读取图像
    img_sitk = sitk.ReadImage(img_path)
    img_array = sitk.GetArrayFromImage(img_sitk)  # [Z, Y, X]
    spacing = list(img_sitk.GetSpacing())  # (x_sp, y_sp, z_sp)
    origin = list(img_sitk.GetOrigin())
    direction = list(img_sitk.GetDirection())

    # 读取 mask
    mask_sitk = sitk.ReadImage(mask_path)
    mask_array = sitk.GetArrayFromImage(mask_sitk)

    # 读取 ROI
    roi = read_roi_txt(roi_path)

    # 扩大 ROI (加 margin)
    # 注意: ROI 坐标是 [X,Y,Z], numpy array 是 [Z,Y,X]
    img_shape = img_array.shape  # [Z, Y, X]
    x_min = max(0, roi['x_min'] - margin)
    y_min = max(0, roi['y_min'] - margin)
    z_min = max(0, roi['z_min'] - margin)
    x_max = min(img_shape[2], roi['x_max'] + margin)
    y_max = min(img_shape[1], roi['y_max'] + margin)
    z_max = min(img_shape[0], roi['z_max'] + margin)

    # 裁切 (numpy 索引 [Z,Y,X])
    img_crop = img_array[z_min:z_max, y_min:y_max, x_min:x_max]
    mask_crop = mask_array[z_min:z_max, y_min:y_max, x_min:x_max]

    crop_shape = img_crop.shape

    # 重映射标签
    mask_remapped = remap_labels(mask_crop)

    # 生成 binary label
    binary_label = (mask_remapped > 0).astype(np.uint8)

    # 提取分类标签
    vessel_type, circulation, laterality = extract_classification_labels(mask_remapped)

    # 统计血管类别
    unique_vessels = np.unique(mask_remapped[mask_remapped > 0])
    vessel_voxel_counts = {int(v): int(np.sum(mask_remapped == v)) for v in unique_vessels}

    # 归一化图像
    img_norm = normalize_mra(img_crop)

    # Pad 到目标尺寸
    img_padded, pad_before = pad_to_target(img_norm, target_size, pad_value=0)
    label_padded, _ = pad_to_target(binary_label, target_size, pad_value=0)
    mask_padded, _ = pad_to_target(mask_remapped, target_size, pad_value=0)

    metadata = {
        'case_id': case_id,
        'modality': modality,
        'original_img_shape': list(img_array.shape),
        'crop_shape': list(crop_shape),
        'padded_shape': list(img_padded.shape),
        'target_size': target_size,
        'pad_before': pad_before.tolist(),
        'spacing': spacing,  # [x_sp, y_sp, z_sp]
        'origin': origin,
        'roi': roi,
        'margin': margin,
        'crop_bounds': {
            'x_min': x_min, 'x_max': x_max,
            'y_min': y_min, 'y_max': y_max,
            'z_min': z_min, 'z_max': z_max,
        },
        'vessel_type': vessel_type,
        'circulation': circulation,
        'laterality': laterality,
        'vessel_voxel_counts': vessel_voxel_counts,
        'foreground_ratio': float(binary_label.sum()) / binary_label.size,
        'num_vessel_classes_present': len(unique_vessels),
    }

    return {
        'image': img_padded,
        'binary_label': label_padded,
        'multiclass_mask': mask_padded,
        'metadata': metadata,
    }


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='TopCoW MRA Preprocessing')
    parser.add_argument('--data_root', type=str,
                        default='/home/ubuntu/data/TopCoW2024_Data_Release')
    parser.add_argument('--output_dir', type=str,
                        default='/home/ubuntu/data/TopCoW_processed')
    parser.add_argument('--modality', type=str, default='mr',
                        choices=['mr', 'ct'])
    parser.add_argument('--margin', type=int, default=0,
                        help='ROI 外扩 voxel 数')
    parser.add_argument('--target_size', type=int, nargs=3,
                        default=[64, 160, 160],
                        help='目标尺寸 [Z, Y, X]')
    parser.add_argument('--train_ratio', type=float, default=1)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    # 验证 target_size 可被 8 整除 (3 次 pool)
    for i, s in enumerate(args.target_size):
        assert s % 8 == 0, f"target_size[{i}]={s} must be divisible by 8"

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    for subdir in ['image', 'binary_label', 'multiclass_mask']:
        os.makedirs(os.path.join(args.output_dir, subdir), exist_ok=True)

    # 找到所有 case
    img_pattern = os.path.join(args.data_root, "imagesTr",
                               f"topcow_{args.modality}_*_0000.nii.gz")
    img_files = sorted(glob.glob(img_pattern))

    case_ids = []
    for f in img_files:
        basename = os.path.basename(f)
        parts = basename.replace('_0000.nii.gz', '').split('_')
        case_ids.append(parts[2])

    print(f"Found {len(case_ids)} {args.modality.upper()} cases")
    print(f"Target size: {args.target_size}")
    print(f"ROI margin: {args.margin} voxels")

    # Train/Test 划分
    np.random.seed(args.seed)
    indices = np.random.permutation(len(case_ids))
    n_train = int(len(case_ids) * args.train_ratio)
    train_indices = sorted(indices[:n_train])
    test_indices = sorted(indices[n_train:])
    train_cases = [case_ids[i] for i in train_indices]
    test_cases = [case_ids[i] for i in test_indices]

    print(f"Train: {len(train_cases)} cases, Test: {len(test_cases)} cases")

    # 统计信息
    all_metadata = OrderedDict()
    crop_sizes = []
    spacings = []
    stats = {'total': 0, 'success': 0, 'failed': 0,
             'vessel_type_dist': {}, 'circulation_dist': {}, 'laterality_dist': {}}

    for case_id in case_ids:
        print(f"\n[{stats['total'] + 1}/{len(case_ids)}] Processing {case_id}...")
        stats['total'] += 1

        result = process_single_case(
            case_id, args.data_root, args.modality,
            args.margin, args.target_size
        )

        if result is None:
            stats['failed'] += 1
            continue

        stats['success'] += 1
        meta = result['metadata']
        crop_sizes.append(meta['crop_shape'])
        spacings.append(meta['spacing'])

        # 保存 NPY
        case_name = f"topcow_{args.modality}_{case_id}"
        np.save(os.path.join(args.output_dir, 'image', f'{case_name}.npy'),
                result['image'])
        np.save(os.path.join(args.output_dir, 'binary_label', f'{case_name}.npy'),
                result['binary_label'])
        np.save(os.path.join(args.output_dir, 'multiclass_mask', f'{case_name}.npy'),
                result['multiclass_mask'])

        split = 'train' if case_id in train_cases else 'test'
        meta['split'] = split

        all_metadata[case_name] = {
            'image': os.path.join(args.output_dir, 'image', f'{case_name}.npy'),
            'label': os.path.join(args.output_dir, 'binary_label', f'{case_name}.npy'),
            'multiclass_mask': os.path.join(args.output_dir, 'multiclass_mask', f'{case_name}.npy'),
            'split': split,
            'generation_type': meta['vessel_type'],
            'structure_type': meta['circulation'],
            'location_type': meta['laterality'],
            'case_id': case_id,
            'original_shape': meta['original_img_shape'],
            'crop_shape': meta['crop_shape'],
            'padded_shape': meta['padded_shape'],
            'spacing': meta['spacing'],
            'foreground_ratio': meta['foreground_ratio'],
            'vessel_voxel_counts': meta['vessel_voxel_counts'],
            'num_vessel_classes_present': meta['num_vessel_classes_present'],
        }

        # 分布统计
        for key, val, dist in [
            ('vessel_type', meta['vessel_type'], stats['vessel_type_dist']),
            ('circulation', meta['circulation'], stats['circulation_dist']),
            ('laterality', meta['laterality'], stats['laterality_dist'])
        ]:
            dist[val] = dist.get(val, 0) + 1

        print(f"  Crop: {meta['crop_shape']} → Pad: {meta['padded_shape']}")
        print(f"  Spacing: [{meta['spacing'][0]:.3f}, {meta['spacing'][1]:.3f}, {meta['spacing'][2]:.3f}] mm")
        print(f"  FG ratio: {meta['foreground_ratio']:.4f}, Vessels: {meta['num_vessel_classes_present']}")
        print(f"  Labels: vtype={meta['vessel_type']}({VESSEL_NAMES.get(meta['vessel_type'], '?')}), "
              f"circ={meta['circulation']}, lat={meta['laterality']}, split={split}")

    # 保存 metadata
    pkl_path = os.path.join(args.output_dir, f'topcow_{args.modality}_metadata.pkl')
    with open(pkl_path, 'wb') as f:
        pickle.dump(all_metadata, f)

    split_info = {
        'train_cases': train_cases,
        'test_cases': test_cases,
        'seed': args.seed,
        'train_ratio': args.train_ratio,
    }
    split_path = os.path.join(args.output_dir, f'topcow_{args.modality}_split.pkl')
    with open(split_path, 'wb') as f:
        pickle.dump(split_info, f)

    # ========================================
    # 汇总统计
    # ========================================
    print("\n" + "=" * 70)
    print("预处理完成")
    print("=" * 70)
    print(f"成功: {stats['success']}/{stats['total']} cases")
    print(f"训练: {len(train_cases)}, 测试: {len(test_cases)}")

    if crop_sizes:
        crop_arr = np.array(crop_sizes)
        print(f"\nROI 裁切尺寸统计 [Z, Y, X]:")
        print(f"  Min:  {crop_arr.min(axis=0)}")
        print(f"  Max:  {crop_arr.max(axis=0)}")
        print(f"  Mean: {crop_arr.mean(axis=0).astype(int)}")

        # 检查是否有尺寸超出 target
        target = np.array(args.target_size)
        oversize = (crop_arr > target).any(axis=1)
        if oversize.sum() > 0:
            print(f"  ⚠️  {oversize.sum()} cases 的 crop 超出 target_size，已被居中裁切!")
        else:
            print(f"  ✓ 所有 case 的 crop 均在 target_size 范围内")

    if spacings:
        sp_arr = np.array(spacings)
        print(f"\nSpacing 统计 [X, Y, Z] (mm):")
        print(f"  Min:  [{sp_arr[:, 0].min():.3f}, {sp_arr[:, 1].min():.3f}, {sp_arr[:, 2].min():.3f}]")
        print(f"  Max:  [{sp_arr[:, 0].max():.3f}, {sp_arr[:, 1].max():.3f}, {sp_arr[:, 2].max():.3f}]")
        print(f"  Mean: [{sp_arr[:, 0].mean():.3f}, {sp_arr[:, 1].mean():.3f}, {sp_arr[:, 2].mean():.3f}]")

    print(f"\nVessel type 分布:")
    for vt in sorted(stats['vessel_type_dist'].keys()):
        print(f"  {VESSEL_NAMES.get(vt, f'?({vt})')}: {stats['vessel_type_dist'][vt]}")

    circ_names = {0: 'anterior', 1: 'posterior', 2: 'communicating', 3: 'background'}
    print(f"\nCirculation 分布:")
    for ci in sorted(stats['circulation_dist'].keys()):
        print(f"  {circ_names.get(ci, '?')}: {stats['circulation_dist'][ci]}")

    lat_names = {0: 'left', 1: 'right', 2: 'midline', 3: 'background'}
    print(f"\nLaterality 分布:")
    for la in sorted(stats['laterality_dist'].keys()):
        print(f"  {lat_names.get(la, '?')}: {stats['laterality_dist'][la]}")

    # 保存配置
    config = {
        'data_root': args.data_root,
        'output_dir': args.output_dir,
        'modality': args.modality,
        'margin': args.margin,
        'target_size': args.target_size,
        'train_ratio': args.train_ratio,
        'seed': args.seed,
        'num_vessel_classes': NUM_VESSEL_CLASSES,
        'num_circulation_classes': NUM_CIRCULATION_CLASSES,
        'num_laterality_classes': NUM_LATERALITY_CLASSES,
    }
    config_path = os.path.join(args.output_dir, 'preprocess_config.pkl')
    with open(config_path, 'wb') as f:
        pickle.dump(config, f)

    print(f"\n输出文件:")
    print(f"  Metadata: {pkl_path}")
    print(f"  Split: {split_path}")
    print(f"  Config: {config_path}")
    print(f"  Data: {args.output_dir}/{{image,binary_label,multiclass_mask}}/")


if __name__ == '__main__':
    main()