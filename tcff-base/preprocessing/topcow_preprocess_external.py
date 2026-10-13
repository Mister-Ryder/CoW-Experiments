# -*- coding: utf-8 -*-
"""
topcow_preprocess_external.py
==============================
MRA_IXI_HH 和 MRA_Lausanne 外部测试集预处理

目录结构 (两个数据集一致):
    {data_root}/
      imagesTr/          图像 NIfTI
      cow_seg_labelsTr/  多类别分割标注
      roi_loc_labelsTr/  CoW ROI 坐标 txt

文件命名规律:
  MRA_IXI_HH:
    image : IXI{id}-MRA_LPS_DFmasked_ROI-braincase.nii.gz   (脑区已预裁切)
    label : IXI{id}-MRA_LPS_DFmasked_ROI-braincase_full_LPS_Mask.nii.gz
    roi   : IXI{id}-MRA_LPS_DFmasked_ROI-cow.txt            (CoW 在脑区裁切图中的坐标)

  MRA_Lausanne:
    image : sub-{id}_angio_LPS.nii.gz
    label : sub-{id}_angio_LPS_full_LPS_Mask.nii.gz
    roi   : sub-{id}_angio_LPS.txt

预处理流程 (与 topcow_preprocess_v2.py 保持一致):
  1. 读取 ROI txt → 带 margin 裁切到 CoW 区域
  2. 百分位归一化
  3. Pad 到 target_size [Z, Y, X]
  4. 生成 binary label + multiclass mask (标签重映射 15→13)
  5. 保存为 NPY + metadata pkl (兼容 TopCoWTestDataset)

用法:
  # IXI_HH
  python topcow_preprocess_external.py \\
      --data_root ./MRA_IXI_HH \\
      --dataset ixi \\
      --output_dir ./ixi_processed

  # Lausanne
  python topcow_preprocess_external.py \\
      --data_root ./MRA_Lausanne \\
      --dataset lausanne \\
      --output_dir ./lausanne_processed

  # 同时处理两个 (分别运行两次, 或用 --output_dir 合并)
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
# 标签映射 (与训练集完全一致)
# ============================================================

LABEL_REMAP = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
    8: 8, 9: 9, 10: 10, 11: 11, 12: 12, 15: 13
}

VESSEL_NAMES = {
    0: "Background", 1: "BA", 2: "R-PCA", 3: "L-PCA",
    4: "R-ICA", 5: "R-MCA", 6: "L-ICA", 7: "L-MCA",
    8: "R-Pcom", 9: "L-Pcom", 10: "Acom",
    11: "R-ACA", 12: "L-ACA", 13: "3rd-A2"
}


# ============================================================
# 数据集配置: 文件名解析规则
# ============================================================

def discover_cases_ixi(data_root):
    """
    MRA_IXI_HH: 图像 stem = IXI{xxx}-MRA_LPS_DFmasked_ROI-braincase
    case_id    = IXI{xxx}  (e.g. IXI012-HH-1211)
    """
    img_dir = os.path.join(data_root, 'imagesTr')
    lbl_dir = os.path.join(data_root, 'cow_seg_labelsTr')
    roi_dir = os.path.join(data_root, 'roi_loc_labelsTr')

    cases = []
    for img_path in sorted(glob.glob(os.path.join(img_dir, 'IXI*braincase.nii.gz'))):
        stem = os.path.basename(img_path).replace('.nii.gz', '')
        # stem = IXI012-HH-1211-MRA_LPS_DFmasked_ROI-braincase
        # case_id = IXI012-HH-1211
        case_id = stem.split('-MRA_')[0]

        lbl_path = os.path.join(lbl_dir, stem + '_full_LPS_Mask.nii.gz')
        # ROI txt: IXI012-HH-1211-MRA_LPS_DFmasked_ROI-cow.txt
        roi_stem = case_id + '-MRA_LPS_DFmasked_ROI-cow'
        roi_path = os.path.join(roi_dir, roi_stem + '.txt')

        cases.append({
            'case_id': case_id,
            'case_name': 'ixi_' + case_id.replace('-', '_'),
            'img_path': img_path,
            'lbl_path': lbl_path,
            'roi_path': roi_path,
        })
    return cases


def discover_cases_lausanne(data_root):
    """
    MRA_Lausanne: 图像 stem = sub-{id}_ses-{date}_angio_LPS
    case_id = sub-{id}_ses-{date}
    """
    img_dir = os.path.join(data_root, 'imagesTr')
    lbl_dir = os.path.join(data_root, 'cow_seg_labelsTr')
    roi_dir = os.path.join(data_root, 'roi_loc_labelsTr')

    cases = []
    for img_path in sorted(glob.glob(os.path.join(img_dir, 'sub-*_angio_LPS.nii.gz'))):
        stem = os.path.basename(img_path).replace('.nii.gz', '')
        # stem = sub-000_ses-20110101_angio_LPS
        # case_id = sub-000_ses-20110101
        case_id = '_'.join(stem.split('_')[:2])   # sub-000_ses-20110101

        lbl_path = os.path.join(lbl_dir, stem + '_full_LPS_Mask.nii.gz')
        roi_path = os.path.join(roi_dir, stem + '.txt')

        cases.append({
            'case_id': case_id,
            'case_name': 'lausanne_' + case_id.replace('-', '').replace('_', '_'),
            'img_path': img_path,
            'lbl_path': lbl_path,
            'roi_path': roi_path,
        })
    return cases


# ============================================================
# ROI txt 解析 (复用 topcow_preprocess_v2.py 的解析器)
# ============================================================

def read_roi_txt(roi_path):
    """
    支持两种格式:
    格式 A:
        Size (Voxels): 143 121 29
        Location (Voxels): 174 146 96
    格式 B (纯数字, 6个):
        143 121 29 174 146 96
    """
    with open(roi_path, 'r') as f:
        content = f.read()

    size_match = re.search(r'Size\s*\(Voxels\)\s*:\s*(\d+)\s+(\d+)\s+(\d+)', content)
    loc_match  = re.search(r'Location\s*\(Voxels\)\s*:\s*(\d+)\s+(\d+)\s+(\d+)', content)

    if size_match and loc_match:
        size   = [int(size_match.group(i)) for i in range(1, 4)]
        origin = [int(loc_match.group(i))  for i in range(1, 4)]
    else:
        numbers = [int(x) for x in re.findall(r'\d+', content)]
        if len(numbers) == 6:
            size, origin = numbers[:3], numbers[3:]
        else:
            raise ValueError(
                f"Cannot parse ROI file {roi_path}. "
                f"Found {len(numbers)} numbers: {numbers}"
            )

    return {
        'size':  size,    # [size_x, size_y, size_z]
        'origin': origin, # [x_min, y_min, z_min]
        'x_min': origin[0], 'y_min': origin[1], 'z_min': origin[2],
        'x_max': origin[0] + size[0],
        'y_max': origin[1] + size[1],
        'z_max': origin[2] + size[2],
    }


# ============================================================
# 核心处理函数
# ============================================================

def remap_labels(mask_array):
    remapped = np.zeros_like(mask_array, dtype=np.uint8)
    for orig, new in LABEL_REMAP.items():
        remapped[mask_array == orig] = new
    return remapped


def normalize_mra(image, p_low=0.5, p_high=99.5):
    image = image.astype(np.float32)
    lo = np.percentile(image, p_low)
    hi = np.percentile(image, p_high)
    if hi - lo < 1e-6:
        return np.zeros_like(image)
    return np.clip((image - lo) / (hi - lo), 0, 1)


def pad_to_target(array, target_size, pad_value=0):
    """居中 pad / 居中 crop → target_size"""
    current = np.array(array.shape)
    target  = np.array(target_size)

    # 超出维度先居中裁切
    for i in range(3):
        if current[i] > target[i]:
            start = (current[i] - target[i]) // 2
            slices = [slice(None)] * 3
            slices[i] = slice(start, start + target[i])
            array = array[tuple(slices)]

    current   = np.array(array.shape)
    pad_total = target - current
    pad_before = pad_total // 2
    pad_after  = pad_total - pad_before
    padding = [(pad_before[i], pad_after[i]) for i in range(3)]
    padded = np.pad(array, padding, mode='constant', constant_values=pad_value)
    return padded, pad_before


def process_case(case_info, margin, target_size):
    """处理单个 case, 返回 dict 或 None"""
    img_path = case_info['img_path']
    lbl_path = case_info['lbl_path']
    roi_path = case_info['roi_path']
    case_id  = case_info['case_id']

    # ---- 检查文件 ----
    missing = []
    for p, name in [(img_path, 'image'), (lbl_path, 'label'), (roi_path, 'roi')]:
        if not os.path.exists(p):
            missing.append(f'{name}: {p}')
    if missing:
        print(f"  [SKIP] Missing files:")
        for m in missing: print(f"    {m}")
        return None

    # ---- 读取图像 ----
    img_sitk = sitk.ReadImage(img_path)
    img_arr  = sitk.GetArrayFromImage(img_sitk)   # [Z, Y, X]
    spacing  = list(img_sitk.GetSpacing())

    lbl_sitk = sitk.ReadImage(lbl_path)
    lbl_arr  = sitk.GetArrayFromImage(lbl_sitk)

    img_shape = img_arr.shape  # [Z, Y, X]

    # ---- ROI 裁切 ----
    roi = read_roi_txt(roi_path)

    # ROI 坐标是 [X, Y, Z], numpy 索引是 [Z, Y, X]
    x_min = max(0, roi['x_min'] - margin)
    y_min = max(0, roi['y_min'] - margin)
    z_min = max(0, roi['z_min'] - margin)
    x_max = min(img_shape[2], roi['x_max'] + margin)
    y_max = min(img_shape[1], roi['y_max'] + margin)
    z_max = min(img_shape[0], roi['z_max'] + margin)

    img_crop = img_arr[z_min:z_max, y_min:y_max, x_min:x_max]
    lbl_crop = lbl_arr[z_min:z_max, y_min:y_max, x_min:x_max]
    crop_shape = img_crop.shape

    # ---- 标签重映射 ----
    mask_remapped = remap_labels(lbl_crop)
    binary_label  = (mask_remapped > 0).astype(np.uint8)

    unique_vessels = np.unique(mask_remapped[mask_remapped > 0])
    fg_ratio = float(binary_label.sum()) / max(binary_label.size, 1)

    # ---- 归一化 ----
    img_norm = normalize_mra(img_crop)

    # ---- Pad 到目标尺寸 ----
    img_padded,  pad_before = pad_to_target(img_norm,      target_size)
    lbl_padded,  _          = pad_to_target(binary_label,  target_size)
    mask_padded, _          = pad_to_target(mask_remapped, target_size)

    meta = {
        'case_id':         case_id,
        'original_shape':  list(img_shape),
        'crop_shape':      list(crop_shape),
        'padded_shape':    list(img_padded.shape),
        'spacing':         spacing,
        'roi':             roi,
        'margin':          margin,
        'crop_bounds':     {'x_min': x_min, 'x_max': x_max,
                            'y_min': y_min, 'y_max': y_max,
                            'z_min': z_min, 'z_max': z_max},
        'foreground_ratio': fg_ratio,
        'num_vessel_classes_present': len(unique_vessels),
    }

    return {
        'image':          img_padded,
        'binary_label':   lbl_padded,
        'multiclass_mask': mask_padded,
        'metadata':       meta,
    }


# ============================================================
# 主函数
# ============================================================

def main():
    p = argparse.ArgumentParser(description='Preprocess MRA_IXI_HH / MRA_Lausanne for testing')
    p.add_argument('--data_root',   type=str, required=True,
                   help='数据集根目录 (含 imagesTr/ cow_seg_labelsTr/ roi_loc_labelsTr/)')
    p.add_argument('--dataset',     type=str, required=True, choices=['ixi', 'lausanne'],
                   help='数据集类型: ixi 或 lausanne')
    p.add_argument('--output_dir',  type=str, default=None,
                   help='输出目录 (默认: ./{dataset}_processed)')
    p.add_argument('--margin',      type=int, default=5,
                   help='ROI 外扩 voxel 数 (默认 5)')
    p.add_argument('--target_size', type=int, nargs=3, default=[64, 160, 160],
                   help='目标体积尺寸 [Z Y X] (默认 64 160 160)')
    args = p.parse_args()

    # 校验 target_size 可被 8 整除
    for i, s in enumerate(args.target_size):
        assert s % 8 == 0, f"target_size[{i}]={s} 必须能被 8 整除"

    if args.output_dir is None:
        args.output_dir = f'./{args.dataset}_processed'

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    for sub in ['image', 'binary_label', 'multiclass_mask']:
        os.makedirs(os.path.join(args.output_dir, sub), exist_ok=True)

    # 发现 case 列表
    if args.dataset == 'ixi':
        cases = discover_cases_ixi(args.data_root)
    else:
        cases = discover_cases_lausanne(args.data_root)

    if not cases:
        print(f"[ERROR] No cases found in {args.data_root}. Check --data_root and --dataset.")
        return

    print(f"\n{'='*60}")
    print(f"Dataset : {args.dataset.upper()}")
    print(f"Root    : {args.data_root}")
    print(f"Cases   : {len(cases)}")
    print(f"Margin  : {args.margin} voxels")
    print(f"Target  : {args.target_size} [Z, Y, X]")
    print(f"Output  : {args.output_dir}")
    print(f"{'='*60}")

    all_metadata = OrderedDict()
    crop_sizes   = []
    n_ok = n_skip = 0

    for i, case_info in enumerate(cases):
        case_name = case_info['case_name']
        case_id   = case_info['case_id']
        print(f"\n[{i+1}/{len(cases)}] {case_id}")

        result = process_case(case_info, args.margin, args.target_size)
        if result is None:
            n_skip += 1
            continue

        n_ok += 1
        meta = result['metadata']
        crop_sizes.append(meta['crop_shape'])

        # 保存 NPY
        np.save(os.path.join(args.output_dir, 'image',          f'{case_name}.npy'), result['image'])
        np.save(os.path.join(args.output_dir, 'binary_label',   f'{case_name}.npy'), result['binary_label'])
        np.save(os.path.join(args.output_dir, 'multiclass_mask',f'{case_name}.npy'), result['multiclass_mask'])

        # metadata 条目 (与 TopCoWTestDataset 期望格式完全兼容)
        all_metadata[case_name] = {
            'image':          os.path.join(args.output_dir, 'image',          f'{case_name}.npy'),
            'label':          os.path.join(args.output_dir, 'binary_label',   f'{case_name}.npy'),
            'multiclass_mask':os.path.join(args.output_dir, 'multiclass_mask',f'{case_name}.npy'),
            'case_id':        case_id,
            'split':          'test',
            'original_shape': meta['original_shape'],
            'crop_shape':     meta['crop_shape'],
            'padded_shape':   meta['padded_shape'],
            'spacing':        meta['spacing'],
            'foreground_ratio':             meta['foreground_ratio'],
            'num_vessel_classes_present':   meta['num_vessel_classes_present'],
        }

        print(f"  Crop : {meta['crop_shape']}  →  Pad: {meta['padded_shape']}")
        print(f"  Spacing: [{meta['spacing'][0]:.3f}, {meta['spacing'][1]:.3f}, {meta['spacing'][2]:.3f}] mm")
        print(f"  FG ratio: {meta['foreground_ratio']:.4f}  |  Vessel classes: {meta['num_vessel_classes_present']}")

    # ---- 保存 metadata ----
    meta_name = f'{args.dataset}_metadata.pkl'
    split_name = f'{args.dataset}_split.pkl'

    pkl_path = os.path.join(args.output_dir, meta_name)
    with open(pkl_path, 'wb') as f:
        pickle.dump(all_metadata, f)

    split_info = {
        'test_cases':  [v['case_id'] for v in all_metadata.values()],
        'train_cases': [],
    }
    split_path = os.path.join(args.output_dir, split_name)
    with open(split_path, 'wb') as f:
        pickle.dump(split_info, f)

    # ---- 统计 ----
    print(f"\n{'='*60}")
    print(f"完成: {n_ok} OK / {n_skip} SKIP / {len(cases)} total")

    if crop_sizes:
        arr = np.array(crop_sizes)
        target = np.array(args.target_size)
        print(f"\nROI crop 尺寸统计 [Z, Y, X]:")
        print(f"  Min : {arr.min(axis=0).tolist()}")
        print(f"  Max : {arr.max(axis=0).tolist()}")
        print(f"  Mean: {arr.mean(axis=0).astype(int).tolist()}")
        oversize = (arr > target).any(axis=1).sum()
        if oversize:
            print(f"  ⚠  {oversize} case(s) exceeded target_size — center-cropped automatically")
        else:
            print(f"  ✓  All cases fit within target_size")

    print(f"\nOutput:")
    print(f"  Metadata : {pkl_path}")
    print(f"  Split    : {split_path}")
    print(f"  Data     : {args.output_dir}/{{image,binary_label,multiclass_mask}}/")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()


# python topcow_preprocess_external.py --data_root ./MRA_IXI_HH --dataset ixi --output_dir ./ixi_processed
# python topcow_preprocess_external.py --data_root ./MRA_Lausanne --dataset lausanne --output_dir ./lausanne_processed