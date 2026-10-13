#!/usr/bin/env python3
"""
topcow_infer.py (v3)
====================
修正版推理脚本：严格按照官方 ROI 框裁切，支持 margin 外扩，自动调整输入尺寸以避免中心裁剪。
输出：
  *_pred_roi.nii.gz       -> ROI 空间的分割（与 image_roi 对齐）
  *_pred_fullbrain.nii.gz -> 全脑空间的分割（与原始 NIfTI 对齐）
"""

import os
import re
import sys
import glob
import json
import argparse
import numpy as np
import torch
import SimpleITK as sitk

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
SAM_MED3D_PATH = '/mnt/e/00-Data/Brain-TopCoW/24/topcow-TGF/SAM-Med3D'
sys.path.insert(0, SAM_MED3D_PATH)

TARGET_SIZE = (64, 160, 160)   # 模型输入尺寸 [Z, Y, X]
NUM_CLASSES = 14

VESSEL_NAMES = {
    0: "Background", 1: "BA", 2: "R-PCA", 3: "L-PCA",
    4: "R-ICA", 5: "R-MCA", 6: "L-ICA", 7: "L-MCA",
    8: "R-Pcom", 9: "L-Pcom", 10: "Acom",
    11: "R-ACA", 12: "L-ACA", 13: "3rd-A2"
}

# ------------------------------------------------------------
# 模型加载（保持原有逻辑）
# ------------------------------------------------------------
def load_model(model_dir, device):
    ckpt_path = os.path.join(model_dir, 'best_model.pth')
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"No best_model.pth in {model_dir}")
    config_path = os.path.join(model_dir, 'config.json')
    foundation_type = fusion_type = model_type = None
    config = {}
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        foundation_type = config.get('foundation_type')
        fusion_type = config.get('fusion_type')
        model_type = config.get('model_type', 'tgf')

    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    sd = state.get('model_state_dict', state)

    try:
        from topcow_foundation_model import (
            TopCoWFoundationModel, build_foundation_encoder, FOUNDATION_TYPES
        )
        if foundation_type and foundation_type in FOUNDATION_TYPES:
            enc = build_foundation_encoder(foundation_type, config.get('foundation_ckpt', None))
            m = TopCoWFoundationModel(enc, fusion_type=fusion_type or 'tgf')
            m.to(device).load_state_dict(sd, strict=True)
            m.eval()
            if hasattr(m, 'set_epoch'):
                m.set_epoch(999999)
            print(f"  Loaded foundation: {foundation_type} + {fusion_type}")
            return m
    except ImportError:
        pass

    from topcow_model import TopCoWTGFModel, TopCoWUNet
    from topcow_model_baselines import TopCoWBaselineFusionModel

    if model_type == 'unet':
        m = TopCoWUNet()
    elif model_type in ('tgf', 'tgf_pretrain'):
        m = TopCoWTGFModel(disable_gating=False)
    elif model_type == 'tgf_nogate':
        m = TopCoWTGFModel(disable_gating=True)
    elif model_type in ('concat', 'crossattn', 'bagf', 'pug', 'dsff'):
        m = TopCoWBaselineFusionModel(fusion_type=model_type)
    else:
        raise ValueError(f"Unknown model_type={model_type}")

    m.load_state_dict(sd, strict=False)
    m.to(device).eval()
    if hasattr(m, 'set_epoch'):
        m.set_epoch(999999)
    print(f"  Loaded baseline: {model_type}")
    return m

# ------------------------------------------------------------
# 推理与空间处理
# ------------------------------------------------------------
@torch.no_grad()
def infer_volume(model, image_np, device, use_tta=True):
    x = torch.from_numpy(image_np).float().unsqueeze(0).unsqueeze(0).to(device)
    if use_tta:
        preds = []
        for fz in (False, True):
            for fy in (False, True):
                a = x.clone()
                if fz: a = a.flip(2)
                if fy: a = a.flip(3)
                out, _, _ = model(a)
                if fy: out = out.flip(3)
                if fz: out = out.flip(2)
                preds.append(out)
        seg = torch.stack(preds).mean(0)
    else:
        seg, _, _ = model(x)
    return torch.argmax(seg[0], dim=0).cpu().numpy().astype(np.uint8)

def read_roi_txt(roi_path):
    with open(roi_path) as f:
        content = f.read()
    sm = re.search(r'Size\s*\(Voxels\)\s*:\s*(\d+)\s+(\d+)\s+(\d+)', content)
    lm = re.search(r'Location\s*\(Voxels\)\s*:\s*(\d+)\s+(\d+)\s+(\d+)', content)
    if sm and lm:
        size   = [int(sm.group(i)) for i in range(1, 4)]   # [X, Y, Z]!
        origin = [int(lm.group(i)) for i in range(1, 4)]   # [X, Y, Z]!
    else:
        nums = [int(x) for x in re.findall(r'\d+', content)]
        if len(nums) == 6:
            size, origin = nums[:3], nums[3:]
        else:
            raise ValueError(f"Cannot parse ROI: {roi_path}")
    # 转换为 [Z, Y, X] 顺序
    return {
        'x_min': origin[0], 'y_min': origin[1], 'z_min': origin[2],
        'x_max': origin[0] + size[0],
        'y_max': origin[1] + size[1],
        'z_max': origin[2] + size[2],
    }

def normalize_mra(image, p_low=0.5, p_high=99.5):
    image = image.astype(np.float32)
    lo, hi = np.percentile(image, p_low), np.percentile(image, p_high)
    if hi - lo < 1e-6:
        return np.zeros_like(image)
    return np.clip((image - lo) / (hi - lo), 0, 1)

def pad_to_fixed_size(array, target_size, pad_value=0):
    """将数组中心填充或裁剪到 target_size，不进行中心裁剪（若输入大于目标则只取中心部分，但会记录偏移）"""
    current = np.array(array.shape)
    target = np.array(target_size)
    offset = np.zeros(3, dtype=int)
    # 如果某维大于目标，则中心裁剪（但会记录偏移）
    for i in range(3):
        if current[i] > target[i]:
            start = (current[i] - target[i]) // 2
            offset[i] = start
            sl = [slice(None)] * 3
            sl[i] = slice(start, start + target[i])
            array = array[tuple(sl)]
    current = np.array(array.shape)
    pad_total = target - current
    pad_before = pad_total // 2
    pad_after = pad_total - pad_before
    padded = np.pad(array, [(pad_before[i], pad_after[i]) for i in range(3)],
                    mode='constant', constant_values=pad_value)
    return padded, pad_before, offset

def unpad(padded, crop_shape, target_size, pad_before):
    result = padded
    for i in range(3):
        cs = min(crop_shape[i], target_size[i])
        pb = int(pad_before[i])
        sl = [slice(None)] * 3
        sl[i] = slice(pb, pb + cs)
        result = result[tuple(sl)]
    return result

def embed_in_fullbrain(roi_arr, full_shape, x_min, y_min, z_min, offset=None):
    """将 ROI 空间数组放回全脑，offset 为 (z_off, y_off, x_off) 中心裁剪偏移"""
    full = np.zeros(full_shape, dtype=roi_arr.dtype)
    z0 = z_min + (offset[0] if offset is not None else 0)
    y0 = y_min + (offset[1] if offset is not None else 0)
    x0 = x_min + (offset[2] if offset is not None else 0)
    zr, yr, xr = roi_arr.shape
    ze = min(z0 + zr, full_shape[0])
    ye = min(y0 + yr, full_shape[1])
    xe = min(x0 + xr, full_shape[2])
    full[z0:ze, y0:ye, x0:xe] = roi_arr[:ze - z0, :ye - y0, :xe - x0]
    return full

def save_nifti(array_zyx, save_path, spacing=(1.,1.,1.), origin=(0.,0.,0.), direction=None, dtype=np.uint8):
    img = sitk.GetImageFromArray(array_zyx.astype(dtype))
    img.SetSpacing(spacing)
    img.SetOrigin(origin)
    if direction is not None:
        img.SetDirection(direction)
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    sitk.WriteImage(img, save_path)

# ------------------------------------------------------------
# 预处理单个 NIfTI
# ------------------------------------------------------------
def preprocess_nifti(img_path, roi_path, margin, target_size):
    img_sitk = sitk.ReadImage(img_path)
    img_np = sitk.GetArrayFromImage(img_sitk)   # [Z,Y,X]
    spacing = img_sitk.GetSpacing()
    origin = img_sitk.GetOrigin()
    direction = img_sitk.GetDirection()
    full_shape = img_np.shape

    if roi_path and os.path.exists(roi_path):
        roi = read_roi_txt(roi_path)
        x_min = max(0, roi['x_min'] - margin)
        y_min = max(0, roi['y_min'] - margin)
        z_min = max(0, roi['z_min'] - margin)
        x_max = min(full_shape[2], roi['x_max'] + margin)
        y_max = min(full_shape[1], roi['y_max'] + margin)
        z_max = min(full_shape[0], roi['z_max'] + margin)
        crop_origin = img_sitk.TransformIndexToPhysicalPoint([int(x_min), int(y_min), int(z_min)])
    else:
        x_min, y_min, z_min = 0, 0, 0
        x_max, y_max, z_max = full_shape[2], full_shape[1], full_shape[0]
        crop_origin = origin

    img_crop = img_np[z_min:z_max, y_min:y_max, x_min:x_max]
    crop_shape = img_crop.shape

    # 归一化
    img_norm = normalize_mra(img_crop)

    # 若 crop_shape 超过 target_size，会进行中心裁剪（保留偏移信息）
    img_padded, pad_before, crop_offset = pad_to_fixed_size(img_norm, target_size)

    meta = {
        'spacing': spacing,
        'origin': origin,
        'direction': direction,
        'crop_origin': crop_origin,
        'full_shape': full_shape,
        'crop_start': (x_min, y_min, z_min),
        'crop_shape': crop_shape,
        'pad_before': pad_before,
        'crop_offset': crop_offset,   # 中心裁剪偏移 (z,y,x)
        'margin': margin,
        'target_size': target_size,
    }
    return img_padded, meta

# ------------------------------------------------------------
# Mode A: 原始 NIfTI 推理
# ------------------------------------------------------------
def discover_cases_raw(data_root, dataset):
    img_dir = os.path.join(data_root, 'imagesTr')
    roi_dir = os.path.join(data_root, 'roi_loc_labelsTr')
    cases = []
    if dataset == 'ixi':
        for img_path in sorted(glob.glob(os.path.join(img_dir, 'IXI*braincase.nii.gz'))):
            stem = os.path.basename(img_path).replace('.nii.gz', '')
            case_id = stem.split('-MRA_')[0]
            roi_key = case_id + '-MRA_LPS_DFmasked_ROI-cow'
            cases.append({
                'case_id': case_id,
                'case_name': 'ixi_' + case_id.replace('-', '_'),
                'img_path': img_path,
                'roi_path': os.path.join(roi_dir, roi_key + '.txt'),
            })
    elif dataset == 'lausanne':
        for img_path in sorted(glob.glob(os.path.join(img_dir, 'sub-*_angio_LPS.nii.gz'))):
            stem = os.path.basename(img_path).replace('.nii.gz', '')
            case_id = '_'.join(stem.split('_')[:2])
            cases.append({
                'case_id': case_id,
                'case_name': 'lausanne_' + case_id,
                'img_path': img_path,
                'roi_path': os.path.join(roi_dir, stem + '.txt'),
            })
    elif dataset == 'topcow':
        modality = 'mr'
        for img_path in sorted(glob.glob(os.path.join(img_dir, f'topcow_{modality}_*_0000.nii.gz'))):
            parts = os.path.basename(img_path).replace('_0000.nii.gz','').split('_')
            case_id = parts[2]
            prefix = f'topcow_{modality}_{case_id}'
            cases.append({
                'case_id': case_id,
                'case_name': prefix,
                'img_path': img_path,
                'roi_path': os.path.join(roi_dir, f'{prefix}.txt'),
            })
    else:
        for img_path in sorted(glob.glob(os.path.join(img_dir, '*.nii.gz'))):
            stem = os.path.basename(img_path).replace('_0000.nii.gz','').replace('.nii.gz','')
            cases.append({
                'case_id': stem,
                'case_name': stem,
                'img_path': img_path,
                'roi_path': None,
            })
    return cases

def infer_raw(args, model, device):
    cases = discover_cases_raw(args.data_root, args.dataset)
    print(f"\n{'='*70}")
    print(f"Mode: raw   Dataset: {args.dataset}   Cases: {len(cases)}")
    print(f"Margin: {args.margin}   TTA: {not args.no_tta}")
    print(f"Target size (model input): {TARGET_SIZE}")
    print(f"Output: {args.output_dir}")
    print(f"{'='*70}")
    os.makedirs(args.output_dir, exist_ok=True)

    for ci in cases:
        case_name = ci['case_name']
        if not os.path.exists(ci['img_path']):
            print(f"\n[SKIP] {case_name}: image not found")
            continue
        print(f"\n{case_name}")
        try:
            img_padded, meta = preprocess_nifti(ci['img_path'], ci['roi_path'], args.margin, TARGET_SIZE)
        except Exception as e:
            print(f"  [ERROR] preprocess: {e}")
            continue

        # 推理
        pred_padded = infer_volume(model, img_padded, device, use_tta=not args.no_tta)

        # 还原 ROI 空间 (unpad)
        pred_roi = unpad(pred_padded, meta['crop_shape'], TARGET_SIZE, meta['pad_before'])
        # 还原全脑
        x_min, y_min, z_min = meta['crop_start']
        pred_full = embed_in_fullbrain(pred_roi, meta['full_shape'],
                                       x_min, y_min, z_min,
                                       offset=meta.get('crop_offset', (0,0,0)))

        # 保存 ROI 分割
        roi_path = os.path.join(args.output_dir, f'{case_name}_pred_roi.nii.gz')
        save_nifti(pred_roi, roi_path,
                   spacing=meta['spacing'], origin=meta['crop_origin'],
                   direction=meta['direction'])
        # 保存全脑分割
        full_path = os.path.join(args.output_dir, f'{case_name}_pred_fullbrain.nii.gz')
        save_nifti(pred_full, full_path,
                   spacing=meta['spacing'], origin=meta['origin'],
                   direction=meta['direction'])

        print(f"  ROI pred     → shape {pred_roi.shape}, origin {meta['crop_origin']}")
        print(f"  Fullbrain    → shape {pred_full.shape}, origin {meta['origin']}")

        if args.save_image:
            img_roi = unpad(img_padded, meta['crop_shape'], TARGET_SIZE, meta['pad_before'])
            img_roi_path = os.path.join(args.output_dir, f'{case_name}_image_roi.nii.gz')
            save_nifti(img_roi, img_roi_path,
                       spacing=meta['spacing'], origin=meta['crop_origin'],
                       direction=meta['direction'], dtype=np.float32)
            print(f"  Image (roi)  → {img_roi_path}")

        # 打印维度信息
        print(f"  Original ROI (txt) : X[{meta['crop_start'][0]}:{meta['crop_start'][0]+meta['crop_shape'][2]}]"
              f" Y[{meta['crop_start'][1]}:{meta['crop_start'][1]+meta['crop_shape'][1]}]"
              f" Z[{meta['crop_start'][2]}:{meta['crop_start'][2]+meta['crop_shape'][0]}]")
        print(f"  + margin {args.margin} → crop shape {meta['crop_shape']}")
        print(f"  → padded to {TARGET_SIZE} (pad_before={meta['pad_before']})")
        if any(meta.get('crop_offset', (0,0,0))):
            print(f"  ⚠ Warning: crop_shape exceeded target_size, center-cropped offset={meta['crop_offset']}")

# ------------------------------------------------------------
# 主函数
# ------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description='TopCoW inference v3')
    p.add_argument('--mode', required=True, choices=['raw', 'preprocessed'])  # preprocessed 未改动暂略
    p.add_argument('--data_root', required=True, help='[raw] 原始数据集根目录')
    p.add_argument('--dataset', default='ixi', choices=['ixi','lausanne','topcow','generic'])
    p.add_argument('--model_dir', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--margin', type=int, default=0, help='ROI 外扩体素数 (默认0，严格遵循txt)')
    p.add_argument('--no_tta', action='store_true')
    p.add_argument('--save_image', action='store_true')
    p.add_argument('--gpu', type=int, default=0)
    args = p.parse_args()

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = load_model(args.model_dir, device)
    if args.mode == 'raw':
        infer_raw(args, model, device)
    else:
        p.error('preprocessed mode not updated in this version')

if __name__ == '__main__':
    main()