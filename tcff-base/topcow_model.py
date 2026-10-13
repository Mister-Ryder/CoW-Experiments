"""
topcow_model.py (v4)
====================
TopCoW TGF: 14类多分类 + LAAM在encoder skip + 可学习temperature EDL

架构:
  共享 Encoder → enc_feats [32, 64, 128, 256]
                        ↓
  ┌─────────────────────┬────────────────────────────────┐
  │   Aux Decoder       │   产出三种粗粒度控制信号        │
  │   level0(128) → circuit(4)    → pred/unc → GAAM      │
  │   level1(64)  → caliber(4)    → pred/unc → SAAM      │
  │   level2(32)  → laterality(4) → pred/unc → LAAM      │
  └─────────────────────┘                                 │
                                                          │
  LAAM: 修改 encoder skip[2] (浅层空间先验)               │
                        ↓                                 │
  Main Decoder:                                           │
    level0: skip[0](128) → GAAM融合 → 128ch               │
    level1: skip[1](64)  → SAAM融合 → 64ch                │
    level2: modified_skip[2](32) → 32ch (LAAM已在skip生效) │
                        ↓                                 │
  Output: 14类 softmax (多分类分割)                        │
  Binary: merge(pred > 0)                                 │

关键改动 vs v3:
- 主 decoder 输出 14 类 (非 2 类)
- LAAM 部署在 encoder skip connection (非 decoder)
- 可学习 temperature 校准 EDL uncertainty
- 深监督头输出 14 类
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from func.model_arch_model10 import SegAirwayModel


# ============================================================
# 标签映射
# ============================================================

CIRCUIT_LUT = [3, 1, 1, 1, 0, 0, 0, 0, 2, 2, 2, 0, 0, 0]
CALIBER_LUT = [3, 0, 1, 1, 0, 1, 0, 1, 2, 2, 2, 1, 1, 1]
LATERALITY_LUT = [3, 2, 1, 0, 1, 1, 0, 0, 1, 0, 2, 1, 0, 2]
NUM_CIRCUIT = 4
NUM_CALIBER = 4
NUM_LATERALITY = 4
NUM_SEG_CLASSES = 14


def _lut(lst):
    return torch.tensor(lst, dtype=torch.long)


# ============================================================
# 轻量辅助 Decoder Block
# ============================================================

class LightDecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, 4, stride=2, padding=1)
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
        )

    def forward(self, skip, x):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


# ============================================================
# 融合算子基类
# ============================================================

class BaseFusion(nn.Module):
    def __init__(self, seg_ch, aux_ch):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Conv3d(aux_ch, seg_ch, 1), nn.BatchNorm3d(seg_ch), nn.ReLU(inplace=True)
        )
        self.fuse = nn.Sequential(
            nn.Conv3d(seg_ch * 2, seg_ch, 3, padding=1),
            nn.BatchNorm3d(seg_ch), nn.ReLU(inplace=True)
        )
        self.W_u = nn.Sequential(nn.Conv3d(seg_ch, 1, 1), nn.Sigmoid())

    def base_fuse(self, seg_feat, aux_feat):
        if seg_feat.shape[2:] != aux_feat.shape[2:]:
            aux_feat = F.interpolate(aux_feat, size=seg_feat.shape[2:],
                                     mode='trilinear', align_corners=False)
        t = self.transform(aux_feat)
        return self.fuse(torch.cat([seg_feat, t], dim=1)) + seg_feat

    def _align_pred(self, pred, target_shape):
        if pred.shape[1:] != target_shape:
            pred = F.interpolate(
                pred.unsqueeze(1).float(), size=target_shape, mode='nearest'
            ).squeeze(1).long()
        return pred

    def _align_unc(self, unc, target_shape):
        if unc is not None and unc.shape[2:] != target_shape:
            unc = F.interpolate(unc, size=target_shape,
                                mode='trilinear', align_corners=False)
        return unc

    def uncertainty_gate(self, enhanced, fused, uncertainty):
        if uncertainty is None:
            return enhanced
        uncertainty = self._align_unc(uncertainty, fused.shape[2:])
        w = self.W_u(fused)
        return enhanced * (1 - w * uncertainty) + fused * (w * uncertainty)


# ============================================================
# GAAM: 深层 — circuit-conditioned + uncertainty gate
# ============================================================

class GAAMFusion(BaseFusion):
    def __init__(self, seg_ch, aux_ch, num_conditions=3):
        super().__init__(seg_ch, aux_ch)
        self.cond_enhancers = nn.ModuleList([
            nn.Sequential(
                nn.Conv3d(seg_ch, seg_ch, 3, padding=1, groups=seg_ch),
                nn.BatchNorm3d(seg_ch), nn.Sigmoid()
            ) for _ in range(num_conditions)
        ])

    def forward(self, seg_feat, aux_feat, prediction=None, uncertainty=None):
        fused = self.base_fuse(seg_feat, aux_feat)
        if prediction is not None:
            prediction = self._align_pred(prediction, fused.shape[2:])
            enhanced = torch.zeros_like(fused)
            for cid, enh in enumerate(self.cond_enhancers):
                mask = (prediction == cid).unsqueeze(1).float()
                enhanced = enhanced + fused * enh(fused) * mask
            bg = (prediction >= len(self.cond_enhancers)).unsqueeze(1).float()
            enhanced = enhanced + fused * bg
        else:
            enhanced = fused * self.cond_enhancers[0](fused)
        return self.uncertainty_gate(enhanced, fused, uncertainty)


# ============================================================
# SAAM: 中层 — caliber-conditioned morphology + uncertainty gate
# ============================================================

class SAAMFusion(BaseFusion):
    def __init__(self, seg_ch, aux_ch, n_branches=4, num_conditions=3):
        super().__init__(seg_ch, aux_ch)
        self.n_branches = n_branches
        self.morphs = nn.ModuleList([
            nn.Conv3d(seg_ch, seg_ch // n_branches, 3, padding=d, dilation=d)
            for d in [1, 1, 2, 3]
        ])
        self.cond_weights = nn.ParameterList([
            nn.Parameter(torch.ones(n_branches) / n_branches)
            for _ in range(num_conditions)
        ])
        self.morph_fuse = nn.Conv3d(seg_ch, seg_ch, 1)

    def forward(self, seg_feat, aux_feat, prediction=None, uncertainty=None):
        fused = self.base_fuse(seg_feat, aux_feat)
        morph_outs = [m(fused) for m in self.morphs]
        if prediction is not None:
            prediction = self._align_pred(prediction, fused.shape[2:])
            enhanced = torch.zeros_like(fused)
            for cid, cw in enumerate(self.cond_weights):
                w = F.softmax(cw, dim=0)
                weighted = torch.cat([morph_outs[j] * w[j] for j in range(self.n_branches)], dim=1)
                mask = (prediction == cid).unsqueeze(1).float()
                enhanced = enhanced + self.morph_fuse(weighted) * mask
            bg = (prediction >= len(self.cond_weights)).unsqueeze(1).float()
            enhanced = enhanced + fused * bg
            enhanced = fused + enhanced
        else:
            enhanced = fused + self.morph_fuse(torch.cat(morph_outs, dim=1))
        return self.uncertainty_gate(enhanced, fused, uncertainty)


# ============================================================
# LAAM: 浅层 — laterality-routed + uncertainty gate
# 部署在 encoder skip connection 上
# ============================================================

class LAAMFusion(BaseFusion):
    def __init__(self, seg_ch, aux_ch, num_regions=3):
        super().__init__(seg_ch, aux_ch)
        self.region_convs = nn.ModuleList([
            nn.Conv3d(seg_ch, seg_ch, 3, padding=1) for _ in range(num_regions)
        ])

    def forward(self, seg_feat, aux_feat, prediction=None, uncertainty=None):
        fused = self.base_fuse(seg_feat, aux_feat)
        if prediction is not None:
            prediction = self._align_pred(prediction, fused.shape[2:])
            enhanced = torch.zeros_like(fused)
            for rid, conv in enumerate(self.region_convs):
                mask = (prediction == rid).unsqueeze(1).float()
                enhanced = enhanced + conv(fused) * mask
            bg = (prediction >= len(self.region_convs)).unsqueeze(1).float()
            enhanced = enhanced + fused * bg
        else:
            enhanced = fused
        return self.uncertainty_gate(enhanced, fused, uncertainty)


# ============================================================
# 主模型
# ============================================================

class TopCoWTGFModel(nn.Module):
    def __init__(self, num_seg_classes=NUM_SEG_CLASSES, disable_gating=False):
        super().__init__()
        self.num_seg_classes = num_seg_classes
        self.disable_gating = disable_gating

        # 共享 Encoder
        base = SegAirwayModel(in_channels=1, out_channels=2)
        self.encoders = base.encoders
        self.seg_decoders = base.decoders
        self.fusion_module_decoder = base.fusion_module_decoder
        self.dropout2 = base.dropout2

        # 替换最终输出: 6ch → 14类 (原 2类)
        self.seg_final = nn.Conv3d(6, num_seg_classes, 1)
        self.seg_activation = nn.Softmax(dim=1)

        # 辅助 Decoder
        self.aux_decoders = nn.ModuleList([
            LightDecoderBlock(256, 128, 128),
            LightDecoderBlock(128, 64, 64),
            LightDecoderBlock(64, 32, 32),
        ])

        # 三个独立预测头
        self.circuit_head = nn.Conv3d(128, NUM_CIRCUIT, 1)
        self.caliber_head = nn.Conv3d(64, NUM_CALIBER, 1)
        self.laterality_head = nn.Conv3d(32, NUM_LATERALITY, 1)

        # 三种融合算子
        self.gaam = GAAMFusion(128, 128, num_conditions=3)   # 深层, main_dec[0]
        self.saam = SAAMFusion(64, 64, num_conditions=3)     # 中层, main_dec[1]
        self.laam = LAAMFusion(32, 32, num_regions=3)        # 浅层, encoder skip

        # 深监督头 (14类)
        self.ds_heads = nn.ModuleList([
            nn.Conv3d(128, num_seg_classes, 1),
            nn.Conv3d(64, num_seg_classes, 1),
            nn.Conv3d(32, num_seg_classes, 1),
        ])

        # 可学习 temperature (EDL 校准)
        self.temperature = nn.Parameter(torch.ones(1))

        # LUT
        self.register_buffer('lat_lut', _lut(LATERALITY_LUT))

    def _edl(self, logits, num_classes):
        """EDL with learnable temperature"""
        evidence = F.relu(logits / self.temperature.clamp(min=0.1))
        alpha = evidence + 1.0
        S = alpha.sum(dim=1, keepdim=True)
        uncertainty = num_classes / S
        pred = torch.argmax(logits, dim=1)
        return pred, uncertainty

    def forward(self, x, return_aux=True, aux_only=False):
        # ===== 1. 共享 Encoder =====
        enc_feats = []
        for encoder in self.encoders:
            x = encoder(x)
            enc_feats.append(x)

        enc_rev = list(reversed(enc_feats))
        bottleneck = enc_rev[0]         # 256ch
        skips = enc_rev[1:]             # [128ch, 64ch, 32ch]

        # ===== 2. 辅助 Decoder =====
        aux_x = bottleneck
        aux_feats = []
        for i, dec in enumerate(self.aux_decoders):
            aux_x = dec(skips[i], aux_x)
            aux_feats.append(aux_x)

        circuit_logits = self.circuit_head(aux_feats[0])
        caliber_logits = self.caliber_head(aux_feats[1])
        laterality_logits = self.laterality_head(aux_feats[2])

        aux_output = {
            'circuit_logits': circuit_logits,
            'caliber_logits': caliber_logits,
            'laterality_logits': laterality_logits,
            'ds_logits': None,
        }

        # Stage 1 预训练: 只需要辅助头输出，跳过 main decoder 节省显存
        if aux_only:
            return None, aux_output, None

        with torch.no_grad():
            circuit_pred, circuit_unc = self._edl(circuit_logits, NUM_CIRCUIT)
            caliber_pred, caliber_unc = self._edl(caliber_logits, NUM_CALIBER)
            lateral_pred, lateral_unc = self._edl(laterality_logits, NUM_LATERALITY)

        if self.disable_gating:
            circuit_unc = None
            caliber_unc = None
            lateral_unc = None

        # ===== 3. LAAM: 修改 encoder skip[2] =====
        lat_route = self.lat_lut[lateral_pred.clamp(0, len(self.lat_lut) - 1)]
        modified_skip2 = self.laam(skips[2], aux_feats[2],
                                   prediction=lat_route, uncertainty=lateral_unc)

        # ===== 4. 主 Decoder + GAAM/SAAM 融合 =====
        main_x = bottleneck
        main_feats = []

        for i, seg_dec in enumerate(self.seg_decoders):
            if i == 2:
                main_x = seg_dec(modified_skip2, main_x)
            else:
                main_x = seg_dec(skips[i], main_x)

            if i == 0:
                main_x = self.gaam(main_x, aux_feats[0],
                                   prediction=circuit_pred, uncertainty=circuit_unc)
            elif i == 1:
                main_x = self.saam(main_x, aux_feats[1],
                                   prediction=caliber_pred, uncertainty=caliber_unc)

            main_feats.append(main_x)

        # ===== 5. 输出 =====
        dec_fusion = self.dropout2(self.fusion_module_decoder(main_feats))
        seg_logits = self.seg_final(dec_fusion)
        seg = self.seg_activation(seg_logits)

        # 深监督
        ds_logits = [head(feat) for feat, head in zip(main_feats, self.ds_heads)]
        aux_output['ds_logits'] = ds_logits

        return seg, aux_output, seg_logits


# ============================================================
# U-Net Baseline (14类, 无辅助)
# ============================================================

class TopCoWUNet(nn.Module):
    def __init__(self, num_seg_classes=NUM_SEG_CLASSES):
        super().__init__()
        base = SegAirwayModel(in_channels=1, out_channels=2)
        self.encoders = base.encoders
        self.decoders = base.decoders
        self.fusion_module_decoder = base.fusion_module_decoder
        self.dropout2 = base.dropout2
        self.seg_final = nn.Conv3d(6, num_seg_classes, 1)
        self.seg_activation = nn.Softmax(dim=1)

    def forward(self, x, return_aux=True, aux_only=False):
        if aux_only:
            return None, {'circuit_logits': None, 'caliber_logits': None,
                          'laterality_logits': None, 'ds_logits': None}, None
        enc_feats = []
        for enc in self.encoders:
            x = enc(x)
            enc_feats.append(x)
        enc_rev = list(reversed(enc_feats))
        main_x = enc_rev[0]
        skips = enc_rev[1:]
        main_feats = []
        for i, dec in enumerate(self.decoders):
            main_x = dec(skips[i], main_x)
            main_feats.append(main_x)
        dec_fusion = self.dropout2(self.fusion_module_decoder(main_feats))
        seg_logits = self.seg_final(dec_fusion)
        seg = self.seg_activation(seg_logits)
        aux = {'circuit_logits': None, 'caliber_logits': None,
               'laterality_logits': None, 'ds_logits': None}
        return seg, aux, seg_logits


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    print("Testing TopCoWTGFModel v4 (multiclass)...")
    model = TopCoWTGFModel(num_seg_classes=14, disable_gating=False)
    x = torch.randn(1, 1, 64, 160, 160)
    model.eval()
    with torch.no_grad():
        seg, aux, logits = model(x)
    print(f"Seg: {seg.shape}")          # [1, 14, D, H, W]
    print(f"Logits: {logits.shape}")
    print(f"Circuit: {aux['circuit_logits'].shape}")
    print(f"Caliber: {aux['caliber_logits'].shape}")
    print(f"Laterality: {aux['laterality_logits'].shape}")
    for i, ds in enumerate(aux['ds_logits']):
        print(f"  DS level {i}: {ds.shape}")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Temperature: {model.temperature.item():.4f}")

    # nogate
    m2 = TopCoWTGFModel(disable_gating=True)
    m2.eval()
    with torch.no_grad():
        s2, _, _ = m2(x)
    print(f"Nogate: {s2.shape}")

    # UNet
    m3 = TopCoWUNet()
    m3.eval()
    with torch.no_grad():
        s3, _, _ = m3(x)
    print(f"UNet: {s3.shape}")

    print("All tests passed!")