"""
topcow_foundation_model.py (multi-model, v4-aligned, TCFF)
==========================================================
多种大模型特征融合: 冻结 Foundation Encoder → TGF/Concat → UNet

支持的 Foundation Encoder:
  sam3d_turbo  — SAM-Med3D turbo (vit_b_ori, 768→384), ECCV 2024
  swinunetr    — SwinUNETR encoder (Swin-B, 48→384), CVPR 2022, 5050 CT 预训练
  mock         — 随机卷积 (测试用)

融合位置严格对齐 v4:
  Level 2 (32ch): encoder skip[2] (LAAM 位置)
  Level 0 (128ch): decoder step 0 后 (GAAM 位置)
  Level 1 (64ch): decoder step 1 后 (SAAM 位置)

TCFF 改动 (vs 旧版):
  - FoundationTGF: 移除 gate_floor, 新增通道级信任适配 + 通道级 EDL
  - return_unc 返回 trust_info (channel_weight, aux_global) 用于 CTC loss
  - 其他融合方法 (concat/crossattn/bagf/pug/dsff) 完全不变

用法:
  --foundation_type sam3d_turbo --foundation_ckpt ./sam_med3d_turbo.pth
  --foundation_type swinunetr  --foundation_ckpt ./model_swinvit.pt
  --foundation_type mock
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from func.model_arch_model10 import SegAirwayModel

NUM_SEG_CLASSES = 14
FOUNDATION_TYPES = ['sam3d_turbo', 'swinunetr', 'mock']


# ============================================================
# Foundation Encoder 基类
# ============================================================

class FoundationEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.output_dim = 0

    def forward(self, x):
        """[B,1,D,H,W] → [B, output_dim, d, h, w]"""
        raise NotImplementedError


# ============================================================
# Mock Encoder (测试用)
# ============================================================

class MockFoundationEncoder(FoundationEncoder):
    def __init__(self, output_dim=384):
        super().__init__()
        self.output_dim = output_dim
        self.net = nn.Sequential(
            nn.Conv3d(1, 64, 3, stride=2, padding=1), nn.BatchNorm3d(64), nn.ReLU(True),
            nn.Conv3d(64, 128, 3, stride=2, padding=1), nn.BatchNorm3d(128), nn.ReLU(True),
            nn.Conv3d(128, output_dim, 3, stride=2, padding=1), nn.BatchNorm3d(output_dim), nn.ReLU(True),
            nn.Conv3d(output_dim, output_dim, 3, stride=2, padding=1), nn.BatchNorm3d(output_dim), nn.ReLU(True),
        )
        print(f"  [MockEncoder] output_dim={output_dim}")

    def forward(self, x):
        return self.net(x)


# ============================================================
# SAM-Med3D turbo Encoder
# ============================================================

class SAMMed3DEncoder(FoundationEncoder):
    """SAM-Med3D turbo: vit_b_ori, embed=768, neck→384, input=128³"""
    def __init__(self, checkpoint_path=None, model_type='vit_b_ori',
                 output_dim=384, input_size=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_size = input_size
        self.sam_encoder = self._load(checkpoint_path, model_type)
        if self.sam_encoder is None:
            self._fallback(output_dim)

    def _load(self, ckpt_path, model_type):
        if not ckpt_path or not os.path.exists(ckpt_path):
            print(f"  [SAM-Med3D] No checkpoint"); return None
        try:
            from segment_anything.build_sam3D import sam_model_registry3D
            sam = sam_model_registry3D[model_type](checkpoint=None)
            state = torch.load(ckpt_path, map_location='cpu')
            if 'model_state_dict' in state:
                state = state['model_state_dict']
            sam.load_state_dict(state, strict=True)
            encoder = sam.image_encoder
            for p in encoder.parameters(): p.requires_grad = False
            n = sum(p.numel() for p in encoder.parameters())
            print(f"  [SAM-Med3D] Loaded {model_type}, params={n:,}, output={self.output_dim}")
            return encoder
        except Exception as e:
            print(f"  [SAM-Med3D] Load failed: {e}"); return None

    def _fallback(self, dim):
        print(f"  [SAM-Med3D] Using fallback conv")
        self.sam_encoder = nn.Sequential(
            nn.Conv3d(1, 64, 7, stride=2, padding=3), nn.BatchNorm3d(64), nn.ReLU(True),
            nn.Conv3d(64, 128, 3, stride=2, padding=1), nn.BatchNorm3d(128), nn.ReLU(True),
            nn.Conv3d(128, 256, 3, stride=2, padding=1), nn.BatchNorm3d(256), nn.ReLU(True),
            nn.Conv3d(256, dim, 3, stride=2, padding=1), nn.BatchNorm3d(dim), nn.ReLU(True),
        )

    def forward(self, x):
        if any(s != self.input_size for s in x.shape[2:]):
            x = F.interpolate(x, size=[self.input_size]*3, mode='trilinear', align_corners=False)
        with torch.no_grad():
            feat = self.sam_encoder(x)
            if isinstance(feat, (tuple, list)): feat = feat[0]
            if feat.dim() == 3:
                B, N, C = feat.shape
                s = round(N ** (1/3))
                feat = feat.transpose(1, 2).reshape(B, C, s, s, s)
        return feat


# ============================================================
# SwinUNETR Encoder (via MONAI)
# ============================================================

class SwinUNETREncoder(FoundationEncoder):
    """SwinUNETR encoder: 3D Swin Transformer, 5050 CT SSL pretrained"""
    def __init__(self, checkpoint_path=None, feature_size=48, input_size=96):
        super().__init__()
        self.output_dim = feature_size * 8  # 48*8 = 384
        self.input_size = input_size
        self.swin_encoder = self._load(checkpoint_path, feature_size)
        if self.swin_encoder is None:
            self._fallback(self.output_dim)

    def _load(self, ckpt_path, feature_size):
        try:
            from monai.networks.nets.swin_unetr import SwinUNETR
            model = SwinUNETR(
                img_size=(96, 96, 96), in_channels=1, out_channels=14,
                feature_size=feature_size, spatial_dims=3,
            )
            if ckpt_path and os.path.exists(ckpt_path):
                state = torch.load(ckpt_path, map_location='cpu')
                if 'state_dict' in state:
                    state = state['state_dict']
                if any(k.startswith('module.') for k in state.keys()):
                    state = {k.replace('module.', ''): v for k, v in state.items()}
                if any(k.startswith('swin_vit') for k in state.keys()):
                    state = {k.replace('swin_vit', 'swinViT'): v for k, v in state.items()}
                model.load_state_dict(state, strict=False)
                print(f"  [SwinUNETR] Loaded pretrained encoder, output={self.output_dim}")
            else:
                print(f"  [SwinUNETR] No checkpoint, random init, output={self.output_dim}")

            encoder = model.swinViT
            for p in encoder.parameters(): p.requires_grad = False
            n = sum(p.numel() for p in encoder.parameters())
            print(f"  [SwinUNETR] Encoder params={n:,}")
            return encoder
        except ImportError:
            print(f"  [SwinUNETR] MONAI not installed: pip install monai")
            return None
        except Exception as e:
            print(f"  [SwinUNETR] Load failed: {e}")
            return None

    def _fallback(self, dim):
        print(f"  [SwinUNETR] Using fallback conv")
        self.swin_encoder = nn.Sequential(
            nn.Conv3d(1, 64, 7, stride=2, padding=3), nn.BatchNorm3d(64), nn.ReLU(True),
            nn.Conv3d(64, 128, 3, stride=2, padding=1), nn.BatchNorm3d(128), nn.ReLU(True),
            nn.Conv3d(128, 256, 3, stride=2, padding=1), nn.BatchNorm3d(256), nn.ReLU(True),
            nn.Conv3d(256, dim, 3, stride=2, padding=1), nn.BatchNorm3d(dim), nn.ReLU(True),
        )
        self._is_fallback = True

    def forward(self, x):
        if any(s != self.input_size for s in x.shape[2:]):
            x = F.interpolate(x, size=[self.input_size]*3, mode='trilinear', align_corners=False)
        with torch.no_grad():
            if hasattr(self, '_is_fallback') and self._is_fallback:
                return self.swin_encoder(x)
            feats = self.swin_encoder(x)
            if isinstance(feats, (list, tuple)):
                return feats[-1]
            return feats


# ============================================================
# Foundation Encoder 工厂
# ============================================================

def build_foundation_encoder(foundation_type, checkpoint_path=None):
    """统一的 encoder 构造函数"""
    if foundation_type == 'sam3d_turbo':
        return SAMMed3DEncoder(checkpoint_path=checkpoint_path)
    elif foundation_type == 'swinunetr':
        return SwinUNETREncoder(checkpoint_path=checkpoint_path)
    elif foundation_type == 'mock':
        return MockFoundationEncoder()
    else:
        raise ValueError(f"Unknown foundation_type: {foundation_type}. "
                         f"Options: {FOUNDATION_TYPES}")


# ============================================================
# 多尺度投影
# ============================================================

class MultiScaleProjector(nn.Module):
    """foundation bottleneck → 3 个 UNet level (128, 64, 32)"""
    def __init__(self, in_dim, out_dims=(128, 64, 32)):
        super().__init__()
        self.proj_deep = nn.Sequential(
            nn.Conv3d(in_dim, out_dims[0], 1), nn.BatchNorm3d(out_dims[0]), nn.ReLU(True))
        self.proj_mid = nn.Sequential(
            nn.ConvTranspose3d(in_dim, out_dims[1], 4, stride=2, padding=1),
            nn.BatchNorm3d(out_dims[1]), nn.ReLU(True))
        self.proj_shallow = nn.Sequential(
            nn.ConvTranspose3d(in_dim, in_dim // 2, 4, stride=2, padding=1),
            nn.BatchNorm3d(in_dim // 2), nn.ReLU(True),
            nn.ConvTranspose3d(in_dim // 2, out_dims[2], 4, stride=2, padding=1),
            nn.BatchNorm3d(out_dims[2]), nn.ReLU(True))

    def forward(self, feat, target_shapes):
        f0 = self.proj_deep(feat)
        f1 = self.proj_mid(feat)
        f2 = self.proj_shallow(feat)
        if f0.shape[2:] != target_shapes[0]:
            f0 = F.interpolate(f0, size=target_shapes[0], mode='trilinear', align_corners=False)
        if f1.shape[2:] != target_shapes[1]:
            f1 = F.interpolate(f1, size=target_shapes[1], mode='trilinear', align_corners=False)
        if f2.shape[2:] != target_shapes[2]:
            f2 = F.interpolate(f2, size=target_shapes[2], mode='trilinear', align_corners=False)
        return [f0, f1, f2]


# ============================================================
# 融合模块: 全部 baseline + TGF (TCFF)
# 所有模块统一接口: forward(seg_feat, aux_feat) → fused_feat
# ============================================================

FUSION_TYPES = ['tgf', 'concat', 'crossattn', 'bagf', 'pug', 'dsff']


# ---------- 以下 5 个 baseline 融合模块与旧版完全一致，不做任何修改 ----------

class ConcatFusion(nn.Module):
    """静态拼接 (2015)"""
    def __init__(self, seg_ch, aux_ch):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv3d(seg_ch + aux_ch, seg_ch, 3, padding=1),
            nn.BatchNorm3d(seg_ch), nn.ReLU(True))

    def forward(self, seg_feat, aux_feat):
        if seg_feat.shape[2:] != aux_feat.shape[2:]:
            aux_feat = F.interpolate(aux_feat, size=seg_feat.shape[2:],
                                     mode='trilinear', align_corners=False)
        return self.fuse(torch.cat([seg_feat, aux_feat], dim=1)) + seg_feat


class CrossAttnFusion(nn.Module):
    """跨注意力融合 (2021): Q=main, KV=aux"""
    def __init__(self, seg_ch, aux_ch, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = seg_ch // num_heads
        self.q = nn.Conv3d(seg_ch, seg_ch, 1)
        self.k = nn.Conv3d(aux_ch, seg_ch, 1)
        self.v = nn.Conv3d(aux_ch, seg_ch, 1)
        self.proj = nn.Sequential(
            nn.Conv3d(seg_ch, seg_ch, 1), nn.BatchNorm3d(seg_ch), nn.ReLU(True))
        self.scale = self.head_dim ** -0.5

    def forward(self, seg_feat, aux_feat):
        if seg_feat.shape[2:] != aux_feat.shape[2:]:
            aux_feat = F.interpolate(aux_feat, size=seg_feat.shape[2:],
                                     mode='trilinear', align_corners=False)
        B, C, D, H, W = seg_feat.shape
        N = D * H * W
        q = self.q(seg_feat).reshape(B, self.num_heads, self.head_dim, N)
        k = self.k(aux_feat).reshape(B, self.num_heads, self.head_dim, N)
        v = self.v(aux_feat).reshape(B, self.num_heads, self.head_dim, N)
        attn = (q.transpose(-1, -2) @ k) * self.scale
        if N > 4096:
            attn = (q * k).sum(dim=2, keepdim=True)
            attn = attn.softmax(dim=-1)
            out = (v * attn).reshape(B, C, D, H, W)
        else:
            attn = attn.softmax(dim=-1)
            out = (v @ attn.transpose(-1, -2)).reshape(B, C, D, H, W)
        return seg_feat + self.proj(out)


class BAGFFusion(nn.Module):
    """双向自适应门控融合 (BAGF, LBMNet 2025)"""
    def __init__(self, seg_ch, aux_ch):
        super().__init__()
        self.adapt = nn.Sequential(
            nn.Conv3d(aux_ch, seg_ch, 1), nn.BatchNorm3d(seg_ch), nn.ReLU(True))
        self.gate_s2a = nn.Sequential(
            nn.Conv3d(seg_ch * 2, seg_ch, 3, padding=1),
            nn.BatchNorm3d(seg_ch), nn.Sigmoid())
        self.gate_a2s = nn.Sequential(
            nn.Conv3d(seg_ch * 2, seg_ch, 3, padding=1),
            nn.BatchNorm3d(seg_ch), nn.Sigmoid())
        self.fuse = nn.Sequential(
            nn.Conv3d(seg_ch * 2, seg_ch, 3, padding=1),
            nn.BatchNorm3d(seg_ch), nn.ReLU(True))

    def forward(self, seg_feat, aux_feat):
        if seg_feat.shape[2:] != aux_feat.shape[2:]:
            aux_feat = F.interpolate(aux_feat, size=seg_feat.shape[2:],
                                     mode='trilinear', align_corners=False)
        aux_t = self.adapt(aux_feat)
        cat = torch.cat([seg_feat, aux_t], dim=1)
        g1 = self.gate_s2a(cat)
        g2 = self.gate_a2s(cat)
        enhanced = torch.cat([seg_feat * g1, aux_t * g2], dim=1)
        return self.fuse(enhanced) + seg_feat


class PUGFusion(nn.Module):
    """Progressive Uncertainty-Guided fusion (PUG-UKAN 2025)"""
    def __init__(self, seg_ch, aux_ch, num_edl_classes=4):
        super().__init__()
        self.adapt = nn.Sequential(
            nn.Conv3d(aux_ch, seg_ch, 1), nn.BatchNorm3d(seg_ch), nn.ReLU(True))
        self.edl_head = nn.Sequential(
            nn.Conv3d(aux_ch, aux_ch // 2, 3, padding=1),
            nn.BatchNorm3d(aux_ch // 2), nn.ReLU(True),
            nn.Conv3d(aux_ch // 2, num_edl_classes, 1))
        self.temperature = nn.Parameter(torch.ones(1))
        self.unc_attn = nn.Sequential(
            nn.Conv3d(seg_ch + 1, seg_ch, 3, padding=1),
            nn.BatchNorm3d(seg_ch), nn.Sigmoid())
        self.fuse = nn.Sequential(
            nn.Conv3d(seg_ch * 2, seg_ch, 3, padding=1),
            nn.BatchNorm3d(seg_ch), nn.ReLU(True))

    def forward(self, seg_feat, aux_feat):
        if seg_feat.shape[2:] != aux_feat.shape[2:]:
            aux_feat = F.interpolate(aux_feat, size=seg_feat.shape[2:],
                                     mode='trilinear', align_corners=False)
        aux_t = self.adapt(aux_feat)
        evidence = F.relu(self.edl_head(aux_feat) / self.temperature.clamp(min=0.1))
        alpha = evidence + 1.0
        unc = evidence.size(1) / alpha.sum(dim=1, keepdim=True)
        attn = self.unc_attn(torch.cat([seg_feat, unc], dim=1))
        guided = aux_t * attn
        return self.fuse(torch.cat([seg_feat, guided], dim=1)) + seg_feat


class DSFFFusion(nn.Module):
    """Deep Supervision Feature Fusion: 多尺度特征加权"""
    def __init__(self, seg_ch, aux_ch):
        super().__init__()
        self.adapt = nn.Sequential(
            nn.Conv3d(aux_ch, seg_ch, 1), nn.BatchNorm3d(seg_ch), nn.ReLU(True))
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool3d(1),
            nn.Conv3d(seg_ch * 2, seg_ch // 4, 1), nn.ReLU(True),
            nn.Conv3d(seg_ch // 4, seg_ch * 2, 1), nn.Sigmoid())
        self.fuse = nn.Sequential(
            nn.Conv3d(seg_ch * 2, seg_ch, 3, padding=1),
            nn.BatchNorm3d(seg_ch), nn.ReLU(True))

    def forward(self, seg_feat, aux_feat):
        if seg_feat.shape[2:] != aux_feat.shape[2:]:
            aux_feat = F.interpolate(aux_feat, size=seg_feat.shape[2:],
                                     mode='trilinear', align_corners=False)
        aux_t = self.adapt(aux_feat)
        cat = torch.cat([seg_feat, aux_t], dim=1)
        w = self.channel_attn(cat)
        return self.fuse(cat * w) + seg_feat


# ---------- 以下是 TCFF 改进后的 FoundationTGF ----------

class FoundationTGF(nn.Module):
    """TCFF-TGF for Foundation Model Features (改进版)

    改动要点 (vs 旧版):
      1. 移除 gate_floor (与气道/双解码器实验统一)
      2. 新增 Pillar-1: 通道级信任适配 (channel_gate)
         — GAP(seg+aux) → MLP → per-channel sigmoid gate → 预筛选 aux 通道
      3. 改 Pillar-2: EDL 从 per-voxel 改为 channel-level (全局)
         — GAP(aux) → Linear EDL head → 全局 uncertainty → broadcast
      4. return_unc 返回 (fused, channel_weight, aux_global) 用于 CTC loss
    """
    def __init__(self, seg_ch, aux_ch, num_edl_classes=4):
        super().__init__()
        self.num_edl_classes = num_edl_classes
        self.seg_ch = seg_ch
        self.aux_ch = aux_ch

        # ── Pillar 1: Channel-level Trust Adaptation ──
        self.channel_gate = nn.Sequential(
            nn.Linear(seg_ch + aux_ch, (seg_ch + aux_ch) // 4),
            nn.ReLU(True),
            nn.Linear((seg_ch + aux_ch) // 4, aux_ch),
            nn.Sigmoid()
        )

        # ── Fusion ──
        self.fuse = nn.Sequential(
            nn.Conv3d(seg_ch + aux_ch, seg_ch, 3, padding=1),
            nn.BatchNorm3d(seg_ch), nn.ReLU(True))

        # ── Pillar 2: Channel-level EDL Trust Estimation ──
        self.edl_head = nn.Sequential(
            nn.Linear(aux_ch, aux_ch // 2),
            nn.ReLU(True),
            nn.Linear(aux_ch // 2, num_edl_classes))
        self.W_u = nn.Sequential(nn.Conv3d(seg_ch, 1, 1), nn.Sigmoid())
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, seg_feat, aux_feat, return_unc=False):
        if seg_feat.shape[2:] != aux_feat.shape[2:]:
            aux_feat = F.interpolate(aux_feat, size=seg_feat.shape[2:],
                                     mode='trilinear', align_corners=False)

        # ---- 全局通道表示 ----
        seg_global = seg_feat.mean(dim=[2, 3, 4])   # [B, C_s]
        aux_global = aux_feat.mean(dim=[2, 3, 4])   # [B, C_a]

        # ---- Pillar 1: 通道信任适配 ----
        channel_weight = self.channel_gate(
            torch.cat([seg_global, aux_global], dim=1))          # [B, C_a]
        aux_filtered = aux_feat * channel_weight[:, :, None, None, None]

        # ---- Fusion ----
        fused = self.fuse(torch.cat([seg_feat, aux_filtered], dim=1)) + seg_feat

        # ---- Pillar 2: 通道级 EDL 不确定性 ----
        edl_logits = self.edl_head(aux_global)                   # [B, K]
        evidence = F.relu(edl_logits / self.temperature.clamp(min=0.1))
        alpha = evidence + 1.0
        S = alpha.sum(dim=1, keepdim=True)                       # [B, 1]
        uncertainty = self.num_edl_classes / S                   # [B, 1]
        uncertainty = uncertainty[:, :, None, None, None]        # [B,1,1,1,1] broadcast

        # ---- 返回 trust_info (训练 CTC loss 用) ----
        if return_unc:
            return fused, channel_weight, aux_global

        # ---- Trust Gate (推理时) ----
        w = self.W_u(fused)                                      # [B,1,D,H,W]
        gate_strength = w * uncertainty                          # broadcast
        return fused * (1 - gate_strength) + seg_feat * gate_strength


def build_fusion(fusion_type, seg_ch, aux_ch):
    """融合模块工厂 — 仅 tgf 分支改动，其余不变"""
    if fusion_type == 'concat':
        return ConcatFusion(seg_ch, aux_ch)
    elif fusion_type == 'crossattn':
        return CrossAttnFusion(seg_ch, aux_ch)
    elif fusion_type == 'bagf':
        return BAGFFusion(seg_ch, aux_ch)
    elif fusion_type == 'pug':
        return PUGFusion(seg_ch, aux_ch)
    elif fusion_type == 'dsff':
        return DSFFFusion(seg_ch, aux_ch)
    elif fusion_type == 'tgf':
        return FoundationTGF(seg_ch, aux_ch)
    else:
        raise ValueError(f"Unknown fusion_type: {fusion_type}. Options: {FUSION_TYPES}")


# ============================================================
# 主模型
# ============================================================

class TopCoWFoundationModel(nn.Module):
    def __init__(self, foundation_encoder, fusion_type='tgf',
                 num_seg_classes=NUM_SEG_CLASSES):
        super().__init__()
        assert fusion_type in FUSION_TYPES, f"Unknown: {fusion_type}. Options: {FUSION_TYPES}"
        self.fusion_type = fusion_type

        self.foundation_encoder = foundation_encoder
        for p in self.foundation_encoder.parameters():
            p.requires_grad = False

        self.projector = MultiScaleProjector(
            in_dim=foundation_encoder.output_dim, out_dims=(128, 64, 32))

        base = SegAirwayModel(in_channels=1, out_channels=2)
        self.encoders = base.encoders
        self.seg_decoders = base.decoders
        self.fusion_module_decoder = base.fusion_module_decoder
        self.dropout2 = base.dropout2
        self.seg_final = nn.Conv3d(6, num_seg_classes, 1)
        self.seg_activation = nn.Softmax(dim=1)

        self.fusion_deep = build_fusion(fusion_type, 128, 128)
        self.fusion_mid = build_fusion(fusion_type, 64, 64)
        self.fusion_shallow = build_fusion(fusion_type, 32, 32)

        self.ds_heads = nn.ModuleList([
            nn.Conv3d(128, num_seg_classes, 1),
            nn.Conv3d(64, num_seg_classes, 1),
            nn.Conv3d(32, num_seg_classes, 1)])

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.parameters())
        print(f"Trainable params: {trainable:,} | Total: {total:,}")

    def set_epoch(self, epoch):
        pass

    def forward(self, x, return_aux=True, aux_only=False, return_unc=False):
        if aux_only:
            return None, {'circuit_logits': None, 'caliber_logits': None,
                          'laterality_logits': None, 'ds_logits': None}, None

        with torch.no_grad():
            found_feat = self.foundation_encoder(x)

        enc_feats = []
        x_enc = x
        for encoder in self.encoders:
            x_enc = encoder(x_enc)
            enc_feats.append(x_enc)
        enc_rev = list(reversed(enc_feats))
        bottleneck, skips = enc_rev[0], enc_rev[1:]

        target_shapes = [s.shape[2:] for s in skips]
        found_multi = self.projector(found_feat, target_shapes)

        # ---- trust_info 收集 (仅 tgf + return_unc) ----
        trust_info = []

        # ---- 浅层融合 (encoder skip) ----
        if return_unc and self.fusion_type == 'tgf':
            modified_skip2, cw_s, ag_s = self.fusion_shallow(
                skips[2], found_multi[2], return_unc=True)
            trust_info.append((cw_s, ag_s))
        else:
            modified_skip2 = self.fusion_shallow(skips[2], found_multi[2])

        # ---- 主 Decoder ----
        main_x = bottleneck
        main_feats = []

        for i, seg_dec in enumerate(self.seg_decoders):
            main_x = seg_dec(modified_skip2 if i == 2 else skips[i], main_x)

            if i == 0:
                if return_unc and self.fusion_type == 'tgf':
                    main_x, cw_d, ag_d = self.fusion_deep(
                        main_x, found_multi[0], return_unc=True)
                    trust_info.append((cw_d, ag_d))
                else:
                    main_x = self.fusion_deep(main_x, found_multi[0])
            elif i == 1:
                if return_unc and self.fusion_type == 'tgf':
                    main_x, cw_m, ag_m = self.fusion_mid(
                        main_x, found_multi[1], return_unc=True)
                    trust_info.append((cw_m, ag_m))
                else:
                    main_x = self.fusion_mid(main_x, found_multi[1])

            main_feats.append(main_x)

        dec_fusion = self.dropout2(self.fusion_module_decoder(main_feats))
        seg_logits = self.seg_final(dec_fusion)
        seg = self.seg_activation(seg_logits)
        ds_logits = [h(f) for f, h in zip(main_feats, self.ds_heads)]

        aux_output = {'circuit_logits': None, 'caliber_logits': None,
                      'laterality_logits': None, 'ds_logits': ds_logits}

        if return_unc and self.fusion_type == 'tgf':
            return seg, aux_output, seg_logits, trust_info

        return seg, aux_output, seg_logits


# ============================================================
# 测试
# ============================================================

if __name__ == "__main__":
    x = torch.randn(1, 1, 64, 160, 160)
    for ft in ['mock']:
        for fuse in ['tgf', 'concat']:
            print(f"\n--- {ft} + {fuse} ---")
            enc = build_foundation_encoder(ft)
            m = TopCoWFoundationModel(enc, fusion_type=fuse)
            m.eval()
            with torch.no_grad():
                seg, aux, logits = m(x)
            print(f"  Seg: {seg.shape}")

        # 测试 return_unc
        print(f"\n--- {ft} + tgf (return_unc) ---")
        enc = build_foundation_encoder(ft)
        m = TopCoWFoundationModel(enc, fusion_type='tgf')
        m.train()
        seg, aux, logits, ti = m(x, return_unc=True)
        print(f"  Seg: {seg.shape}, trust_info levels: {len(ti)}")
        for j, (cw, ag) in enumerate(ti):
            print(f"    level {j}: cw={cw.shape}, ag={ag.shape}")

    print("\nAll passed!")