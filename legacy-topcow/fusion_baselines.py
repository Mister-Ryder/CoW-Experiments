"""
fusion_baselines.py
===================
5种融合baseline: Concat / CrossAttn / BAGF(2025) / PUG(2025) / DSFF(NeurIPS2024)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatFusion(nn.Module):
    """静态拼接融合 (2015+)"""
    def __init__(self, seg_ch, aux_ch):
        super().__init__()
        self.transform = nn.Sequential(
            nn.Conv3d(aux_ch, seg_ch, 1), nn.BatchNorm3d(seg_ch), nn.ReLU(inplace=True))
        self.fuse = nn.Sequential(
            nn.Conv3d(seg_ch * 2, seg_ch, 3, padding=1), nn.BatchNorm3d(seg_ch), nn.ReLU(inplace=True),
            nn.Conv3d(seg_ch, seg_ch, 3, padding=1), nn.BatchNorm3d(seg_ch), nn.ReLU(inplace=True))

    def forward(self, seg_feat, aux_feat, **kwargs):
        if seg_feat.shape[2:] != aux_feat.shape[2:]:
            aux_feat = F.interpolate(aux_feat, size=seg_feat.shape[2:], mode='trilinear', align_corners=False)
        return self.fuse(torch.cat([seg_feat, self.transform(aux_feat)], dim=1)) + seg_feat


class CrossAttentionFusion(nn.Module):
    """跨注意力融合 (2021+): Q=main, K=V=aux"""
    def __init__(self, seg_ch, aux_ch, num_heads=4, pool_size=8):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = seg_ch // num_heads
        self.pool_size = pool_size
        self.transform_aux = nn.Conv3d(aux_ch, seg_ch, 1)
        self.q_proj = nn.Conv3d(seg_ch, seg_ch, 1)
        self.k_proj = nn.Conv3d(seg_ch, seg_ch, 1)
        self.v_proj = nn.Conv3d(seg_ch, seg_ch, 1)
        self.out_proj = nn.Sequential(nn.Conv3d(seg_ch, seg_ch, 1), nn.BatchNorm3d(seg_ch), nn.ReLU(inplace=True))

    def forward(self, seg_feat, aux_feat, **kwargs):
        B, C, D, H, W = seg_feat.shape
        if seg_feat.shape[2:] != aux_feat.shape[2:]:
            aux_feat = F.interpolate(aux_feat, size=(D, H, W), mode='trilinear', align_corners=False)
        aux_feat = self.transform_aux(aux_feat)
        ps = self.pool_size
        seg_pool = F.adaptive_avg_pool3d(seg_feat, (min(D, ps), min(H, ps), min(W, ps)))
        aux_pool = F.adaptive_avg_pool3d(aux_feat, (min(D, ps), min(H, ps), min(W, ps)))
        Q = self.q_proj(seg_pool)
        K = self.k_proj(aux_pool)
        V = self.v_proj(aux_pool)
        d, h, w = Q.shape[2], Q.shape[3], Q.shape[4]
        N = d * h * w
        Q = Q.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        K = K.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        V = V.view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)
        attn = F.softmax(torch.matmul(Q, K.transpose(-2, -1)) * (self.head_dim ** -0.5), dim=-1)
        out = torch.matmul(attn, V).permute(0, 1, 3, 2).contiguous().view(B, C, d, h, w)
        if out.shape[2:] != seg_feat.shape[2:]:
            out = F.interpolate(out, size=(D, H, W), mode='trilinear', align_corners=False)
        return self.out_proj(out) + seg_feat


class BAGFFusion(nn.Module):
    """双向自适应门控融合 (LBMNet, 2025)"""
    def __init__(self, seg_ch, aux_ch):
        super().__init__()
        self.transform_aux = nn.Sequential(
            nn.Conv3d(aux_ch, seg_ch, 1), nn.BatchNorm3d(seg_ch), nn.ReLU(inplace=True))
        self.gate_main = nn.Sequential(nn.Conv3d(seg_ch, seg_ch, 3, padding=1), nn.BatchNorm3d(seg_ch), nn.Sigmoid())
        self.gate_aux = nn.Sequential(nn.Conv3d(seg_ch, seg_ch, 3, padding=1), nn.BatchNorm3d(seg_ch), nn.Sigmoid())
        self.refine = nn.Sequential(nn.Conv3d(seg_ch, seg_ch, 3, padding=1), nn.BatchNorm3d(seg_ch), nn.ReLU(inplace=True))

    def forward(self, seg_feat, aux_feat, **kwargs):
        if seg_feat.shape[2:] != aux_feat.shape[2:]:
            aux_feat = F.interpolate(aux_feat, size=seg_feat.shape[2:], mode='trilinear', align_corners=False)
        aux_t = self.transform_aux(aux_feat)
        fused = seg_feat * self.gate_aux(aux_t) + aux_t * self.gate_main(seg_feat)
        return self.refine(fused) + seg_feat


class PUGFusion(nn.Module):
    """证据不确定性引导注意力融合 (arXiv:2510.08949, 2025)
    核心区别于TGF: 高uncertainty→增强融合关注度 (vs TGF的高uncertainty→减少融合)"""
    def __init__(self, seg_ch, aux_ch):
        super().__init__()
        self.transform_aux = nn.Sequential(
            nn.Conv3d(aux_ch, seg_ch, 1), nn.BatchNorm3d(seg_ch), nn.ReLU(inplace=True))
        self.unc_proj = nn.Sequential(
            nn.Conv3d(1, seg_ch, 1), nn.ReLU(inplace=True), nn.Conv3d(seg_ch, seg_ch, 1), nn.Sigmoid())
        self.unc_scale = nn.Parameter(torch.ones(1))
        self.refine = nn.Sequential(
            nn.Conv3d(seg_ch * 2, seg_ch, 3, padding=1), nn.BatchNorm3d(seg_ch), nn.ReLU(inplace=True))
        self.fallback = nn.Sequential(
            nn.Conv3d(seg_ch * 2, seg_ch, 3, padding=1), nn.BatchNorm3d(seg_ch), nn.ReLU(inplace=True))

    def forward(self, seg_feat, aux_feat, uncertainty=None, **kwargs):
        if seg_feat.shape[2:] != aux_feat.shape[2:]:
            aux_feat = F.interpolate(aux_feat, size=seg_feat.shape[2:], mode='trilinear', align_corners=False)
        aux_t = self.transform_aux(aux_feat)
        if uncertainty is not None:
            if uncertainty.shape[2:] != seg_feat.shape[2:]:
                uncertainty = F.interpolate(uncertainty, size=seg_feat.shape[2:], mode='trilinear', align_corners=False)
            attn = self.unc_proj(uncertainty * self.unc_scale)
            out = self.refine(torch.cat([seg_feat, aux_t * attn], dim=1))
        else:
            out = self.fallback(torch.cat([seg_feat, aux_t], dim=1))
        return out + seg_feat


class DSFFFusion(nn.Module):
    """动态稀疏特征融合 (E2ENet, NeurIPS 2024)
    学习通道级稀疏门控 + 空间门控，自适应选择融合通道"""
    def __init__(self, seg_ch, aux_ch, density=0.5):
        super().__init__()
        self.density = density
        self.seg_ch = seg_ch
        self.transform_aux = nn.Sequential(
            nn.Conv3d(aux_ch, seg_ch, 1), nn.BatchNorm3d(seg_ch), nn.ReLU(inplace=True))
        self.channel_scores = nn.Parameter(torch.randn(1, seg_ch, 1, 1, 1) * 0.01)
        self.spatial_gate = nn.Sequential(
            nn.Conv3d(seg_ch * 2, seg_ch, 1), nn.BatchNorm3d(seg_ch), nn.Sigmoid())
        self.refine = nn.Sequential(
            nn.Conv3d(seg_ch, seg_ch, 3, padding=1), nn.BatchNorm3d(seg_ch), nn.ReLU(inplace=True))

    def _soft_topk_mask(self, scores):
        k = max(1, int(self.seg_ch * self.density))
        threshold = torch.topk(scores.view(-1), k, largest=True)[0][-1]
        return torch.sigmoid((scores - threshold) * 10.0)

    def forward(self, seg_feat, aux_feat, **kwargs):
        if seg_feat.shape[2:] != aux_feat.shape[2:]:
            aux_feat = F.interpolate(aux_feat, size=seg_feat.shape[2:], mode='trilinear', align_corners=False)
        aux_t = self.transform_aux(aux_feat)
        sparse_mask = self._soft_topk_mask(torch.sigmoid(self.channel_scores))
        spatial_w = self.spatial_gate(torch.cat([seg_feat, aux_t], dim=1))
        return self.refine(seg_feat + aux_t * sparse_mask * spatial_w) + seg_feat


def create_fusion_module(name, seg_ch, aux_ch):
    name = name.lower()
    modules = {
        'concat': ConcatFusion, 'crossattn': CrossAttentionFusion,
        'bagf': BAGFFusion, 'pug': PUGFusion, 'dsff': DSFFFusion,
    }
    if name not in modules:
        raise ValueError(f"Unknown: {name}. Options: {list(modules.keys())}")
    return modules[name](seg_ch, aux_ch)


if __name__ == "__main__":
    B, D, H, W, C = 1, 16, 40, 40, 64
    for name in ['concat', 'crossattn', 'bagf', 'pug', 'dsff']:
        m = create_fusion_module(name, C, C)
        out = m(torch.randn(B, C, D, H, W), torch.randn(B, C, D, H, W), uncertainty=torch.rand(B, 1, D, H, W))
        print(f"{name:>10}: out={out.shape}, params={sum(p.numel() for p in m.parameters()):,}")