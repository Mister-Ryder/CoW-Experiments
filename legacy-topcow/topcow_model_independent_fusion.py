"""
topcow_model_independent_fusion.py (TCFF)
=========================================
将独立训练的分类器特征作为辅助源, 与UNet分割网络融合

支持融合方式:
  - tgf:     TCFF 完整版 (GAAM/SAAM/LAAM 条件路由 + EDL 不确定性门控)
  - concat:  静态拼接 (baseline)
  - crossattn / bagf / dsff: 其他 baseline (走 else 分支)

TCFF 改动 (vs 旧版):
  - 移除 SimpleTGFGate (无条件路由, 有 gate_floor → 性能差)
  - TGF 模式改用 GAAM/SAAM/LAAM (与双解码器实验完全一致的融合算子)
  - 利用冻结分类器已有的 circuit/caliber/laterality 预测做条件路由
  - 利用冻结分类器 logits 做 EDL 不确定性估计
  - 非 TGF 分支代码完全不变
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from func.model_arch_model10 import SegAirwayModel
from topcow_anatomical_classifier import TopCoWAnatomicalClassifier
from fusion_baselines import ConcatFusion, create_fusion_module

# 从双解码器模型导入已验证的 TGF 融合算子
from topcow_model import (
    GAAMFusion, SAAMFusion, LAAMFusion,
    LATERALITY_LUT, NUM_CIRCUIT, NUM_CALIBER, NUM_LATERALITY
)


class TopCoWIndependentClassifierFusionModel(nn.Module):
    def __init__(self, classifier, fusion_type='tgf', num_seg_classes=14,
                 disable_gating=False):
        super().__init__()
        self.classifier = classifier
        for param in self.classifier.parameters():
            param.requires_grad = False  # 冻结分类器

        self.fusion_type = fusion_type
        self.disable_gating = disable_gating  # True = 只保留条件路由, 关掉不确定性门控

        # V4 UNet (主网络)
        base = SegAirwayModel(in_channels=1, out_channels=2)
        self.encoders = base.encoders
        self.seg_decoders = base.decoders
        self.fusion_module_decoder = base.fusion_module_decoder
        self.dropout2 = base.dropout2
        self.seg_final = nn.Conv3d(6, num_seg_classes, 1)
        self.seg_activation = nn.Softmax(dim=1)

        # ---- 融合模块: TGF vs 其他 ----
        if fusion_type == 'tgf':
            # TCFF: 使用与双解码器完全一致的 GAAM/SAAM/LAAM
            self.gaam = GAAMFusion(128, 128, num_conditions=3)    # 深层 (circuit)
            self.saam = SAAMFusion(64, 64, num_conditions=3)      # 中层 (caliber)
            self.laam = LAAMFusion(32, 32, num_regions=3)         # 浅层 (laterality)

            # 可学习 temperature (EDL 校准, 与双解码器一致)
            self.temperature = nn.Parameter(torch.ones(1))

            # Laterality LUT (条件路由用)
            self.register_buffer('lat_lut',
                                 torch.tensor(LATERALITY_LUT, dtype=torch.long))
        else:
            # 非 TGF: 保持原逻辑不变
            self.adapters = nn.ModuleList([
                nn.Conv3d(128, 128, 1),   # deep -> level0
                nn.Conv3d(64, 64, 1),     # mid -> level1
                nn.Conv3d(32, 32, 1),     # shallow -> level2
            ])
            self.fusions = nn.ModuleList([
                create_fusion_module(fusion_type, 128, 128) if fusion_type != 'concat'
                    else ConcatFusion(128, 128),
                create_fusion_module(fusion_type, 64, 64) if fusion_type != 'concat'
                    else ConcatFusion(64, 64),
                create_fusion_module(fusion_type, 32, 32) if fusion_type != 'concat'
                    else ConcatFusion(32, 32),
            ])

        # 深监督头
        self.ds_heads = nn.ModuleList([
            nn.Conv3d(128, num_seg_classes, 1),
            nn.Conv3d(64, num_seg_classes, 1),
            nn.Conv3d(32, num_seg_classes, 1),
        ])

    def _edl(self, logits, num_classes):
        """EDL with learnable temperature (与双解码器 _edl 一致)"""
        evidence = F.relu(logits / self.temperature.clamp(min=0.1))
        alpha = evidence + 1.0
        S = alpha.sum(dim=1, keepdim=True)
        uncertainty = num_classes / S
        pred = torch.argmax(logits, dim=1)
        return pred, uncertainty

    def forward(self, x):
        # ===== 1. 冻结分类器前向 =====
        with torch.no_grad():
            cls_out = self.classifier(x)
            cls_feats = cls_out['multi_scale_features']  # [deep(128), mid(64), shallow(32)]

        # ===== 2. 编码器 =====
        enc_feats = []
        x_enc = x
        for encoder in self.encoders:
            x_enc = encoder(x_enc)
            enc_feats.append(x_enc)
        enc_rev = list(reversed(enc_feats))
        bottleneck = enc_rev[0]   # 256ch
        skips = enc_rev[1:]       # [128, 64, 32]

        # ===== 3. 融合 + 解码 =====
        if self.fusion_type == 'tgf':
            return self._forward_tgf(cls_out, cls_feats, bottleneck, skips)
        else:
            return self._forward_baseline(cls_feats, bottleneck, skips)

    def _forward_tgf(self, cls_out, cls_feats, bottleneck, skips):
        """TGF 路径: GAAM/SAAM/LAAM + 条件路由 + EDL 不确定性门控"""

        # ---- 从冻结分类器获取预测和不确定性 ----
        with torch.no_grad():
            circuit_pred, circuit_unc = self._edl(
                cls_out['circuit_logits'], NUM_CIRCUIT)
            caliber_pred, caliber_unc = self._edl(
                cls_out['caliber_logits'], NUM_CALIBER)
            lateral_pred, lateral_unc = self._edl(
                cls_out['laterality_logits'], NUM_LATERALITY)

        # nogate 消融: 保留条件路由, 关掉不确定性门控
        if self.disable_gating:
            circuit_unc = None
            caliber_unc = None
            lateral_unc = None

        # ---- LAAM: 修改 encoder skip[2] (浅层) ----
        lat_route = self.lat_lut[lateral_pred.clamp(0, len(self.lat_lut) - 1)]
        modified_skip2 = self.laam(skips[2], cls_feats[2],
                                   prediction=lat_route,
                                   uncertainty=lateral_unc)

        # ---- 主 Decoder + GAAM/SAAM 融合 ----
        main_x = bottleneck
        main_feats = []

        for i, seg_dec in enumerate(self.seg_decoders):
            if i == 2:
                main_x = seg_dec(modified_skip2, main_x)
            else:
                main_x = seg_dec(skips[i], main_x)

            if i == 0:
                # GAAM: 深层 (circuit-conditioned)
                main_x = self.gaam(main_x, cls_feats[0],
                                   prediction=circuit_pred,
                                   uncertainty=circuit_unc)
            elif i == 1:
                # SAAM: 中层 (caliber-conditioned)
                main_x = self.saam(main_x, cls_feats[1],
                                   prediction=caliber_pred,
                                   uncertainty=caliber_unc)

            main_feats.append(main_x)

        # ---- 输出 ----
        dec_fusion = self.dropout2(self.fusion_module_decoder(main_feats))
        seg_logits = self.seg_final(dec_fusion)
        seg = self.seg_activation(seg_logits)

        ds_logits = [head(feat) for feat, head in zip(main_feats, self.ds_heads)]
        aux_output = {
            'circuit_logits': cls_out['circuit_logits'],
            'caliber_logits': cls_out['caliber_logits'],
            'laterality_logits': cls_out['laterality_logits'],
            'ds_logits': ds_logits,
        }
        return seg, aux_output, seg_logits

    def _forward_baseline(self, cls_feats, bottleneck, skips):
        """非 TGF 路径: 与旧版完全一致"""

        # 浅层 (i=2)
        shallow_adapted = F.interpolate(
            self.adapters[2](cls_feats[2]), size=skips[2].shape[2:],
            mode='trilinear', align_corners=False)
        modified_skip2 = self.fusions[2](skips[2], shallow_adapted)

        # 解码器
        main_x = bottleneck
        main_feats = []
        for i, seg_dec in enumerate(self.seg_decoders):
            if i == 2:
                main_x = seg_dec(modified_skip2, main_x)
            else:
                main_x = seg_dec(skips[i], main_x)
                adapted_cls_feat = F.interpolate(
                    self.adapters[i](cls_feats[i]), size=main_x.shape[2:],
                    mode='trilinear', align_corners=False)
                main_x = self.fusions[i](main_x, adapted_cls_feat)
            main_feats.append(main_x)

        # 输出
        dec_fusion = self.dropout2(self.fusion_module_decoder(main_feats))
        seg_logits = self.seg_final(dec_fusion)
        seg = self.seg_activation(seg_logits)

        ds_logits = [head(feat) for feat, head in zip(main_feats, self.ds_heads)]
        aux_output = {
            'circuit_logits': None, 'caliber_logits': None,
            'laterality_logits': None, 'ds_logits': ds_logits,
        }
        return seg, aux_output, seg_logits


if __name__ == "__main__":
    x = torch.randn(1, 1, 64, 160, 160)
    classifier = TopCoWAnatomicalClassifier()

    for ft in ['tgf', 'concat']:
        print(f"\n--- fusion_type={ft} ---")
        model = TopCoWIndependentClassifierFusionModel(
            classifier, fusion_type=ft)
        model.eval()
        with torch.no_grad():
            seg, aux, logits = model(x)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Seg: {seg.shape}, trainable: {trainable:,}")
        if ft == 'tgf':
            print(f"  Circuit logits: {aux['circuit_logits'].shape}")
            print(f"  Temperature: {model.temperature.item():.4f}")

    print("\nAll passed!")