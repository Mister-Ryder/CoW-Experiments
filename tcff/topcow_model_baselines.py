"""
topcow_model_baselines.py
=========================
同架构 + 辅助 decoder，用 baseline 融合算子替换 TGF 的 GAAM/SAAM/LAAM
支持: concat, crossattn, bagf, pug, dsff
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from func.model_arch_model10 import SegAirwayModel
from fusion_baselines import create_fusion_module

NUM_SEG_CLASSES = 14
NUM_CIRCUIT = 4
NUM_CALIBER = 4
NUM_LATERALITY = 4


class LightDecoderBlock(nn.Module):
    def __init__(self, in_ch, skip_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, in_ch, 4, stride=2, padding=1)
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch + skip_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True))

    def forward(self, skip, x):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        return self.conv(torch.cat([skip, x], dim=1))


class TopCoWBaselineFusionModel(nn.Module):
    def __init__(self, fusion_type='concat', num_seg_classes=NUM_SEG_CLASSES):
        super().__init__()
        self.fusion_type = fusion_type
        self.num_seg_classes = num_seg_classes

        base = SegAirwayModel(in_channels=1, out_channels=2)
        self.encoders = base.encoders
        self.seg_decoders = base.decoders
        self.fusion_module_decoder = base.fusion_module_decoder
        self.dropout2 = base.dropout2
        self.seg_final = nn.Conv3d(6, num_seg_classes, 1)
        self.seg_activation = nn.Softmax(dim=1)

        self.aux_decoders = nn.ModuleList([
            LightDecoderBlock(256, 128, 128),
            LightDecoderBlock(128, 64, 64),
            LightDecoderBlock(64, 32, 32)])

        self.circuit_head = nn.Conv3d(128, NUM_CIRCUIT, 1)
        self.caliber_head = nn.Conv3d(64, NUM_CALIBER, 1)
        self.laterality_head = nn.Conv3d(32, NUM_LATERALITY, 1)

        self.fusion_deep = create_fusion_module(fusion_type, 128, 128)
        self.fusion_mid = create_fusion_module(fusion_type, 64, 64)
        self.fusion_shallow = create_fusion_module(fusion_type, 32, 32)

        self.ds_heads = nn.ModuleList([
            nn.Conv3d(128, num_seg_classes, 1),
            nn.Conv3d(64, num_seg_classes, 1),
            nn.Conv3d(32, num_seg_classes, 1)])

        self.temperature = nn.Parameter(torch.ones(1))

    def _edl_uncertainty(self, logits, num_classes):
        evidence = F.relu(logits / self.temperature.clamp(min=0.1))
        alpha = evidence + 1.0
        S = alpha.sum(dim=1, keepdim=True)
        return num_classes / S

    def forward(self, x, return_aux=True, aux_only=False):
        enc_feats = []
        for encoder in self.encoders:
            x = encoder(x)
            enc_feats.append(x)
        enc_rev = list(reversed(enc_feats))
        bottleneck = enc_rev[0]
        skips = enc_rev[1:]

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
            'ds_logits': None}

        if aux_only:
            return None, aux_output, None

        # PUG 需要 uncertainty
        unc_kw = [{}, {}, {}]
        if self.fusion_type == 'pug':
            with torch.no_grad():
                unc_kw[0] = {'uncertainty': self._edl_uncertainty(circuit_logits, NUM_CIRCUIT)}
                unc_kw[1] = {'uncertainty': self._edl_uncertainty(caliber_logits, NUM_CALIBER)}
                unc_kw[2] = {'uncertainty': self._edl_uncertainty(laterality_logits, NUM_LATERALITY)}

        # 浅层融合 (encoder skip)
        modified_skip2 = self.fusion_shallow(skips[2], aux_feats[2], **unc_kw[2])

        # 主 Decoder
        main_x = bottleneck
        main_feats = []
        for i, seg_dec in enumerate(self.seg_decoders):
            if i == 2:
                main_x = seg_dec(modified_skip2, main_x)
            else:
                main_x = seg_dec(skips[i], main_x)
            if i == 0:
                main_x = self.fusion_deep(main_x, aux_feats[0], **unc_kw[0])
            elif i == 1:
                main_x = self.fusion_mid(main_x, aux_feats[1], **unc_kw[1])
            main_feats.append(main_x)

        dec_fusion = self.dropout2(self.fusion_module_decoder(main_feats))
        seg_logits = self.seg_final(dec_fusion)
        seg = self.seg_activation(seg_logits)
        ds_logits = [head(feat) for feat, head in zip(main_feats, self.ds_heads)]
        aux_output['ds_logits'] = ds_logits
        return seg, aux_output, seg_logits


if __name__ == "__main__":
    x = torch.randn(1, 1, 64, 160, 160)
    for ft in ['concat', 'crossattn', 'bagf', 'pug', 'dsff']:
        m = TopCoWBaselineFusionModel(fusion_type=ft)
        m.eval()
        with torch.no_grad():
            seg, aux, logits = m(x)
        print(f"{ft:>10}: seg={seg.shape}, params={sum(p.numel() for p in m.parameters()):,}")