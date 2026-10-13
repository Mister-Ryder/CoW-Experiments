"""
topcow_anatomical_classifier.py
================================
用于TopCoW的独立血管解剖分类器

输出:
  - circuit_logits: [B, 4, D, H, W]
  - caliber_logits: [B, 4, D, H, W]
  - laterality_logits: [B, 4, D, H, W]
  - multi_scale_features: 三个尺度的特征图, 分别对应分类器的深层、中层、浅层输出
    (通道数分别为 [128, 64, 32], 与UNet解码器对齐)

训练后冻结, 作为独立辅助源送入分割网络进行融合实验。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CIRCUIT = 4
NUM_CALIBER = 4
NUM_LATERALITY = 4


class LightEncoderBlock(nn.Module):
    """轻量编码器块"""
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, stride=stride, padding=1),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch), nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


class TopCoWAnatomicalClassifier(nn.Module):
    """
    独立血管解剖分类器, 不共享分割网络的编码器
    """
    def __init__(self, in_channels=1):
        super().__init__()
        # 独立编码器 (与V4主编码器结构不同, 保证特征异源)
        self.enc1 = LightEncoderBlock(in_channels, 32)
        self.down1 = nn.MaxPool3d(2)
        self.enc2 = LightEncoderBlock(32, 64)
        self.down2 = nn.MaxPool3d(2)
        self.enc3 = LightEncoderBlock(64, 128)
        self.down3 = nn.MaxPool3d(2)
        self.enc4 = LightEncoderBlock(128, 256)

        # 特征投影适配器 (将编码器特征投影到与 UNet decoder 对齐的通道)
        self.proj_deep = nn.Conv3d(256, 128, 1)   # 深层 -> level0 (128ch)
        self.proj_mid  = nn.Conv3d(128, 64, 1)   # 中层 -> level1 (64ch)
        self.proj_shallow = nn.Conv3d(64, 32, 1)  # 浅层 -> level2 (32ch)

        # 分类头 (用于预训练)
        self.circuit_head = nn.Conv3d(128, NUM_CIRCUIT, 1)
        self.caliber_head = nn.Conv3d(64, NUM_CALIBER, 1)
        self.laterality_head = nn.Conv3d(32, NUM_LATERALITY, 1)

    def forward(self, x):
        # 编码器前向
        e1 = self.enc1(x)       # 32ch, 原始分辨率
        e2 = self.enc2(self.down1(e1))  # 64ch, 1/2
        e3 = self.enc3(self.down2(e2))  # 128ch, 1/4
        e4 = self.enc4(self.down3(e3))  # 256ch, 1/8

        # 投影到融合通道
        feat_deep = self.proj_deep(e4)      # 128ch, 1/8
        feat_mid  = self.proj_mid(e3)       # 64ch, 1/4
        feat_shallow = self.proj_shallow(e2) # 32ch, 1/2

        # 分类logits (上采样到输入分辨率)
        circuit_logits = self.circuit_head(feat_deep)
        caliber_logits = self.caliber_head(feat_mid)
        laterality_logits = self.laterality_head(feat_shallow)

        return {
            'circuit_logits': circuit_logits,
            'caliber_logits': caliber_logits,
            'laterality_logits': laterality_logits,
            'multi_scale_features': [feat_deep, feat_mid, feat_shallow],
        }


if __name__ == "__main__":
    x = torch.randn(2, 1, 64, 160, 160)
    model = TopCoWAnatomicalClassifier()
    out = model(x)
    for k, v in out.items():
        if isinstance(v, list):
            print(f"{k}: {[t.shape for t in v]}")
        else:
            print(f"{k}: {v.shape}")