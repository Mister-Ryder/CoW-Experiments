# AirwayAnatomicalClassifier.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size,
                              padding=padding, stride=stride)
        self.bn = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block for channel attention"""

    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class AirwayAnatomicalClassifier(nn.Module):
    def __init__(self, in_channels=1,
                 generation_classes=7,  # 气道代数类别数
                 structure_classes=7,  # 结构类型类别数
                 location_classes=7):  # 空间位置类别数
        super(AirwayAnatomicalClassifier, self).__init__()

        # 编码器部分 - 共享特征提取
        self.encoder = nn.ModuleList([
            ConvBlock(in_channels, 32),  # L1: 32x128x128x128
            ConvBlock(32, 32, stride=2),  # L1: 32x64x64x64
            ConvBlock(32, 64),  # L2: 64x64x64x64
            ConvBlock(64, 64, stride=2),  # L2: 64x32x32x32
            ConvBlock(64, 128),  # L3: 128x32x32x32
            ConvBlock(128, 128, stride=2),  # L3: 128x16x16x16
            ConvBlock(128, 256),  # L4: 256x16x16x16
            ConvBlock(256, 256, stride=2)  # L4: 256x8x8x8
        ])

        # 添加通道注意力机制
        self.attention_blocks = nn.ModuleList([
            SEBlock(64, reduction=8),
            SEBlock(128, reduction=8),
            SEBlock(256, reduction=8)
        ])

        # 多尺度特征聚合
        self.multi_scale_fusion = nn.Sequential(
            nn.Conv3d(256 + 128 + 64, 256, kernel_size=1, padding=0),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True)
        )

        # 全局特征提取
        self.global_avg_pool = nn.AdaptiveAvgPool3d(1)
        self.global_max_pool = nn.AdaptiveMaxPool3d(1)

        # 1. 气道代数分类头
        self.generation_head = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, generation_classes)
        )

        # 2. 结构类型分类头
        self.structure_head = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, structure_classes)
        )

        # 3. 空间位置分类头
        self.location_head = nn.Sequential(
            nn.Linear(256 * 2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, location_classes)
        )

        # 4. 解剖学引导图生成头 (低分辨率特征图)
        self.guidance_head = nn.Sequential(
            ConvBlock(256, 128, kernel_size=1, padding=0),
            nn.Conv3d(128, 16, kernel_size=1, padding=0)  # 16通道的低维解剖学特征图
        )

    def forward(self, x):
        # 提取所有编码器特征
        features = []
        for i, encoder_block in enumerate(self.encoder):
            x = encoder_block(x)
            if i in [3, 5, 7]:  # 保存关键特征层
                features.append(x)

        # 应用注意力机制
        features[0] = self.attention_blocks[0](features[0])  # 64x32x32x32
        features[1] = self.attention_blocks[1](features[1])  # 128x16x16x16
        features[2] = self.attention_blocks[2](features[2])  # 256x8x8x8

        # 多尺度特征融合
        # 上采样低层特征，以便与最低分辨率特征结合
        feat_size = features[2].shape[2:]
        upsampled_feat1 = F.interpolate(features[0], size=feat_size, mode='trilinear', align_corners=False)
        upsampled_feat2 = F.interpolate(features[1], size=feat_size, mode='trilinear', align_corners=False)

        # 合并特征
        multi_scale_feat = torch.cat([features[2], upsampled_feat2, upsampled_feat1], dim=1)
        fused_feat = self.multi_scale_fusion(multi_scale_feat)

        # 创建解剖学引导图
        guidance_map = self.guidance_head(fused_feat)

        # 全局池化 (avg+max双路径)
        avg_feat = self.global_avg_pool(fused_feat).view(fused_feat.size(0), -1)
        max_feat = self.global_max_pool(fused_feat).view(fused_feat.size(0), -1)
        global_feat = torch.cat([avg_feat, max_feat], dim=1)

        # 各个分类头的输出
        generation_logits = self.generation_head(global_feat)
        structure_logits = self.structure_head(global_feat)
        location_logits = self.location_head(global_feat)

        features_out=features
        del features

        # 返回包含各种特征和预测的字典
        return {
            'generation': generation_logits,
            'structure': structure_logits,
            'location': location_logits,
            'guidance_map': guidance_map,
            'encoder_features': features_out,  # 编码器不同层级的特征
            # 'fused_feature': fused_feat,  # 融合的全局特征
            # 'global_feature': global_feat  # 池化后的全局特征向量
        }


# 测试代码
if __name__ == "__main__":
    # 创建一个假的输入张量
    batch_size = 2
    input_tensor = torch.randn(batch_size, 1, 128, 128, 128)

    # 初始化模型
    model = AirwayAnatomicalClassifier()

    # 前向传播
    output = model(input_tensor)

    # 打印输出形状
    print("Generation logits shape:", output['generation'].shape)
    print("Structure logits shape:", output['structure'].shape)
    print("Location logits shape:", output['location'].shape)
    print("Guidance map shape:", output['guidance_map'].shape)

    for i, feat in enumerate(output['encoder_features']):
        print(f"Encoder feature {i} shape:", feat.shape)

    print("Fused feature shape:", output['fused_feature'].shape)
    print("Global feature shape:", output['global_feature'].shape)