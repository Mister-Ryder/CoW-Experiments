# AnatomicalFusionModules.py - 精简版
import torch
import torch.nn as nn
import torch.nn.functional as F


class AnatomicalGuidanceModule(nn.Module):
    """解剖学引导融合模块基类"""

    def __init__(self, seg_channels, cls_channels):
        super(AnatomicalGuidanceModule, self).__init__()
        self.seg_channels = seg_channels
        self.cls_channels = cls_channels

        # 特征转换
        self.transform = nn.Sequential(
            nn.Conv3d(cls_channels, seg_channels, kernel_size=1),
            nn.BatchNorm3d(seg_channels),
            nn.ReLU(inplace=True)
        )

        # 特征融合
        self.fusion = nn.Sequential(
            nn.Conv3d(seg_channels * 2, seg_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(seg_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, seg_feat, cls_feat):
        # 确保空间维度匹配
        if seg_feat.shape[2:] != cls_feat.shape[2:]:
            cls_feat = F.interpolate(cls_feat, size=seg_feat.shape[2:],
                                     mode='trilinear', align_corners=False)

        # 变换分类特征
        transformed_cls_feat = self.transform(cls_feat)

        # 拼接并融合
        combined = torch.cat([seg_feat, transformed_cls_feat], dim=1)
        output = self.fusion(combined)

        del cls_feat
        # 残差连接
        return output + seg_feat


class GenerationGuidanceModule(AnatomicalGuidanceModule):
    """代数特征引导模块 - 适用于深层解码器"""

    def __init__(self, seg_channels, cls_channels):
        super(GenerationGuidanceModule, self).__init__(seg_channels, cls_channels)

        # 特化于气道代数的增强网络
        self.generation_enhance = nn.Sequential(
            nn.Conv3d(seg_channels, seg_channels, kernel_size=3, padding=1, groups=seg_channels),
            nn.BatchNorm3d(seg_channels),
            nn.Sigmoid()
        )

        # 不确定性门控模块
        self.uncertainty_gate = nn.Sequential(
            nn.Conv3d(seg_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    # def forward(self, seg_feat, cls_feat, gen_pred=None, gen_probs=None):
    #     # 基本融合
    #     fused = super(GenerationGuidanceModule, self).forward(seg_feat, cls_feat)
    #
    #     # 代数特定增强
    #     enhance_mask = self.generation_enhance(fused)
    #
    #     # 如果有代数预测，可以进一步调整增强强度
    #     if gen_pred is not None:
    #         # 获取批次大小
    #         batch_size = seg_feat.size(0)
    #
    #         # 为细小气道(类别5-6)样本增强权重
    #         fine_airway_mask = (gen_pred >= 5).float().view(batch_size, 1, 1, 1, 1)
    #         enhance_mask = enhance_mask * (1 + 0.2 * fine_airway_mask)
    #
    #     # 应用不确定性处理
    #     if gen_probs is not None:
    #         # 计算不确定性 - 最大概率的补
    #         uncertainty = 1.0 - torch.max(gen_probs, dim=1)[0]
    #         uncertainty = uncertainty.view(-1, 1, 1, 1, 1)
    #
    #         # 计算不确定性门控
    #         uncertainty_weight = self.uncertainty_gate(fused)
    #
    #         # 在高不确定性区域，保持更多原始特征
    #         fused_enhanced = fused * enhance_mask
    #         enhanced = fused_enhanced * (1 - uncertainty_weight * uncertainty) + fused * (uncertainty_weight * uncertainty)
    #     else:
    #         # 没有概率信息时的标准增强
    #         enhanced = fused * enhance_mask
    #     del seg_feat, cls_feat,fused
    #     return enhanced

    def forward(self, seg_feat, cls_feat, gen_pred=None, gen_probs=None):
        fused = super().forward(seg_feat, cls_feat)
        enhance_mask = self.generation_enhance(fused)

        # 处理代数预测分支
        if gen_pred is not None:
            fine_airway_mask = (gen_pred >= 5).float().view(seg_feat.size(0), 1, 1, 1, 1)
            enhance_mask = enhance_mask * (1 + 0.2 * fine_airway_mask)
            fine_airway_mask = fine_airway_mask.detach()  # 可安全detach

        # 处理不确定性分支
        if gen_probs is not None:
            uncertainty = 1.0 - torch.max(gen_probs, dim=1)[0].view(-1, 1, 1, 1, 1)
            uncertainty_weight = self.uncertainty_gate(fused)
            fused_enhanced = fused * enhance_mask
            enhanced = fused_enhanced * (1 - uncertainty_weight * uncertainty) + fused * (
                        uncertainty_weight * uncertainty)
            # 分离中间变量
            uncertainty = uncertainty.detach()
            # fused_enhanced = fused_enhanced.detach()
        else:
            enhanced = fused * enhance_mask

        # # 显式释放中间变量
        # del seg_feat, cls_feat, fused, enhance_mask
        # if gen_pred is not None:
        #     del fine_airway_mask
        # if gen_probs is not None:
        #     del uncertainty, fused_enhanced
        # torch.cuda.empty_cache()
        return enhanced


class StructureGuidanceModule(AnatomicalGuidanceModule):
    """结构特征引导模块 - 适用于中层解码器"""

    def __init__(self, seg_channels, cls_channels):
        super(StructureGuidanceModule, self).__init__(seg_channels, cls_channels)

        # 形态学卷积核 - 为四种主要结构类型设计
        self.morphology_convs = nn.ModuleList([
            nn.Conv3d(seg_channels, seg_channels // 4, kernel_size=3, padding=1),  # 直管道
            nn.Conv3d(seg_channels, seg_channels // 4, kernel_size=3, padding=1),  # 圆形管道
            nn.Conv3d(seg_channels, seg_channels // 4, kernel_size=3, padding=2, dilation=2),  # 分叉
            nn.Conv3d(seg_channels, seg_channels // 4, kernel_size=3, padding=3, dilation=3)  # 多分支
        ])

        # 形态特征合并
        self.morph_fusion = nn.Conv3d(seg_channels, seg_channels, kernel_size=1)
        
        # 不确定性处理模块
        self.confidence_gate = nn.Sequential(
            nn.Conv3d(seg_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    # def forward(self, seg_feat, cls_feat, struct_pred=None, struct_probs=None):
    #     # 基本融合
    #     fused = super(StructureGuidanceModule, self).forward(seg_feat, cls_feat)
    #
    #     # 获取基础形态学特征
    #     morph_feats = []
    #     for conv in self.morphology_convs:
    #         morph_feats.append(conv(fused))
    #
    #     # 合并形态学特征
    #     morph_feat = torch.cat(morph_feats, dim=1)
    #     morph_enhanced = self.morph_fusion(morph_feat)
    #
    #     # 如果有结构预测，应用特定的形态学处理
    #     if struct_pred is not None:
    #         batch_size = seg_feat.size(0)
    #
    #         # 获取批次中每个样本的预测结构类型
    #         is_bifurcation = (struct_pred == 3).float().view(batch_size, 1, 1, 1, 1)
    #
    #         # 对分叉结构应用轻微增强
    #         bifurcation_enhance = 1.0 + 0.1 * is_bifurcation
    #         enhanced = fused + morph_enhanced * bifurcation_enhance
    #
    #         # 处理结构分类不确定性
    #         if struct_probs is not None:
    #             # 获取结构分类的最大概率作为确信度
    #             confidence = torch.max(struct_probs, dim=1)[0].view(batch_size, 1, 1, 1, 1)
    #
    #             # 计算不确定性门控
    #             confidence_weight = self.confidence_gate(fused)
    #
    #             # 低确信度区域，应用更保守的特征融合
    #             enhanced = enhanced * confidence + fused * (1 - confidence)
    #     else:
    #         # 没有预测时的默认处理
    #         enhanced = fused + morph_enhanced
    #     del seg_feat, cls_feat,fused,morph_feats
    #     return enhanced

    def forward(self, seg_feat, cls_feat, struct_pred=None, struct_probs=None):
        fused = super().forward(seg_feat, cls_feat)

        # 形态学特征提取
        morph_feats = []
        for conv in self.morphology_convs:
            morph_feats.append(conv(fused))
        morph_feat = torch.cat(morph_feats, dim=1)
        morph_enhanced = self.morph_fusion(morph_feat)

        # 结构预测分支
        if struct_pred is not None:
            is_bifurcation = (struct_pred == 3).float().view(seg_feat.size(0), 1, 1, 1, 1)
            bifurcation_enhance = 1.0 + 0.1 * is_bifurcation
            enhanced = fused + morph_enhanced * bifurcation_enhance
            # 分离中间变量
            is_bifurcation = is_bifurcation.detach()
            # bifurcation_enhance = bifurcation_enhance.detach()

            # 不确定性处理分支
            if struct_probs is not None:
                confidence = torch.max(struct_probs, dim=1)[0].view(seg_feat.size(0), 1, 1, 1, 1)
                # confidence_weight = self.confidence_gate(fused)
                enhanced = enhanced * confidence + fused * (1 - confidence)
                confidence = confidence.detach()
                # confidence_weight = confidence_weight.detach()
        else:
            enhanced = fused + morph_enhanced

        # # 显式释放中间变量
        # del seg_feat, cls_feat, fused, morph_feat, morph_enhanced
        # if struct_pred is not None:
        #     del is_bifurcation, bifurcation_enhance
        # if struct_probs is not None:
        #     del confidence, confidence_weight
        # torch.cuda.empty_cache()
        return enhanced


class LocationGuidanceModule(AnatomicalGuidanceModule):
    """位置特征引导模块 - 适用于浅层编码器"""

    def __init__(self, seg_channels, cls_channels):
        super(LocationGuidanceModule, self).__init__(seg_channels, cls_channels)

        # 区域自适应卷积
        self.region_convs = nn.ModuleList([
            nn.Conv3d(seg_channels, seg_channels, kernel_size=3, padding=1),  # 上肺
            nn.Conv3d(seg_channels, seg_channels, kernel_size=3, padding=1),  # 中肺
            nn.Conv3d(seg_channels, seg_channels, kernel_size=3, padding=1),  # 下肺
            nn.Conv3d(seg_channels, seg_channels, kernel_size=3, padding=1),  # 左肺
            nn.Conv3d(seg_channels, seg_channels, kernel_size=3, padding=1),  # 右肺
            nn.Conv3d(seg_channels, seg_channels, kernel_size=3, padding=1)  # 气管
        ])
        
        # 不确定性门控
        self.uncertainty_gate = nn.Sequential(
            nn.Conv3d(seg_channels, 1, kernel_size=1),
            nn.Sigmoid()
        )

    # def forward(self, seg_feat, cls_feat, loc_pred=None, loc_probs=None):
    #     # 基本融合
    #     fused = super(LocationGuidanceModule, self).forward(seg_feat, cls_feat)
    #
    #     # 如果有位置预测，应用区域自适应处理
    #     if loc_pred is not None:
    #         batch_size = seg_feat.size(0)
    #
    #         # 初始化增强特征
    #         enhanced = torch.zeros_like(fused)
    #
    #         # 为每个位置类型应用特定的区域处理
    #         for i in range(1, 7):  # 跳过类别0(未定位)
    #             region_mask = (loc_pred == i)
    #             if region_mask.sum() > 0:
    #                 region_indices = region_mask.nonzero(as_tuple=True)[0]
    #                 region_feats = self.region_convs[i - 1](fused[region_indices])
    #                 enhanced[region_indices] = region_feats
    #
    #         # 处理未定位情况
    #         unlocated_mask = (loc_pred == 0)
    #         if unlocated_mask.sum() > 0:
    #             unlocated_indices = unlocated_mask.nonzero(as_tuple=True)[0]
    #             enhanced[unlocated_indices] = fused[unlocated_indices]
    #
    #         # 处理位置不确定性
    #         if loc_probs is not None:
    #             # 获取位置分类的最大概率
    #             max_prob = torch.max(loc_probs, dim=1)[0].view(batch_size, 1, 1, 1, 1)
    #
    #             # 计算不确定性门控
    #             uncertainty_weight = self.uncertainty_gate(fused)
    #
    #             # 低确信度区域，保持更多原始特征
    #             enhanced = enhanced * (1 - uncertainty_weight * (1 - max_prob)) + fused * (uncertainty_weight * (1 - max_prob))
    #     else:
    #         enhanced = fused
    #     del seg_feat, cls_feat,fused
    #     return enhanced

    def forward(self, seg_feat, cls_feat, loc_pred=None, loc_probs=None):
        fused = super().forward(seg_feat, cls_feat)
        enhanced = torch.zeros_like(fused)

        # 位置预测分支
        if loc_pred is not None:
            for i in range(1, 7):
                region_mask = (loc_pred == i)
                if region_mask.sum() > 0:
                    region_indices = region_mask.nonzero(as_tuple=True)[0]
                    region_feats = self.region_convs[i - 1](fused[region_indices])
                    enhanced[region_indices] = region_feats
                    # region_feats = region_feats.detach()  # 分离区域特征

            # 处理未定位区域
            unlocated_mask = (loc_pred == 0)
            if unlocated_mask.sum() > 0:
                unlocated_indices = unlocated_mask.nonzero(as_tuple=True)[0]
                enhanced[unlocated_indices] = fused[unlocated_indices]

            # 不确定性处理分支
            if loc_probs is not None:
                max_prob = torch.max(loc_probs, dim=1)[0].view(seg_feat.size(0), 1, 1, 1, 1)
                uncertainty_weight = self.uncertainty_gate(fused)
                enhanced = enhanced * (1 - uncertainty_weight * (1 - max_prob)) + fused * (
                            uncertainty_weight * (1 - max_prob))
                max_prob = max_prob.detach()
                # uncertainty_weight = uncertainty_weight.detach()
        else:
            enhanced = fused

        # # 显式释放中间变量
        # del seg_feat, cls_feat, fused
        # if loc_pred is not None:
        #     del region_feats
        # if loc_probs is not None:
        #     del max_prob, uncertainty_weight
        # torch.cuda.empty_cache()
        return enhanced