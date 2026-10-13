# EnhancedAirwaySegmentation.py - 内存优化版
import torch
import torch.nn as nn
import torch.nn.functional as F
import gc

from func.AnatomicalFusionModules import GenerationGuidanceModule, StructureGuidanceModule, LocationGuidanceModule
from func.model_arch_model10 import SegAirwayModel
from func.AnatomicalClassifier import AirwayAnatomicalClassifier


class EnhancedAirwaySegmentation(nn.Module):
    def __init__(self, segmentation_model_path=None, classifier_model_path=None):
        super(EnhancedAirwaySegmentation, self).__init__()

        # 直接在内部创建模型实例，而不是从外部接收
        self.base_model = SegAirwayModel(in_channels=1, out_channels=2)
        self.classifier = AirwayAnatomicalClassifier(
            in_channels=1,
            generation_classes=7,
            structure_classes=7,
            location_classes=7
        )

        # 加载预训练权重（如果提供）
        if segmentation_model_path:
            self._load_weights(self.base_model, segmentation_model_path)

        if classifier_model_path:
            self._load_weights(self.classifier, classifier_model_path)

            # 默认冻结分类器
            for param in self.classifier.parameters():
                param.requires_grad = False

        # 创建解剖学引导模块
        encoder_channels = [32, 64, 128, 256]  # 基础网络的编码器通道数
        decoder_channels = [128, 64, 32]  # 基础网络的解码器通道数
        classifier_channels = [64, 128, 256]  # 分类器的特征通道数

        # 位置引导模块 - 用于浅层编码器
        self.location_guidance = LocationGuidanceModule(
            seg_channels=encoder_channels[1],  # 编码器第二层
            cls_channels=classifier_channels[0]  # 分类器浅层特征
        )

        # 结构引导模块 - 用于中层解码器
        self.structure_guidance = StructureGuidanceModule(
            seg_channels=decoder_channels[1],  # 解码器中层
            cls_channels=classifier_channels[1]  # 分类器中层特征
        )

        # 代数引导模块 - 用于深层解码器
        self.generation_guidance = GenerationGuidanceModule(
            seg_channels=decoder_channels[0],  # 解码器深层
            cls_channels=classifier_channels[2]  # 分类器深层特征
        )

        # 解剖学预测融合输出层
        self.anatomical_output_fusion = nn.Sequential(
            nn.Conv3d(22, decoder_channels[2], kernel_size=3, padding=1),  # 22
            nn.BatchNorm3d(decoder_channels[2]),
            nn.ReLU(inplace=True),
            nn.Conv3d(decoder_channels[2], 2, kernel_size=1)  # 二分类输出 (背景/气道)
        )

        # 分割反馈到分类器的通路
        self.seg_to_cls_feedback = nn.ModuleDict({
            'generation': nn.Sequential(
                nn.Conv3d(decoder_channels[0], classifier_channels[2], kernel_size=1),
                nn.BatchNorm3d(classifier_channels[2]),
                nn.ReLU(inplace=True)
            ),
            'structure': nn.Sequential(
                nn.Conv3d(decoder_channels[1], classifier_channels[1], kernel_size=1),
                nn.BatchNorm3d(classifier_channels[1]),
                nn.ReLU(inplace=True)
            ),
            'location': nn.Sequential(
                nn.Conv3d(encoder_channels[1], classifier_channels[0], kernel_size=1),
                nn.BatchNorm3d(classifier_channels[0]),
                nn.ReLU(inplace=True)
            )
        })

        # 分类器不确定性估计
        self.uncertainty_estimator = nn.ModuleDict({
            'generation': nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(decoder_channels[0], 7, kernel_size=1)
            ),
            'structure': nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(decoder_channels[1], 7, kernel_size=1)
            ),
            'location': nn.Sequential(
                nn.AdaptiveAvgPool3d(1),
                nn.Conv3d(encoder_channels[1], 7, kernel_size=1)
            )
        })

        # 自适应阈值模块 - 用于推理时
        self.adaptive_threshold = nn.Sequential(
            nn.Conv3d(3, 8, kernel_size=3, padding=1),
            nn.BatchNorm3d(8),
            nn.ReLU(inplace=True),
            nn.Conv3d(8, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.gau_final = nn.Conv3d(in_channels=2, out_channels=1, kernel_size=3, padding=1, stride=1)
        self.gau_activation = nn.Sigmoid()

    def _load_weights(self, model, checkpoint_path):
        """加载预训练权重并处理可能的前缀问题"""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('model_state_dict', checkpoint)  # 支持直接加载state_dict

        # 移除"module."前缀（如果有）
        new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}

        # 使用严格=False加载，允许部分参数不匹配
        model.load_state_dict(new_state_dict, strict=False)
        print(f"成功加载权重: {checkpoint_path}")

        # 主动释放内存
        del checkpoint, state_dict, new_state_dict
        torch.cuda.empty_cache()

    def unfreeze_classifier_deep_layers(self):
        """解冻分类器的深层特征 - 第二阶段训练"""
        for i, param in enumerate(self.classifier.parameters()):
            if i >= len(list(self.classifier.parameters())) // 2:
                param.requires_grad = True
        print("已解冻分类器的深层特征")

    def unfreeze_all_classifier(self):
        """解冻所有分类器参数 - 第三阶段训练"""
        for param in self.classifier.parameters():
            param.requires_grad = True
        print("已解冻所有分类器参数")

    def forward(self, x, joint_training=False):
        # 获取分类器预测和特征
        cls_training = self.training and joint_training

        with torch.set_grad_enabled(cls_training):  # 仅在联合训练时计算分类器梯度
            cls_output = self.classifier(x)

        cls_features = cls_output['encoder_features']  # 获取分类器特征
        guidance_map = cls_output['guidance_map']  # 获取引导图

        # 获取分类预测
        gen_logits = cls_output['generation']
        struct_logits = cls_output['structure']
        loc_logits = cls_output['location']

        with torch.no_grad():
            # 获取分类预测的概率分布
            gen_probs = F.softmax(gen_logits, dim=1)
            struct_probs = F.softmax(struct_logits, dim=1)
            loc_probs = F.softmax(loc_logits, dim=1)

            # 获取分类预测的类别
            gen_pred = torch.argmax(gen_logits, dim=1)
            struct_pred = torch.argmax(struct_logits, dim=1)
            loc_pred = torch.argmax(loc_logits, dim=1)

        # 运行分割网络的编码器
        encoder_features = []
        x_enc = x

        # 创建存储反馈特征的字典（用于联合训练）
        feedback_features = {}

        # 运行编码器
        for i, encoder in enumerate(self.base_model.encoders):
            if i == 1:  # 在第二个编码器层应用位置引导
                x_enc = encoder(x_enc)
                feedback_features['location'] = x_enc  # 存储位置特征用于反馈
                x_enc = self.location_guidance(x_enc, cls_features[0], loc_pred, loc_probs)
            else:
                x_enc = encoder(x_enc)

            # 存储编码器特征以便解码器使用
            encoder_features.append(x_enc)

        # 翻转编码器特征以匹配解码器顺序
        encoder_features = list(reversed(encoder_features))

        # 运行解码器
        x_dec = encoder_features[0]  # 从瓶颈层开始
        encoder_features = encoder_features[1:]  # 移除瓶颈层

        decoder_features = []
        # #gau添加
        # decoders_features_center = []
        # x_dec_center = x_dec

        # 运行解码器
        for i, decoder in enumerate(self.base_model.decoders):
            # gau添加
            # for i, (decoder,decoder_c) in enumerate(self.base_model.decoders,self.base_model.decoders_center):

            if i == 0:  # 第一个解码器 - 应用代数引导
                x_dec = decoder(encoder_features[i], x_dec)
                feedback_features['generation'] = x_dec  # 存储代数特征用于反馈
                x_dec = self.generation_guidance(x_dec, cls_features[2], gen_pred, gen_probs)

            elif i == 1:  # 第二个解码器 - 应用结构引导
                x_dec = decoder(encoder_features[i], x_dec)
                feedback_features['structure'] = x_dec  # 存储结构特征用于反馈
                x_dec = self.structure_guidance(x_dec, cls_features[1], struct_pred, struct_probs)

            else:  # 最后一个解码器 - 常规处理
                x_dec = decoder(encoder_features[i], x_dec)

            decoder_features.append(x_dec)

        # 融合解码器特征
        decoder_fusion = self.base_model.fusion_module_decoder(decoder_features)

        # 将引导图上采样到解码器输出尺寸
        if guidance_map.shape[2:] != decoder_fusion.shape[2:]:
            resized_guidance_map = F.interpolate(guidance_map, size=decoder_fusion.shape[2:],
                                                 mode='trilinear', align_corners=False)
            del guidance_map  # 明确删除原始引用
            guidance_map = resized_guidance_map  # 重新赋值

        # 融合引导图和解码器输出
        final_fusion = torch.cat([decoder_fusion, guidance_map], dim=1)
        segmentation_temp = self.anatomical_output_fusion(final_fusion)
        segmentation = self.base_model.final_activation2(segmentation_temp)
        gau = self.gau_activation(self.gau_final(segmentation_temp))

        # #gau添加
        # final_fusion_center = torch.cat([decoders_fusion_center, guidance_map], dim=1)
        # gau = self.anatomical_output_fusion(final_fusion_center)
        # gau = self.base_model.final_activation3(self.base_model.final_conv3(gau))

        # 计算分类器不确定性和增强特征（联合训练时）
        cls_uncertainty = {}
        if self.training and joint_training:
            # 使用分割特征增强分类器的判别能力
            enhanced_cls_features = []

            # 生成增强特征
            for key, feature in feedback_features.items():
                enhanced_feature = self.seg_to_cls_feedback[key](feature)

                # 匹配尺寸
                if key == 'generation' and enhanced_feature.shape[2:] != cls_features[2].shape[2:]:
                    enhanced_feature = F.interpolate(enhanced_feature, size=cls_features[2].shape[2:],
                                                     mode='trilinear', align_corners=False)
                elif key == 'structure' and enhanced_feature.shape[2:] != cls_features[1].shape[2:]:
                    enhanced_feature = F.interpolate(enhanced_feature, size=cls_features[1].shape[2:],
                                                     mode='trilinear', align_corners=False)
                elif key == 'location' and enhanced_feature.shape[2:] != cls_features[0].shape[2:]:
                    enhanced_feature = F.interpolate(enhanced_feature, size=cls_features[0].shape[2:],
                                                     mode='trilinear', align_corners=False)

                enhanced_cls_features.append((key, enhanced_feature))

                # 计算不确定性估计
                uncertainty_logits = self.uncertainty_estimator[key](feature)
                cls_uncertainty[key] = uncertainty_logits

            # 将增强特征添加到输出中供联合训练使用
            cls_output['enhanced_features'] = enhanced_cls_features
            cls_output['uncertainty_estimates'] = cls_uncertainty
        del feedback_features

        # 应用解剖学自适应阈值（推理时使用）
        if 0:
            with torch.no_grad():
                # 计算分类器不确定性 - 使用熵作为度量
                gen_entropy = -torch.sum(gen_probs * torch.log(gen_probs + 1e-10), dim=1, keepdim=True)
                struct_entropy = -torch.sum(struct_probs * torch.log(struct_probs + 1e-10), dim=1, keepdim=True)

                # 确保entropy有正确的维度
                if len(gen_entropy.shape) == 2:  # 如果只有[batch_size, 1]维度
                    gen_entropy = gen_entropy.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # 添加三个空间维度
                if len(struct_entropy.shape) == 2:  # 如果只有[batch_size, 1]维度
                    struct_entropy = struct_entropy.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # 添加三个空间维度

                # 形状调整为与分割输出匹配
                if gen_entropy.shape[2:] != segmentation.shape[2:]:
                    gen_entropy = F.interpolate(gen_entropy, size=segmentation.shape[2:],
                                                mode='trilinear', align_corners=False)
                if struct_entropy.shape[2:] != segmentation.shape[2:]:
                    struct_entropy = F.interpolate(struct_entropy, size=segmentation.shape[2:],
                                                   mode='trilinear', align_corners=False)

                # 对不确定性进行标准化
                gen_uncertainty = gen_entropy / torch.log(torch.tensor(7.0, device=gen_entropy.device))  # 7类，最大熵为log(7)
                struct_uncertainty = struct_entropy / torch.log(torch.tensor(7.0, device=struct_entropy.device))

                # 为细小气道样本降低阈值
                fine_airway_mask = (gen_pred >= 5).float().view(-1, 1, 1, 1, 1)
                # 分叉处增强连通性
                bifurcation_mask = (struct_pred == 3).float().view(-1, 1, 1, 1, 1)
                # 直管道区域增强
                tube_mask = (struct_pred == 1).float().view(-1, 1, 1, 1, 1)

                # 不确定性补偿 - 在高不确定性区域降低阈值增强力度
                gen_certainty = 1.0 - gen_uncertainty
                struct_certainty = 1.0 - struct_uncertainty

                # 应用自适应增强
                # 基本增强
                base_enhance = 1.0 + 0.3 * fine_airway_mask + 0.2 * bifurcation_mask + 0.2 * tube_mask

                # 不确定性调整
                uncertainty_factor = gen_certainty * struct_certainty

                # 将增强强度与不确定性关联 - 在高确定性区域应用更强的增强
                adaptive_enhance = base_enhance * uncertainty_factor

                # 应用到分割输出
                segmentation[:, 1:2] = segmentation[:, 1:2] * adaptive_enhance

                # # 应用自适应阈值模块进行细粒度调整
                # threshold_input = torch.cat([
                #     segmentation[:, 1:2],  # 只使用前景通道
                #     gen_uncertainty.expand_as(segmentation[:, 0:1]),
                #     struct_uncertainty.expand_as(segmentation[:, 0:1])
                # ], dim=1)
                # threshold_factor = self.adaptive_threshold(threshold_input)
                # segmentation[:, 1:2] = segmentation[:, 1:2] * threshold_factor

        return segmentation, cls_output, gau
