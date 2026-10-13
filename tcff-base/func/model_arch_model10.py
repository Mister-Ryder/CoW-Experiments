# model
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
from functools import partial


def create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding, stride, dilation=1):
    """
    Create a list of modules with together constitute a single conv layer with non-linearity
    and optional batchnorm/groupnorm.
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size(int or tuple): size of the convolving kernel
        order (string): order of things, e.g.
            'cr' -> conv + ReLU
            'gcr' -> groupnorm + conv + ReLU
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
            'bcr' -> batchnorm + conv + ReLU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
    Return:
        list of tuple (name, module)
    """
    assert 'c' in order, "Conv layer MUST be present"
    assert order[0] not in 'rle', 'Non-linearity cannot be the first operation in the layer'

    modules = []
    for i, char in enumerate(order):
        if char == 'r':
            modules.append(('ReLU', nn.ReLU(inplace=True)))
        elif char == 'l':
            modules.append(('LeakyReLU', nn.LeakyReLU(negative_slope=0.1, inplace=True)))
        elif char == 'e':
            modules.append(('ELU', nn.ELU(inplace=True)))
        elif char == 'c':
            # add learnable bias only in the absence of batchnorm/groupnorm
            bias = not ('g' in order or 'b' in order)
            modules.append(('conv',
                            nn.Conv3d(in_channels, out_channels, kernel_size, padding=padding, bias=bias, stride=stride,
                                      dilation=dilation)))
        elif char == 'g':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                num_channels = in_channels
            else:
                num_channels = out_channels

            # use only one group if the given number of groups is greater than the number of channels
            if num_channels < num_groups:
                num_groups = 1

            assert num_channels % num_groups == 0, f'Expected number of channels in input to be divisible by num_groups. num_channels={num_channels}, num_groups={num_groups}'
            modules.append(('groupnorm', nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)))
        elif char == 'b':
            is_before_conv = i < order.index('c')
            if is_before_conv:
                modules.append(('batchnorm', nn.BatchNorm3d(in_channels)))
            else:
                modules.append(('batchnorm', nn.BatchNorm3d(out_channels)))
        else:
            raise ValueError(f"Unsupported layer type '{char}'. MUST be one of ['b', 'g', 'r', 'l', 'e', 'c']")

    return modules


class SingleConv(nn.Sequential):
    """
    Basic convolutional module consisting of a Conv3d, non-linearity and optional batchnorm/groupnorm. The order
    of operations can be specified via the `order` parameter
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output channels
        kernel_size (int or tuple): size of the convolving kernel
        order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple):
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, order='gcr', num_groups=8, padding=1, stride=1,
                 dilation=1):
        super(SingleConv, self).__init__()

        for name, module in create_conv(in_channels, out_channels, kernel_size, order, num_groups, padding=padding,
                                        stride=stride, dilation=dilation):
            self.add_module(name, module)


class AttModule(nn.Module):
    def __init__(self, channel, mid_channel=8):
        super(AttModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, mid_channel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channel, channel, bias=False),
            nn.Sigmoid())

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)


class Encoder(nn.Module):
    """
    A single encoder module consisting of

    (1) two consecutive convolution layers (e.g. BatchNorm3d+ReLU+Conv3d).
    We use (Conv3d+ReLU+GroupNorm3d) by default.
    This can be changed however by providing the 'order' argument, e.g. in order
    to change to Conv3d+BatchNorm3d+ELU use order='cbe'.
    Use padded convolutions to make sure that the output (H_out, W_out) is the same
    as (H_in, W_in), so that you don't have to crop in the decoder path.

    (2) the pooling layer (one may specify the MaxPool kernel_size to be different
    than the standard (2,2,2), e.g. if the volumetric data is anisotropic
    (make sure to use complementary scale_factor in the decoder path) followed by
    a DoubleConv module.

    Args:
        in_channels (int): number of input channels
        middle_channels (int): number of middle channels
        out_channels (int): number of output channels
        apply_pooling (bool): if True use pooling
        conv_kernel_size (int or tuple): size of the convolving kernel
        pool_kernel_size (int or tuple): the size of the window
        pool_type (str): pooling layer: 'max' or 'avg'
        conv_layer_order (string): determines the order of layers, e.g.
            'cr' -> conv + ReLU
            'crg' -> conv + ReLU + groupnorm
            'cl' -> conv + LeakyReLU
            'ce' -> conv + ELU
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
    """

    def __init__(self, in_channels, middle_channels, out_channels, apply_pooling=True, conv_kernel_size=3,
                 pool_kernel_size=2, pool_type='max', conv_layer_order='gcr', num_groups=8, padding=1, stride=1):
        super(Encoder, self).__init__()
        assert pool_type in ['max', 'avg']
        if apply_pooling:
            if pool_type == 'max':
                self.pooling = nn.MaxPool3d(kernel_size=pool_kernel_size)
            else:
                self.pooling = nn.AvgPool3d(kernel_size=pool_kernel_size)
        else:
            self.pooling = None

        # conv1
        self.conv1 = SingleConv(in_channels, middle_channels, conv_kernel_size, conv_layer_order, num_groups,
                                padding=padding, stride=stride)
        # conv2
        self.conv2 = SingleConv(middle_channels, out_channels, conv_kernel_size, conv_layer_order, num_groups,
                                padding=padding, stride=stride)

        #
        self.dilation_conv = SingleConv(middle_channels, middle_channels, (3, 3, 3), conv_layer_order, num_groups,
                                        padding=(4, 4, 4), stride=stride, dilation=4)
        self.att = AttModule(channel=middle_channels)

    def forward(self, x):
        if self.pooling is not None:
            x = self.pooling(x)
        x = self.conv1(x)
        x = x + self.att(self.dilation_conv(x))
        x = self.conv2(x)

        return x


class Decoder(nn.Module):
    """
    A single module for decoder path consisting of the upsampling layer
    (either learned ConvTranspose3d or nearest neighbor interpolation) followed by a basic module (e.g. double conv like encoder).
    Args:
        in_channels (int): number of input channels
        middle_channels (int): number of middle channels
        out_channels (int): number of output channels
        conv_kernel_size (int or tuple): size of the conv kernel
        deconv_kernel_size (int or tuple): size of the deconv kernel
        scale_factor (tuple): used as the multiplier for the image H/W/D in
            case of nn.Upsample or as stride in case of ConvTranspose3d, must reverse the MaxPool3d operation
            from the corresponding encoder
        conv_layer_order (string): determines the order of layers
            in `DoubleConv` module. See `DoubleConv` for more info.
        num_groups (int): number of groups for the GroupNorm
        padding (int or tuple): add zero-padding added to all three sides of the input
        is_deconv (bool):
    """

    def __init__(self, in_channels, upsample_out_channels, conv_in_channels, conv_middle_channels, out_channels,
                 conv_kernel_size=3, conv_layer_order='gcr', num_groups=8, conv_padding=1, conv_stride=1,
                 deconv_kernel_size=4, deconv_stride=(2, 2, 2), deconv_padding=1):
        super(Decoder, self).__init__()
        # deconv
        self.upsample = nn.ConvTranspose3d(in_channels, upsample_out_channels, kernel_size=deconv_kernel_size,
                                           stride=deconv_stride, padding=deconv_padding)

        # concat joining
        self.joining = partial(self._joining, concat=True)

        # conv1
        self.conv1 = SingleConv(conv_in_channels, conv_middle_channels, conv_kernel_size, conv_layer_order, num_groups,
                                padding=conv_padding, stride=conv_stride)
        # conv2
        self.conv2 = SingleConv(conv_middle_channels, out_channels, conv_kernel_size, conv_layer_order, num_groups,
                                padding=conv_padding, stride=conv_stride)

        #
        self.dilation_conv = SingleConv(conv_middle_channels, conv_middle_channels, (3, 3, 3), conv_layer_order,
                                        num_groups, padding=(4, 4, 4), stride=conv_stride, dilation=4)
        self.att = AttModule(channel=conv_middle_channels)

    def forward(self, encoder_features, x):
        x = self.upsample(x)
        x = self.joining(encoder_features, x)

        x = self.conv1(x)
        x = x + self.att(self.dilation_conv(x))
        x = self.conv2(x)

        return x

    @staticmethod
    def _joining(encoder_features, x, concat):
        if concat:
            return torch.cat((encoder_features, x), dim=1)
        else:
            return encoder_features + x

class DistanceSmallEncoder(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(DistanceSmallEncoder, self).__init__()
        # Distance Module
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(mid_channels, 1, kernel_size=1)
        self.dist_residual_conv = nn.Conv3d(in_channels, 1, kernel_size=1)  # Residual connection for distance

    def forward(self, x):
        # Distance Module
        dist_features = F.relu(self.conv1(x))
        distance_map = torch.sigmoid(self.conv2(dist_features))
        distance_map = distance_map + self.dist_residual_conv(x)  # Add residual
        distance_map = torch.sigmoid(distance_map)  # Apply sigmoid to ensure values are in [0, 1]
        # distance_map = F.interpolate(distance_map, size=[32,128,128], mode='trilinear', align_corners=False)

        return distance_map

class AttentionSmallEncoder(nn.Module):
    def __init__(self, in_channels, mid_channels, out_channels):
        super(AttentionSmallEncoder, self).__init__()
        # Attention Module
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(mid_channels, 1, kernel_size=1)
        self.att_residual_conv = nn.Conv3d(in_channels, 1, kernel_size=1)  # Residual connection for attention

        self.upsample = nn.ConvTranspose3d(1, 256, kernel_size=4,
                                           stride=(2, 2, 2), padding=1)

    def forward(self, x):
        # Attention Module
        features = F.relu(self.conv1(x))
        attention_map = torch.sigmoid(self.conv2(features))
        attention_map = attention_map +self.att_residual_conv(x)  # Add residual
        attention_map = torch.sigmoid(attention_map)  # Apply sigmoid again to ensure values are in [0, 1]
        # attended_features = x * attention_map

        # attention_map = F.interpolate(attention_map, size=[32,128,128], mode='trilinear', align_corners=False)

        return  attention_map

class MultiHeadAttention3D(nn.Module):
    def __init__(self, in_channels, num_heads):
        super(MultiHeadAttention3D, self).__init__()
        self.num_heads = num_heads
        self.depth = in_channels // num_heads

        self.query_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.key_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)
        self.value_conv = nn.Conv3d(in_channels, in_channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, key):
        batch_size, channels, depth, height, width = x.size()

        # 将查询、键和值转换为多头格式
        def reshape_to_heads(tensor):
            return tensor.view(batch_size, self.num_heads, self.depth, depth * height * width).permute(0, 1, 3, 2)

        query = reshape_to_heads(self.query_conv(x))
        key = reshape_to_heads(self.key_conv(key))
        value = reshape_to_heads(self.value_conv(x))

        # 计算注意力得分
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / (self.depth ** 0.5)
        attention = self.softmax(attention_scores)
        attention = self.dropout(attention)

        # 应用注意力权重到值上
        out = torch.matmul(attention, value)
        out = out.permute(0, 1, 3, 2).contiguous().view(batch_size, channels, depth, height, width)
        return out


class GaussianGuidedAttention3D(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(GaussianGuidedAttention3D, self).__init__()
        self.in_channels = in_channels
        self.multihead_attention = MultiHeadAttention3D(in_channels, num_heads)
        self.gaussian_expand_conv = nn.Conv3d(1, in_channels, kernel_size=1)
        # self.norm = nn.LayerNorm([in_channels, None, None, None])  # 注意LayerNorm在3D中的使用可能需要调整
        self.norm = nn.GroupNorm(num_groups=8, num_channels=in_channels)

    def forward(self, x, gaussian):

        gaussian_expanded = self.gaussian_expand_conv(gaussian)
        # 使用多头注意力机制
        out1 = self.multihead_attention(x, gaussian_expanded)
        out = out1
        out = self.norm(out + x)  # 添加残差连接和Layer Normalization
        return out

class AttentionGuide(nn.Module):
    def __init__(self, in_channels, num_heads=8):
        super(AttentionGuide, self).__init__()
        self.in_channels = in_channels
        # self.multihead_attention = MultiHeadAttention3D(in_channels, num_heads)
        self.gaussian_expand_conv = nn.Conv3d(1, in_channels, kernel_size=1)
        # self.norm = nn.LayerNorm([in_channels, None, None, None])  # 注意LayerNorm在3D中的使用可能需要调整
        self.norm = nn.GroupNorm(num_groups=8, num_channels=in_channels)

    def forward(self, x, gaussian):
        gaussian_expanded = self.gaussian_expand_conv(gaussian)
        out = self.norm(gaussian_expanded + x)  # 添加残差连接和Layer Normalization
        return out

class FusionModule(nn.Module):
    def __init__(self, in_channels_list, out_channels=2, scale_factors=[1,2,4,8], mode='trilinear', align_corners=True):
        super(FusionModule, self).__init__()
        self.convs = nn.ModuleList([SingleConv(in_channels, out_channels, kernel_size=3, order='gcr', num_groups=8, padding=1, stride=1) for in_channels in in_channels_list])
        self.scale_factors = scale_factors

    def forward(self, encoders_features):
        outputs = []

        for i, x in enumerate(encoders_features):
            x = self.convs[i](x)
            x = F.interpolate(x, scale_factor=self.scale_factors[-i-1], mode='trilinear', align_corners=True)
            # x = F.interpolate(x, scale_factor=target_size, mode='trilinear', align_corners=True)
            outputs.append(x)
        # concatenate all tensors in the list along the channel dimension
        outputs = torch.cat(outputs, dim=1)
        return outputs

'''
class ResizeConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0):
        super(ResizeConv, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)

    def forward(self, x):
        return self.conv(x)

class FusionModule(nn.Module):
    def __init__(self, in_channels_list, out_channels=2, size=(32,32,32)):
        super(FusionModule, self).__init__()
        self.convs = nn.ModuleList([
            SingleConv(in_channels, out_channels, kernel_size=3, order='gcr', num_groups=8, padding=1, stride=1)
            for in_channels in in_channels_list
        ])
        self.resize_convs = nn.ModuleList([
            ResizeConv(out_channels, out_channels, kernel_size=1) for _ in in_channels_list
        ])
        self.target_size = size

    def forward(self, encoders_features):
        # Apply initial convolutions
        conv_outputs = [conv(x) for conv, x in zip(self.convs, encoders_features)]

        # Apply resizing through convolutions
        resized_outputs = [F.interpolate(x, size=self.target_size, mode='trilinear', align_corners=True) for x in
                           conv_outputs]

        # Apply resize convolutions
        resized_outputs = [resize_conv(x) for resize_conv, x in zip(self.resize_convs, resized_outputs)]

        # Concatenate all tensors in the list along the channel dimension
        outputs = torch.cat(resized_outputs, dim=1)
        return outputs
'''

class ExtensionModule(nn.Module):
    def __init__(self, in_channels, num_blocks, layer_order='gcr'):
        super(ExtensionModule, self).__init__()
        self.layers = nn.ModuleList(
            [SingleConv(in_channels, in_channels,3,layer_order,num_groups=2) for _ in range(num_blocks)])
        self.dilation_conv = SingleConv(in_channels, in_channels, (3, 3, 3), layer_order,
                                        2, padding=(4, 4, 4), dilation=4)
        self.att = AttModule(channel=in_channels)


    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
            x = x + self.att(self.dilation_conv(x))
        return x

class droplayer(nn.Module):
    def __init__(self, channel_num=1, thr = 0.1):
        super(droplayer, self).__init__()
        self.channel_num = channel_num
        self.threshold = thr
    def forward(self, x):
        if self.training:
            r = torch.rand(x.shape[0],self.channel_num,1,1,1).cuda()
            r[r<self.threshold] = 0
            r[r>=self.threshold] = 1
            r = r*self.channel_num/(r.sum()+0.01)
            return x*r
        else:
            return x


class SegAirwayModel(nn.Module):
    """
    Args:
        in_channels (int): number of input channels
        out_channels (int): number of output segmentation masks;
            Note that that the of out_channels might correspond to either
            different semantic classes or to different binary segmentation mask.
            It's up to the user of the class to interpret the out_channels and
            use the proper loss criterion during training (i.e. CrossEntropyLoss (multi-class)
            or BCEWithLogitsLoss (two-class) respectively)
        layer_order (string): determines the order of layers
            in `SingleConv` module. e.g. 'crg' stands for Conv3d+ReLU+GroupNorm3d.
            See `SingleConv` for more info
    """

    def __init__(self, in_channels, out_channels, layer_order='gcr', **kwargs):
        super(SegAirwayModel, self).__init__()

        # create encoder
        encoder_1 = Encoder(in_channels=in_channels, middle_channels=16, out_channels=32, apply_pooling=False,
                            conv_kernel_size=3, pool_kernel_size=2, pool_type='max', conv_layer_order=layer_order,
                            num_groups=8, padding=1, stride=1)
        encoder_2 = Encoder(in_channels=32, middle_channels=32, out_channels=64, apply_pooling=True, conv_kernel_size=3,
                            pool_kernel_size=2, pool_type='max', conv_layer_order=layer_order, num_groups=8, padding=1,
                            stride=1)
        encoder_3 = Encoder(in_channels=64, middle_channels=64, out_channels=128, apply_pooling=True,
                            conv_kernel_size=3, pool_kernel_size=2, pool_type='max', conv_layer_order=layer_order,
                            num_groups=8, padding=1, stride=1)
        encoder_4 = Encoder(in_channels=128, middle_channels=128, out_channels=256, apply_pooling=True,
                            conv_kernel_size=3, pool_kernel_size=2, pool_type='max', conv_layer_order=layer_order,
                            num_groups=8, padding=1, stride=1)

        self.encoders = nn.ModuleList([encoder_1, encoder_2, encoder_3, encoder_4])

        # create decoder
        decoder_1 = Decoder(in_channels=256, upsample_out_channels=256, conv_in_channels=384, conv_middle_channels=128,
                            out_channels=128, conv_kernel_size=3, conv_layer_order=layer_order, num_groups=2,
                            conv_padding=1, conv_stride=1, deconv_kernel_size=4, deconv_stride=(2, 2, 2),
                            deconv_padding=1)
        decoder_2 = Decoder(in_channels=128, upsample_out_channels=128, conv_in_channels=192, conv_middle_channels=64,
                            out_channels=64, conv_kernel_size=3, conv_layer_order=layer_order, num_groups=8,
                            conv_padding=1, conv_stride=1, deconv_kernel_size=4, deconv_stride=(2, 2, 2),
                            deconv_padding=1)
        decoder_3 = Decoder(in_channels=64, upsample_out_channels=64, conv_in_channels=96, conv_middle_channels=32,
                            out_channels=32, conv_kernel_size=3, conv_layer_order=layer_order, num_groups=8,
                            conv_padding=1, conv_stride=1, deconv_kernel_size=4, deconv_stride=(2, 2, 2),
                            deconv_padding=1)

        self.decoders = nn.ModuleList([decoder_1, decoder_2, decoder_3])
        # in the last layer a 1×1 convolution reduces the number of output
        # channels to the number of labels
        self.final_conv = nn.Conv3d(in_channels=8, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        self.final_activation = nn.Softmax(dim=1)
        self.final_conv2 = nn.Conv3d(in_channels=6, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        self.final_activation2 = nn.Softmax(dim=1)


        self.fusion_module_encoder = FusionModule(in_channels_list=[256,128,64,32], out_channels=2, scale_factors=[1,2,4,8])
        self.fusion_module_decoder= FusionModule(in_channels_list=[128, 64, 32], out_channels=2, scale_factors=[1,2,4])
        self.dropout1 = droplayer(channel_num=8, thr=0.1)
        self.dropout2 = droplayer(channel_num=6, thr=0.1)

        self.extension_module = ExtensionModule(in_channels=14, num_blocks=1, layer_order=layer_order)
        # self.extension_module2 = ExtensionModule(in_channels=14, num_blocks=1, layer_order=layer_order)

        self.final_conv3 = nn.Conv3d(in_channels=14, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        # self.final_activation3 = nn.Softmax(dim=1)
        self.final_activation3=nn.Sigmoid()

    def forward(self, x):
        with_gaussian=True
        # encoder part
        encoders_features = []
        decoders_features = []
        for idx, encoder in enumerate(self.encoders):
            x = encoder(x)
            # reverse the encoder outputs to be aligned with the decoder
            encoders_features.insert(0, x)

        # remove the last encoder's output from the list
        # !!remember: it's the 1st in the list
        encoders_fusion= self.dropout1(self.fusion_module_encoder(encoders_features))
        encoders_features = encoders_features[1:]

        # decoder part
        for decoder, encoder_features in zip(self.decoders, encoders_features):
            # pass the output from the corresponding encoder and the output
            # of the previous decoder
            x = decoder(encoder_features, x)
            decoders_features.append(x) #正向插入
            #x_without_gaussian = decoder(encoder_features, x_without_gaussian)
            # skl = decoder_skl(encoder_features, skl)

        decoders_fusion = self.dropout2(self.fusion_module_decoder(decoders_features))

        x_label = self.final_conv(encoders_fusion)
        x_label = self.final_activation(x_label)
        x_label2=self.final_conv2(decoders_fusion)
        x_label2=self.final_activation2(x_label2)

        x_gau=x_label
        # x = self.extension_module(torch.cat([encoders_fusion, decoders_fusion], dim=1))
        #
        # x_gau = self.final_conv3(x)
        # x_gau = self.final_activation3(x_gau)

        # x2 = self.extension_module2(torch.cat([encoders_fusion, decoders_fusion], dim=1))
        #
        # x_gau2 = self.final_conv3(x2)
        # x_gau2 = self.final_activation3(x_gau2)

        return x_label,x_label2,x_gau


import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class ResUNet3d(nn.Module):
    """
    Resnet+convdownsampleing+deepsuprivsion
    共9498260个可训练的参数, 接近九百五十万
    """

    @staticmethod
    def _block(in_channels, features, name, prob=0.2):
        block = nn.Sequential(OrderedDict([
            (name + "conv1", nn.Conv3d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False, ),),
            (name + "norm1", nn.GroupNorm(num_groups=8, num_channels=features)),
            (name + "droupout1", nn.Dropout3d(p=prob, inplace=True)),
            (name + "relu1", nn.ReLU(inplace=True)),
            (name + "conv2", nn.Conv3d(
                in_channels=features,
                out_channels=features,
                kernel_size=3,
                padding=1,
                bias=False, ),),
            (name + "norm2", nn.GroupNorm(num_groups=8, num_channels=features)),
            (name + "droupout2", nn.Dropout3d(p=prob, inplace=True)),
            (name + "relu2", nn.ReLU(inplace=True)),
        ]))
        return block

    @staticmethod
    def _downsamplingblock(in_channels, features, name, prob=0.2):
        block = nn.Sequential(OrderedDict([
            (name + "conv1", nn.Conv3d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=2,
                padding=2,
                bias=False, ),),
            (name + "norm1", nn.GroupNorm(num_groups=8, num_channels=features)),
            (name + "droupout1", nn.Dropout3d(p=prob, inplace=True)),
            (name + "relu1", nn.ReLU(inplace=True)),
        ]))
        return block

    @staticmethod
    def _upsamplingblock(in_channels, features, name, prob=0.2):
        block = nn.Sequential(OrderedDict([
            (name + "conv1", nn.ConvTranspose3d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=2,
                padding=2,
                bias=False, ),),
            (name + "norm1", nn.GroupNorm(num_groups=8, num_channels=features)),
            (name + "droupout1", nn.Dropout3d(p=prob, inplace=True)),
            (name + "relu1", nn.ReLU(inplace=True)),
        ]))
        return block

    @staticmethod
    def _outputblock(in_channels, features, scale_factor, name):
        block = nn.Sequential(OrderedDict([
            (name + "conv1", nn.Conv3d(
                in_channels=in_channels,
                out_channels=features,
                kernel_size=1,
                padding=1,
                bias=False, ),),
            (name + "up1", nn.Upsample(scale_factor=(scale_factor, scale_factor, scale_factor), mode='trilinear')),
        ]))
        return block


    def __init__(self, in_channels, out_channels, init_features=16,crop_size=(32, 32, 32)):
        super(ResUNet3d,self).__init__()

        self.features = init_features
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.crop_size=crop_size

        self.encoder_stage1 = ResUNet3d._block(self.in_channels, self.features, 'encoder_stage1')
        self.down_conv1 = ResUNet3d._downsamplingblock(self.features, self.features * 2, 'down_conv1')
        self.encoder_stage2 = ResUNet3d._block(self.features * 2, self.features * 2, 'encoder_stage2')
        self.down_conv2 = ResUNet3d._downsamplingblock(self.features * 2, self.features * 4, 'down_conv2')
        self.encoder_stage3 = ResUNet3d._block(self.features * 4, self.features * 4, 'encoder_stage3')
        self.down_conv3 = ResUNet3d._downsamplingblock(self.features * 4, self.features * 8, 'down_conv3')
        self.encoder_stage4 = ResUNet3d._block(self.features * 8, self.features * 8, 'encoder_stage4')
        self.down_conv4 = ResUNet3d._downsamplingblock(self.features * 8, self.features * 16, 'down_conv4')
        self.encoder_stage5 = ResUNet3d._block(self.features * 16, self.features * 16, 'encoder_stage5')
        self.up_conv1 = ResUNet3d._upsamplingblock(self.features * 16, self.features * 8, 'up_conv1')
        self.decoder_stage1 = ResUNet3d._block(self.features * 16, self.features * 8, 'decoder_stage1')
        self.map1 = ResUNet3d._outputblock(self.features * 8, self.out_channels, 8, 'map1')
        self.up_conv2 = ResUNet3d._upsamplingblock(self.features * 8, self.features * 4, 'up_conv2')
        self.decoder_stage2 = ResUNet3d._block(self.features * 8, self.features * 4, 'decoder_stage2')
        self.map2 = ResUNet3d._outputblock(self.features * 4, self.out_channels, 4, 'map2')
        self.up_conv3 = ResUNet3d._upsamplingblock(self.features * 4, self.features * 2, 'up_conv3')
        self.decoder_stage3 = ResUNet3d._block(self.features * 4, self.features * 2, 'decoder_stage3')
        self.map3 = ResUNet3d._outputblock(self.features * 2, self.out_channels, 2, 'map3')
        self.up_conv4 = ResUNet3d._upsamplingblock(self.features * 2, self.features, 'up_conv4')
        self.decoder_stage4 = ResUNet3d._block(self.features* 2, self.features, 'decoder_stage4')
        self.map4 = ResUNet3d._outputblock(self.features, self.out_channels, 1, 'map4')


        self.fusion_module_encoder = FusionModule(in_channels_list=[256, 128, 64, 32,16], out_channels=2,
                                                  size=self.crop_size)
        self.fusion_module_decoder = FusionModule(in_channels_list=[128, 64, 32, 16], out_channels=2,
                                                  size=self.crop_size)

        self.final_conv = nn.Conv3d(in_channels=10, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        # self.final_activation = nn.Softmax(dim=1)
        self.final_conv2 = nn.Conv3d(in_channels=8, out_channels=out_channels, kernel_size=3, padding=1, stride=1)
        # self.final_activation2 = nn.Sigmoid()
        self.dropout1 = droplayer(channel_num=10, thr=0.1)
        self.dropout2 = droplayer(channel_num=8, thr=0.1)

    def forward(self, inputs):
        long_range1 = self.encoder_stage1(inputs)

        short_range1 = self.down_conv1(long_range1)
        long_range2 = self.encoder_stage2(short_range1) + short_range1

        short_range2 = self.down_conv2(long_range2)
        long_range3 = self.encoder_stage3(short_range2) + short_range2

        short_range3 = self.down_conv3(long_range3)
        long_range4 = self.encoder_stage4(short_range3) + short_range3

        short_range4 = self.down_conv4(long_range4)
        long_range5 = self.encoder_stage5(short_range4) + short_range4

        short_range6 = self.up_conv1(long_range5)
        outputs1 = self.decoder_stage1(torch.cat([short_range6, long_range4], dim=1)) + short_range6
        output1_logit = self.map1(outputs1)

        short_range7 = self.up_conv2(outputs1)
        outputs2 = self.decoder_stage2(torch.cat([short_range7, long_range3], dim=1)) + short_range7
        output2_logit = self.map2(outputs2)

        short_range8 = self.up_conv3(outputs2)
        outputs3 = self.decoder_stage3(torch.cat([short_range8, long_range2], dim=1)) + short_range8
        output3_logit = self.map3(outputs3)

        short_range9 = self.up_conv4(outputs3)
        outputs4 = self.decoder_stage4(torch.cat([short_range9, long_range1], dim=1)) + short_range9
        output4_logit = self.map4(outputs4)

        encoders_features= [long_range5,long_range4 ,long_range3 ,long_range2 ,long_range1]
        encoders_fusion = self.final_conv(self.dropout1(self.fusion_module_encoder(encoders_features)))

        decoders_features =[outputs1,outputs2,outputs3,outputs4]
        decoders_fusion = self.final_conv2(self.dropout2(self.fusion_module_decoder(decoders_features)))

        if self.out_channels == 1:
            output1 = torch.sigmoid(encoders_fusion)
            output2 = torch.sigmoid(decoders_fusion)

        if self.out_channels > 1:
            output1 = torch.softmax(encoders_fusion, dim=1)
            output2 = torch.softmax(decoders_fusion, dim=1)
        _ =output1

        return output1, output2,_


        # if self.out_channels == 1:
        #     output1 = torch.sigmoid(output1_logit)
        #     output2 = torch.sigmoid(output2_logit)
        #     output3 = torch.sigmoid(output3_logit)
        #     output4 = torch.sigmoid(output4_logit)
        # if self.out_channels > 1:
        #     output1 = torch.softmax(output1_logit, dim=1)
        #     output2 = torch.softmax(output2_logit, dim=1)
        #     output3 = torch.softmax(output3_logit, dim=1)
        #     output4 = torch.softmax(output4_logit, dim=1)
        # return output1_logit, output2_logit, output3_logit, output4_logit, output1, output2, output3, output4


