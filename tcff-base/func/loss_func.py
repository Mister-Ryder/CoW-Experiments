# loss function
# from func.structure_acc import cal_structure_acc, get_clusters_and_centers_of_slices

import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch
import numpy as np


def dice_loss_weights(pred, target, weights):
    """
    This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    smooth = 0.01

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    weights_flat = weights.contiguous().view(-1)

    intersection = 2. * torch.sum(torch.mul(torch.mul(iflat, tflat), weights_flat))

    # A_sum = torch.sum(torch.mul(torch.mul(iflat, iflat), weights_flat))
    # B_sum = torch.sum(torch.mul(torch.mul(tflat, tflat), weights_flat))
    A_sum = torch.sum(torch.mul(iflat, iflat))
    B_sum = torch.sum(torch.mul(tflat, tflat))

    return 1 - ((intersection + smooth) / (A_sum + B_sum + smooth))


def dice_loss(pred, target):
    """
    This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    smooth = 1.

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)

    intersection = 2. * torch.sum(torch.mul(iflat, tflat))

    A_sum = torch.sum(torch.mul(iflat, iflat))
    B_sum = torch.sum(torch.mul(tflat, tflat))

    return 1 - ((intersection + smooth) / (A_sum + B_sum + smooth))


def mae_loss(pred, target):
    """
    Mean Absolute Error (MAE) Loss
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    return torch.mean(torch.abs(pred - target))


def mse_loss(pred, target):
    """
    Mean Squared Error (MSE) Loss
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """
    return torch.mean((pred - target) ** 2)


def dice_loss_power_weights(pred, target, weights, alpha=0.5, delta=0.000001):
    smooth = 1
    delta = 0.1

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)
    weights_flat = weights.contiguous().view(-1)

    intersection = 2. * torch.sum(torch.mul(torch.mul(torch.pow(iflat + delta, alpha), tflat), weights_flat))

    A_sum = torch.sum(
        torch.mul(torch.mul(torch.pow(iflat + delta, alpha), torch.pow(iflat + delta, alpha)), weights_flat))
    B_sum = torch.sum(torch.mul(torch.mul(tflat, tflat), weights_flat))

    return 1 - ((intersection + smooth) / (A_sum + B_sum + smooth))


def dice_loss_power(pred, target, alpha=0.5, delta=0.000001):
    smooth = 1
    delta = 0.1

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)

    intersection = 2. * torch.sum(torch.mul(torch.pow(iflat + delta, alpha), tflat))

    A_sum = torch.sum(torch.mul(torch.pow(iflat + delta, alpha), torch.pow(iflat + delta, alpha)))
    B_sum = torch.sum(torch.mul(tflat, tflat))

    return 1 - ((intersection + smooth) / (A_sum + B_sum + smooth))


def dice_accuracy(pred, target):
    """
    This definition generalize to real valued pred and target vector.
    This should be differentiable.
    pred: tensor with first dimension as batch
    target: tensor with first dimension as batch
    """

    # have to use contiguous since they may from a torch.view op
    iflat = pred.contiguous().view(-1)
    tflat = target.contiguous().view(-1)

    intersection = 2. * torch.sum(torch.mul(iflat, tflat))

    A_sum = torch.sum(torch.mul(iflat, iflat))
    B_sum = torch.sum(torch.mul(tflat, tflat))

    return (intersection) / (A_sum + B_sum + 0.0001)


class ConnectivityLoss3D(nn.Module):
    def __init__(self):
        super(ConnectivityLoss3D, self).__init__()
        # 卷积核初始化为两个通道
        self.kernel = torch.tensor([[[[1, 1, 1],
                                      [1, 1, 1],
                                      [1, 1, 1]],
                                     [[1, 1, 1],
                                      [1, 1, 1],
                                      [1, 1, 1]],
                                     [[1, 1, 1],
                                      [1, 1, 1],
                                      [1, 1, 1]]]], dtype=torch.float32)
        self.kernel = self.kernel.view(1, 1, 3, 3, 3)
        # 为每个输入通道复制卷积核
        # self.kernel = self.kernel.repeat(2, 1, 1, 1, 1)

    def compute_connectivity(self, skeleton):
        device = skeleton.device
        kernel = self.kernel.to(device)

        # 确保卷积核的通道数与输入相匹配
        connectivity = torch.nn.functional.conv3d(skeleton, kernel, padding=1)#, groups=2)

        # 计算至少与1个其他体素相连的体素数量
        connected_voxels = torch.sum(connectivity > 1, dim=[2, 3, 4])

        # 计算连通性损失
        total_voxels = torch.prod(torch.tensor(skeleton.shape[2:])).float()
        connectivity_loss = (total_voxels - connected_voxels.float()) / total_voxels

        return connectivity_loss.mean()

    def forward(self, pred_skeleton, true_skeleton):
        # pred_skeleton = pred_skeleton.sigmoid() > 0.5
        pred_connectivity_loss = self.compute_connectivity(pred_skeleton)
        # true_skeleton = true_skeleton > 0.5
        true_connectivity_loss = self.compute_connectivity(true_skeleton)

        connectivity_loss = torch.abs(pred_connectivity_loss - true_connectivity_loss)

        return connectivity_loss

def create_window(window_size, window_sigma, device):
    x = torch.arange(window_size, device=device).float()  # 确保是float类型以进行除法
    gauss = torch.exp(-(x - window_size // 2) ** 2 / (2 * window_sigma ** 2))
    gauss = gauss / gauss.sum()  # 归一化
    gauss = gauss.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # 调整形状
    window = gauss.repeat(1, 1, window_size, window_size, window_size)
    return window

def ssim_loss(pred, target, window_size=11, window_sigma=1.5, data_range=1.0, k1=0.01, k2=0.03):
    C1 = (k1 * data_range) ** 2
    C2 = (k2 * data_range) ** 2

    pred = pred.view(-1, 1, 32, 128, 128)
    target = target.view(-1, 1, 32, 128, 128)
    ssim_total = ssim(pred, target, window_size, window_sigma, C1, C2)

    return 1.0 - ssim_total

def ssim(pred, target, window_size, window_sigma, C1, C2):
    window = create_window(window_size, window_sigma, pred.device)
    mu1 = F.conv3d(pred, window, padding=window_size//2)
    mu2 = F.conv3d(target, window, padding=window_size//2)

    pred_sq = pred * pred
    target_sq = target * target
    pred_target = pred * target

    sigma1_sq = F.conv3d(pred_sq, window, padding=window_size//2) - mu1 ** 2
    sigma2_sq = F.conv3d(target_sq, window, padding=window_size//2) - mu2 ** 2
    sigma12 = F.conv3d(pred_target, window, padding=window_size//2) - mu1 * mu2

    # 避免除零错误，增加小的常数 epsilon
    epsilon = 1e-8
    luminance = (2 * mu1 * mu2 + C1) / (mu1 ** 2 + mu2 ** 2 + C1 + epsilon)
    contrast = (2 * sigma12 + C2) / (sigma1_sq + sigma2_sq + C2 + epsilon)
    structure = (sigma12 + C2/2) / (torch.sqrt(sigma1_sq * sigma2_sq) + C2/2 + epsilon)

    ssim_map = luminance * contrast * structure
    return ssim_map.mean()