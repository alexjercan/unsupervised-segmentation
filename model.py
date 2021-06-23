# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
# - https://arxiv.org/pdf/2007.09990.pdf
# - https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip/blob/master/demo.py
#

from util import plot_raw_surfaces
from config import DEVICE
import torch.nn.functional as F
from general import generate_layers, generate_surfaces
import torch
import torch.nn as nn


class UNetBlock(nn.Module):
    def __init__(self, input_channel, output_channel, down_size=True):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.conv3 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(output_channel)
        self.max_pool = nn.MaxPool2d(2, 2) if down_size else nn.Identity()
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.max_pool(x)
        return x


class UNetBlockT(nn.Module):
    def __init__(self, prev_channel, input_channel, output_channel, up_sample=True):
        super(UNetBlockT, self).__init__()
        self.up_sampling = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False) if up_sample else nn.Identity()
        self.conv1 = nn.Conv2d(prev_channel + input_channel, output_channel, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(output_channel)
        self.conv2 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(output_channel)
        self.conv3 = nn.Conv2d(output_channel, output_channel, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(output_channel)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, prev_feature_map, x):
        x = self.up_sampling(x)
        x = torch.cat((x, prev_feature_map), dim=1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        return x


class UNetFeature(nn.Module):
    def __init__(self, in_channels=3):
        super(UNetFeature, self).__init__()
        self.in_channels = in_channels
        self.down_block1 = UNetBlock(in_channels, 16, False)
        self.down_block2 = UNetBlock(16, 32, True)
        self.down_block3 = UNetBlock(32, 64, True)
        self.down_block4 = UNetBlock(64, 128, True)
        self.down_block5 = UNetBlock(128, 256, True)
        self.down_block6 = UNetBlock(256, 512, True)
        self.down_block7 = UNetBlock(512, 1024, True)

        self.mid_conv1 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(1024)
        self.mid_conv2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(1024)
        self.mid_conv3 = torch.nn.Conv2d(1024, 1024, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(1024)

        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        x1 = self.down_block1(x)
        x2 = self.down_block2(x1)
        x3 = self.down_block3(x2)
        x4 = self.down_block4(x3)
        x5 = self.down_block5(x4)
        x6 = self.down_block6(x5)
        x7 = self.down_block7(x6)

        x7 = self.relu(self.bn1(self.mid_conv1(x7)))
        x7 = self.relu(self.bn2(self.mid_conv2(x7)))
        x7 = self.relu(self.bn3(self.mid_conv3(x7)))

        return x1, x2, x3, x4, x5, x6, x7


class UNetFCN(nn.Module):
    def __init__(self, out_channels=3):
        super(UNetFCN, self).__init__()
        self.up_block1 = UNetBlockT(512, 1024, 512)
        self.up_block2 = UNetBlockT(256, 512, 256)
        self.up_block3 = UNetBlockT(128, 256, 128)
        self.up_block4 = UNetBlockT(64, 128, 64)
        self.up_block5 = UNetBlockT(32, 64, 32)
        self.up_block6 = UNetBlockT(16, 32, 16)

        self.last_conv1 = nn.Conv2d(16, 16, 3, padding=1)
        self.last_bn = nn.BatchNorm2d(16)
        self.last_conv2 = nn.Conv2d(16, out_channels, 1, padding=0)
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x1, x2, x3, x4, x5, x6, x7):
        x = self.up_block1(x6, x7)
        x = self.up_block2(x5, x)
        x = self.up_block3(x4, x)
        x = self.up_block4(x3, x)
        x = self.up_block5(x2, x)
        x = self.up_block6(x1, x)

        x = self.relu(self.last_bn(self.last_conv1(x)))
        x = self.last_conv2(x)
        return x


class Model(nn.Module):
    def __init__(self, num_classes=10, num_layers=3):
        super().__init__()
        self.feature = UNetFeature()
        self.predict = UNetFCN(out_channels=num_classes)
        self.num_layers = num_layers

    def forward(self, imgs, depths):
        layers = generate_layers(imgs, depths, self.num_layers)
        # plot_raw_surfaces(imgs.permute(0, 2, 3, 1), torch.stack(layers, dim=-1))
        features = [self.feature(layer) for layer in layers]
        x = [self.predict(*x) for x in features]
        return torch.stack(x, dim=-1)


class ModelSmallBlock(nn.Module):
    def __init__(self, nChannel):
        self.nConv = 3
        super(ModelSmallBlock, self).__init__()
        self.conv1 = nn.Conv2d(3, nChannel, kernel_size=3, stride=1, padding=1 )
        self.bn1 = nn.BatchNorm2d(nChannel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()
        for i in range(self.nConv-1):
            self.conv2.append( nn.Conv2d(nChannel, nChannel, kernel_size=3, stride=1, padding=1 ) )
            self.bn2.append( nn.BatchNorm2d(nChannel) )
        self.conv3 = nn.Conv2d(nChannel, nChannel, kernel_size=1, stride=1, padding=0 )
        self.bn3 = nn.BatchNorm2d(nChannel)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu( x )
        x = self.bn1(x)
        for i in range(self.nConv-1):
            x = self.conv2[i](x)
            x = F.relu( x )
            x = self.bn2[i](x)
        x = self.conv3(x)
        x = self.bn3(x)
        return x


class ModelSmall(nn.Module):
    def __init__(self, num_classes=10, num_layers=3):
        super().__init__()
        self.predict = ModelSmallBlock(num_classes)
        self.num_layers = num_layers

    def forward(self, imgs, depths):
        layers = generate_layers(imgs, depths, self.num_layers)
        # plot_raw_surfaces(imgs.permute(0, 2, 3, 1), torch.stack(layers, dim=-1))
        x = [self.predict(x) for x in layers]
        return torch.stack(x, dim=-1)


class ContinuityLoss(nn.Module):
    def __init__(self):
        super(ContinuityLoss, self).__init__()
        self.y_loss = nn.L1Loss()
        self.z_loss = nn.L1Loss()

    def forward(self, predictions):
        device = predictions.device

        hp_y = predictions[:, 1:, :, :, :] - predictions[:, 0:-1, :, :, :]
        hp_z = predictions[:, :, 1:, :, :] - predictions[:, :, 0:-1, :, :]

        hp_y_target = torch.zeros_like(hp_y, device=device)
        hp_z_target = torch.zeros_like(hp_z, device=device)

        return (self.y_loss(hp_y, hp_y_target) + self.z_loss(hp_z, hp_z_target))


class SurfaceLoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(SurfaceLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.eps = eps


    def forward(self, predictions, normals, depths):
        num_layers = predictions.shape[-1]
        _, targets = torch.max(predictions, 1)

        surfaces = generate_surfaces(normals)
        layers = generate_layers(surfaces, depths, num_layers)
        layers = torch.stack(layers, dim=-1).squeeze(1) * targets
        return self.loss(predictions, layers)


class LossFunction(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(LossFunction, self).__init__()
        self.c_loss = ContinuityLoss()
        self.s_loss = SurfaceLoss()

        self.alpha = alpha
        self.beta = beta

        self.c_loss_val = 0
        self.s_loss_val = 0

    def forward(self, predictions, data):
        (normals, depths) = data

        c_loss = self.c_loss(predictions) * self.alpha
        s_loss = self.s_loss(predictions, normals, depths) * self.beta

        self.c_loss_val = c_loss.item()
        self.s_loss_val = s_loss.item()

        return c_loss + s_loss

    def show(self):
        loss = self.c_loss_val + self.s_loss_val
        return f'(total:{loss:.4f} c:{self.c_loss_val:.4f} s:{self.s_loss_val:.4f})'


class SupervisedLossFunction(nn.Module):
    def __init__(self):
        super(SupervisedLossFunction, self).__init__()
        self.loss = nn.CrossEntropyLoss()

        self.loss_val = 0

    def forward(self, predictions, data):
        (seg13, depths) = data
        num_layers = predictions.shape[-1]

        # seg13 = torch.clamp(seg13, 0, 9)

        layers = generate_layers(seg13, depths, num_layers)
        layers = torch.stack(layers, dim=-1).squeeze(1)

        loss = self.loss(predictions, layers)

        self.loss_val = loss.item()

        return loss

    def show(self):
        loss = self.loss_val
        return f'(total:{loss:.4f})'


if __name__ == "__main__":
    img = torch.rand((4, 3, 256, 256))
    depth = torch.rand((4, 1, 256, 256))
    model = Model(num_classes=10, num_layers=2)

    model = model.to(DEVICE)
    img = img.to(DEVICE)
    depth = depth.to(DEVICE)

    pred = model(img, depth)
    pred = pred.cpu()
    assert pred.shape == (4, 10, 256, 256, 2), f"Model {pred.shape}"

    print("model ok")
