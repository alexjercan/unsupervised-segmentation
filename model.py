# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
# - https://arxiv.org/pdf/2007.09990.pdf
# - https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip/blob/master/demo.py
#

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return self.bn(self.relu(self.conv(x)))


class CNNArray(nn.Module):
    def __init__(self, channels, num_conv=2):
        super().__init__()
        self.conv = nn.Sequential(
            *[CNNBlock(channels, channels, kernel_size=3, stride=1, padding=1) for _ in range(num_conv)]
        )

    def forward(self, x):
        return self.conv(x)


class Model(nn.Module):
    def __init__(self, in_channels=3, channels=30, num_conv=2):
        super(Model, self).__init__()
        self.conv1 = CNNBlock(in_channels, channels,
                              kernel_size=3, stride=1, padding=1)
        self.conv2 = CNNArray(channels, num_conv)
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class LossFunction(nn.Module):
    def __init__(self):
        super(LossFunction, self).__init__()
        self.loss = nn.CrossEntropyLoss()
        self.y_loss = nn.L1Loss()
        self.z_loss = nn.L1Loss()
        
        self.loss_val = 0
        self.y_loss_val = 0
        self.z_loss_val = 0

    def forward(self, predictions, targets):
        (predictions, hp_y, hp_z) = predictions
        (target, hp_y_target, hp_z_target) = targets
        
        # print(predictions.shape, target.shape)
        # exit()
        
        loss = self.loss(predictions, target) * 1.0
        y_loss = self.y_loss(hp_y, hp_y_target) * 1.0
        z_loss = self.z_loss(hp_z, hp_z_target) * 1.0
        
        self.loss_val = loss.item()
        self.y_loss_val = y_loss.item()
        self.z_loss_val = z_loss.item()
        
        return loss + y_loss + z_loss

    def show(self):
        loss = self.loss_val + self.y_loss_val + self.z_loss_val
        return f'(total:{loss:.4f} x_entropy:{self.loss_val} y:{self.y_loss_val} z:{self.z_loss_val})'


if __name__ == "__main__":
    img = torch.rand((4, 3, 256, 256))
    model = Model()
    pred = model(img)
    assert pred.shape == (4, 30, 256, 256), f"Model {pred.shape}"

    print("model ok")
