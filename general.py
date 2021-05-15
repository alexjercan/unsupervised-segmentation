# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

import os
import torch


def tensors_to_device(tensors, device):
    return (tensor.to(device, non_blocking=True) for tensor in tensors)


def generate_surfaces(normals, eps=1e-8):
    surfaces = (torch.abs(normals) >= eps)
    surfaces = torch.logical_or(surfaces[:, 0:1, :, :], torch.logical_or(surfaces[:, 1:2, :, :], surfaces[:, 2:3, :, :])).long()
    return surfaces


def generate_layers(imgs, depths, k):
    bs = imgs.shape[0]
    intervals = generate_intervals(depths, k)
    layers = [torch.stack([generate_layer(imgs[i], depths[i], intervals[i], j) for i in range(bs)]) for j in range(k)]
    return layers


def generate_layer(img, depth, intervals, j):
   return torch.where(torch.logical_and(intervals[j] <= depth, depth <= intervals[j + 1]), img, torch.zeros_like(img))


def generate_intervals(depths, k):
    return [generate_interval(depth, k) for depth in depths]


def generate_interval(depth, k):
    depths = torch.unique(depth)
    depths = depths[depths < 1]
    min_d = torch.min(depths)
    max_d = torch.max(depths)
    eps = (max_d - min_d) / k
    return [min_d + i * eps for i in range(k+1)]


def init_weights(m):
    if type(m) == torch.nn.Conv2d or type(m) == torch.nn.Conv3d or \
            type(m) == torch.nn.ConvTranspose2d or type(m) == torch.nn.ConvTranspose3d:
        torch.nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.BatchNorm2d or type(m) == torch.nn.BatchNorm3d:
        torch.nn.init.constant_(m.weight, 1)
        torch.nn.init.constant_(m.bias, 0)
    elif type(m) == torch.nn.Linear:
        torch.nn.init.normal_(m.weight, 0, 0.01)
        torch.nn.init.constant_(m.bias, 0)


def set_parameter_requires_grad(model):
    for param in model.parameters():
        param.requires_grad = False


def load_checkpoint(model, checkpoint_file, device):
    checkpoint = torch.load(checkpoint_file, map_location=device)
    init_epoch = checkpoint['epoch_idx'] + 1
    model.load_state_dict(checkpoint['state_dict'])

    return init_epoch, model


def save_checkpoint(epoch_idx, model, dir_checkpoints):
    file_name = 'checkpoint-epoch-%03d.pth' % (epoch_idx)
    output_path = os.path.join(dir_checkpoints, file_name)
    if not os.path.exists(dir_checkpoints):
        os.makedirs(dir_checkpoints)
    checkpoint = {
        'epoch_idx': epoch_idx,
        'state_dict': model.state_dict(),
    }
    torch.save(checkpoint, output_path)
