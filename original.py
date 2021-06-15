# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
# - https://arxiv.org/pdf/2007.09990.pdf
# - https://github.com/kanezaki/pytorch-unsupervised-segmentation-tip/blob/master/demo.py
#

import torch
import torch.optim
from config import DEVICE
from general import init_weights, load_checkpoint
from metrics import MetricFunction, print_single_error
import torch.nn as nn
from tqdm import tqdm
from model import ModelSmallBlock
from general import tensors_to_device


class OgModel(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.predict = ModelSmallBlock(num_classes)

    def forward(self, imgs):
        pred = self.predict(imgs)
        return pred


class OgLossFunction(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0):
        super(OgLossFunction, self).__init__()
        self.s_loss = nn.CrossEntropyLoss()
        self.y_loss = nn.L1Loss()
        self.z_loss = nn.L1Loss()

        self.s_loss_val = 0
        self.c_loss_val = 0

        self.alpha = alpha
        self.beta = beta

    def forward(self, predictions):
        hp_y = predictions[:, 1:, :, :] - predictions[:, 0:-1, :, :]
        hp_z = predictions[:, :, 1:, :] - predictions[:, :, 0:-1, :]

        hp_y_target = torch.zeros_like(hp_y)
        hp_z_target = torch.zeros_like(hp_z)

        _, target = torch.max(predictions, 1)

        s_loss = self.s_loss(predictions, target) * self.alpha
        c_loss = (self.y_loss(hp_y, hp_y_target) +
                  self.z_loss(hp_z, hp_z_target)) * self.beta

        self.s_loss_val = s_loss.item()
        self.c_loss_val = c_loss.item()

        return s_loss + c_loss

    def show(self):
        loss = self.s_loss_val + self.c_loss_val
        return f'(total:{loss:.4f} s:{self.s_loss_val:.4f} c:{self.c_loss_val:.4f})'


def og_train_one_epoch(model, dataloader, loss_fn, metric_fn, solver, epoch_idx):
    loop = tqdm(dataloader, position=0, leave=True)
    for i, tensors in enumerate(loop):
        imgs, normals, depths = tensors_to_device(tensors, DEVICE)

        predictions = model(imgs)

        loss = loss_fn(predictions)
        metric_fn.evaluate(predictions.unsqueeze(-1), (normals, depths))

        model.zero_grad()
        loss.backward()
        solver.step()

        loop.set_postfix(loss=loss_fn.show(), epoch=epoch_idx)
    loop.close()


def og_run_test(model, dataloader, loss_fn, metric_fn):
    loop = tqdm(dataloader, position=0, leave=True)

    for _, tensors in enumerate(loop):
        imgs, normals, depths = tensors_to_device(tensors, DEVICE)
        with torch.no_grad():
            predictions = model(imgs)

            loss_fn(predictions)
            metric_fn.evaluate(predictions.unsqueeze(-1), (normals, depths))
    loop.close()


def og_run_test_nyuv2(model, dataloader, loss_fn, metric_fn):
    loop = tqdm(dataloader, position=0, leave=True)

    for _, tensors in enumerate(loop):
        imgs, seg13, normals, depths = tensors_to_device(tensors, DEVICE)
        with torch.no_grad():
            predictions = model(imgs)

            loss_fn(predictions)
            metric_fn.evaluate(predictions.unsqueeze(-1), (seg13, normals, depths))
    loop.close()


def run_all(train_dataloader, test_dataloader, LEARNING_RATE, WEIGHT_DECAY, MILESTONES, GAMMA, LOAD_TRAIN_MODEL, CHECKPOINT_TRAIN_FILE, CHECKPOINT_TEST_FILE, NUM_EPOCHS, BATCH_SIZE, LOAD_TEST_MODEL):
    model = OgModel(num_classes=10)
    model.apply(init_weights)
    solver = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()),
                             lr=LEARNING_RATE, momentum=0.9,
                             dampening=0.1, weight_decay=WEIGHT_DECAY)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        solver, milestones=MILESTONES, gamma=GAMMA)
    model = model.to(DEVICE)
    loss_fn = OgLossFunction()

    epoch_idx = 0
    if LOAD_TRAIN_MODEL:
        epoch_idx, model = load_checkpoint(
            model, CHECKPOINT_TRAIN_FILE, DEVICE)

    model.train()
    for epoch_idx in range(epoch_idx, NUM_EPOCHS):
        metric_fn = MetricFunction(BATCH_SIZE)
        og_train_one_epoch(model, train_dataloader, loss_fn,
                           metric_fn, solver, epoch_idx)
        print_single_error(epoch_idx, loss_fn.show(), metric_fn.show())
        lr_scheduler.step()

    if LOAD_TEST_MODEL:
        epoch_idx, model = load_checkpoint(model, CHECKPOINT_TEST_FILE, DEVICE)

    model.eval()
    metric_fn = MetricFunction(BATCH_SIZE)
    og_run_test(model, test_dataloader, loss_fn, metric_fn)
    print_single_error(epoch_idx, loss_fn.show(), metric_fn.show())

if __name__ == "__main__":
    img = torch.rand((4, 3, 256, 256))
    model = OgModel(num_classes=10)
    pred = model(img)
    assert pred.shape == (4, 10, 256, 256), f"Model {pred.shape}"

    print("model ok")
