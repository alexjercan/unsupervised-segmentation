# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

import os
import cv2
import torch
import argparse
import albumentations as A
import torch.nn as nn
from torchvision.models.segmentation.fcn import FCNHead
from torchvision.models.segmentation.segmentation import fcn_resnet50
import my_albumentations as M

from config import parse_detect_config, DEVICE, read_yaml_config
from model import Model, ModelSmall
from util import plot_predictions, save_predictions
from general import generate_layers, load_checkpoint
from dataset import LoadAnimation, LoadImages, create_dataloader_nyuv2
from tqdm import tqdm


def generatePredictions(model, dataset):
    loop = tqdm(dataset, position=0, leave=True)

    for og_img, img, depth, path in loop:
        with torch.no_grad():
            img = img.to(DEVICE, non_blocking=True).unsqueeze(0)
            depth = depth.to(DEVICE, non_blocking=True).unsqueeze(0)

            predictions = model(img, depth)
            yield og_img, predictions, depth, path

    loop.close()


def generatePredictions_fcn(model, dataset):
    def runmodel(model, imgs, depths):
        layers = generate_layers(imgs, depths, k=3)
        x = [model(x)['out'] for x in layers]
        return torch.stack(x, dim=-1)

    loop = tqdm(dataset, position=0, leave=True)

    for og_img, img, depth, path in loop:
        with torch.no_grad():
            img = img.to(DEVICE, non_blocking=True).unsqueeze(0)
            depth = depth.to(DEVICE, non_blocking=True).unsqueeze(0)

            predictions = runmodel(model, img, depth)
            yield og_img, predictions, depth, path

    loop.close()


def detect_fcn(model=None, config=None):
    torch.backends.cudnn.benchmark = True

    config = parse_detect_config() if not config else config

    transform = A.Compose(
        [
            M.MyLongestMaxSize(max_size=config.IMAGE_SIZE),
            M.MyPadIfNeeded(min_height=config.IMAGE_SIZE, min_width=config.IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
            A.Normalize(),
            M.MyToTensorV2(),
        ],
        additional_targets={
            'depth' : 'depth',
        }
    )

    dataset = LoadImages(config.JSON, transform=transform)

    if not model:
        model = fcn_resnet50(pretrained=True, num_classes=21)
        model.classifier = FCNHead(2048, channels=14)
        model = model.to(DEVICE)
        epoch, model = load_checkpoint(model, config.CHECKPOINT_FILE, DEVICE)

    model.eval()
    for img, predictions, depths, path in generatePredictions_fcn(model, dataset):
        # plot_predictions([img], predictions, depths, [path])
        save_predictions([img], predictions, depths, [path])

def detect(model=None, config=None):
    torch.backends.cudnn.benchmark = True

    config = parse_detect_config() if not config else config

    transform = A.Compose(
        [
            M.MyLongestMaxSize(max_size=config.IMAGE_SIZE),
            M.MyPadIfNeeded(min_height=config.IMAGE_SIZE, min_width=config.IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0, mask_value=0),
            A.Normalize(),
            M.MyToTensorV2(),
        ],
        additional_targets={
            'depth' : 'depth',
        }
    )

    # dataset = LoadImages(config.JSON, transform=transform)
    dataset = LoadAnimation(os.path.join("..", "DrivingDepth"), transform=transform)

    if not model:
        # model = Model()
        model = ModelSmall(num_classes=100, num_layers=3)
        model = model.to(DEVICE)
        _, model = load_checkpoint(model, config.CHECKPOINT_FILE, DEVICE)
        # model.predict.conv3 = nn.Conv2d(100, 13, kernel_size=1, stride=1, padding=0)
        # model.predict.bn3 = nn.BatchNorm2d(13)

    model.eval()
    for img, predictions, depths, path in generatePredictions(model, dataset):
        # plot_predictions([img], predictions, depths, [path])
        save_predictions([img], predictions, depths, [path])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='run inference on model')
    parser.add_argument('-f', type=str, default="detect.yaml", help='detect config file')
    opt = parser.parse_args()

    config_detect = parse_detect_config(read_yaml_config(opt.f))

    detect_fcn(config=config_detect)