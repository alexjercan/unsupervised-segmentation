# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

from torchvision.models.segmentation.segmentation import fcn_resnet50
from util import plot_predictions
import torch
import argparse
import albumentations as A
import my_albumentations as M
import torch.nn as nn

from tqdm import tqdm
from metrics import MetricFunction, MetricFunctionNYUv2, print_single_error
from config import parse_test_config, DEVICE, read_yaml_config
from model import Model, LossFunction, ModelSmall, SupervisedLossFunction
from original import og_run_test_nyuv2, OgModel, OgLossFunction
from general import generate_layers, load_checkpoint, tensors_to_device
from dataset import create_dataloader, create_dataloader_nyuv2


def run_test(model, dataloader, loss_fn, metric_fn):
    loop = tqdm(dataloader, position=0, leave=True)

    for _, tensors in enumerate(loop):
        imgs, normals, depths = tensors_to_device(tensors, DEVICE)
        with torch.no_grad():
            predictions = model(imgs, depths)

            loss_fn(predictions, (normals, depths))
            metric_fn.evaluate(predictions, (normals, depths))
    loop.close()


def run_test_nyuv2(model, dataloader, loss_fn, metric_fn):
    loop = tqdm(dataloader, position=0, leave=True)

    for i, tensors in enumerate(loop):
        imgs, seg13, normals, depths = tensors_to_device(tensors, DEVICE)
        with torch.no_grad():
            predictions = model(imgs, depths)

            loss_fn(predictions, (normals, depths))
            metric_fn.evaluate(predictions, (seg13, normals, depths))
    loop.close()


def run_test_nyuv2_fcn(model, dataloader, loss_fn, metric_fn):
    def runmodel(model, imgs, depths):
        layers = generate_layers(imgs, depths, k=3)
        x = [model(x)['out'] for x in layers]
        return torch.stack(x, dim=-1)

    loop = tqdm(dataloader, position=0, leave=True)

    for i, tensors in enumerate(loop):
        imgs, seg13, normals, depths = tensors_to_device(tensors, DEVICE)
        with torch.no_grad():
            predictions = runmodel(model, imgs, depths)

            loss_fn(predictions, (normals, depths))
            metric_fn.evaluate(predictions, (seg13, normals, depths))
    loop.close()


def test_nyuv2(model=None, config=None):
    epoch = 0
    torch.backends.cudnn.benchmark = True

    config = parse_test_config() if not config else config

    _, dataloader = create_dataloader_nyuv2(batch_size=config.BATCH_SIZE, train=True)

    if not model:
        # model = OgModel()
        # model = Model()
        model = ModelSmall(num_classes=100)
        model = model.to(DEVICE)
        epoch, model = load_checkpoint(model, config.CHECKPOINT_FILE, DEVICE)
        model.predict.conv3 = nn.Conv2d(100, 13, kernel_size=1, stride=1, padding=0)
        model.predict.bn3 = nn.BatchNorm2d(13)

    loss_fn = LossFunction()
    # loss_fn = OgLossFunction()
    metric_fn = MetricFunctionNYUv2(config.BATCH_SIZE)

    model.eval()
    run_test_nyuv2(model, dataloader, loss_fn, metric_fn)
    print(metric_fn.best_iou, metric_fn.best_index)
    print(metric_fn.ious)
    print(sorted(range(len(metric_fn.ious)), key=lambda k: metric_fn.ious[k]))
    # og_run_test_nyuv2(model, dataloader, loss_fn, metric_fn)
    print_single_error(epoch, loss_fn.show(), metric_fn.show())


def test_nyuv2_fcn(model=None, config=None):
    epoch = 0
    torch.backends.cudnn.benchmark = True

    config = parse_test_config() if not config else config

    _, dataloader = create_dataloader_nyuv2(batch_size=config.BATCH_SIZE, train=True)

    if not model:
        model = fcn_resnet50(pretrained=False, num_classes=14)
        model = model.to(DEVICE)
        epoch, model = load_checkpoint(model, config.CHECKPOINT_FILE, DEVICE)

    loss_fn = LossFunction()
    metric_fn = MetricFunctionNYUv2(config.BATCH_SIZE)

    model.eval()
    run_test_nyuv2_fcn(model, dataloader, loss_fn, metric_fn)
    print_single_error(epoch, loss_fn.show(), metric_fn.show())


def test(model=None, config=None):
    epoch = 0
    torch.backends.cudnn.benchmark = True

    config = parse_test_config() if not config else config

    transform = A.Compose(
        [
            A.Normalize(),
            M.MyToTensorV2(),
        ],
        additional_targets={
            'normal': 'normal',
            'depth' : 'depth',
        }
    )

    _, dataloader = create_dataloader(config.DATASET_ROOT, config.JSON_PATH,
                                      batch_size=config.BATCH_SIZE, transform=transform,
                                      workers=config.WORKERS, pin_memory=config.PIN_MEMORY, shuffle=config.SHUFFLE)

    if not model:
        model = Model()
        model = model.to(DEVICE)
        epoch, model = load_checkpoint(model, config.CHECKPOINT_FILE, DEVICE)

    loss_fn = LossFunction()
    metric_fn = MetricFunction(config.BATCH_SIZE)

    model.eval()
    run_test(model, dataloader, loss_fn, metric_fn)
    print_single_error(epoch, loss_fn.show(), metric_fn.show())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='test model')
    parser.add_argument('--test', type=str, default="test.yaml", help='test config file')
    opt = parser.parse_args()

    config_test = parse_test_config(read_yaml_config(opt.test))

    # test(config=config_test)
    test_nyuv2_fcn(config=config_test)
