# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

import torch
import argparse
import albumentations as A
import my_albumentations as M

from tqdm import tqdm
from metrics import MetricFunction, print_single_error
from config import parse_test_config, DEVICE, read_yaml_config
from model import Model, LossFunction
from general import load_checkpoint, tensors_to_device
from dataset import create_dataloader


def run_test(model, dataloader, loss_fn, metric_fn):
    loop = tqdm(dataloader, position=0, leave=True)
    
    for _, imgs in enumerate(loop):
        with torch.no_grad():
            imgs = imgs.to(DEVICE, non_blocking=True)

            predictions = model(imgs)
            hp_y = predictions[:, 1:, :, :] - predictions[:, 0:-1, :, :]
            hp_z = predictions[:, :, 1:, :] - predictions[:, :, 0:-1, :]
            
            hp_y_target = torch.zeros_like(hp_y)
            hp_z_target = torch.zeros_like(hp_z)
            
            predictions = predictions.view(predictions.shape[0], predictions.shape[1], -1)
            _, target = torch.max(predictions, 1)
            
            predictions = tuple(tensors_to_device((predictions, hp_y, hp_z), DEVICE))
            targets = tuple(tensors_to_device((target, hp_y_target, hp_z_target), DEVICE))

            loss_fn(predictions, targets)
            metric_fn.evaluate(predictions, targets)
    loop.close()


def test(model=None, config=None):
    epoch = 0
    torch.backends.cudnn.benchmark = True

    config = parse_test_config() if not config else config

    transform = A.Compose(
        [
            A.Normalize(mean=0, std=1),
            M.MyToTensorV2(),
        ]
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

    test(config=config_test)
