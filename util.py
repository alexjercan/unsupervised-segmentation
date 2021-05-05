# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

from pathlib import Path


def load_image(path):
    img = img2rgb(path)  # RGB
    assert img is not None, 'Image Not Found ' + path
    return img


def img2rgb(path):
    if not os.path.isfile(path):
        return None

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


label_colors = np.random.randint(255, size=(100, 3))


def plot_predictions(images, predictions, paths):
    _, predictions = torch.max(predictions, 1)

    predictions = predictions.cpu().numpy()

    for img, pred, path in zip(images, predictions, paths):
        rgb = np.array([label_colors[c % 100] for c in pred]).astype(np.float32) / 255
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(path)
        ax1.axis('off')
        ax1.imshow(img)
        ax2.axis('off')
        ax2.imshow(rgb)
        plt.show()


def save_predictions(predictions, paths):
    _, predictions = torch.max(predictions, dim=1)

    predictions = predictions.cpu().numpy()

    for pred, path in zip(predictions, paths):
        rgb = np.array([label_colors[c % 100] for c in pred]).astype(np.float32) / 255
        pred_path = str(Path(path).with_suffix(".exr"))

        cv2.imwrite(pred_path, rgb)

        plt.axis('off')
        plt.imshow(rgb)
        plt.savefig(str(Path(path).with_suffix(".png")))
