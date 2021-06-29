# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

from general import layers_to_canvas, squash_layers
import os
import cv2
import torch
import numpy as np
import albumentations as A
import matplotlib.pyplot as plt

from pathlib import Path


def load_image(path):
    img = img2rgb(path)  # RGB
    assert img is not None, 'Image Not Found ' + path
    return img


def load_depth(path, max_depth=80):
    img = exr2depth(path, maxvalue=max_depth)  # 1 channel depth
    assert img is not None, 'Image Not Found ' + path
    return img


def load_normal(path):
    img = exr2normal(path)  # 3 channel normal
    assert img is not None, 'Image Not Found ' + path
    return img


def img2rgb(path):
    if not os.path.isfile(path):
        return None

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def exr2depth(path, maxvalue=80):
    if not os.path.isfile(path):
        return None

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE | cv2.IMREAD_ANYDEPTH)

    img[img > maxvalue] = maxvalue
    img = img / maxvalue

    return np.array(img).astype(np.float32).reshape((img.shape[0], img.shape[1], -1))


def exr2normal(path):
    if not os.path.isfile(path):
        return None

    return cv2.imread(path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH) * 2 - 1


def crop2content(img):
    # convert to grayscale
    gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

    # invert gray image
    gray = 255 - gray

    # threshold
    thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY)[1]

    # apply close and open morphology to fill tiny black and white holes and save as mask
    kernel = np.ones((3,3), np.uint8)
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # get contours (presumably just one around the nonzero pixels)
    contours = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]
    cntr = contours[0]
    x,y,w,h = cv2.boundingRect(cntr)

    # make background transparent by placing the mask into the alpha channel
    new_img = cv2.cvtColor(img, cv2.COLOR_BGR2BGRA)
    new_img[:, :, 3] = mask

    # then crop it to bounding rectangle
    crop = new_img[y:y+h, x:x+w]
    crop = cv2.cvtColor(crop, cv2.COLOR_BGRA2RGB)

    # cv2.imshow("THRESH", thresh)
    # cv2.imshow("MASK", mask)
    # cv2.imshow("CROP", crop)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return crop, thresh, mask


def plot_raw_surfaces(imgs, surfaces):
    num_layers = surfaces.shape[-1]
    surfaces = (surfaces - torch.min(surfaces))
    surfaces = surfaces / torch.max(surfaces)
    _, ax = plt.subplots(1, num_layers + 1)
    for i in range(num_layers):
        ax[i].axis('off')
        ax[i].imshow(surfaces[0, :, :, :, i].permute(1, 2, 0))
    ax[-1].axis('off')
    ax[-1].imshow(imgs[0])
    plt.show()
    plt.close()


np.random.seed(42)
label_colors = np.random.randint(255, size=(100, 3))


def plot_predictions(images, predictions, depths, paths):
    plt.rcParams['figure.figsize'] = [12, 8]
    plt.rcParams['figure.dpi'] = 200

    _, predictions = torch.max(predictions, 1)
    device = predictions.device
    canvas = torch.zeros(predictions.shape[:-1], dtype=torch.long, device=device)
    predictions = torch.stack(squash_layers(predictions, depths, predictions.shape[-1]))
    for pred in predictions.permute(1, 0, 2, 3):
        canvas = torch.where(pred != -1, pred, canvas)
    # plot_raw_surfaces(images, predictions.permute(0, 2, 3, 1).unsqueeze(1))
    predictions = canvas.cpu().numpy()

    for img, pred, path in zip(images, predictions, paths):
        rgb = np.array([label_colors[c % 100] for c in pred]).astype(np.float32) / 255
        m = max(img.shape[:-1])
        rgb = A.resize(rgb, width=m, height=m, interpolation=cv2.INTER_NEAREST)
        rgb = A.center_crop(rgb, *img.shape[:-1])

        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(path)
        ax1.axis('off')
        ax1.imshow(img)
        ax2.axis('off')
        ax2.imshow(rgb)
        plt.show()


def save_predictions(images, predictions, depths, paths):
    # plt.rcParams['figure.figsize'] = [12, 8]
    plt.axis('off')
    plt.rcParams['figure.dpi'] = 200
    plt.subplots_adjust(left=0, bottom=0, right=1, top=1, wspace=0, hspace=0)

    _, predictions = torch.max(predictions, 1)
    canvas = layers_to_canvas(predictions)
    predictions = canvas.cpu().numpy()

    for img, pred, path in zip(images, predictions, paths):
        rgb = np.array([label_colors[c % 100] for c in pred]).astype(np.float32) / 255
        m = max(img.shape[:-1])
        rgb = A.resize(rgb, width=m, height=m, interpolation=cv2.INTER_NEAREST)
        rgb = A.center_crop(rgb, *img.shape[:-1])

        pred_path = str(Path(path).with_suffix(".png"))
        cv2.imwrite(pred_path, cv2.cvtColor((rgb * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))

        # plt.imshow(rgb)
        # plt.savefig(str(Path(path).with_suffix(".png")))
        # plt.close();


def save_predictions_fg(images, predictions, depths, paths):
    plt.axis('off')
    plt.rcParams['figure.dpi'] = 200

    confidences, predictions = torch.max(predictions, 1)

    for img, conf, pred, path in zip(images, confidences, predictions, paths):
        pred_path = str(Path(path).with_suffix(".png"))

        plt.imshow(img)
        plt.title(f'Object{pred+1}:{conf}')
        plt.savefig(pred_path)
        plt.close();