# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

import os
import json
from copy import copy
import cv2

from torch.utils.data import Dataset, DataLoader
from util import load_depth, load_image, load_normal
from nyuv2 import NYUv2
from torchvision import transforms


def create_dataloader(dataset_root, json_path, batch_size=2, transform=None, workers=8, pin_memory=True, shuffle=True):
    dataset = BDataset(dataset_root, json_path, transform=transform)
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=nw, pin_memory=pin_memory, shuffle=shuffle)
    return dataset, dataloader


def create_dataloader_nyuv2(batch_size=2, transform=None, workers=8, pin_memory=True, shuffle=True):
    t = transforms.Compose([transforms.Resize((256, 256)), transforms.ToTensor()])
    dataset = NYUv2(root="../NYUv2", download=True,
        rgb_transform=t, seg_transform=t, sn_transform=t, depth_transform=t)
    batch_size = min(batch_size, len(dataset))
    nw = min([os.cpu_count(), batch_size if batch_size > 1 else 0, workers])
    dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=nw, pin_memory=pin_memory, shuffle=shuffle)
    return dataset, dataloader


class BDataset(Dataset):
    def __init__(self, dataset_root, json_path, transform=None):
        super(BDataset, self).__init__()
        self.dataset_root = dataset_root
        self.json_path = os.path.join(dataset_root, json_path)
        self.transform = transform

        with open(self.json_path, "r") as f:
            self.json_data = json.load(f)

    def __len__(self):
        return len(self.json_data)

    def __getitem__(self, index):
        data = self.__load__(index)
        data = self.__transform__(data)
        return data

    def __load__(self, index):
        img_path = os.path.join(self.dataset_root, self.json_data[index]["image"])
        normal_path = os.path.join(self.dataset_root, self.json_data[index]["normal"])
        depth_path = os.path.join(self.dataset_root, self.json_data[index]["depth"])

        img = load_image(img_path)
        normal = load_normal(normal_path)
        depth = load_depth(depth_path)

        return img, normal, depth

    def __transform__(self, data):
        img, normal, depth = data

        if self.transform is not None:
            augmentations = self.transform(image=img, normal=normal, depth=depth)
            img = augmentations["image"]
            normal = augmentations["normal"]
            depth = augmentations["depth"]

        return img, normal, depth


class LoadImages():
    def __init__(self, json_data, transform=None):
        self.json_data = json_data
        self.transform = transform
        self.count = 0

    def __len__(self):
        return len(self.json_data)

    def __iter__(self):
        self.count = 0
        return self

    def __next__(self):
        index = self.count

        if self.count == self.__len__():
            raise StopIteration
        self.count += 1

        data = self.__load__(index)
        data =  self.__transform__(data)
        return data

    def __load__(self, index):
        img_path = self.json_data[index]["image"]
        depth_path = self.json_data[index]["depth"]
        output_img_path = self.json_data[index]["output"]

        img = load_image(img_path)
        depth = load_depth(depth_path)

        return img, depth, output_img_path

    def __transform__(self, data):
        img, depth, output_path = data
        og_img = copy(img)

        if self.transform is not None:
            augmentations = self.transform(image=img, depth=depth)
            img = augmentations["image"]
            depth = augmentations["depth"]

        return og_img, img, depth, output_path


if __name__ == "__main__":
    from config import JSON, IMAGE_SIZE
    import albumentations as A
    import my_albumentations as M
    import matplotlib.pyplot as plt

    def visualize(image):
        plt.figure(figsize=(10, 10))
        plt.axis('off')
        plt.imshow(image)
        plt.show()

    my_transform = A.Compose(
        [
            M.MyRandomResizedCrop(width=IMAGE_SIZE, height=IMAGE_SIZE),
            M.MyHorizontalFlip(p=0.5),
            M.MyVerticalFlip(p=0.1),
            A.OneOf([
                A.MotionBlur(p=0.2),
                A.MedianBlur(blur_limit=3, p=0.1),
                A.Blur(blur_limit=3, p=0.1),
            ], p=0.2),
            A.OneOf([
                M.MyOpticalDistortion(p=0.3),
                M.MyGridDistortion(p=0.1),
            ], p=0.2),
            A.OneOf([
                A.IAASharpen(),
                A.IAAEmboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.Normalize(),
            M.MyToTensorV2(),
        ],
        additional_targets={
            'normal': 'normal',
            'depth': 'depth',
        }
    )

    img_transform = A.Compose(
        [
            M.MyLongestMaxSize(max_size=IMAGE_SIZE),
            M.MyPadIfNeeded(min_height=IMAGE_SIZE, min_width=IMAGE_SIZE, border_mode=cv2.BORDER_CONSTANT, value=0),
            A.Normalize(),
            M.MyToTensorV2(),
        ],
        additional_targets={
            'depth': 'depth'
        }
    )

    _, dataloader = create_dataloader("../bdataset_scene", "test.json", transform=my_transform)
    imgs, normals, depths = next(iter(dataloader))
    assert imgs.shape == (2, 3, 256, 256), f"dataset error {imgs.shape}"
    assert normals.shape == (2, 3, 256, 256), f"dataset error {normals.shape}"
    assert depths.shape == (2, 1, 256, 256), f"dataset error {depths.shape}"

    dataset = LoadImages(JSON, transform=img_transform)
    og_img, img, depth, path = next(iter(dataset))
    assert img.shape == (3, 256, 256), f"dataset error {img.shape}"
    assert depth.shape == (1, 256, 256), f"dataset error {depth.shape}"

    print("dataset ok")
