# -*- coding: utf-8 -*-
#
# Developed by Alex Jercan <jercan_alex27@yahoo.com>
#
# References:
#

import os
import json
from copy import copy

from torch.utils.data import Dataset, DataLoader
from util import load_depth, load_image, load_normal


def create_dataloader(dataset_root, json_path, batch_size=2, transform=None, workers=8, pin_memory=True, shuffle=True):
    dataset = BDataset(dataset_root, json_path, transform=transform)
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
        output_img_path = self.json_data[index]["output"]

        img = load_image(img_path)

        return img, output_img_path

    def __transform__(self, data):
        img, output_path = data
        og_img = copy(img)

        if self.transform is not None:
            augmentations = self.transform(image=img)
            img = augmentations["image"]

        return og_img, img, output_path


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
                M.MyIAAPiecewiseAffine(p=0.3),
            ], p=0.2),
            A.OneOf([
                A.IAASharpen(),
                A.IAAEmboss(),
                A.RandomBrightnessContrast(),
            ], p=0.3),
            A.Normalize(mean=0, std=1),
            M.MyToTensorV2(),
        ],
        additional_targets={
            'normal': 'normal',
            'depth': 'depth',
        }
    )

    img_transform = A.Compose(
        [
            A.Normalize(mean=0, std=1),
            M.MyToTensorV2(),
        ]
    )

    _, dataloader = create_dataloader("../bdataset", "test.json", transform=my_transform)
    imgs, normals, depths = next(iter(dataloader))
    assert imgs.shape == (2, 3, 256, 256), f"dataset error {imgs.shape}"
    assert normals.shape == (2, 3, 256, 256), f"dataset error {normals.shape}"
    assert depths.shape == (2, 1, 256, 256), f"dataset error {depths.shape}"

    dataset = LoadImages(JSON, transform=img_transform)
    og_img, img, path = next(iter(dataset))
    assert og_img.shape == (256, 256, 3), f"dataset error {og_img.shape}"
    assert img.shape == (3, 256, 256), f"dataset error {img.shape}"

    print("dataset ok")
