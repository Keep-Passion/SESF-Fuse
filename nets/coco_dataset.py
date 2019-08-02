import os
import random
import cv2
import numpy as np
import PIL.Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from nets.nets_utility import *
import torchvision.transforms.functional as F


class COCODataset(Dataset):
    def __init__(self, input_dir, crop_size=256, transform=None, need_crop=False, need_augment=False):
        self._images_basename = os.listdir(input_dir)
        if '.ipynb_checkpoints' in self._images_basename:
            self._images_basename.remove('.ipynb_checkpoints')
        self._images_address = [os.path.join(input_dir, item) for item in sorted(self._images_basename)]
        self._crop_size = crop_size
        self._transform = transform
        self._origin_transform = transforms.Compose([
            transforms.ToTensor()
        ])
        self._need_crop = need_crop
        self._need_augment = need_augment

    def __len__(self):
        return len(self._images_address)

    def __getitem__(self, idx):
        image = cv2.imread(self._images_address[idx], 0) / 255.0
        image = cv2.resize(image, (256, 256))  # 调整训练图像尺寸
        if self._need_crop:
            roi_image_np = self._random_crop(image)
        else:
            roi_image_np = image
        roi_image_pil = self._rand_augment(roi_image_np)
        if self._transform is not None:
            roi_image_tensor = self._transform(roi_image_pil)
        else:
            roi_image_tensor = self._origin_transform(roi_image_pil)
        return roi_image_tensor

    def _rand_augment(self, image):
        image_pil = PIL.Image.fromarray(image.astype(np.float32))
        if self._need_augment:
            image_pil = self._rand_horizontal_flip(image_pil)
        return image_pil

    def _rand_rotated(self, image_pil):
        rotate_angle = random.choice([0, 90, 180, 270])
        image_pil = F.rotate(image_pil, rotate_angle, expand=True)
        return image_pil

    def _rand_horizontal_flip(self, image_pil):
        # 0.5的概率水平翻转
        if random.random() < 0.5:
            image_pil = F.hflip(image_pil)
        return image_pil

    def _rand_vertical_flip(self, image_pil):
        # 0.5的概率水平翻转
        if random.random() < 0.5:
            image_pil = F.vflip(image_pil)
        return image_pil

    def _random_crop(self, image_np):
        h, w = image_np.shape[:2]
        start_row = random.randint(0, h - self._crop_size)
        start_col = random.randint(0, w - self._crop_size)
        roi_image_np = image_np[start_row: start_row + self._crop_size, start_col: start_col + self._crop_size]
        return roi_image_np


if __name__ == "__main__":
    batch_size = 3
    gpu_device = "cuda:0"
    shuffle = True
    # address
    project_address = os.getcwd()
    train_dir = os.path.join(os.path.join(os.path.join(project_address, "datasets"), "used_for_nets"), "litt_train2017")
    val_dir = os.path.join(os.path.join(os.path.join(project_address, "datasets"), "used_for_nets"), "litt_val2017")

    print('train_dir', train_dir)
    # datasets
    data_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.4500517361627943], [0.26465333914691797])
    ])

    image_datasets = {}
    image_datasets['train'] = COCODataset(train_dir, transform=data_transforms)
    image_datasets['val'] = COCODataset(val_dir, transform=data_transforms)

    dataloaders = {}
    dataloaders['train'] = DataLoader(
        image_datasets['train'],
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=1
    )
    dataloaders['val'] = DataLoader(
        image_datasets['val'],
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=1
    )
    datasets_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    print('datasets_size', datasets_sizes)