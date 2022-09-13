import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from src.utils.utility import read_image
import tifffile as tifi
from torchvision.transforms import ToTensor
import numpy as np


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        image_name = self.img_labels.iloc[idx, 1]
        image = read_image("./Data/IMG/" + image_name)
        image = (image - np.min(image)) / (np.max(image) - np.min(image))
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image is None:
            print(image_name)
        if self.transform:
            image = self.transform(image)

        return image
