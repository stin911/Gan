import cv2
import numpy as np
import pandas as pd
import tifffile as tifi
import torch.nn as nn
import torch
import os


def weights_init(m):
    """

    :param m:
    :return:
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def read_image(path):
    """
    :param path:
    :return:
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    return img


def show_image(image):
    """
    :param image:
    :return:
    """
    cv2.imshow('image', image)
    cv2.waitKey(10000)
    cv2.destroyAllWindows()


def label_date(data):
    """
    :param data:
    :return:
    """
    if data == "LAA":
        return 0
    else:
        return 1


def just_to():
    a = pd.read_csv(".//Data//train.csv")
    print(a)
    list_p = []
    for e in os.listdir("./Data/IMG/"):
        print(e[:-4])
        list_p.append(e[:-4])
    data = a.loc[a["image_id"].isin(list_p)]
    data.to_csv("./Data/train_part.csv")


def getmax_min_images(folder):
    list_dim = []
    for e in os.listdir(folder):
        image = read_image(folder + e)
        list_dim.append(np.shape(image))
    print(max(list_dim), min(list_dim))


def show_image_dataloader(dataloader):
    for e in dataloader:
        a = torch.permute(e, (1, 2, 0))
        a = a.detach().cpu().numpy()
        show_image(a)
