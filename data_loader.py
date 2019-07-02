# -*- coding: utf-8 -*-

########################################################################################################
#   Dataset loader for img with csv file or MNIST.
#   
#   By Adam Mo
#   2/7/2019
########################################################################################################

from __future__ import print_function, division

import struct

import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from skimage import transform
# from skimage import io, transform
# import matplotlib.pyplot as plt
import pandas as pd
import os

import pdb

from math import sqrt

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")


class ImgDataset(Dataset):
    """All images with labels dataset."""

    def __init__(self, list_file, dir_name='', transform=None):
        """
        Args:
            list_file (string): Path to the csv file with annotations.
            dir_name (string): Path for directory of data
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data_lines = open(list_file).readlines()
        self.dir_name = dir_name
        self.transform = transform
        # print(len(self.image_lines))

    def __len__(self):
        return len(self.data_lines)

    def __getitem__(self, idx):
        data_line = self.data_lines[idx]

        data_line = data_line.replace('\n', '').replace('\r', '').split(',')

        id = int(data_line[0])
        label = int(data_line[1])
        # img = np.load(os.path.join(self.dir_name, data_line[2]))               # if RBG imgs
        img = np.load(os.path.join(self.dir_name, data_line[2]))[:, np.newaxis]  # if the GREY img has only ONE channel

        if self.transform:
            img = self.transform(img)

        data_dict = {
            'id': id, 'label': label, 'img': img
        }

        return data_dict


class MNISTDataset(Dataset):

    """Abandon."""

    def __init__(self, mnist_file, label_file, transform=None):
        """
        Args:
            mnist_file (string): Path to the mnist file.
            label_file (string): Path to the label file.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        with open(label_file, 'rb') as lbpath:
            magic, n = struct.unpack('>II', lbpath.read(8))
            labels = np.fromfile(lbpath, dtype=np.uint8)
        with open(mnist_file, 'rb') as imgpath:
            magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
            images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        self.transform = transform
        self.img = images
        self.lbl = labels

    def __len__(self):
        return len(self.lbl)

    def __getitem__(self, idx):
        img = np.reshape(self.img[idx], [28, 28])[:, :, np.newaxis]    # if the GREY img has only ONE channel
        label = self.lbl[idx]

        if self.transform:
            img = self.transform(img)

        data_dict = {
            'lbl': label, 'img': img
        }

        return data_dict


def test_for_img():
    """
    Test for the ImgDataset
    """
    batch_size = 5
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        # transforms.RandomCrop(32, padding=4),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])

    readpath = 'full_datafile.csv'

    trainset = ImgDataset(readpath, transform=transform_train, dir_name='')
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # print(trainloader)

    import matplotlib.pyplot as plt

    for batch_idx, datai in enumerate(trainloader):
        plt.imshow(datai['img'][0][0][11].numpy())
        plt.show()
        input()


def test_for_mnist():
    batch_size = 5
    # Data
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.ToTensor(),
    ])

    mnistpath = 'data/train-images.idx3-ubyte'
    labelpath = 'data/train-labels.idx1-ubyte'

    trainset = MNISTDataset(mnistpath, labelpath, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # print(trainloader)

    import matplotlib.pyplot as plt

    for batch_idx, datai in enumerate(trainloader):
        plt.imshow(datai['img'][0][0].numpy(), cmap=plt.cm.gray)
        plt.show()
        input()


if __name__ == '__main__':
    test_for_mnist()
