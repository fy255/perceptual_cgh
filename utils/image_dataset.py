import os
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import cv2
import utils


class imageDataset(Dataset):
    def __init__(self, imgs_dir, channel, image_res, homography_res):
        self.imgs_dir = imgs_dir
        self.channel = channel
        self.image_res = image_res
        self.homography_res = homography_res
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]

    @classmethod
    def im2Amp(cls, image, channel):
        image = utils.img_linearize(image, channel)
        img_nd = np.sqrt(image)  # to amplitude
        img_nd = np.transpose(img_nd, axes=(2, 0, 1))  # HWC to CHW
        return img_nd

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        fileName = self.ids[i]
        idx = self.ids[i] + '.*'
        img_file_path = os.path.join(self.imgs_dir, idx)
        img_file = glob(img_file_path)
        img = cv2.imread(img_file[0])
        img = self.im2Amp(img, self.channel)
        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'image_name': fileName
        }


class holoDataset(Dataset):
    def __init__(self, imgs_dir, image_res, homography_res):
        self.imgs_dir = imgs_dir
        self.image_res = image_res
        self.homography_res = homography_res
        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]

    @classmethod
    def holo2Angle(cls, image):
        img_1d = image[..., 0]
        img_1d_float = img_1d.astype(np.float64) / 255 * 2 * np.pi - np.pi
        return img_1d_float, img_1d

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        fileName = self.ids[i]
        idx = self.ids[i] + '.*'
        img_file_path = os.path.join(self.imgs_dir, idx)
        img_file = glob(img_file_path)
        img = cv2.imread(img_file[0])
        holo_angle, holo = self.holo2Angle(img)
        return {
            'holo_angle': torch.from_numpy(holo_angle).type(torch.FloatTensor),
            'holo': holo
        }
