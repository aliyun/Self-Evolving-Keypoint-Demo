
import utils

import glob
import math
import os
import random
import sys

import numpy as np
from PIL import Image
import torch
import torch.utils.data
import torchvision
import torchvision.transforms.functional as TF

class LocalFeaturePair(torch.utils.data.Dataset):
    # This class load each image along with keypoints in it from dataset.
    # Then we transform the input image twice to get a pair of images.
    # The corresponding keypoints are also calculated.
    # We return the two transformed images with keypoints coordinates at last.
    def __init__(
        self, root_path_img, root_path_keypoint,
        new_height = 240, new_width = 320,
        down_ratio = 2, map_size = 640):
        super(LocalFeaturePair, self).__init__()
        #print('Read image from ' + root_path_img +
        #      ', read keypoints from ' + root_path_keypoint + '.')
        self.img_path_list = []
        self.keypoints_path_list = []
        # load each image path and its label path
        for name in os.listdir(root_path_img):
            if name[-4:] == '.jpg':
                img_path = os.path.join(root_path_img, name)
                self.img_path_list.append(img_path)
                keypoints_path = os.path.join(
                    root_path_keypoint, name[:-4] + '.npz')
                self.keypoints_path_list.append(keypoints_path)

        # Prepare transform.
        self.new_height = new_height
        self.new_width = new_width
        self.down_ratio = down_ratio
        self.map_size = map_size

        self.degrees = (-20, 20)
        self.translate = (0.04, 0.04)
        self.scale = (0.8, 1.2)
        self.shear = (-20, 20)
        self.resample_map = Image.NEAREST
        self.resample_img = Image.BILINEAR
        self.fillcolor = 0

        self.transformer_color_jitter = torchvision.transforms.ColorJitter(
            brightness=(0.6, 1.4), contrast=(0.6, 1.4),
            saturation=(0.6, 1.4), hue=(-0.2, 0.2))

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        keypoints_path = self.keypoints_path_list[index]
        # Load img.
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        # Load keypoints.
        keypoints_data = np.load(keypoints_path)
        score_init = torch.from_numpy(
            keypoints_data['score'].astype(np.float32))
        keypoints_map = keypoints_data['keypoints_map'].astype(np.uint8) * 255
        keypoints_map = TF.to_pil_image(keypoints_map)

        # Random affine the image and keypoints_map to get data 0.
        affine_params_keypoint = utils.get_affine_params(
            degrees=self.degrees, translate=self.translate,
            scale_ranges=self.scale, shears=self.shear,
            img_size=keypoints_map.size, random_method='uniform')
        keypoints_map0 = TF.affine(
            keypoints_map, *affine_params_keypoint, resample=self.resample_map,
            fillcolor=self.fillcolor)

        center = (keypoints_map.size[0] * 0.5 + 0.5,
                  keypoints_map.size[1] * 0.5 + 0.5)
        matrix0 = utils.get_affine_matrix(center, *affine_params_keypoint)

        affine_params_img = (
            affine_params_keypoint[0],
            (affine_params_keypoint[1][0] * self.down_ratio,
             affine_params_keypoint[1][1] * self.down_ratio),
            affine_params_keypoint[2], affine_params_keypoint[3])
        img0 = TF.affine(
            img, *affine_params_img, resample=self.resample_img,
            fillcolor=self.fillcolor)

        # Pad img & keypoints_map if it is needed.
        if img0.size[0] < self.new_width:
            img0 = TF.pad(
                img0, (self.new_width - img0.size[0], 0),
                fill = 0, padding_mode = 'constant')
            keypoints_map0 = TF.pad(
                keypoints_map0,
                (int(self.new_width / self.down_ratio) - keypoints_map0.size[0],
                 0),
                fill = 0, padding_mode = 'constant')
        if img0.size[1] < self.new_height:
            img0 = TF.pad(
                img0, (0, self.new_height - img0.size[1]),
                fill = 0, padding_mode = 'constant')
            keypoints_map0 = TF.pad(
                keypoints_map0,
                (0, int(self.new_height / self.down_ratio) -
                    keypoints_map0.size[1]),
                fill = 0, padding_mode = 'constant')
        height_start = random.randint(
            0, int(img0.size[1]-self.new_height) / self.down_ratio)
        width_start = random.randint(
            0, int(img0.size[0]-self.new_width) / self.down_ratio)
        img0 = TF.crop(
            img0,
            height_start * self.down_ratio, width_start * self.down_ratio,
            self.new_height, self.new_width)
        keypoints_map0 = TF.crop(
            keypoints_map0, height_start, width_start,
            int(self.new_height / self.down_ratio),
            int(self.new_width / self.down_ratio))
        matrix0[0,2] -= height_start
        matrix0[1,2] -= width_start

        img0 = self.transformer_color_jitter(img0)
        img0 = img0.convert('L')
        img0 = TF.to_tensor(img0)

        if random.random() < 0.2:
            img0 = utils.random_blur(img0)

        # Random affine the image and keypoints_map to get data 1.
        affine_params_keypoint = utils.get_affine_params(
            degrees=self.degrees, translate=self.translate,
            scale_ranges=self.scale, shears=self.shear,
            img_size=keypoints_map.size, random_method='uniform')
        keypoints_map1 = TF.affine(
            keypoints_map, *affine_params_keypoint, resample=self.resample_map,
            fillcolor=self.fillcolor)

        center = (keypoints_map.size[0] * 0.5 + 0.5,
                  keypoints_map.size[1] * 0.5 + 0.5)
        matrix1 = utils.get_affine_matrix(center, *affine_params_keypoint)

        affine_params_img = (
            affine_params_keypoint[0],
            (affine_params_keypoint[1][0] * self.down_ratio,
             affine_params_keypoint[1][1] * self.down_ratio),
            affine_params_keypoint[2], affine_params_keypoint[3])
        img1 = TF.affine(
            img, *affine_params_img, resample=self.resample_img,
            fillcolor=self.fillcolor)

        # Pad img & keypoints_map if it is needed.
        if img1.size[0] < self.new_width:
            img1 = TF.pad(
                img1, (0, 0, self.new_width - img1.size[0], 0),
                fill = 0, padding_mode = 'constant')
            keypoints_map1 = TF.pad(
                keypoints_map1,
                (0, 0,
                 int(self.new_width / self.down_ratio) - keypoints_map1.size[0], 0),
                fill = 0, padding_mode = 'constant')
        if img1.size[1] < self.new_height:
            img1 = TF.pad(
                img1, (0, 0, 0, self.new_height - img1.size[1]),
                fill = 0, padding_mode = 'constant')
            keypoints_map1 = TF.pad(
                keypoints_map1,
                (0, 0, 0,
                 int(self.new_height / self.down_ratio) - keypoints_map1.size[1]),
                fill = 0, padding_mode = 'constant')
        height_start = random.randint(
            0, int(img1.size[1]-self.new_height) / self.down_ratio)
        width_start = random.randint(
            0, int(img1.size[0]-self.new_width) / self.down_ratio)
        img1 = TF.crop(
            img1,
            height_start * self.down_ratio, width_start * self.down_ratio,
            self.new_height, self.new_width)
        keypoints_map1 = TF.crop(
            keypoints_map1, height_start, width_start,
            int(self.new_height / self.down_ratio),
            int(self.new_width / self.down_ratio))
        matrix1[0,2] -= height_start
        matrix1[1,2] -= width_start

        img1 = self.transformer_color_jitter(img1)
        img1 = img1.convert('L')
        img1 = TF.to_tensor(img1)

        if random.random() < 0.2:
            img1 = utils.random_blur(img1)

        keypoints_map = TF.pad(
            keypoints_map,
            (0, 0, self.map_size - keypoints_map.size[0],
             self.map_size - keypoints_map.size[1]),
            fill = 0, padding_mode = 'constant')
        keypoints_map = TF.to_tensor(keypoints_map).squeeze()
        score = torch.zeros([self.map_size, self.map_size])
        score[:score_init.shape[0], :score_init.shape[1]] = score_init

        return img0, img1, keypoints_map, score, matrix0, matrix1

