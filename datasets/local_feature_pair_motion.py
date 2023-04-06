
import glob
import math
import os
import random
import sys

import cv2
import numpy as np
from PIL import Image
import torch
import torch.utils.data
import torchvision
import torchvision.transforms.functional as TF

sys.path.append('..')
import utils

class LocalFeaturePairMotion(torch.utils.data.Dataset):
    # This class load each image along with keypoints in it from dataset.
    # Then we transform the input image twice to get a pair of images. The corresponding keypoints are also calculated.
    # We return the two transformed images with keypoints coordinates at last.
    def __init__(self, root_path_img, root_path_keypoint, root_path_label,
            new_height = 240, new_width = 320, down_ratio = 2, map_size = 1024):
        super(LocalFeaturePairMotion, self).__init__()
        #print('Read image from ' + root_path_img + ', read keypoints from ' + root_path_keypoint + '.')
        self.img_path_list = []
        self.keypoints_path_list = []
        self.label_path_list = []
        # load each image path and its label path
        for name in os.listdir(root_path_img):
            if name[-4:] == '.png':
                img_path = os.path.join(root_path_img, name)
                self.img_path_list.append(img_path)
                keypoints_path = os.path.join(root_path_keypoint, name[:-4] + '.npz')
                self.keypoints_path_list.append(keypoints_path)
                label_path = os.path.join(root_path_label, name[:-15] + 'gtFine_labelIds.png')
                self.label_path_list.append(label_path)

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
        self.resample_label = Image.NEAREST
        self.fillcolor = 0

        self.transformer_color_jitter = torchvision.transforms.ColorJitter(
                brightness=(0.6, 1.4), contrast=(0.6, 1.4),
                saturation=(0.6, 1.4), hue=(-0.2, 0.2))

    def __len__(self):
        return len(self.img_path_list)

    def __getitem__(self, index):
        img_path = self.img_path_list[index]
        keypoints_path = self.keypoints_path_list[index]
        label_path = self.label_path_list[index]
        # Load img.
        with open(img_path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        # Load keypoints.
        keypoints_data = np.load(keypoints_path)
        score_init = torch.from_numpy(keypoints_data['score'].astype(np.float32))
        keypoints_map = keypoints_data['keypoints_map'].astype(np.uint8) * 255
        keypoints_map = TF.to_pil_image(keypoints_map)
        # Load label.
        with open(label_path, 'rb') as f:
            label = Image.open(f)
            label = label.convert('L')

        # Random affine the image and keypoints_map to get data 0.
        affine_params_keypoint = utils.get_affine_params(
                degrees=self.degrees, translate=self.translate, scale_ranges=self.scale,
                shears=self.shear, img_size=keypoints_map.size, random_method='uniform')
        keypoints_map0 = TF.affine(keypoints_map, *affine_params_keypoint, resample=self.resample_map,
                                  fillcolor=self.fillcolor)

        center = (keypoints_map.size[0] * 0.5 + 0.5, keypoints_map.size[1] * 0.5 + 0.5)
        matrix0 = utils.get_affine_matrix(center, *affine_params_keypoint)

        affine_params_img = (affine_params_keypoint[0],
                (affine_params_keypoint[1][0] * self.down_ratio, affine_params_keypoint[1][1] * self.down_ratio),
                affine_params_keypoint[2], affine_params_keypoint[3])
        img0 = TF.affine(img, *affine_params_img, resample=self.resample_img, fillcolor=self.fillcolor)
        label0 = TF.affine(label, *affine_params_img, resample=self.resample_label, fillcolor=self.fillcolor)

        # Pad img & keypoints_map if it is needed.
        if img0.size[0] < self.new_width:
            img0 = TF.pad(img0, (self.new_width - img0.size[0], 0), fill = 0, padding_mode = 'constant')
            keypoints_map0 = TF.pad(keypoints_map0, (int(self.new_width / self.down_ratio) - keypoints_map0.size[0], 0),
                                  fill = 0, padding_mode = 'constant')
            label0 = TF.pad(label0, (self.new_width - label0.size[0], 0), fill = 0, padding_mode = 'constant')
        if img0.size[1] < self.new_height:
            img0 = TF.pad(img0, (0, self.new_height - img0.size[1]), fill = 0, padding_mode = 'constant')
            keypoints_map0 = TF.pad(keypoints_map0, (0, int(self.new_height / self.down_ratio) - keypoints_map0.size[1]),
                                  fill = 0, padding_mode = 'constant')
            label0 = TF.pad(label0, (0, self.new_height - label0.size[1]), fill = 0, padding_mode = 'constant')
        height_start = random.randint(0, int(img0.size[1]-self.new_height) / self.down_ratio)
        width_start = random.randint(0, int(img0.size[0]-self.new_width) / self.down_ratio)
        img0 = TF.crop(img0, height_start * self.down_ratio, width_start * self.down_ratio,
                      self.new_height, self.new_width)
        keypoints_map0 = TF.crop(keypoints_map0, height_start, width_start,
                                int(self.new_height / self.down_ratio),
                                int(self.new_width / self.down_ratio))
        label0 = TF.crop(label0, height_start * self.down_ratio, width_start * self.down_ratio,
                      self.new_height, self.new_width)
        matrix0[0,2] -= height_start
        matrix0[1,2] -= width_start

        img0 = self.transformer_color_jitter(img0)
        img0 = img0.convert('L')
        img0 = TF.to_tensor(img0)

        label0 = TF.to_tensor(label0)
        label0 = label0 * 255.
        label0 = label0.type(torch.long)
        label0 = self.transform_label(label0)

        if random.random() < 0.2:
            img0 = utils.random_blur(img0)

        # Random affine the image and keypoints_map to get data 1.
        affine_params_keypoint = utils.get_affine_params(
                degrees=self.degrees, translate=self.translate, scale_ranges=self.scale,
                shears=self.shear, img_size=keypoints_map.size, random_method='uniform')
        keypoints_map1 = TF.affine(keypoints_map, *affine_params_keypoint, resample=self.resample_map,
                                  fillcolor=self.fillcolor)

        center = (keypoints_map.size[0] * 0.5 + 0.5, keypoints_map.size[1] * 0.5 + 0.5)
        matrix1 = utils.get_affine_matrix(center, *affine_params_keypoint)

        affine_params_img = (affine_params_keypoint[0],
                (affine_params_keypoint[1][0] * self.down_ratio, affine_params_keypoint[1][1] * self.down_ratio),
                affine_params_keypoint[2], affine_params_keypoint[3])
        img1 = TF.affine(img, *affine_params_img, resample=self.resample_img, fillcolor=self.fillcolor)
        label1 = TF.affine(label, *affine_params_img, resample=self.resample_label, fillcolor=self.fillcolor)

        # Pad img & keypoints_map if it is needed.
        if img1.size[0] < self.new_width:
            img1 = TF.pad(img1, (0, 0, self.new_width - img1.size[0], 0), fill = 0, padding_mode = 'constant')
            keypoints_map1 = TF.pad(keypoints_map1, (0, 0, int(self.new_width / self.down_ratio) - keypoints_map1.size[0], 0),
                                  fill = 0, padding_mode = 'constant')
            label1 = TF.pad(label1, (self.new_width - label1.size[0], 0), fill = 0, padding_mode = 'constant')
        if img1.size[1] < self.new_height:
            img1 = TF.pad(img1, (0, 0, 0, self.new_height - img1.size[1]), fill = 0, padding_mode = 'constant')
            keypoints_map1 = TF.pad(keypoints_map1, (0, 0, 0, int(self.new_height / self.down_ratio) - keypoints_map1.size[1]),
                                  fill = 0, padding_mode = 'constant')
            label1 = TF.pad(label1, (0, self.new_height - label1.size[1]), fill = 0, padding_mode = 'constant')
        height_start = random.randint(0, int(img1.size[1]-self.new_height) / self.down_ratio)
        width_start = random.randint(0, int(img1.size[0]-self.new_width) / self.down_ratio)
        img1 = TF.crop(img1, height_start * self.down_ratio, width_start * self.down_ratio,
                      self.new_height, self.new_width)
        keypoints_map1 = TF.crop(keypoints_map1, height_start, width_start,
                                int(self.new_height / self.down_ratio),
                                int(self.new_width / self.down_ratio))
        label1 = TF.crop(label1, height_start * self.down_ratio, width_start * self.down_ratio,
                      self.new_height, self.new_width)
        matrix1[0,2] -= height_start
        matrix1[1,2] -= width_start

        img1 = self.transformer_color_jitter(img1)
        img1 = img1.convert('L')
        img1 = TF.to_tensor(img1)

        label1 = TF.to_tensor(label1)
        label1 = label1 * 255.
        label1 = label1.type(torch.long)
        label1 = self.transform_label(label1)

        if random.random() < 0.2:
            img1 = utils.random_blur(img1)

        keypoints_map = TF.pad(keypoints_map, (0, 0, self.map_size - keypoints_map.size[0], self.map_size - keypoints_map.size[1]),
                fill = 0, padding_mode = 'constant')
        keypoints_map = TF.to_tensor(keypoints_map).squeeze()
        score = torch.zeros([self.map_size, self.map_size])
        score[:score_init.shape[0], :score_init.shape[1]] = score_init

        return img0, img1, keypoints_map, score, label0, label1, matrix0, matrix1

    def transform_label(self, label):
        # transform semantic label to motion attribute label
        # process sky/unstable, static(static, long-term static), moving(short-term static, moving), in turn

        # init label_final
        label_final = torch.zeros(label.shape, dtype=label.dtype)
        label_final[:] = 0  # 0: unlabeled

        # process sky/unstable
        label_final[label == 23] = 1  # sky/unstable

        # process static(static, long-term static)
        label_static = torch.zeros(label.shape, dtype=label.dtype)
        label_static[:] = 0
        label_static[label == 4] = 1  # static
        label_static[label == 6] = 1  # ground
        label_static[label == 7] = 1  # road
        label_static[label == 8] = 1  # sidewalk
        label_static[label == 9] = 1  # parking
        label_static[label == 10] = 1  # rail track
        label_static[label == 11] = 1  # building
        label_static[label == 12] = 1  # wall
        label_static[label == 13] = 1  # fence
        label_static[label == 14] = 1  # guard rail
        label_static[label == 15] = 1  # bridge
        label_static[label == 16] = 1  # tunnel
        label_static[label == 17] = 1  # pole
        label_static[label == 18] = 1  # polegroup
        label_static[label == 19] = 1  # traffic light
        label_static[label == 20] = 1  # traffic sign
        label_static[label == 21] = 1  # vegetation
        label_static[label == 22] = 1  # terrain
        label_final[label_static == 1] = 2

        # process moving(short-term static, moving)
        label_moving = torch.zeros(label.shape, dtype=label.dtype)
        label_moving[:] = 0
        label_moving[label == 1] = 1  # ego vehicle
        label_moving[label == 2] = 1  # rectification border
        label_moving[label == 3] = 1  # out of roi
        label_moving[label == 5] = 1  # dynamic
        label_moving[label == 24] = 1  # person
        label_moving[label == 25] = 1  # rider
        label_moving[label == 26] = 1  # car
        label_moving[label == 27] = 1  # truck
        label_moving[label == 28] = 1  # bus
        label_moving[label == 29] = 1  # caravan
        label_moving[label == 30] = 1  # trailer
        label_moving[label == 31] = 1  # train
        label_moving[label == 32] = 1  # motorcycle
        label_moving[label == 33] = 1  # bicycle
        #label_moving = cv2.dilate(label_moving, kernel = 3, iterations = 1)
        label_final[label_moving == 1] = 3

        return label_final

