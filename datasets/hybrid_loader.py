
# First party.
from .dataset import get_dataset

# Standard.
import logging
import random

# Third party.
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

class HybridLoader(torch.utils.data.Dataset):
    # This class load each image from datasets that can be web images,
    # sfm data, or their combination.
    def __init__(self, dataset_info_list, new_height = 240, new_width = 320,
                 index_em = 0, sub_set = 1):
        super(HybridLoader, self).__init__()
        logging.debug('Init hybrid datasets.')
        self.new_height = new_height
        self.new_width = new_width
        self.sample_list = []
        self.datasets = []
        # Process each dataset according to its type.
        for dataset_info in dataset_info_list:
            dataset = get_dataset(
                dataset_info,
                start = index_em % sub_set,
                stride = sub_set)
            idx_dataset = len(self.datasets)
            for idx_image in range(len(dataset)):
                self.sample_list.append([idx_dataset, idx_image])
            self.datasets.append(dataset)

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, index):
        idx_dataset, idx_image = self.sample_list[index]
        # Get data pair.
        outs = self.datasets[idx_dataset].get_data_pair(idx_image, num_refs = 1)

        (img0, img1, keypoints_map0, keypoints_map1,
            score_map0, score_map1, grid_target2ref) = outs

        img0 = img0[0, :, :, :]
        img1 = img1[0, :, :, :]
        keypoints_map0 = keypoints_map0[0, :, :, :]
        keypoints_map1 = keypoints_map1[0, :, :, :]
        score_map0 = score_map0[0, :, :, :]
        score_map1 = score_map1[0, :, :, :]
        grid_target2ref = grid_target2ref[0, :, :, :]

        _, height, width = img0.shape

        # Pad data if they are two small.
        if width < self.new_width:
            img0 = F.pad(img0, (0, self.new_width - width))
            keypoints_map0 = F.pad(keypoints_map0, (0, self.new_width - width))
            score_map0 = F.pad(score_map0, (0, self.new_width - width))

            img1 = F.pad(img1, (0, self.new_width - width))
            keypoints_map1 = F.pad(keypoints_map1, (0, self.new_width - width))
            score_map1 = F.pad(score_map1, (0, self.new_width - width))

            grid_target2ref[:, :, 0] = (
                (grid_target2ref[:, :, 0] + 1.) * width / self.new_width - 1)
            grid_target2ref = F.pad(
                grid_target2ref, (0, 0, 0, self.new_width - width), value = 1)

            width = self.new_width

        if height < self.new_height:
            img0 = F.pad(img0, (0, 0, 0, self.new_height - height))
            keypoints_map0 = F.pad(
                keypoints_map0, (0, 0, 0, self.new_height - height))
            score_map0 = F.pad(
                score_map0, (0, 0, 0, self.new_height - height))

            img1 = F.pad(img1, (0, 0, 0, self.new_height - height))
            keypoints_map1 = F.pad(
                keypoints_map1, (0, 0, 0, self.new_height - height))
            score_map1 = F.pad(
                score_map1, (0, 0, 0, self.new_height - height))

            grid_target2ref[:, :, 1] = (
                (grid_target2ref[:, :, 1] + 1.) * height / self.new_height - 1)
            grid_target2ref = F.pad(
                grid_target2ref,
                (0, 0, 0, 0, 0, self.new_height - height),
                value = 1)

            height = self.new_height

        # Random crop the data.
        top = random.randint(0, height - self.new_height)
        left = random.randint(0, width - self.new_width)

        img0_crop = img0[
            :, top : top + self.new_height, left : left + self.new_width]
        keypoints_map0_crop = keypoints_map0[
            :, top : top + self.new_height, left : left + self.new_width]
        score_map0_crop = score_map0[
            :, top : top + self.new_height, left : left + self.new_width]

        img1_crop = img1[
            :, top : top + self.new_height, left : left + self.new_width]
        keypoints_map1_crop = keypoints_map1[
            :, top : top + self.new_height, left : left + self.new_width]
        score_map1_crop = score_map1[
            :, top : top + self.new_height, left : left + self.new_width]

        grid_target2ref[:, :, 0] = (
            ((grid_target2ref[:, :, 0] + 1.) / 2. * width - left) /
            self.new_width * 2. - 1.)
        grid_target2ref[:, :, 1] = (
            ((grid_target2ref[:, :, 1] + 1.) / 2. * height - top) /
            self.new_height * 2. - 1.)
        grid_target2ref_crop = grid_target2ref[
            top : top + self.new_height, left : left + self.new_width, :]

        return (img0_crop, img1_crop,
            keypoints_map0_crop, keypoints_map1_crop,
            score_map0_crop, score_map1_crop,
            grid_target2ref_crop)

if __name__ == '__main__':
    print('Test HybridLoader ... ')

