
from .transformer import TransformerColor, TransformerAffine

# Standard.
import copy
import logging
import os
import random

# Third party.
import cv2
import numpy as np
from third_party.colmap.scripts.python.read_write_model import (
    read_model, qvec2rotmat)
from third_party.colmap.scripts.python.read_dense import read_array
import h5py
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def get_dataset(dataset_info, start = 0, stride = 1,
                device = torch.device('cpu')):
    if dataset_info['type'].upper() == 'WEBIMAGES':
        return WebImageDataset(dataset_info, start, stride, device)
    elif dataset_info['type'].upper() == 'SFM':
        return SFMDataset(dataset_info, start, stride)
    else:
        logging.critical('Unknow dataset type {0}'.format(dataset.type))
        return None

class WebImageDataset():
    def __init__(self, dataset_info, start = 0, stride = 1,
                 device = torch.device('cpu')):
        logging.debug(
            'Prepare WEBIMAGES dataset {0}'.format(dataset_info['name']))
        self.root_path_image = dataset_info['root_path_image']
        self.root_path_keypoints = dataset_info['root_path_keypoints']
        self.image_type = dataset_info['image_type'].upper()
        self.device = device
        self.image_path_list = []
        self.keypoints_path_list = []
        # Load each image path and its keypoints path.
        for image_name in sorted(os.listdir(dataset_info['root_path_image'])):
            if image_name[-4:].upper() not in ['.JPG', '.PNG', '.PPM']:
                continue
            image_path = os.path.join(self.root_path_image, image_name)
            self.image_path_list.append(image_path)
            keypoints_path = os.path.join(
                self.root_path_keypoints, image_name + '.npz')
            self.keypoints_path_list.append(keypoints_path)

        self.image_path_list = self.image_path_list[start::stride]
        self.keypoints_path_list = self.keypoints_path_list[start::stride]

        # Color jitter transformer.
        self.transformer_color = TransformerColor()
        # Affine transformer.
        self.transformer_affine = TransformerAffine()

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        return self.image_path_list[index]

    def get_image_pair(self, index, num_refs = 1):
        if self.image_type in {'COLOR'}:
            img_bgr = cv2.imread(
                self.image_path_list[index], cv2.IMREAD_COLOR)
        elif self.image_type in {'BAYER_RG'}:
            img_bayer = cv2.imread(
                self.image_path_list[index], cv2.IMREAD_GRAYSCALE)
            img_bgr = cv2.cvtColor(img_bayer, cv2.COLOR_BAYER_RG2BGR)
        else:
            logging.error(
                'Can not read image with {0} format.'.format(
                    self.image_type))

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_rgb_color_jitter = self.transformer_color(img_rgb)
        img_target = cv2.cvtColor(img_rgb_color_jitter, cv2.COLOR_RGB2GRAY)
        img_target = TF.to_tensor(img_target).unsqueeze(0).to(
            self.device)

        img_refs = []
        grid_target2refs = []

        for index_ref in range(num_refs):
            # Transform and save the results.
            img_rgb_ref = self.transformer_color(img_rgb)
            outs = self.transformer_affine(img_rgb_ref)

            img_rgb_ref_affine = outs[0]
            img_gray_ref_affine = cv2.cvtColor(
                img_rgb_ref_affine, cv2.COLOR_RGB2GRAY)
            img_gray_ref_affine = TF.to_tensor(img_gray_ref_affine).to(
                self.device)
            img_refs.append(img_gray_ref_affine.unsqueeze(0))

            grid_target2ref = torch.tensor(outs[1]).unsqueeze(0)
            grid_target2refs.append(grid_target2ref)

        img_refs = torch.cat(img_refs, dim = 0).to(self.device)
        grid_target2refs = torch.cat(grid_target2refs, dim = 0
            ).to(torch.float).to(self.device)

        return (img_target, img_refs, grid_target2refs)

    def get_data_pair(self, index, num_refs = 1):
        # Load keypoints and score.
        outs = np.load(self.get_keypoints_path(index))
        keypoints_target = outs['keypoints_map']
        score_target = outs['score']
        keypoints_target = torch.tensor(keypoints_target
            ).unsqueeze(0).unsqueeze(0).to(self.device)
        score_target = torch.tensor(score_target
            ).unsqueeze(0).unsqueeze(0).to(self.device)

        outs = self.get_image_pair(index, num_refs)
        (img_target, img_refs, grid_target2refs) = outs

        keypoints_refs = []
        score_refs = []
        for index_ref in range(num_refs):
            keypoints_ref = F.grid_sample(
                keypoints_target.to(torch.float),
                grid_target2refs[index_ref:index_ref+1, :, :, :],
                mode = 'nearest')
            keypoints_refs.append(keypoints_ref.to(torch.long))

            score_ref = F.grid_sample(
                score_target,
                grid_target2refs[index_ref:index_ref+1, :, :, :],
                padding_mode = 'border')
            score_refs.append(score_ref)

        keypoints_refs = torch.cat(keypoints_refs, dim = 0).to(self.device)
        score_refs = torch.cat(score_refs, dim = 0).to(self.device)

        return (img_target, img_refs, keypoints_target, keypoints_refs,
            score_target, score_refs, grid_target2refs)

    def get_image_path(self, index):
        return self.image_path_list[index]

    def get_keypoints_path(self, index):
        return self.keypoints_path_list[index]

class SFMDataset():
    def __init__(self, dataset_info, start = 0, stride = 1):
        logging.debug('Prepare SFM dataset {0}'.format(dataset_info['name']))
        self.ratio_select_self = 0.5
        self.depth_epsilon = 10.
        # Load sfm data info.
        self.root_path_image = dataset_info['root_path_image']
        self.root_path_keypoints = dataset_info['root_path_keypoints']
        self.root_path_depth = dataset_info['root_path_depth']
        self.root_path_depth_clean = dataset_info['root_path_depth_clean']

        self.cameras, self.images, _ = read_model(
            path = dataset_info['root_path_sfm'], ext='.bin')

        self.indices = [i for i in self.cameras]
        #self.indices = self.indices[::10]

        self.indices = self.indices[start::stride]
        self.cameras = {idx:self.cameras[idx] for idx in self.indices}
        self.images = {idx:self.images[idx] for idx in self.indices}

        self.image_path_list = []
        self.keypoints_path_list = []
        self.depth_path_list = []
        self.depth_clean_path_list = []
        for index in self.indices:
            image_path = os.path.join(
                dataset_info['root_path_image'], self.images[index].name)
            self.image_path_list.append(image_path)
            keypoints_path = os.path.join(
                dataset_info['root_path_keypoints'],
                self.images[index].name + '.npz')
            self.keypoints_path_list.append(keypoints_path)
            depth_path = os.path.join(
                dataset_info['root_path_depth'],
                self.images[index].name + '.photometric.bin')
            self.depth_path_list.append(depth_path)
            depth_clean_path = os.path.join(
                dataset_info['root_path_depth_clean'],
                self.images[index].name.split('.')[0] + '.h5')
            self.depth_clean_path_list.append(depth_clean_path)

    def __len__(self):
        return len(self.image_path_list)

    def __getitem__(self, index):
        return self.image_path_list[index]

    def get_image_pair(self, index, num_refs = 1):
        img_target = self.get_image(index)
        img_refs = []
        correspondence_target2refs = []

        indexes_neighbour = self.get_neighbour_indexes(index, num_refs)

        for index_neighbour in indexes_neighbour:
            img_ref = self.get_image(index_neighbour)
            img_refs.append(img_ref)
            correspondence_target2ref = self.get_correspondence(
                index, index_neighbour)
            correspondence_target2refs.append(correspondence_target2ref)

        return (img_target, img_refs, correspondence_target2refs)

    def get_data_pair(self, index, num_refs = 1):
        img_target = self.get_image(index)
        outs = self.get_keypoints(index)
        keypoints_target, score_target = outs

        img_refs = []
        correspondence_target2refs = []
        keypoints_refs = []
        score_refs = []

        indexes_neighbour = self.get_neighbour_indexes(index, num_refs)

        for index_neighbour in indexes_neighbour:
            img_ref = self.get_image(index_neighbour)
            img_refs.append(img_ref)

            correspondence_target2ref = self.get_correspondence(
                index, index_neighbour)
            correspondence_target2refs.append(correspondence_target2ref)

            outs = self.get_keypoints(index_neighbour)
            keypoints_ref, score_ref = outs
            keypoints_refs.append(keypoints_ref)
            score_refs.append(score_ref)

        return (img_target, img_refs, keypoints_target, keypoints_refs,
            score_target, score_refs,
            correspondence_target2refs)

    def get_neighbour_indexes(self, index, num_refs):
        indexes = []
        for index_ref in range(num_refs):
            if random.random() < self.ratio_select_self:
                indexes.append(index)
            else:
                # Random select a neighbour.
                while True:
                    index_neighbour = random.randint(0, len(self) - 1)
                    if not index_neighbour == index:
                        break
                indexes.append(index_neighbour)
        return indexes

    def get_image(self, index):
        img = cv2.imread(
            self.image_path_list[index], cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def get_keypoints(self, index):
        outs = np.load(self.get_keypoints_path(index))
        keypoints = outs['keypoints_map']
        score = outs['score']
        return keypoints, score

    def get_correspondence(self, index0, index1):
        sfm_data0 = self.get_sfm_data(index0)
        img0 = self.get_image(index0)
        height0, width0, _ = img0.shape
        # Process the correspondence.
        if index0 == index1:
            correspondence_0_to_1 = np.zeros(
                [height0, width0, 2], dtype = np.int)
            for row in range(height0):
                correspondence_0_to_1[row, :, 0] = row
            for col in range(width0):
                correspondence_0_to_1[:, col, 1] = col
        else:
            depth0 = sfm_data0['depth_clean']
            depth0 = depth0[:height0, :width0]
            K0 = sfm_data0['K']
            R0 = sfm_data0['R']
            T0 = sfm_data0['T']
            rows0, cols0 = np.where(depth0 > 0)
            ys0 = rows0
            xs0 = cols0
            zs0 = depth0[rows0, cols0]
            u_xys0 = np.concatenate(
                [xs0, ys0, np.ones([xs0.shape[0]])], axis = 0).reshape([3, -1])
            n_xyzs0 = np.dot(np.linalg.inv(K0), u_xys0)
            n_xyzs0 = n_xyzs0 * zs0 / n_xyzs0[2, :]
            xyz0_world = np.dot(R0.T, n_xyzs0 - T0[:,None])

            sfm_data1 = self.get_sfm_data(index1)
            img1 = self.get_image(index1)
            height1, width1, _ = img1.shape
            depth1 = sfm_data1['depth_clean']
            height_depth1, width_depth1 = depth1.shape
            K1 = sfm_data1['K']
            R1 = sfm_data1['R']
            T1 = sfm_data1['T']

            n_xyzs1 = np.dot(R1, xyz0_world) + T1[:,None]
            depth1_pro = n_xyzs1[2, :]
            u_xys1 = np.dot(K1, n_xyzs1)
            z1 = u_xys1[2,:]
            u_xys1 = u_xys1 / z1

            rows1 = u_xys1[1,:]
            rows1 = rows1.astype(np.int)
            cols1 = u_xys1[0,:]
            cols1 = cols1.astype(np.int)

            valid_points = rows1 >= 0
            valid_points = valid_points & (rows1 < height1)
            valid_points = valid_points & (rows1 < height_depth1)
            valid_points = valid_points & (cols1 >= 0)
            valid_points = valid_points & (cols1 < width1)
            valid_points = valid_points & (cols1 < width_depth1 - 1)

            rows0 = rows0[valid_points]
            cols0 = cols0[valid_points]
            rows1 = rows1[valid_points]
            cols1 = cols1[valid_points]
            depth1_pro = depth1_pro[valid_points]

            depth1_common = depth1[rows1, cols1]
            valid_points = (
                np.abs(depth1_pro - depth1_common) < self.depth_epsilon)

            rows0 = rows0[valid_points]
            cols0 = cols0[valid_points]
            rows1 = rows1[valid_points]
            cols1 = cols1[valid_points]

            correspondence_0_to_1 = np.zeros(
                [height0, width0, 2], dtype = np.int)
            correspondence_0_to_1[rows0, cols0, 0] = rows1
            correspondence_0_to_1[rows0, cols0, 1] = cols1

        return correspondence_0_to_1

    def get_sfm_data(self, index):
        idx = self.indices[index]

        depth = read_array(self.depth_path_list[index])
        min_depth, max_depth = np.percentile(depth, [5, 95])
        depth[depth < min_depth] = min_depth
        depth[depth > max_depth] = max_depth

        # reformat data
        q = self.images[idx].qvec
        R = qvec2rotmat(q)
        T = self.images[idx].tvec
        p = self.images[idx].xys
        pars = self.cameras[idx].params
        K = np.array([[pars[0], 0, pars[2]], [0, pars[1], pars[3]], [0, 0, 1]])

        # get also the clean depth maps
        base = '.'.join(self.images[idx].name.split('.')[:-1])
        with h5py.File(self.depth_clean_path_list[index], 'r') as f:
            depth_clean = f['depth'][()]

        return {
            'depth': depth,
            'depth_clean': depth_clean,
            'K': K,
            'q': q,
            'R': R,
            'T': T,
            'xys': p} # x,y of keypoints

    def get_image_path(self, index):
        return self.image_path_list[index]

    def get_keypoints_path(self, index):
        return self.keypoints_path_list[index]

