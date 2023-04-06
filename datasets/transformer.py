
# First party.
import utils

# Standard party.
import logging

# Third party.
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF

# ColorTransformer.
class TransformerColor(object):
    def __init__(self):
        self.transformer_color_jitter = torchvision.transforms.ColorJitter(
            brightness=(0.6, 1.4), contrast=(0.6, 1.4),
            saturation=(0.6, 1.4), hue=(-0.2, 0.2))
        return

    def __call__(self, img_in):
        img_out = TF.to_pil_image(img_in)
        img_out = self.transformer_color_jitter(img_out)
        img_out = np.array(img_out)
        return img_out

# AffineTransformer.
class TransformerAffine(object):
    def __init__(self):
        self.degrees = (-40, 40)
        self.translate = (0.04, 0.04)
        self.scale = (0.7, 1.4)
        self.shear = (-40, 40)
        self.random_method = 'uniform'

        return

    def __call__(self, img_in):
        if len(img_in.shape) == 2:
            img = torch.from_numpy(img_in).unsqueeze(0).unsqueeze(0)
        elif len(img_in.shape) == 3:
            img = torch.from_numpy(img_in).unsqueeze(0).permute([0, 3, 1, 2])
        else:
            logging.error('Wrong image with shape {0}'.format(img_in.shape))

        num, channel, height, width = img.shape
        img = img.to(torch.double)

        affine_params_img = utils.get_affine_params(
            degrees = self.degrees,
            translate = self.translate,
            scale_ranges = self.scale,
            shears = self.shear,
            random_method = self.random_method)

        theta = utils.get_affine_matrix_theta(*affine_params_img)
        theta_inverse = utils.get_affine_matrix_theta_inverse(
            *affine_params_img)

        theta = torch.from_numpy(theta).unsqueeze(0)
        theta_inverse = torch.from_numpy(theta_inverse).unsqueeze(0)

        grid10 = F.affine_grid(theta, torch.Size((num, channel, height, width)))
        grid01 = F.affine_grid(
            theta_inverse, torch.Size((num, channel, height, width)))

        img_affined = F.grid_sample(img, grid10, padding_mode = 'zeros')

        if len(img_in.shape) == 2:
            img_out = img_affined.detach().cpu().squeeze().numpy()
        elif len(img_in.shape) == 3:
            img_out = img_affined.detach().cpu().squeeze().permute(
                [1, 2, 0]).numpy()
        img_out = img_out.astype(np.uint8)

        grid01 = grid01.detach().cpu().squeeze().numpy().copy()
        grid10 = grid10.detach().cpu().squeeze().numpy().copy()

        theta = theta.detach().cpu().squeeze().numpy()
        theta_inverse = theta_inverse.detach().cpu().squeeze().numpy()

        # grid01: Height x Width x 2, last dim, 0: col, 1: row

        return (img_out, grid01, grid10, theta, theta_inverse)

if __name__ == '__main__':
    print('Test Transformer ... ')

