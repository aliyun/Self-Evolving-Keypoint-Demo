
# First party.
from datasets import PrefetchReader, TransformerColor, TransformerAffine
from utils import (
    get_affine_params, get_affine_matrix, non_maximum_suppress, random_blur)

# Standard.
import logging
import os
import random

# Third party.
import cv2
import numpy as np
from PIL import Image
from scipy.spatial.distance import squareform
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF

def calculate_mean_ratio_local(
    descriptor0, descriptors1, grids_01, idx_near = 1, radius = 3, stride = 2):
    # Calculate each ratio and their mean.
    _, dim, height, width = descriptor0.shape
    num_refs, _, _, _ = descriptors1.shape
    ratioes = torch.zeros(
        [num_refs, height, width],
        dtype = descriptor0.dtype,
        device = descriptor0.device)
    num_maps = torch.zeros(
        [num_refs, height, width],
        dtype = descriptor0.dtype,
        device = descriptor0.device)
    offsets = []
    for col_offset in range(-radius, radius + 1):
        for row_offset in range(-radius, radius + 1):
            if col_offset == 0 and row_offset == 0:
                continue
            offsets.append([2. * col_offset * stride / width,
                            2. * row_offset * stride / height])
    offsets = torch.tensor(
        offsets, dtype = grids_01.dtype, device = grids_01.device)
    num_offset = offsets.shape[0]
    for idx_ref in range(num_refs):
        descriptor1 = descriptors1[idx_ref:idx_ref+1, :, :, :]
        grid_01 = grids_01[idx_ref:idx_ref+1, :, :, :]

        num_map = torch.zeros(
            [1, height, width],
            dtype = descriptor0.dtype,
            device = descriptor0.device)
        ns, rows, cols = torch.where(num_map == 0)
        valid_points = torch.where(
            (grid_01[ns, rows, cols, 0] - 2. * stride * radius / width > -1.) &
            (grid_01[ns, rows, cols, 0] + 2. * stride * radius / width < 1.) &
            (grid_01[ns, rows, cols, 1] - 2. * stride * radius / height > -1.) &
            (grid_01[ns, rows, cols, 1] + 2. * stride * radius / height < 1.))
        ns = ns[valid_points]
        xs = rows[valid_points]
        ys = cols[valid_points]
        num_map[ns, xs, ys] = 1
        num_maps[idx_ref, :, :] = num_map[0, :, :]

        # Calculate repeatability.
        descriptor1_reverse = F.grid_sample(descriptor1, grid_01)
        descriptor1_reverse = F.normalize(descriptor1_reverse, dim = 1)
        repeatability = descriptor0 * descriptor1_reverse
        repeatability = torch.sqrt(2. - 2. * repeatability.sum(dim=1) + 1e-4)
        repeatability = repeatability * num_map

        # Calculate distinctness.
        distinctness = torch.zeros(
            [num_offset, height, width],
            dtype = descriptor0.dtype,
            device = descriptor0.device)
        for idx_offset in range(num_offset):
            grid_01_offset = torch.zeros(
                grid_01.shape, dtype = grid_01.dtype, device = grid_01.device)
            grid_01_offset[0, :, :, 0] = (
                grid_01[0, :, :, 0] + offsets[idx_offset, 0])
            grid_01_offset[0, :, :, 1] = (
                grid_01[0, :, :, 1] + offsets[idx_offset, 1])

            descriptor1_reverse = F.grid_sample(descriptor1, grid_01_offset)
            descriptor1_reverse = F.normalize(descriptor1_reverse, dim = 1)
            distinctness_tmp = descriptor0 * descriptor1_reverse
            distinctness_tmp = torch.sqrt(2. - 2. * distinctness_tmp.sum(dim=1) + 1e-4)
            distinctness_tmp = distinctness_tmp * num_map
            distinctness[idx_offset, :, :] = distinctness_tmp[0, :, :]
        distinctness, _ = distinctness.min(dim = 0)
        distinctness = distinctness.unsqueeze(0)

        #ratioes[idx_ref, :, :] = distinctness[0, :, :] - repeatability[0, :, :]
        ratioes[idx_ref, :, :] = (distinctness[0, :, :] /
            (repeatability[0, :, :] + 3*1e-2))

    #ratioes[num_maps == 0] = 1e3
    #ratio, _ = ratioes.min(dim = 0)
    #ratio[num_maps.sum(dim = 0) == 0] = 0

    ratio = ratioes.sum(dim = 0) / (num_maps.sum(dim = 0) + 1e-2)

    return ratio

    _, dim, height, width = descriptor0.shape
    num_refs, _, _, _ = descriptors1.shape
    # Calculate repeatability.
    num_maps = torch.zeros(
        [num_refs, height, width],
        dtype = descriptor0.dtype,
        device = descriptor0.device)
    ns, rows, cols = torch.where(num_maps == 0)
    valid_points = torch.where(
        (grid_01[ns, rows, cols, 0] > -0.98) &
        (grid_01[ns, rows, cols, 0] < 0.98) &
        (grid_01[ns, rows, cols, 1] > -0.98) &
        (grid_01[ns, rows, cols, 1] < 0.98))
    ns = ns[valid_points]
    xs = rows[valid_points]
    ys = cols[valid_points]

    num_maps[ns, xs, ys] = 1
    num_maps_sum = torch.sum(num_maps, dim = 0)

    repeatability = descriptor0 * descriptors1
    repeatability = torch.sqrt(2. - 2. * repeatability.sum(dim=1))

    repeatability = (
        torch.sum(repeatability * num_maps, dim = 0) / (num_maps_sum + 1e-8))

    #repeatability[num_maps == 0] = 0
    #repeatability, _ = torch.max(repeatability, dim = 0)
    #repeatability[num_maps_sum == 0] = 2

    # Calculate distinctness.
    num_maps = torch.ones(
        [num_refs, height, width],
        dtype = descriptor0.dtype,
        device = descriptor0.device)
    num_maps[:, 0:radius*stride, :] = 0
    num_maps[:, :, 0:radius*stride] = 0
    num_maps[:, -radius*stride::, :] = 0
    num_maps[:, :, -radius*stride::] = 0
    ns, rows, cols = torch.where(num_maps == 1)
    valid_points = torch.where(
        (grid_01[ns, rows, cols, 0] > -0.98) &
        (grid_01[ns, rows, cols, 0] < 0.98) &
        (grid_01[ns, rows, cols, 1] > -0.98) &
        (grid_01[ns, rows, cols, 1] < 0.98))
    ns = ns[valid_points]
    xs = rows[valid_points]
    ys = cols[valid_points]

    num_maps[:] = 0
    num_maps[ns, xs, ys] = 1
    num_maps_sum = torch.sum(num_maps, dim = 0)

    distinctness = torch.zeros(
        [num_refs, height, width],
        dtype = descriptor0.dtype,
        device = descriptor0.device)
    diameter = radius * 2 + 1
    num_locals = diameter ** 2
    rows_all, cols_all = torch.where(num_maps[0, :, :] > -1)
    descriptor0 = descriptor0.permute(2, 3, 0, 1)
    descriptor0 = descriptor0.reshape([height * width, 1, dim])
    for idx_ref in range(num_refs):
        torch.cuda.empty_cache()
        rows_tmp = []
        cols_tmp = []
        for row_offset in range(-radius, radius + 1):
            for col_offset in range(-radius, radius + 1):
                rows_tmp.append(rows_all + row_offset * stride)
                cols_tmp.append(cols_all + col_offset * stride)

        rows_tmp = torch.cat(rows_tmp, dim = 0)
        cols_tmp = torch.cat(cols_tmp, dim = 0)

        rows_tmp[rows_tmp < 0] = 0
        rows_tmp[rows_tmp >= height] = height - 1
        cols_tmp[cols_tmp < 0] = 0
        cols_tmp[cols_tmp >= width] = width - 1

        descriptors1_tmp = descriptors1[idx_ref, :, rows_tmp, cols_tmp]
        descriptors1_tmp = descriptors1_tmp.reshape(
            [dim, num_locals, height, width])
        descriptors1_tmp = descriptors1_tmp.permute(2, 3, 0, 1)
        descriptors1_tmp = descriptors1_tmp.reshape(
            [height * width, dim, num_locals])

        distinctness_tmp = torch.bmm(descriptor0, descriptors1_tmp)

        distinctness_tmp = torch.sqrt(2. - 2. * distinctness_tmp.squeeze())
        distinctness_tmp, _ = torch.topk(
            distinctness_tmp, idx_near + 1, dim = 1, largest = False)
        distinctness_tmp = distinctness_tmp[:, -1]
        distinctness[idx_ref, :, :] = distinctness_tmp.reshape([height, width])

    #distinctness = distinctness * num_maps
    #distinctness = (torch.sum(distinctness, dim = 0) / (num_maps_sum + 1e-1))

    distinctness[num_maps == 0] = 1e2
    distinctness, _ = torch.min(distinctness, dim = 0)
    distinctness[num_maps_sum == 0] = 0

    ratio = distinctness
    #ratio = distinctness / (repeatability + 1e-1)
    #ratio = distinctness - repeatability

    return ratio

def calculate_ratio_local_grid(
    descriptor0, descriptor1, correspondence, idx_near = 1, radius = 3,
    stride = 2):

    diameter = 2 * radius + 1
    num_local = diameter ** 2
    dims_feature, height0, width0 = descriptor0.shape
    dims_feature, height1, width1 = descriptor1.shape

    ratio = np.zeros([height0, width0])
    num_map = np.zeros([height0, width0])

    xs, ys = np.where(ratio == 0)
    coord0 = np.concatenate((xs, ys)).reshape([2, xs.shape[0]])
    coord0 = coord0.round().astype(np.int64)
    coord1 = np.concatenate((correspondence[xs, ys, 0],
                             correspondence[xs, ys, 1]))
    coord1 = coord1.reshape([2, xs.shape[0]]).round().astype(np.int64)

    border = stride * radius
    valid_points = coord0[0, :] > 0
    valid_points = valid_points & (coord0[0, :] < height0)
    valid_points = valid_points & (coord0[1, :] > 0)
    valid_points = valid_points & (coord0[1, :] < width0)
    valid_points = valid_points & (coord1[0, :] > border)
    valid_points = valid_points & (coord1[0, :] < height1 - border)
    valid_points = valid_points & (coord1[1, :] > border )
    valid_points = valid_points & (coord1[1, :] < width1 - border)
    coord0 = coord0[:, valid_points]
    coord1 = coord1[:, valid_points]

    num_points = coord0.shape[1]
    if num_points == 0:
        return ratio, num_map

    num_map[coord0[0,:], coord0[1,:]] = 1

    descriptor0_mat = descriptor0[:, coord0[0,:], coord0[1,:]].transpose(0,
        1).reshape([num_points,1,dims_feature])

    x_offset = np.arange(- stride * radius, stride * radius + 1, stride)
    x_offset = np.tile(np.repeat(x_offset, diameter), num_points)
    y_offset = np.arange(- stride * radius, stride * radius + 1, stride)
    y_offset = np.tile(np.tile(y_offset, diameter), num_points)

    correlation_mat = torch.zeros(
        (num_points, num_local), device = descriptor0.device)
    for idx_local in range(num_local):
        torch.cuda.empty_cache()
        xy1 = coord1.copy()
        xy1[0,:] = xy1[0,:] - x_offset[idx_local::num_local]
        xy1[1,:] = xy1[1,:] - y_offset[idx_local::num_local]
        descriptor1_mat = descriptor1[:, xy1[0,:], xy1[1,:]].reshape(
            [dims_feature, num_points, 1]).transpose(0, 1)

        correlation_mat[:, idx_local] = torch.bmm(
            descriptor0_mat, descriptor1_mat).reshape([num_points])
    x_same = np.arange(num_points)
    y_same = np.repeat([2*radius*(radius+1)], num_points)
    correlation_same = correlation_mat[x_same, y_same].clone()
    correlation_mat[x_same, y_same] = -1
    correlation_topk, _ = torch.topk(correlation_mat, idx_near, dim = 1)
    correlation_rankk = correlation_topk[:, -1]
    #ratio_k = (
    #    (torch.sqrt(2. * (1. - correlation_rankk))) /
    #    (torch.sqrt(2. * (1. - correlation_same))))
    ratio_k = torch.sqrt(2. * (1. - correlation_rankk))

    ratio[coord0[0,:], coord0[1,:]] = ratio_k.detach().cpu().numpy()

    return ratio, num_map


def calculate_ratio_local(descriptor0, descriptor1, matrix, idx_near = 1,
        radius = 3, verbose = False):
    diameter = 2*radius + 1
    num_local = diameter ** 2
    dims_feature, height, width = descriptor0.shape
    ratio = np.zeros([height, width])

    xs, ys = np.where(ratio == 0)
    coord0 = np.concatenate((xs, ys)).reshape([2, xs.shape[0]])
    coord = np.ones([3, coord0.shape[1]])
    coord[0:2, :] = coord0
    coord0 = coord0.round().astype(np.int64)
    coord1 = np.matmul(matrix, coord).round().astype(np.int64)

    border = 2*radius
    valid_points = coord0[0, :] >= 0
    valid_points = valid_points & (coord0[0, :] < height)
    valid_points = valid_points & (coord0[1, :] >= 0)
    valid_points = valid_points & (coord0[1, :] < width)
    valid_points = valid_points & (coord1[0, :] > border)
    valid_points = valid_points & (coord1[0, :] < height - border)
    valid_points = valid_points & (coord1[1, :] > border )
    valid_points = valid_points & (coord1[1, :] < width - border)
    coord0 = coord0[:, valid_points]
    coord1 = coord1[:, valid_points]
    num_points = coord0.shape[1]
    num_map_affined = np.zeros([height, width])
    num_map_affined[coord0[0,:], coord0[1,:]] = 1

    descriptor0_mat = (
        descriptor0[:, coord0[0,:], coord0[1,:]].transpose(0, 1).reshape(
            [num_points,1,dims_feature]))

    x_offset = np.arange(-2 * radius, 2 * radius + 1, 2)
    x_offset = np.tile(np.repeat(x_offset, diameter), num_points)
    y_offset = np.arange(-2 * radius, 2 * radius + 1, 2)
    y_offset = np.tile(np.tile(y_offset, diameter), num_points)
    xy1 = np.repeat(coord1, num_local, axis = 1)
    xy1[0,:] = xy1[0,:] - x_offset
    xy1[1,:] = xy1[1,:] - y_offset

    descriptor1_mat = descriptor1[:, xy1[0,:], xy1[1,:]].reshape(
            [dims_feature,num_points,(2*radius+1) ** 2]).transpose(0, 1)

    correlation_mat = torch.bmm(descriptor0_mat, descriptor1_mat).reshape(
            [num_points, -1])
    x_same = np.arange(num_points)
    y_same = np.repeat([2*radius*(radius+1)], num_points)
    correlation_same = correlation_mat[x_same, y_same].clone()
    correlation_mat[x_same, y_same] = -1
    correlation_topk, _ = torch.topk(correlation_mat, idx_near, dim=1)
    correlation_rankk = correlation_topk[:,-1]
    ratio_k = (
        torch.sqrt(2. * (1. - correlation_rankk)) /
        torch.sqrt(2. * (1. - correlation_same)))

    ratio[coord0[0,:], coord0[1,:]] = ratio_k.detach().cpu().numpy()

    if verbose == False:
        return ratio, num_map_affined
    else:
        repeatability = np.zeros([height, width])
        distinctness = np.zeros([height, width])
        repeatability[coord0[0,:], coord0[1,:]] = (
            torch.sqrt(2. * (1. - correlation_same)).cpu().numpy())
        distinctness[coord0[0,:], coord0[1,:]] = (
            torch.sqrt(2. * (1. - correlation_rankk)).cpu().numpy())
        return ratio, num_map_affined, repeatability, distinctness

def calculate_ratio(descriptor0, descriptor1, matrix, idx_near = 5):
    dims_feature, height, width = descriptor0.shape
    ratio = np.zeros([height, width])

    xs, ys = np.where(ratio == 0)
    coord0 = np.concatenate((xs, ys)).reshape([2, xs.shape[0]])
    coord = np.ones([3, coord0.shape[1]])
    coord[0:2, :] = coord0
    coord0 = coord0.round().astype(np.int64)
    coord1 = np.matmul(matrix, coord).round().astype(np.int64)

    valid_points = coord0[0, :] > 0
    valid_points = valid_points & (coord0[0, :] < height)
    valid_points = valid_points & (coord0[1, :] > 0)
    valid_points = valid_points & (coord0[1, :] < width)
    valid_points = valid_points & (coord1[0, :] > 0)
    valid_points = valid_points & (coord1[0, :] < height)
    valid_points = valid_points & (coord1[1, :] > 0)
    valid_points = valid_points & (coord1[1, :] < width)
    coord0 = coord0[:, valid_points]
    coord1 = coord1[:, valid_points]
    num_map_affined = np.zeros([height, width])
    num_map_affined[coord0[0,:], coord0[1,:]] = 1

    descriptor0_mat = descriptor0[:, coord0[0,:], coord0[1,:]].transpose(0, 1)
    descriptor1_mat = descriptor1[:, coord1[0,:], coord1[1,:]]
    correlation_mat = torch.mm(descriptor0_mat, descriptor1_mat)
    x_diagonal = np.range(correlation_mat.shape[0])
    y_diagonal = np.range(correlation_mat.shape[0])
    correlation_same = correlation_mat[x_diagonal, y_diagonal]
    correlation_topk, _ = torch.topk(correlation_mat, idx_near, dim=1)
    correlation_rankk = correlation_topk[:,-1]
    ratio_k = (
        torch.sqrt(2. * (1. - correlation_rankk)) /
        (torch.sqrt(2. * (1. - correlation_same)) + 1e-6))

    ratio[coord0[0,:], coord0[1,:]] = ratio_k.detach().cpu().numpy()

    return ratio, num_map_affined

def get_ratio_affine_adaption_multi_scale(
    model, img_path, down_ratio_descriptor, num_affines=1, verbose = False):
    # Affine parameters.
    degrees = (-40, 40)
    translate = (0.04, 0.04)
    scale = (0.7, 1.4)
    shear = (-40, 40)
    fillcolor = 0
    # Color jitter transformer.
    transformer_color_jitter = torchvision.transforms.ColorJitter(
            brightness=(0.6, 1.4), contrast=(0.6, 1.4),
            saturation=(0.6, 1.4), hue=(-0.2, 0.2))

    with open(img_path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
    width_img, height_img = img.size
    # Process the initial image.
    img_affined = transformer_color_jitter(img)
    img_affined = TF.to_tensor(img_affined.convert('L')).unsqueeze(0).to(
            device=next(model.parameters()).device)
    if random.random() < 0.2:
        img_affined = random_blur(img_affined)
    with torch.no_grad():
        outs = model.forward(img_affined)
        descriptor_high = outs[1]
        descriptor_low = outs[2]
        descriptor_high = F.normalize(descriptor_high, dim=1)
        descriptor_low = F.normalize(descriptor_low, dim=1)
    _, _, height_map_high, width_map_high = descriptor_high.shape
    _, _, height_map_low, width_map_low = descriptor_low.shape
    ratio_high = np.zeros([height_map_high, width_map_high])
    ratio_low = np.zeros([height_map_low, width_map_low])
    if verbose == True:
        imgs = np.zeros([num_affines+1, height_map_low, width_map_low])
        imgs[0,:,:] = img_affined[0,0,:,:].cpu().numpy()
        ratios = np.zeros([num_affines, height_map_low, width_map_low])
        repeatability_high = np.zeros([height_map_high, width_map_high])
        repeatability_low = np.zeros([height_map_low, width_map_low])
        distinctness_high = np.zeros([height_map_high, width_map_high])
        distinctness_low = np.zeros([height_map_low, width_map_low])
    num_map_high = np.zeros([height_map_high, width_map_high]) + 1e-6
    num_map_low = np.zeros([height_map_low, width_map_low]) + 1e-6
    # Process each affined image.
    for index_affine in range(num_affines):
        # Random affine input image.
        affine_params_img = get_affine_params(
            degrees=degrees, translate=translate, scale_ranges=scale,
            shears=shear, img_size=img.size, random_method='uniform')
        affine_params_map_high = (affine_params_img[0],
                (affine_params_img[1][0] / down_ratio_descriptor,
                 affine_params_img[1][1] / down_ratio_descriptor),
                affine_params_img[2], affine_params_img[3])
        affine_params_map_low = (affine_params_img[0],
                (affine_params_img[1][0], affine_params_img[1][1]),
                affine_params_img[2], affine_params_img[3])
        img_affined = transformer_color_jitter(img)
        img_affined = TF.affine(
            img_affined, *affine_params_img, resample=Image.BILINEAR,
            fillcolor=fillcolor)
        img_affined = TF.to_tensor(img_affined.convert('L')).unsqueeze(0).to(
                device=next(model.parameters()).device)
        if random.random() < 0.2:
            img_affined = random_blur(img_affined)

        center_map_high = (width_map_high * 0.5 + 0.5,
                           height_map_high * 0.5 + 0.5)
        matrix_map_high = get_affine_matrix(
            center_map_high, *affine_params_map_high)
        center_map_low = (width_map_low * 0.5 + 0.5,
                          height_map_low * 0.5 + 0.5)
        matrix_map_low = get_affine_matrix(
            center_map_low, *affine_params_map_low)

        # Compute result.
        with torch.no_grad():
            torch.cuda.empty_cache()
            outs = model.forward(img_affined)
            descriptor_high_affined = outs[1]
            descriptor_low_affined = outs[2]
            descriptor_high_affined = F.normalize(
                descriptor_high_affined, dim = 1)
            descriptor_low_affined = F.normalize(
                descriptor_low_affined, dim = 1)
            torch.cuda.empty_cache()
            if verbose == False:
                ratio_high_affined, num_map_high_affined = (
                    calculate_ratio_local(
                        descriptor_high.squeeze(),
                        descriptor_high_affined.squeeze(),
                        matrix_map_high))
            else:
                (ratio_high_affined, num_map_high_affined,
                 repeatability_high_affined, distinctness_high_affined) = (
                    calculate_ratio_local(
                        descriptor_high.squeeze(),
                        descriptor_high_affined.squeeze(),
                        matrix_map_high, verbose = True))
            ratio_high_affined = ratio_high_affined.squeeze()
            torch.cuda.empty_cache()
            if verbose == False:
                ratio_low_affined, num_map_low_affined = calculate_ratio_local(
                    descriptor_low.squeeze(),
                    descriptor_low_affined.squeeze(),
                    matrix_map_low)
            else:
                (ratio_low_affined, num_map_low_affined,
                 repeatability_low_affined, distinctness_low_affined) = (
                    calculate_ratio_local(
                        descriptor_low.squeeze(),
                        descriptor_low_affined.squeeze(),
                        matrix_map_low, verbose = True))
            ratio_low_affined = ratio_low_affined.squeeze()

        ratio_high += ratio_high_affined * num_map_high_affined
        num_map_high += num_map_high_affined

        ratio_low += ratio_low_affined * num_map_low_affined
        num_map_low += num_map_low_affined

        if verbose == True:
            imgs[1+index_affine,:,:] = img_affined[0,0,:,:].cpu().numpy()
            ratios[index_affine,:,:] = 0.5 * (
                ratio_low_affined * num_map_low_affined +
                cv2.resize(ratio_high_affined * num_map_high_affined,
                           (width_map_low, height_map_low),
                           interpolation=cv2.INTER_LINEAR))
            repeatability_high += (
                repeatability_high_affined * num_map_high_affined)
            repeatability_low += repeatability_low_affined * num_map_low_affined
            distinctness_high += (
                distinctness_high_affined * num_map_high_affined)
            distinctness_low += distinctness_low_affined * num_map_low_affined

    ratio_high = ratio_high / num_map_high
    ratio_high = cv2.resize(
        ratio_high, (width_map_low, height_map_low),
        interpolation=cv2.INTER_LINEAR)
    ratio_low = ratio_low / num_map_low
    ratio = 0.5 * (ratio_low + ratio_high)

    if verbose == False:
        return ratio
    else:
        repeatability_high = repeatability_high / num_map_high
        repeatability_high = cv2.resize(repeatability_high,
                (width_map_low, height_map_low), interpolation=cv2.INTER_LINEAR)
        repeatability_low = repeatability_low / num_map_low
        repeatability = 0.5 * (repeatability_low + repeatability_high)

        distinctness_high = distinctness_high / num_map_high
        distinctness_high = cv2.resize(distinctness_high,
                (width_map_low, height_map_low), interpolation=cv2.INTER_LINEAR)
        distinctness_low = distinctness_low / num_map_low
        distinctness = 0.5 * (distinctness_low + distinctness_high)

        return (ratio, ratio_low, ratio_high, imgs,
                ratios, repeatability, distinctness)

def get_ratio_affine_adaption(
    model, img_path, down_ratio_descriptor, num_affines=1):
    # Affine parameters.
    degrees = (-40, 40)
    translate = (0.04, 0.04)
    scale = (0.7, 1.4)
    shear = (-40, 40)
    fillcolor = 0
    # Color jitter transformer.
    transformer_color_jitter = torchvision.transforms.ColorJitter(
            brightness=(0.6, 1.4), contrast=(0.6, 1.4),
            saturation=(0.6, 1.4), hue=(-0.2, 0.2))

    with open(img_path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
    width_img, height_img = img.size
    # Process the initial image.
    img_affined = transformer_color_jitter(img)
    img_affined = TF.to_tensor(img_affined.convert('L')).unsqueeze(0).to(
            device=next(model.parameters()).device)
    if random.random() < 0.2:
        img_affined = random_blur(img_affined)
    with torch.no_grad():
        torch.cuda.empty_cache()
        outs = model.forward(img_affined)
        descriptor = outs[1]
        descriptor = F.interpolate(
            descriptor,
            (int(height_img/down_ratio_descriptor),
             int(width_img/down_ratio_descriptor)),
            mode='bilinear')
        descriptor = F.normalize(descriptor, dim=1)
    _, _, height_map, width_map= descriptor.shape
    ratio = np.zeros([height_map, width_map])
    num_map = np.zeros([height_map, width_map]) + 1e-6
    # Process each affined image.
    for index_affine in range(num_affines):
        # Random affine input image.
        affine_params_img = get_affine_params(
            degrees=degrees, translate=translate, scale_ranges=scale,
            shears=shear, img_size=img.size, random_method='uniform')
        affine_params_map = (
            affine_params_img[0],
            (affine_params_img[1][0] / down_ratio_descriptor,
             affine_params_img[1][1] / down_ratio_descriptor),
            affine_params_img[2], affine_params_img[3])
        img_affined = transformer_color_jitter(img)
        img_affined = TF.affine(
            img_affined, *affine_params_img, resample=Image.BILINEAR,
            fillcolor=fillcolor)
        img_affined = TF.to_tensor(img_affined.convert('L')).unsqueeze(0).to(
                device=next(model.parameters()).device)
        if random.random() < 0.2:
            img_affined = random_blur(img_affined)

        center_map = (width_map * 0.5 + 0.5, height_map * 0.5 + 0.5)
        matrix_map = get_affine_matrix(center_map, *affine_params_map)

        # Compute result.
        with torch.no_grad():
            torch.cuda.empty_cache()
            outs = model.forward(img_affined)
            descriptor_affined = outs[1]
            descriptor_affined = F.interpolate(
                descriptor_affined,
                (int(height_img/down_ratio_descriptor),
                 int(width_img/down_ratio_descriptor)),
                mode='bilinear')
            descriptor_affined = F.normalize(descriptor_affined, dim=1)
            torch.cuda.empty_cache()
            ratio_affined, num_map_affined = calculate_ratio_local(
                descriptor.squeeze(), descriptor_affined.squeeze(), matrix_map)
            ratio_affined = ratio_affined.squeeze()

        ratio += ratio_affined * num_map_affined
        num_map += num_map_affined

    ratio = ratio / num_map
    ratio = cv2.resize(
        ratio, (width_img, height_img), interpolation=cv2.INTER_LINEAR)

    return ratio

def get_ratio_affine_adaption_local(
    model, img_path, down_ratio_descriptor, num_affines=1):
    # Affine parameters.
    degrees = (-40, 40)
    translate = (0.04, 0.04)
    scale = (0.7, 1.4)
    shear = (-40, 40)
    fillcolor = 0
    # Color jitter transformer.
    transformer_color_jitter = torchvision.transforms.ColorJitter(
            brightness=(0.6, 1.4), contrast=(0.6, 1.4),
            saturation=(0.6, 1.4), hue=(-0.2, 0.2))

    with open(img_path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
    width_img, height_img = img.size
    # Process the initial image.
    img_affined = transformer_color_jitter(img)
    img_affined = TF.to_tensor(img_affined.convert('L')).unsqueeze(0).to(
            device=next(model.parameters()).device)
    if random.random() < 0.2:
        img_affined = random_blur(img_affined)
    with torch.no_grad():
        outs = model.forward(img_affined)
        descriptor = outs[2]
        descriptor = F.normalize(descriptor, dim=1)
    _, _, height_map, width_map= descriptor.shape
    ratio = np.zeros([height_map, width_map])
    num_map = np.zeros([height_map, width_map]) + 1e-6
    # Process each affined image.
    for index_affine in range(num_affines):
        # Random affine input image.
        affine_params_img = get_affine_params(
            degrees=degrees, translate=translate, scale_ranges=scale,
            shears=shear, img_size=img.size, random_method='uniform')
        affine_params_map = (affine_params_img[0],
                (affine_params_img[1][0], affine_params_img[1][1]),
                affine_params_img[2], affine_params_img[3])
        img_affined = transformer_color_jitter(img)
        img_affined = TF.affine(
            img_affined, *affine_params_img, resample=Image.BILINEAR,
            fillcolor=fillcolor)
        img_affined = TF.to_tensor(img_affined.convert('L')).unsqueeze(0).to(
                device=next(model.parameters()).device)
        if random.random() < 0.2:
            img_affined = random_blur(img_affined)

        center_map = (width_map * 0.5 + 0.5, height_map * 0.5 + 0.5)
        matrix_map = get_affine_matrix(center_map, *affine_params_map)

        # Compute result.
        with torch.no_grad():
            outs = model.forward(img_affined)
            descriptor_affined = outs[2]
            descriptor_affined = F.normalize(descriptor_affined, dim=1)
            ratio_affined, num_map_affined = calculate_ratio_local(
                descriptor.squeeze(), descriptor_affined.squeeze(), matrix_map)
            ratio_affined = ratio_affined.squeeze()

        ratio += ratio_affined * num_map_affined
        num_map += num_map_affined

    ratio = ratio / num_map

    return ratio

def get_ratio_affine_adaption_multi_scale_from_image_pair(
    args, img_target, img_refs, grid_target2refs, model):

    _, _, height_target, width_target = img_target.shape
    num_refs, dim, height_ref, width_ref = img_refs.shape

    torch.cuda.empty_cache()
    with torch.no_grad():
        # Get descriptors of img_target.
        outs = model.forward(img_target)

        descriptor_high = outs[1]
        descriptor_high = F.normalize(descriptor_high, dim=1)

        descriptor_low = outs[2]
        descriptor_low = F.normalize(descriptor_low, dim=1)

        # Get descriptors of img_refs.
        outs = model.forward(img_refs)

        descriptor_high_refs = outs[1]
        grid_target2refs_high = F.interpolate(
            grid_target2refs.permute(0, 3, 1, 2),
            (int(height_target/4), int(width_target/4)),
            mode='bilinear').permute(0, 2, 3, 1)

        descriptor_low_refs = outs[2]

        # Calculate keypoints.
        torch.cuda.empty_cache()

        ratio_high = calculate_mean_ratio_local(
            descriptor_high,
            descriptor_high_refs,
            grid_target2refs_high,
            stride = 2)
        ratio_high = F.interpolate(
            ratio_high.unsqueeze(0).unsqueeze(0),
            (height_target, width_target),
            mode='bilinear').squeeze()

        ratio_low = calculate_mean_ratio_local(
            descriptor_low,
            descriptor_low_refs,
            grid_target2refs,
            stride = 2)

        #ratio = ratio_low
        #ratio = ratio_high
        ratio = 0.5 * (ratio_low + ratio_high)

    return ratio

def get_ratio(
    args, dataset_reader, model,
    num_sparses = [0], num_points = [0], num_pixels = [0], index = 0):

    num_imgs_with_sparse_points = 0
    num_points_sum = 0
    num_pixels_sum = 0
    prefetch_reader = PrefetchReader(dataset_reader, args.num_refs)
    for idx in range(len(prefetch_reader)):
        data = prefetch_reader[idx]
        image_path = data['image_path']
        keypoints_path = data['keypoints_path']
        img_target = data['img_target']
        img_refs = data['img_refs']
        grid_target2refs = data['grid_target2refs']

        torch.cuda.empty_cache()
        #_, img_name = os.path.split(img_path)
        #keypoints_path = os.path.join(dataset_reader.keypoints_dir, img_name)
        #ratio = get_ratio_affine_adaption_multi_scale(model, img_path,
        #        down_ratio_descriptor=1, num_affines = 20)
        #ratio = get_ratio_affine_adaption(model, img_path,
        #        down_ratio_descriptor=1, num_affines = 20)
        ratio = get_ratio_affine_adaption_multi_scale_from_image_pair(
            args, img_target, img_refs, grid_target2refs, model)
        # NMS only process one input map each time.
        keypoints_map = non_maximum_suppress(
            ratio,
            confidence_threshold = args.confidence_threshold_reliability,
            maxinum_points = args.max_keypoints, radius = args.nms_radius)
        keypoints_map[0:4*args.nms_radius, :] = 0
        keypoints_map[:, 0:4*args.nms_radius] = 0
        keypoints_map[-4*args.nms_radius::, :] = 0
        keypoints_map[:, -4*args.nms_radius::] = 0

        # Save the keypoints into numpy file.
        np.savez(keypoints_path, keypoints_map = keypoints_map,
                 score = ratio.squeeze().detach().cpu().numpy())
        cv2.imwrite(keypoints_path + '_keypoints.png', keypoints_map.astype(np.uint8) * 255)
        ratio_np = ratio.squeeze().detach().cpu().numpy()
        #ratio_np = (ratio_np - ratio_np.min()) / (ratio_np.max() - ratio_np.min())
        cv2.imwrite(keypoints_path + '_score.png', ratio_np * 192)

        if keypoints_map.sum() < args.threshold_sparse_points:
            num_imgs_with_sparse_points += 1
        num_points_sum += keypoints_map.sum()
        num_pixels_sum += keypoints_map.size

    num_sparses[index] = num_imgs_with_sparse_points
    num_points[index] = num_points_sum
    num_pixels[index] = num_pixels_sum
    return num_imgs_with_sparse_points, num_points_sum, num_pixels_sum

