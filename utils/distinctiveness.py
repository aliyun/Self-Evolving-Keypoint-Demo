
# First party.
from .util import get_affine_params, get_affine_matrix, non_maximum_suppress

# Standard.
import os

# Third party.
import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF

def calculate_local_distinctiveness(descriptor, radius = 1, local_range = 8):
    batch_size, dims_feature, height, width = descriptor.shape
    distinctiveness = torch.zeros(
        [batch_size, height, width], device = descriptor.device)
    for index_batch in range(batch_size):
        # Process an img in each iteration.
        descriptor_img = descriptor[index_batch,:,:,:]
        for index_height in range(local_range, height-local_range):
            descriptor_a = (
                descriptor_img[:, index_height,
                    local_range:width-local_range].transpose(
                        0, 1).unsqueeze(1).contiguous())
            descriptor_b = (
                descriptor_img[:,
                    index_height-local_range:index_height+local_range+1,
                    :].unfold(2, 2*local_range+1, 1).transpose(0, 2).reshape(
                            [width-2*local_range, dims_feature,
                            (2*local_range+1)**2]).contiguous())
            correlations = torch.bmm(
                descriptor_a, descriptor_b).squeeze().reshape(
                    [width-2*local_range, 2*local_range+1, 2*local_range+1])
            correlations[:,
                local_range-radius:local_range+radius+1,
                local_range-radius:local_range+radius+1] = -1
            correlations, _ = correlations.reshape(
                [width-2*local_range, (2*local_range+1)**2]).max(dim=1)
            correlations = torch.clamp(correlations, min=-1, max=1)
            distinctiveness[index_batch, index_height,
                local_range:width-local_range] = torch.sqrt(
                    2. * (1. - correlations)) / 2.

    distinctiveness = distinctiveness.detach().cpu().numpy()
    return distinctiveness

def calculate_distinctiveness(descriptor, radius = 1):
    batch_size, dims_feature, height, width = descriptor.shape
    distinctiveness = np.zeros([batch_size, height, width])
    AllAtOnce = height*width < 2**13
    for index_batch in range(batch_size):
        # Process an img in each iteration.
        descriptor_img = descriptor[index_batch,:,:,:].clone().reshape(
            [dims_feature, height, width])
        if AllAtOnce:
            descriptor_img = descriptor_img.reshape(
                [dims_feature, height*width]).transpose(0, 1).contiguous()
            distance = F.pdist(descriptor_img).cpu().numpy()
            distance = squareform(distance)
            distance = distance.reshape([height, width, height, width])
            for index_height in range(radius, height-radius):
                for index_width in range(radius, width-radius):
                    distance[index_height, index_width,
                        index_height-radius:index_height+radius+1,
                        index_width-radius:index_width+radius+1] = 10
            distance = distance.reshape(
                [height*width, height*width]).min(axis=1)
            distance = distance.reshape([height, width])
            distinctiveness[index_batch, :, :] = distance / 2.
        else:
            for index_height in range(radius, height-radius):
                correlations = torch.ones(
                    [width, height*width], device = descriptor.device)
                correlations = torch.mm(
                    descriptor_img[:, index_height, :].transpose(0, 1),
                    descriptor_img.reshape([dims_feature, height*width]))
                correlations = correlations.reshape([width, height, width])
                correlations[0:radius, :, :] = 1
                correlations[width-radius:width, :, :] = 1
                for index_width in range(radius, width-radius):
                    correlations[index_width,
                        index_height-radius:index_height+radius+1,
                        index_width-radius:index_width+radius+1] = -1
                correlations, _ = correlations.reshape(
                    [width, height*width]).max(dim=1)
                correlations = torch.clamp(correlations, min=-1, max=1)
                distance = torch.sqrt(
                    2. * (1. - correlations)).detach().cpu().numpy()
                distinctiveness[index_batch, index_height, :] = distance / 2.

    return distinctiveness

def get_similarity_affine_adaption(
    model, img_path, down_ratio_descriptor, num_affines=1):
    # Affine parameters.
    degrees = (-15, 15)
    translate = (0.04, 0.04)
    scale = (0.9, 1.1)
    shear = (-15, 15)
    fillcolor = 0
    # Color jitter transformer.
    transformer_color_jitter = torchvision.transforms.ColorJitter(
            brightness=(0.8, 1.2), contrast=(0.8, 1.2),
            saturation=(0.8, 1.2), hue=(-0.2, 0.2))

    with open(img_path, 'rb') as f:
        img = Image.open(f)
        img = img.convert('RGB')
    width_img, height_img = img.size
    # Process the initial image.
    img_affined = transformer_color_jitter(img)
    img_affined = TF.to_tensor(img_affined.convert('L')).unsqueeze(0).to(
            device=next(model.parameters()).device)
    with torch.no_grad():
        _, descriptor_high, descriptor_low = model.forward(img_affined)
        descriptor_high = F.normalize(descriptor_high, dim=1)
        descriptor_low = F.normalize(descriptor_low, dim=1)
        distinctiveness_high = calculate_distinctiveness(descriptor_high)
        distinctiveness_high = distinctiveness_high.squeeze()
        distinctiveness_low = calculate_local_distinctiveness(descriptor_low)
        distinctiveness_low = distinctiveness_low.squeeze()
    height_map_high, width_map_high = distinctiveness_high.shape
    num_map_high = np.ones([height_map_high, width_map_high])
    height_map_low, width_map_low = distinctiveness_low.shape
    num_map_low = np.ones([height_map_low, width_map_low])
    # Process each affined image.
    for index_affine in range(num_affines-1):
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
        img_affined = TF.to_tensor(
            img_affined.convert('L')).unsqueeze(0).to(
                device=next(model.parameters()).device)

        # Compute result.
        with torch.no_grad():
            _, descriptor_high_affined, descriptor_low_affined = model.forward(
                img_affined)
            descriptor_high_affined = F.normalize(
                descriptor_high_affined, dim=1)
            descriptor_low_affined = F.normalize(
                descriptor_low_affined, dim=1)
            distinctiveness_high_affined = calculate_distinctiveness(
                descriptor_high_affined)
            distinctiveness_high_affined = (
                distinctiveness_high_affined.squeeze())
            distinctiveness_high_affined[0:2,:] = 0
            distinctiveness_high_affined[-2::,:] = 0
            distinctiveness_high_affined[0:2,:] = 0
            distinctiveness_high_affined[:,-2::] = 0
            distinctiveness_low_affined = calculate_local_distinctiveness(
                descriptor_low_affined)
            distinctiveness_low_affined = distinctiveness_low_affined.squeeze()
            distinctiveness_low_affined[0:8,:] = 0
            distinctiveness_low_affined[-8::,:] = 0
            distinctiveness_low_affined[:,0:8] = 0
            distinctiveness_low_affined[:,-8::] = 0

        # Invers affine resultmap high.
        num_map_high_affined = np.ones([height_map_high, width_map_high])
        num_map_high_affined[0:2,:] = 0
        num_map_high_affined[-2::,:] = 0
        num_map_high_affined[:,0:2] = 0
        num_map_high_affined[:,-2::] = 0
        center_map_high = (width_map_high * 0.5 + 0.5,
                           height_map_high * 0.5 + 0.5)
        matrix_map_high = get_affine_matrix(
            center_map_high, *affine_params_map_high)
        distinctiveness_high_affined = TF.to_pil_image(
            np.expand_dims(distinctiveness_high_affined, 2).astype(np.float32))
        distinctiveness_high_affined_inverse = (
            distinctiveness_high_affined.transform(
                (width_map_high, height_map_high), Image.AFFINE,
                matrix_map_high,
                resample=Image.NEAREST, fillcolor=fillcolor))
        distinctiveness_high_affined_inverse = TF.to_tensor(
            distinctiveness_high_affined_inverse).numpy()

        num_map_high_affined = TF.to_pil_image(
            np.expand_dims(num_map_high_affined, 2).astype(np.float32))
        num_map_high_affined_inverse = num_map_high_affined.transform(
            (width_map_high, height_map_high), Image.AFFINE, matrix_map_high,
            resample=Image.NEAREST, fillcolor=fillcolor)
        num_map_high_affined_inverse = TF.to_tensor(
            num_map_high_affined_inverse).numpy()

        distinctiveness_high_affined_inverse = (
            distinctiveness_high_affined_inverse.squeeze())
        num_map_high_affined_inverse = num_map_high_affined_inverse.squeeze()

        distinctiveness_high += (distinctiveness_high_affined_inverse *
            num_map_high_affined_inverse)
        num_map_high += num_map_high_affined_inverse

        # Invers affine resultmap low.
        num_map_low_affined = np.ones([height_map_low, width_map_low])
        num_map_low_affined[0:8,:] = 0
        num_map_low_affined[-8::,:] = 0
        num_map_low_affined[:,0:8] = 0
        num_map_low_affined[:,-8::] = 0
        center_map_low = (width_map_low * 0.5 + 0.5,
                          height_map_low * 0.5 + 0.5)
        matrix_map_low = get_affine_matrix(
            center_map_low, *affine_params_map_low)
        distinctiveness_low_affined = TF.to_pil_image(
            np.expand_dims(distinctiveness_low_affined, 2).astype(np.float32))
        distinctiveness_low_affined_inverse = (
            distinctiveness_low_affined.transform(
                (width_map_low, height_map_low), Image.AFFINE, matrix_map_low,
                resample=Image.NEAREST, fillcolor=fillcolor))
        distinctiveness_low_affined_inverse = TF.to_tensor(
            distinctiveness_low_affined_inverse).numpy()

        num_map_low_affined = TF.to_pil_image(
            np.expand_dims(num_map_low_affined, 2).astype(np.float32))
        num_map_low_affined_inverse = num_map_low_affined.transform(
            (width_map_low, height_map_low), Image.AFFINE, matrix_map_low,
            resample=Image.NEAREST, fillcolor=fillcolor)
        num_map_low_affined_inverse = TF.to_tensor(
            num_map_low_affined_inverse).numpy()

        distinctiveness_low_affined_inverse = (
            distinctiveness_low_affined_inverse.squeeze())
        num_map_low_affined_inverse = num_map_low_affined_inverse.squeeze()

        distinctiveness_low += (
            distinctiveness_low_affined_inverse * num_map_low_affined_inverse)
        num_map_low += num_map_low_affined_inverse

    distinctiveness_high = distinctiveness_high / num_map_high
    distinctiveness_high = cv2.resize(distinctiveness_high,
            (width_map_low, height_map_low), interpolation=cv2.INTER_LINEAR)
    distinctiveness_low = distinctiveness_low / num_map_low

    distinctiveness = 0.5 * (distinctiveness_low + distinctiveness_high)
    return distinctiveness

def get_similarity(
    model, img_list, keypoints_dir, confidence_threshold,
    threshold_sparse_points, down_ratio_descriptor,
    device = torch.device('cpu'),
    num_sparses = [0], num_points = [0], num_pixels = [0], index = 0):
    model = model.to(device)
    num_imgs_with_sparse_points = 0
    num_points_sum = 0
    num_pixels_sum = 0
    for img_path in img_list:
        torch.cuda.empty_cache()
        _, img_name = os.path.split(img_path)
        keypoints_path = os.path.join(keypoints_dir, img_name)
        distinctiveness = get_similarity_affine_adaption(
            model, img_path, down_ratio_descriptor, num_affines = 5)
        # NMS only process one input map each time.
        keypoints_map = non_maximum_suppress(
            distinctiveness, confidence_threshold = confidence_threshold,
            maxinum_points = 1000, radius = 8)
        # Save the keypoints into numpy file.
        np.save(keypoints_path, keypoints_map)
        if keypoints_map.sum() < threshold_sparse_points:
            num_imgs_with_sparse_points += 1
        num_points_sum += keypoints_map.sum()
        num_pixels_sum += keypoints_map.size
    num_sparses[index] = num_imgs_with_sparse_points
    num_points[index] = num_points_sum
    num_pixels[index] = num_pixels_sum
    return num_imgs_with_sparse_points, num_points_sum, num_pixels_sum

