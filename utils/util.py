
# Standard.
import logging
import math
import os
import random

# Third party.
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

def get_weight_map(score_map_init):
    # Parameters.
    kernel_size_half = 2
    kernel = np.zeros([2*kernel_size_half+1, 2*kernel_size_half+1])
    kernel[kernel_size_half, kernel_size_half] = 1
    kernel = scifilters.gaussian_filter(kernel, sigma=1)
    kernel = kernel / kernel[kernel_size_half, kernel_size_half]
    kernel = torch.tensor(
        kernel, device=score_map_init.device, dtype=torch.float32).detach(
            ).view([1, 1, 2*kernel_size_half+1, 2*kernel_size_half+1])
    # Process.
    num_maps, height, width = score_map_init.shape
    score_map = score_map_init.clone().detach().type(torch.float32).view(
        [num_maps, 1, height, width])
    weight_map = torch.ones(
        [num_maps, 1, height, width], device=score_map.device)
    weight_map_near_keypoint = F.conv2d(
        score_map, kernel, padding=kernel_size_half)
    weight_map = weight_map - weight_map_near_keypoint + score_map
    weight_map = weight_map.view([num_maps, height, width])
    return weight_map

def reformat_loc_gt(loc_gt_init, cell):
    batch_size, height, width = loc_gt_init.shape
    detector_label = torch.zeros(
        [int(batch_size * height/cell * width/cell),
        cell ** 2, 1], device = loc_gt_init.device, dtype=loc_gt_init.dtype)
    for i_cell in range(cell ** 2):
        detector_label[:, i_cell] = i_cell + 1
    loc_gt_init = loc_gt_init.reshape(
        [batch_size, int(height/cell), cell,
         int(width/cell), cell]).transpose(2,3).reshape([-1, 1, cell ** 2])
    loc_gt_num = loc_gt_init.sum(dim=2).detach().cpu().numpy().astype(np.int64)
    xs, ys = np.where(loc_gt_num > 1)
    for i in range(xs.shape[0]):
        x = xs[i]
        y = ys[i]
        i_label = random.randint(0, loc_gt_num[x,y]-1)
        for z in range(cell ** 2):
            if loc_gt_init[x,y,z] == 1:
                if i_label != 0:
                    loc_gt_init[x,y,z] = 0
                i_label -= 1
    loc_gt = torch.bmm(loc_gt_init, detector_label)
    loc_gt = loc_gt.reshape([batch_size, int(height/cell), int(width/cell)])
    return loc_gt

def get_loc_gt(keypoints_maps, matrixs, height_out=None, width_out=None):
    num_maps, height, width = keypoints_maps.shape
    if height_out is None:
        height_out = height
    if width_out is None:
        width_out = width
    loc_gt = torch.zeros([num_maps, height_out, width_out],
            device = keypoints_maps.device, dtype = keypoints_maps.dtype)
    for i_map in range(num_maps):
        xs, ys = np.where(keypoints_maps[i_map,:,:].cpu().numpy() > 0)
        matrix = matrixs[i_map, :, :].cpu().numpy()
        coord_init = np.concatenate((xs, ys)).reshape([2, xs.shape[0]])
        coord_init_homo = np.ones([coord_init.shape[0]+1, coord_init.shape[1]])
        coord_init_homo[0:2, :] = coord_init

        coord_new = np.matmul(matrix, coord_init_homo)
        coord_new = coord_new.round().astype(np.int64)
        # Filter the points outside the image.
        valid_points = []
        for index_point in range(coord_new.shape[1]):
            if (coord_new[0][index_point] > 0 and
                coord_new[0][index_point] < height_out and
                coord_new[1][index_point] > 0 and
                coord_new[1][index_point] < width_out):
                valid_points.append(index_point)
        coord_new = coord_new[:, valid_points]
        loc_gt[i_map, coord_new[0, :], coord_new[1, :]] = 1
    return loc_gt

def get_all_keypoints_coord(
    height, width, matrix0_init, matrix1_init, down_ratio):
    # Get matrix to map.
    matrix0 = np.zeros([3,3])
    matrix0[0,:] = matrix0_init.detach().cpu().numpy()[0,:] / down_ratio
    matrix0[1,:] = matrix0_init.detach().cpu().numpy()[1,:] / down_ratio
    matrix0[2, 2] = 1

    matrix1 = np.zeros([3,3])
    matrix1[0,:] = matrix1_init.detach().cpu().numpy()[0,:] / down_ratio
    matrix1[1,:] = matrix1_init.detach().cpu().numpy()[1,:] / down_ratio
    matrix1[2, 2] = 1

    # Inverse matrix to map.
    a = matrix0[0,0]
    b = matrix0[0,1]
    c = matrix0[0,2]
    d = matrix0[1,0]
    e = matrix0[1,1]
    f = matrix0[1,2]
    matrix0_inverse = np.array(
        [[e, -b, b*f - c*e], [-d, a, c*d - a*f], [0, 0, 0]]) / (a*e - b*d)
    matrix0_inverse[2, 2] = 1
    a = matrix1[0,0]
    b = matrix1[0,1]
    c = matrix1[0,2]
    d = matrix1[1,0]
    e = matrix1[1,1]
    f = matrix1[1,2]
    matrix1_inverse = np.array(
        [[e, -b, b*f - c*e], [-d, a, c*d - a*f], [0, 0, 0]]) / (a*e - b*d)
    matrix1_inverse[2, 2] = 1

    # Process each point.
    height_map = int(height / down_ratio)
    width_map = int(width / down_ratio)
    keypoints_map = np.ones([height_map, width_map])
    keypoints_map[:2, :] = 0
    keypoints_map[-2:, :] = 0
    keypoints_map[:, :2] = 0
    keypoints_map[:, -2:] = 0
    xs, ys = np.where(keypoints_map > 0)
    coord = np.concatenate((xs, ys)).reshape([2, xs.shape[0]])

    coord0 = np.ones([coord.shape[0]+1, coord.shape[1]])
    coord0[0:2, :] = coord
    coord0 = coord0.round().astype(np.int64)

    coord1 = np.matmul(matrix1, np.matmul(matrix0_inverse, coord0)
            ).round().astype(np.int64)
    coord1_inverse = np.matmul(matrix0, np.matmul(matrix1_inverse, coord1)
            ).round().astype(np.int64)

    # Select valid points.
    valid_points = coord1[0, :] > 0
    valid_points = valid_points & (coord1[0, :] < height_map)
    valid_points = valid_points & (coord1[1, :] > 0)
    valid_points = valid_points & (coord1[1, :] < width_map)
    valid_points = valid_points & (coord0[0, :] == coord1_inverse[0,:])
    valid_points = valid_points & (coord0[1, :] == coord1_inverse[1,:])

    coord0 = coord0[0:2, valid_points]
    coord1 = coord1[0:2, valid_points]

    coord0 = coord0.transpose(1, 0)
    coord1 = coord1.transpose(1, 0)

    return coord0, coord1

def get_keypoints_coord(keypoints_map, matrix0, matrix1, img_height, img_width):
    xs, ys = np.where(keypoints_map.cpu().numpy() > 0)
    matrix0 = matrix0.cpu().numpy()
    matrix1 = matrix1.cpu().numpy()

    coord = np.concatenate((xs, ys)).reshape([2, xs.shape[0]])
    coord_homo = np.ones([coord.shape[0]+1, coord.shape[1]])
    coord_homo[0:2, :] = coord

    coord0 = np.matmul(matrix0, coord_homo)
    coord0 = coord0.round().astype(np.int64)
    coord1 = np.matmul(matrix1, coord_homo)
    coord1 = coord1.round().astype(np.int64)

    # Filter the points outside the image.
    valid_points = coord0[0, :] > 0
    valid_points = valid_points & (coord0[0, :] < img_height)
    valid_points = valid_points & (coord0[1, :] > 0)
    valid_points = valid_points & (coord0[1, :] < img_width)
    valid_points = valid_points & (coord1[0, :] > 0)
    valid_points = valid_points & (coord1[0, :] < img_height)
    valid_points = valid_points & (coord1[1, :] > 0)
    valid_points = valid_points & (coord1[1, :] < img_width)

    coord0 = coord0[:, valid_points]
    coord1 = coord1[:, valid_points]

    coord0 = coord0.transpose(1, 0)
    coord1 = coord1.transpose(1, 0)

    return coord0, coord1

def get_common_keypoints_coord(keypoints_map0, grid_01):
    height, width = keypoints_map0.shape
    xs0, ys0 = torch.where(keypoints_map0 > 0)
    valid_points = torch.where(
        (grid_01[xs0, ys0, 0] > -1) &
        (grid_01[xs0, ys0, 0] < 1) &
        (grid_01[xs0, ys0, 1] > -1) &
        (grid_01[xs0, ys0, 1] < 1))
    xs0 = xs0[valid_points]
    ys0 = ys0[valid_points]
    xs1 = ((grid_01[xs0, ys0, 1] + 1.) / 2. * height).to(xs0.dtype)
    ys1 = ((grid_01[xs0, ys0, 0] + 1.) / 2. * width).to(xs0.dtype)
    coord0 = torch.cat((xs0.unsqueeze(1), ys0.unsqueeze(1)), dim = 1)
    coord1 = torch.cat((xs1.unsqueeze(1), ys1.unsqueeze(1)), dim = 1)
    return coord0, coord1

    xs0, ys0 = torch.where(keypoints_map0 > 0)
    valid_points = torch.where(correspondence_01[xs0, ys0, 0] > 0)
    xs0 = xs0[valid_points]
    ys0 = ys0[valid_points]
    xs1 = correspondence_01[xs0, ys0, 0]
    ys1 = correspondence_01[xs0, ys0, 1]
    coord0 = torch.cat((xs0.unsqueeze(1), ys0.unsqueeze(1)), dim = 1)
    coord1 = torch.cat((xs1.unsqueeze(1), ys1.unsqueeze(1)), dim = 1)
    return coord0, coord1

def get_loc_gt_neighbour(loc_gt0, correspondence_01):
    loc_gt1 = torch.zeros(
        loc_gt0.shape, dtype = loc_gt0.dtype, device = loc_gt0.device)
    for idx_sample in range(loc_gt0.shape[0]):
        xs0, ys0 = torch.where(loc_gt0[idx_sample, :, :] > 0)
        xs1 = correspondence_01[idx_sample, xs0, ys0, 0]
        ys1 = correspondence_01[idx_sample, xs0, ys0, 1]
        loc_gt1[idx_sample, xs1, ys1] = 1
        loc_gt1[idx_sample, 0, 0] = 0
    return loc_gt1

# use non maximax suppress (nms) to find robust keypoints
def non_maximum_suppress(
    responce_map, confidence_threshold = 0.2,
    maxinum_points = 500, radius = 4):
#    responce_map_init[:10, :] = 0
#    responce_map_init[-10:, :] = 0
#    responce_map_init[:, :10] = 0
#    responce_map_init[:, -10:] = 0
    # Speed up NMS using maxpooling.
    with torch.no_grad():
        #responce_map = torch.from_numpy(responce_map_init
        #    ).unsqueeze(0).unsqueeze(0)
        responce_map = responce_map.unsqueeze(0).unsqueeze(0)
        responce_map_pooling = torch.nn.functional.max_pool2d(
            responce_map, kernel_size = radius * 2 + 1,
            stride = 1, padding = radius).squeeze()
        responce_map = responce_map.squeeze()
        responce_map_pooling = responce_map_pooling.squeeze()
        xs, ys = torch.where((responce_map > confidence_threshold) & (
            responce_map == responce_map_pooling))
        grid = torch.zeros(responce_map.shape, dtype=torch.int32)
        if xs.shape[0] > maxinum_points:
            indicates = responce_map[xs, ys].sort(descending=True)[1]
            indicates = indicates[:maxinum_points]
            grid[xs[indicates], ys[indicates]] = 1
        else:
            grid[xs, ys] = 1
        grid = grid.numpy().astype(np.int)
    return grid

    responce_map = responce_map_init.copy()
    height, width = responce_map.shape
    responce_map[0:2*radius, :] = 0
    responce_map[:, 0:2*radius] = 0
    responce_map[height-2*radius:height, :] = 0
    responce_map[:, width-2*radius:width] = 0
    xs, ys = np.where(responce_map > confidence_threshold)
    in_corners = np.zeros((3, len(xs))) # Populate point data sized 3xN.
    in_corners[0, :] = xs
    in_corners[1, :] = ys
    in_corners[2, :] = responce_map[xs, ys]

    grid = np.zeros((height, width)).astype(np.int) # Track NMS data.
    # Store indices of points.
    inds_map_to_point = np.zeros((height, width)).astype(int)
    # Sort by confidence and round to nearest int.
    inds_sort = np.argsort(-in_corners[2,:])
    corners = in_corners[:,inds_sort]
    rcorners = corners[:2,:].round().astype(int) # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
        return grid
    if rcorners.shape[1] == 1:
        grid[rcorners[0,0], rcorners[1,0]] = 1
        return grid
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
        grid[rcorners[0,i], rcorners[1,i]] = -1
        inds_map_to_point[rc[0], rc[1]] = i
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
        if grid[rc[0], rc[1]] == -1: # If not yet suppressed.
            grid[rc[0]-radius:rc[0]+radius+1, rc[1]-radius:rc[1]+radius+1] = 0
            grid[rc[0], rc[1]] = 1
            count += 1
            if count == maxinum_points:
                break
    grid[grid < 0] = 0

    return grid

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)

def list_files(root, extensions=None):
    files = []
    root = os.path.expanduser(root)
    if not os.path.isdir(root):
        return files

    for dirpath, _, filenames in sorted(os.walk(root)):
        for filename in sorted(filenames):
            if (None == extensions or
                has_file_allowed_extension(filename, extensions)):
                path = os.path.join(dirpath, filename)
                files.append(path)

    return files

def global_orthogonal_regularization(anchor, negative):

    neg_dis = torch.sum(torch.mul(anchor,negative),1)
    dim = anchor.size(1)
    gor = (torch.pow(torch.mean(neg_dis),2) +
           torch.clamp(torch.mean(torch.pow(neg_dis,2))-1.0/dim, min=0.0))

    return gor

def distance_matrix_vector(anchor, positive):
    """Given batch of anchor descriptors and
       positive descriptors calculate distance matrix.
    """

    d1_sq = torch.sum(anchor * anchor, dim=1).unsqueeze(-1)
    d2_sq = torch.sum(positive * positive, dim=1).unsqueeze(-1)

    dist_square = (
        d1_sq.repeat(1, positive.size(0)) +
        torch.t(d2_sq.repeat(1, anchor.size(0))) -
        2.0 * torch.bmm(
            anchor.unsqueeze(0), torch.t(positive).unsqueeze(0)).squeeze(0))
    dist_square = torch.clamp(dist_square, min = 0.0)

    return torch.sqrt(dist_square)

def distance_vectors_pairwise(anchor, positive, negative = None):
    """Given batch of anchor descriptors and positive descriptors
       calculate distance matrix.
    """

    a_sq = torch.sum(anchor * anchor, dim=1)
    p_sq = torch.sum(positive * positive, dim=1)

    eps = 1e-8
    d_a_p = torch.sqrt(
        a_sq + p_sq - 2*torch.sum(anchor * positive, dim = 1) + eps)
    if negative is not None:
        n_sq = torch.sum(negative * negative, dim=1)
        d_a_n = torch.sqrt(
            a_sq + n_sq - 2*torch.sum(anchor * negative, dim = 1) + eps)
        d_p_n = torch.sqrt(
            p_sq + n_sq - 2*torch.sum(positive * negative, dim = 1) + eps)
        return d_a_p, d_a_n, d_p_n
    return d_a_p

# resize image to size 32x32
cv2_scale36 = lambda x: cv2.resize(
    x, dsize=(36, 36), interpolation=cv2.INTER_LINEAR)
cv2_scale = lambda x: cv2.resize(
    x, dsize=(32, 32), interpolation=cv2.INTER_LINEAR)
# reshape image
np_reshape = lambda x: np.reshape(x, (32, 32, 1))

class L1Norm(nn.Module):
    def __init__(self):
        super(L1Norm,self).__init__()
        self.eps = 1e-10
    def forward(self, x):
        norm = torch.sum(torch.abs(x), dim = 1) + self.eps
        x= x / norm.expand_as(x)
        return x


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False

def get_affine_params(
    degrees, translate, scale_ranges, shears,
    img_size = (1, 1), random_method='uniform'):
    """Get parameters for affine transformation
    Returns:
        sequence: params to be passed to the affine transformation
    """
    support_methods = ['uniform', 'normal', 'gaussian']
    assert(random_method in support_methods)
    if random_method == 'uniform':
        angle = random.uniform(degrees[0], degrees[1])
    elif random_method in ['normal', 'gaussian']:
        number = (max(min(random.normalvariate(0,1), 2.), -2.) + 2.) / 4.
        angle = number * (degrees[1] - degrees[0]) + degrees[0]
    if translate is not None:
        max_dx = translate[0] * img_size[0]
        max_dy = translate[1] * img_size[1]
        if random_method == 'uniform':
            translations = (random.uniform(-max_dx, max_dx),
                            random.uniform(-max_dy, max_dy))
        elif random_method in ['normal', 'gaussian']:
            number0 = (max(min(random.normalvariate(0,1), 2.), -2.) + 2.) / 4.
            number1 = (max(min(random.normalvariate(0,1), 2.), -2.) + 2.) / 4.
            translations = (number0 * (2. * max_dx) - max_dx,
                            number1 * (2. * max_dy) - max_dy)
    else:
        translations = (0, 0)

    if scale_ranges is not None:
        if random_method == 'uniform':
            scale_x = random.uniform(scale_ranges[0], scale_ranges[1])
            scale_y = random.uniform(scale_ranges[0], scale_ranges[1])
            scales = (scale_x, scale_y)
        elif random_method in ['normal', 'gaussian']:
            number = (max(min(random.normalvariate(0,1), 2.), -2.) + 2.) / 4.
            scale_x = (number * (scale_ranges[1] - scale_ranges[0]) +
                     scale_ranges[0])
            scale_y = (number * (scale_ranges[1] - scale_ranges[0]) +
                     scale_ranges[0])
            scales = (scale_x, scale_y)
    else:
        scales = (1.0, 1.0)

    if shears is not None:
        if random_method == 'uniform':
            shear = random.uniform(shears[0], shears[1])
        elif random_method in ['normal', 'gaussian']:
            number = (max(min(random.normalvariate(0,1), 2.), -2.) + 2.) / 4.
            shear = number * (shears[1] - shears[0]) + shears[0]
    else:
        shear = 0.0

    return angle, translations, scales, shear

def get_affine_matrix(center, angle, translate, scale, shear):
    # Helper method to compute matrix for affine transformation

    # We need compute affine transformation matrix: M = T * C * RSS * C^-1
    # where T is translation matrix: [1, 0, tx | 0, 1, ty | 0, 0, 1]
    #       C is translation matrix to keep center: [1, 0, cx | 0, 1, cy | 0, 0, 1]
    #       RSS is rotation with scale and shear matrix
    #       RSS(a, scale, shear) = [ cos(a)*scale    -sin(a + shear)*scale     0]
    #                              [ sin(a)*scale    cos(a + shear)*scale     0]
    #                              [     0                  0          1]

    # As the coordinated system of PIL is (width, height), the parameters should be (width, height).

    # However, for convenient, the result 2 * 3 matrix is (height, width)

    angle = math.radians(angle)
    shear = math.radians(shear)
    shear = [shear, 0]

    # Rotation matrix with scale and shear
    matrix = [
        math.cos(angle + shear[1]), -math.sin(angle+shear[0]), 0,
        math.sin(angle + shear[1]), math.cos(angle+shear[0]), 0
    ]
    matrix = [scale * m for m in matrix]

    # Apply inverse of center translation: RSS * C^-1
    matrix[2] = -matrix[0] * center[0] - matrix[1] * center[1]
    matrix[5] = -matrix[3] * center[0] - matrix[4] * center[1]

    # Apply translation and center translation: T * C * RSS * C^-1
    matrix[2] += translate[0] + center[0]
    matrix[5] += translate[1] + center[1]

    matrix_width_height = np.array(matrix).reshape([2,3])
    matrix_height_width = np.zeros([2,3])
    matrix_height_width[0,0] = matrix_width_height[1,1]
    matrix_height_width[0,1] = matrix_width_height[1,0]
    matrix_height_width[0,2] = matrix_width_height[1,2]
    matrix_height_width[1,0] = matrix_width_height[0,1]
    matrix_height_width[1,1] = matrix_width_height[0,0]
    matrix_height_width[1,2] = matrix_width_height[0,2]

    return matrix_height_width.astype(np.float)

def get_affine_matrix_inverse(center, angle, translate, scale, shear):
    # Return the inverse matrix calculated by get_affine_matrix()

    matrix = get_affine_matrix(center, angle, translate, scale, shear)

    a = matrix[0, 0]
    b = matrix[0, 1]
    c = matrix[0, 2]
    d = matrix[1, 0]
    e = matrix[1, 1]
    f = matrix[1, 2]
    matrix_inverse = (np.array([[e, -b, b*f - c*e], [-d, a, c*d - a*f]]) /
        (a*e - b*d))

    return matrix_inverse.astype(np.float)

def get_affine_matrix_theta(angle, translate, scale, shear):
    angle = math.radians(angle)
    shear = math.radians(shear)

    # Rotation matrix with scale and shear
    matrix = [
        [
            scale[0] * math.cos(angle),
            - scale[1] * math.sin(angle + shear),
            translate[0],
        ],
        [
            scale[0] * math.sin(angle),
            scale[1] * math.cos(angle + shear),
            translate[1],
        ],
    ]
    matrix = np.array(matrix)

    return matrix.astype(np.float)

def get_affine_matrix_theta_inverse(angle, translate, scale, shear):
    # Return the inverse matrix calculated by get_affine_matrix()

    matrix = get_affine_matrix_theta(angle, translate, scale, shear)

    a = matrix[0, 0]
    b = matrix[0, 1]
    c = matrix[0, 2]
    d = matrix[1, 0]
    e = matrix[1, 1]
    f = matrix[1, 2]
    matrix_inverse = (np.array([[e, -b, b*f - c*e], [-d, a, c*d - a*f]]) /
        (a*e - b*d))

    return matrix_inverse.astype(np.float)

def blur_img_2d_heterogeneous(img_in, blur_kernel):
    # TODO: To be completed.
    dims = img_in.dim()
    if dims != 2:
        print("Do not support blur image with dims {0} now.".format(dims))
    h, w = img_in.shape
    num_pixels, kernel_size = blur_kernel.shape
    kernel_diameter = int(kernel_size ** 0.5)
    kernel_radius = int((kernel_diameter - 1) / 2)
    # Blur the input image.
    img_in_col = im2col(img_in)
    img_blur = torch.mm(blur_kernel, img_in)
    img_blur = img_blur.reshape([h,w])
    return img_blur

def generate_random_blur_kernel_heterogeneous(h, w):
    # TODO: To be completed.
    # Random generate parameters.
    max_offset = 0.02
    x_offset = random.uniform(0, max_offset)
    y_offset = random.uniform(0, max_offset)
    z_offset = random.uniform(-max_offset, max_offset)
    z_rotation = random.uniform(-max_offset*math.pi, max_offset*math.pi)
    # Calculate blur kernel.
    num_pixels = h * w
    kernel_radius = math.floor(max(h,w) * max_offset / 2)
    kernel_diameter = kernel_radius * 2 + 1
    kernel_size = kernel_diameter ** 2
    blur_kernel = torch.zeros([h, w, kernel_diameter, kernel_diameter])
    for i_h in range(h):
        for i_w in range(w):
            blur_kernel[i_h, i_w, :, :] = 0

def blur_img_2d_homogeneous(img_in, blur_kernel):
    dims = img_in.dim()
    if dims != 2:
        print("Do not support blur image with dims {0} now.".format(dims))
    blur_kernel = blur_kernel.to(device=img_in.device)
    h, w = img_in.shape
    kernel_diameter, _ = blur_kernel.shape
    kernel_radius = int((kernel_diameter - 1) / 2)
    img_blur = torch.nn.functional.conv2d(img_in.reshape([1,1,h,w]),
            blur_kernel.reshape([1,1,kernel_diameter,kernel_diameter]),
            padding=kernel_radius)
    return img_blur

def generate_random_blur_kernel_homogeneous(h, w):
    # Random generate parameters.
    max_h_w = max(h,w)
    max_offset = 0.01
    max_var = max_h_w*max_offset
    h_offset = math.ceil(random.uniform(0, max_offset) * max_h_w)
    w_offset = math.ceil(random.uniform(-max_offset, max_offset * max_h_w))
    gaussian_offset = random.uniform(0, max_offset)
    hw_var = random.uniform(0, max_var)
    gaussian_var = random.uniform(0, 0.2*max_var)
    kernel_radius = math.ceil(max_h_w * max_offset)
    kernel_diameter = kernel_radius * 2 + 1
    blur_kernel = torch.zeros([kernel_diameter, kernel_diameter])
    for i_h in range(kernel_diameter):
        for i_w in range(kernel_diameter):
            blur_kernel[i_h][i_w] = (1./(2*math.pi*gaussian_var)* math.exp(
                -((i_h-kernel_radius)**2 +
                  (i_w-kernel_radius)**2)/(2*gaussian_var)))
    for i_h in range(-h_offset+kernel_radius, h_offset+kernel_radius):
        i_w = int(-w_offset+kernel_radius +
                  (2*w_offset+1) * (i_h-kernel_radius+h_offset) /
                  (2*h_offset+1))
        blur_kernel[i_h][i_w] += (1./(2*math.pi*gaussian_var)* math.exp(
            -((i_h-kernel_radius)**2 + (i_w-kernel_radius)**2)/(2*hw_var)))
    blur_kernel = blur_kernel / blur_kernel.sum()
    return blur_kernel

def random_blur_img_2d(img_in):
    dims = img_in.dim()
    if dims != 2:
        print("Do not support blur image with dims {0} now.".format(dims))
    h, w = img_in.shape
    blur_kernel = generate_random_blur_kernel_homogeneous(h, w)
    return blur_img_2d_homogeneous(img_in, blur_kernel)

def random_blur_img_3d(tensor_in):
    dims = tensor_in.dim()
    if dims != 3:
        print("Do not support blur tensor with dims {0} now.".format(dims))
    c, h, w = tensor_in.shape
    blur_kernel = generate_random_blur_kernel_homogeneous(h, w)
    img_blur = torch.zeros([c,h,w], device = tensor_in.device)
    for i_channel in range(c):
        img_blur[i_channel,:,:] = blur_img_2d_homogeneous(
            tensor_in[i_channel,:,:], blur_kernel)
    return img_blur

def random_blur(tensor_in, is_one_image = False):
    dims = tensor_in.dim()
    if dims < 2 or dims > 4:
        print("Do not support blur tensor with dims {0} now.".format(dims))
        return tensor_in
    if dims == 2:
        return random_blur_img_2d(tensor_in)
    if dims == 3:
        if is_one_image:
            return random_blur_img_3d(tensor_in)
        else:
            n, h, w = tensor_in.shape
            img_blur = torch.zeros([n,h,w], device = tensor_in.device)
            for i_img in range(n):
                img_blur[i_img,:,:] = random_blur_img_2d(tensor_in[i_img,:,:])
            return img_blur
    if dims == 4:
        n, c, h, w = tensor_in.shape
        img_blur = torch.zeros([n,c,h,w], device = tensor_in.device)
        for i_img in range(n):
            img_blur[i_img,:,:,:] = random_blur_img_3d(tensor_in[i_img,:,:,:])
        return img_blur
    return tensor_in

