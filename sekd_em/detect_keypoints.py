
# First party packages.
from datasets import (
    get_dataset, PrefetchReader, TransformerColor, TransformerAffine)
import utils

# Standard packages.
import copy
import logging
import os
import shutil
import threading

# Third party packages.
import cv2
import numpy as np
import torch
import torch.multiprocessing as mp
torch.multiprocessing.set_start_method('spawn', force=True)
import torch.nn.functional as F
import torchvision.transforms.functional as TF

def detect_keypoints(args, index_em = 0, log_writer = None, model = None,
                     random = False):
    if random == True:
        detect_keypoints_random(args, index_em, log_writer)
    else:
        detect_keypoints_via_detector(args, index_em, log_writer, model)
    return

def get_detector_probability_affine_adaption(
    args, model, img_target, img_refs, grid_target2refs):

    _, _, height_target, width_target = img_target.shape
    # Process target image.
    torch.cuda.empty_cache()
    with torch.no_grad():
        outs = model.forward(img_target)
        score = outs[0]
        if args.detector_loss == 'focal_loss':
            score = F.softmax(score, dim = 1)
            score = score[0,1,:,:]
        elif args.detector_loss == 'l2_loss':
            score = score[0,0,:,:]
    num_map = torch.ones(score.shape).to(img_target.device)

    # Process refs with additional affine adaption.
    with torch.no_grad():
        outs = model.forward(img_refs)
        score_refs = outs[0]
        if args.detector_loss == 'focal_loss':
            score_refs = F.softmax(score_refs, dim = 1)
            score_refs = score_refs[:, 1:, :, :]
        elif args.detector_loss == 'l2_loss':
            score_refs = score_refs[:, 0:1, :, :]
        score_refs[:,:,0:28,:] = 0
        score_refs[:,:,-28::,:] = 0
        score_refs[:,:,:,0:28] = 0
        score_refs[:,:,:,-28::] = 0

        num_map_refs = torch.ones(score_refs.shape).to(img_target.device)
        num_map_refs[:,:,0:28,:] = 0
        num_map_refs[:,:,-28::,:] = 0
        num_map_refs[:,:,:,0:28] = 0
        num_map_refs[:,:,:,-28::] = 0

        score_refs_inverse = F.grid_sample(
            score_refs, grid_target2refs, padding_mode = 'zeros')
        num_map_refs_inverse = F.grid_sample(
            num_map_refs, grid_target2refs, padding_mode = 'zeros')

        score_refs_inverse = score_refs_inverse.sum(dim = 0).squeeze()
        num_map_refs_inverse = num_map_refs_inverse.sum(dim = 0).squeeze()

    score = score + score_refs_inverse
    num_map = num_map + num_map_refs_inverse
    score = score / num_map

    return score

class GetDetectKeypointsThread(threading.Thread):
    def __init__(self, args, dataset_reader, model, num_sparses,
                 num_points, num_pixels, index):
        threading.Thread.__init__(self)
        self.args = args
        self.dataset_reader = dataset_reader
        self.model = model
        self.num_sparses = num_sparses
        self.num_points = num_points
        self.num_pixels = num_pixels
        self.index = index

    def run(self):
        get_detect_keypoints(
            self.args, self.dataset_reader, self.model, self.num_sparses,
            self.num_points, self.num_pixels, self.index)
        return

def get_detect_keypoints(
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

        score = get_detector_probability_affine_adaption(
            args, model, img_target, img_refs, grid_target2refs)

        # Non maximum suppress.
        keypoints_map = utils.non_maximum_suppress(
            score,
            confidence_threshold = args.confidence_threshold_detector,
            maxinum_points = args.max_keypoints, radius = args.nms_radius)
        keypoints_map[0:1*args.nms_radius, :] = 0
        keypoints_map[:, 0:1*args.nms_radius] = 0
        keypoints_map[-1*args.nms_radius::, :] = 0
        keypoints_map[:, -1*args.nms_radius::] = 0

        num_points_sum += keypoints_map.sum()
        num_pixels_sum += keypoints_map.size
        if keypoints_map.sum() < args.threshold_sparse_points:
            num_imgs_with_sparse_points += 1
        np.savez(keypoints_path, keypoints_map = keypoints_map,
                 score = score.squeeze().detach().cpu().numpy())
        cv2.imwrite(keypoints_path + '_keypoints.png', keypoints_map.astype(np.uint8) * 255)
        cv2.imwrite(keypoints_path + '_score.png', score.squeeze().detach().cpu().numpy() * 255)

    num_sparses[index] = num_imgs_with_sparse_points
    num_points[index] = num_points_sum
    num_pixels[index] = num_pixels_sum
    return num_imgs_with_sparse_points, num_points_sum, num_pixels_sum

def detect_keypoints_via_detector(args, index_em, log_writer, model):
    # Get keypoints via detector.
    logging.info('Get keypoints via detector.')
    num_images = 0.
    num_imgs_with_sparse_points = 0
    num_points_sum = 0
    num_pixels_sum = 0
    model.eval()

    if not args.use_cuda:
        model = model.cpu()
        # Process each dataset.
        for dataset_info in args.dataset_info_list:
            logging.info('Processing dataset {0}'.format(dataset_info['name']))
            # Remove all files in keypoints dir.
            if os.path.isdir(dataset_info['root_path_keypoints']):
                shutil.rmtree(dataset_info['root_path_keypoints'])
            os.makedirs(dataset_info['root_path_keypoints'])
            dataset_reader = get_dataset(
                dataset_info,
                start = index_em % args.sub_set,
                stride = args.sub_set,
                device = torch.device('cpu'))
            num_images += len(dataset_reader)
            outs = get_detect_keypoints(
                args,
                dataset_reader,
                model)
            num_imgs_with_sparse_points += outs[0]
            num_points_sum += outs[1]
            num_pixels_sum += outs[2]
    else:
        num_gpus = len(args.gpu_ids)
        num_processes_each_gpu = args.num_processes_each_gpu
        num_processes = int(np.sum(num_processes_each_gpu))
        models_share = []
        devices = []
        for i_gpu in range(num_gpus):
            for i_gpu_process in range(num_processes_each_gpu[i_gpu]):
                model_share_tmp = type(model)()
                model_share_tmp.load_state_dict(model.state_dict())
                model_share_tmp.eval()
                device_tmp = torch.device(
                    'cuda:{0}'.format(args.gpu_ids[i_gpu]))
                model_share_tmp = model_share_tmp.to(device_tmp)
                devices.append(device_tmp)
                models_share.append(model_share_tmp)

        # Process each dataset.
        for dataset_info in args.dataset_info_list:
            logging.info('Processing dataset {0}'.format(dataset_info['name']))
            # Remove all files in keypoints dir.
            if os.path.isdir(dataset_info['root_path_keypoints']):
                shutil.rmtree(dataset_info['root_path_keypoints'])
            os.makedirs(dataset_info['root_path_keypoints'])
            num_sparses = [0 for i in range(num_processes)]
            num_points = [0 for i in range(num_processes)]
            num_pixels = [0 for i in range(num_processes)]
            processes = []
            for i_process in range(num_processes):
                dataset_reader = get_dataset(
                    dataset_info,
                    start = index_em % args.sub_set + i_process * args.sub_set,
                    stride = num_processes * args.sub_set,
                    device = devices[i_process])
                num_images += len(dataset_reader)
                p = GetDetectKeypointsThread(
                    args, dataset_reader, models_share[i_process],
                    num_sparses, num_points, num_pixels, i_process)
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            num_imgs_with_sparse_points += np.sum(num_sparses)
            num_points_sum += np.sum(num_points)
            num_pixels_sum += np.sum(num_pixels)
    avg_points = num_points_sum / num_images
    avg_pixels = num_pixels_sum / num_images
    args.weight_class[0] = avg_points / (avg_pixels - avg_points)

    logging.info(
        '{0} images are with less than {1} points via detector.'.format(
            num_imgs_with_sparse_points, args.threshold_sparse_points))
    logging.info(
        '{0} points are detected in an image averagely.'.format(avg_points))
    if log_writer is not None:
        log_writer.add_scalars(
            'detect_keypoints',
            {'num_imgs_with_sparse_points': num_imgs_with_sparse_points,
             'avg_points': avg_points},
            index_em)
        #img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        #log_writer.add_image(
        #    'Input_image_for_detector', np.expand_dims(img, axis = 0),
        #    index_em)
        #log_writer.add_image(
        #    'Detector_result', np.expand_dims(keypoints_map, axis = 0),
        #    index_em)

    return

def detect_keypoints_random(args, index_em, log_writer):
    # This function random get keypoints.
    logging.info('Get keypoints random.')
    # Process each dataset.
    num_imgs_with_sparse_points = 0
    sum_points = 0.
    sum_pixels = 0.
    num_images = 0.
    for dataset_info in args.dataset_info_list:
        logging.info('Processing dataset {0}'.format(dataset_info['name']))
        dataset_reader = get_dataset(
            dataset_info,
            start = index_em % args.sub_set,
            stride = args.sub_set)
        # Remove all files in keypoints dir.
        if os.path.isdir(dataset_reader.root_path_keypoints):
            shutil.rmtree(dataset_reader.root_path_keypoints)
        os.makedirs(dataset_reader.root_path_keypoints)
        num_images += len(dataset_reader)
        # Get keypoints random.
        for idx in range(len(dataset_reader)):
            image_path = dataset_reader.get_image_path(idx)
            keypoints_path = dataset_reader.get_keypoints_path(idx)
            # Load img file and calculate the keypoints.
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            keypoints_map = np.zeros(img.shape, dtype = np.float32)
            #xs = np.random.randint(0, keypoints_map.shape[0], args.max_keypoints)
            #ys = np.random.randint(0, keypoints_map.shape[1], args.max_keypoints)
            xs = np.random.randint(0, keypoints_map.shape[0], 1000)
            ys = np.random.randint(0, keypoints_map.shape[1], 1000)
            keypoints_map[xs, ys] = 1
            sum_points += keypoints_map.sum()
            sum_pixels += keypoints_map.size
            if keypoints_map.sum() < args.threshold_sparse_points:
                num_imgs_with_sparse_points += 1
            score = np.ones(img.shape, dtype = np.float32) * 0.5
            # Save the keypoints into numpy file.
            np.savez(keypoints_path, keypoints_map = keypoints_map,
                     score = score)
    avg_points = sum_points / num_images
    avg_pixels = sum_pixels / num_images
    args.weight_class[0] = avg_points / (avg_pixels - avg_points)

    logging.info(
        '{0} images are with less than {1} points via detector.'.format(
            num_imgs_with_sparse_points, args.threshold_sparse_points))
    logging.info(
        '{0} points are detected in an image averagely.'.format(avg_points))
    if log_writer is not None:
        log_writer.add_scalars(
            'detect_keypoints',
            {'num_imgs_with_sparse_points': num_imgs_with_sparse_points,
             'avg_points': avg_points},
            index_em)
        log_writer.add_image(
            'Input_image_for_detector', np.expand_dims(img, axis = 0),
            index_em)
        log_writer.add_image(
            'Detector_result', np.expand_dims(keypoints_map, axis = 0),
            index_em)
    return

if __name__ == '__main__':
    print('Detect keypoints')

