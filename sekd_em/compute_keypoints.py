
# First party packages.
from datasets import get_dataset
from .ratio import (
    get_ratio, get_ratio_affine_adaption,
    get_ratio_affine_adaption_local,
    get_ratio_affine_adaption_multi_scale)


# Standard packages.
import copy
import logging
import os
import shutil
import threading

# Third party packages.
import numpy as np
import torch
import torch.multiprocessing as mp
torch.multiprocessing.set_start_method('spawn', force=True)

class GetRatioThread(threading.Thread):
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
        get_ratio(self.args, self.dataset_reader, self.model, self.num_sparses,
                  self.num_points, self.num_pixels, self.index)
        return

def compute_keypoints(args, model, index_em = 0, log_writer=None):
    logging.info('Get keypoints via descriptor.')
    model.eval()
    # Get keypoints via reliability.
    num_images = 0
    num_imgs_with_sparse_points = 0
    num_points_sum = 0
    num_pixels_sum = 0
    if not args.use_cuda:
        model = model.cpu()
        for dataset_info in args.dataset_info_list:
            logging.info('Processing dataset {0}'.format(dataset_info['name']))
            # Remove all files in keypoints_dir
            if os.path.exists(dataset_info['root_path_keypoints']):
                shutil.rmtree(dataset_info['root_path_keypoints'])
            os.mkdir(dataset_info['root_path_keypoints'])

            dataset_reader = get_dataset(
                dataset_info,
                start = index_em % args.sub_set,
                stride = args.sub_set,
                device = torch.device('cpu'))
            num_images += len(dataset_reader)
            outs = get_ratio(
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

        for dataset_info in args.dataset_info_list:
            logging.info('Processing dataset {0}'.format(dataset_info['name']))
            # Remove all files in keypoints_dir
            if os.path.exists(dataset_info['root_path_keypoints']):
                shutil.rmtree(dataset_info['root_path_keypoints'])
            os.mkdir(dataset_info['root_path_keypoints'])

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
                p = GetRatioThread(
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
    args.weight_class[0] = (avg_points / (avg_pixels - avg_points))

    logging.info(
        '{0} images are with less than {1} points via reliability.'.format(
        num_imgs_with_sparse_points, args.threshold_sparse_points))
    logging.info(
        '{0} points are detected in an image averagely.'.format(avg_points))
    if log_writer is not None:
        log_writer.add_scalars(
            'get_keypoints_via_features',
            {'num_imgs_with_sparse_points': num_imgs_with_sparse_points,
             'avg_points': avg_points},
            index_em)
    return

if __name__ == '__main__':
    print('Compute keypoints')

