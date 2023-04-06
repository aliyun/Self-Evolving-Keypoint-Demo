
# First party packages.
import net
import sekd_em

# Standard packages.
import argparse
import datetime
import logging
import os
import shutil

# Third party packages.
import torch
from torch.utils.tensorboard import SummaryWriter
import yaml

def parse_args():
    # Process args.
    parser = argparse.ArgumentParser(description='PyTorch SEKD')
    parser.add_argument(
        '--exp_dir', default='data/exp',
        help='The root folder to save experiment results.')
    parser.add_argument(
        '--experiment_name', default= 'sekd_em',
        help='The folder to save experiment results.')
    parser.add_argument(
        '--dataset_config', type=str, default='src/config/dataset_config.yaml',
        help='Config file of training datasets.')
    parser.add_argument(
        '--sub_set', type=int, default=1,
        help='Oly use 1/sub_set data.')

    parser.add_argument(
        '--use_cuda', type=bool, default=True,
        help='If use cuda for acceleration (default: True).')
    parser.add_argument(
        '--gpu_ids', type=str, default='0',
        help='GPU id(s) used for acceleration.')
    parser.add_argument(
        '--num_processes_each_gpu', type=str, default='1',
        help='Num processes for each gpu.')

    parser.add_argument(
        '--optimizer', default='adam', type=str, metavar='OPT',
        help='The optimizer to use (default: adam).')
    parser.add_argument(
        '--batch_size', type=int, default=8, metavar='BS',
        help='Input batch size for training.')
    parser.add_argument(
        '--num_workers', default=0, type=int,
        help='Number of workers to be created during loading data.')
    parser.add_argument(
        '--lr', type=float, default=1e-3, metavar='LR',
        help='The initial learning rate.')
    parser.add_argument(
        '--weight_decay', default=1e-8, type=float, metavar='WDecay',
        help='The weight decay.')

    parser.add_argument(
        '--iterations_em', type=int, default=2,
        help='The number of em-iterations to train SEKD.')
    parser.add_argument(
        '--epoches_detector', type=int, default=2,
        help='Epoches to train SEKD detector in each em iteration.')
    parser.add_argument(
        '--epoches_descriptor', type=int, default=2,
        help='Epoches to train SEKD descriptor in each em iteration.')

    parser.add_argument(
        '--max_keypoints', type = int, default = 2000,
        help='Maximum number of keypoints detected or calculated.')

    parser.add_argument(
        '--nms_radius', type = int, default = 4,
        help='Radius of non maximum suppression algorithm.')

    parser.add_argument(
        '--num_refs', type = int, default = 20,
        help='Maximum number of keypoints detected or calculated.')

    parser.add_argument(
        '--detector_loss', default='focal_loss', type=str,
        help=('detector loss: {focal_loss, l2_loss}.'))

    parser.add_argument(
        '--model_name', default='SEKD', type=str,
        help=('Method to evaluate: {SEKD, SEKDLarge, SEKDScale, SEKDMotion, ' +
              'SEKDMobile, SEKDMobileCV2, SEKDMobile2CV2, SEKDUNet}.'))

    parser.add_argument(
        '--height', type = int, default = 240,
        help='Input image height during training.')
    parser.add_argument(
        '--width', type = int, default = 320,
        help='Input image width during training.')
    parser.add_argument(
        '--down_ratio_descriptor', type = int, default = 1,
        help='Down ratio of output descriptor map.')
    parser.add_argument(
        '--threshold_sparse_points', type = int, default = 50,
        help='Below this number, the image is regarded as with sparse points.')
    parser.add_argument(
        '--confidence_threshold_detector', type = float, default = 0.4,
        help='Confidence threshold of detector.')
    parser.add_argument(
        '--confidence_threshold_reliability', type = float, default = 0.9,
        help='Confidence threshold when compute keypoints.')

    args = parser.parse_args()

    with open(args.dataset_config, 'r') as f:
        args.dataset_info_list = yaml.load(f, Loader = yaml.FullLoader)
    if args.use_cuda:
        gpu_ids = [
            int(index) for index in args.gpu_ids.replace(',', ' ').split()]
        args.gpu_ids = gpu_ids
        num_processes_each_gpu = [
            int(index) for index in
            args.num_processes_each_gpu.replace(',', ' ').split()]
        args.num_processes_each_gpu = num_processes_each_gpu

    args.weight_class = [1. for i in range(2)]
    args.weight_loss_repeatability = 1.
    args.use_all_points_for_descriptor = False

    return args

def train_sekd_em(args, model, log_writer, dir_model):
    # Alternately train the detector and descriptor.
    for index_em in range(args.iterations_em):
        logging.info("EM iteration: " + str(index_em))
        # Step E: train the descriptor only.
        # Step E-1. Get the keypoints using the trained detector.
        if args.use_cuda:
            torch.cuda.empty_cache()

        if index_em != 0:
            if index_em == 0:
                sekd_em.detect_keypoints(args, index_em, log_writer, random = True)
            else:
                sekd_em.detect_keypoints(args, index_em, log_writer, model)

            # Step E-2. Train the descritpor on the detected keypoints using
            # triplet loss while preserving detector unchanged.
            if args.use_cuda:
                torch.cuda.empty_cache()
            if index_em == 0:
                model = sekd_em.update_descriptor(
                    args, model, dir_model, index_em, log_writer,
                    weight_detector=0)
            else:
                model = sekd_em.update_descriptor(
                    args, model, dir_model, index_em, log_writer)

        # Step M: train the detector only.
        # Step M-1. Get the soft&hard keypoints label according to reliability
        # of the current descriptors.
        if args.use_cuda:
            torch.cuda.empty_cache()
        sekd_em.compute_keypoints(args, model, index_em, log_writer)

        # Step M-2. Train detector while preserving the descriptor unchanged.
        if args.use_cuda:
            torch.cuda.empty_cache()
        model = sekd_em.update_detector(
            args, model, dir_model, index_em, log_writer)

    return

if __name__ == '__main__':
    logging.basicConfig(
        format='%(asctime)s: %(levelname)-6s ' +
                '[%(filename)s:%(lineno)d] %(message)s',
        datefmt='%d-%m-%Y:%H:%M:%S',
        level=logging.INFO)
    args = parse_args()
    logging.info('Args: {0}'.format(args))
    dir_result = '{}/{}/{}/'.format(
        args.exp_dir, args.experiment_name,
        datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    dir_log = os.path.join(dir_result, 'log')
    dir_model = os.path.join(dir_result, 'model')

    # Create result directory.
    if not os.path.exists(dir_result):
        os.makedirs(dir_result)
    # Create tensorboard log writer at loggin directory.
    log_writer = SummaryWriter(dir_log)
    # Create model save directory.
    if not os.path.exists(dir_model):
        os.makedirs(dir_model)

    model = net.get_sekd_model(args.model_name)

    train_sekd_em(args, model, log_writer, dir_model)

    log_writer.close()
