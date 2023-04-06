
# First party packages.
from datasets import HybridLoader
from loss import focal_loss
import utils

# Standard packages.
import copy
import logging

# Third party packages.
import numpy as np
import torch
import torch.nn.functional as F

def create_optimizer(model, args):
    # setup optimizer
    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(
            model.parameters(), lr = args.lr, momentum = 0.9,
            weight_decay = args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(), lr = args.lr, weight_decay = args.weight_decay)
    else:
        logging.critical(
            'Not supported optimizer: {0}'.format(args.optimizer))

    return optimizer

def update_detector(args, model, dir_model, index_em = 0, log_writer = None):
    ''' Only train the detector while try the best to preserve the descriptor
        unchanged.
    '''
    logging.info('Update detector.')
    model_init = copy.deepcopy(model)
    model_init.eval()
    model.train()
    if True == args.use_cuda:
        model_init = model_init.to(
            torch.device('cuda:{0}'.format(args.gpu_ids[0])))
        model = model.to(torch.device('cuda:{0}'.format(args.gpu_ids[0])))
    # Init data loader.
    train_loader = torch.utils.data.DataLoader(
        HybridLoader(args.dataset_info_list, args.height, args.width,
                     index_em % args.sub_set, args.sub_set),
        batch_size = args.batch_size, shuffle = True,
        num_workers = args.num_workers)
    # Init optimizer.
    optimizer = create_optimizer(model, args)
    # Init scheduler.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience = 1)
    # Train model to detect keypoints obtained from descriptor's reliability.
    for epoch in range(args.epoches_detector):
        logging.info('epoch: {0}'.format(epoch))
        # Load data.
        avg_loss = 0
        avg_loss_repeatability = 0
        avg_loss_detector = 0
        avg_loss_descriptor = 0
        num_batches = 0
        for batch_idx, data in enumerate(train_loader):
            imgs0 = data[0]
            imgs1 = data[1]
            gt_keypoints_maps0 = data[2]
            gt_keypoints_maps1 = data[3]
            gt_score_maps0 = data[4]
            gt_score_maps1 = data[5]
            grid_01 = data[6]
            if True == args.use_cuda:
                imgs0 = imgs0.to(
                    torch.device('cuda:{0}'.format(args.gpu_ids[0])))
                imgs1 = imgs1.to(
                    torch.device('cuda:{0}'.format(args.gpu_ids[0])))
                gt_keypoints_maps0 = gt_keypoints_maps0.to(
                    torch.device('cuda:{0}'.format(args.gpu_ids[0])))
                gt_keypoints_maps1 = gt_keypoints_maps1.to(
                    torch.device('cuda:{0}'.format(args.gpu_ids[0])))
                gt_score_maps0 = gt_score_maps0.to(
                        torch.device('cuda:{0}'.format(args.gpu_ids[0])))
                gt_score_maps1 = gt_score_maps1.to(
                        torch.device('cuda:{0}'.format(args.gpu_ids[0])))
                grid_01 = grid_01.to(
                        torch.device('cuda:{0}'.format(args.gpu_ids[0])))
            # Forward.
            outs = model.forward(imgs0)
            score0, descriptor0 = outs[0], outs[1]
            descriptor0 = F.normalize(descriptor0, dim = 1)
            outs = model.forward(imgs1)
            score1, descriptor1 = outs[0], outs[1]
            descriptor1 = F.normalize(descriptor1, dim = 1)
            with torch.no_grad():
                outs = model_init.forward(imgs0)
                descriptor0_init = outs[1]
                descriptor0_init = F.normalize(descriptor0_init, dim = 1)
                outs = model_init.forward(imgs1)
                descriptor1_init = outs[1]
                descriptor1_init = F.normalize(descriptor1_init, dim = 1)
            loc_gt0 = gt_keypoints_maps0.type(torch.long).clone()
            loc_gt1 = gt_keypoints_maps1.type(torch.long).clone()
            # Calculate loss.
            loss_descriptor = 0.1 * 0.5 * (
                F.mse_loss(descriptor0, descriptor0_init) +
                F.mse_loss(descriptor1, descriptor1_init))
            loss_repeatability = torch.tensor(
                0., device = loss_descriptor.device)
            for idx_sample in range(gt_keypoints_maps0.shape[0]):
                gt_keypoints_maps0[idx_sample,:,:] = 1
                coord0, coord1 = utils.get_common_keypoints_coord(
                    gt_keypoints_maps0[idx_sample, 0, :, :],
                    grid_01[idx_sample, :, :, :])
                if coord0.shape[0] < 2:
                    continue
                if args.detector_loss == 'focal_loss':
                    loc0_at_keypoints = score0[idx_sample, :, coord0[:, 0],
                        coord0[:, 1]].transpose(0,1)
                    loc1_at_keypoints = score1[idx_sample, :, coord1[:, 0],
                        coord1[:, 1]].transpose(0,1)
                    loc0_prob = F.softmax(loc0_at_keypoints, dim = 1)
                    loc1_prob = F.softmax(loc1_at_keypoints, dim = 1)
                    weight_repeatability = 0.5 * (
                        loc0_prob[:,1].clone().detach() +
                        loc1_prob[:,1].clone().detach()).unsqueeze(1)
                    loss_repeatability += (
                        args.weight_loss_repeatability * 0.5 * torch.mean(
                            weight_repeatability * (
                                F.kl_div(torch.log(loc0_prob+1e-8),
                                         loc1_prob, reduction='none') +
                                F.kl_div(torch.log(loc1_prob+1e-8),
                                         loc0_prob, reduction='none'))))
                elif args.detector_loss == 'l2_loss':
                    loc0_at_keypoints = score0[idx_sample, 0, coord0[:, 0],
                        coord0[:, 1]]
                    loc1_at_keypoints = score1[idx_sample, 0, coord1[:, 0],
                        coord1[:, 1]]
                    loss_repeatability += (
                        args.weight_loss_repeatability *
                        F.mse_loss(loc0_at_keypoints, loc1_at_keypoints))
            loss_repeatability /= gt_keypoints_maps0.shape[0]
            if args.detector_loss == 'focal_loss':
                loss_detector = 0.5 * (
                    focal_loss(
                        score0, loc_gt0, weight = args.weight_class, dim = 1) +
                    focal_loss(
                        score1, loc_gt1, weight = args.weight_class, dim = 1))
            elif args.detector_loss == 'l2_loss':
                loss_detector = 0.5 * (
                        F.mse_loss(score0[:, 0:1, :, :], gt_score_maps0) +
                        F.mse_loss(score1[:, 0:1, :, :], gt_score_maps1))
            loss = loss_detector + loss_repeatability + loss_descriptor
            # Backward.
            loss.backward()
            # Update weight.
            optimizer.step()
            optimizer.zero_grad()

            num_batches = num_batches + 1
            avg_loss += loss.detach().cpu().numpy()
            avg_loss_repeatability += loss_repeatability.detach().cpu().numpy()
            avg_loss_detector += loss_detector.detach().cpu().numpy()
            avg_loss_descriptor += loss_descriptor.detach().cpu().numpy()
        # Log avg_loss.
        avg_loss = avg_loss / num_batches
        avg_loss_repeatability = avg_loss_repeatability / num_batches
        avg_loss_detector = avg_loss_detector / num_batches
        avg_loss_descriptor = avg_loss_descriptor / num_batches
        logging.info('avg_loss: {0}'.format(avg_loss))
        if log_writer is not None:
            log_writer.add_scalars(
                'train_detector_loss',
                {'avg_loss': avg_loss,
                 'avg_loss_repeatability': avg_loss_repeatability,
                 'avg_loss_detector': avg_loss_detector,
                 'avg_loss_descriptor': avg_loss_descriptor},
                epoch + index_em * args.epoches_detector)
        torch.save(
            {'state_dict': model.state_dict()},
            '{0}/checkpoint_detector_{1}_{2}.pth'.format(
                dir_model, index_em, epoch))
        # Update learning rate.
        scheduler.step(avg_loss)

    model = model.cpu()
    return model

if __name__ == '__main__':
    print('Update detector')

