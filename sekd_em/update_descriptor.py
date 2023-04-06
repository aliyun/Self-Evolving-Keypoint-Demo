
# First party packages.
from datasets import HybridLoader
from loss import loss_triplet
import utils

# Standard packages.
import copy
import logging

# Third party packages.
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
            model.parameters(), lr = args.lr,
            weight_decay = args.weight_decay)
    else:
        logging.critical(
            'Not supported optimizer: {0}'.format(args.optimizer))

    return optimizer

def update_descriptor(
    args, model, dir_model, index_em = 0, log_writer = None,
    weight_detector = 1):
    ''' Only train the descriptor while try the best to preserve the detector
        unchanged.
    '''
    logging.info('Update descriptor.')
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
    # Train the descriptor using triplet loss on the detected keypoints.
    for epoch in range(args.epoches_descriptor):
        logging.info('epoch: {0}'.format(epoch))
        # Load data.
        avg_loss = 0
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
                gt_score_maps0 = gt_score_maps0.to(
                        torch.device('cuda:{0}'.format(args.gpu_ids[0])))
                gt_score_maps1 = gt_score_maps1.to(
                        torch.device('cuda:{0}'.format(args.gpu_ids[0])))
            # Forward.
            outs = model.forward(imgs0)
            score0, descriptor0 = outs[0], outs[1]
            descriptor0 = F.interpolate(
                descriptor0, (args.height, args.width), mode = 'bilinear')
            descriptor0 = F.normalize(descriptor0, dim = 1)
            outs = model.forward(imgs1)
            score1, descriptor1 = outs[0], outs[1]
            descriptor1 = F.interpolate(
                descriptor1, (args.height, args.width), mode = 'bilinear')
            descriptor1 = F.normalize(descriptor1, dim = 1)
            with torch.no_grad():
                outs = model_init.forward(imgs0)
                score_init0 = outs[0]
                outs = model_init.forward(imgs1)
                score_init1 = outs[0]
            # Calculate loss detector.
            loss_detector = weight_detector * 0.5 * (
                F.mse_loss(score0, score_init0)
                + F.mse_loss(score1, score_init1))
            # Calculate loss descriptor.
            loss_descriptor = torch.tensor(0., device = loss_detector.device)
            for idx_sample in range(descriptor0.shape[0]):
                coord0, coord1 = utils.get_common_keypoints_coord(
                    gt_keypoints_maps0[idx_sample, 0, :, :],
                    grid_01[idx_sample, :, :, :])
                if coord0.shape[0] < 2:
                    continue
                if True == args.use_cuda:
                    coord0 = coord0.to(
                        torch.device('cuda:{0}'.format(args.gpu_ids[0])))
                    coord1 = coord1.to(
                        torch.device('cuda:{0}'.format(args.gpu_ids[0])))
                loss_descriptor += loss_triplet(
                    descriptor0[idx_sample, :, coord0[:, 0],
                                coord0[:, 1]].transpose(0,1),
                    descriptor1[idx_sample, :, coord1[:, 0],
                                coord1[:, 1]].transpose(0,1),
                    anchor_swap = True)
            assert descriptor0.shape[0] != 0
            loss_descriptor = loss_descriptor / descriptor0.shape[0]
            loss = loss_detector + loss_descriptor
            # Backward.
            loss.backward()
            # Update weight.
            optimizer.step()
            optimizer.zero_grad()

            num_batches = num_batches + 1
            avg_loss += loss.detach().cpu().numpy()
            avg_loss_detector += loss_detector.detach().cpu().numpy()
            avg_loss_descriptor += loss_descriptor.detach().cpu().numpy()
        # Log avg_loss.
        avg_loss = avg_loss / num_batches
        avg_loss_detector = avg_loss_detector / num_batches
        avg_loss_descriptor = avg_loss_descriptor / num_batches
        logging.info('avg_loss: {0}'.format(avg_loss))
        if log_writer is not None:
            log_writer.add_scalars(
                'train_descriptor_loss',
                {'avg_loss': avg_loss,
                 'avg_loss_detector': avg_loss_detector,
                 'avg_loss_descriptor': avg_loss_descriptor},
                epoch + index_em * args.epoches_descriptor)
        torch.save(
            {'state_dict': model.state_dict()},
            '{0}/checkpoint_descriptor_{1}_{2}.pth'.format(
                dir_model, index_em, epoch))
        # Update learning rate.
        scheduler.step(avg_loss)

    model = model.cpu()
    return model

if __name__ == '__main__':
    print('Update descriptor')

