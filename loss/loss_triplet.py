
import torch
import torch.nn as nn
import sys

sys.path.append('..')
from utils import distance_matrix_vector

def loss_triplet(anchor, positive, anchor_swap = False, anchor_ave = False,
                 margin = 0.8, batch_reduce = 'min',
                 loss_type = "ratio_test",
                 mining_self = False, sample_weight=None):
    """HardNet margin loss -
            calculates loss based on distance matrix based on
            positive distance and closest negative distance.
    """
    ratio_threshold = 0.85

    assert anchor.size() == positive.size(), "Input sizes between positive and negative must be equal."
    assert anchor.dim() == 2, "Inputd must be a 2D matrix."
    eps = 1e-8
    dist_matrix = distance_matrix_vector(anchor, positive)
    eye = torch.eye(dist_matrix.size(1), requires_grad = True, device = anchor.device)

    # steps to filter out same patches that occur in distance matrix as negatives
    pos = torch.diag(dist_matrix)
    dist_without_min_on_diag = dist_matrix + eye * 10.
    # Pick out the distance < 0.008 and save into mask.
    mask = (dist_without_min_on_diag.ge(0.008).float() - 1.0) * (-1.)
    mask = mask.type_as(dist_without_min_on_diag) * 10
    # Exclude the distance < 0.008 from min.
    dist_without_min_on_diag = dist_without_min_on_diag + mask
    if batch_reduce == 'min':
        min_neg, _ = torch.min(dist_without_min_on_diag, dim = 1)
        if anchor_swap:
            min_neg2, _ = torch.min(dist_without_min_on_diag, dim = 0)
            min_neg = torch.min(min_neg, min_neg2)
        if mining_self:
            dist_matrix_a = distance_matrix_vector(anchor, anchor) + eps
            dist_matrix_p = distance_matrix_vector(positive, positive) + eps
            dist_without_min_on_diag_a = dist_matrix_a + eye * 10
            dist_without_min_on_diag_p = dist_matrix_p + eye * 10
            min_neg_a, _ = torch.min(dist_without_min_on_diag_a, dim = 1)
            min_neg_p, _ = torch.min(dist_without_min_on_diag_p, dim = 1)
            min_neg_3 = torch.min(min_neg_p, min_neg_a)
            min_neg = torch.min(min_neg, min_neg_3)
    elif batch_reduce == 'average':
        pos = pos.repeat(anchor.size(0)).view(-1,1).squeeze(0)
        min_neg = dist_without_min_on_diag.view(-1,1)
        if anchor_swap:
            min_neg2 = torch.t(dist_without_min_on_diag).contiguous().view(-1,1)
            min_neg = torch.min(min_neg,min_neg2)
        min_neg = min_neg.squeeze(0)
    elif batch_reduce == 'random':
        idxs = torch.randperm(anchor.size()[0], requires_grad = True).long()
        if anchor.is_cuda:
            idxs = idxs.cuda(anchor.get_device())
        min_neg = dist_without_min_on_diag.gather(1,idxs.view(-1,1))
        if anchor_swap:
            min_neg2 = torch.t(dist_without_min_on_diag).gather(1,idxs.view(-1,1)) 
            min_neg = torch.min(min_neg,min_neg2)
        min_neg = torch.t(min_neg).squeeze(0)
        pos = pos
    else:
        print ('Unknown batch reduce mode. Try min, average or random')
        sys.exit(1)

    if loss_type == "ratio_test":
        loss_clamp = (pos.clone().detach() / min_neg.clone().detach()
                      > ratio_threshold)
        loss = loss_clamp * (pos - min_neg)
        loss = loss / (torch.sum(loss_clamp) + eps)
    elif loss_type == "triplet_margin":
        loss = torch.clamp(margin + pos - min_neg, min=0.0)
        loss = loss / torch.sum(loss.clone().detach() > 0)
    elif loss_type == 'softmax':
        exp_pos = torch.exp(2.0 - pos);
        exp_den = exp_pos + torch.exp(2.0 - min_neg) + eps;
        loss = - torch.log( exp_pos / exp_den )
    elif loss_type == 'contrastive':
        loss = torch.clamp(margin - min_neg, min=0.0) + pos;
        loss = loss / torch.sum(loss.clone().detach() > 0)
    else:
        print ('Unknown loss type. Try triplet_margin, softmax or contrastive')
        sys.exit(1)

    if sample_weight is not None:
        loss = sample_weight * loss

    loss = torch.sum(loss)

    return loss

