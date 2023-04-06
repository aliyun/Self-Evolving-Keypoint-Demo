
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def one_hot(index, num_classes):
    size = index.size() + (num_classes,)
    view = index.size() + (1,)

    mask = torch.zeros(*size)
    index = index.view(*view)
    ones = torch.ones(*view)
    if index.is_cuda:
        mask = mask.cuda(index.get_device())
        ones = ones.cuda(index.get_device())

    return mask.scatter_(1, index.long(), ones)

class FocalLoss(nn.Module):

    def __init__(self, weight = None, ignore_index = None, dim = -1, gamma=1, eps=1e-7):
        super(FocalLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.dim = dim
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target, weight_element = None):
        prob = torch.transpose(input, -1, self.dim)
        prob = prob.contiguous().view(-1, input.size(self.dim))
        if dim >= 0:
            target = target.unsqueeze(self.dim)
        else:
            target = target.unsqueeze(self.dim-1)
        target = torch.transpose(target, -1, self.dim)
        target = target.contiguous().view(-1)

        y = one_hot(target, prob.size(-1))
        if ignore_index is not None:
            y[:, ignore_index] = 0

        logit = F.softmax(prob, dim=-1)
        logit = logit.clamp(self.eps, 1. - self.eps)
        loss = -1. * y * torch.log(logit) # cross entropy
        # Weight loss.
        if weight_element is not None:
            weight_element = weight_element.view(-1, 1)
            loss = loss * weight_element
        if self.weight is not None:
            assert(len(self.weight) == loss.size(-1))
            loss = loss * torch.tensor(self.weight, dtype=loss.dtype, device=loss.device)
        loss = loss * (1 - logit) ** self.gamma # focal loss

        return loss.mean()

def focal_loss(input, target, weight = None, ignore_index = None, dim = -1, gamma=1, eps=1e-7, weight_element = None):
    prob = torch.transpose(input, -1, dim)
    prob = prob.contiguous().view(-1, input.size(dim))
    if dim >= 0:
        target = target.unsqueeze(dim)
    else:
        target = target.unsqueeze(dim-1)
    target = torch.transpose(target, -1, dim)
    target = target.contiguous().view(-1)

    y = one_hot(target, prob.size(-1))
    if ignore_index is not None:
        y[:, ignore_index] = 0

    logit = F.softmax(prob, dim=-1)
    logit = logit.clamp(eps, 1. - eps)
    loss = -1. * y * torch.log(logit) # cross entropy
    # Weight loss.
    if weight_element is not None:
        weight_element = weight_element.view(-1, 1)
        loss = loss * weight_element
    if weight is not None:
        assert(len(weight) == loss.size(-1))
        loss = loss * torch.tensor(weight, dtype=loss.dtype, device=loss.device)
    loss = loss * (1 - logit) ** gamma # focal loss

    return loss.mean()

class BinaryFocalLoss(nn.Module):

    def __init__(self, gamma=1, eps=1e-7):
        super(BinaryFocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps

    def forward(self, input, target):
        prob = input.clamp(self.eps, 1. - self.eps)
        loss = -1. * ((1-prob)**self.gamma * target * torch.log(prob) +
                prob**self.gamma * (1-target) * torch.log(1-prob))

        return loss.mean()

def binary_focal_loss(input, target, gamma=1, eps=1e-7):
    prob = input.clamp(eps, 1. - eps)
    loss = -1. * ((1-prob)**gamma * target * torch.log(prob) +
            prob**gamma * (1-target) * torch.log(1-prob))

    return loss.mean()

