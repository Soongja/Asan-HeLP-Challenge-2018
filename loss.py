"""
Reference:
- https://github.com/bermanmaxim/LovaszSoftmax/blob/master/pytorch/lovasz_losses.py
"""

import numpy as np
import torch
import torch.nn as nn
from itertools import filterfalse


def one_hot_embedding(labels, num_classes):
    """Embedding labels to one-hot form.

    Args:
      labels: (LongTensor) class labels, sized [N,].
      num_classes: (int) number of classes.

    Returns:
      (tensor) encoded labels, sized [N, #classes].
    """
    y = torch.eye(num_classes)
    return y[labels]


def mean(l, ignore_nan=False, empty=0):
    """
    nanmean compatible with generators.
    """
    l = iter(l)
    if ignore_nan:
        l = filterfalse(np.isnan, l)
    try:
        n = 1
        acc = next(l)
    except StopIteration:
        if empty == 'raise':
            raise ValueError('Empty mean')
        return empty
    for n, v in enumerate(l, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n


def lovasz_grad(gt_sorted):
    """
    Computes gradient of the Lovasz extension w.r.t sorted errors
    See Alg. 1 in paper
    """
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1. - intersection / union
    if p > 1: # cover 1-pixel case
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard


def lovasz_softmax_flat(probas, labels, only_present=False):
    """
    Multi-class Lovasz-Softmax loss
      probas: [P, C] Variable, class probabilities at each prediction (between 0 and 1)
      labels: [P] Tensor, ground truth labels (between 0 and C - 1)
      only_present: average only on classes present in ground truth
    """
    C = probas.size(1)
    losses = []
    for c in range(C):
        fg = (labels == c).float() # foreground for class c
        if only_present and fg.sum() == 0:
            continue
        errors = (fg - probas[:, c]).abs()
        errors_sorted, perm = torch.sort(errors, 0, descending=True)
        perm = perm.data
        fg_sorted = fg[perm]
        losses.append(torch.dot(errors_sorted, lovasz_grad(fg_sorted)))
    return mean(losses)


class LovaszLoss(nn.Module):
    def __init__(self):
        super(LovaszLoss, self).__init__()

    def forward(self, pred, target):
        return lovasz_softmax_flat(pred, target)


def weighted_categorical_dice(pred, target, n_classes, class_weights):
    if torch.cuda.is_available():
        pred = pred.cuda()
        target = target.cuda()

    smooth = 1.
    dice = 0.
    for c in range(n_classes):
        pflat = pred[:, c].view(-1)
        tflat = target[:, c].view(-1)

        intersection = (pflat * tflat).sum()

        w = class_weights[c]
        dice += w * ((2. * intersection + smooth) / (pflat.sum() + tflat.sum() + smooth))

    return dice


class WeightedDiceLoss(nn.Module):
    def __init__(self, n_classes, class_weights):
        super(WeightedDiceLoss, self).__init__()
        self.n_classes = n_classes
        self.class_weights = class_weights

    def forward(self, pred, target):
        # inputs must have shapes and dtypes as below.
        # pred: (N, n_classes) - contiguous float tensor
        # target: (N) - long tensor

        target = one_hot_embedding(target, self.n_classes)

        return 1 - weighted_categorical_dice(pred, target, self.n_classes, self.class_weights)


class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights):
        super(WeightedCrossEntropyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))

    def forward(self, pred, target):
        # inputs must have shapes and dtypes as below.
        # pred: (N, n_classes) - contiguous float tensor
        # target: (N) - long tensor
        ce = self.ce(pred, target)

        return ce


class Weighted_CE_Dice(nn.Module):
    def __init__(self, n_classes, class_weights):
        super(Weighted_CE_Dice, self).__init__()
        self.ce = nn.CrossEntropyLoss(weight=torch.tensor(class_weights))
        self.dice = WeightedDiceLoss(n_classes, class_weights)

    def forward(self, pred, target):
        # inputs must have shapes and dtypes as below.
        # pred: (N, n_classes) - contiguous float tensor
        # target: (N) - long tensor
        ce = self.ce(pred, target)
        dice = self.dice(pred, target)

        return ce + dice
