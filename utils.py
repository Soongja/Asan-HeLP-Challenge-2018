import numpy as np
import torch
import torch.optim as optim


# K-fold cross validataion

# Developer: Alejandro Debus
# Email: aledebus@gmail.com

def partitions(number, k):
    '''
    Distribution of the folds

    Args:
        number: number of patients
        k: folds number
    '''
    n_partitions = np.ones(k) * int(number/k)
    n_partitions[0:(number % k)] += 1
    return n_partitions


def get_indices(n_splits, subjects, frames=1):
    '''
    Indices of the set test

    Args:
        n_splits: folds number
        subjects: number of patients
        frames: length of the sequence of each patient
    '''
    l = partitions(subjects, n_splits)
    fold_sizes = l * frames
    indices = np.arange(subjects * frames).astype(int)
    current = 0
    for fold_size in fold_sizes:
        start = current
        stop =  current + fold_size
        current = stop
        yield(indices[int(start):int(stop)])


def k_folds(n_splits, subjects, frames=1):
    '''
    Generates folds for cross validation

    Args:
        n_splits: folds number
        subjects: number of patients
        frames: length of the sequence of each patient
    '''
    indices = np.arange(subjects * frames).astype(int)
    for test_idx in get_indices(n_splits, subjects, frames):
        train_idx = np.setdiff1d(indices, test_idx)
        yield train_idx, test_idx

#######################################################################


def _create_optimizer(opt, model):
    lr = opt.lr
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return optimizer


def _create_scheduler(optimizer, milestones):
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=patience, verbose=True)
    return scheduler
