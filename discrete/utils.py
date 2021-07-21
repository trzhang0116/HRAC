import torch
import numpy as np
import os
import time


def euclidean_dist(x, y, squared=False):
    """
    Compute (Squared) Euclidean distance between two tensors.
    Args:
        x: input tensor with size N x D.
        y: input tensor with size M x D.
        return: distance matrix with size N x M.
    """
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception('Invalid input shape.')

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)
    dist = torch.pow(x - y, 2).sum(2)

    if squared:
        return dist
    else:
        return torch.sqrt(dist+1e-12)


def to_tensor(x, device='cpu'):
    x = torch.from_numpy(x).float()
    x = x.to(device)
    return x


def make_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def single_input_transform(obs_raw, device='cpu'):
    obs_var = to_tensor(obs_raw, device=device)
    obs_var = obs_var.unsqueeze(0)
    return obs_var


def print_localtime():
    localtime = time.localtime()
    print("  [%d.%d.%d-%d:%d:%d]" % (localtime.tm_year, localtime.tm_mon, localtime.tm_mday,
                                   localtime.tm_hour, localtime.tm_min, localtime.tm_sec))
