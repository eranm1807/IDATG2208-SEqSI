import copy
import json
import math
import os
import random
import shutil
import tarfile
import time
import urllib.request
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms.functional as TF

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

DATASETS = {'cifar10': dict(in_ch=3, flip=True, pad=4)}

def load_data(name, root='./data', device='cpu', n_val=5000, split_seed=0):
    if name == 'cifar10':
        tr = torchvision.datasets.CIFAR10(root, train=True, download=True)
        te = torchvision.datasets.CIFAR10(root, train=False, download=True)
        x_tr = torch.tensor(tr.data).permute(0, 3, 1, 2).contiguous()
        x_te = torch.tensor(te.data).permute(0, 3, 1, 2).contiguous()
        y_tr, y_te = (torch.tensor(tr.targets), torch.tensor(te.targets))
    else:
        raise ValueError(name)
    g = torch.Generator().manual_seed(split_seed)
    perm = torch.randperm(len(x_tr), generator=g)
    val_idx, tr_idx = (perm[:n_val], perm[n_val:])
    d = dict(x_train=x_tr[tr_idx], y_train=y_tr[tr_idx], x_val=x_tr[val_idx], y_val=y_tr[val_idx], x_test=x_te, y_test=y_te)
    d = {k: v.to(device) for k, v in d.items()}
    d['name'] = name
    return d

def to_float(x_uint8):
    return x_uint8.float().div_(255.0)

def to_uint8(x_float):
    return (x_float.clamp(0, 1) * 255.0).round().to(torch.uint8)