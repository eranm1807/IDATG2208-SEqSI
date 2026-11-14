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

class ZeroSumConv2d(nn.Conv2d):

    def __init__(self, in_ch, out_ch, k=3):
        super().__init__(in_ch, out_ch, k, padding=k // 2, padding_mode='replicate', bias=False)

    def centred_weight(self):
        return self.weight - self.weight.mean(dim=(1, 2, 3), keepdim=True)

    def forward(self, x):
        return self._conv_forward(x, self.centred_weight(), None)
VARIANTS = {'standard': dict(bias=True, zero_sum_first=False), 'seqsi': dict(bias=False, zero_sum_first=True)}

class SmallCNN(nn.Module):

    def __init__(self, in_ch=3, n_classes=10, widths=(64, 128, 256), convs_per_stage=2, bias=True, zero_sum_first=False):
        super().__init__()
        layers, c, first = ([], in_ch, True)
        for w in widths:
            for _ in range(convs_per_stage):
                if first and zero_sum_first:
                    conv = ZeroSumConv2d(c, w)
                else:
                    conv = nn.Conv2d(c, w, 3, padding=1, bias=bias)
                first = False
                layers.append(conv)
                layers.append(nn.ReLU(inplace=True))
                c = w
            layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.head = nn.Linear(c, n_classes, bias=bias)
        self.apply(_init_weights)

    def forward(self, x):
        x = self.features(x)
        x = x.mean(dim=(2, 3))
        return self.head(x)

def _init_weights(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0.0, 0.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

def build_model(variant, in_ch, n_classes=10, convs_per_stage=2):
    net = SmallCNN(in_ch=in_ch, n_classes=n_classes, convs_per_stage=convs_per_stage, **VARIANTS[variant])
    return net

def count_params(model):
    return sum((p.numel() for p in model.parameters()))
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