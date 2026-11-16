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
DATASETS = {'cifar10': dict(in_ch=3, flip=True, pad=4), 'fashionmnist': dict(in_ch=1, flip=False, pad=2)}

def load_data(name, root='./data', device='cpu', n_val=5000, split_seed=0):
    if name == 'cifar10':
        tr = torchvision.datasets.CIFAR10(root, train=True, download=True)
        te = torchvision.datasets.CIFAR10(root, train=False, download=True)
        x_tr = torch.tensor(tr.data).permute(0, 3, 1, 2).contiguous()
        x_te = torch.tensor(te.data).permute(0, 3, 1, 2).contiguous()
        y_tr, y_te = (torch.tensor(tr.targets), torch.tensor(te.targets))
    elif name == 'fashionmnist':
        tr = torchvision.datasets.FashionMNIST(root, train=True, download=True)
        te = torchvision.datasets.FashionMNIST(root, train=False, download=True)
        x_tr, x_te = (tr.data.unsqueeze(1), te.data.unsqueeze(1))
        y_tr, y_te = (tr.targets.clone(), te.targets.clone())
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

def gpu_augment(x, pad=4, flip=True):
    n, c, h, w = x.shape
    dev = x.device
    xp = F.pad(x, (pad, pad, pad, pad), mode='reflect')
    i = torch.randint(0, 2 * pad + 1, (n,), device=dev)
    j = torch.randint(0, 2 * pad + 1, (n,), device=dev)
    rows = (i[:, None] + torch.arange(h, device=dev))[:, None, :, None]
    cols = (j[:, None] + torch.arange(w, device=dev))[:, None, None, :]
    out = xp[torch.arange(n, device=dev)[:, None, None, None], torch.arange(c, device=dev)[None, :, None, None], rows, cols]
    if flip:
        m = torch.rand(n, device=dev) < 0.5
        out = torch.where(m[:, None, None, None], out.flip(3), out)
    return out

def train_model(model, data, dataset, epochs=30, bs=128, lr=0.002, wd=0.0005, seed=0, device='cuda', amp=True, log=print):
    set_seed(seed)
    cfg = DATASETS[dataset]
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    n = len(data['x_train'])
    steps_per_epoch = math.ceil(n / bs)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch)
    use_amp = amp and device == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
    g = torch.Generator().manual_seed(seed)
    hist = []
    t0 = time.time()
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, generator=g).to(data['x_train'].device)
        tot_loss = torch.zeros((), device=device)
        tot_correct = torch.zeros((), device=device)
        for k in range(steps_per_epoch):
            idx = perm[k * bs:(k + 1) * bs]
            x = to_float(data['x_train'][idx])
            y = data['y_train'][idx]
            x = gpu_augment(x, pad=cfg['pad'], flip=cfg['flip'])
            with torch.autocast('cuda', dtype=torch.float16, enabled=use_amp):
                out = model(x)
                loss = F.cross_entropy(out, y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot_loss += loss.detach().float() * len(y)
            tot_correct += (out.argmax(1) == y).sum()
        val_acc = accuracy(model, data['x_val'], data['y_val'])
        row = dict(epoch=ep + 1, train_loss=tot_loss.item() / n, train_acc=tot_correct.item() / n, val_acc=val_acc, time_s=time.time() - t0)
        hist.append(row)
        if log is not None and (ep == 0 or (ep + 1) % 5 == 0 or ep + 1 == epochs):
            log(f"  epoch {ep + 1:3d}/{epochs}  loss {row['train_loss']:.4f}  train {row['train_acc']:.4f}  val {val_acc:.4f}  ({row['time_s']:.0f}s)")
    return hist
B_VALUES = [-0.8, -0.6, -0.4, -0.2, 0.0, 0.2, 0.4, 0.6, 0.8]
A_VALUES = [0.1, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]

def brightness(x, b, clip):
    y = x + b
    return y.clamp(0, 1) if clip else y

def contrast(x, a, clip):
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    y = a * (x - m) + m
    return y.clamp(0, 1) if clip else y
CORRUPTIONS = {'gamma': [1.25, 1.5, 2.0, 2.5, 3.0], 'gaussian_noise': [0.04, 0.06, 0.08, 0.09, 0.1], 'gaussian_blur': [0.4, 0.6, 0.7, 0.8, 1.0], 'jpeg': [80, 65, 58, 50, 40]}
_CORRUPTION_CACHE = {}

def _jpeg(x_uint8, quality):
    from torchvision.io import decode_jpeg, encode_jpeg
    xs = x_uint8.cpu()
    out = torch.empty_like(xs)
    for i in range(len(xs)):
        out[i] = decode_jpeg(encode_jpeg(xs[i], quality=int(quality)))
    return out

def make_corrupted(x_uint8, name, level, seed=1234, cache_key=None):
    key = (cache_key, name, level) if cache_key is not None else None
    if key is not None and key in _CORRUPTION_CACHE:
        return _CORRUPTION_CACHE[key].to(x_uint8.device)
    dev = x_uint8.device
    if name == 'jpeg':
        out = _jpeg(x_uint8, level)
    else:
        outs = []
        g = torch.Generator(device=dev).manual_seed(seed)
        for k in range(0, len(x_uint8), 1000):
            x = to_float(x_uint8[k:k + 1000])
            if name == 'gamma':
                y = x.clamp(0, 1).pow(level)
            elif name == 'gaussian_noise':
                y = x + level * torch.randn(x.shape, generator=g, device=dev)
            elif name == 'gaussian_blur':
                ks = 2 * math.ceil(3 * level) + 1
                y = TF.gaussian_blur(x, kernel_size=[ks, ks], sigma=[level, level])
            else:
                raise ValueError(name)
            outs.append(to_uint8(y))
        out = torch.cat(outs)
    if key is not None:
        _CORRUPTION_CACHE[key] = out.cpu()
    return out.to(dev)

@torch.no_grad()
def predict_logits(model, x_uint8, transform=None, bs=1000):
    model.eval()
    outs = []
    for k in range(0, len(x_uint8), bs):
        x = to_float(x_uint8[k:k + bs])
        if transform is not None:
            x = transform(x)
        outs.append(model(x).float())
    return torch.cat(outs)

def accuracy(model, x, y, transform=None):
    return (predict_logits(model, x, transform).argmax(1) == y).float().mean().item()

def metrics_from_logits(z, y, t_raw=1.0, t_norm=1.0):
    p = z.softmax(1)
    return dict(acc=(z.argmax(1) == y).float().mean().item(), conf=p.max(1).values.mean().item())

def shift_sweep(model, x, y, kind, values, clip, t_raw=1.0, t_norm=1.0):
    fn = brightness if kind == 'brightness' else contrast
    rows = []
    for v in values:
        z = predict_logits(model, x, lambda t, v=v: fn(t, v, clip))
        rows.append(dict(kind=kind, value=float(v), clip=bool(clip), **metrics_from_logits(z, y, t_raw, t_norm)))
    return rows

@torch.no_grad()
def invariance_check(model, x_uint8, n=1000, seed=0, use_double=True):
    m = copy.deepcopy(model).eval()
    dtype = torch.float64 if use_double else torch.float32
    m = m.to(dtype)
    g = torch.Generator().manual_seed(seed)
    x = to_float(x_uint8[:n]).to(dtype)
    a = torch.exp(torch.empty(n).uniform_(math.log(0.25), math.log(4), generator=g))
    b = torch.empty(n).uniform_(-1, 1, generator=g)
    a = a.to(x.device, dtype)[:, None, None, None]
    b = b.to(x.device, dtype)[:, None, None, None]
    f0 = torch.cat([m(x[k:k + 250]) for k in range(0, n, 250)])
    f1 = torch.cat([m((a * x + b)[k:k + 250]) for k in range(0, n, 250)])
    a_vec = a[:, 0, 0, 0, None]
    t_eq = a_vec * f0
    rel_eq = (f1 - t_eq).norm(dim=1) / t_eq.norm(dim=1).clamp_min(1e-30)
    rel_inv = (f1 - f0).norm(dim=1) / f0.norm(dim=1).clamp_min(1e-30)
    agree = (f1.argmax(1) == f0.argmax(1)).float().mean().item()
    return dict(n=n, max_rel_err=rel_eq.max().item(), median_rel_err=rel_eq.median().item(), max_rel_err_inv=rel_inv.max().item(), prediction_agreement=agree)

def evaluate_model(model, data, cifar10c_dir=None, log=print):
    dataset = data['name']
    xv, yv = (data['x_val'], data['y_val'])
    xt, yt = (data['x_test'], data['y_test'])
    res = {}
    res['clean'] = metrics_from_logits(predict_logits(model, xt), yt)
    res['invariance'] = invariance_check(model, xt)
    rows = []
    for clip in (False, True):
        rows += shift_sweep(model, xt, yt, 'brightness', B_VALUES, clip)
        rows += shift_sweep(model, xt, yt, 'contrast', A_VALUES, clip)
    res['affine'] = rows
    return res

def save_json(obj, path):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(obj, f, indent=1)
    os.replace(tmp, path)