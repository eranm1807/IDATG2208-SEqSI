import itertools
import os
import time
import torch
import seqsi_lib as L
BASE = dict(dataset='cifar10', variant='standard', seed=0, epochs=30, convs_per_stage=2, bn='none', preproc='none', aug='none')
DATASETS = ['cifar10', 'fashionmnist']
SEEDS5 = list(range(5))

def grid(**kw):
    keys = list(kw)
    return [dict(BASE, **dict(zip(keys, vals))) for vals in itertools.product(*(kw[k] for k in keys))]
GROUPS = {'E': grid(dataset=['cifar10'], variant=['standard', 'seqsi'], seed=[0]), 'D_main': grid(dataset=DATASETS, variant=['standard', 'seqsi'], seed=SEEDS5)}
ORDER = ['E', 'D_main']

def run_name(cfg):
    return f"{cfg['dataset']}_{cfg['variant']}_cps{cfg['convs_per_stage']}_bn-{cfg['bn']}_pre-{cfg['preproc']}_aug-{cfg['aug']}_s{cfg['seed']}"
_DATA = {}

def get_data(dataset, root, device):
    if dataset not in _DATA:
        _DATA.clear()
        _DATA[dataset] = L.load_data(dataset, root=root, device=device)
    return _DATA[dataset]

def _build(cfg):
    in_ch = L.DATASETS[cfg['dataset']]['in_ch']
    return L.build_model(cfg['variant'], in_ch, convs_per_stage=cfg['convs_per_stage'])

def run_one(cfg, out_dir, data_root, device):
    name = run_name(cfg)
    path = os.path.join(out_dir, 'runs', name + '.json')
    data = get_data(cfg['dataset'], data_root, device)
    L.set_seed(cfg['seed'])
    model = _build(cfg)
    t0 = time.time()
    hist = L.train_model(model, data, cfg['dataset'], epochs=cfg['epochs'], seed=cfg['seed'], device=device)
    train_time = time.time() - t0
    t1 = time.time()
    ev = L.evaluate_model(model, data)
    eval_time = time.time() - t1
    res = dict(name=name, config=cfg, params=L.count_params(model), train_time_s=train_time, eval_time_s=eval_time, history=hist, **ev)
    L.save_json(res, path)
    return f"clean acc {ev['clean']['acc']:.4f} | inv.err {ev['invariance']['max_rel_err']:.1e} | {train_time / 60:.1f}+{eval_time / 60:.1f} min"

def run_groups(groups, out_dir='results', data_root='./data', device=None):
    if groups == ['all'] or groups == 'all':
        groups = ORDER
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    for g in groups:
        for cfg in GROUPS[g]:
            print(f'{g}: {run_name(cfg)}')
            msg = run_one(cfg, out_dir, data_root, device)
            print(f'   -> {msg}')
    print('Ferdig.')