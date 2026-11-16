import argparse
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

def unique_configs(groups):
    order, info = ([], {})
    for g in groups:
        for cfg in GROUPS[g]:
            name = run_name(cfg)
            if name not in info:
                order.append(name)
                info[name] = [g, cfg]
    return [tuple(info[n]) for n in order]
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
    if os.path.exists(path):
        return 'skipped'
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
    todo = unique_configs(groups)
    done = [c for _, c in todo if os.path.exists(os.path.join(out_dir, 'runs', run_name(c) + '.json'))]
    print(f'{len(todo)} runs in {groups}: {len(done)} done, {len(todo) - len(done)} left. Device: {device}')
    t_start = time.time()
    for i, (g, cfg) in enumerate(todo, start=1):
        name = run_name(cfg)
        if os.path.exists(os.path.join(out_dir, 'runs', name + '.json')):
            continue
        print(f'[{i}/{len(todo)}] {g}: {name}')
        msg = run_one(cfg, out_dir, data_root, device)
        print(f'   -> {msg}   (total {(time.time() - t_start) / 60:.0f} min)')
    print('Done.')
if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--groups', nargs='+', default=['all'], help=f"one or more of {ORDER}, or 'all'")
    ap.add_argument('--out', default='results')
    ap.add_argument('--data', default='./data')
    a = ap.parse_args()
    run_groups(a.groups, out_dir=a.out, data_root=a.data)