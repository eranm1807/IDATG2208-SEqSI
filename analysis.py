import glob
import json
import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
if __name__ == '__main__':
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
MODELS = ['Standard', 'SEqSI', 'Standard + aug.', 'Standard + std.']
STYLE = {'Standard': dict(color='#D55E00', marker='o'), 'SEqSI': dict(color='#0072B2', marker='s'), 'Standard + aug.': dict(color='#E69F00', marker='^'), 'Standard + std.': dict(color='#009E73', marker='D')}
DS_NAME = {'cifar10': 'CIFAR-10', 'fashionmnist': 'Fashion-MNIST'}
C10C_SHORT = {'gaussian_blur': 'blur', 'gaussian_noise': 'noise', 'jpeg_compression': 'JPEG'}
CORR_NAME = {'gamma': 'Gamma', 'gaussian_noise': 'Gaussian noise', 'gaussian_blur': 'Gaussian blur', 'jpeg': 'JPEG'}
plt.rcParams.update({'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7, 'ytick.labelsize': 7, 'axes.spines.top': False, 'axes.spines.right': False, 'axes.grid': True, 'grid.color': '#e6e6e6', 'grid.linewidth': 0.6, 'lines.linewidth': 1.5, 'lines.markersize': 4, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02})
FULL_W = 5.5

def label_of(cfg):
    base = {'standard': 'Standard', 'seqsi': 'SEqSI'}[cfg['variant']]
    if cfg.get('preproc') == 'standardize':
        base += ' + std.'
    if cfg.get('aug') == 'photometric':
        base += ' + aug.'
    return base

def family_of(cfg):
    if cfg['convs_per_stage'] != 2:
        return f"depth{cfg['convs_per_stage']}"
    if cfg['bn'] != 'none':
        return f"bn-{cfg['bn']}"
    return 'main'

def load_runs(run_dir):
    runs = []
    for p in sorted(glob.glob(os.path.join(run_dir, '*.json'))):
        with open(p) as f:
            runs.append(json.load(f))
    return runs

def build_frames(runs):
    R, AFF, COR, SPA, C10 = ([], [], [], [], [])
    for r in runs:
        c = r['config']
        key = dict(dataset=c['dataset'], model=label_of(c), family=family_of(c), seed=c['seed'], cps=c['convs_per_stage'], bn=c['bn'])
        aff = pd.DataFrame(r['affine'])
        pure, clipped = (aff[~aff['clip']], aff[aff['clip']])
        cor = pd.DataFrame(r['corruptions'])
        spa = pd.DataFrame(r['spatial'])
        inv = r['invariance']
        row = dict(**key, params=r['params'], gpu=r['gpu'], train_time_min=r['train_time_s'] / 60, eval_time_min=r.get('eval_time_s', 0) / 60, clean_acc=r['clean']['acc'], clean_ece=r['clean']['ece'], clean_ece_ts=r['clean']['ece_ts'], clean_ece_norm=r['clean']['ece_norm'], t_raw=r['temperature']['t_raw'], inv_err=inv['max_rel_err'], inv_err_inv=inv.get('max_rel_err_inv', np.nan), inv_agree=inv['prediction_agreement'], affine_pure_mean=pure['acc'].mean(), affine_pure_worst=pure['acc'].min(), affine_clip_mean=clipped['acc'].mean(), affine_clip_worst=clipped['acc'].min(), corr_mean=cor['acc'].mean(), corr_worst=cor['acc'].min(), spatial_mean=spa['acc'].mean(), spatial_worst=spa['acc'].min(), pure_conf_range=pure['conf'].max() - pure['conf'].min(), pure_conf_norm_range=pure['conf_norm'].max() - pure['conf_norm'].min(), pure_ece_mean=pure['ece'].mean(), pure_ece_norm_mean=pure['ece_norm'].mean())
        for name, g in cor.groupby('corruption'):
            row[f'corr_{name}'] = g['acc'].mean()
        if 'cifar10c' in r:
            c10 = pd.DataFrame(r['cifar10c'])
            row['c10c_mean'] = c10['acc'].mean()
            for name, g in c10.groupby('corruption'):
                row[f'c10c_{name}'] = g['acc'].mean()
            C10.append(c10.assign(**key))
        R.append(row)
        AFF.append(aff.assign(**key))
        COR.append(cor.assign(**key))
        SPA.append(spa.assign(**key))
    cat = lambda xs: pd.concat(xs, ignore_index=True) if xs else pd.DataFrame()
    return (pd.DataFrame(R), cat(AFF), cat(COR), cat(SPA), cat(C10))

def ci95(x):
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n == 0:
        return (np.nan, np.nan)
    if n < 2:
        return (float(x.mean()), np.nan)
    h = stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)
    return (float(x.mean()), float(h))

def curve(df, by, col='acc'):
    rows = []
    for v, g in df.groupby(by):
        m, h = ci95(g[col])
        rows.append((v, m, h, len(g)))
    return pd.DataFrame(rows, columns=[by, 'mean', 'ci', 'n']).sort_values(by)

def _line(ax, c, label, ls='-', band=True, style_of=None):
    st = dict(STYLE.get(style_of or label, dict(color='black', marker='o')))
    if style_of is not None:
        st['markerfacecolor'] = 'white'
    ax.plot(c.iloc[:, 0], c['mean'], ls=ls, label=label, **st)
    if band and c['ci'].notna().any():
        ax.fill_between(c.iloc[:, 0], c['mean'] - c['ci'], c['mean'] + c['ci'], color=st['color'], alpha=0.15, linewidth=0)

def _present(df, models):
    return [m for m in models if m in set(df['model'])]

def _legend_below(fig, axes, ncol=4, bottom=0.08):
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=ncol, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, bottom, 1, 1))

def _contrast_axis(ax, values, narrow=False):
    ax.set_xscale('log')
    ticks = [v for v in values if v in (0.1, 0.25, 0.5, 1, 2, 4)] if narrow else values
    ax.set_xticks(ticks)
    ax.set_xticklabels([f'{v:g}' for v in ticks])
    ax.minorticks_off()

def fig_affine(AFF, dataset, models):
    d = AFF[(AFF['dataset'] == dataset) & (AFF['family'] == 'main')]
    models = _present(d, models)
    fig, axes = plt.subplots(2, 2, figsize=(FULL_W, 3.6), sharey=True)
    for r, clip in enumerate([False, True]):
        for cidx, kind in enumerate(['brightness', 'contrast']):
            ax = axes[r, cidx]
            sub = d[(d['clip'] == clip) & (d['kind'] == kind)]
            for m in models:
                _line(ax, curve(sub[sub['model'] == m], 'value'), m)
            ax.set_ylim(0, 1)
            if kind == 'contrast':
                _contrast_axis(ax, sorted(sub['value'].unique()))
            ax.set_title(f"{kind}, {('clipped to [0,1]' if clip else 'pure affine')}")
            ax.set_xlabel('brightness shift $b$' if kind == 'brightness' else 'contrast factor $a$')
        axes[r, 0].set_ylabel('test accuracy')
    _legend_below(fig, axes.ravel(), ncol=len(models))
    return fig

def fig_corruptions(COR, R, dataset, models):
    d = COR[(COR['dataset'] == dataset) & (COR['family'] == 'main')]
    rr = R[(R['dataset'] == dataset) & (R['family'] == 'main')]
    models = _present(d, models)
    names = list(dict.fromkeys(d['corruption']))
    fig, axes = plt.subplots(1, len(names), figsize=(FULL_W, 1.9), sharey=True)
    for ax, name in zip(np.atleast_1d(axes), names):
        for m in models:
            c = curve(d[(d['corruption'] == name) & (d['model'] == m)], 'severity')
            mc, hc = ci95(rr[rr['model'] == m]['clean_acc'])
            c = pd.concat([pd.DataFrame([[0, mc, hc, 0]], columns=c.columns), c])
            _line(ax, c, m)
        ax.set_title(CORR_NAME.get(name, name))
        ax.set_xlabel('severity (0 = clean)')
        ax.set_xticks(range(6))
    np.atleast_1d(axes)[0].set_ylabel('test accuracy')
    _legend_below(fig, np.atleast_1d(axes), ncol=len(models), bottom=0.2)
    return fig

def fig_cifar10c(R, models):
    d = R[(R['dataset'] == 'cifar10') & (R['family'] == 'main')]
    cols = [c for c in d.columns if c.startswith('c10c_') and c != 'c10c_mean']
    if not cols:
        return None
    models = _present(d.dropna(subset=['c10c_mean']), models)
    names = [c[5:] for c in cols]
    fig, ax = plt.subplots(figsize=(FULL_W, 2.0))
    w = 0.8 / max(len(models), 1)
    for i, m in enumerate(models):
        dm = d[d['model'] == m]
        ms, hs = zip(*[ci95(dm[c]) for c in cols])
        x = np.arange(len(cols)) + (i - (len(models) - 1) / 2) * w
        ax.bar(x, ms, width=w * 0.9, color=STYLE[m]['color'], label=m, yerr=[0 if np.isnan(h) else h for h in hs], capsize=1.5, error_kw=dict(lw=0.6))
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([C10C_SHORT.get(n, n) for n in names])
    ax.set_ylabel('accuracy (mean over\nseverities 1-5)')
    ax.set_ylim(0, 1)
    ax.legend(ncol=min(4, len(models)), frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.18))
    return fig

def fig_calibration(AFF, dataset, models=('Standard', 'SEqSI')):
    d = AFF[(AFF['dataset'] == dataset) & (AFF['family'] == 'main') & (AFF['kind'] == 'contrast') & ~AFF['clip']]
    models = _present(d, list(models))
    fig, axes = plt.subplots(1, 3, figsize=(FULL_W, 1.9))
    for m in models:
        dm = d[d['model'] == m]
        _line(axes[0], curve(dm, 'value', 'acc'), m)
        _line(axes[1], curve(dm, 'value', 'conf'), m)
        _line(axes[1], curve(dm, 'value', 'conf_norm'), m + ' (norm. logits)', ls='--', band=False, style_of=m)
        _line(axes[2], curve(dm, 'value', 'ece'), m)
        _line(axes[2], curve(dm, 'value', 'ece_norm'), m + ' (norm. logits)', ls='--', band=False, style_of=m)
    for ax, t in zip(axes, ['accuracy', 'mean confidence', 'ECE']):
        _contrast_axis(ax, sorted(d['value'].unique()), narrow=True)
        ax.set_xlabel('contrast factor $a$')
        ax.set_title(t)
    axes[0].set_ylim(0, 1)
    axes[1].set_ylim(0, 1.02)
    h, l = axes[1].get_legend_handles_labels()
    fig.legend(h, l, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return fig

def fig_reliability(runs, dataset, models=('Standard', 'SEqSI'), settings=('clean', 'contrast_0.25_pure', 'contrast_3.0_pure')):
    pick = [r for r in runs if r['config']['dataset'] == dataset and family_of(r['config']) == 'main' and (label_of(r['config']) in models)]
    models = [m for m in models if any((label_of(r['config']) == m for r in pick))]
    if not models:
        return None
    fig, axes = plt.subplots(len(models), len(settings), figsize=(FULL_W, 1.75 * len(models)), squeeze=False)
    for i, m in enumerate(models):
        rs = [r for r in pick if label_of(r['config']) == m]
        for j, s in enumerate(settings):
            ax = axes[i, j]
            bins = pd.concat([pd.DataFrame(r['reliability'][s]['raw']) for r in rs])
            bins = bins[bins['count'] > 0]
            agg = bins.assign(wa=bins['acc'] * bins['count'], wc=bins['conf'] * bins['count']).groupby(['lo', 'hi'], as_index=False)[['wa', 'wc', 'count']].sum()
            centre = (agg['lo'] + agg['hi']) / 2
            acc = agg['wa'] / agg['count']
            ax.bar(centre, acc, width=(agg['hi'] - agg['lo']) * 0.92, color=STYLE[m]['color'], alpha=0.85, label='accuracy')
            ax.plot([0, 1], [0, 1], ls='--', color='#555555', lw=0.8)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_title(f"{m}: {s.replace('_', ' ')}")
            if i == len(models) - 1:
                ax.set_xlabel('confidence')
            if j == 0:
                ax.set_ylabel('accuracy')
    fig.tight_layout()
    return fig

def fig_spatial(SPA, dataset, models):
    d = SPA[(SPA['dataset'] == dataset) & (SPA['family'] == 'main')]
    models = _present(d, models)
    fig, ax = plt.subplots(figsize=(FULL_W / 2, 1.9))
    for m in models:
        _line(ax, curve(d[d['model'] == m], 'value'), m)
    ax.set_ylim(0, 1)
    ax.set_xlabel('illumination ramp strength $s$')
    ax.set_ylabel('test accuracy')
    ax.legend(frameon=False, fontsize=6)
    return fig

def fig_depth_bn(R, dataset):
    d = R[(R['dataset'] == dataset) & R['model'].isin(['Standard', 'SEqSI']) & (R['bn'].eq('none') | R['cps'].eq(2))]
    depth = d[d['bn'] == 'none']
    bn = d[d['cps'] == 2]
    if depth['cps'].nunique() < 2 and bn['bn'].nunique() < 2:
        return None
    fig, axes = plt.subplots(1, 2, figsize=(FULL_W, 2.0), sharey=True)
    for ax, sub, col, ticks, title in [(axes[0], depth, 'cps', [1, 2, 3], 'depth (convs per stage)'), (axes[1], bn, 'bn', ['none', 'scale', 'standard'], 'batch normalisation')]:
        for m in ['Standard', 'SEqSI']:
            for metric, ls in [('clean_acc', '-'), ('affine_pure_worst', '--')]:
                c = curve(sub[sub['model'] == m], col, metric)
                if c.empty:
                    continue
                c = c.set_index(col).reindex(ticks).reset_index()
                c[col] = range(len(ticks))
                _line(ax, c, m, ls=ls)
        ax.set_xticks(range(len(ticks)))
        ax.set_xticklabels([str(t) for t in ticks])
        ax.set_xlabel(title)
    axes[0].set_ylabel('accuracy')
    axes[0].set_ylim(0, 1)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=STYLE[m]['color'], marker=STYLE[m]['marker'], label=m) for m in ['Standard', 'SEqSI']]
    handles += [Line2D([], [], color='black', ls='-', label='clean'), Line2D([], [], color='black', ls='--', label='worst pure affine shift')]
    fig.legend(handles=handles, loc='lower center', ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    return fig

def _pm(x, pct=True, digits=1):
    m, h = ci95(x)
    if np.isnan(m):
        return '--'
    f = 100 if pct else 1
    if np.isnan(h):
        return f'{m * f:.{digits}f}'
    return f'{m * f:.{digits}f}$\\pm${h * f:.{digits}f}'
MAIN_COLS = [('clean_acc', 'Clean'), ('affine_pure_worst', 'Pure aff.\\ (worst)'), ('affine_clip_mean', 'Clipped (mean)'), ('affine_clip_worst', 'Clipped (worst)'), ('corr_mean', 'Corrupt.\\ (mean)'), ('spatial_worst', 'Spatial (worst)'), ('clean_ece', 'ECE')]

def table_main(R):
    main = R[R['family'] == 'main']
    lines = ['\\begin{table}[t]', '\\centering', '\\small', '\\setlength{\\tabcolsep}{3.5pt}', '\\caption{Test accuracy and calibration (\\%, mean $\\pm$ 95\\,\\% CI over seeds). \\emph{Pure aff.}: worst case over all unclipped brightness and contrast shifts; \\emph{Clipped}: the same shifts clipped to $[0,1]$; \\emph{Corrupt.}: mean over gamma, noise, blur and JPEG at severities 1--5; \\emph{Spatial}: worst case over the illumination ramp.}', '\\label{tab:main}', '\\begin{tabular}{l' + 'c' * len(MAIN_COLS) + '}', '\\toprule', 'Model & ' + ' & '.join((n for _, n in MAIN_COLS)) + ' \\\\', '\\midrule']
    for ds in ['cifar10', 'fashionmnist']:
        d = main[main['dataset'] == ds]
        if d.empty:
            continue
        lines.append(f'\\multicolumn{{{len(MAIN_COLS) + 1}}}{{l}}{{\\textit{{{DS_NAME[ds]}}}}} \\\\')
        for m in _present(d, MODELS):
            dm = d[d['model'] == m]
            cells = [_pm(dm[c]) for c, _ in MAIN_COLS]
            lines.append(f'{m} & ' + ' & '.join(cells) + ' \\\\')
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    return '\n'.join(lines)

def table_invariance(R):
    main = R[R['family'] == 'main']
    lines = ['\\begin{table}[t]', '\\centering', '\\small', '\\caption{Numerical invariance check on 1{,}000 test images with random $a\\in[0.25,4]$ and $b\\in[-1,1]$ (float64). Relative error of $f(a\\mathbf{x}+b)$ against $a f(\\mathbf{x})$ (largest over images and seeds) and fraction of unchanged predictions.}', '\\label{tab:invariance}', '\\begin{tabular}{llcc}', '\\toprule', 'Dataset & Model & Max.\\ rel.\\ error & Unchanged predictions \\\\', '\\midrule']
    for ds in ['cifar10', 'fashionmnist']:
        d = main[main['dataset'] == ds]
        for m in _present(d, MODELS):
            dm = d[d['model'] == m]
            lines.append(f"{DS_NAME[ds]} & {m} & {dm['inv_err'].max():.1e} & {100 * dm['inv_agree'].min():.1f}\\,\\% \\\\")
    lines += ['\\bottomrule', '\\end{tabular}', '\\end{table}']
    return '\n'.join(lines)

def table_compute(R):
    g = R.groupby(['dataset', 'family']).agg(runs=('seed', 'size'), train_h=('train_time_min', lambda x: x.sum() / 60), eval_h=('eval_time_min', lambda x: x.sum() / 60))
    gpu = ', '.join(sorted(set(R['gpu'])))
    lines = ['\\begin{table}[h]', '\\centering', '\\small', f'\\caption{{Compute used for all experiments (GPU: {gpu}).}}', '\\label{tab:compute}', '\\begin{tabular}{llrrr}', '\\toprule', 'Dataset & Experiment group & Runs & Training (h) & Evaluation (h) \\\\', '\\midrule']
    for (ds, fam), r in g.iterrows():
        lines.append(f"{DS_NAME[ds]} & {fam} & {int(r['runs'])} & {r['train_h']:.2f} & {r['eval_h']:.2f} \\\\")
    lines += ['\\midrule', f"Total & & {int(g['runs'].sum())} & {g['train_h'].sum():.2f} & {g['eval_h'].sum():.2f} \\\\", '\\bottomrule', '\\end{tabular}', '\\end{table}']
    return '\n'.join(lines)
SUMMARY_COLS = ['clean_acc', 'affine_pure_worst', 'affine_clip_mean', 'affine_clip_worst', 'corr_mean', 'spatial_worst', 'c10c_mean', 'clean_ece', 'pure_conf_range', 'pure_conf_norm_range', 'inv_err', 'inv_agree']

def summary_text(R):
    out = [f"Runs: {len(R)} | GPU: {', '.join(sorted(set(R['gpu'])))} | total training time: {R['train_time_min'].sum() / 60:.2f} h"]
    for (ds, fam), d in R.groupby(['dataset', 'family']):
        out.append(f'\n## {ds} / {fam}')
        for m in _present(d, MODELS + sorted(set(d['model']) - set(MODELS))):
            dm = d[d['model'] == m]
            parts = [f'n={len(dm)}', f"params={int(dm['params'].iloc[0]):,}"]
            for c in SUMMARY_COLS:
                if c in dm and (not dm[c].isna().all()):
                    mm, h = ci95(dm[c])
                    if c == 'inv_err':
                        parts.append(f'{c}={dm[c].max():.1e}')
                    else:
                        parts.append(f'{c}={mm:.4f}' + ('' if np.isnan(h) else f'±{h:.4f}'))
            out.append(f'- {m}: ' + ', '.join(parts))
    return '\n'.join(out)

def make_all(run_dir='results/runs', out_dir='results', show=False):
    runs = load_runs(run_dir)
    if not runs:
        print('No runs found in', run_dir)
        return None
    R, AFF, COR, SPA, C10 = build_frames(runs)
    fig_dir, tab_dir = (os.path.join(out_dir, 'figures'), os.path.join(out_dir, 'tables'))
    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(tab_dir, exist_ok=True)
    made = []

    def save(fig, name):
        if fig is None:
            return
        fig.savefig(os.path.join(fig_dir, name + '.pdf'))
        fig.savefig(os.path.join(fig_dir, name + '.png'), dpi=200)
        made.append(name)
        if show:
            plt.show()
        plt.close(fig)
    for ds in sorted(R['dataset'].unique()):
        save(fig_affine(AFF, ds, ['Standard', 'SEqSI']), f'affine_{ds}')
        if len(_present(AFF[AFF['dataset'] == ds], MODELS)) > 2:
            save(fig_affine(AFF, ds, ['Standard', 'SEqSI', 'Standard + aug.', 'Standard + std.']), f'affine_all_{ds}')
        save(fig_corruptions(COR, R, ds, ['Standard', 'SEqSI', 'Standard + aug.', 'Standard + std.']), f'corruptions_{ds}')
        save(fig_calibration(AFF, ds), f'calibration_{ds}')
        save(fig_reliability(runs, ds), f'reliability_{ds}')
        save(fig_spatial(SPA, ds, ['Standard', 'SEqSI', 'Standard + aug.', 'Standard + std.']), f'spatial_{ds}')
        save(fig_depth_bn(R, ds), f'depth_bn_{ds}')
    if 'c10c_mean' in R:
        save(fig_cifar10c(R, ['Standard', 'SEqSI', 'Standard + aug.', 'Standard + std.']), 'cifar10c')
    tables = {'main': table_main(R), 'invariance': table_invariance(R), 'compute': table_compute(R)}
    for k, v in tables.items():
        if v:
            with open(os.path.join(tab_dir, k + '.tex'), 'w') as f:
                f.write(v + '\n')
    R.to_csv(os.path.join(out_dir, 'runs_summary.csv'), index=False)
    text = summary_text(R)
    with open(os.path.join(out_dir, 'summary.md'), 'w') as f:
        f.write(text + '\n')
    print(f"Figures: {', '.join(made)}")
    print(f"Tables: {', '.join((k for k, v in tables.items() if v))}")
    return text