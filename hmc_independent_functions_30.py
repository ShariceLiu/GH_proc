"""Twenty independent single-function regression experiments, four models each.

Every fit uses only one function's 30 random training observations. There is
no shared fitted theta or transfer from other functions. Generate full coherent
181-point GHP-beta realizations once, then score the remaining 151 locations.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from GH_proc_w_betaprior import Multivariate_GH_1d
from hmc_1d_split_experiments import fit_model, write_json, MODELS, LABELS, COLORS
from hmc_multiple_functions_experiment import TRUTH, aggregate
from fit_beta_mu_1d_experiment import _save_csv


def prepare(root, *, simulation_seed=13026, location_seed=14026):
    root.mkdir(parents=True, exist_ok=True)
    settings = dict(n_functions=20, n_grid=181, n_train=30, n_test=151,
                    simulation_seed=simulation_seed, location_seed=location_seed,
                    parameters=TRUTH, domain=[0., 10.],
                    design='Independent uniform sampling of 30 grid indices without replacement '
                           'per function. Same split for all models. No forced endpoints.',
                    inference='Independent HMC fit per function and model; no pooling across functions.')
    if (root/'dataset.npz').exists():
        if json.loads((root/'generation.json').read_text()) != settings:
            raise ValueError('Use a new output directory for changed simulation settings')
        return
    x = np.linspace(0., 10., settings['n_grid'])
    children = np.random.SeedSequence(simulation_seed).spawn(settings['n_functions'])
    locations = np.random.default_rng(location_seed)
    fs, ys, betas, ws, trains, tests = [], [], [], [], [], []
    for index, child in enumerate(children):
        f, y, beta, W = Multivariate_GH_1d(x, **TRUTH, beta_prior=True,
                                          return_latents=True, seed=np.random.default_rng(child))
        train = np.sort(locations.choice(len(x), settings['n_train'], replace=False))
        test = np.setdiff1d(np.arange(len(x)), train)
        folder = root/f'function_{index:02d}'
        folder.mkdir(exist_ok=True)
        np.savez_compressed(folder/'dataset.npz', x=x, f=f, y=y, beta=beta, W=W,
                            train_ids=train, test_ids=test, all_unobserved_ids=test)
        fs.append(f); ys.append(y); betas.append(beta); ws.append(W)
        trains.append(train); tests.append(test)
    np.savez_compressed(root/'dataset.npz', x=x, f=fs, y=ys, beta=betas, W=ws,
                        train_ids=trains, test_ids=tests)
    write_json(root/'generation.json', settings)
    _save_csv(root/'generating_W.csv', [dict(function_id=i, W=float(w)) for i, w in enumerate(ws)])


def run_fits(root, models, function_ids, *, warmup=1000, draws=1500,
             target_accept=.99, max_retries=1, force=False):
    failed = []
    for index in function_ids:
        folder = root/f'function_{index:02d}'
        with np.load(folder/'dataset.npz') as saved:
            data = {key: saved[key].copy() for key in saved.files}
        for model in models:
            path = folder/f'{model}_report.json'
            if not force and path.exists():
                report = json.loads(path.read_text())
                if report['diagnostics_ok'] and (folder/f'{model}_predictions.npz').exists():
                    print(f'Function {index+1}/20, {model}: reusing completed fit', flush=True)
                    continue
            print(f'INDEPENDENT FUNCTION {index+1}/20 | {model}', flush=True)
            report = fit_model(data, 'sparse', model, folder, seed=15026+100*index,
                               warmup=warmup, draws=draws, target_accept=target_accept,
                               max_retries=max_retries)
            if not report['diagnostics_ok']:
                failed.append(dict(function_id=index, model=model))
    if failed:
        raise RuntimeError(f'Resolve these fits before aggregation: {failed}')


def summarize(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    with np.load(root/'dataset.npz') as saved:
        data = {key: saved[key].copy() for key in saved.files}
    reports, rows, params = {}, [], []
    for index in range(len(data['f'])):
        for model in MODELS:
            folder = root/f'function_{index:02d}'
            report = json.loads((folder/f'{model}_report.json').read_text())
            if not report['diagnostics_ok']:
                raise ValueError(f'Function {index}, {model}: HMC diagnostics failed')
            reports[index, model] = report
            rows.append(dict(function_id=index, generating_W=float(data['W'][index]), **report['scores']))
            params.extend(dict(function_id=index, model=model, **p) for p in report['summary'])
    _save_csv(root/'per_function_scores.csv', rows)
    _save_csv(root/'parameters.csv', params)
    totals = []
    rng = np.random.default_rng(16026)
    boot = rng.integers(0, len(data['f']), size=(10000, len(data['f'])))
    uncertainty, paired = [], []
    for model in MODELS:
        local = [r for r in rows if r['model'] == model]
        baseline = [r for r in rows if r['model'] == 'gp']
        totals.append(dict(model=model, n_functions=len(local), n_train_per_function=30,
                           n_test_per_function=151, **aggregate(local),
                           max_rhat=max(r['max_rhat'] for r in local),
                           min_bulk_ess=min(r['min_bulk_ess'] for r in local),
                           min_tail_ess=min(r['min_tail_ess'] for r in local),
                           divergences=sum(r['divergences'] for r in local),
                           diagnostics_ok=all(r['diagnostics_ok'] for r in local)))
        for target in ('f', 'y'):
            for metric in ('rmse', 'mean_marginal_nlpd'):
                key = f'{target}_{metric}'
                a, b = np.array([r[key] for r in local]), np.array([r[key] for r in baseline])
                if metric == 'rmse':
                    av, bv = np.sqrt((a[boot]**2).mean(axis=1)), np.sqrt((b[boot]**2).mean(axis=1))
                    estimate, reference = np.sqrt(np.mean(a*a)), np.sqrt(np.mean(b*b))
                else:
                    av, bv = a[boot].mean(axis=1), b[boot].mean(axis=1)
                    estimate, reference = a.mean(), b.mean()
                uncertainty.append(dict(model=model, metric=key, estimate=float(estimate),
                                        lower=float(np.quantile(av, .025)), upper=float(np.quantile(av, .975))))
                paired.append(dict(model=model, metric=key, difference_from_gp=float(estimate-reference),
                                   lower=float(np.quantile(av-bv, .025)), upper=float(np.quantile(av-bv, .975))))
    _save_csv(root/'scores.csv', totals)
    _save_csv(root/'score_bootstrap_intervals.csv', uncertainty)
    _save_csv(root/'paired_comparison.csv', paired)
    for page in range(4):
        fig, axes = plt.subplots(5, 4, figsize=(16, 13), sharex=True, sharey='row', constrained_layout=True)
        for row, index in enumerate(range(page*5, (page+1)*5)):
            for col, model in enumerate(MODELS):
                ax = axes[row, col]
                with np.load(root/f'function_{index:02d}'/f'{model}_predictions.npz') as p:
                    ax.plot(data['x'], data['f'][index], 'k--', lw=1, label='True f')
                    ax.plot(data['x'], p['f_mean'], color=COLORS[model], label='Posterior mean')
                    ax.fill_between(data['x'], *p['f_ci'].T, color=COLORS[model], alpha=.22, label='Pointwise 95% interval')
                ids = data['train_ids'][index]
                ax.scatter(data['x'][ids], data['y'][index, ids], s=12, c='black', alpha=.6, label='30 training y')
                ax.set_title(LABELS[model], fontsize=10)
                ax.grid(alpha=.2)
                if col == 0:
                    ax.set_ylabel(f'Function {index+1}, W={data["W"][index]:.2f}\nf')
                if row == 4:
                    ax.set_xlabel('x')
        axes[0, 0].legend(fontsize=6)
        fig.suptitle('Independent single-function fits | 30 random observations each | no pooling of hyperparameters')
        fig.savefig(root/f'regression_page_{page+1}.png', dpi=145)
        plt.close(fig)
    labels_tex = dict(gp='GP', tp_mu_gp='STP + GP mean', mu_gp=r'GHP ($\mu$)', beta_gp=r'GHP ($\beta$)')
    lines = [r'% Requires \usepackage{booktabs}', r'\begin{table}[htbp]', r'\centering',
             r'\caption{Independent regression experiments on 20 functions generated from the GHP '
             r'$\beta$ model, each with an independent $W$. Each model is refitted using 30 randomly '
             r'selected observations per function and evaluated on the remaining 151 points. '
             r'RMSE pools squared errors across all held-out points; NLPD averages pointwise '
             r'negative log predictive densities.}', r'\label{tab:independent_single_function_regression}',
             r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Model & RMSE ($f$) & NLPD ($f$) & RMSE ($y$) & NLPD ($y$) \\', r'\midrule']
    for row in totals:
        lines.append(labels_tex[row['model']]+' & '+' & '.join(f'{row[key]:.6f}' for key in
                     ('f_rmse', 'f_mean_marginal_nlpd', 'y_rmse', 'y_mean_marginal_nlpd'))+r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    (root/'comparison_table.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_json(root/'report.json', dict(generation=json.loads((root/'generation.json').read_text()),
               completed_fits=len(rows), scores=totals, score_intervals=uncertainty, paired_comparison=paired,
               all_diagnostics_ok=all(r['diagnostics_ok'] for r in rows),
               max_rhat=max(r['max_rhat'] for r in rows), min_bulk_ess=min(r['min_bulk_ess'] for r in rows),
               min_tail_ess=min(r['min_tail_ess'] for r in rows),
               retained_divergences=sum(r['divergences'] for r in rows),
               fits_with_retries=sum(len(r['attempts'])>1 for r in reports.values()),
               aggregation='Equal 151 held-out points per function. RMSE=sqrt(mean of per-function '
                           'MSE); NLPD=mean of per-function mean pointwise NLPD. Bootstrap uses '
                           'whole independent functions with paired indices across models.'))
    print(json.dumps(totals, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='figure/independent_functions_30_hmc')
    parser.add_argument('--stage', choices=['all', 'prepare', 'fit', 'summarize'], default='all')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--function-ids', nargs='+', type=int, choices=range(20), default=list(range(20)))
    parser.add_argument('--warmup', type=int, default=1000)
    parser.add_argument('--draws', type=int, default=1500)
    parser.add_argument('--target-accept', type=float, default=.99)
    parser.add_argument('--max-retries', type=int, default=1)
    parser.add_argument('--force', action='store_true')
    args = parser.parse_args()
    root = Path(args.output_dir)
    with threadpool_limits(limits=1):
        if args.stage in ('all', 'prepare'):
            prepare(root)
        if args.stage in ('all', 'fit'):
            run_fits(root, args.models, args.function_ids, warmup=args.warmup, draws=args.draws,
                     target_accept=args.target_accept, max_retries=args.max_retries, force=args.force)
        if args.stage in ('all', 'summarize'):
            summarize(root)
