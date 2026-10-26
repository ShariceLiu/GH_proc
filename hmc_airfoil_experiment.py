"""Single-function Airfoil regression with GP, STP+GP mean, GHP mu and beta.

Run python hmc_airfoil_experiment.py. Each model has two ARD SE kernels.
Fixed random row split, training-only standardization, fixed mixing priors.
HMC samples the kernel/noise parameters and (except GP) a single global W.
No latent f ground truth exists: report held-out y RMSE/NLPD/coverage only.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from GH_tabular_hmc import MODELS, ZERO_MODELS, ALL_MODELS, has_mixing, default_center, sample_parameters, predict_mixture
from GP_TP_hmc import mixture_logpdf
from load_real_datasets import load_datasets

LABELS = {'gp': 'GP', 'tp_mu_gp': 'STP + GP mean', 'mu_gp': 'GHP mu', 'beta_gp': 'GHP beta'}


def write_json(path, obj):
    def convert(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    Path(path).write_text(json.dumps(obj, default=convert, indent=2)+'\n', encoding='utf-8')


def prepare(root, n_train=300, split_seed=17026):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    X, target = load_datasets(['airfoil'])['airfoil']
    x, y = X.to_numpy(dtype=float), target.to_numpy(dtype=float)
    if not 2 <= n_train < len(y):
        raise ValueError('Need at least two training rows and one test row')
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError('Nonfinite data; explicit preprocessing required')
    order = np.random.default_rng(split_seed).permutation(len(y))
    train, test = np.sort(order[:n_train]), np.sort(order[n_train:])
    x_mean, x_scale = x[train].mean(axis=0), x[train].std(axis=0)
    y_mean, y_scale = y[train].mean(), y[train].std()
    if np.any(x_scale == 0) or y_scale == 0:
        raise ValueError('Cannot standardize constant training columns')
    data = dict(x=x, y=y, train_ids=train, test_ids=test, x_mean=x_mean, x_scale=x_scale,
                y_mean=y_mean, y_scale=y_scale, x_scaled=(x-x_mean)/x_scale,
                y_scaled=(y-y_mean)/y_scale, feature_names=np.array(X.columns, dtype=str),
                target_name=np.array(str(target.name)), split_seed=np.array(split_seed))
    path = root/'dataset.npz'
    if path.exists():
        with np.load(path) as old:
            for key, value in data.items():
                if not np.array_equal(old[key], value):
                    raise ValueError('Data/split changed; choose a new output folder')
    else:
        np.savez_compressed(path, **data)
    config = dict(dataset='airfoil', n_rows=len(y), n_features=x.shape[1], n_train=len(train),
                  n_test=len(test), split_seed=split_seed, split='uniform random rows without replacement',
                  feature_names=list(X.columns), target=str(target.name),
                  standardization=dict(x_mean=x_mean, x_scale=x_scale, y_mean=y_mean, y_scale=y_scale,
                                       source='training rows only', ddof=0),
                  fixed_priors=dict(GHP=dict(l=1., omega=1.), STP=dict(nu=5., ig_shape=2.5, ig_scale=1.5)),
                  kernel='sum of two SE kernels, separate inverse squared length scale per input/component',
                  log_prior_center=default_center(data['y_scaled'][train], x.shape[1]), log_prior_sd=2.,
                  source_sha256={name: hashlib.sha256((Path('data/real_datasets/airfoil')/name).read_bytes()).hexdigest()
                                 for name in ['features.csv', 'target.csv']},
                  scores='original target units; pointwise Gaussian-mixture predictive density for y')
    write_json(root/'experiment.json', config)
    return data


def scores(y, prediction):
    logp = mixture_logpdf(y, prediction['component_means'], prediction['component_y_variances'],
                          weights=prediction['weights'])
    ci, mean = prediction['y_ci'], prediction['y_mean']
    return dict(y_rmse=float(np.sqrt(np.mean((y-mean)**2))),
                y_mean_marginal_nlpd=float(-logp.mean()),
                y_coverage_95=float(np.mean((y >= ci[:, 0]) & (y <= ci[:, 1]))),
                y_mean_interval_width=float(np.mean(ci[:, 1]-ci[:, 0])))


def fit_model(root, data, model, *, chains=4, warmup=1000, draws=1500, seed=18026,
              target_accept=.95, max_retries=1):
    folder = Path(root)/model
    folder.mkdir(parents=True, exist_ok=True)
    report_path = folder/'report.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report['diagnostics_ok'] and (folder/'predictions.npz').exists():
            print(f'{model}: reusing completed fit', flush=True)
            return report
    train, test = data['train_ids'], data['test_ids']
    x, y = data['x_scaled'][train], data['y_scaled'][train]
    attempts_file = folder/'attempts.json'
    attempts = json.loads(attempts_file.read_text()) if attempts_file.exists() else []
    start = time.perf_counter()
    resumed = bool(attempts and attempts[-1]['diagnostics_ok'] and
                   (folder/f'attempt_{attempts[-1]["attempt"]}_samples.npz').exists())
    if resumed:
        previous = attempts[-1]
        number, settings = previous['attempt'], previous['settings']
        chains = settings['chains']
        with np.load(folder/f'attempt_{number}_samples.npz') as saved:
            fit = {key: saved[key].copy() for key in saved.files}
        fit.update(model=model, diagnostics=previous['diagnostics'],
                   effective_diagnostics=previous['effective_diagnostics'],
                   initialization=previous.get('initialization', []))
        summary = {key: previous[key] for key in ['max_rhat', 'min_bulk_ess', 'min_tail_ess',
                                                'divergences', 'diagnostics_ok', 'acceptance', 'swap_acceptance']}
        # JSON diagnostics are lists; parameter summaries index them directly.
        print(f'{model}: resuming prediction from saved attempt {number}', flush=True)
    for retry in ([] if resumed else range(max_retries+1)):
        number = len(attempts)+1
        settings = dict(chains=chains, warmup=warmup*2**retry, draws=draws*2**retry,
                        seed=seed+100*ALL_MODELS.index(model)+10000*(number-1),
                        target_accept=min(.995, target_accept+.03*retry), leapfrog_steps=(12, 30))
        print(f'{model}: attempt {number}, settings={settings}', flush=True)
        fit = sample_parameters(y, x, model=model, **settings)
        summary = {key: fit[key] for key in ['max_rhat', 'min_bulk_ess', 'min_tail_ess', 'divergences',
                                           'diagnostics_ok', 'acceptance', 'swap_acceptance']}
        attempt = dict(attempt=number, settings=settings, **summary,
                       diagnostics=fit['diagnostics'], effective_diagnostics=fit['effective_diagnostics'],
                       warmup_divergences=int(np.sum(fit['warmup_divergent'])),
                       initialization=fit['initialization'], sampling_seconds=time.perf_counter()-start)
        attempts.append(attempt)
        np.savez_compressed(folder/f'attempt_{number}_samples.npz', parameter_names=fit['parameter_names'],
                            **{key: value for key, value in fit.items() if isinstance(value, np.ndarray)})
        write_json(attempts_file, attempts)
        print(json.dumps(summary), flush=True)
        if fit['diagnostics_ok']:
            break
    sampling_seconds = attempts[-1].get('sampling_seconds') if resumed else time.perf_counter()-start
    prediction = predict_mixture(y, x, data['x_scaled'][test], fit,
                                 output_mean=float(data['y_mean']), output_scale=float(data['y_scale']))
    np.savez_compressed(folder/'predictions.npz', test_ids=test, **prediction)
    metric = scores(data['y'][test], prediction)
    flat = np.exp(fit['log_samples'].reshape(-1, fit['log_samples'].shape[-1]))
    parameters = []
    for j, name in enumerate(fit['parameter_names']):
        quantiles = np.quantile(flat[:, j], [.025, .5, .975])
        parameters.append(dict(parameter=name, mean=float(flat[:, j].mean()), std=float(flat[:, j].std()),
                               lower=float(quantiles[0]), median=float(quantiles[1]), upper=float(quantiles[2]),
                               **{key: float(value[j]) for key, value in fit['diagnostics'].items()}))
    pd.DataFrame(parameters).to_csv(folder/'parameters_standardized.csv', index=False)
    # Amplitudes/noise are variances; ARD weights are inverse length squared.
    if model in ZERO_MODELS:
        factors = np.r_[np.full(2, float(data['y_scale'])**2), 1./data['x_scale']**2]
    else:
        factors = np.r_[np.full(3, float(data['y_scale'])**2),
                        1./data['x_scale']**2, 1./data['x_scale']**2]
    if has_mixing(model):
        factors = np.r_[factors, 1.]
    original_parameters = []
    for j, entry in enumerate(parameters):
        converted = dict(entry)
        for key in ['mean', 'std', 'lower', 'median', 'upper']:
            converted[key] *= float(factors[j])
        original_parameters.append(converted)
    pd.DataFrame(original_parameters).to_csv(folder/'parameters_original_units.csv', index=False)
    chain_metrics = []
    for chain in range(chains):
        section = slice(chain*settings['draws'], (chain+1)*settings['draws'])
        means = prediction['component_means'][section]
        variances = prediction['component_y_variances'][section]
        logp = mixture_logpdf(data['y'][test], means, variances)
        chain_metrics.append(dict(chain=chain+1, y_rmse=float(np.sqrt(np.mean((means.mean(axis=0)-data['y'][test])**2))),
                                  y_mean_marginal_nlpd=float(-logp.mean())))
    pd.DataFrame(chain_metrics).to_csv(folder/'per_chain_scores.csv', index=False)
    pd.DataFrame(dict(row_id=test, y=data['y'][test], predicted_y=prediction['y_mean'],
                      y_lower=prediction['y_ci'][:, 0], y_upper=prediction['y_ci'][:, 1],
                      f_lower=prediction['f_ci'][:, 0], f_upper=prediction['f_ci'][:, 1])).to_csv(folder/'predictions.csv', index=False)
    report = dict(model=model, n_train=len(train), n_test=len(test), **metric, **summary,
                  selected_attempt=number, selected_samples_file=f'attempt_{number}_samples.npz',
                  resumed_predictions=resumed,
                  settings=settings, sampling_seconds=sampling_seconds, total_seconds=time.perf_counter()-start,
                  prior_center=fit['prior_center'], prior_sd=fit['prior_sd'], parameters=parameters,
                  parameters_original_units=original_parameters,
                  initialization=fit['initialization'], diagnostics=fit['diagnostics'],
                  effective_diagnostics=fit['effective_diagnostics'])
    write_json(report_path, report)
    print(f'{model}: {metric}', flush=True)
    return report


def summarize(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    with np.load(root/'dataset.npz') as saved:
        y, test = saved['y'].copy(), saved['test_ids'].copy()
        x_scale, y_scale = saved['x_scale'].copy(), float(saved['y_scale'])
    reports, rows = [], []
    for model in MODELS:
        file = root/model/'report.json'
        if not file.exists():
            continue
        report = json.loads(file.read_text())
        # Also upgrades reports produced while this presentation code was being added.
        factors = np.r_[np.full(3, y_scale**2), 1./x_scale**2, 1./x_scale**2]
        if model != 'gp':
            factors = np.r_[factors, 1.]
        original = []
        for j, entry in enumerate(report['parameters']):
            converted = dict(entry)
            for key in ['mean', 'std', 'lower', 'median', 'upper']:
                converted[key] *= float(factors[j])
            original.append(converted)
        pd.DataFrame(original).to_csv(root/model/'parameters_original_units.csv', index=False)
        report['parameters_original_units'] = original
        write_json(file, report)
        reports.append(report)
        rows.append({key: report[key] for key in ['model', 'n_train', 'n_test', 'y_rmse',
                    'y_mean_marginal_nlpd', 'y_coverage_95', 'y_mean_interval_width',
                    'max_rhat', 'min_bulk_ess', 'min_tail_ess', 'divergences', 'diagnostics_ok']})
    if not rows:
        return
    pd.DataFrame(rows).to_csv(root/'scores.csv', index=False)
    write_json(root/'summary.json', dict(completed_models=len(rows), all_diagnostics_ok=all(r['diagnostics_ok'] for r in rows), scores=rows))
    lines = [r'% Requires \usepackage{booktabs}', r'\begin{table}[htbp]', r'\centering',
             rf'\caption{{Airfoil regression: {rows[0]["n_train"]:,} training and {rows[0]["n_test"]:,} test observations. '
             r'HMC posterior predictions; scores in original target units. '
             r'STP includes an independent GP mean.}', r'\label{tab:airfoil_regression}',
             r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Model & RMSE ($y$) & NLPD ($y$) & 95\% coverage & Mean width \\', r'\midrule']
    latex_names = {'gp': 'GP', 'tp_mu_gp': 'STP + GP mean', 'mu_gp': r'GHP ($\mu$)', 'beta_gp': r'GHP ($\beta$)'}
    for row in rows:
        suffix = '' if row['diagnostics_ok'] else r'$^{\dagger}$'
        lines.append(latex_names[row['model']]+suffix+' & '+' & '.join(f'{row[key]:.4f}' for key in
                     ['y_rmse', 'y_mean_marginal_nlpd', 'y_coverage_95', 'y_mean_interval_width'])+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    if not all(row['diagnostics_ok'] for row in rows):
        lines += [r'\par\small $\dagger$ HMC diagnostics require further investigation.']
    lines += [r'\end{table}']
    (root/'comparison_table.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    fig, axes = plt.subplots(2, 4, figsize=(17, 8), constrained_layout=True)
    for col, model in enumerate(MODELS):
        match = [r for r in reports if r['model'] == model]
        if not match:
            axes[0, col].set_visible(False)
            axes[1, col].set_visible(False)
            continue
        report = match[0]
        with np.load(root/model/'predictions.npz') as pred:
            mean, ci = pred['y_mean'], pred['y_ci']
        observed = y[test]
        ax = axes[0, col]
        ax.scatter(observed, mean, s=9, alpha=.45)
        limits = [min(observed.min(), mean.min()), max(observed.max(), mean.max())]
        ax.plot(limits, limits, 'k--', lw=1)
        flag = '' if report['diagnostics_ok'] else ' [CHECK HMC]'
        ax.set(title=f'{LABELS[model]}{flag}\nRMSE={report["y_rmse"]:.3f}; NLPD={report["y_mean_marginal_nlpd"]:.3f}',
               xlabel='Observed target', ylabel='Posterior predictive mean')
        # All held-out rows sorted by predictive mean: descriptive display, not an x-axis function.
        ids = np.argsort(mean)
        ax = axes[1, col]
        ax.fill_between(np.arange(len(test)), ci[ids, 0], ci[ids, 1], alpha=.25, label='95% predictive interval for y')
        ax.plot(mean[ids], lw=1, label='Predictive mean')
        ax.scatter(np.arange(len(test)), observed[ids], c='black', s=4, alpha=.35, label='Held-out observations')
        ax.set(xlabel='Test rows ordered by predictive mean', ylabel='Target (original units)',
               title=f'Coverage={report["y_coverage_95"]:.1%}; mean width={report["y_mean_interval_width"]:.2f}')
        ax.legend(fontsize=7)
    fig.suptitle(f'Airfoil: one 5D regression function | shared random {rows[0]["n_train"]:,} / {rows[0]["n_test"]:,} split')
    fig.savefig(root/'regression.png', dpi=155)
    plt.close(fig)
    for report in reports:
        folder = root/report['model']
        with np.load(folder/report['selected_samples_file']) as sample:
            q, names = sample['log_samples'], sample['parameter_names']
        fig, axes = plt.subplots(7, 2, figsize=(12, 15), constrained_layout=True)
        for j, ax in enumerate(axes.flat):
            if j >= q.shape[-1]:
                ax.set_visible(False)
                continue
            for chain in q:
                ax.plot(chain[:, j], lw=.4, alpha=.65)
            ax.set_title(f'log {names[j]} | R-hat={report["diagnostics"]["rhat"][j]:.3f}', fontsize=9)
        fig.suptitle(LABELS[report['model']]+' retained HMC traces (standardized units)')
        fig.savefig(folder/'traces.png', dpi=110)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('figure/airfoil_single_function_hmc'))
    parser.add_argument('--stage', choices=['prepare', 'fit', 'summarize', 'all'], default='all')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--n-train', type=int, default=300)
    parser.add_argument('--split-seed', type=int, default=17026)
    parser.add_argument('--seed', type=int, default=18026)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=1000)
    parser.add_argument('--draws', type=int, default=1500)
    parser.add_argument('--target-accept', type=float, default=.95)
    parser.add_argument('--max-retries', type=int, default=1)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        data = prepare(args.output_dir, args.n_train, args.split_seed)
        if args.stage in ('fit', 'all'):
            for model in args.models:
                fit_model(args.output_dir, data, model, chains=args.chains, warmup=args.warmup,
                          draws=args.draws, seed=args.seed, target_accept=args.target_accept,
                          max_retries=args.max_retries)
        if args.stage in ('summarize', 'all'):
            summarize(args.output_dir)


if __name__ == '__main__':
    main()
