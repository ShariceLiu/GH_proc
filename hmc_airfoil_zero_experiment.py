"""Add one-SE GP/STP/GHP to Airfoil and score CRPS for all seven models.

Reuse the exact saved 300/1203 split and training-only standardization.
Zero means apply in standardized response units. No original fits are changed.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import json
import multiprocessing
from pathlib import Path
import time

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from threadpoolctl import threadpool_limits

from GH_tabular_hmc import MODELS, ZERO_MODELS, ALL_MODELS, default_center
from hmc_airfoil_experiment import fit_model, scores, write_json
from predictive_crps import gaussian_mixture_crps
from score_single_function_crps import independent_check

LABELS = dict(gp='GP (two SE)', tp_mu_gp='STP + GP mean', mu_gp='GHP (GP mu)',
              beta_gp='GHP (GP beta)', gp_zero='GP (mu=0, one SE)',
              stp_zero='STP (mu=0, one SE)', ghp_zero='GHP (mu=beta=0, one SE)')


def case_folder(root, model):
    return Path(root)/model if model in MODELS else Path(root)/'zero_mean_baselines'/model


def prepare(root):
    root = Path(root)
    with np.load(root/'dataset.npz') as saved:
        data = {k: saved[k].copy() for k in saved.files}
    train, test = data['train_ids'], data['test_ids']
    assert len(train) == 300 and len(test) == 1203 and data['x'].shape == (1503, 5)
    assert_array_equal(np.sort(np.r_[train, test]), np.arange(len(data['y'])))
    assert_allclose(data['x_mean'], data['x'][train].mean(axis=0), atol=1e-12)
    assert_allclose(data['x_scale'], data['x'][train].std(axis=0), atol=1e-12)
    assert_allclose(data['y_mean'], data['y'][train].mean(), atol=1e-12)
    assert_allclose(data['y_scale'], data['y'][train].std(), atol=1e-12)
    assert_allclose(data['x_scaled'], (data['x']-data['x_mean'])/data['x_scale'])
    assert_allclose(data['y_scaled'], (data['y']-data['y_mean'])/data['y_scale'])
    folder = root/'zero_mean_baselines'
    folder.mkdir(parents=True, exist_ok=True)
    sources = [root/'dataset.npz', root/'experiment.json', root/'scores.csv']
    for model in MODELS:
        sources.extend([root/model/'report.json', root/model/'predictions.npz'])
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    provenance = folder/'source_provenance.json'
    if provenance.exists() and json.loads(provenance.read_text()) != hashes:
        raise ValueError('Original Airfoil data/results changed; review provenance')
    if not provenance.exists():
        write_json(provenance, hashes)
    if not (folder/'experiment.json').exists():
        write_json(folder/'experiment.json', dict(n_train=300, n_test=1203, split_seed=int(data['split_seed']),
               models=list(ZERO_MODELS), kernel='One ARD SE kernel with five inverse squared length scales',
               mean='Zero for standardized y; fixed training-y mean restored in original units',
               learned='v_0, k, w[0:5], plus W for STP/GHP',
               fixed_gig=dict(l=1., omega=1.), fixed_stp_nu=5.,
               prior_center=default_center(data['y_scaled'][train], 5, 'gp_zero'), prior_sd=2.,
               prior_convention='Keep original residual-kernel and noise priors; remove second GP',
               scores='Only y: latent f is not observed. RMSE and CRPS in dB.'))
    return data


def fit_worker(root, model, settings):
    root = Path(root)
    with np.load(root/'dataset.npz') as saved:
        data = {k: saved[k].copy() for k in saved.files}
    log_folder = root/'zero_mean_baselines'/'logs'
    log_folder.mkdir(exist_ok=True)
    with (log_folder/f'{model}.log').open('a', encoding='utf-8') as log:
        with redirect_stdout(log), redirect_stderr(log), threadpool_limits(limits=1):
            report = fit_model(root/'zero_mean_baselines', data, model, **settings)
    if not report['diagnostics_ok']:
        raise RuntimeError(f'{model} needs further HMC convergence work')
    return {k: report[k] for k in ['model', 'y_rmse', 'y_mean_marginal_nlpd', 'max_rhat',
                                  'min_bulk_ess', 'min_tail_ess', 'divergences', 'diagnostics_ok']}


def fit_all(root, settings, workers):
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs = [pool.submit(fit_worker, str(root), model, settings) for model in ZERO_MODELS]
        for job in as_completed(jobs):
            print(json.dumps(job.result()), flush=True)


def score_crps(root, models=ALL_MODELS):
    root = Path(root)
    data = prepare(root)
    truth = data['y'][data['test_ids']]
    folder = root/'crps'
    folder.mkdir(exist_ok=True)
    settings = dict(atol=1e-8, rtol=1e-8, tail_sd=10., batch_size=16)
    for model in models:
        source = case_folder(root, model)/'predictions.npz'
        report = json.loads((source.parent/'report.json').read_text())
        if not report['diagnostics_ok']:
            raise RuntimeError(f'{model}: HMC diagnostic check failed')
        provenance = dict(prediction_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                          dataset_sha256=hashlib.sha256((root/'dataset.npz').read_bytes()).hexdigest(),
                          settings=settings)
        report_path = folder/f'{model}.json'
        out = folder/f'{model}.npz'
        if report_path.exists() and out.exists():
            cached = json.loads(report_path.read_text())
            if cached['provenance'] != provenance:
                raise ValueError(f'{model}: CRPS sources changed')
            print(f'{model}: reusing CRPS={cached["y_crps"]:.8f}', flush=True)
            continue
        started = time.perf_counter()
        with np.load(source) as p:
            assert_array_equal(p['test_ids'], data['test_ids'])
            weights = p['weights']
            means, variances = p['component_means'], p['component_y_variances']
            previous = scores(truth, p)
            for key, value in previous.items():
                assert_allclose(value, report[key], atol=1e-10)
            print(f'{model}: y CRPS, {len(weights)} components x {len(truth)} test rows', flush=True)
            pointwise, numerical = gaussian_mixture_crps(truth, means, variances, weights, **settings)
            checks = []
            for j in np.linspace(0, len(truth)-1, 5, dtype=int):
                checks.append(dict(test_row=int(data['test_ids'][j]), **independent_check(
                    truth[j], means[:, j], variances[:, j], weights, pointwise[j])))
        np.savez_compressed(out, test_ids=data['test_ids'], y_crps=pointwise)
        result = dict(model=model, y_crps=float(pointwise.mean()), provenance=provenance,
                      validation_passed=True, all_components_used=True, original_units='dB',
                      numerical=numerical, independent_checks=checks, seconds=time.perf_counter()-started)
        write_json(report_path, result)
        print(f'{model}: mean y CRPS={pointwise.mean():.8f}', flush=True)


def summarize(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    data = prepare(root)
    rows, pointwise_rows = [], []
    test = data['test_ids']
    for model in ALL_MODELS:
        report = json.loads((case_folder(root, model)/'report.json').read_text())
        crps = json.loads((root/'crps'/f'{model}.json').read_text())
        rows.append(dict(**{k: report[k] for k in ['model', 'n_train', 'n_test', 'y_rmse',
                         'y_mean_marginal_nlpd', 'y_coverage_95', 'y_mean_interval_width', 'max_rhat',
                         'min_bulk_ess', 'min_tail_ess', 'divergences', 'diagnostics_ok']},
                         y_crps=crps['y_crps'], n_signal_components=2 if model in MODELS else 1))
        with np.load(root/'crps'/f'{model}.npz') as saved:
            assert_array_equal(saved['test_ids'], test)
            pointwise_rows.extend(dict(model=model, test_row=int(index), y_true=float(data['y'][index]),
                                       y_crps=float(value)) for index, value in zip(test, saved['y_crps']))
    table = pd.DataFrame(rows)
    table.to_csv(root/'scores_with_zero_mean_crps.csv', index=False)
    pd.DataFrame(pointwise_rows).to_csv(root/'crps'/'pointwise_crps.csv', index=False)
    latex = dict(LABELS, mu_gp=r'GHP ($\mu$ GP)', beta_gp=r'GHP ($\beta$ GP)',
                 gp_zero=r'GP ($\mu=0$, one SE)', stp_zero=r'STP ($\mu=0$, one SE)',
                 ghp_zero=r'GHP ($\mu=\beta=0$, one SE)')
    lines = [r'% Requires \usepackage{booktabs}', r'\begin{table}[htbp]', r'\centering', r'\small',
             r'\caption{Airfoil regression: the same 300 training and 1,203 held-out rows for all models. '
             r'Scores use HMC posterior predictive mixtures; RMSE and CRPS are in dB. '
             r'GHP fixes $\lambda=\omega=1$; STP fixes $\nu=5$. '
             r'Zero means refer to the training-standardized response. '
             r'Latent $f$ is unknown, so all scores concern $y$.}',
             r'\label{tab:airfoil_seven_models}', r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Model & RMSE ($y$) & NLPD ($y$) & CRPS ($y$) & 95\% coverage \\', r'\midrule']
    for row in rows:
        if row['model'] == ZERO_MODELS[0]:
            lines.append(r'\midrule')
        lines.append(latex[row['model']]+' & '+ ' & '.join(f'{row[k]:.5f}' for k in
                     ['y_rmse', 'y_mean_marginal_nlpd', 'y_crps'])+f' & {100*row["y_coverage_95"]:.2f}'+r'\% \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (root/'comparison_with_zero_mean_crps.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    fig, axes = plt.subplots(7, 2, figsize=(13, 21), constrained_layout=True)
    for i, row in enumerate(rows):
        model = row['model']
        with np.load(case_folder(root, model)/'predictions.npz') as p:
            mean, ci = p['y_mean'], p['y_ci']
        truth = data['y'][test]
        ax = axes[i, 0]
        ax.scatter(truth, mean, s=8, alpha=.4)
        limits = [min(truth.min(), mean.min()), max(truth.max(), mean.max())]
        ax.plot(limits, limits, 'k--', lw=1)
        ax.set(xlabel='Observed y (dB)', ylabel='Predictive mean (dB)',
               title=f'{LABELS[model]} | RMSE={row["y_rmse"]:.3f}; CRPS={row["y_crps"]:.3f}')
        ids = np.argsort(mean)
        ax = axes[i, 1]
        ax.fill_between(np.arange(len(test)), ci[ids, 0], ci[ids, 1], alpha=.25, label='95% predictive interval')
        ax.plot(mean[ids], lw=1, label='Predictive mean')
        ax.scatter(np.arange(len(test)), truth[ids], c='black', s=3, alpha=.3, label='Held-out y')
        ax.set(xlabel='Test rows ordered by predictive mean', ylabel='y (dB)',
               title=f'NLPD={row["y_mean_marginal_nlpd"]:.3f}; coverage={row["y_coverage_95"]:.1%}')
        if i == 0:
            ax.legend(fontsize=7)
    fig.suptitle('Airfoil | same 300 / 1,203 split | four two-SE models and three one-SE baselines')
    fig.savefig(root/'regression_with_zero_mean_crps.png', dpi=140)
    plt.close(fig)
    for model in ZERO_MODELS:
        folder = case_folder(root, model)
        report = json.loads((folder/'report.json').read_text())
        with np.load(folder/report['selected_samples_file']) as saved:
            q, names = saved['log_samples'], saved['parameter_names']
        fig, axes = plt.subplots(4, 2, figsize=(11, 10), constrained_layout=True)
        for j, ax in enumerate(axes.flat):
            if j >= q.shape[-1]:
                ax.set_visible(False)
                continue
            for chain in q:
                ax.plot(chain[:, j], lw=.4, alpha=.6)
            ax.set_title(f'log {names[j]} | R-hat={report["diagnostics"]["rhat"][j]:.4f}')
        fig.suptitle(LABELS[model]+' (standardized units)')
        fig.savefig(folder/'traces.png', dpi=120)
        plt.close(fig)
    write_json(root/'summary_with_zero_mean_crps.json', dict(completed_models=7,
               all_diagnostics_ok=all(r['diagnostics_ok'] for r in rows),
               max_rhat=max(r['max_rhat'] for r in rows),
               min_bulk_ess=min(r['min_bulk_ess'] for r in rows),
               min_tail_ess=min(r['min_tail_ess'] for r in rows),
               retained_divergences=sum(r['divergences'] for r in rows), scores=rows))
    print(table[['model', 'y_rmse', 'y_mean_marginal_nlpd', 'y_crps', 'y_coverage_95']].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('figure/airfoil_single_function_hmc'))
    parser.add_argument('--stage', choices=['prepare', 'fit', 'crps', 'summarize', 'validate', 'all'], default='all')
    parser.add_argument('--models', nargs='+', choices=ALL_MODELS, default=list(ALL_MODELS), help='CRPS model subset')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--seed', type=int, default=24026)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=1000)
    parser.add_argument('--draws', type=int, default=1500)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    with threadpool_limits(limits=1):
        prepare(args.root)
        if args.stage in ['fit', 'all']:
            settings = dict(chains=args.chains, warmup=args.warmup, draws=args.draws,
                            seed=args.seed, target_accept=.95, max_retries=1)
            config = args.root/'zero_mean_baselines'/'sampling_settings.json'
            if config.exists() and json.loads(config.read_text()) != settings:
                raise ValueError('Sampling settings differ from existing experiment')
            write_json(config, settings)
            fit_all(args.root, settings, args.workers)
        if args.stage in ['crps', 'all']:
            score_crps(args.root, args.models)
        if args.stage in ['summarize', 'all']:
            summarize(args.root)
        if args.stage in ['validate', 'all']:
            from validate_airfoil_zero_experiment import validate
            validate(args.root)


if __name__ == '__main__':
    main()
