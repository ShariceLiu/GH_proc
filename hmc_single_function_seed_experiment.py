"""Repeat the nine-model 30/151 experiment on a new 1D GHP-beta realization.

Default data seed is 6 (previous experiment: 5). Generation happens once;
every model sees the same saved noisy training values and locations. Outputs
are isolated by seed. Completed fits are reused, including for resumed runs.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from fit_beta_mu_1d_experiment import simulate_1d
from hmc_1d_split_experiments import make_split, write_json, fit_model as fit_two_se
from hmc_zero_mean_30_experiment import fit_model as fit_one_se, LABELS
from hmc_fixed_mu_30_experiment import gig_prior_mean, summarize as summarize_fixed_mu


MODELS = ('gp', 'tp_mu_gp', 'mu_gp', 'beta_gp', 'gp_zero', 'stp_zero',
          'ghp_zero', 'ghp_known_beta', 'ghp_fixed_mu')


def paths(root, model):
    root = Path(root)
    if model in MODELS[:4]:
        folder = root/'sparse'
        return folder, folder/f'{model}_report.json', folder/f'{model}_predictions.npz'
    group = ('zero_mean_baselines' if model in MODELS[4:7] else
             'known_beta' if model == 'ghp_known_beta' else 'fixed_mu_prior_mean')
    folder = root/group/model
    return folder, folder/'report.json', folder/'predictions.npz'


def load_data(root):
    with np.load(Path(root)/'sparse'/'dataset.npz') as saved:
        return {key: saved[key].copy() for key in saved.files}


def prepare(root, seed, settings):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    folder = root/'sparse'
    folder.mkdir(exist_ok=True)
    configuration = dict(data_seed=seed, hmc=settings, models=list(MODELS), n_train=30, n_test=151,
                         n_points=181, domain=[0., 10.], prior_log_sd=2.,
                         fixed_gig=dict(l=1., omega=1.), fixed_stp_nu=5.,
                         fixed_mu_multiplier=gig_prior_mean(),
                         split='Same approximately equally spaced training locations as seed 5',
                         generation='simulate_1d(seed, n_points=181, l=1, omega=1), single BLAS thread',
                         mean_information='Only final two models receive beta_true; true W is never supplied to fitting')
    config_path = root/'configuration.json'
    if config_path.exists():
        if json.loads(config_path.read_text()) != configuration:
            raise ValueError('Configuration differs from saved run; use a new output directory')
    else:
        if (folder/'dataset.npz').exists():
            raise ValueError('Existing data without matching configuration; use a fresh output folder')
        write_json(config_path, configuration)
    if not (folder/'dataset.npz').exists():
        with threadpool_limits(limits=1):
            original = simulate_1d(seed=seed)
        data = make_split(original, 'sparse', sparse_points=30)
        data['mu_fixed'] = gig_prior_mean()*data['beta']
        np.savez_compressed(folder/'dataset.npz', **{k: v for k, v in data.items()
                                                   if isinstance(v, np.ndarray) or k == 'W'})
        write_json(folder/'split.json', dict(data_seed=seed, generating_parameters=original['parameters'],
                   generating_W=original['W'], n_train=30, n_test=151,
                   train_ids=data['train_ids'], test_ids=data['test_ids']))
    data = load_data(root)
    # Read only the earlier locations, not responses or any selected fit parameters.
    original_root = Path('figure/single_function_30_hmc')
    with np.load(original_root/'sparse'/'dataset.npz') as old:
        for key in ['x', 'train_ids', 'test_ids']:
            np.testing.assert_array_equal(data[key], old[key])
        if seed != 5 and np.array_equal(data['y'], old['y']):
            raise ValueError('Expected a different generated realization')
    files = [original_root/'sparse'/'dataset.npz', original_root/'scores_with_crps.csv']
    sources = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    provenance = dict(previous_experiment=sources,
                      new_dataset_sha256=hashlib.sha256((folder/'dataset.npz').read_bytes()).hexdigest())
    provenance_path = root/'source_provenance.json'
    if provenance_path.exists():
        if json.loads(provenance_path.read_text()) != provenance:
            raise ValueError('Saved new data or previous comparison sources changed')
    else:
        write_json(provenance_path, provenance)
    print(f'Data seed {seed}: W={float(data["W"]):.8f}; 30 train / 151 test', flush=True)
    return data


def fit_worker(root, model, settings):
    root = Path(root)
    folder, report_path, prediction_path = paths(root, model)
    if report_path.exists() and prediction_path.exists():
        report = json.loads(report_path.read_text())
        if report['diagnostics_ok']:
            return dict(model=model, reused=True, scores=report['scores'])
    data = load_data(root)
    logs = root/'logs'
    logs.mkdir(exist_ok=True)
    with (logs/f'{model}.log').open('a', encoding='utf-8') as log:
        with redirect_stdout(log), redirect_stderr(log), threadpool_limits(limits=1):
            if model in MODELS[:4]:
                report = fit_two_se(data, 'sparse', model, root/'sparse', **settings)
            else:
                one_settings = dict(settings, seed=settings['seed']+1000)
                report = fit_one_se(data, folder.parent, model, **one_settings)
    if not report['diagnostics_ok']:
        raise RuntimeError(f'{model} did not pass HMC diagnostics; see {report_path}')
    return dict(model=model, reused=False, scores=report['scores'])


def fit_all(root, settings, workers=3):
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs = {pool.submit(fit_worker, str(root), model, settings): model for model in MODELS}
        for job in as_completed(jobs):
            result = job.result()
            print(json.dumps(result), flush=True)


def summarize(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    data = load_data(root)
    config = json.loads((root/'configuration.json').read_text())
    rows = []
    for model in MODELS:
        _, report_path, _ = paths(root, model)
        report = json.loads(report_path.read_text())
        if not report['diagnostics_ok']:
            raise RuntimeError(f'Resolve {model} diagnostics before summarizing')
        rows.append(dict(report['scores'], scenario='sparse', data_seed=config['data_seed'],
                         n_signal_components=2 if model in MODELS[:4] else 1,
                         known_beta=model == 'ghp_known_beta',
                         component_swap_acceptance=report['scores'].get('component_swap_acceptance', 0.)))
    table = pd.DataFrame(rows)
    table.iloc[:4].to_csv(root/'scores.csv', index=False)
    table.iloc[:7].to_csv(root/'scores_with_zero_mean.csv', index=False)
    table.iloc[:8].to_csv(root/'scores_with_known_beta.csv', index=False)
    # This summarizer is data-independent and reads this root's saved dataset.
    summarize_fixed_mu(root)
    # Preserve the simulation seed on the added ninth row too.
    nine = pd.read_csv(root/'scores_with_fixed_mu.csv')
    nine['data_seed'] = config['data_seed']
    nine.to_csv(root/'scores_with_fixed_mu.csv', index=False)
    labels = dict(LABELS, ghp_known_beta='GHP (known beta; W inferred)',
                  ghp_fixed_mu='GHP (fixed mu = prior E[W] beta)')
    fig, axes = plt.subplots(9, 2, figsize=(13, 24), sharex=True, sharey=True, constrained_layout=True)
    for i, row in enumerate(rows):
        model = row['model']
        with np.load(paths(root, model)[2]) as p:
            for j, target in enumerate(['f', 'y']):
                ax = axes[i, j]
                ax.plot(data['x'], data['f'], 'k--', lw=1., label='True f')
                ax.plot(data['x'], p[f'{target}_mean'], color=f'C{i}', label='Posterior mean')
                ax.fill_between(data['x'], *p[f'{target}_ci'].T, color=f'C{i}', alpha=.23,
                                label='Pointwise 95% interval')
                train, test = data['train_ids'], data['test_ids']
                ax.scatter(data['x'][train], data['y'][train], c='black', s=10, alpha=.65, label='Training y')
                if target == 'y':
                    ax.scatter(data['x'][test], data['y'][test], c='gray', marker='x', s=8, alpha=.45, label='Test y')
                ax.set(title=f'{labels[model]} | {target} RMSE={row[target+"_rmse"]:.4f}; '
                             f'NLPD={row[target+"_mean_marginal_nlpd"]:.4f}', ylabel=target)
                ax.grid(alpha=.18)
                if i == 0:
                    ax.legend(fontsize=7, ncol=2)
    for ax in axes[-1]:
        ax.set_xlabel('x')
    fig.suptitle(f'Data seed {config["data_seed"]} | 30 training / 151 test | final two models receive true beta')
    fig.savefig(root/'regression_all_models.png', dpi=140)
    plt.close(fig)
    summary = dict(data_seed=config['data_seed'], generating_W=float(data['W']), completed_models=len(rows),
                   all_diagnostics_ok=True, max_rhat=max(r['max_rhat'] for r in rows),
                   min_bulk_ess=min(r['min_bulk_ess'] for r in rows),
                   min_tail_ess=min(r['min_tail_ess'] for r in rows),
                   retained_divergences=sum(r['divergences'] for r in rows), scores=rows)
    write_json(root/'summary.json', summary)


def compare_seeds(root):
    root = Path(root)
    old = pd.read_csv('figure/single_function_30_hmc/scores_with_crps.csv').assign(data_seed=5)
    new = pd.read_csv(root/'scores_with_crps.csv')
    columns = ['model', 'data_seed', 'f_rmse', 'f_mean_marginal_nlpd', 'f_crps',
               'y_rmse', 'y_mean_marginal_nlpd', 'y_crps']
    pd.concat([old[columns], new[columns]], ignore_index=True).to_csv(root/'comparison_by_seed.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-seed', type=int, default=6)
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--stage', choices=['prepare', 'fit', 'summarize', 'crps', 'validate', 'all'], default='all')
    parser.add_argument('--workers', type=int, default=3)
    parser.add_argument('--sampler-seed', type=int, default=22026)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=2000)
    parser.add_argument('--draws', type=int, default=3000)
    args = parser.parse_args()
    if args.data_seed < 0 or args.workers < 1:
        parser.error('Need nonnegative data seed and positive worker count')
    root = args.output_dir or Path(f'figure/single_function_30_hmc_seed{args.data_seed}')
    if root.resolve() == Path('figure/single_function_30_hmc').resolve():
        parser.error('Use a new folder; the original seed-5 experiment is preserved')
    settings = dict(chains=args.chains, warmup=args.warmup, draws=args.draws,
                    seed=args.sampler_seed, target_accept=.99, max_retries=1)
    with threadpool_limits(limits=1):
        prepare(root, args.data_seed, settings)
        if args.stage in ['fit', 'all']:
            fit_all(root, settings, args.workers)
        if args.stage in ['summarize', 'all']:
            summarize(root)
        if args.stage in ['validate', 'all']:
            from validate_single_function_seed_experiment import validate
            validate(root)
        if args.stage in ['crps', 'all']:
            from score_single_function_crps import run
            run(root)
            compare_seeds(root)


if __name__ == '__main__':
    main()
