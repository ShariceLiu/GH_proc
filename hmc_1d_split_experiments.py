"""Sparse, gap and extrapolation comparisons on the SAME saved 1D realization.

Run: python hmc_1d_split_experiments.py
Models: two-SE GP, TP residual + independent GP mean, GHP mu, GHP beta.
All kernel/noise hyperparameters are refitted for each split with HMC.
GIG lambda=omega=1 and TP nu=5 are fixed. No new functions are generated.
Use --scenarios/--models for individual runs, --summarize to rebuild figures.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from fit_beta_mu_1d_experiment import _json_default, _save_csv
from hmc_beta_mu_1d_experiment import load_dataset
from hmc_gp_tp_1d_experiment import score_predictions, diagnostics_summary
from GH_proc_joint_hmc import sample_joint_parameters, predict_joint_mixture
from GP_TP_hmc import sample_baseline_parameters, predict_baseline_mixture


SCENARIOS = ('sparse', 'gap', 'extrapolation')
MODELS = ('gp', 'tp_mu_gp', 'mu_gp', 'beta_gp')
LABELS = {'gp': 'GP', 'tp_mu_gp': 'STP + GP mean',
          'mu_gp': 'GHP (mu)', 'beta_gp': 'GHP (beta)'}
COLORS = dict(zip(MODELS, ['C2', 'C3', 'C1', 'C0']))
SPLIT_LABELS = {'sparse': 'Sparse observations', 'gap': 'Gap: x in [4, 6]',
                'extrapolation': 'Extrapolation: x > 6'}


def make_split(original, scenario, sparse_points=15):
    """Subsample the original training set; only x determines the split.

    Sparse: evenly spread training locations, including domain endpoints;
    score every unused location. Gap/extrapolation: remove original training
    locations in the target region and score ONLY that region. Other unused
    points remain available for secondary scores, never for training/priors.
    """
    if scenario not in SCENARIOS:
        raise ValueError(f'Unknown scenario: {scenario}')
    x = np.asarray(original['x'])
    base = np.asarray(original['train_ids'], dtype=int)
    if scenario == 'sparse':
        if not isinstance(sparse_points, (int, np.integer)) or not 2 <= sparse_points < len(base):
            raise ValueError('sparse_points must be between 2 and the original training count minus one')
        train = base[np.rint(np.linspace(0, len(base)-1, sparse_points)).astype(int)]
        test = np.setdiff1d(np.arange(len(x)), train)
    elif scenario == 'gap':
        train = base[(x[base] < 4.) | (x[base] > 6.)]
        test = np.flatnonzero((x >= 4.) & (x <= 6.))
    else:
        train = base[x[base] <= 6.]
        test = np.flatnonzero(x > 6.)
    if not len(train) or not len(test) or np.intersect1d(train, test).size:
        raise ValueError('Split requires nonempty disjoint training and test sets')
    data = dict(original)
    data.update(train_ids=train, test_ids=test,
                original_train_ids=base.copy(),
                all_unobserved_ids=np.setdiff1d(np.arange(len(x)), train),
                unused_ids=np.setdiff1d(np.arange(len(x)), np.r_[train, test]))
    return data


def prior_center_from_training(y):
    """Same empirical log-prior convention as the earlier experiments."""
    scale = float(np.var(y))
    scale = scale if scale > 1e-12 else 1.
    return np.log([.5*scale, .5*scale, .1*scale, 1., .25])


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=_json_default), encoding='utf-8')


def fit_model(data, scenario, model, folder, *, chains=4, warmup=1000, draws=1500,
              seed=5026, prior_sd=2., target_accept=.9, max_retries=1):
    """Refit on this split; retry only for failed sampler diagnostics.

    Every attempt is saved. On a retry, double warmup/draws and increase
    target acceptance; predictive scores never determine which run to use.
    """
    train = data['train_ids']
    y, x = data['y'][train], data['x'][train]
    center = prior_center_from_training(y)
    model_seed = seed+100*SCENARIOS.index(scenario)+MODELS.index(model)
    attempts_path = folder/f'{model}_attempts.json'
    attempts = json.loads(attempts_path.read_text(encoding='utf-8')) if attempts_path.exists() else []
    attempt_offset = len(attempts)
    started = time.perf_counter()
    for attempt in range(max_retries+1):
        settings = dict(chains=chains, warmup=warmup*2**attempt, draws=draws*2**attempt,
                        seed=model_seed+10000*(attempt_offset+attempt), target_accept=min(.999, target_accept+.07*attempt),
                        leapfrog_steps=(20, 50), prior_center=center, prior_sd=prior_sd)
        attempt_number = attempt_offset+attempt+1
        print(f'{scenario} / {model}: attempt {attempt_number}, {len(train)} train, '
              f'{settings["warmup"]} warmup + {settings["draws"]} draws per chain', flush=True)
        if model == 'gp':
            fit = sample_baseline_parameters(y, x, model='gp', **settings)
        else:
            fit = sample_joint_parameters(y, x, model=model, l=1., omega=1., nu=5., **settings)
        attempt_summary = dict(attempt=attempt_number, settings=settings, **diagnostics_summary(fit),
                               warmup_divergences=int(np.sum(fit['warmup_divergent'])))
        attempts.append(attempt_summary)
        np.savez_compressed(folder/f'{model}_attempt_{attempt_number}_samples.npz',
                            parameter_names=fit['parameter_names'],
                            **{key: value for key, value in fit.items() if isinstance(value, np.ndarray)})
        write_json(folder/f'{model}_attempts.json', attempts)
        print(json.dumps({key: attempt_summary[key] for key in
                          ['max_rhat', 'min_bulk_ess', 'min_tail_ess', 'divergences', 'diagnostics_ok']}), flush=True)
        if fit['diagnostics_ok']:
            break
    sampling_seconds = time.perf_counter()-started
    print(f'{scenario} / {model}: averaging {settings["draws"]*chains} conditional predictions', flush=True)
    predict = predict_baseline_mixture if model == 'gp' else predict_joint_mixture
    prediction = predict(y, x, data['x'], fit)
    np.savez_compressed(folder/f'{model}_predictions.npz', **prediction)
    scores = dict(scenario=scenario, model=model, n_train=len(train), n_test=len(data['test_ids']),
                  **score_predictions(data, prediction, previous=model != 'gp'), **diagnostics_summary(fit))
    secondary = dict(data, test_ids=data['all_unobserved_ids'])
    # Per-chain estimates indicate Monte Carlo variation; these are not independent datasets.
    per_chain = []
    for chain in range(chains):
        indices = slice(chain*settings['draws'], (chain+1)*settings['draws'])
        chain_p = dict(prediction)
        chain_p['weights'] = np.full(settings['draws'], 1/settings['draws'])
        for target in ['f', 'y']:
            key = 'component_means' if model == 'gp' else f'component_{target}_means'
            chain_p[key] = prediction[key][indices]
            chain_p[f'{target}_mean'] = chain_p[key].mean(axis=0)
            chain_p[f'component_{target}_variances'] = prediction[f'component_{target}_variances'][indices]
        chain_scores = score_predictions(data, chain_p, previous=model != 'gp')
        per_chain.append(dict(chain=chain+1, **{key: value for key, value in chain_scores.items()
                                             if key.endswith('rmse') or key.endswith('nlpd')}))
    _save_csv(folder/f'{model}_per_chain_scores.csv', per_chain)
    report = dict(scenario=scenario, model=model, scores=scores,
                  all_unobserved_scores=score_predictions(secondary, prediction, previous=model != 'gp'),
                  attempts=attempts, selected_attempt=attempt_number,
                  selected_samples_file=f'{model}_attempt_{attempt_number}_samples.npz',
                  sampling_seconds=sampling_seconds, total_seconds=time.perf_counter()-started,
                  **{key: fit[key] for key in ['parameter_names', 'prior_center', 'prior_sd',
                     'diagnostics', 'diagnostics_ok', 'summary', 'initialization', 'step_size', 'inverse_mass']})
    if model != 'gp':
        report['effective_diagnostics'] = fit['effective_diagnostics']
    write_json(folder/f'{model}_report.json', report)
    print(f'{scenario} / {model}: f RMSE={scores["f_rmse"]:.5f}, f NLPD={scores["f_mean_marginal_nlpd"]:.5f}', flush=True)
    return report


def shade_region(ax, scenario):
    if scenario == 'gap':
        ax.axvspan(4., 6., color='gray', alpha=.09)
    elif scenario == 'extrapolation':
        ax.axvspan(6., 10., color='gray', alpha=.09)
        ax.axvline(6., color='gray', lw=.8, ls=':')


def plot_panel(ax, data, p, row, scenario, model, target='f'):
    x, train, test = data['x'], data['train_ids'], data['test_ids']
    shade_region(ax, scenario)
    ax.plot(x, data['f'], 'k--', lw=1.2, label='True f')
    ax.plot(x, p[f'{target}_mean'], color=COLORS[model], label='Posterior mean')
    ax.fill_between(x, *p[f'{target}_ci'].T, color=COLORS[model], alpha=.22, label='Pointwise 95% interval')
    ax.scatter(x[train], data['y'][train], s=12, color='black', alpha=.6, label='Training y', zorder=3)
    if target == 'y':
        ax.scatter(x[test], data['y'][test], s=12, marker='x', c='gray', alpha=.5, label='Held-out y')
    flag = '' if row['diagnostics_ok'] else ' [CHECK HMC]'
    ax.set_title(f'{LABELS[model]}{flag}\n{target} RMSE={row[target+"_rmse"]:.3f}; '
                 f'NLPD={row[target+"_mean_marginal_nlpd"]:.3f}', fontsize=10)
    ax.set(xlabel='x', ylabel=target)
    ax.grid(alpha=.2)


def summarize(root):
    """Read completed fits; regenerate score tables and figures without resampling."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    scores, parameters, completed = [], [], {}
    for scenario in SCENARIOS:
        folder = root/scenario
        if not (folder/'dataset.npz').exists():
            continue
        with np.load(folder/'dataset.npz') as saved:
            data = {key: saved[key].copy() for key in saved.files}
        models = {}
        for model in MODELS:
            path = folder/f'{model}_report.json'
            if not path.exists():
                continue
            report = json.loads(path.read_text(encoding='utf-8'))
            with np.load(folder/f'{model}_predictions.npz') as saved:
                p = {f'{t}_{stat}': saved[f'{t}_{stat}'].copy()
                     for t in ['f', 'y'] for stat in ['mean', 'ci']}
            models[model] = (p, report)
            scores.append(report['scores'])
            parameters.extend(dict(scenario=scenario, model=model, **row) for row in report['summary'])
        if not models:
            continue
        completed[scenario] = (data, models)
        _save_csv(folder/'scores.csv', [r for r in scores if r['scenario'] == scenario])
        _save_csv(folder/'parameters.csv', [r for r in parameters if r['scenario'] == scenario])
        fig, axes = plt.subplots(2, 4, figsize=(17, 7), sharex=True, sharey='row', constrained_layout=True)
        for col, model in enumerate(MODELS):
            if model not in models:
                for ax in axes[:, col]:
                    ax.set_visible(False)
                continue
            p, report = models[model]
            for row, target in enumerate(['f', 'y']):
                plot_panel(axes[row, col], data, p, report['scores'], scenario, model, target)
            axes[0, col].legend(fontsize=7)
        fig.suptitle(f'{SPLIT_LABELS[scenario]} | {len(data["train_ids"])} training, '
                     f'{len(data["test_ids"])} scored test points | same saved realization')
        fig.savefig(folder/'regression.png', dpi=155)
        plt.close(fig)
        fig, axes = plt.subplots(6, 4, figsize=(16, 12), constrained_layout=True)
        for col, model in enumerate(MODELS):
            if model not in models:
                for ax in axes[:, col]:
                    ax.set_visible(False)
                continue
            report = models[model][1]
            with np.load(folder/report['selected_samples_file']) as saved:
                samples = saved['log_samples']
            for j in range(6):
                if j == samples.shape[2]:
                    axes[j, col].set_visible(False)
                    continue
                for chain in samples:
                    axes[j, col].plot(chain[:, j], lw=.35, alpha=.6)
                axes[j, col].set_ylabel('log '+report['parameter_names'][j], fontsize=8)
                axes[j, col].set_title(f'{LABELS[model]}: R-hat={report["diagnostics"]["rhat"][j]:.3f}, '
                                       f'ESS={report["diagnostics"]["ess_bulk"][j]:.0f}', fontsize=8)
        fig.savefig(folder/'traces.png', dpi=130)
        plt.close(fig)
    if not scores:
        return
    _save_csv(root/'scores.csv', scores)
    _save_csv(root/'parameters.csv', parameters)
    fig, axes = plt.subplots(len(completed), 4, figsize=(17, 3.5*len(completed)),
                             sharex=True, sharey='row', squeeze=False, constrained_layout=True)
    for row, (scenario, (data, models)) in enumerate(completed.items()):
        for col, model in enumerate(MODELS):
            if model not in models:
                axes[row, col].set_visible(False)
                continue
            p, report = models[model]
            plot_panel(axes[row, col], data, p, report['scores'], scenario, model)
        axes[row, 0].set_ylabel(SPLIT_LABELS[scenario]+'\nLatent f')
    fig.suptitle('Posterior means and pointwise 95% intervals | dashed black: true f | points: training y', fontsize=13)
    fig.savefig(root/'overview.png', dpi=160)
    plt.close(fig)
    labels_tex = {'gp': 'GP', 'tp_mu_gp': 'STP + GP mean',
                  'mu_gp': r'GHP ($\mu$)', 'beta_gp': r'GHP ($\beta$)'}
    lines = [r'% Requires \usepackage{booktabs}', r'\begin{table}[htbp]', r'\centering',
             r'\caption{Single-function regression with reduced observations. '
             r'Gap and extrapolation scores use only the designated unobserved region. '
             r'Predictions average over the HMC posterior.}',
             r'\label{tab:split_regression}', r'\begin{tabular}{llrrrr}', r'\toprule',
             r'Split & Model & RMSE ($f$) & NLPD ($f$) & RMSE ($y$) & NLPD ($y$) \\', r'\midrule']
    for i, row in enumerate(scores):
        if i and row['scenario'] != scores[i-1]['scenario']:
            lines.append(r'\midrule')
        lines.append(row['scenario'].capitalize()+' & '+labels_tex[row['model']]+' & '+
                     ' & '.join(f'{row[key]:.6f}' for key in
                                ['f_rmse', 'f_mean_marginal_nlpd', 'y_rmse', 'y_mean_marginal_nlpd'])+r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    (root/'comparison_table.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_json(root/'summary.json', dict(completed_fits=len(scores),
               all_diagnostics_ok=all(r['diagnostics_ok'] for r in scores),
               fixed_mixing_parameters=dict(GHP=dict(l=1., omega=1.),
                                             STP=dict(nu=5., ig_shape=2.5, ig_scale=1.5)),
               max_rhat=max(r['max_rhat'] for r in scores),
               min_bulk_ess=min(r['min_bulk_ess'] for r in scores),
               min_tail_ess=min(r['min_tail_ess'] for r in scores), scores=scores))
    (root/'README.md').write_text(
        '# Same-function sparse, gap and extrapolation experiments\n\n'
        'Run `python hmc_1d_split_experiments.py`. Rebuild figures/tables with '
        '`python hmc_1d_split_experiments.py --summarize`. Individual runs can use '
        '`--scenarios gap --models gp`, for example. Existing result files for selected '
        'model predictions/reports are replaced on rerun; all sampler attempts '
        'and other scenarios/models are retained.\n\n'
        'All cases load the exact earlier seed-5 realization, including its original noise. '
        'Training is always a subset of the original 91 training locations. '
        'Sparse uses 15 evenly spread training locations by default (including endpoints) '
        'and scores every unused point. Gap removes training points in [4,6] and scores '
        'only that interval. Extrapolation retains training points up to 6 and scores x>6. '
        'Other unobserved locations are scored separately in each model report.\n\n'
        'Each model is refitted for each split. All five kernel/noise parameters are '
        'sampled; mixing models also sample W. GIG lambda=omega=1, TP nu=5 '
        '(inverse-gamma shape=2.5, scale=1.5). STP denotes a Student-t residual plus '
        'an independent GP mean: conditional covariance K2+W*K1. Two SE components '
        'in every model. Normal log-prior sd=2; centers use only that split\'s training '
        'y variance, with the same formula for all models. No test values or latent '
        'truth are used for fitting or priors. These empirical prior centers can '
        'differ across splits.\n\n'
        'Default HMC: 4 chains, 1000 warmup + 1500 retained draws each; component '
        'exchange proposals use the full posterior ratio. A failed diagnostic check '
        'triggers one longer, more conservative run. All attempt diagnostics/samples '
        'are saved. Selection uses diagnostics only, never prediction scores. '
        'If a run remains flagged, it can be retried with, for example, '
        '`--scenarios sparse --models gp mu_gp beta_gp --warmup 2500 --draws 3000 '
        '--target-accept 0.995 --max-retries 0`. Actual settings, all diagnostics '
        'and every attempt are recorded in JSON.\n\n'
        '- overview.png: latent f for every split/model.\n'
        '- scores.csv, parameters.csv, comparison_table.tex: combined numerical results.\n'
        '- Each split has regression.png (f/y), traces.png, dataset.npz and split.json.\n'
        '- *_report.json and *_attempts.json: priors, diagnostics, sampler settings and scores.\n'
        '- *_attempt_N_samples.npz: all retained and warmup log states and diagnostics.\n'
        '- *_predictions.npz: Gaussian-mixture components/weights, full covariance, '
        'means and exact pointwise 95% mixture intervals. Joint models also store mu '
        'or beta and its contribution to f.\n'
        '- *_per_chain_scores.csv: Monte Carlo agreement, not replicate experiments.\n\n'
        'RMSE(f) uses latent truth; RMSE(y) uses held-out noisy observations. NLPD '
        'is the negative mean log marginal mixture density. Coverage refers to '
        'pointwise intervals on one correlated realization and is not a repeated-data '
        'calibration estimate. No multiple-function simulations are included.\n', encoding='utf-8')


def run(*, output_dir='figure/single_function_split_hmc', scenarios=SCENARIOS, models=MODELS,
        sparse_points=15, no_summarize=False, **settings):
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    original = load_dataset()
    for scenario in scenarios:
        data = make_split(original, scenario, sparse_points)
        folder = root/scenario
        folder.mkdir(parents=True, exist_ok=True)
        if (folder/'dataset.npz').exists():
            with np.load(folder/'dataset.npz') as previous:
                for key in ['x', 'y', 'f', 'train_ids', 'test_ids']:
                    if not np.array_equal(data[key], previous[key]):
                        raise ValueError('Use a new output folder when changing data or the split')
        np.savez_compressed(folder/'dataset.npz', **{key: value for key, value in data.items()
                            if isinstance(value, np.ndarray) or key == 'W'})
        write_json(folder/'split.json', dict(scenario=scenario, dataset_source=data['source'],
                   generating_parameters=data['parameters'], generating_W=data['W'],
                   n_train=len(data['train_ids']), n_test=len(data['test_ids']),
                   train_ids=data['train_ids'], test_ids=data['test_ids'], unused_ids=data['unused_ids']))
        for model in models:
            fit_model(data, scenario, model, folder, **settings)
    if not no_summarize:
        summarize(root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='figure/single_function_split_hmc')
    parser.add_argument('--scenarios', nargs='+', choices=SCENARIOS, default=list(SCENARIOS))
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--sparse-points', type=int, default=15)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=1000)
    parser.add_argument('--draws', type=int, default=1500)
    parser.add_argument('--seed', type=int, default=5026)
    parser.add_argument('--prior-sd', type=float, default=2.)
    parser.add_argument('--target-accept', type=float, default=.9)
    parser.add_argument('--max-retries', type=int, default=1)
    parser.add_argument('--summarize', action='store_true')
    parser.add_argument('--no-summarize', action='store_true', help='Defer figures/tables for independent scenario runs')
    args = vars(parser.parse_args())
    only_summary = args.pop('summarize')
    if args['max_retries'] < 0:
        parser.error('--max-retries must be nonnegative')
    with threadpool_limits(limits=1):
        if only_summary:
            summarize(args['output_dir'])
        else:
            run(**args)
