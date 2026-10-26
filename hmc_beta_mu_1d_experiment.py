"""Joint (theta,W) HMC and conditional Gaussian regression for both 1D models.

Run: python hmc_beta_mu_1d_experiment.py --no-show
Results: figure/beta_vs_mu_1d_joint_hmc/ (keeps earlier MAP outputs intact).
Reuses the exact saved 91-training/90-test dataset from the MAP comparison.
If it is absent, regenerates the same seed=5, lambda=omega=1 dataset.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import special
from threadpoolctl import threadpool_limits

from fit_beta_mu_1d_experiment import simulate_1d, _json_default, _save_csv
from GH_proc_joint_hmc import sample_joint_parameters, predict_joint_mixture


def load_dataset():
    saved = Path('figure/beta_vs_mu_1d_map_predictions.npz')
    if saved.exists():
        metadata = json.loads(saved.with_name('beta_vs_mu_1d_map_report.json').read_text(encoding='utf-8'))
        if (metadata['seed'] != 5 or metadata['n_points'] != 181
                or metadata['l_fixed'] != 1 or metadata['omega_fixed'] != 1):
            raise ValueError('Saved MAP data must be the seed=5, 181-point, lambda=omega=1 experiment')
        data = dict(parameters=metadata['generating_parameters'])
        mapping = dict(x='x', y='y', f='true_f', beta='true_beta', W='true_W',
                       effect='true_effect', train_ids='train_ids', test_ids='test_ids')
        with np.load(saved) as previous:
            for key, stored in mapping.items():
                data[key] = previous[stored].copy()
        # Load saved values directly: SVD-based Gaussian simulation can change
        # with BLAS threading even at the same random seed.
        data['W'] = float(data['W'])
        data['source'] = str(saved)
    else:
        data = simulate_1d()
        data['source'] = 'simulate_1d(seed=5, n_points=181, l=1, omega=1)'
    return data


def score_model(data, prediction):
    test = data['test_ids']
    scores = {}
    for name, truth in [('f', data['f']), ('y', data['y']), ('effect', data['effect'])]:
        mean, ci = prediction[f'{name}_mean'][test], prediction[f'{name}_ci'][test]
        scores[f'{name}_rmse'] = float(np.sqrt(np.mean((mean-truth[test])**2)))
        scores[f'{name}_coverage_95'] = float(np.mean((truth[test] >= ci[:, 0]) & (truth[test] <= ci[:, 1])))
        scores[f'{name}_mean_interval_width'] = float(np.mean(ci[:, 1]-ci[:, 0]))
    means = prediction['component_y_means'][:, test]
    variances = prediction['component_y_variances'][:, test]
    logpdf = -0.5*(np.log(2*np.pi*variances)+(data['y'][test]-means)**2/variances)
    scores['y_mean_marginal_nlpd'] = float(-np.mean(special.logsumexp(logpdf, axis=0)-np.log(len(means))))
    return scores


def plot_results(data, predictions, fits, folder, show):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x, train, test = data['x'], data['train_ids'], data['test_ids']
    fig, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True, sharey='row', constrained_layout=True)
    for column, (model, p) in enumerate(predictions.items()):
        label = 'Beta GP' if model == 'beta_gp' else 'Mu GP (independent of W)'
        for row, name in enumerate(['effect', 'f', 'y']):
            ax = axes[row, column]
            truth = data['effect'] if name == 'effect' else data['f']
            ax.plot(x, truth, 'k--', lw=1.4, label='True W beta' if name == 'effect' else 'True latent f')
            ax.plot(x, p[f'{name}_mean'], color=f'C{column}', label='Posterior mean')
            ax.fill_between(x, *p[f'{name}_ci'].T, color=f'C{column}', alpha=0.22,
                            label='Pointwise 95% interval')
            ax.set_ylabel(dict(effect='h = W beta or mu', f='Latent f', y='Noisy y')[name])
            ax.grid(alpha=0.2)
            if name == 'f':
                ax.scatter(x[train], data['y'][train], s=10, c='gray', alpha=0.5, label='Training y')
            elif name == 'y':
                ax.scatter(x[test], data['y'][test], s=13, c='black', marker='x', label='Held-out y')
            ax.legend(fontsize=8, loc='best')
        axes[0, column].set_title(label)
        axes[2, column].set_xlabel('x')
    fig.suptitle('Joint posterior sampling of hyperparameters and W | fixed lambda = omega = 1\n'
                 'Bands average conditional Gaussians over all retained draws')
    fig.savefig(folder/'regression.png', dpi=160)
    if show:
        plt.show()
    plt.close(fig)

    fig, axes = plt.subplots(6, 2, figsize=(14, 13), constrained_layout=True)
    for col, (model, fit) in enumerate(fits.items()):
        for j, name in enumerate(fit['parameter_names']):
            ax = axes[j, col]
            for chain, values in enumerate(fit['log_samples']):
                ax.plot(values[:, j], lw=0.5, alpha=0.65, label=f'Chain {chain+1}')
            ax.set_ylabel(f'log {name}')
            ax.set_title(f'{model}: R-hat={fit["diagnostics"]["rhat"][j]:.3f}, '
                         f'bulk ESS={fit["diagnostics"]["ess_bulk"][j]:.0f}', fontsize=9)
        axes[-1, col].set_xlabel('Retained iteration (warmup excluded)')
        axes[0, col].legend(fontsize=7, ncol=4)
    fig.savefig(folder/'traces.png', dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for j, ax in enumerate(axes.flat):
        for color, (model, fit) in enumerate(fits.items()):
            values = fit['log_samples'][..., j].ravel()
            ax.hist(values, bins=55, density=True, histtype='step', lw=1.6, color=f'C{color}', label=model)
        ax.set_xlabel(['log v_0', 'log latent variance', 'log k', 'log w', 'log latent w', 'log W'][j])
        ax.set_ylabel('Posterior density in log coordinates')
        ax.legend(fontsize=8)
    fig.savefig(folder/'parameter_posteriors.png', dpi=160)
    plt.close(fig)


def run_experiment(*, output_dir='figure/beta_vs_mu_1d_joint_hmc', chains=4,
                   warmup=1000, draws=1500, sampler_seed=2026, prior_sd=2.,
                   target_accept=0.9, leapfrog_steps=(20, 50), show=False):
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    data = load_dataset()
    np.savez_compressed(folder/'dataset.npz', **{key: data[key] for key in
                        ['x', 'y', 'f', 'beta', 'effect', 'W', 'train_ids', 'test_ids']})
    train = data['train_ids']
    y, x = data['y'][train], data['x'][train]
    predictions, fits, scores, parameters, reports = {}, {}, [], [], {}
    print(f'Same 1D dataset: {len(train)} train, {len(data["test_ids"])} held out; '
          f'true W={data["W"]:.6f}', flush=True)
    for model_index, model in enumerate(['beta_gp', 'mu_gp']):
        started = time.perf_counter()
        print(f'Joint HMC: {model}', flush=True)
        fit = sample_joint_parameters(y, x, model=model, l=1., omega=1., prior_sd=prior_sd,
                                      chains=chains, warmup=warmup, draws=draws,
                                      seed=sampler_seed+model_index, target_accept=target_accept,
                                      leapfrog_steps=leapfrog_steps, verbose=True)
        sampling_seconds = time.perf_counter()-started
        np.savez_compressed(folder/f'{model}_samples.npz', **{key: value for key, value in fit.items()
                            if isinstance(value, np.ndarray)}, parameter_names=fit['parameter_names'])
        print(f'{model}: sampling done; averaging all {chains*draws} conditional Gaussians', flush=True)
        prediction = predict_joint_mixture(y, x, data['x'], fit)
        np.savez_compressed(folder/f'{model}_predictions.npz', **prediction)
        metrics = score_model(data, prediction)
        diagnostics = fit['diagnostics']
        metrics.update(max_rhat=float(np.max(diagnostics['rhat'])),
                       min_bulk_ess=float(np.min(diagnostics['ess_bulk'])),
                       min_tail_ess=float(np.min(diagnostics['ess_tail'])),
                       divergences=int(np.sum(fit['divergent'])),
                       acceptance=float(np.mean(fit['acceptance'])),
                       component_swap_acceptance=float(np.mean(fit['swap_accepted'])),
                       diagnostics_ok=fit['diagnostics_ok'])
        scores.append(dict(model=model, **metrics))
        parameters.extend(dict(model=model, **row) for row in fit['summary'])
        reports[model] = dict(prior_center=fit['prior_center'], prior_sd=fit['prior_sd'],
                              parameter_names=fit['parameter_names'], diagnostics=diagnostics,
                              effective_diagnostics=fit['effective_diagnostics'],
                              diagnostics_ok=fit['diagnostics_ok'], scores=metrics,
                              step_size=fit['step_size'], inverse_mass=fit['inverse_mass'],
                              warmup_divergences=int(np.sum(fit['warmup_divergent'])),
                              initialization=fit['initialization'], summary=fit['summary'],
                              sampling_seconds=sampling_seconds,
                              total_seconds=time.perf_counter()-started)
        fits[model], predictions[model] = fit, prediction
        print(f'{model}: f RMSE={metrics["f_rmse"]:.6f}; y RMSE={metrics["y_rmse"]:.6f}; '
              f'max R-hat={metrics["max_rhat"]:.4f}; min bulk ESS={metrics["min_bulk_ess"]:.0f}; '
              f'divergences={metrics["divergences"]}', flush=True)
        # Persist completed model summaries before starting the next model.
        _save_csv(folder/'scores.csv', scores)
        _save_csv(folder/'parameters.csv', parameters)
        report = dict(dataset_source=data['source'], seed=5, n_points=len(data['x']),
                      generating_parameters=data['parameters'], true_W=data['W'],
                      train_ids=data['train_ids'], test_ids=data['test_ids'],
                      lambda_fixed=1., omega_fixed=1., chains=chains, warmup=warmup, draws=draws,
                      sampler_seed=sampler_seed, target_accept=target_accept,
                      leapfrog_steps=leapfrog_steps,
                      prior_note='Independent untruncated Normal priors on log hyperparameters; '
                      'centers use training y variance, as in the MAP experiment. These empirical '
                      'prior settings are held fixed. mu-GP prior is independent of W.',
                      sampler_note='Metropolis-corrected HMC, not NUTS; beta/mu/f marginalized '
                      'analytically, W sampled jointly with theta. Log-W Jacobian included. '
                      'Unit-determinant effective log-amplitude coordinates. After each HMC step, '
                      'a swap of the two SE components is accepted using the full posterior ratio. Initialization '
                      'optimization does not fix any posterior parameters.',
                      prediction_note='All retained states have equal weights. Full covariance '
                      'is mean conditional covariance plus covariance of conditional means. '
                      '95% bands invert the marginal Gaussian-mixture CDF.', models=reports)
        (folder/'report.json').write_text(json.dumps(report, indent=2, default=_json_default), encoding='utf-8')
    plot_results(data, predictions, fits, folder, show)
    (folder/'README.md').write_text(
        '# Joint Bayesian 1D beta/mu comparison\n\n'
        'Run `python hmc_beta_mu_1d_experiment.py --no-show` from the repository root.\n\n'
        '- `regression.png`: separate panels for each model; posterior means and pointwise 95% bands.\n'
        '- `traces.png`, `parameter_posteriors.png`: chains and marginal parameter densities.\n'
        '- `scores.csv`: `f_rmse` uses true latent f; `y_rmse` uses noisy held-out y.\n'
        '- `parameters.csv`: posterior means, std, variance, quantiles, R-hat and ESS (including W).\n'
        '- `report.json`: complete settings, empirical prior choices and diagnostics.\n'
        '- `dataset.npz`: exact data, latent truth and split; truth is used only for evaluation.\n'
        '- `*_samples.npz`: all retained joint log/physical parameter samples and sampler diagnostics.\n'
        '- `*_predictions.npz`: means, full covariances, within/between covariance terms, marginal '
        'mixture components and credible intervals, averaging every retained sample.\n\n'
        'This compares the original beta GP with an independent mu GP, not the equivalent '
        'reparameterized mu|W GP. lambda=omega=1. Priors on log hyperparameters match the MAP '
        'centers/sd, but are untruncated. Parameter uncertainty is included conditional on these '
        'training-derived prior settings. Check report diagnostics before interpreting results.\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='figure/beta_vs_mu_1d_joint_hmc')
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=1000)
    parser.add_argument('--draws', type=int, default=1500)
    parser.add_argument('--sampler-seed', type=int, default=2026)
    parser.add_argument('--prior-sd', type=float, default=2.)
    parser.add_argument('--target-accept', type=float, default=0.9)
    parser.add_argument('--leapfrog-min', type=int, default=20)
    parser.add_argument('--leapfrog-max', type=int, default=50)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    # Small dense matrices are much faster with one BLAS thread; context restores
    # the caller's thread settings and avoids changing any global environment.
    with threadpool_limits(limits=1):
        run_experiment(output_dir=args.output_dir, chains=args.chains, warmup=args.warmup,
                       draws=args.draws, sampler_seed=args.sampler_seed, prior_sd=args.prior_sd,
                       target_accept=args.target_accept,
                       leapfrog_steps=(args.leapfrog_min, args.leapfrog_max), show=not args.no_show)
