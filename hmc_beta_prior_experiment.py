"""Simulated 2D GH data: infer both SE kernels and noise using HMC.

Run: python hmc_beta_prior_experiment.py --no-show
Output: figure/betaprior_hmc_2d_{summary.csv,fit.json,samples.npz,...png}.
Uses native x/y units and known mu=0; the true hyperparameters are used only
to simulate and evaluate. One W is shared by the whole simulated dataset.
"""

import argparse
import csv
import json
from pathlib import Path
import time

import numpy as np

from GH_proc_w_betaprior import Multivariate_GH, beta_f_inf
from GH_proc_w_betaprior_fit import score_predictions
from GH_proc_w_betaprior_hmc import sample_kernel_parameters, predict_hyperparameter_mixture


def simulate_2d(n_samples=100, train_fraction=0.8, seed=5):
    """Jointly simulate train/test latents, then split; return truth for checking."""
    if not isinstance(n_samples, (int, np.integer)) or n_samples < 12:
        raise ValueError('n_samples must be an integer >= 12')
    if not 0 < train_fraction < 1:
        raise ValueError('train_fraction must lie between zero and one')
    input_seed, latent_seed, split_seed = np.random.SeedSequence(seed).spawn(3)
    rng = np.random.default_rng(input_seed)
    xs = rng.uniform(-3, 3, size=(n_samples, 2))
    truth = dict(l=-0.5, omega=1., w=np.array([2., 1.2]), v_0=0.2, k=0.05,
                 mu=0., sigma_v0=1., sigma_w=np.array([0.25, 0.6]))
    f, ys, beta, W = Multivariate_GH(xs, **truth, seed=latent_seed, return_latents=True)
    order = np.random.default_rng(split_seed).permutation(n_samples)
    count = max(2, min(n_samples-1, int(train_fraction*n_samples)))
    return dict(xs=xs, ys=ys, f=f, beta=beta, W=W, truth=truth,
                train=order[:count], test=order[count:])


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n-samples', type=int, default=100)
    parser.add_argument('--seed', type=int, default=5, help='Simulation/split seed')
    parser.add_argument('--sampler-seed', type=int, default=2026)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=500)
    parser.add_argument('--draws', type=int, default=800, help='Retained draws per chain')
    parser.add_argument('--log-prior-sd', type=float, default=1.5)
    parser.add_argument('--target-accept', type=float, default=0.85)
    parser.add_argument('--prediction-draws', type=int, default=64)
    parser.add_argument('--output-prefix', default='figure/betaprior_hmc_2d')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    import matplotlib
    if args.no_show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = simulate_2d(args.n_samples, seed=args.seed)
    train, test = data['train'], data['test']
    xs, ys = data['xs'][train], data['ys'][train]
    prefix = Path(args.output_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    print(f'2D simulation: {len(train)} training, {len(test)} held out, true W={data["W"]:.4f}', flush=True)
    # Explicit, broad prior centers; different from the generating parameters.
    # Fix these before fitting. sd=1.5 is approximately a factor 19 each way
    # for a central 95% prior interval on a positive parameter.
    centers = dict(v_0=0.5, sigma_v0=0.5, k=0.1, w=[1., 1.], sigma_w=[0.25, 0.25])
    started = time.perf_counter()
    fit = sample_kernel_parameters(
        ys, xs, l=-0.5, omega=1., mu=0., prior_centers=centers,
        log_prior_sd=args.log_prior_sd, chains=args.chains, warmup=args.warmup,
        draws=args.draws, seed=args.sampler_seed, target_accept=args.target_accept, verbose=True)
    sampling_seconds = time.perf_counter() - started
    truth = data['truth']
    true_vector = np.r_[truth['v_0'], truth['sigma_v0'], truth['k'], truth['w'], truth['sigma_w']]
    summary = [dict(row, truth=float(value)) for row, value in zip(fit['summary'], true_vector)]
    with Path(f'{prefix}_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary[0]))
        writer.writeheader()
        writer.writerows(summary)
    np.savez_compressed(
        f'{prefix}_samples.npz', log_samples=fit['log_samples'], parameter_samples=fit['parameter_samples'],
        warmup_log_samples=fit['warmup_samples'], parameter_names=fit['parameter_names'],
        starts=fit['starts'], log_density=fit['log_density'], acceptance=fit['acceptance'],
        acceptance_probability=fit['acceptance_probability'], energy_error=fit['energy_error'],
        divergent=fit['divergent'], warmup_divergent=fit['warmup_divergent'],
        step_size=fit['step_size'], inverse_mass=fit['inverse_mass'],
        xs=data['xs'], ys=data['ys'], true_f=data['f'], true_beta=data['beta'],
        true_W=data['W'], true_parameters=true_vector, train=train, test=test)
    print('\nparameter       true     median         95% interval       R-hat  bulk ESS', flush=True)
    for row in summary:
        print(f'{row["parameter"]:13s} {row["truth"]:7.3f} {row["median"]:9.3f} '
              f'[{row["lower"]:8.3f}, {row["upper"]:8.3f}] '
              f'{row["rhat"]:8.4f} {row["ess_bulk"]:9.0f}', flush=True)
    acceptance = fit['acceptance'].mean(axis=1)
    divergences = fit['divergent'].sum(axis=1)
    print(f'Acceptance by chain: {acceptance}; retained divergences: {divergences}', flush=True)
    print(f'Computing predictions using {args.prediction_draws} hyperparameter draws...', flush=True)
    prediction = predict_hyperparameter_mixture(
        ys, xs, data['xs'][test], fit, max_draws=args.prediction_draws)
    map_prediction = beta_f_inf(ys, xs, data['xs'][test], **fit['map_fit']['params'])
    oracle_prediction = beta_f_inf(ys, xs, data['xs'][test], **truth)
    metrics = {name: score_predictions(data['ys'][test], posterior)
               for name, posterior in [('hmc', prediction), ('map', map_prediction), ('known_parameters', oracle_prediction)]}
    for name, latent in [('beta', data['beta'][test]), ('f', data['f'][test])]:
        interval = prediction[f'{name}_ci']
        metrics['hmc'][f'{name}_rmse'] = float(np.sqrt(np.mean((latent-prediction[f'{name}_mean'])**2)))
        metrics['hmc'][f'{name}_coverage'] = float(np.mean((latent >= interval[:, 0]) & (latent <= interval[:, 1])))
    np.savez_compressed(f'{prefix}_predictions.npz', x_test=data['xs'][test],
                        y_test=data['ys'][test], true_beta=data['beta'][test], true_f=data['f'][test],
                        **{key: value for key, value in prediction.items()
                           if key.endswith(('_mean', '_var', '_std', '_ci'))})
    report = dict(arguments=vars(args), true_parameters=truth, true_W=data['W'], summary=summary,
                  map_fit=fit['map_fit'], log_bounds=fit['log_bounds'],
                  acceptance=acceptance, divergences=divergences,
                  warmup_divergences=fit['warmup_divergent'].sum(axis=1),
                  sampling_seconds=sampling_seconds, metrics=metrics,
                  prediction_hyperparameter_draws=prediction['hyperparameter_draws'])
    Path(f'{prefix}_fit.json').write_text(json.dumps(report, indent=2, default=_json_default))
    print('Held-out prediction:', json.dumps(metrics, indent=2), flush=True)

    names = fit['parameter_names']
    fig, axes = plt.subplots(len(names), 2, figsize=(13, 2.2*len(names)), squeeze=False)
    for d, name in enumerate(names):
        for chain in range(args.chains):
            warm = fit['warmup_samples'][chain, :, d]
            retained = fit['log_samples'][chain, :, d]
            color = f'C{chain % 10}'
            axes[d, 0].plot(np.arange(-args.warmup, 0), warm, color=color, alpha=0.25, lw=0.5)
            axes[d, 0].plot(retained, color=color, alpha=0.7, lw=0.6)
            axes[d, 1].hist(retained, bins=35, density=True, histtype='step', color=color,
                             label=f'chain {chain+1}')
        axes[d, 0].axvline(0, color='gray', ls=':', lw=1)
        axes[d, 0].axhline(np.log(true_vector[d]), color='black', ls='--', label='truth')
        axes[d, 1].axvline(np.log(true_vector[d]), color='black', ls='--')
        axes[d, 0].set_ylabel(f'log {name}')
        axes[d, 1].set_title(f'R-hat={summary[d]["rhat"]:.3f}; bulk ESS={summary[d]["ess_bulk"]:.0f}', fontsize=10)
    axes[0, 1].legend(fontsize=8)
    axes[-1, 0].set_xlabel('Iteration (negative = discarded warm-up)')
    axes[-1, 1].set_xlabel('Log parameter')
    fig.suptitle('HMC: SE kernel and noise parameters, fixed GIG(-0.5, 1)')
    fig.tight_layout()
    fig.savefig(f'{prefix}_traces.png', dpi=160)

    fig, ax = plt.subplots(figsize=(9, 5))
    for d, row in enumerate(summary):
        # Normalize by truth so all seven parameters can share one log axis.
        ax.errorbar(row['median']/true_vector[d], d,
                    xerr=[[ (row['median']-row['lower'])/true_vector[d] ],
                          [ (row['upper']-row['median'])/true_vector[d] ]], fmt='o', color='C0', capsize=4)
    ax.axvline(1, color='black', ls='--', label='Generating value')
    ax.set(xscale='log', yticks=np.arange(len(names)), yticklabels=names,
           xlabel='Parameter / generating value (median and 95% posterior interval)',
           title='Recovery from one 2D realization')
    ax.legend()
    fig.tight_layout()
    fig.savefig(f'{prefix}_parameters.png', dpi=160)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, name, latent in zip(axes, ['beta', 'f', 'y'],
                                 [data['beta'][test], data['f'][test], data['ys'][test]]):
        mean, interval = prediction[f'{name}_mean'], prediction[f'{name}_ci']
        ax.vlines(latent, interval[:, 0], interval[:, 1], color='C0', alpha=0.5)
        ax.scatter(latent, mean, color='C0', s=18)
        low, high = min(latent.min(), interval.min()), max(latent.max(), interval.max())
        ax.plot([low, high], [low, high], 'k--', lw=1)
        ax.set(xlabel=f'True held-out {name}', ylabel='Posterior mean and 95% interval', title=name)
    fig.suptitle('Predictions integrate W and sampled kernel/noise parameters')
    fig.tight_layout()
    fig.savefig(f'{prefix}_predictions.png', dpi=160)
    print(f'Saved {prefix}_*. Sampling and MAP took {sampling_seconds:.1f}s.', flush=True)
    if not args.no_show:
        plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()
