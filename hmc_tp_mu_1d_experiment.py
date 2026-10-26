"""Student-t residual plus independent GP mean on the saved 1D dataset.

Run: python hmc_tp_mu_1d_experiment.py --no-show
mu~GP(0,K2), f|mu,W~GP(mu,W*K1), y|f,W~N(f,W*kI).
W~InvGamma(nu/2,(nu-2)/2), with fixed nu=5 by default.
The marginal f is a convolution of a GP and TP, not a standard TP.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

from GH_proc_joint_hmc import sample_joint_parameters, predict_joint_mixture
from fit_beta_mu_1d_experiment import _json_default, _save_csv
from hmc_beta_mu_1d_experiment import load_dataset
from hmc_gp_tp_1d_experiment import (load_previous_results, score_predictions,
                                    diagnostics_summary, plot_results, save_latex_table)


def load_saved_gp(data, folder):
    """Reuse the unchanged GP baseline only after checking its exact dataset."""
    folder = Path(folder)
    with np.load(folder/'dataset.npz') as saved:
        for key in ['x', 'y', 'f', 'train_ids', 'test_ids']:
            if not np.array_equal(saved[key], data[key]):
                raise ValueError(f'Saved GP dataset mismatch: {key}')
    report = json.loads((folder/'report.json').read_text(encoding='utf-8'))['models']['gp']
    with np.load(folder/'gp_samples.npz') as saved:
        fit = {key: saved[key].copy() for key in saved.files}
    fit.update(diagnostics=report['diagnostics'], diagnostics_ok=report['diagnostics_ok'],
               summary=report['summary'])
    with np.load(folder/'gp_predictions.npz') as saved:
        prediction = {key: saved[key].copy() for key in saved.files}
    return fit, prediction, report


def plot_mu(data, prediction, folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4), constrained_layout=True)
    ax.plot(data['x'], prediction['mu_mean'], label='Posterior mean of mu')
    ax.fill_between(data['x'], *prediction['mu_ci'].T, alpha=.2, label='Pointwise 95% interval')
    ax.plot(data['x'], data['effect'], 'k--', label='Generating effect W * beta (reference)')
    ax.set(xlabel='x', ylabel='Additive GP mean', title='Student-t residual + independent GP mean')
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(folder/'mu_posterior.png', dpi=160)
    plt.close(fig)


def run_experiment(*, output_dir='figure/tp_mu_1d_joint_hmc',
                   gp_source='figure/gp_tp_1d_hmc', nu=5., chains=4,
                   warmup=1000, draws=1500, seed=4026, prior_sd=2.,
                   target_accept=.9, leapfrog_steps=(20, 50), show=False):
    data = load_dataset()
    gp_fit, gp_prediction, gp_report = load_saved_gp(data, gp_source)
    scores, predictions, previous_source = load_previous_results(data)
    scores.append(dict(model='gp', **score_predictions(data, gp_prediction), **diagnostics_summary(gp_fit)))
    predictions['gp'] = gp_prediction
    folder = Path(output_dir)
    if folder.resolve() in {Path(gp_source).resolve(), Path(previous_source).resolve() if previous_source else None}:
        raise ValueError('Use a separate output folder for the new model')
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(folder/'dataset.npz', **{key: data[key] for key in
                        ['x', 'y', 'f', 'beta', 'effect', 'W', 'train_ids', 'test_ids']})
    train = data['train_ids']
    y, x = data['y'][train], data['x'][train]
    print(f'Same saved 1D dataset: {len(train)} train / {len(data["test_ids"])} held out; nu={nu:g}', flush=True)
    started = time.perf_counter()
    fit = sample_joint_parameters(y, x, model='tp_mu_gp', nu=nu, prior_sd=prior_sd,
                                  chains=chains, warmup=warmup, draws=draws, seed=seed,
                                  target_accept=target_accept, leapfrog_steps=leapfrog_steps, verbose=True)
    sampling_seconds = time.perf_counter()-started
    np.savez_compressed(folder/'tp_mu_gp_samples.npz', parameter_names=fit['parameter_names'],
                        **{key: value for key, value in fit.items() if isinstance(value, np.ndarray)})
    print(f'Averaging {chains*draws} conditional Gaussian predictions for f, mu and y', flush=True)
    prediction = predict_joint_mixture(y, x, data['x'], fit)
    np.savez_compressed(folder/'tp_mu_gp_predictions.npz', **prediction)
    metrics = dict(model='tp_mu_gp', **score_predictions(data, prediction, previous=True),
                   **diagnostics_summary(fit))
    scores.append(metrics)
    predictions['tp_mu_gp'] = prediction
    _save_csv(folder/'scores.csv', scores)
    _save_csv(folder/'parameters.csv', [dict(model='tp_mu_gp', **row) for row in fit['summary']])
    _save_csv(folder/'W_summary.csv', [dict(model='tp_mu_gp', **row) for row in fit['summary'] if row['parameter'] == 'W'])
    model_report = {key: fit[key] for key in ['prior_center', 'prior_sd', 'parameter_names',
                    'diagnostics', 'effective_diagnostics', 'diagnostics_ok', 'summary',
                    'initialization', 'step_size', 'inverse_mass']}
    model_report.update(scores=metrics, sampling_seconds=sampling_seconds,
                        warmup_divergences=int(np.sum(fit['warmup_divergent'])))
    report = dict(dataset_source=data['source'], generating_parameters=data['parameters'],
                  generating_W=data['W'], train_ids=train, test_ids=data['test_ids'],
                  previous_models_source=previous_source, gp_source=str(gp_source),
                  model='mu~GP(0,K2) independent of W; f|mu,W~GP(mu,W*K1); y|f,W~N(f,W*kI)',
                  covariance_f_given_W='K2+W*K1', covariance_y_given_W='K2+W*(K1+kI)',
                  fixed_nu=nu, W_prior=dict(distribution='inverse-gamma', shape=nu/2, scale=(nu-2)/2),
                  prior_note='Independent Normal priors on log(v_0,mu_v0,k,w,mu_w), sd=prior_sd. '
                  'Centers use only training y variance, matching the earlier GHP fits.',
                  sampler_note='Joint HMC of theta and W in effective log coordinates, '
                  'including the log-W Jacobian. Latent functions marginalized analytically. '
                  'Posterior-corrected kernel exchanges; optimization only initializes chains.',
                  prediction_note='All retained draws equally weighted. Gaussian mixture moments '
                  'include within-conditional and between-draw covariance. Intervals are pointwise '
                  'equal-tail mixture quantiles. This GP+TP sum is not a standard Student-t process.',
                  chains=chains, warmup=warmup, draws=draws, seed=seed, prior_sd=prior_sd,
                  target_accept=target_accept, leapfrog_steps=leapfrog_steps,
                  models=dict(gp=gp_report, tp_mu_gp=model_report))
    (folder/'report.json').write_text(json.dumps(report, indent=2, default=_json_default), encoding='utf-8')
    plot_results(data, predictions, dict(gp=gp_fit, tp_mu_gp=fit), scores, folder, nu, show, tp_model='tp_mu_gp')
    plot_mu(data, prediction, folder)
    save_latex_table(folder, scores)
    (folder/'README.md').write_text(
        '# Student-t residual + independent GP mean\n\n'
        'Run `python hmc_tp_mu_1d_experiment.py --no-show` from the repository root. '
        'Requires the saved GP baseline in `figure/gp_tp_1d_hmc/` (override with `--gp-source`).\n\n'
        'mu ~ GP(0,K2), independently of W; f | mu,W ~ GP(mu,W*K1); '
        'y | f,W ~ N(f,W*k*I). Thus f | W has covariance K2+W*K1. '
        f'W ~ InvGamma({nu/2:g}, scale={((nu-2)/2):g}), fixed nu={nu:g}. '
        'The residual is Student-t, while the sum with mu is not a standard TP. '
        'Both SE variance amplitudes, both inverse squared length scales, noise k '
        'and W are sampled jointly. Priors and parameter naming match the GHP mean model.\n\n'
        'Uses exactly the saved 91/90 split. GP and GIG beta/mu results are reused '
        'after checking identical arrays. Old baseline results stay in their original folders.\n\n'
        '- regression.png: f and noisy-y predictions for GP and the new model.\n'
        '- model_comparison.png, scores.csv, comparison_table.tex: comparison with GHP models.\n'
        '- mu_posterior.png: inferred additive mean; generating W*beta is only a reference.\n'
        '- traces.png, parameter_posteriors.png: retained chain diagnostics and distributions.\n'
        '- parameters.csv, W_summary.csv, report.json: moments, settings and diagnostics.\n'
        '- tp_mu_gp_samples.npz: all physical/effective log states and sampler outputs.\n'
        '- tp_mu_gp_predictions.npz: f, mu, effect (=mu) and y Gaussian-mixture '
        'components, weights, full covariance and pointwise 95% intervals.\n\n'
        'f RMSE/NLPD use latent truth; y RMSE/NLPD use held-out noisy observations. '
        'Predictive means coincide but variances differ by E[W*k | y]. '
        'W summaries are Monte Carlo estimates from retained HMC draws.\n', encoding='utf-8')
    print(json.dumps(metrics, indent=2), flush=True)
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='figure/tp_mu_1d_joint_hmc')
    parser.add_argument('--gp-source', default='figure/gp_tp_1d_hmc')
    parser.add_argument('--nu', type=float, default=5.)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=1000)
    parser.add_argument('--draws', type=int, default=1500)
    parser.add_argument('--sampler-seed', type=int, default=4026)
    parser.add_argument('--prior-sd', type=float, default=2.)
    parser.add_argument('--target-accept', type=float, default=.9)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    if not np.isfinite(args.nu) or args.nu <= 2:
        parser.error('--nu must be finite and greater than two')
    with threadpool_limits(limits=1):
        run_experiment(output_dir=args.output_dir, gp_source=args.gp_source, nu=args.nu,
                       chains=args.chains, warmup=args.warmup, draws=args.draws,
                       seed=args.sampler_seed, prior_sd=args.prior_sd,
                       target_accept=args.target_accept, show=not args.no_show)
