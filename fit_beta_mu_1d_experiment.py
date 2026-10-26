"""1D comparison: learn both SE kernels and noise by MAP, then regress.

Run: python fit_beta_mu_1d_experiment.py --no-show

Generate mu=0, beta~GP(0,Sigma_beta), f=W*beta+g, g|W~GP(0,W*K),
y=f+noise with noise variance W*k. Fix lambda=omega=1 by default.
Use the same training observations to fit these two models independently:
  beta_gp: mu=0, learn K, Sigma_beta and k;
  mu_gp: beta=0, mu~GP(0,Sigma_mu) independent of W, learn K, Sigma_mu and k.

Each fit has five positive hyperparameters in 1D. Both use the same empirical
Gaussian priors on log parameters, starting points, bounds, and optimizer.
W and the latent functions are marginalized in the likelihood; plug MAP
parameters into regression afterwards. No hyperparameter uncertainty in bands.
No preprocessing: original x/y units and the known zero prior means are kept.
The true parameters and latent draws are only used to generate/evaluate data.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from GH_proc_w_betaprior import Multivariate_GH_1d, beta_f_inf
from GH_proc_w_betaprior_fit import fit_kernel_parameters, score_predictions
from GH_proc_mu_prior import beta_effect_posterior, mu_f_inf
from GH_proc_mu_prior_fit import fit_mu_kernel_parameters


def simulate_1d(seed=5, n_points=181, l=1., omega=1.):
    """Same 1D grid/generating kernels as the original beta-prior demo."""
    if not isinstance(n_points, (int, np.integer)) or n_points < 9:
        raise ValueError('n_points must be an integer >= 9')
    x = np.linspace(0., 10., n_points)
    parameters = dict(l=l, omega=omega, mu=0., v_0=0.2, w=2., k=0.05,
                      sigma_v0=1., sigma_w=0.15)
    f, y, beta, W = Multivariate_GH_1d(
        x, **parameters, beta_prior=True, return_latents=True, seed=seed)
    return dict(x=x, y=y, f=f, beta=beta, W=W, effect=W*beta, parameters=parameters,
                train_ids=np.arange(0, n_points, 2), test_ids=np.arange(1, n_points, 2))


def fit_1d_models(y_train, x_train, *, l=1., omega=1., seed=5,
                   n_restarts=3, maxiter=200, log_prior_sd=2., verbose=True):
    """Training-only MAP stage; this function never receives held-out labels.

    Both covariances are parameterized by their variance amplitude and inverse
    squared length scale. The latent covariance is optimized in BOTH models.
    All five positive parameters have independent log-Normal priors; MAP is
    found in log coordinates. Prior centers use only training-response variance.
    """
    x = np.asarray(x_train, dtype=float)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    if x.ndim != 1:
        raise ValueError('This experiment fits 1D inputs only')
    if log_prior_sd is None:
        raise ValueError('This comparison requires MAP with a proper hyperparameter prior')
    common = dict(l=l, omega=omega, seed=seed, n_restarts=n_restarts, maxiter=maxiter,
                  log_prior_sd=log_prior_sd, verbose=verbose)
    if verbose:
        print('Fitting beta-GP model: K, Sigma_beta and k', flush=True)
    beta_fit = fit_kernel_parameters(y_train, x, mu=0., fit_mu=False, **common)
    if verbose:
        print('Fitting mu-GP model: K, Sigma_mu and k', flush=True)
    mu_fit = fit_mu_kernel_parameters(y_train, x, mu0=0., fit_mu0=False, **common)
    return dict(beta_gp=beta_fit, mu_gp=mu_fit)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def _save_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _plots(data, predictions, prefix, show):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x, train, test = data['x'], data['train_ids'], data['test_ids']
    fig, axes = plt.subplots(3, 1, figsize=(11, 11), sharex=True, constrained_layout=True)
    labels = dict(beta_gp='MAP beta-GP model', mu_gp='MAP mu-GP model')
    for ax, name, truth in [(axes[0], 'effect', data['effect']), (axes[1], 'f', data['f']),
                            (axes[2], 'y', data['f'])]:
        ax.plot(x, truth, 'k--', lw=1.7, label='True W beta' if name == 'effect' else 'True latent f')
        for color, key in enumerate(labels):
            p = predictions[key]
            ax.plot(x, p[f'{name}_mean'], color=f'C{color}', label=labels[key])
            ax.fill_between(x, p[f'{name}_ci'][:, 0], p[f'{name}_ci'][:, 1], color=f'C{color}', alpha=0.17)
        ax.set_ylabel({'effect': 'Mean contribution h', 'f': 'Latent f', 'y': 'New observations y'}[name])
    axes[1].scatter(x[train], data['y'][train], color='gray', alpha=0.6, s=11, label='Training y')
    axes[2].scatter(x[test], data['y'][test], color='black', marker='x', s=14, label='Held-out y')
    axes[0].set_title('h = W beta in the beta model; h = mu in the mu model')
    axes[1].set_title('Latent-function regression')
    axes[2].set_title('Noisy predictive intervals and held-out observations')
    for ax in axes:
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=0.15)
    axes[-1].set_xlabel('x')
    fig.suptitle(f'1D MAP then regression; fixed lambda={data["parameters"]["l"]:g}, '
                 f'omega={data["parameters"]["omega"]:g}; true W={data["W"]:.3f}\n'
                 'Shading: pointwise 95% intervals conditional on fitted hyperparameters')
    fig.savefig(f'{prefix}_regression.png', dpi=160)
    if show:
        plt.show()
    plt.close(fig)


def run_experiment(*, seed=5, n_points=181, l=1., omega=1., n_restarts=3,
                   maxiter=200, log_prior_sd=2.,
                   save_prefix='figure/beta_vs_mu_1d_map', show=False):
    data = simulate_1d(seed=seed, n_points=n_points, l=l, omega=omega)
    train, test, x, y = data['train_ids'], data['test_ids'], data['x'], data['y']
    print(f'1D beta-GP simulation: {len(train)} train / {len(test)} held out; true W={data["W"]:.6f}', flush=True)
    fits = fit_1d_models(y[train], x[train], l=l, omega=omega, seed=seed,
                         n_restarts=n_restarts, maxiter=maxiter, log_prior_sd=log_prior_sd)
    predictions, scores, parameters, diagnostics = {}, [], [], {}
    for name, fit in fits.items():
        if name == 'beta_gp':
            p = beta_f_inf(y[train], x[train], x, **fit['params'], **fit['quad_options'])
            effect = beta_effect_posterior(p)
            latent, variance_key, weight_key = 'beta', 'sigma_v0', 'sigma_w'
        else:
            p = mu_f_inf(y[train], x[train], x, **fit['params'], **fit['quad_options'])
            effect = {stat: p[f'mu_{stat}'] for stat in ('mean', 'var', 'std', 'ci')}
            latent, variance_key, weight_key = 'mu', 'mu_v0', 'mu_w'
        m = len(x)
        subset = dict(weights=p['weights'], y_mean=p['y_mean'][test], y_ci=p['y_ci'][test],
                      component_means=p['component_means'][:, np.r_[test, m+test]],
                      component_y_variances=p['component_y_variances'][:, test])
        metrics = score_predictions(y[test], subset)
        for target, truth, mean, interval in [('f', data['f'], p['f_mean'], p['f_ci']),
                                              ('effect', data['effect'], effect['mean'], effect['ci'])]:
            metrics[f'{target}_rmse'] = float(np.sqrt(np.mean((mean[test]-truth[test])**2)))
            metrics[f'{target}_coverage'] = float(np.mean((truth[test] >= interval[test, 0]) & (truth[test] <= interval[test, 1])))
        W_mean = float(p['weights'] @ p['W'])
        W_var = float(p['weights'] @ (p['W']-W_mean)**2)
        scores.append(dict(model=name, converged=fit['converged'], **metrics,
                           W_mean=W_mean, W_var=W_var, W_std=float(np.sqrt(W_var)),
                           training_log_marginal_likelihood=fit['log_marginal_likelihood'],
                           negative_log_posterior_up_to_constant=fit['objective']))
        theta = fit['params']
        weight, latent_weight = float(np.asarray(theta['w']).item()), float(np.asarray(theta[weight_key]).item())
        parameters.append(dict(model=name, l_fixed=l, omega_fixed=omega, GP_prior_mean_fixed=0.,
                                v_0=theta['v_0'], w=weight, residual_length_scale=1/np.sqrt(weight),
                                k=theta['k'], latent_GP=latent, latent_variance=theta[variance_key],
                                latent_weight=latent_weight, latent_length_scale=1/np.sqrt(latent_weight),
                                W_mean=W_mean, W_std=np.sqrt(W_var)))
        predictions[name] = {f'{target}_{stat}': p[f'{target}_{stat}']
                             for target in (latent, 'f', 'y') for stat in ('mean', 'var', 'std', 'ci')}
        predictions[name].update({f'effect_{stat}': effect[stat] for stat in ('mean', 'var', 'std', 'ci')})
        predictions[name].update(W=p['W'], weights=p['weights'], component_means=p['component_means'],
                                  component_y_variances=p['component_y_variances'])
        diagnostics[name] = p['quadrature']
        print(f'{name}: held-out y RMSE={metrics["rmse"]:.6f}, NLPD={metrics["mean_marginal_nlpd"]:.6f}, '
              f'f RMSE={metrics["f_rmse"]:.6f}, h RMSE={metrics["effect_rmse"]:.6f}', flush=True)
        print('  Fitted parameters:', theta, flush=True)
        del p, effect
    prefix = Path(save_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    _save_csv(Path(f'{prefix}_scores.csv'), scores)
    _save_csv(Path(f'{prefix}_parameters.csv'), parameters)
    report = dict(seed=seed, n_points=n_points, l_fixed=l, omega_fixed=omega,
                  generating_parameters=data['parameters'], true_W=data['W'],
                  preprocessing='none; original units; zero prior means fixed',
                  train_ids=train, test_ids=test, log_prior_sd=log_prior_sd,
                  n_restarts=n_restarts, maxiter=maxiter, fits=fits, scores=scores,
                  parameters=parameters, regression_quadrature=diagnostics)
    Path(f'{prefix}_report.json').write_text(json.dumps(report, indent=2, default=_json_default))
    arrays = {f'{model}_{key}': value for model, prediction in predictions.items()
              for key, value in prediction.items()}
    np.savez_compressed(f'{prefix}_predictions.npz', x=x, y=y, true_f=data['f'],
                        true_beta=data['beta'], true_W=data['W'], true_effect=data['effect'],
                        train_ids=train, test_ids=test, **arrays)
    _plots(data, predictions, prefix, show)
    print(f'Saved {prefix}_*', flush=True)
    return dict(data=data, fits=fits, predictions=predictions, scores=scores, parameters=parameters)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--n-points', type=int, default=181)
    parser.add_argument('--l', type=float, default=1.)
    parser.add_argument('--omega', type=float, default=1.)
    parser.add_argument('--n-restarts', type=int, default=3)
    parser.add_argument('--maxiter', type=int, default=200)
    parser.add_argument('--log-prior-sd', type=float, default=2.)
    parser.add_argument('--save-prefix', default='figure/beta_vs_mu_1d_map')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    run_experiment(seed=args.seed, n_points=args.n_points, l=args.l, omega=args.omega,
                    n_restarts=args.n_restarts, maxiter=args.maxiter, log_prior_sd=args.log_prior_sd,
                    save_prefix=args.save_prefix, show=not args.no_show)
