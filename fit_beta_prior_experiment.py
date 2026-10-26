"""Fit fixed-GIG beta-prior models and compare omega on one validation split.

Examples:
    python fit_beta_prior_experiment.py --no-show
    python fit_beta_prior_experiment.py --dataset synthetic --no-show
    python fit_beta_prior_experiment.py --n-samples 400 --omegas 0.5 1 2 5

The wine default uses the local white-wine CSV also referenced by wine.py.
Targets are treated as continuous. Scaling is learned on the training split
only. Outputs report predictions in original target units; beta is rescaled
without adding the target mean. Hyperparameter uncertainty is not included.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from GH_proc_w_betaprior import Multivariate_GH
from GH_proc_w_betaprior_fit import compare_fixed_omega


def standardize_split(xs, ys, train_ids, valid_ids):
    """Use training statistics only; never inspect validation labels for scaling."""
    x_mean = xs[train_ids].mean(axis=0)
    x_scale = xs[train_ids].std(axis=0)
    x_scale = np.where(x_scale > 0, x_scale, 1.)
    y_mean, y_scale = float(ys[train_ids].mean()), float(ys[train_ids].std())
    y_scale = y_scale if y_scale > 0 else 1.
    scaler = dict(x_mean=x_mean, x_scale=x_scale, y_mean=y_mean, y_scale=y_scale)
    return ((xs[train_ids]-x_mean)/x_scale, (ys[train_ids]-y_mean)/y_scale,
            (xs[valid_ids]-x_mean)/x_scale, (ys[valid_ids]-y_mean)/y_scale, scaler)


def _json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def run_experiment(dataset='wine', csv_path=None, n_samples=240, train_fraction=0.8,
                    omegas=(0.5, 1, 2, 5), seed=5, n_restarts=3, maxiter=200,
                    save_prefix=None, show=True):
    if n_samples < 10 or not 0 < train_fraction < 1:
        raise ValueError("Use n_samples >= 10 and 0 < train_fraction < 1")
    rng = np.random.default_rng(seed)
    truth = None
    if dataset == 'wine':
        csv_path = Path(csv_path) if csv_path else Path(__file__).resolve().parent.parent / 'data/winequality/winequality-white.csv'
        with csv_path.open(newline='') as stream:
            reader = csv.reader(stream, delimiter=';')
            header = next(reader)
            data = np.array([[float(value) for value in row] for row in reader])
        target = header.index('quality')
        if n_samples > len(data):
            raise ValueError("n_samples exceeds the dataset size")
        source_ids = rng.choice(len(data), n_samples, replace=False)
        ys = data[source_ids, target]
        xs = np.delete(data[source_ids], target, axis=1)
    elif dataset == 'synthetic':
        xs = rng.uniform(-3, 3, size=(n_samples, 2))
        fs, ys, beta, W = Multivariate_GH(xs, l=-0.5, omega=1, w=[2., 1.2],
                                         v_0=0.2, sigma_w=[0.25, 0.6], sigma_v0=1,
                                         k=0.05, mu=0, return_latents=True, seed=seed)
        truth = dict(fs=fs, beta=beta, W=W)
        source_ids = np.arange(n_samples)
    else:
        raise ValueError("dataset must be 'wine' or 'synthetic'")
    if not np.all(np.isfinite(xs)) or not np.all(np.isfinite(ys)):
        raise ValueError("Dataset contains nonfinite values")
    n_train = int(train_fraction*n_samples)
    if not 2 <= n_train < n_samples:
        raise ValueError("The split must contain at least two training points and one validation point")
    order = rng.permutation(n_samples)
    train_ids, valid_ids = order[:n_train], order[n_train:]
    xt, yt, xv, yv, scaler = standardize_split(xs, ys, train_ids, valid_ids)
    print(f"Dataset={dataset}: {len(train_ids)} training, {len(valid_ids)} validation, D={xs.shape[1]}", flush=True)
    comparison = compare_fixed_omega(yt, xt, yv, xv, omegas=omegas, l=-0.5,
                                     fit_options=dict(n_restarts=n_restarts, maxiter=maxiter, seed=seed))
    scale, offset = scaler['y_scale'], scaler['y_mean']
    rows, predictions, fits = [], {}, []
    for record in comparison['records']:
        metrics, fit, p = record['metrics'], record['fit'], record['posterior']
        row = dict(omega=record['omega'], converged=fit['converged'],
                   training_log_likelihood_standardized=fit['log_marginal_likelihood'],
                   validation_rmse=scale*metrics['rmse'], validation_mae=scale*metrics['mae'],
                   validation_mean_marginal_nlpd=metrics['mean_marginal_nlpd']+np.log(scale),
                   validation_95pct_coverage=metrics['interval_coverage'],
                   validation_mean_interval_width=scale*metrics['mean_interval_width'],
                   posterior_mean_W=float(p['weights'] @ p['W']),
                   beta_mean_shift_rmse=scale*record['beta_mean_shift_rmse'])
        if truth is not None:
            row['beta_rmse'] = float(np.sqrt(np.mean((scale*p['beta_mean']-truth['beta'][valid_ids])**2)))
            row['f_rmse'] = float(np.sqrt(np.mean((offset+scale*p['f_mean']-truth['fs'][valid_ids])**2)))
        rows.append(row)
        original_params = dict(fit['params'])
        for key in ('v_0', 'sigma_v0', 'k'):
            original_params[key] *= scale**2
        for key in ('w', 'sigma_w'):
            original_params[key] = np.asarray(original_params[key]) / scaler['x_scale']**2
        original_params['mu'] = offset+scale*original_params['mu']
        fits.append(dict(omega=record['omega'], **fit, params_original_units=original_params))
        label = f"omega_{record['omega']:g}"
        for name in ('beta', 'f', 'y'):
            shift = 0 if name == 'beta' else offset
            predictions[f'{label}_{name}_mean'] = shift + scale*p[f'{name}_mean']
            predictions[f'{label}_{name}_ci'] = shift + scale*p[f'{name}_ci']
            predictions[f'{label}_{name}_std'] = scale*p[f'{name}_std']
        predictions[f'{label}_W'] = p['W']
        predictions[f'{label}_weights'] = p['weights']
    print('\nOriginal target units:', flush=True)
    for row in rows:
        print(f"omega={row['omega']:g}: RMSE={row['validation_rmse']:.4f}, NLPD="
              f"{row['validation_mean_marginal_nlpd']:.4f}, coverage={row['validation_95pct_coverage']:.3f}, "
              f"beta shift={row['beta_mean_shift_rmse']:.4f}", flush=True)

    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    ws = [r['omega'] for r in rows]
    for ax, key, title in ((axes[0, 0], 'validation_rmse', 'Validation RMSE'),
                            (axes[0, 1], 'validation_mean_marginal_nlpd', 'Mean marginal predictive NLPD')):
        ax.plot(ws, [r[key] for r in rows], 'o-')
        ax.set(xscale='log', xlabel='Fixed omega', title=title)
        ax.grid(alpha=0.2)
    baseline = next(r for r in comparison['records'] if r['omega'] == comparison['baseline_omega'])
    beta_order = np.argsort(baseline['posterior']['beta_mean'])
    for record in comparison['records']:
        axes[1, 0].plot(scale*record['posterior']['beta_mean'][beta_order], label=f"omega={record['omega']:g}")
    axes[1, 0].set(xlabel='Validation points sorted by baseline beta mean', ylabel='Posterior mean beta',
                   title='Sensitivity of inferred beta')
    axes[1, 0].legend()
    base = baseline['posterior']
    y_order = np.argsort(ys[valid_ids], kind='stable')
    positions = np.arange(len(valid_ids))
    axes[1, 1].fill_between(positions, offset+scale*base['y_ci'][y_order, 0],
                            offset+scale*base['y_ci'][y_order, 1], alpha=0.25, label='95% predictive interval for y')
    axes[1, 1].scatter(positions, ys[valid_ids][y_order], c='black', s=14, label='Observed y')
    axes[1, 1].plot(positions, offset+scale*base['y_mean'][y_order], label='Predictive mean')
    axes[1, 1].set(xlabel='Validation points sorted by observed y', ylabel='y',
                   title=f"Baseline omega={comparison['baseline_omega']:g}")
    axes[1, 1].legend(fontsize=8)
    fig.suptitle(f"{dataset}: fixed lambda=-0.5; {n_train} train / {len(valid_ids)} validation", fontsize=14)

    prefix = Path(save_prefix or f'figure/betaprior_fit_{dataset}')
    prefix.parent.mkdir(parents=True, exist_ok=True)
    with prefix.with_name(prefix.name+'_summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with prefix.with_name(prefix.name+'_fit.json').open('w') as stream:
        json.dump(dict(dataset=dataset, csv_path=str(csv_path) if csv_path else None,
                       seed=seed, l=-0.5, scaler=scaler, train_source_ids=source_ids[train_ids],
                       validation_source_ids=source_ids[valid_ids], baseline_omega=comparison['baseline_omega'],
                       fits=fits, metrics=rows), stream, default=_json_default, indent=2)
    np.savez_compressed(prefix.with_name(prefix.name+'_predictions.npz'),
                         x_valid=xs[valid_ids], y_valid=ys[valid_ids], **predictions)
    fig.savefig(prefix.with_name(prefix.name+'_comparison.png'), dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return dict(comparison=comparison, metrics=rows, scaler=scaler, train_ids=train_ids,
                valid_ids=valid_ids, source_ids=source_ids, truth=truth)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['wine', 'synthetic'], default='wine')
    parser.add_argument('--csv', dest='csv_path')
    parser.add_argument('--n-samples', type=int, default=240)
    parser.add_argument('--train-fraction', type=float, default=0.8)
    parser.add_argument('--omegas', type=float, nargs='+', default=[0.5, 1, 2, 5])
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--n-restarts', type=int, default=3)
    parser.add_argument('--maxiter', type=int, default=200)
    parser.add_argument('--save-prefix')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    run_experiment(dataset=args.dataset, csv_path=args.csv_path, n_samples=args.n_samples,
                    train_fraction=args.train_fraction, omegas=args.omegas, seed=args.seed,
                    n_restarts=args.n_restarts, maxiter=args.maxiter,
                    save_prefix=args.save_prefix, show=not args.no_show)
