"""Compare W*beta GP regression against an independent additive GP mu.

python compare_beta_mu_gp.py --no-show
python compare_beta_mu_gp.py --seeds 5 --no-show

Data and known parameters match the 2D GP-beta demo: a 13x13 grid, 85 observed
locations, mu=0, beta~GP(0,Sigma), GIG(1,1). The seed-5 realization reproduces
the earlier GP-beta maps. All seeds use the same observation locations.

Models:
  beta_gp: f=W*beta+g, beta~GP(0,Sigma), g|W~GP(0,W*K).
  mu_gp_same: f=mu+g, mu~GP(0,Sigma) independently of W, beta=0.
  mu_gp_matched: mu covariance E_prior[W^2]*Sigma; equal unconditional
                 covariance of y to beta_gp, but a different joint prior.

Compare the common latent contribution h=W*beta (first model) with h=mu
(alternatives). E[W*beta|y] is calculated component-wise, preserving dependence.
There is no parameter fitting, target centering, or use of true W for inference.
These controls test the effect of the W coupling and prior scale. Predictive
similarity from a few realizations does not identify which latent term exists.
"""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy import special

from GH_proc_w_betaprior import Multivariate_GH, beta_f_inf
from GH_proc_w_betaprior_fit import score_predictions
from GH_proc_mu_prior import beta_effect_posterior, mu_f_inf


LABELS = dict(beta_gp='GP beta: h = W beta', mu_gp_same='GP mu: same covariance',
              mu_gp_matched='GP mu: matched marginal covariance')


def gig_second_moment(l, omega):
    """E[W^2]=K_(l+2)(omega)/K_l(omega), for the scale-one GIG convention."""
    if not np.isfinite(l) or not np.isfinite(omega) or omega <= 0:
        raise ValueError('l must be finite and omega positive')
    value = float(special.kve(l+2, omega)/special.kve(l, omega))
    if not np.isfinite(value) or value <= 0:
        raise ValueError('The GIG second moment is not numerically finite')
    return value


def _score(posterior, data, test):
    m = len(data['ys'])
    subset = dict(weights=posterior['weights'], y_mean=posterior['y_mean'][test],
                  y_ci=posterior['y_ci'][test],
                  component_means=posterior['component_means'][:, np.r_[test, m+test]],
                  component_y_variances=posterior['component_y_variances'][:, test])
    return score_predictions(data['ys'][test], subset)


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


def _plot_example(example, prefix, show):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data, fits = example['data'], example['fits']
    coords, x1, x2, train = data['coordinates'], data['x1'], data['x2'], data['train']
    shape = x1.shape
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    for row, name, truth in [(0, 'effect', data['effect']), (1, 'f', data['fs'])]:
        arrays = [truth] + [fits[key][f'{name}_mean'] for key in LABELS]
        low, high = min(a.min() for a in arrays), max(a.max() for a in arrays)
        titles = [f'True {"h = W beta" if name == "effect" else "f"}'] + list(LABELS.values())
        for col, (values, title) in enumerate(zip(arrays, titles)):
            ax = axes[row, col]
            mesh = ax.pcolormesh(x1, x2, values.reshape(shape), shading='auto',
                                 cmap='viridis', vmin=low, vmax=high)
            ax.set(title=title, xlabel='x1', ylabel='x2', aspect='equal')
            if col:
                ax.scatter(data['xs'][train, 0], data['xs'][train, 1], s=9,
                            facecolors='none', edgecolors='white', linewidths=0.5)
        fig.colorbar(mesh, ax=axes[row, :], shrink=0.85, label='Mean contribution h' if row == 0 else 'Latent f')
    fig.suptitle(f'Known parameters; seed {data["seed"]}; true W={data["W"]:.3f}; circles = observed y locations')
    fig.savefig(f'{prefix}_maps.png', dpi=150)

    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, constrained_layout=True)
    middle = len(coords)//2
    ids = np.arange(middle*len(coords), (middle+1)*len(coords))
    for ax, name, truth in [(axes[0], 'effect', data['effect']), (axes[1], 'f', data['fs'])]:
        ax.plot(coords, truth[ids], 'k--', label='Truth', lw=2)
        for color, (key, label) in enumerate(LABELS.items()):
            p = fits[key]
            ax.plot(coords, p[f'{name}_mean'][ids], color=f'C{color}', label=label)
            ax.fill_between(coords, p[f'{name}_ci'][ids, 0], p[f'{name}_ci'][ids, 1], color=f'C{color}', alpha=0.12)
        ax.set_ylabel('Mean contribution h' if name == 'effect' else 'Latent f')
        ax.legend(fontsize=8)
    observed = np.intersect1d(ids, train)
    axes[1].scatter(data['xs'][observed, 0], data['ys'][observed], color='black', s=18)
    axes[1].set_xlabel('x1')
    fig.suptitle(f'Slice at x2={coords[middle]:g}; shaded regions = pointwise 95% intervals')
    fig.savefig(f'{prefix}_slice.png', dpi=150)
    if show:
        plt.show()
    plt.close('all')


def run_comparison(*, seeds=(5, 6, 7, 8, 9), grid_size=13, n_train=85,
                   l=1., omega=1., save_prefix='figure/beta_vs_mu_gp', show=False):
    """Run paired comparisons on shared data; return rows, aggregates and first example.

    Default l/omega intentionally match GH_proc_w_betaprior_multid.py, not
    the later MAP fitting experiments. Change them explicitly to test another
    GIG setting. Other covariance parameters remain those of the 2D maps.
    """
    if not isinstance(grid_size, (int, np.integer)) or grid_size < 3:
        raise ValueError('grid_size must be an integer >= 3')
    if not isinstance(n_train, (int, np.integer)) or not 1 <= n_train < grid_size**2:
        raise ValueError('n_train must leave at least one held-out grid location')
    seeds = list(seeds)
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError('Provide a nonempty sequence of distinct seeds')
    factor = gig_second_moment(l, omega)
    coords = np.linspace(-3, 3, grid_size)
    x1, x2 = np.meshgrid(coords, coords, indexing='xy')
    xs = np.column_stack((x1.ravel(), x2.ravel()))
    train = np.sort(np.random.default_rng(6).choice(len(xs), size=n_train, replace=False))
    test = np.setdiff1d(np.arange(len(xs)), train)
    params = dict(l=l, omega=omega, mu=0., v_0=0.2, w=[2., 1.2], k=0.05,
                  sigma_v0=1., sigma_w=[0.25, 0.6])
    mu_params = {name: params[name] for name in ('l', 'omega', 'v_0', 'w', 'k')}
    mu_params.update(mu0=0., mu_w=params['sigma_w'])
    records, quadrature, arrays = [], [], {}
    example = None
    for seed in seeds:
        f, y, beta, W = Multivariate_GH(xs, **params, beta_prior=True, seed=seed, return_latents=True)
        data = dict(seed=seed, xs=xs, fs=f, ys=y, beta=beta, W=W, effect=W*beta,
                    coordinates=coords, x1=x1, x2=x2, train=train, test=test)
        compact = {}
        print(f'Seed {seed}: true W={W:.4f}; {len(train)} train / {len(test)} held out', flush=True)
        for key in LABELS:
            scale = factor if key == 'mu_gp_matched' else 1.
            if key == 'beta_gp':
                posterior = beta_f_inf(y[train], xs[train], xs, **params, quad_tol=1e-7)
                effect = beta_effect_posterior(posterior)
            else:
                posterior = mu_f_inf(y[train], xs[train], xs, **mu_params,
                                     mu_v0=scale*params['sigma_v0'], quad_tol=1e-7)
                effect = {name: posterior[f'mu_{name}'] for name in ('mean', 'var', 'std', 'ci')}
            metrics = _score(posterior, data, test)
            wmean = float(posterior['weights'] @ posterior['W'])
            wvar = float(posterior['weights'] @ (posterior['W']-wmean)**2)
            row = dict(seed=seed, model=key, true_W=W, latent_prior_variance=scale*params['sigma_v0'],
                       W_mean=wmean, W_var=wvar, log_marginal_likelihood=posterior['log_marginal_likelihood'], **metrics)
            for name, truth, mean, interval in [
                ('f', f, posterior['f_mean'], posterior['f_ci']),
                ('effect', W*beta, effect['mean'], effect['ci'])]:
                row[f'{name}_rmse'] = float(np.sqrt(np.mean((mean[test]-truth[test])**2)))
                row[f'{name}_coverage'] = float(np.mean((truth[test] >= interval[test, 0]) & (truth[test] <= interval[test, 1])))
            records.append(row)
            quadrature.append(dict(seed=seed, model=key, **posterior['quadrature']))
            compact[key] = {f'{name}_{stat}': posterior[f'{name}_{stat}']
                            for name in ('f', 'y') for stat in ('mean', 'var', 'std', 'ci')}
            compact[key].update({f'effect_{stat}': effect[stat] for stat in ('mean', 'var', 'std', 'ci')})
            compact[key].update(W=posterior['W'], weights=posterior['weights'])
            for name, value in compact[key].items():
                arrays[f'seed_{seed}_{key}_{name}'] = value
            print(f'  {key:15s}: y RMSE={metrics["rmse"]:.4f}, NLPD={metrics["mean_marginal_nlpd"]:.4f}, '
                  f'h RMSE={row["effect_rmse"]:.4f}, E[W|y]={wmean:.4f}', flush=True)
            del posterior, effect
        for name in ('ys', 'fs', 'beta', 'effect', 'W'):
            arrays[f'seed_{seed}_true_{name}'] = data[name]
        if example is None:
            example = dict(data=data, fits=compact)
    metrics = ['rmse', 'mean_marginal_nlpd', 'interval_coverage', 'mean_interval_width',
               'f_rmse', 'f_coverage', 'effect_rmse', 'effect_coverage']
    aggregates = []
    for key in LABELS:
        rows = [row for row in records if row['model'] == key]
        entry = dict(model=key, realizations=len(rows))
        for metric in metrics:
            values = np.array([row[metric] for row in rows])
            entry[metric+'_mean'] = float(values.mean())
            entry[metric+'_sd_across_realizations'] = float(values.std(ddof=1)) if len(values) > 1 else None
        aggregates.append(entry)
    # Positive paired differences favor the generating beta model.
    paired = []
    for key in ('mu_gp_same', 'mu_gp_matched'):
        entry = dict(model=key)
        for metric in ('rmse', 'mean_marginal_nlpd'):
            differences = [next(r[metric] for r in records if r['seed'] == seed and r['model'] == key)
                           - next(r[metric] for r in records if r['seed'] == seed and r['model'] == 'beta_gp')
                           for seed in seeds]
            entry[metric+'_difference_by_seed'] = differences
            entry[metric+'_mean_difference'] = float(np.mean(differences))
        paired.append(entry)
    result = dict(parameters=params, seeds=seeds, grid_size=grid_size, n_train=n_train,
                  split_seed=6, E_prior_W_squared=factor, records=records,
                  aggregates=aggregates, paired_differences=paired, quadrature=quadrature)
    prefix = Path(save_prefix)
    prefix.parent.mkdir(parents=True, exist_ok=True)
    _save_csv(Path(f'{prefix}_scores.csv'), records)
    _save_csv(Path(f'{prefix}_aggregate.csv'), aggregates)
    Path(f'{prefix}_report.json').write_text(json.dumps(result, indent=2, default=_json_default))
    np.savez_compressed(f'{prefix}_predictions.npz', xs=xs, train_ids=train, test_ids=test, **arrays)
    _plot_example(example, prefix, show)
    print('\nAverages across realizations:', flush=True)
    for row in aggregates:
        print(f'{row["model"]:15s}: y RMSE={row["rmse_mean"]:.4f}, NLPD={row["mean_marginal_nlpd_mean"]:.4f}, '
              f'f RMSE={row["f_rmse_mean"]:.4f}, h RMSE={row["effect_rmse_mean"]:.4f}', flush=True)
    print(f'Saved {prefix}_*; E_prior[W^2]={factor:.6f}', flush=True)
    result['example'] = example
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seeds', nargs='+', type=int, default=[5, 6, 7, 8, 9])
    parser.add_argument('--grid-size', type=int, default=13)
    parser.add_argument('--n-train', type=int, default=85)
    parser.add_argument('--l', type=float, default=1.)
    parser.add_argument('--omega', type=float, default=1.)
    parser.add_argument('--save-prefix', default='figure/beta_vs_mu_gp')
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    run_comparison(seeds=args.seeds, grid_size=args.grid_size, n_train=args.n_train,
                    l=args.l, omega=args.omega, save_prefix=args.save_prefix, show=not args.no_show)
