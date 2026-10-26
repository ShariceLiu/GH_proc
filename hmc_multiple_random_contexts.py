"""Compare five versus fifteen random observations of the same new functions.

Reuse shared-parameter HMC fits from hmc_multiple_functions_experiment.py.
Draw locations without replacement, independently per test function, using
only its grid and a fixed seed. The five locations are a subset of the fifteen.
Both conditions are scored on the same remaining 106 points per function.
"""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import special
from threadpoolctl import threadpool_limits

from GH_multiple_hmc import MODELS, context_log_weights, resample_context
from GH_proc_joint_hmc import predict_joint_mixture
from GP_TP_hmc import predict_baseline_mixture
from fit_beta_mu_1d_experiment import _save_csv
from hmc_gp_tp_1d_experiment import score_predictions
from hmc_multiple_functions_experiment import LABELS, COLORS, aggregate, load_data, write_json


def random_locations(n_grid, n_functions, *, seed=10026, counts=(5, 15)):
    """Outcome-independent nested random designs; do not force endpoints."""
    if not 0 < counts[0] < counts[1] < n_grid or n_functions < 1:
        raise ValueError('Require 0 < small count < large count < grid size')
    rng = np.random.default_rng(seed)
    draws = np.stack([rng.choice(n_grid, counts[1], replace=False) for _ in range(n_functions)])
    small, large = np.sort(draws[:, :counts[0]], axis=1), np.sort(draws, axis=1)
    test = np.stack([np.setdiff1d(np.arange(n_grid), ids) for ids in large])
    return small, large, test


def file_hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(folder, source, *, seed=10026):
    """Freeze original realizations and fit provenance; never simulate anew."""
    source = source.resolve()
    data = load_data(source)
    provenance = {}
    for model in MODELS:
        fit = json.loads((source/model/'fit.json').read_text())
        if not fit['diagnostics_ok']:
            raise ValueError(f'{model}: training HMC diagnostics failed')
        for key, original in [('training_function_ids', 'train_function_ids'),
                              ('training_point_ids', 'train_ids')]:
            if not np.array_equal(fit[key], data[original]):
                raise ValueError(f'{model}: training split does not match the saved dataset')
        provenance[model] = dict(samples=fit['selected_samples'],
                                 sha256=file_hash(source/model/fit['selected_samples']))
    settings = dict(source=str(source), source_dataset_sha256=file_hash(source/'dataset.npz'),
                    fits=provenance, location_seed=seed, context_counts=[5, 15],
                    generating_settings=json.loads((source/'generation.json').read_text()),
                    design='Uniform sampling without replacement on the saved grid, independently '
                           'per test function. Five is nested in fifteen. No forced endpoints. '
                           'All models and both counts use the same held-out set for each function.')
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'dataset.npz').exists():
        if json.loads((folder/'settings.json').read_text()) != settings:
            raise ValueError('Use a new output directory for changed locations, data or source fits')
        return
    small, large, test = random_locations(len(data['x']), len(data['test_function_ids']), seed=seed)
    data.pop('context_ids')
    data.pop('test_ids')
    data.update(context_5_ids=small, context_15_ids=large, evaluation_ids=test)
    np.savez_compressed(folder/'dataset.npz', **data)
    write_json(folder/'settings.json', settings)
    rows = [dict(function_id=int(function), count=count, grid_index=int(i), x=float(data['x'][i]))
            for j, function in enumerate(data['test_function_ids']) for count in (5, 15)
            for i in data[f'context_{count}_ids'][j]]
    _save_csv(folder/'observation_locations.csv', rows)


def checked_weights(theta, x, y, model):
    """Refine log-W quadrature until each theta's context evidence converges."""
    nodes = 192
    t, logw = context_log_weights(theta, x, y[None], model=model,
                                  quadrature_nodes=nodes, chunk_size=32)
    error = 0.
    if model != 'gp':
        for _ in range(3):
            fine_t, fine_logw = context_log_weights(theta, x, y[None], model=model,
                                                   quadrature_nodes=nodes*2, chunk_size=32)
            error = float(np.max(np.abs(special.logsumexp(logw, axis=2)
                                        - special.logsumexp(fine_logw, axis=2))))
            t, logw, nodes = fine_t, fine_logw, nodes*2
            if error < 1e-5:
                break
        if error >= 1e-5:
            raise ValueError(f'{model}: W quadrature has not converged: {error}')
    return t, logw[0], error


def predict_model(folder, data, model, *, counts=(5, 15), draws=6000, seed=11026):
    settings = json.loads((folder/'settings.json').read_text())
    source = Path(settings['source'])
    sample_path = source/model/settings['fits'][model]['samples']
    if file_hash(sample_path) != settings['fits'][model]['sha256']:
        raise ValueError('Source HMC posterior has changed since preparation')
    with np.load(sample_path) as saved:
        theta = saved['theta_log_samples'].reshape(-1, 5)
    predictor = predict_baseline_mixture if model == 'gp' else predict_joint_mixture
    for count in counts:
        destination = folder/f'context_{count}'/model
        destination.mkdir(parents=True, exist_ok=True)
        rows, diagnostics = [], []
        for j, function in enumerate(data['test_function_ids']):
            ids = data[f'context_{count}_ids'][j]
            x, y = data['x'][ids], data['y'][function, ids]
            print(f'{model}, {count} random observations: function {j+1}/{len(data["test_function_ids"])}', flush=True)
            t, logw, error = checked_weights(theta, x, y, model)
            sample_seed = seed + 10000*count + 100*MODELS.index(model) + j
            q, diag = resample_context(theta, logw, t, model=model, draws=draws, seed=sample_seed)
            diag.update(model=model, context_count=count, function_id=int(function),
                        quadrature_nodes=len(t), max_quadrature_log_error=error,
                        prediction_draws=draws, resampling_seed=sample_seed)
            if diag['theta_importance_ess'] < max(200., .03*len(theta)) or diag['edge_mass'] > 1e-8:
                raise ValueError(f'Further context inference needed: {diag}')
            fit = dict(model=model, log_samples=q[None], nu=None)
            prediction = predictor(y, x, data['x'], fit)
            local = dict(f=data['f'][function], y=data['y'][function], test_ids=data['evaluation_ids'][j])
            row = dict(model=model, context_count=count, function_id=int(function),
                       true_W=float(data['W'][function]),
                       **score_predictions(local, prediction, previous=model != 'gp'))
            if model != 'gp':
                ws = np.exp(q[:, 5])
                diag.update(W_mean=float(ws.mean()), W_std=float(ws.std()),
                            W_lower=float(np.quantile(ws, .025)), W_upper=float(np.quantile(ws, .975)))
            np.savez_compressed(destination/f'function_{function:02d}_prediction.npz',
                                posterior_log_parameters=q, context_ids=ids,
                                evaluation_ids=data['evaluation_ids'][j],
                                **{key: value for key, value in prediction.items()
                                   if not key.startswith('component_')})
            rows.append(row)
            diagnostics.append(diag)
            _save_csv(destination/'per_function_scores.csv', rows)
            write_json(destination/'prediction_diagnostics.json', diagnostics)
            print(f'  theta weight ESS={diag["theta_importance_ess"]:.0f}, '
                  f'quadrature={len(t)}, f RMSE={row["f_rmse"]:.4f}', flush=True)


def read_scores(path):
    with path.open() as stream:
        return [{key: value if key == 'model' else float(value) for key, value in row.items()}
                for row in csv.DictReader(stream)]


def paired_difference(a, b, indices, metric):
    """Whole-function paired bootstrap, conditional on the trained ensemble."""
    av, bv = np.array([r[metric] for r in a]), np.array([r[metric] for r in b])
    if metric.endswith('_rmse'):
        estimate = np.sqrt(np.mean(av**2))-np.sqrt(np.mean(bv**2))
        replicates = np.sqrt(np.mean(av[indices]**2, axis=1))-np.sqrt(np.mean(bv[indices]**2, axis=1))
    else:
        estimate, replicates = np.mean(av-bv), np.mean((av-bv)[indices], axis=1)
    return dict(difference=float(estimate), lower=float(np.quantile(replicates, .025)),
                upper=float(np.quantile(replicates, .975)))


def summarize(folder, data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    settings = json.loads((folder/'settings.json').read_text())
    rows, all_rows, scores, diagnostics = [], [], {}, {}
    for count in (5, 15):
        for model in MODELS:
            destination = folder/f'context_{count}'/model
            local = read_scores(destination/'per_function_scores.csv')
            if [int(r['function_id']) for r in local] != list(data['test_function_ids']):
                raise ValueError(f'Incomplete predictions: {count}, {model}')
            scores[count, model] = local
            all_rows.extend(local)
            diag = json.loads((destination/'prediction_diagnostics.json').read_text())
            diagnostics[f'{count}_{model}'] = diag
            rows.append(dict(model=model, context_count=count, n_test_functions=len(local),
                             evaluation_points_per_function=data['evaluation_ids'].shape[1],
                             **aggregate(local), min_theta_importance_ess=min(d['theta_importance_ess'] for d in diag),
                             max_quadrature_log_error=max(d['max_quadrature_log_error'] for d in diag)))
    _save_csv(folder/'scores.csv', rows)
    _save_csv(folder/'per_function_scores.csv', all_rows)
    rng = np.random.default_rng(12026)
    n = len(data['test_function_ids'])
    indices = rng.integers(0, n, size=(5000, n))
    paired = []
    for model in MODELS:
        for target in ('f', 'y'):
            for metric in ('rmse', 'mean_marginal_nlpd', 'coverage_95', 'mean_interval_width'):
                key = f'{target}_{metric}'
                paired.append(dict(model=model, comparison='15 minus 5', metric=key,
                                   **paired_difference(scores[15, model], scores[5, model], indices, key)))
                paired.append(dict(model=model, comparison='model minus GP with 15', metric=key,
                                   **paired_difference(scores[15, model], scores[15, 'gp'], indices, key)))
    _save_csv(folder/'paired_comparison.csv', paired)
    for count in (5, 15):
        for page in range((n+4)//5):
            positions = list(range(page*5, min(n, (page+1)*5)))
            fig, axes = plt.subplots(len(positions), 4, figsize=(16, 2.6*len(positions)),
                                     squeeze=False, sharex=True, sharey='row', constrained_layout=True)
            for row, j in enumerate(positions):
                function = data['test_function_ids'][j]
                ids = data[f'context_{count}_ids'][j]
                for col, model in enumerate(MODELS):
                    ax = axes[row, col]
                    with np.load(folder/f'context_{count}'/model/f'function_{function:02d}_prediction.npz') as pred:
                        ax.plot(data['x'], data['f'][function], 'k--', lw=1, label='True f')
                        ax.plot(data['x'], pred['f_mean'], color=COLORS[model], label='Posterior mean')
                        ax.fill_between(data['x'], *pred['f_ci'].T, color=COLORS[model], alpha=.22,
                                        label='Pointwise 95% interval')
                    ax.scatter(data['x'][ids], data['y'][function, ids], s=17, c='black', label=f'{count} observed y')
                    ax.set_title(LABELS[model], fontsize=10)
                    ax.grid(alpha=.2)
                    if col == 0:
                        ax.set_ylabel(f'Test {j+1}, W={data["W"][function]:.2f}\nf')
                    if row == len(positions)-1:
                        ax.set_xlabel('x')
            axes[0, 0].legend(fontsize=6)
            fig.suptitle(f'{count} random observations per new function | shared kernels learned from 30 training functions')
            fig.savefig(folder/f'context_{count}'/f'regression_page_{page+1}.png', dpi=145)
            plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    keys = ['f_rmse', 'f_mean_marginal_nlpd', 'f_mean_interval_width']
    for ax, key, title in zip(axes, keys, ['Latent f RMSE', 'Latent f NLPD', 'Latent f 95% interval width']):
        for i, model in enumerate(MODELS):
            values = [next(r[key] for r in rows if r['model'] == model and r['context_count'] == c) for c in (5, 15)]
            ax.plot([5, 15], values, 'o-', color=COLORS[model], label=LABELS[model])
        ax.set_xticks([5, 15])
        ax.set_xlabel('Random observations per test function')
        ax.set_ylabel(title)
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle('Same 20 test functions and common held-out points; nested random locations')
    fig.savefig(folder/'comparison.png', dpi=170)
    plt.close(fig)
    tex_labels = dict(gp='GP', tp_mu_gp='STP + GP mean', mu_gp=r'GHP ($\mu$)', beta_gp=r'GHP ($\beta$)')
    lines = [r'% Requires \usepackage{booktabs}', r'\begin{table}[htbp]', r'\centering',
             r'\caption{Regression on 20 new GHP-beta functions, using five or fifteen nested random '
             r'observations per function. Shared hyperparameter posteriors are learned from the same '
             r'30 training functions. Both conditions use the same 106 held-out points per function. '
             r'NLPD is the mean pointwise negative log predictive density.}',
             r'\label{tab:random_context_comparison}', r'\begin{tabular}{lrrrrr}', r'\toprule',
             r'Model & Observations & RMSE ($f$) & NLPD ($f$) & RMSE ($y$) & NLPD ($y$) \\', r'\midrule']
    for i, model in enumerate(MODELS):
        if i:
            lines.append(r'\addlinespace')
        for count in (5, 15):
            r = next(r for r in rows if r['model'] == model and r['context_count'] == count)
            lines.append(tex_labels[model]+f' & {count} & '+' & '.join(f'{r[k]:.6f}' for k in
                         ['f_rmse', 'f_mean_marginal_nlpd', 'y_rmse', 'y_mean_marginal_nlpd'])+r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    (folder/'comparison_table.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_json(folder/'report.json', dict(settings=settings, scores=rows, paired_comparison=paired,
                                        prediction_diagnostics=diagnostics,
                                        inference='Reuse all 8000 shared-theta HMC states from the original '
                                        '30 training functions. Each new function independently updates theta '
                                        'and its fresh W from its own random observations only. Checked log-W '
                                        'quadrature, theta importance weights, stratified posterior resampling, '
                                        'conditional Gaussian moments and actual mixture quantiles.',
                                        scope='One training ensemble, 20 new functions and one random nested '
                                        'location design each. Bootstrap resamples whole test functions, '
                                        'conditional on the fitted training ensemble.'))
    (folder/'README.md').write_text(
        '# Random observations on new GHP-beta functions\n\n'
        'Run `python hmc_multiple_random_contexts.py`. Stages: `--stage prepare`, '
        '`--stage predict --models beta_gp --counts 5 15`, `--stage summarize`.\n\n'
        'Reuse the exact 30 training and 20 test functions, and all four trained '
        'HMC posteriors, from `../multiple_functions_ghp_beta/`. Training functions '
        'still have their original 31 observations; this experiment changes the '
        'observations revealed on each NEW test function. Kernel parameters remain '
        'uncertain and each context updates their posterior separately. Fixed '
        'GIG lambda=omega=1; STP is a TP residual plus an independent GP mean, '
        'with fixed nu=5 (IG shape=2.5, scale=1.5).\n\n'
        'Location seed 10026. Sample 15 distinct indices uniformly from the 121-point '
        'grid separately for each function; the first five sampled indices define '
        'the smaller context. Sort only after sampling. No endpoints are forced '
        'and no draws are rejected based on locations or outcomes. All four models '
        'use the same observations. Both context sizes are evaluated on the same '
        '106 remaining points per function. Thus the five-point scores here differ '
        'from the earlier equally spaced five-point experiment.\n\n'
        'Fresh W and theta are updated jointly by context likelihood weighting of '
        'the training theta samples and converged log-W quadrature. Training W '
        'draws are not reused as test-function W. Importance ESS and quadrature '
        'diagnostics must pass. Each Gaussian mixture uses 6000 stratified draws, '
        'including within-component and between-component uncertainty.\n\n'
        '- `dataset.npz`, `observation_locations.csv`: exact functions and nested splits.\n'
        '- `settings.json`: source paths, hashes and random seed.\n'
        '- `scores.csv`, `comparison_table.tex`, `comparison.png`: paired comparison.\n'
        '- `paired_comparison.csv`: bootstrap intervals using whole test functions.\n'
        '- `context_15/regression_page_*.png`: all 20 regressions with 15 observations.\n'
        '- `context_5/regression_page_*.png`: corresponding five-observation regressions.\n'
        '- `context_*/MODEL/`: conditional posterior draws, predictions, scores and diagnostics.\n\n'
        'The sample count comparison is paired. Conclusions still depend on this '
        'training ensemble and one random observation design per test function.\n', encoding='utf-8')
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', default='figure/multiple_functions_ghp_beta')
    parser.add_argument('--output-dir', default='figure/multiple_functions_random_contexts')
    parser.add_argument('--stage', choices=['all', 'prepare', 'predict', 'summarize'], default='all')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--counts', nargs='+', type=int, choices=[5, 15], default=[5, 15])
    parser.add_argument('--prediction-draws', type=int, default=6000)
    args = parser.parse_args()
    folder = Path(args.output_dir)
    with threadpool_limits(limits=1):
        prepare(folder, Path(args.source))
        if args.stage != 'prepare':
            data = load_data(folder)
            if args.stage in ('all', 'predict'):
                for model in args.models:
                    predict_model(folder, data, model, counts=args.counts, draws=args.prediction_draws)
            if args.stage in ('all', 'summarize'):
                summarize(folder, data)
