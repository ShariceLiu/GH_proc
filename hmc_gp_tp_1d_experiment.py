"""Two-SE Gaussian/Student-t baselines on the saved beta/mu 1D dataset.

Run: python hmc_gp_tp_1d_experiment.py --no-show
Default output: figure/gp_tp_1d_hmc/. TP nu=5 is fixed; all five positive
kernel/noise parameters are sampled. Saved beta/mu results are also scored
for latent f NLPD and included in comparisons, without refitting them.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import optimize, special, stats
from threadpoolctl import threadpool_limits

from fit_beta_mu_1d_experiment import _json_default, _save_csv
from hmc_beta_mu_1d_experiment import load_dataset
from GP_TP_hmc import sample_baseline_parameters, predict_baseline_mixture, mixture_logpdf


def score_predictions(data, prediction, *, previous=False):
    """Held-out f/y scores; NLPD evaluates their distinct mixture distributions."""
    test = data['test_ids']
    result = {}
    for name, truth in [('f', data['f']), ('y', data['y'])]:
        mean, ci = prediction[f'{name}_mean'][test], prediction[f'{name}_ci'][test]
        result[f'{name}_rmse'] = float(np.sqrt(np.mean((mean-truth[test])**2)))
        result[f'{name}_coverage_95'] = float(np.mean((truth[test] >= ci[:, 0]) & (truth[test] <= ci[:, 1])))
        result[f'{name}_mean_interval_width'] = float(np.mean(ci[:, 1]-ci[:, 0]))
        component = prediction[f'component_{name}_means'] if previous else prediction['component_means']
        result[f'{name}_mean_marginal_nlpd'] = float(-np.mean(mixture_logpdf(
            truth[test], component[:, test], prediction[f'component_{name}_variances'][:, test],
            df=np.inf if previous else float(prediction['component_df']), weights=prediction['weights'])))
    return result


def diagnostics_summary(fit):
    diagnostics = fit['diagnostics']
    return dict(max_rhat=float(np.max(diagnostics['rhat'])),
                min_bulk_ess=float(np.min(diagnostics['ess_bulk'])),
                min_tail_ess=float(np.min(diagnostics['ess_tail'])),
                divergences=int(np.sum(fit['divergent'])),
                acceptance=float(np.mean(fit['acceptance'])),
                component_swap_acceptance=float(np.mean(fit['swap_accepted'])),
                diagnostics_ok=bool(fit['diagnostics_ok']))


def load_previous_results(data, folder='figure/beta_vs_mu_1d_joint_hmc'):
    """Read saved GIG beta/mu fits only if their dataset/split exactly matches."""
    folder = Path(folder)
    if not (folder/'report.json').exists():
        return [], {}, None
    with np.load(folder/'dataset.npz') as saved:
        for key in ['x', 'y', 'f', 'train_ids', 'test_ids']:
            if not np.array_equal(saved[key], data[key]):
                raise ValueError(f'Previous beta/mu data do not match: {key}')
    report = json.loads((folder/'report.json').read_text(encoding='utf-8'))
    rows, predictions = [], {}
    for name in ['beta_gp', 'mu_gp']:
        with np.load(folder/f'{name}_predictions.npz') as prediction:
            scores = score_predictions(data, prediction, previous=True)
            old = report['models'][name]['scores']
            scores.update({key: old[key] for key in ['max_rhat', 'min_bulk_ess', 'min_tail_ess',
                                                    'divergences', 'acceptance',
                                                    'component_swap_acceptance', 'diagnostics_ok']})
            rows.append(dict(model=name, **scores))
            predictions[name] = {f'{target}_{stat}': prediction[f'{target}_{stat}'].copy()
                                 for target in ['f', 'y'] for stat in ['mean', 'ci']}
    return rows, predictions, str(folder)


def summarize_tp_scale(prediction):
    """Moments and quantiles of the theta-averaged inverse-gamma posterior."""
    shape, scales = float(prediction['W_shape']), prediction['component_W_scales']
    weights = prediction['weights']
    quantiles = []
    for probability in [.025, .5, .975]:
        unit_quantile = stats.invgamma.ppf(probability, a=shape)
        lower, upper = scales.min()*unit_quantile, scales.max()*unit_quantile
        if np.isclose(lower, upper, rtol=1e-14, atol=0):
            value = (lower+upper)/2
        else:
            value = optimize.brentq(lambda w: weights@special.gammaincc(shape, scales/w)-probability,
                                    lower, upper)
        quantiles.append(float(value))
    return dict(model='tp', parameter='W', mean=float(prediction['W_mean']),
                std=float(prediction['W_std']), variance=float(prediction['W_var']),
                median=quantiles[1], lower=quantiles[0], upper=quantiles[2])


def plot_results(data, predictions, fits, scores, folder, nu, show, *, tp_model='tp'):
    import matplotlib
    if not show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    x, train, test = data['x'], data['train_ids'], data['test_ids']
    labels = dict(gp='Gaussian process', tp=f'Student-t process (nu={nu:g})',
                  beta_gp='Beta GP (GIG mixing)', mu_gp='Mu GP (GIG mixing)',
                  tp_mu_gp=f'Student-t residual + independent GP mean (nu={nu:g})')
    colors = dict(gp='C2', tp='C3', beta_gp='C0', mu_gp='C1', tp_mu_gp='C3')
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey='row', constrained_layout=True)
    for col, model in enumerate(['gp', tp_model]):
        p = predictions[model]
        for row, target in enumerate(['f', 'y']):
            ax = axes[row, col]
            ax.plot(x, data['f'], 'k--', lw=1.5, label='True latent f')
            ax.plot(x, p[f'{target}_mean'], color=colors[model], label='Posterior mean')
            ax.fill_between(x, *p[f'{target}_ci'].T, color=colors[model], alpha=.22, label='Pointwise 95% interval')
            ids = train if target == 'f' else test
            ax.scatter(x[ids], data['y'][ids], s=12, marker='o' if target == 'f' else 'x',
                       c='gray' if target == 'f' else 'black', alpha=.65,
                       label='Training y' if target == 'f' else 'Held-out y')
            ax.set_ylabel('Latent f' if target == 'f' else 'Noisy observations y')
            ax.grid(alpha=.2)
            ax.legend(fontsize=8)
        axes[0, col].set_title(labels[model])
        axes[1, col].set_xlabel('x')
    fig.suptitle('Two learned SE components + learned noise | posterior hyperparameter averaging')
    fig.savefig(folder/'regression.png', dpi=160)
    if show:
        plt.show()
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True, constrained_layout=True)
    for ax, row in zip(axes.flat, scores):
        model, p = row['model'], predictions[row['model']]
        ax.plot(x, data['f'], 'k--', label='True latent f')
        ax.plot(x, p['f_mean'], color=colors[model], label='Posterior mean')
        ax.fill_between(x, *p['f_ci'].T, color=colors[model], alpha=.22, label='Pointwise 95% interval')
        ax.scatter(x[train], data['y'][train], s=9, c='gray', alpha=.45)
        ax.set_title(f'{labels[model]} | f RMSE={row["f_rmse"]:.5f}')
        ax.set_xlabel('x')
        ax.set_ylabel('Latent f')
        ax.grid(alpha=.2)
        ax.legend(fontsize=8)
    for ax in axes.flat[len(scores):]:
        ax.set_visible(False)
    fig.savefig(folder/'model_comparison.png', dpi=160)
    plt.close(fig)

    n_parameters = max(len(fit['parameter_names']) for fit in fits.values())
    fig, axes = plt.subplots(n_parameters, 2, figsize=(13, 2.2*n_parameters), constrained_layout=True)
    for col, (model, fit) in enumerate(fits.items()):
        for j, name in enumerate(fit['parameter_names']):
            for chain in fit['log_samples']:
                axes[j, col].plot(chain[:, j], lw=.4, alpha=.65)
            axes[j, col].set_ylabel(f'log {name}')
            axes[j, col].set_title(f'{model}: R-hat={fit["diagnostics"]["rhat"][j]:.3f}, '
                                    f'bulk ESS={fit["diagnostics"]["ess_bulk"][j]:.0f}', fontsize=9)
        axes[-1, col].set_xlabel('Retained iteration (warmup excluded)')
        for j in range(len(fit['parameter_names']), n_parameters):
            axes[j, col].set_visible(False)
    fig.savefig(folder/'traces.png', dpi=140)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 7), constrained_layout=True)
    for j, ax in enumerate(list(axes.flat)[:5]):
        for model, fit in fits.items():
            ax.hist(fit['log_samples'][..., j].ravel(), bins=50, density=True,
                    histtype='step', color=colors[model], lw=1.5, label=model)
        ax.set_xlabel('log '+fits['gp']['parameter_names'][j])
        ax.legend()
    p = predictions[tp_model]
    summary = (summarize_tp_scale(p) if tp_model == 'tp'
               else next(row for row in fits[tp_model]['summary'] if row['parameter'] == 'W'))
    grid = np.linspace(max(.001, summary['lower']*.3), summary['upper']*1.15, 250)
    if tp_model == 'tp':
        density = np.array([p['weights']@stats.invgamma.pdf(w, a=p['W_shape'], scale=p['component_W_scales']) for w in grid])
        axes[1, 2].plot(grid, density, color=colors[tp_model], label='TP posterior')
    else:
        axes[1, 2].hist(fits[tp_model]['parameter_samples'][..., 5].ravel(), bins=70,
                        density=True, histtype='step', color=colors[tp_model], label='Posterior W')
        axes[1, 2].set_xlim(grid[0], grid[-1])
    axes[1, 2].plot(grid, stats.invgamma.pdf(grid, a=nu/2, scale=(nu-2)/2), 'k--', label='Fixed IG prior')
    axes[1, 2].set_xlabel('W (inverse-gamma mixing)')
    axes[1, 2].legend()
    fig.savefig(folder/'parameter_posteriors.png', dpi=160)
    plt.close(fig)


def save_latex_table(folder, scores):
    labels = dict(gp='GP', tp='Student-$t$', beta_gp='Beta GP', mu_gp='Mu GP',
                  tp_mu_gp='Student-$t$ + GP mean')
    lines = [r'% Requires \usepackage{booktabs}', r'\begin{table}[htbp]', r'    \centering',
             r'    \caption{Held-out regression results on the same 1D dataset. '
             r'All models average over hyperparameter uncertainty.}',
             r'    \label{tab:four_model_regression}', r'    \begin{tabular}{lrrrr}',
             r'        \toprule', r'        Model & RMSE ($f$) & NLPD ($f$) & RMSE ($y$) & NLPD ($y$) \\',
             r'        \midrule']
    for row in scores:
        lines.append('        '+labels[row['model']]+' & '+ ' & '.join(
            f'{row[key]:.6f}' for key in ['f_rmse', 'f_mean_marginal_nlpd', 'y_rmse', 'y_mean_marginal_nlpd'])+r' \\')
    lines.extend([r'        \bottomrule', r'    \end{tabular}', r'\end{table}'])
    (folder/'comparison_table.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def run_experiment(*, output_dir='figure/gp_tp_1d_hmc', nu=5., chains=4, warmup=1000,
                   draws=1500, seed=3026, prior_sd=2., target_accept=.9,
                   leapfrog_steps=(20, 50), show=False):
    folder = Path(output_dir)
    folder.mkdir(parents=True, exist_ok=True)
    data = load_dataset()
    np.savez_compressed(folder/'dataset.npz', **{key: data[key] for key in
                        ['x', 'y', 'f', 'beta', 'effect', 'W', 'train_ids', 'test_ids']})
    scores, predictions, previous_source = load_previous_results(data)
    train = data['train_ids']
    y, x = data['y'][train], data['x'][train]
    fits, parameters, reports = {}, [], {}
    print(f'Exact saved 1D data: {len(train)} training / {len(data["test_ids"])} held out; TP nu={nu:g}', flush=True)
    for index, model in enumerate(['gp', 'tp']):
        started = time.perf_counter()
        print(f'Sampling {model}: two SE kernels and noise', flush=True)
        fit = sample_baseline_parameters(y, x, model=model, nu=nu, chains=chains,
                                         warmup=warmup, draws=draws, seed=seed+index,
                                         prior_sd=prior_sd, target_accept=target_accept,
                                         leapfrog_steps=leapfrog_steps, verbose=True)
        sampling_seconds = time.perf_counter()-started
        np.savez_compressed(folder/f'{model}_samples.npz', parameter_names=fit['parameter_names'],
                            **{key: value for key, value in fit.items() if isinstance(value, np.ndarray)})
        print(f'{model}: averaging {chains*draws} conditional predictions', flush=True)
        p = predict_baseline_mixture(y, x, data['x'], fit)
        np.savez_compressed(folder/f'{model}_predictions.npz', **p)
        metrics = dict(model=model, **score_predictions(data, p), **diagnostics_summary(fit))
        scores.append(metrics)
        fits[model], predictions[model] = fit, p
        parameters.extend(dict(model=model, **row) for row in fit['summary'])
        reports[model] = dict(prior_center=fit['prior_center'], prior_sd=fit['prior_sd'],
                              diagnostics=fit['diagnostics'], diagnostics_ok=fit['diagnostics_ok'],
                              scores=metrics, initialization=fit['initialization'], summary=fit['summary'],
                              step_size=fit['step_size'], inverse_mass=fit['inverse_mass'],
                              sampling_seconds=sampling_seconds, total_seconds=time.perf_counter()-started,
                              warmup_divergences=int(np.sum(fit['warmup_divergent'])))
        if model == 'tp':
            reports[model]['W_summary'] = summarize_tp_scale(p)
            _save_csv(folder/'W_summary.csv', [reports[model]['W_summary']])
        _save_csv(folder/'scores.csv', scores)
        _save_csv(folder/'parameters.csv', parameters)
        report = dict(dataset_source=data['source'], previous_models_source=previous_source,
                      generating_parameters=data['parameters'], generating_W=data['W'],
                      train_ids=data['train_ids'], test_ids=data['test_ids'], zero_prior_mean=True,
                      kernel='v1*exp(-w1*distance^2/2)+v2*exp(-w2*distance^2/2)',
                      fixed_tp_nu=nu, tp_W_prior=dict(distribution='inverse-gamma', shape=nu/2, scale=(nu-2)/2),
                      prior_note='Independent untruncated Normal priors on log(v1,v2,k,w1,w2). '
                      'Centers depend on training-response variance and are held fixed, as in earlier fits.',
                      sampler='Metropolis-corrected HMC with posterior-corrected kernel exchanges. '
                      'Latent functions and TP W integrated analytically; optimization only initializes chains.',
                      prediction='All retained hyperparameter draws have equal weight. GP components are Gaussian; '
                      'TP components have df=nu+n_train. Stored covariances are not Student-t shape matrices. '
                      'Includes within-conditional and between-hyperparameter covariance, exact mixture quantiles '
                      'and marginal NLPD for both f and y. GP/TP have no separately identified beta/mu effect.',
                      paper='https://proceedings.mlr.press/v33/shah14.pdf', chains=chains,
                      warmup=warmup, draws=draws, sampler_seed=seed, target_accept=target_accept,
                      leapfrog_steps=leapfrog_steps, models=reports)
        (folder/'report.json').write_text(json.dumps(report, indent=2, default=_json_default), encoding='utf-8')
        print(f'{model}: f RMSE={metrics["f_rmse"]:.6f}; f NLPD={metrics["f_mean_marginal_nlpd"]:.6f}; '
              f'max R-hat={metrics["max_rhat"]:.4f}; min ESS={metrics["min_bulk_ess"]:.0f}; '
              f'divergences={metrics["divergences"]}', flush=True)
    plot_results(data, predictions, fits, scores, folder, nu, show)
    save_latex_table(folder, scores)
    (folder/'README.md').write_text(
        '# GP and Student-t process baselines\n\n'
        'Run `python hmc_gp_tp_1d_experiment.py --no-show` from the repository root. '
        'Default TP nu=5 is fixed (IG shape=2.5, scale=1.5). Use `--nu` and a different '
        '`--output-dir` for a sensitivity run. Both models learn two SE amplitudes, '
        'two inverse squared length scales and noise by HMC. Zero prior mean.\n\n'
        'The exact earlier 91/90 data split is reused. Saved beta/mu fits are read '
        'without modification and checked for identical data before comparison. '
        'Their newly computed f NLPD is included in scores.csv.\n\n'
        '- regression.png: GP/TP latent f and fresh noisy y posterior predictions.\n'
        '- model_comparison.png: latent f predictions including the earlier beta/mu fits.\n'
        '- traces.png, parameter_posteriors.png: sampler traces and posterior densities.\n'
        '- scores.csv, comparison_table.tex: held-out errors, coverage, widths and NLPD.\n'
        '- parameters.csv, W_summary.csv: hyperparameter and TP mixing posterior summaries.\n'
        '- report.json: priors, sampler settings, diagnostics and model conventions.\n'
        '- *_samples.npz: all retained HMC hyperparameter states and warmup diagnostics.\n'
        '- *_predictions.npz: mixture means/variances/df/weights, full covariances and intervals; '
        'TP also stores inverse-gamma posterior component parameters.\n\n'
        'f and y have the same predictive mean but different uncertainty and scoring targets. '
        'NLPD is minus the average log mixture density, not average component NLPD. '
        'Continuous-density NLPD can be negative. TP signal and noise share W, which '
        'is integrated exactly, not fixed at its mean. Conditional TP covariance is '
        '(nu-2+Q)/(nu+n-2) times the Gaussian Schur complement; its Student-t shape '
        'matrix uses denominator nu+n. Hyperparameter priors use empirical training '
        'scale centers; inference is conditional on these fixed prior settings.\n', encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='figure/gp_tp_1d_hmc')
    parser.add_argument('--nu', type=float, default=5.)
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=1000)
    parser.add_argument('--draws', type=int, default=1500)
    parser.add_argument('--sampler-seed', type=int, default=3026)
    parser.add_argument('--prior-sd', type=float, default=2.)
    parser.add_argument('--target-accept', type=float, default=.9)
    parser.add_argument('--leapfrog-min', type=int, default=20)
    parser.add_argument('--leapfrog-max', type=int, default=50)
    parser.add_argument('--no-show', action='store_true')
    args = parser.parse_args()
    if not np.isfinite(args.nu) or args.nu <= 2:
        parser.error('--nu must be finite and greater than two')
    with threadpool_limits(limits=1):
        run_experiment(output_dir=args.output_dir, nu=args.nu, chains=args.chains,
                       warmup=args.warmup, draws=args.draws, seed=args.sampler_seed,
                       prior_sd=args.prior_sd, target_accept=args.target_accept,
                       leapfrog_steps=(args.leapfrog_min, args.leapfrog_max), show=not args.no_show)
