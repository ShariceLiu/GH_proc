"""Shared-beta simulation: plug-in GP/STP/GHP regression on new curves.

No HMC. Training uses shared-function marginalization and log-coordinate MAP.
Each test curve gets its own posterior W from its revealed observations only.
The estimated shared function and kernels are fixed during test prediction.
Two oracle checks use true GHP parameters or a GP with matching prior moments.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy import integrate, linalg, special, stats
from threadpoolctl import threadpool_limits

from GH_shared_function_map import MODELS, fit_shared_function, gig_mean
from GH_fixed_function import fixed_function_prediction, marginal_distribution, mixing_quadrature
from predictive_crps import gaussian_mixture_crps, gaussian_mixture_crps_closed_form

ORACLES = ('ghp_oracle', 'gp_matched_oracle')
LABELS = dict(gp='GP, shared mu', stp='STP, shared mu', ghp_mu='GHP, shared mu',
              ghp_beta='GHP, shared beta', ghp_oracle='GHP, known parameters/beta',
              gp_matched_oracle='GP, matched prior moments (oracle)')


def write_json(path, obj):
    def default(value):
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)
    path.write_text(json.dumps(obj, indent=2, default=default, allow_nan=False)+'\n', encoding='utf-8')


def prepare(root, seed, n_train, n_test):
    settings = dict(seed=seed, n_training_functions=n_train, n_test_functions=n_test,
                    grid_size=121, train_points=31, domain=[0., 10.],
                    truth=dict(lam=1., omega=1., v0=.2, w=2., k=.05, mu=0.),
                    beta='0.8 + 0.45*sin(0.6*x) + 0.15*cos(1.3*x)',
                    shared_beta=True, context_counts=[0, 5, 15],
                    context_selection='nested random subsets, independently chosen for each test curve')
    root.mkdir(parents=True, exist_ok=True)
    if (root/'dataset.npz').exists():
        if json.loads((root/'generation.json').read_text()) != settings:
            raise ValueError('Existing data settings differ; choose a new output directory')
    else:
        x = np.linspace(0, 10, 121)
        beta = .8+.45*np.sin(.6*x)+.15*np.cos(1.3*x)
        K = .2*np.exp(-(x[:, None]-x[None, :])**2)
        L = linalg.cholesky(K+1e-12*np.eye(len(x)), lower=True)
        f, y, W = [], [], []
        for child in np.random.SeedSequence(seed).spawn(n_train+n_test):
            rng = np.random.default_rng(child)
            mixing = stats.geninvgauss.rvs(1., 1., random_state=rng)
            function = mixing*beta+np.sqrt(mixing)*(L@rng.normal(size=len(x)))
            observations = function+np.sqrt(.05*mixing)*rng.normal(size=len(x))
            f.append(function); y.append(observations); W.append(mixing)
        context_rng = np.random.default_rng(seed+10000)
        orders = np.stack([context_rng.permutation(len(x)) for _ in range(n_test)])
        np.savez_compressed(root/'dataset.npz', x=x, beta=beta, f=f, y=y, W=W,
                            train_ids=np.arange(0, 121, 4),
                            train_functions=np.arange(n_train), test_functions=np.arange(n_train, n_train+n_test),
                            context_orders=orders)
        write_json(root/'generation.json', settings)
    with np.load(root/'dataset.npz') as saved:
        return {key: saved[key].copy() for key in saved.files}


def fit_all(root, data, starts):
    train = data['train_ids']
    Y = data['y'][data['train_functions']][:, train]
    dataset_hash = hashlib.sha256((root/'dataset.npz').read_bytes()).hexdigest()
    for model in MODELS:
        folder = root/model
        folder.mkdir(exist_ok=True)
        if (folder/'fit.json').exists():
            fit = json.loads((folder/'fit.json').read_text())
            if fit['source_sha256'] != dataset_hash or fit['starts_requested'] != starts:
                raise ValueError('Saved fit provenance differs; choose a new output directory')
            print(f'{model}: reusing fit', flush=True)
            continue
        print(f'{model}: {starts} optimization starts, {Y.shape[0]} curves x {Y.shape[1]} points', flush=True)
        fit = fit_shared_function(Y, data['x'][train], data['x'], model, starts=starts)
        fit.update(source_sha256=dataset_hash, starts_requested=starts)
        write_json(folder/'fit.json', fit)
        np.savez_compressed(folder/'shared_function.npz', x=data['x'], function=fit['function'],
                            population_mean=fit['population_mean'])
        print(f'{model}: MAP log posterior {fit["log_posterior"]:.4f}, '
              f'gradient error {fit["max_gradient_check_error"]:.2g}', flush=True)


def matched_gp_prediction(x, y_context, ids, beta):
    """Oracle Gaussian process with exactly the generating GHP prior moments."""
    ew = gig_mean()
    varw = special.kve(3., 1.)/special.kve(1., 1.)-ew**2
    K = .2*np.exp(-(x[:, None]-x[None, :])**2)
    C = ew*K+varw*np.outer(beta, beta)
    m = ew*beta
    noise = ew*.05
    if len(ids):
        A = C[np.ix_(ids, ids)]+noise*np.eye(len(ids))
        cross = C[:, ids]
        L = linalg.cholesky(A, lower=True)
        mean = m+cross@linalg.cho_solve((L, True), y_context-m[ids])
        solve = linalg.solve_triangular(L, cross.T, lower=True)
        cov = C-solve.T@solve
    else:
        mean, cov = m, C
    cov = (cov+cov.T)/2
    result = dict(mixing='gp', location=mean, conditional_beta=np.zeros(len(x)),
                  f_scale_cov=cov, y_scale_cov=cov+noise*np.eye(len(x)), W_mean=1., W_var=0.)
    for name in ('f', 'y'):
        result[f'{name}_mean'] = mean.copy()
        result[f'{name}_cov'] = result[f'{name}_scale_cov']
        result[f'{name}_var'] = np.diag(result[f'{name}_cov']).copy()
        dist = marginal_distribution(result, name)
        result[f'{name}_ci'] = np.column_stack([dist.ppf(.025), dist.ppf(.975)])
    return result


def student_crps(truth, mean, scale, df):
    """Student-t CRPS, Jordan/Krueger/Lerch (2019); scale is NOT the SD.

https://scoringrules.readthedocs.io/en/latest/generated/scoringrules.crps_t.html
"""
    z = (truth-mean)/scale
    constant = 2*np.sqrt(df)/(df-1)*np.exp(special.betaln(.5, df-.5)-2*special.betaln(.5, df/2))
    return scale*(z*(2*stats.t.cdf(z, df)-1)+2*stats.t.pdf(z, df)*(df+z*z)/(df-1)-constant)


def score_prediction(pred, truth_f, truth_y, test, *, check=False):
    """Pointwise density/CRPS and independent moment/CDF checks."""
    scores, pointwise, validation = {}, {}, {}
    if pred['mixing'] == 'gig':
        W, weights, diagnostics = mixing_quadrature(pred)
        np.testing.assert_allclose(weights@W, pred['W_mean'], rtol=1e-8)
        np.testing.assert_allclose(weights@(W-pred['W_mean'])**2, pred['W_var'], atol=1e-9, rtol=1e-7)
        means = pred['location'][None, :]+W[:, None]*pred['conditional_beta']
        validation['W_quadrature'] = diagnostics
    for name, truth in [('f', truth_f), ('y', truth_y)]:
        if truth is None:
            continue  # Real datasets have no observed latent-function truth.
        dist = marginal_distribution(pred, name)
        mean, variance, ci = pred[f'{name}_mean'], pred[f'{name}_var'], pred[f'{name}_ci']
        logpdf = dist.logpdf(truth)
        np.testing.assert_allclose(dist.mean(), mean, atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(dist.var(), variance, atol=1e-8, rtol=1e-8)
        np.testing.assert_allclose(dist.cdf(ci[:, 0]), .025, atol=1e-7)
        np.testing.assert_allclose(dist.cdf(ci[:, 1]), .975, atol=1e-7)
        if pred['mixing'] == 'gig':
            variances = W[:, None]*np.diag(pred[f'{name}_scale_cov'])[None, :]
            mix_logpdf = special.logsumexp(np.log(weights[:, None])+stats.norm.logpdf(truth, means, np.sqrt(variances)), axis=0)
            np.testing.assert_allclose(mix_logpdf, logpdf, atol=2e-6, rtol=1e-7)
            for endpoint, probability in [(0, .025), (1, .975)]:
                mix_cdf = weights@special.ndtr((ci[:, endpoint]-means)/np.sqrt(variances))
                np.testing.assert_allclose(mix_cdf, probability, atol=2e-7)
            crps, info = gaussian_mixture_crps(truth[test], means[:, test], variances[:, test], weights,
                                               atol=1e-8, rtol=1e-8)
            validation[f'{name}_max_logpdf_error'] = float(np.max(np.abs(mix_logpdf-logpdf)))
            validation[f'{name}_crps_max_error_estimate'] = float(np.max(info['integration_error_estimate']))
            if check:
                chosen = [0, len(test)//2, len(test)-1]
                ref = gaussian_mixture_crps_closed_form(truth[test[chosen]], means[:, test[chosen]], variances[:, test[chosen]], weights)
                np.testing.assert_allclose(crps[chosen], ref, atol=2e-7)
        elif pred['mixing'] == 'ig':
            df = 2*pred['W_shape']
            scale = np.sqrt(pred['W_scale']/pred['W_shape']*np.diag(pred[f'{name}_scale_cov']))
            crps = student_crps(truth[test], mean[test], scale[test], df)
            if check:
                j = test[len(test)//2]
                z = (truth[j]-mean[j])/scale[j]
                left = integrate.quad(lambda u: stats.t.cdf(u, df)**2, -np.inf, z, epsabs=1e-10)[0]
                right = integrate.quad(lambda u: stats.t.sf(u, df)**2, z, np.inf, epsabs=1e-10)[0]
                np.testing.assert_allclose(crps[len(test)//2], scale[j]*(left+right), atol=1e-8)
        else:
            crps = gaussian_mixture_crps_closed_form(truth[test], mean[None, test], variance[None, test])
        skewness = np.asarray(dist.stats(moments='s'))
        if skewness.ndim == 0:
            skewness = np.full(len(truth), skewness)
        scores.update({f'{name}_rmse': float(np.sqrt(np.mean((truth[test]-mean[test])**2))),
                       f'{name}_nlpd': float(-logpdf[test].mean()), f'{name}_crps': float(crps.mean()),
                       f'{name}_coverage_95': float(np.mean((truth[test]>=ci[test, 0]) & (truth[test]<=ci[test, 1]))),
                       f'{name}_interval_width': float(np.mean(ci[test, 1]-ci[test, 0])),
                       f'{name}_mean_abs_skewness': float(np.mean(np.abs(skewness[test])))})
        pointwise.update({f'{name}_nlpd': -logpdf[test], f'{name}_crps': crps,
                          f'{name}_skewness': skewness})
    return scores, pointwise, validation


def predict_all(root, data):
    x = data['x']
    for model in (*MODELS, *ORACLES):
        folder = root/model
        folder.mkdir(exist_ok=True)
        fit = json.loads((folder/'fit.json').read_text()) if model in MODELS else None
        for count in (0, 5, 15):
            case = folder/f'context_{count}'
            case.mkdir(exist_ok=True)
            rows, validations = [], []
            prior_prediction = None
            for j, function_id in enumerate(data['test_functions']):
                saved_report = case/f'function_{j:02d}.json'
                if saved_report.exists():
                    report = json.loads(saved_report.read_text())
                    rows.append(report['scores']); validations.append(report['validation'])
                    continue
                ids = np.sort(data['context_orders'][j, :count])
                test = np.setdiff1d(np.arange(len(x)), ids)
                y_context = data['y'][function_id, ids]
                if count == 0 and prior_prediction is not None:
                    pred = prior_prediction
                elif model == 'gp_matched_oracle':
                    pred = matched_gp_prediction(x, y_context, ids, data['beta'])
                else:
                    if model == 'ghp_oracle':
                        parameters = dict(v0=.2, k=.05, w=2.)
                        latent = data['beta']
                        mixing, kind = 'gig', 'beta'
                    else:
                        v, a, k, w, hw = np.exp(fit['log_parameters'][:5])
                        parameters = dict(v0=v, k=k, w=w)
                        latent = np.asarray(fit['function'])
                        kind = 'beta' if model == 'ghp_beta' else 'mu'
                        mixing = 'gp' if model == 'gp' else 'ig' if model == 'stp' else 'gig'
                    kwargs = {f'{kind}_train': latent[ids], f'{kind}_pred': latent}
                    pred = fixed_function_prediction(y_context, x[ids], x, mixing=mixing, **parameters, **kwargs)
                if count == 0:
                    prior_prediction = pred
                score, point, validation = score_prediction(pred, data['f'][function_id], data['y'][function_id], test,
                                                             check=(j == 0))
                score.update(model=model, context_count=count, function_id=int(function_id), n_test=len(test),
                             W_true=float(data['W'][function_id]), W_mean=float(pred['W_mean']), W_var=float(pred['W_var']))
                rows.append(score); validations.append(validation)
                np.savez_compressed(case/f'function_{j:02d}.npz', **pred, **point, context_ids=ids, test_ids=test)
                write_json(saved_report, dict(scores=score, validation=validation))
                if (j+1)%5 == 0:
                    print(f'{model}: {count} context observations, {j+1}/{len(data["test_functions"])} curves', flush=True)
            pd.DataFrame(rows).to_csv(case/'per_function_scores.csv', index=False)
            write_json(case/'validation.json', dict(all_checks_passed=True, curves=validations))


def summarize(root, data):
    frames = [pd.read_csv(root/model/f'context_{count}/per_function_scores.csv')
              for model in (*MODELS, *ORACLES) for count in (0, 5, 15)]
    all_scores = pd.concat(frames, ignore_index=True)
    all_scores.to_csv(root/'per_function_scores.csv', index=False)
    rows = []
    for model in (*MODELS, *ORACLES):
        for count in (0, 5, 15):
            values = all_scores[(all_scores.model == model) & (all_scores.context_count == count)]
            row = dict(model=model, context_count=count, n_test_functions=len(values))
            for target in ('f', 'y'):
                for metric in ('rmse', 'nlpd', 'crps', 'coverage_95', 'interval_width', 'mean_abs_skewness'):
                    a = values[f'{target}_{metric}'].to_numpy()
                    row[f'{target}_{metric}'] = float(np.sqrt(np.mean(a*a)) if metric == 'rmse' else a.mean())
            rows.append(row)
    summary = pd.DataFrame(rows)
    summary.to_csv(root/'scores.csv', index=False)
    paired = []
    rng = np.random.default_rng(22005)
    for count in (0, 5, 15):
        for model in ('stp', 'ghp_mu', 'ghp_beta', 'ghp_oracle'):
            reference = 'gp_matched_oracle' if model == 'ghp_oracle' else 'gp'
            a = all_scores[(all_scores.model == model)&(all_scores.context_count == count)].sort_values('function_id')
            b = all_scores[(all_scores.model == reference)&(all_scores.context_count == count)].sort_values('function_id')
            np.testing.assert_array_equal(a.function_id, b.function_id)
            ids = rng.integers(0, len(a), (5000, len(a)))
            for target in ('f', 'y'):
                for metric in ('rmse', 'nlpd', 'crps'):
                    va, vb = a[f'{target}_{metric}'].to_numpy(), b[f'{target}_{metric}'].to_numpy()
                    if metric == 'rmse':
                        delta = np.sqrt(np.mean(va[ids]**2, axis=1))-np.sqrt(np.mean(vb[ids]**2, axis=1))
                        estimate = np.sqrt(np.mean(va*va))-np.sqrt(np.mean(vb*vb))
                    else:
                        delta = np.mean((va-vb)[ids], axis=1); estimate = np.mean(va-vb)
                    lo, hi = np.quantile(delta, [.025, .975])
                    paired.append(dict(model=model, reference=reference, context_count=count, target=target,
                                       metric=metric, difference=estimate, lower=lo, upper=hi))
    pd.DataFrame(paired).to_csv(root/'paired_comparison.csv', index=False)
    parameter_rows = []
    x, beta = data['x'], data['beta']
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(x, beta, 'k--', label='True shared beta')
    axes[1].plot(x, gig_mean()*beta, 'k--', label='True population mean')
    for model in MODELS:
        fit = json.loads((root/model/'fit.json').read_text())
        v, a, k, w, hw = np.exp(fit['log_parameters'][:5])
        h = np.asarray(fit['population_mean'])
        parameter_rows.append(dict(model=model, v0=v, mean_GP_variance=a, k=k, w=w, mean_GP_w=hw,
                                   beta_GP_variance=a/gig_mean()**2 if model == 'ghp_beta' else None,
                                   population_mean_rmse=float(np.sqrt(np.mean((h-gig_mean()*beta)**2))),
                                   max_gradient_check_error=fit['max_gradient_check_error']))
        axes[1].plot(x, h, label=LABELS[model])
        if model == 'ghp_beta':
            axes[0].plot(x, fit['function'], label='Estimated shared beta')
    for ax in axes:
        ax.set_xlabel('x'); ax.legend(fontsize=8)
    axes[0].set_title('Shared beta estimated using training curves only')
    axes[1].set_title('Population means: all models have a shared GP prior')
    fig.tight_layout(); fig.savefig(root/'shared_functions.png', dpi=170); fig.savefig(root/'shared_functions.pdf'); plt.close(fig)
    pd.DataFrame(parameter_rows).to_csv(root/'parameters.csv', index=False)
    # A view of the intended population skewness before observing a new curve.
    # Oracle GHP and oracle GP have equal means/variances at every x.
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, index in zip(axes, [24, 96]):
        with np.load(root/'ghp_oracle/context_0/function_00.npz') as p:
            center, sd = p['y_mean'][index], np.sqrt(p['y_var'][index])
        grid = np.linspace(center-3.5*sd, center+6*sd, 400)
        for model in ('gp_matched_oracle', 'ghp_oracle', 'ghp_beta', 'gp'):
            with np.load(root/model/'context_0/function_00.npz') as saved:
                p = {key: saved[key].copy() for key in saved.files}
            p['mixing'] = str(p['mixing'])
            p['location'] = p['location'][[index]]
            p['conditional_beta'] = p['conditional_beta'][[index]]
            p['y_scale_cov'] = p['y_scale_cov'][np.ix_([index], [index])]
            ax.plot(grid, marginal_distribution(p, 'y').pdf(grid), label=LABELS[model], lw=1.5)
        ax.set_title(f'New curve before observations, x={x[index]:.1f}')
        ax.set_xlabel('y'); ax.set_ylabel('Predictive density'); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(root/'population_densities.png', dpi=170)
    fig.savefig(root/'population_densities.pdf'); plt.close(fig)
    lines = [r'\begin{table}[htbp]', r'\centering\small',
             r'\caption{Shared-beta plug-in experiment (seed 5): 30 training curves, 20 test curves. Mixing $W$ is integrated for each test curve. Oracle rows use generating information.}',
             r'\label{tab:shared_beta_plugin}', r'\begin{tabular}{llrrrrrr}', r'\toprule',
             r' & & \multicolumn{3}{c}{Latent $f$} & \multicolumn{3}{c}{Observation $y$} \\',
             r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}',
             r'Context & Model & RMSE & NLPD & CRPS & RMSE & NLPD & CRPS \\']
    tex_names = dict(gp=r'GP ($\hat\mu$)', stp=r'STP ($\hat\mu$)', ghp_mu=r'GHP ($\hat\mu$)',
                     ghp_beta=r'GHP ($\hat\beta$)', ghp_oracle='GHP oracle', gp_matched_oracle='GP matched-moment oracle')
    for count in (0, 5, 15):
        lines.append(r'\midrule')
        for model in (*MODELS, *ORACLES):
            r = summary[(summary.model == model)&(summary.context_count == count)].iloc[0]
            cols = [f'{t}_{s}' for t in ('f', 'y') for s in ('rmse', 'nlpd', 'crps')]
            lines.append(f'{count} & {tex_names[model]} & '+' & '.join(f'{r[c]:.5f}' for c in cols)+r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    (root/'comparison.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    # Plot every test curve, in generation order, at both observation counts.
    for count in (5, 15):
        with PdfPages(root/f'regression_context_{count}.pdf') as pdf:
            for start in range(0, len(data['test_functions']), 5):
                functions = data['test_functions'][start:start+5]
                fig, axes = plt.subplots(len(functions), 4, figsize=(16, 2.6*len(functions)), squeeze=False, sharex=True, sharey='row')
                for r, function_id in enumerate(functions):
                    for c, model in enumerate(MODELS):
                        j = start+r
                        with np.load(root/model/f'context_{count}/function_{j:02d}.npz') as p:
                            ax = axes[r, c]; ids = p['context_ids']
                            ax.fill_between(x, *p['f_ci'].T, alpha=.22, color=f'C{c}')
                            ax.plot(x, p['f_mean'], color=f'C{c}', lw=1.2)
                            ax.plot(x, data['f'][function_id], 'k--', lw=1)
                            ax.scatter(x[ids], data['y'][function_id, ids], c='black', s=12)
                            ax.set_title(f'{LABELS[model]}, curve {j+1}, W={data["W"][function_id]:.2f}', fontsize=9)
                fig.suptitle(f'{count} random observations per test curve; shaded: pointwise 95% latent intervals')
                fig.tight_layout(); pdf.savefig(fig)
                if start == 0:
                    fig.savefig(root/f'regression_context_{count}.png', dpi=140)
                plt.close(fig)
    print(summary[['model', 'context_count', 'f_rmse', 'f_nlpd', 'y_rmse', 'y_nlpd', 'y_crps']].to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('figure/shared_beta_plugin_seed5'))
    parser.add_argument('--seed', type=int, default=5)
    parser.add_argument('--train-functions', type=int, default=30)
    parser.add_argument('--test-functions', type=int, default=20)
    parser.add_argument('--starts', type=int, default=8)
    parser.add_argument('--stage', choices=('all', 'fit', 'predict', 'summarize'), default='all')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        data = prepare(args.output, args.seed, args.train_functions, args.test_functions)
        if args.stage in ('all', 'fit'):
            fit_all(args.output, data, args.starts)
        if args.stage in ('all', 'predict'):
            predict_all(args.output, data)
        if args.stage in ('all', 'summarize'):
            summarize(args.output, data)
