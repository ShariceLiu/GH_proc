"""Airfoil MAP comparison, learning GIG lambda/omega and STP nu.

Reuses the exact 300/1203 split of the original Airfoil HMC experiment.
Run: python learned_mixing_airfoil_experiment.py
Fit checkpoints permit resuming. Predictions store marginal summaries only;
mixture components regenerate cheaply from map.json and dataset.npz.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import optimize, special, stats
from threadpoolctl import threadpool_limits

from GH_learned_mixing_ard import ARDLearnedMixingMAP
from GH_proc_w_betaprior import _mixture_intervals
from GH_tabular_hmc import conditional_marginals
from GH_zero_tabular_bayes import mixture_crps, independent_crps


ROOT = Path('figure/airfoil_learned_mixing_map')
SOURCE = Path('figure/airfoil_single_function_hmc')
LABELS = {'gp': 'GP (two SE)', 'tp_mu_gp': r'STP + GP $\mu$',
          'mu_gp': r'GHP $\mu$', 'beta_gp': r'GHP $\beta$'}


def fit_key(model, learned=True):
    return 'gp' if model == 'gp' else model+('_learned' if learned else '_fixed')


def write_json(path, data):
    def convert(value):
        if isinstance(value, np.ndarray): return value.tolist()
        if isinstance(value, np.generic): return value.item()
        raise TypeError(type(value).__name__)
    path.write_text(json.dumps(data, indent=2, default=convert, allow_nan=False), encoding='utf-8')


def prepare(root):
    root.mkdir(parents=True, exist_ok=True)
    blob = (SOURCE/'dataset.npz').read_bytes()
    digest = hashlib.sha256(blob).hexdigest()
    dest = root/'dataset.npz'
    if dest.exists() and dest.read_bytes() != blob:
        raise RuntimeError('Existing dataset differs from the original Airfoil split')
    if not dest.exists(): dest.write_bytes(blob)
    with np.load(dest) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    train, test = data['train_ids'], data['test_ids']
    assert len(train) == 300 and len(test) == 1203 and data['x'].shape == (1503, 5)
    assert not np.intersect1d(train, test).size
    np.testing.assert_array_equal(np.union1d(train, test), np.arange(1503))
    for name in ('x', 'y'):
        np.testing.assert_allclose(data[name+'_mean'], data[name][train].mean(0), atol=1e-12)
        np.testing.assert_allclose(data[name+'_scale'], data[name][train].std(0), atol=1e-12)
        np.testing.assert_allclose(data[name+'_scaled'],
            (data[name]-data[name+'_mean'])/data[name+'_scale'], atol=1e-12)
    old = json.loads((SOURCE/'experiment.json').read_text())
    write_json(root/'configuration.json', dict(dataset='airfoil', source=str(SOURCE/'dataset.npz'),
        dataset_sha256=digest, split_seed=17026, n_train=300, n_test=1203,
        feature_names=old['feature_names'], standardization=old['standardization'],
        models=list(LABELS), inference='Marginal hyperparameter MAP with W integrated',
        fixed_mixing_controls=dict(lambda_=1., omega=1., nu=5.),
        log_kernel_prior_center=old['log_prior_center'], log_kernel_prior_sd=2.,
        mixing_priors=dict(lambda_normal=[1., 2.], log_omega_normal=[0., 1.],
                           log_nu_minus_two_normal=[float(np.log(3.)), 1.]),
        prior_array_convention='[mean, standard deviation]',
        map_measure='d log(theta) for GP; additionally d lambda d log(omega) for GHP, or d log(nu-2) for STP',
        gp_mixing='No mixing distribution; W=1 only as a computational convention',
        prediction='Hyperparameters fixed at MAP; W and both Gaussian processes integrated',
        scoring='Held-out observed y only, in original units; latent f is unknown'))
    (root/'method.txt').write_text(
        'Exact saved Airfoil split: seed 17026, 300 training / 1203 test rows.\n'
        'Five inputs; all rows are one function, with one W in mixing models.\n'
        'Inputs and response standardized using training data only.\n'
        'Each of the two SE kernels has its own variance and five ARD weights:\n'
        'K(x,x\') = v exp(-0.5 sum_d w_d (x_d-x\'_d)^2).\n'
        'GP (two SE): y ~ N(0, K1+K2+kI); no mixing distribution.\n'
        'GHP beta: y|W ~ N(0, W(K1+kI)+W^2 K_beta).\n'
        'GHP mu: y|W ~ N(0, W(K1+kI)+K_mu), with beta=0.\n'
        'STP + GP mean: y|W ~ N(0, W(K1+kI)+K_mu).\n'
        'The STP residual has an added independent GP mean; its marginal is\n'
        'not a single standard multivariate Student-t distribution.\n'
        'The original training response mean is restored for prediction.\n'
        'Kernel priors: independent Normal log parameters, SD 2, exactly the\n'
        'same centers as the original Airfoil HMC experiment.\n'
        'Mixing priors: lambda~N(1,2^2), log omega~N(0,1),\n'
        'log(nu-2)~N(log 3,1). GIG(lambda,omega,scale=1);\n'
        'W~IG(nu/2,(nu-2)/2) for STP, so E[W]=1.\n'
        'GP MAP maximizes p(log theta|y). GHP MAP maximizes\n'
        'p(log theta,lambda,log omega|y); STP maximizes\n'
        'p(log theta,log(nu-2)|y). MAP is coordinate-dependent.\n'
        'Both latent GPs are integrated analytically and W by adaptive\n'
        'Gauss-Legendre quadrature in log W, including the Jacobian.\n'
        'All parameter-dependent density normalizers are retained.\n'
        'Multistart L-BFGS-B uses analytic scores except the five-point\n'
        'Bessel order derivative. Checks include stationary gradients,\n'
        'local Hessian curvature and stricter quadrature refinement.\n'
        'Predictive means/variances and 95% equal-tail intervals use the\n'
        'conditional Gaussian mixture over p(W|training y,MAP parameters).\n'
        'Mu/beta uncertainty is retained; neither is a fixed plug-in function.\n'
        'No hyperparameter posterior uncertainty is included in MAP intervals.\n'
        'comparison.tex/csv contains the four models with learned mixing\n'
        'parameters where applicable. comparison_with_fixed_mixing.tex/csv\n'
        'also includes the fixed-mixing MAP controls.\n'
        'Matched fixed-mixing MAP controls refit all kernel/noise parameters,\n'
        'keeping lambda=omega=1 or nu=5. Historical HMC results are separate.\n'
        'Held-out y RMSE, NLPD, CRPS, coverage and width are in original units.\n'
        'No latent-f accuracy scores can be computed on this real dataset.\n'
        'Test responses are used only after fitting. Numerical validation\n'
        'does not establish a global optimum or identify a mixing population\n'
        'from this single independent W realization.\n', encoding='utf-8')
    return data, np.array(old['log_prior_center'])


def fit_map(target, folder, *, learned=True, starts=16):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder/'map.json'
    if path.exists():
        result = json.loads(path.read_text())
        if len(result['attempts']) >= starts: return result
    h0 = target.initial_h()
    initial = np.r_[target.center, h0] if learned else target.center.copy()
    bounds = [(-15., 15.)]*target.qdim
    if learned and target.has_mixing:
        bounds += [(-15., 15.), (-8., 8.)] if target.hdim == 2 else [(-8., 8.)]
    def expand(value): return value if learned else np.r_[value, h0]
    def objective(value):
        posterior, gradient = target.marginal(expand(value))
        return -posterior, -gradient[:len(initial)]
    rng = np.random.default_rng(170260+target.hdim)
    records = []
    checkpoint = folder/'map_starts.json'
    existing = json.loads(checkpoint.read_text()) if checkpoint.exists() else []
    begun = time.perf_counter()
    for i in range(starts):
        guess = initial.copy()
        if i >= 2: guess += rng.normal(0, .85, len(guess))
        if i % 2:
            guess[[0, 1]] = guess[[1, 0]]
            a, b = np.arange(3, 3+target.d), np.arange(3+target.d, target.qdim)
            guess[np.r_[a, b]] = guess[np.r_[b, a]]
        if i < len(existing):
            records.append(existing[i])
            continue
        fit = optimize.minimize(objective, guess, jac=True, method='L-BFGS-B', bounds=bounds,
            options=dict(maxiter=1500, ftol=1e-13, gtol=3e-6, maxls=40, maxcor=15))
        negative, negative_grad = objective(fit.x)
        good = bool(fit.success and np.isfinite(negative) and max(abs(negative_grad)) < 2e-4)
        records.append(dict(start=i, converged=good, log_posterior=float(-negative),
            max_gradient=float(max(abs(negative_grad))), message=str(fit.message),
            iterations=int(fit.nit), evaluations=int(fit.nfev), coordinates=fit.x.tolist()))
        write_json(checkpoint, records)
        print(f'{target.model} {"learned" if learned else "fixed"} {i+1}/{starts}: '
              f'log posterior={-negative:.7f}, gradient={max(abs(negative_grad)):.2g}, passed={good}', flush=True)
    good_records = [r for r in records if r['converged']]
    if not good_records: raise RuntimeError('No converged MAP starts; see map_starts.json')
    best = max(good_records, key=lambda r: r['log_posterior'])
    point = np.asarray(best['coordinates'])
    if not all(lo+1e-3 < v < hi-1e-3 for v, (lo, hi) in zip(point, bounds)):
        raise RuntimeError('Selected MAP touches a numerical bound')
    value, gradient = objective(point)
    eps = np.eye(len(point))*1e-4
    numerical = np.array([(objective(point+d)[0]-objective(point-d)[0])/2e-4 for d in eps])
    np.testing.assert_allclose(gradient, numerical, atol=1e-4, rtol=0)
    hessian = np.column_stack([(objective(point+d)[1]-objective(point-d)[1])/2e-4 for d in eps])
    curvature = float(np.linalg.eigvalsh((hessian+hessian.T)/2).min())
    if curvature <= 0: raise RuntimeError('Selected MAP is not a strict local maximum')
    u = expand(point)
    refined, refined_gradient = target.marginal(u, tol=1e-11, tail=50.)
    np.testing.assert_allclose(refined, -value, atol=1e-7, rtol=0)
    np.testing.assert_allclose(refined_gradient[:len(point)], -gradient, atol=1e-6, rtol=0)
    t, weights, info, _ = target.quadrature(u, tol=1e-11, tail=50.)
    W = np.exp(t)
    report = dict(model=target.model,
        mixing=('learned' if learned else 'fixed') if target.has_mixing else 'not applicable', u=u,
        log_posterior=-value, selected_start=best['start'], attempts=records,
        max_gradient=float(max(abs(gradient))), gradient_check_error=float(max(abs(gradient-numerical))),
        min_negative_log_posterior_hessian_eigenvalue=curvature,
        quadrature_refinement_change=float(abs(refined+value)), quadrature=info,
        W_mean=float(weights@W), W_std=float(np.sqrt(weights@(W-weights@W)**2)),
        W_variance=float(weights@(W-weights@W)**2), elapsed_seconds=time.perf_counter()-begun,
        prior_center=target.center, prior_sd=target.sd, bounds=bounds)
    write_json(path, report)
    return json.loads(path.read_text())


def predict_and_score(target, data, fit, folder):
    u = np.asarray(fit['u'])
    fingerprint = hashlib.sha256(u.tobytes()+(folder.parent/'dataset.npz').read_bytes()).hexdigest()
    score_path = folder/'scores.json'
    if score_path.exists() and (folder/'prediction.npz').exists():
        saved = json.loads(score_path.read_text())
        if saved['fit_data_sha256'] == fingerprint: return saved
    test = data['test_ids']
    xp, truth = data['x_scaled'][test], data['y'][test]
    components = target.predict_components(u, xp)
    weights, t = components['weights'], components['log_W']
    checks, results = {}, dict(test_ids=test, log_W=t, weights=weights)
    # Independent dense conditioning on posterior 1/50/99% W quantiles.
    indices = np.unique(np.searchsorted(np.cumsum(weights), [.01, .5, .99]))
    dense_error = 0.
    for i in indices:
        pars = np.r_[u[:target.qdim], t[i]] if target.has_mixing else u
        direct = conditional_marginals(target.y, target.x, xp[[0, 600, 1202]],
            pars, model=target.model)
        for key, expected in [('component_means', direct['mean']), ('component_f_variances', direct['f_variance'])]:
            actual = components[key][i, [0, 600, 1202]]
            np.testing.assert_allclose(actual, expected, atol=1e-8, rtol=1e-8)
            dense_error = max(dense_error, float(max(abs(actual-expected))))
    checks['dense_conditioning_max_error'] = dense_error
    scale, offset = float(data['y_scale']), float(data['y_mean'])
    mean_components = offset+scale*components['component_means']
    mean = weights@mean_components
    for kind in ('f', 'y'):
        variance_components = scale**2*components[f'component_{kind}_variances']
        variance = weights@(variance_components+(mean_components-mean)**2)
        intervals = _mixture_intervals(mean_components, variance_components, weights, .95)
        cdf = np.array([weights@special.ndtr((intervals[:,j]-mean_components)/np.sqrt(variance_components)) for j in (0, 1)])
        error = float(np.max(abs(cdf-np.array([.025, .975])[:, None])))
        if error > 1e-9: raise AssertionError('Mixture quantile CDF mismatch')
        checks[kind+'_quantile_cdf_error'] = error
        results.update({kind+'_mean': mean, kind+'_var': variance, kind+'_ci': intervals})
    variance_components = scale**2*components['component_y_variances']
    with np.errstate(divide='ignore'):
        logpdf = special.logsumexp(np.log(weights)[:, None]+stats.norm.logpdf(truth, mean_components, np.sqrt(variance_components)), axis=0)
    crps, refinements = np.empty(len(test)), []
    for start in range(0, len(test), 64):
        sl = slice(start, start+64)
        crps[sl], refinement = mixture_crps(truth[sl], mean_components[:,sl], np.sqrt(variance_components[:,sl]), weights, atol=1e-6)
        refinements.append(refinement)
    for j in (0, 600, 1202):
        reference, _ = independent_crps(truth[j], mean_components[:,j], np.sqrt(variance_components[:,j]), weights)
        np.testing.assert_allclose(crps[j], reference, atol=2e-6, rtol=0)
    results.update(y_logpdf=logpdf, y_crps=crps, y_truth=truth,
        native_mean=scale*(weights@components['component_native_means']))
    native = components['component_native_means']
    results['native_var'] = scale**2*(weights@(components['component_native_variances']+(native-weights@native)**2))
    intervals = results['y_ci']
    scores = dict(model=target.model, mixing=fit['mixing'], inference='MAP', n_train=300, n_test=len(test),
        y_rmse=float(np.sqrt(np.mean((mean-truth)**2))), y_nlpd=float(-logpdf.mean()), y_crps=float(crps.mean()),
        y_coverage=float(np.mean((truth>=intervals[:,0])&(truth<=intervals[:,1]))),
        y_interval_width=float(np.mean(intervals[:,1]-intervals[:,0])),
        fit_data_sha256=fingerprint, checks=checks, crps_refinements=refinements)
    np.savez_compressed(folder/'prediction.npz', **results)
    write_json(score_path, scores)
    print(json.dumps({k:scores[k] for k in ('model', 'mixing', 'y_rmse', 'y_nlpd', 'y_crps', 'y_coverage')}), flush=True)
    return scores


def summarize(root, data, targets, fits, scores):
    columns = ['model', 'mixing', 'inference', 'n_train', 'n_test', 'y_rmse', 'y_nlpd', 'y_crps', 'y_coverage', 'y_interval_width']
    table = pd.DataFrame(scores)[columns]
    primary = table[table.mixing != 'fixed']
    primary.to_csv(root/'comparison.csv', index=False)
    table.to_csv(root/'comparison_with_fixed_mixing.csv', index=False)
    tex = [r'\begin{table}[htbp]', r'\centering', r'\small',
        r'\caption{Airfoil regression on the original seed-17026 split: 300 training and 1,203 test rows. Hyperparameters use marginal MAP; $W$ and latent functions are integrated during prediction. Scores concern observed $y$ in original units. Fixed controls use $\lambda=\omega=1$ or $\nu=5$.}',
        r'\label{tab:airfoil_learned_mixing_map}', r'\begin{tabular}{llrrrrr}', r'\toprule',
        r'Model & Mixing & RMSE & NLPD & CRPS & Coverage (\%) & Width \\', r'\midrule']
    for row in scores:
        tex.append(f"{LABELS[row['model']]} & {row['mixing'].capitalize()} & {row['y_rmse']:.5f} & {row['y_nlpd']:.5f} & {row['y_crps']:.5f} & {100*row['y_coverage']:.2f} & {row['y_interval_width']:.4f} \\")
        tex[-1] += '\\'
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (root/'comparison_with_fixed_mixing.tex').write_text('\n'.join(tex)+'\n', encoding='utf-8')
    tex = [r'\begin{table}[htbp]', r'\centering', r'\small',
        r'\caption{Airfoil regression with 300 training and 1,203 test rows (split seed 17026). All four models use marginal MAP estimates of their two SE kernels and noise parameters. GHP also learns $\lambda,\omega$ and STP learns $\nu$. Predictions integrate latent functions and, where applicable, $W$, but not hyperparameter uncertainty. Scores evaluate observed $y$ in original units.}',
        r'\label{tab:airfoil_four_models_map}', r'\begin{tabular}{lrrrrr}', r'\toprule',
        r'Model & RMSE & NLPD & CRPS & 95\% coverage (\%) & Interval width \\', r'\midrule']
    for row in primary.to_dict('records'):
        tex.append(f"{LABELS[row['model']]} & {row['y_rmse']:.5f} & {row['y_nlpd']:.5f} & {row['y_crps']:.5f} & {100*row['y_coverage']:.2f} & {row['y_interval_width']:.4f} " + r'\\')
    tex += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (root/'comparison.tex').write_text('\n'.join(tex)+'\n', encoding='utf-8')
    params, mixing = [], []
    for key, fit in fits.items():
        target = targets[fit['model']]
        u = np.asarray(fit['u'])
        extra = ([u[-2], np.exp(u[-1])] if target.hdim == 2 else
                 [2+np.exp(u[-1])] if target.hdim == 1 else [])
        values = np.r_[np.exp(u[:target.qdim]), extra]
        factors = np.r_[np.repeat(float(data['y_scale'])**2, 3),
                        1/data['x_scale']**2, 1/data['x_scale']**2, np.ones(target.hdim)]
        for name, value, factor in zip(target.parameter_names, values, factors):
            params.append(dict(model=target.model, mixing=fit['mixing'], parameter=name,
                standardized_value=value, original_units_value=value*factor))
        mixing.append(dict(model=target.model, mixing=fit['mixing'],
            lambda_=float(u[-2]) if target.hdim == 2 else None,
            omega=float(np.exp(u[-1])) if target.hdim == 2 else None,
            nu=float(2+np.exp(u[-1])) if target.model == 'tp_mu_gp' else None,
            W_mean=fit['W_mean'] if target.has_mixing else None,
            W_std=fit['W_std'] if target.has_mixing else None,
            W_variance=fit['W_variance'] if target.has_mixing else None))
    pd.DataFrame(params).to_csv(root/'parameters.csv', index=False)
    pd.DataFrame(mixing).to_csv(root/'mixing_parameters.csv', index=False)
    old = pd.read_csv(SOURCE/'scores_with_zero_mean_crps.csv')
    old = old[old.model.isin(LABELS)].rename(columns={'y_mean_marginal_nlpd':'y_nlpd',
        'y_coverage_95':'y_coverage', 'y_mean_interval_width':'y_interval_width'})
    old['mixing'], old['inference'] = 'fixed', 'Full Bayes HMC (historical)'
    old.loc[old.model == 'gp', 'mixing'] = 'not applicable'
    pd.concat([table, old[columns]], ignore_index=True).to_csv(root/'comparison_with_previous_hmc.csv', index=False)
    fig, axes = plt.subplots(len(LABELS), 2, figsize=(11, 3.5*len(LABELS)), constrained_layout=True)
    for row, model in enumerate(LABELS):
        with np.load(root/fit_key(model)/'prediction.npz') as p:
            truth, mean, ci = p['y_truth'], p['y_mean'], p['y_ci']
            axes[row,0].scatter(truth, mean, s=10, alpha=.4)
            limits = [min(truth.min(), mean.min()), max(truth.max(), mean.max())]
            axes[row,0].plot(limits, limits, 'k--', lw=1)
            axes[row,0].set(xlabel='Held-out observed y', ylabel='Predictive mean', title=LABELS[model]+'; MAP')
            order = np.argsort(truth)[::12]
            axes[row,1].errorbar(np.arange(len(order)), mean[order],
                yerr=np.maximum(np.array([mean[order]-ci[order,0], ci[order,1]-mean[order]]), 0),
                fmt='.', alpha=.65, label='Predictive mean and 95% interval')
            axes[row,1].scatter(np.arange(len(order)), truth[order], s=10, c='black', label='Observed y')
            axes[row,1].set(xlabel='Subset of test rows, ordered by observed y', ylabel='y (original units)')
            axes[row,1].legend(fontsize=8)
    fig.savefig(root/'regression.pdf')
    fig.savefig(root/'regression.png', dpi=150)
    plt.close(fig)
    write_json(root/'validation.json', dict(passed=True, dataset_split_and_standardization=True,
        fits={key: {field:fit[field] for field in ['max_gradient', 'gradient_check_error',
            'min_negative_log_posterior_hessian_eigenvalue', 'quadrature_refinement_change']} for key, fit in fits.items()},
        predictions={row['model']+'_'+row['mixing']:row['checks'] for row in scores}))
    print(primary.to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT)
    parser.add_argument('--starts', type=int, default=16)
    args = parser.parse_args()
    data, center = prepare(args.output)
    train = data['train_ids']
    targets = {model:ARDLearnedMixingMAP(data['y_scaled'][train], data['x_scaled'][train], model, prior_center=center) for model in LABELS}
    fits, scores = {}, []
    for learned in (True, False):
        for model, target in targets.items():
            if not learned and not target.has_mixing:
                continue
            key = fit_key(model, learned)
            folder = args.output/key
            fit = fit_map(target, folder, learned=learned, starts=args.starts)
            fits[key] = fit
            scores.append(predict_and_score(target, data, fit, folder))
    summarize(args.output, data, targets, fits, scores)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
