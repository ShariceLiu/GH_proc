"""Seed-5 point-estimate comparison, retaining W uncertainty in prediction.

1. Maximize p(log(theta),log(W)|y_train), integrating Gaussian latent GPs.
2. At that optimum take the conditional GP mean for beta or mu, and fix it.
3. Discard the fitted W for prediction: recompute its conjugate posterior
   conditional on the fixed theta/function. Integrate it using GH/t formulas.

This is a plug-in approximation, not full Bayes or a joint mode over function
values and hyperparameters. No HMC states, true functions or test responses
are used for parameter/function estimation. The original HMC prior settings
and labelled kernel components are retained. MAP is coordinate dependent;
here the density is explicitly defined with respect to log parameters.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg, optimize, special, stats
from threadpoolctl import threadpool_limits

from GH_proc_joint_hmc import (make_joint_target, effective_coordinates,
                               physical_coordinates, conditional_prediction)
from GP_TP_hmc import make_baseline_target
from GH_fixed_function import fixed_function_prediction, marginal_distribution, mixing_quadrature
from predictive_crps import gaussian_mixture_crps, gaussian_mixture_crps_closed_form

MODELS = ('gp', 'tp_mu_gp', 'mu_gp', 'beta_gp')
LABELS = {'gp': 'GP, fixed mu', 'tp_mu_gp': 'STP, fixed mu',
          'mu_gp': 'GHP, fixed mu', 'beta_gp': 'GHP, fixed beta'}


def fit_point_estimate(y, x, *, model, starts=24, seed=505):
    """Multistart log-coordinate MAP; only training x/y accepted."""
    if starts < 2:
        raise ValueError('Use at least two starts to check both kernel assignments')
    scale = max(float(np.var(y)), 1e-12)
    center = np.log([.5*scale, .5*scale, .1*scale, 1., .25])
    if model == 'gp':
        target = make_baseline_target(y, x, model='gp', prior_center=center)
        initial = center.copy()
    else:
        target = make_joint_target(y, x, model=model, prior_center=center)
        initial_w = np.log(3/5) if model == 'tp_mu_gp' else np.arcsinh(1.)
        initial = effective_coordinates(np.r_[center, initial_w], model)
    rng = np.random.default_rng(seed)
    attempts, solutions = [], []
    for index in range(starts):
        guess = initial.copy()
        if index >= 2:
            guess += rng.normal(0, 1.5, len(guess))
        if index % 2:
            guess[[0, 1]] = guess[[1, 0]]
            guess[[3, 4]] = guess[[4, 3]]
        def objective(z):
            value, grad = target(z)
            return -value, -grad
        fit = optimize.minimize(objective, guess, jac=True, method='L-BFGS-B',
                                options=dict(maxiter=2000, ftol=1e-14, gtol=1e-8, maxls=40))
        value, grad = target(fit.x)
        good = bool(fit.success and np.isfinite(value) and np.max(np.abs(grad)) < 1e-4)
        attempts.append(dict(start=index, converged=good, log_posterior=float(value),
                             max_abs_gradient=float(np.max(np.abs(grad))), message=str(fit.message)))
        if good:
            solutions.append((value, fit.x, index))
    if not solutions:
        raise RuntimeError(f'{model}: no stationary optimization result')
    value, z, index = max(solutions, key=lambda item: item[0])
    q = z.copy() if model == 'gp' else physical_coordinates(z, model)
    # Independent finite differences check the implemented gradient and local curvature.
    h = 1e-4
    offsets = np.eye(len(z))*h
    numerical_gradient = np.array([(target(z+d)[0]-target(z-d)[0])/(2*h) for d in offsets])
    hessian = np.column_stack([(target(z+d)[1]-target(z-d)[1])/(2*h) for d in offsets])
    max_eigenvalue = float(np.linalg.eigvalsh((hessian+hessian.T)/2).max())
    np.testing.assert_allclose(numerical_gradient, target(z)[1], atol=2e-6, rtol=1e-4)
    if max_eigenvalue >= 0:
        raise RuntimeError('Selected result is not a strict local posterior maximum')
    return q, dict(model=model, prior_center=center.tolist(), prior_sd=2.,
                   map_measure='d log(theta) d log(W); GP omits W',
                   selection='largest posterior among converged starts; no test scores',
                   selected_start=index, log_posterior=value, attempts=attempts,
                   max_hessian_eigenvalue=max_eigenvalue,
                   gradient_check_max_error=float(np.max(np.abs(numerical_gradient-target(z)[1]))))


def estimate_function(y, x, xp, q, model):
    """Conditional mean/mode of the labelled mean GP, at the selected theta,W."""
    # The GP is exactly the mu model with W fixed to one.
    joint_q = np.r_[q, 0.] if model == 'gp' else q
    joint_model = 'mu_gp' if model == 'gp' else model
    p = conditional_prediction(y, x, xp, joint_q, model=joint_model)
    estimate = p['effect_mean']/p['W'] if model == 'beta_gp' else p['effect_mean']
    return estimate, p


def score_and_validate(data, prediction):
    """Closed-form density/quantiles; numerical CRPS with independent checks."""
    test = data['test_ids']
    W, weights, diagnostics = mixing_quadrature(prediction)
    np.testing.assert_allclose(weights@W, prediction['W_mean'], rtol=1e-9)
    np.testing.assert_allclose(weights@(W-prediction['W_mean'])**2, prediction['W_var'],
                               atol=1e-10, rtol=1e-8)
    means = prediction['location'][None, :]+W[:, None]*prediction['conditional_beta']
    scores, validation, point_scores = {}, dict(quadrature=diagnostics), {}
    for name in ('f', 'y'):
        truth = data[name][test]
        dist = marginal_distribution(prediction, name)
        mean, ci = prediction[f'{name}_mean'][test], prediction[f'{name}_ci'][test]
        logpdf = dist.logpdf(data[name])[test]
        variance = W[:, None]*np.diag(prediction[f'{name}_scale_cov'])[None, :]
        mixture_logpdf = special.logsumexp(
            np.log(weights[:, None])+stats.norm.logpdf(data[name], means, np.sqrt(variance)), axis=0)
        np.testing.assert_allclose(mixture_logpdf[test], logpdf, atol=2e-7, rtol=1e-8)
        np.testing.assert_allclose(dist.mean(), prediction[f'{name}_mean'], atol=1e-9)
        np.testing.assert_allclose(dist.var(), prediction[f'{name}_var'], atol=1e-9)
        np.testing.assert_allclose(dist.cdf(prediction[f'{name}_ci'][:, 0]), .025, atol=1e-8)
        np.testing.assert_allclose(dist.cdf(prediction[f'{name}_ci'][:, 1]), .975, atol=1e-8)
        crps, crps_info = gaussian_mixture_crps(truth, means[:, test], variance[:, test], weights,
                                               atol=1e-9, rtol=1e-9)
        # Algebraic finite-mixture CRPS independently checks the CDF integration.
        selected = np.array([0, len(test)//2, len(test)-1])
        reference = gaussian_mixture_crps_closed_form(truth[selected], means[:, test[selected]],
                                                       variance[:, test[selected]], weights)
        np.testing.assert_allclose(crps[selected], reference, atol=1e-8, rtol=1e-8)
        scores.update({f'{name}_rmse': float(np.sqrt(np.mean((truth-mean)**2))),
                       f'{name}_mean_marginal_nlpd': float(-logpdf.mean()),
                       f'{name}_crps': float(crps.mean()),
                       f'{name}_coverage_95': float(np.mean((truth >= ci[:, 0]) & (truth <= ci[:, 1]))),
                       f'{name}_mean_interval_width': float(np.mean(ci[:, 1]-ci[:, 0]))})
        validation[name] = dict(max_logpdf_quadrature_error=float(np.max(np.abs(mixture_logpdf[test]-logpdf))),
                                max_crps_reference_error=float(np.max(np.abs(crps[selected]-reference))),
                                crps_integration=crps_info)
        point_scores.update({f'{name}_nlpd': -logpdf, f'{name}_crps': crps})
    return scores, validation, point_scores


def write_json(path, value):
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        raise TypeError(type(obj).__name__)
    path.write_text(json.dumps(value, indent=2, default=convert, allow_nan=False)+'\n', encoding='utf-8')


def run(source, output, starts):
    output.mkdir(parents=True, exist_ok=True)
    dataset_path = source/'sparse/dataset.npz'
    with np.load(dataset_path) as saved:
        data = {key: saved[key].copy() for key in saved.files}
    train, test = data['train_ids'], data['test_ids']
    if len(train) != 30 or len(test) != 151 or np.intersect1d(train, test).size:
        raise ValueError('Expected the saved 30/151 split')
    # Copy without changing any generated data, including the original noise.
    np.savez_compressed(output/'dataset.npz', **data)
    write_json(output/'experiment.json', dict(data_seed=5, source_dataset=str(dataset_path),
               source_sha256=hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
               n_train=len(train), n_test=len(test), lambda_fixed=1., omega_fixed=1., nu_fixed=5.,
               estimation='log-coordinate theta/W MAP, then conditional GP mean; no second refit',
               prediction='fix theta and estimated function, recompute and integrate posterior W',
               uses_hmc=False, uses_true_functions_for_fitting=False))
    rows, parameter_rows, predictions = [], [], {}
    for model in MODELS:
        print(f'{model}: fitting {starts} starts', flush=True)
        q, report = fit_point_estimate(data['y'][train], data['x'][train], model=model, starts=starts)
        estimate, full = estimate_function(data['y'][train], data['x'][train], data['x'], q, model)
        values = np.exp(q)
        kwargs = (dict(beta_train=estimate[train], beta_pred=estimate) if model == 'beta_gp'
                  else dict(mu_train=estimate[train], mu_pred=estimate))
        mixing = 'gp' if model == 'gp' else 'ig' if model == 'tp_mu_gp' else 'gig'
        prediction = fixed_function_prediction(data['y'][train], data['x'][train], data['x'],
                         v0=values[0], k=values[2], w=values[3], mixing=mixing, **kwargs)
        # At the original MAP W, plugging in the conditional GP mean must
        # preserve the full Gaussian conditional f mean exactly.
        map_w = 1. if model == 'gp' else values[5]
        np.testing.assert_allclose(prediction['location']+map_w*prediction['conditional_beta'],
                                   full['f_mean'], atol=2e-10, rtol=1e-10)
        covariance_difference = full['f_cov']-map_w*prediction['f_scale_cov']
        min_eig = np.linalg.eigvalsh((covariance_difference+covariance_difference.T)/2).min()
        if min_eig < -1e-9:
            raise AssertionError('Fixing the function increased same-W conditional uncertainty')
        scores, validation, point_scores = score_and_validate(data, prediction)
        report.update(physical_log_parameters=q, W_at_estimation_mode=map_w,
                      scores=scores, validation=validation,
                      covariance_difference_min_eigenvalue=float(min_eig),
                      W_prediction_mean=prediction['W_mean'], W_prediction_var=prediction['W_var'])
        write_json(output/f'{model}_report.json', report)
        np.savez_compressed(output/f'{model}_predictions.npz', **prediction,
                            fixed_function=estimate, physical_log_parameters=q, **point_scores)
        predictions[model] = prediction
        rows.append(dict(model=model, data_seed=5, n_train=len(train), n_test=len(test), **scores))
        parameter_rows.append(dict(model=model, v_0=values[0], latent_v0=values[1], k=values[2],
                                   w=values[3], latent_w=values[4], W_at_estimation_mode=map_w,
                                   W_prediction_mean=prediction['W_mean'], W_prediction_var=prediction['W_var']))
        print(f'{model}: {scores}', flush=True)
    scores = pd.DataFrame(rows)
    scores.to_csv(output/'scores.csv', index=False)
    pd.DataFrame(parameter_rows).to_csv(output/'parameters.csv', index=False)
    full = pd.read_csv(source/'scores_with_crps.csv')
    full = full[full.model.isin(MODELS)].copy()
    full['method'] = 'Full Bayesian (existing HMC)'
    comparison = pd.concat([full, scores.assign(method='Fixed theta/function, W integrated')], ignore_index=True)
    columns = ['method', 'model', 'f_rmse', 'f_mean_marginal_nlpd', 'f_crps',
               'y_rmse', 'y_mean_marginal_nlpd', 'y_crps', 'f_coverage_95', 'y_coverage_95',
               'f_mean_interval_width', 'y_mean_interval_width']
    comparison[columns].to_csv(output/'comparison_with_full_bayesian.csv', index=False)
    lines = [r'\begin{table}[htbp]', r'\centering', r'\small',
             r'\caption{Seed 5: 30 training observations and 151 held-out points. Point estimates of kernels and GP-prior mean/skew functions; mixing $W$ integrated in prediction.}',
             r'\label{tab:seed5_fixed_functions}', r'\begin{tabular}{lrrrrrr}', r'\toprule',
             r' & \multicolumn{3}{c}{Latent $f$} & \multicolumn{3}{c}{Observation $y$} \\',
             r'\cmidrule(lr){2-4}\cmidrule(lr){5-7}', r'Model & RMSE & NLPD & CRPS & RMSE & NLPD & CRPS \\', r'\midrule']
    labels = [r'GP (fixed $\hat\mu$)', r'STP (fixed $\hat\mu$)',
              r'GHP (fixed $\hat\mu$, $\beta=0$)', r'GHP (fixed $\hat\beta$, $\mu=0$)']
    for label, row in zip(labels, rows):
        lines.append(label+' & '+' & '.join(f'{row[c]:.5f}' for c in columns[2:8])+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (output/'comparison.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    fig, axes = plt.subplots(4, 2, figsize=(12, 13), sharex=True)
    for row, model in enumerate(MODELS):
        p = predictions[model]
        for col, name in enumerate(('f', 'y')):
            ax = axes[row, col]
            ax.fill_between(data['x'], *p[f'{name}_ci'].T, color='C0', alpha=.23, label='95% pointwise interval')
            ax.plot(data['x'], p[f'{name}_mean'], color='C0', label='Prediction')
            ax.plot(data['x'], data['f'], color='black', ls='--', lw=1, label='True f')
            ax.scatter(data['x'][train], data['y'][train], s=15, color='C3', label='Training y', zorder=3)
            if name == 'y':
                ax.scatter(data['x'][test], data['y'][test], s=8, color='gray', alpha=.4, label='Test y')
            ax.set_title(LABELS[model]+f': {name}')
            ax.set_xlabel('x')
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    fig.suptitle('Seed 5: fixed parameter/function estimates; W integrated')
    fig.tight_layout()
    fig.savefig(output/'regression.png', dpi=170)
    fig.savefig(output/'regression.pdf')
    plt.close(fig)
    print(scores.to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path('figure/single_function_30_hmc'))
    parser.add_argument('--output', type=Path, default=Path('figure/single_function_30_map_fixed_functions_seed5'))
    parser.add_argument('--starts', type=int, default=24)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.source, args.output, args.starts)
