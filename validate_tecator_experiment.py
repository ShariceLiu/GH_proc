"""Independent split, conditional prediction, mixture, and score checks."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from scipy import linalg, special, stats
from threadpoolctl import threadpool_limits

from explore_tecator import load_tecator
from GH_multiple_hmc import make_target, coordinates, context_log_weights, MODELS
from hmc_tecator_experiment import DEFAULT_OUTPUT, PRIOR_CENTER, load_data, checked_weights
from hmc_multiple_functions_experiment import write_json


def validate_contexts(folder, models):
    """Compare spectral and direct new-curve likelihoods at four saved states.

    This checks numerical formulas, independent of whether these checkpoints
    have passed the final HMC diagnostic threshold.
    """
    data = load_data(folder)
    checks = []
    for model in models:
        report = json.loads((folder/model/'fit.json').read_text())
        with np.load(folder/model/report['selected_samples']) as saved:
            q = saved['theta_log_samples']
            theta = q[:, q.shape[1]//2]
        for j, row in enumerate(data['test_rows']):
            ids = data['context_ids'][j]
            x, y = data['x'][ids], data['residuals'][row, ids]
            t, spectral, quad_error = checked_weights(theta, x, y, model)
            _, direct = context_log_weights(theta, x, y[None], model=model,
                                             quadrature_nodes=len(t), backend='dense')
            difference = float(np.max(abs(special.logsumexp(spectral, axis=1)
                                           -special.logsumexp(direct[0], axis=1))))
            if difference > 1e-4:
                raise AssertionError(f'{model}, sample {row+1}: evidence error {difference}')
            checks.append(dict(model=model, sample_id=int(row+1), quadrature_nodes=len(t),
                               max_quadrature_log_error=quad_error,
                               max_spectral_dense_log_evidence_error=difference,
                               checkpoint=report['selected_samples']))
        print(f'{model}: all {len(data["test_rows"])} context likelihoods agree with direct Cholesky.', flush=True)
    write_json(folder/'context_validation.json', dict(passed=True, checks=checks))


def validate(folder, models=MODELS):
    data, raw = load_data(folder), load_tecator()
    samples = raw['samples']
    train, test = data['train_rows'], data['test_rows']
    assert_array_equal(data['y'], raw['spectra'])
    assert_allclose(data['common_mean'], data['y'][train].mean(axis=0), atol=1e-14)
    assert_allclose(data['common_scale'], np.std(data['y'][train]-data['common_mean']))
    compositions = samples[['moisture_pct', 'fat_pct', 'protein_pct']].to_numpy()
    assert len(set(map(tuple, compositions[train]))) == len(train)
    assert len(set(map(tuple, compositions[test]))) == len(test)
    assert not set(map(tuple, compositions[train])).intersection(map(tuple, compositions[test]))
    assert not np.intersect1d(samples.spectrum_group_id.iloc[train],
                             samples.spectrum_group_id.iloc[test]).size
    for context, evaluation in zip(data['context_ids'], data['evaluation_ids']):
        assert not np.intersect1d(context, evaluation).size
        assert_array_equal(np.union1d(context, evaluation), np.arange(100))
    checks = []
    for model in models:
        report = json.loads((folder/model/'fit.json').read_text())
        assert report['diagnostics_ok']
        with np.load(folder/model/report['selected_samples']) as posterior:
            q = posterior['log_samples'][0, 0]
        groups = [(data['x'], data['residuals'][train])]
        z = coordinates(q, model)
        dense = make_target(groups, model=model, prior_center=PRIOR_CENTER)
        spectral = make_target(groups, model=model, prior_center=PRIOR_CENTER, backend='shared_grid')
        dense_value, dense_grad = dense(z)
        spectral_value, spectral_grad = spectral(z)
        assert_allclose(spectral_value, dense_value, atol=.003, rtol=1e-8)
        assert_allclose(spectral_grad, dense_grad, atol=.05, rtol=.001)
        score_rows = pd.read_csv(folder/model/'per_function_scores.csv')
        assert_array_equal(score_rows.sample_id, test+1)
        max_mean_error, max_variance_error, max_quantile_error = 0., 0., 0.
        for j, row in enumerate(test):
            with np.load(folder/model/f'sample_{row+1:03d}.npz') as pred:
                context, evaluation = data['context_ids'][j], data['evaluation_ids'][j]
                assert_array_equal(pred['context_ids'], context)
                assert_array_equal(pred['evaluation_ids'], evaluation)
                x = data['x'][context]
                y = data['residuals'][row, context]
                count = len(pred['posterior_log_parameters'])
                for index in [0, count//2, count-1]:
                    q = pred['posterior_log_parameters'][index]
                    v, a, k, w, sw = np.exp(q[:5])
                    W = 1. if model == 'gp' else np.exp(q[5])
                    A, B, noise = W*v, W**(2 if model == 'beta_gp' else 0)*a, W*k
                    def kernel(left, right):
                        distance = (left[:, None]-right[None, :])**2
                        return A*np.exp(-.5*w*distance)+B*np.exp(-.5*sw*distance)
                    C = kernel(x, x)+noise*np.eye(len(x))
                    cross = kernel(data['x'], x)
                    chol = linalg.cho_factor(C, lower=True)
                    mean = data['common_mean']+data['common_scale']*cross@linalg.cho_solve(chol, y)
                    variance = data['common_scale']**2*(A+B+noise-np.sum(
                        cross*linalg.cho_solve(chol, cross.T).T, axis=1))
                    max_mean_error = max(max_mean_error, float(np.max(abs(mean-pred['component_means'][index]))))
                    max_variance_error = max(max_variance_error, float(np.max(abs(variance-pred['component_y_variances'][index]))))
                    assert_allclose(mean, pred['component_means'][index], atol=2e-7, rtol=1e-7)
                    assert_allclose(variance, pred['component_y_variances'][index], atol=2e-8, rtol=1e-5)
                means, variances = pred['component_means'], pred['component_y_variances']
                weights = pred['weights']
                assert_allclose(weights.sum(), 1.)
                expected_mean = weights@means
                expected_variance = weights@(variances+(means-expected_mean)**2)
                assert_allclose(pred['y_mean'], expected_mean, atol=1e-11)
                assert_allclose(pred['y_var'], expected_variance, atol=1e-11)
                for side, probability in enumerate([.025, .975]):
                    cdf = weights@special.ndtr((pred['y_ci'][:, side]-means)/np.sqrt(variances))
                    error = float(np.max(abs(cdf-probability)))
                    max_quantile_error = max(max_quantile_error, error)
                    assert_allclose(cdf, probability, atol=2e-6)
                truth = data['y'][row, evaluation]
                logpdf = stats.norm.logpdf(truth, loc=means[:, evaluation], scale=np.sqrt(variances[:, evaluation]))
                nlpd = -special.logsumexp(logpdf+np.log(weights[:, None]), axis=0)
                se = (truth-expected_mean[evaluation])**2
                assert_allclose(pred['nlpd'], nlpd, atol=2e-8)
                assert_allclose(pred['squared_error'], se, atol=1e-12)
                assert_allclose(score_rows.iloc[j].y_nlpd, nlpd.mean(), atol=2e-8)
                assert_allclose(score_rows.iloc[j].y_rmse, np.sqrt(se.mean()), atol=1e-10)
                assert_allclose(score_rows.iloc[j].y_crps, pred['crps'].mean(), atol=1e-10)
                assert np.all(pred['crps'] >= 0)
        checks.append(dict(model=model, n_test_functions=len(test),
                           spectral_dense_log_density_difference=float(spectral_value-dense_value),
                           max_conditional_mean_error=max_mean_error,
                           max_conditional_variance_error=max_variance_error,
                           max_mixture_quantile_probability_error=max_quantile_error,
                           passed=True))
    if set(models) == set(MODELS):
        summary = pd.read_csv(folder/'scores.csv')
        individual = pd.read_csv(folder/'per_function_scores.csv')
        for model in MODELS:
            rows = individual.loc[individual.model.eq(model)]
            row = summary.loc[summary.model.eq(model)].iloc[0]
            assert_allclose(row.y_rmse, np.sqrt(np.mean(rows.y_rmse**2)))
            for field in ['y_nlpd', 'y_crps', 'y_coverage_95', 'y_interval_width']:
                assert_allclose(row[field], rows[field].mean())
    filename = 'validation.json' if set(models) == set(MODELS) else 'validation_'+'_'.join(models)+'.json'
    write_json(folder/filename, dict(passed=True, checks=checks,
               split_and_training_only_preprocessing=True, original_unit_metrics=True))
    print(json.dumps(checks, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    parser.add_argument('--contexts-only', action='store_true')
    args = parser.parse_args()
    with threadpool_limits(1):
        if args.contexts_only:
            validate_contexts(args.output, args.models)
        else:
            validate(args.output, args.models)
