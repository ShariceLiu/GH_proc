"""Reconstruct saved fixed-function predictions and check independent GH CDFs."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg, special
from threadpoolctl import threadpool_limits

from GH_shared_function_map import MODELS, shared_calculation, gig_mean
from GH_fixed_function import mixing_quadrature
from shared_beta_plugin_experiment import ORACLES, matched_gp_prediction, write_json


def validate(root):
    with np.load(root/'dataset.npz') as saved:
        data = {key: saved[key].copy() for key in saved.files}
    train, functions, x = data['train_ids'], data['train_functions'], data['x']
    assert not np.intersect1d(functions, data['test_functions']).size
    checks, max_cdf_error, max_moment_error = 0, 0., 0.
    for model in (*MODELS, *ORACLES):
        fit = json.loads((root/model/'fit.json').read_text()) if model in MODELS else None
        if fit is not None:
            q = np.asarray(fit['log_parameters'])
            recomputed = shared_calculation(data['y'][functions][:, train], x[train], q, model, xp=x)
            expected_function = recomputed['h_pred']/gig_mean() if model == 'ghp_beta' else recomputed['h_pred']
            np.testing.assert_allclose(expected_function, fit['function'], atol=1e-10)
        for count in (0, 5, 15):
            for j, function in enumerate(data['test_functions']):
                with np.load(root/model/f'context_{count}/function_{j:02d}.npz') as saved:
                    p = {key: saved[key].copy() for key in saved.files}
                p['mixing'] = str(p['mixing'])
                ids, test = p['context_ids'], p['test_ids']
                np.testing.assert_array_equal(ids, np.sort(data['context_orders'][j, :count]))
                np.testing.assert_array_equal(test, np.setdiff1d(np.arange(len(x)), ids))
                if model == 'gp_matched_oracle':
                    reference = matched_gp_prediction(x, data['y'][function, ids], ids, data['beta'])
                    for name in ('f', 'y'):
                        np.testing.assert_allclose(reference[f'{name}_mean'], p[f'{name}_mean'], atol=1e-10)
                        np.testing.assert_allclose(reference[f'{name}_cov'], p[f'{name}_cov'], atol=1e-10)
                else:
                    if model == 'ghp_oracle':
                        v, k, w, kind, estimated = .2, .05, 2., 'beta', data['beta']
                    else:
                        v, _, k, w, _ = np.exp(fit['log_parameters'][:5])
                        kind, estimated = fit['function_kind'], np.asarray(fit['function'])
                    beta = estimated if kind == 'beta' else np.zeros(len(x))
                    mu = estimated if kind == 'mu' else np.zeros(len(x))
                    K = v*np.exp(-.5*w*(x[:, None]-x[None, :])**2)
                    A = K[np.ix_(ids, ids)]+k*np.eye(count)
                    cross = K[:, ids]
                    r = data['y'][function, ids]-mu[ids]
                    B = linalg.solve(A, cross.T).T if count else np.empty((len(x), 0))
                    location = mu+B@r
                    b = beta-B@beta[ids]
                    S = K-B@cross.T
                    np.testing.assert_allclose(location, p['location'], atol=1e-9)
                    np.testing.assert_allclose(b, p['conditional_beta'], atol=1e-9)
                    np.testing.assert_allclose(S, p['f_scale_cov'], atol=1e-9)
                    np.testing.assert_allclose(p['y_scale_cov']-p['f_scale_cov'], k*np.eye(len(x)), atol=1e-10)
                    if p['mixing'] == 'gig':
                        chi = 1+r@linalg.solve(A, r) if count else 1.
                        psi = 1+beta[ids]@linalg.solve(A, beta[ids]) if count else 1.
                        np.testing.assert_allclose([p['W_lambda'], p['W_chi'], p['W_psi']], [1-count/2, chi, psi], atol=1e-9)
                if p['mixing'] == 'gig':
                    W, weights, _ = mixing_quadrature(p)
                    cm = p['location']+W[:, None]*p['conditional_beta']
                    mean = weights@cm
                    for name in ('f', 'y'):
                        cv = W[:, None]*np.diag(p[f'{name}_scale_cov'])
                        var = weights@(cv+(cm-mean)**2)
                        np.testing.assert_allclose(mean, p[f'{name}_mean'], atol=1e-8, rtol=1e-8)
                        np.testing.assert_allclose(var, p[f'{name}_var'], atol=1e-8, rtol=1e-8)
                        max_moment_error = max(max_moment_error, float(np.max(np.abs(var-p[f'{name}_var']))))
                        for endpoint, probability in [(0, .025), (1, .975)]:
                            cdf = weights@special.ndtr((p[f'{name}_ci'][:, endpoint]-cm)/np.sqrt(cv))
                            error = float(np.max(np.abs(cdf-probability)))
                            max_cdf_error = max(max_cdf_error, error)
                            np.testing.assert_allclose(cdf, probability, atol=2e-7)
                for name in ('f', 'y'):
                    truth = data[name][function, test]
                    report = json.loads((root/model/f'context_{count}/function_{j:02d}.json').read_text())['scores']
                    np.testing.assert_allclose(np.sqrt(np.mean((truth-p[f'{name}_mean'][test])**2)), report[f'{name}_rmse'], atol=1e-12)
                    np.testing.assert_allclose(np.mean(p[f'{name}_nlpd']), report[f'{name}_nlpd'], atol=1e-12)
                    np.testing.assert_allclose(np.mean(p[f'{name}_crps']), report[f'{name}_crps'], atol=1e-12)
                checks += 1
    result = dict(all_checks_passed=True, prediction_files_checked=checks,
                  shared_functions_reconstructed_from_training_only=True,
                  max_GH_interval_cdf_error=max_cdf_error, max_GH_variance_error=max_moment_error,
                  source_data_and_predictions_unchanged=True)
    write_json(root/'validation.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('figure/shared_beta_plugin_seed5'))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        validate(args.root)
