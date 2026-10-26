"""Check training-only beta estimate and every plug-in predictive component."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from scipy import linalg, special, stats
from threadpoolctl import threadpool_limits

from GH_proc_w_betaprior_hmc import chain_diagnostics
from GH_zero_mean_hmc import PLUGIN_BETA_MODEL, coordinates, prior_center_from_training
from hmc_1d_split_experiments import write_json
from summarize_1d_training_comparison import validate_mixture


def validate(root='figure/single_function_30_hmc'):
    root = Path(root)
    folder, source = root/'plugin_beta', root/'sparse'
    for name, expected in json.loads((folder/'source_provenance.json').read_text()).items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected
    with np.load(folder/'dataset.npz') as saved:
        data = {k: saved[k] for k in saved.files}
    with np.load(source/'dataset.npz') as original:
        for key in original.files:
            assert_array_equal(data['beta_true' if key == 'beta' else key], original[key])
    train, test = data['train_ids'], data['test_ids']
    x, y, beta = data['x'], data['y'][train], data['beta']
    assert_array_equal(np.sort(np.r_[train, test]), np.arange(len(x)))
    assert not np.intersect1d(train, test).size
    # Reconstruct beta_hat independently using only x and y_train. No beta/f
    # truth or held-out y enters this calculation or the fitted likelihood.
    source_report = json.loads((source/'beta_gp_report.json').read_text())
    with np.load(source/source_report['selected_samples_file']) as saved:
        source_q = saved['log_samples']
    d = (x[:, None]-x[None, :])**2
    estimated = np.zeros(len(x))
    for values in np.exp(source_q.reshape(-1, 6)):
        v, sv, k, w, sw, W = values
        K = v*np.exp(-.5*w*d[np.ix_(train, train)])
        Sigma = sv*np.exp(-.5*sw*d[:, train])
        C = W*K+W**2*Sigma[train]+W*k*np.eye(len(train))
        estimated += W*Sigma@np.linalg.solve(C, y)/(source_q.shape[0]*source_q.shape[1])
    assert_allclose(beta, estimated, atol=1e-10)
    with np.load(source/'beta_gp_predictions.npz') as p:
        assert_allclose(beta, p['beta_mean'], atol=1e-11)
    # The plug-in estimate must actually differ from the supplied true-beta case.
    assert np.max(np.abs(beta-data['beta_true'])) > 1e-3
    case = folder/PLUGIN_BETA_MODEL
    report = json.loads((case/'report.json').read_text())
    assert_allclose(report['prior_center'], prior_center_from_training(y))
    assert_allclose(report['prior_sd'], 2.)
    with np.load(case/report['selected_samples_file']) as samples:
        q, z = samples['log_samples'], samples['effective_log_samples']
        logp = samples['log_density'].reshape(-1)
        divergent = samples['divergent']
        assert_array_equal(samples['beta_train'], beta[train])
    assert q.shape == (report['settings']['chains'], report['settings']['draws'], 4)
    assert_allclose(coordinates(q, PLUGIN_BETA_MODEL), z, atol=1e-12)
    for values, key in [(q, 'diagnostics'), (z, 'effective_diagnostics')]:
        computed = chain_diagnostics(values)
        for name in computed:
            assert_allclose(computed[name], report[key][name], atol=1e-10)
    assert int(divergent.sum()) == report['divergences']
    assert report['diagnostics_ok'], 'HMC convergence needs attention'
    with np.load(case/'predictions.npz') as saved:
        p = {k: saved[k] for k in saved.files}
    validate_mixture(p, 'gp')
    flat = q.reshape(-1, 4)
    assert int(p['draws_used']) == len(flat)
    assert_allclose(p['weights'], 1./len(flat))
    assert_array_equal(p['beta_fixed'], beta)
    assert_array_equal(p['beta_mean'], beta)
    assert_allclose(p['beta_var'], 0.)
    assert_allclose(p['beta_cov'], 0.)
    W = np.exp(flat[:, -1])
    assert_allclose(p['W_mean'], W.mean())
    assert_allclose(p['W_var'], W.var())
    assert_allclose(p['effect_mean'], W.mean()*beta)
    assert_allclose(p['effect_cov'], W.var()*np.outer(beta, beta))
    effective_noise = np.exp(z.reshape(-1, 4)[:, 1])
    assert_allclose(p['component_y_variances']-p['component_f_variances'],
                    np.broadcast_to(effective_noise[:, None], p['component_y_variances'].shape), atol=1e-12)
    assert_allclose(p['y_cov']-p['f_cov'], effective_noise.mean()*np.eye(len(x)), atol=1e-11)
    within = np.zeros_like(p['f_cov'])
    density_indices = {c*q.shape[1]+j for c in range(q.shape[0])
                       for j in np.linspace(0, q.shape[1]-1, 8, dtype=int)}
    for index, values in enumerate(np.exp(flat)):
        v, k, w, scale = values
        K = scale*v*np.exp(-.5*w*d)
        C = K[np.ix_(train, train)]+scale*k*np.eye(len(train))
        cross = K[:, train]
        L = linalg.cholesky(C, lower=True, check_finite=False)
        expected_mean = scale*beta+cross@linalg.cho_solve((L, True), y-scale*beta[train], check_finite=False)
        expected_cov = K-cross@linalg.cho_solve((L, True), cross.T, check_finite=False)
        assert_allclose(p['component_means'][index], expected_mean, atol=1e-10)
        assert_allclose(p['component_f_variances'][index], np.diag(expected_cov), atol=1e-10)
        within += expected_cov/len(flat)
        if index in density_indices:
            density = stats.multivariate_normal.logpdf(y, mean=scale*beta[train], cov=C)
            density += stats.norm.logpdf(flat[index, :3], report['prior_center'], report['prior_sd']).sum()
            density += stats.geninvgauss.logpdf(scale, 1., 1.)+np.log(scale)
            assert_allclose(density, logp[index], atol=1e-9)
    assert_allclose(p['f_within_cov'], within, atol=1e-10)
    centered = p['component_means']-p['f_mean']
    between = centered.T@centered/len(flat)
    assert_allclose(p['f_between_cov'], between, atol=1e-11)
    assert_allclose(p['f_cov'], within+between, atol=1e-10)
    for target in ['f', 'y']:
        means = p['component_means'][:, test]
        variances = p[f'component_{target}_variances'][:, test]
        truth = data[target][test]
        density = -.5*(np.log(2*np.pi*variances)+(truth-means)**2/variances)
        nlpd = -np.mean(special.logsumexp(density, axis=0)-np.log(len(flat)))
        assert_allclose(nlpd, report['scores'][f'{target}_mean_marginal_nlpd'], atol=1e-11)
        rmse = np.sqrt(np.mean((p[f'{target}_mean'][test]-truth)**2))
        assert_allclose(rmse, report['scores'][f'{target}_rmse'], atol=1e-12)
        ci = p[f'{target}_ci'][test]
        assert_allclose(np.mean((truth >= ci[:, 0]) & (truth <= ci[:, 1])), report['scores'][f'{target}_coverage_95'])
        assert_allclose(np.mean(ci[:, 1]-ci[:, 0]), report['scores'][f'{target}_mean_interval_width'])
    original = pd.read_csv(root/'scores_with_fixed_mu.csv')
    combined = pd.read_csv(root/'scores_with_plugin_beta.csv')
    pd.testing.assert_frame_equal(original, combined.iloc[:len(original)][original.columns].reset_index(drop=True))
    result = dict(validation_passed=True, previous_results_unchanged=True, same_30_151_split=True,
                  beta_hat_reconstructed_using_training_y_only=True,
                  source_beta_components_checked=source_q.shape[0]*source_q.shape[1],
                  fixed_beta_differs_from_true_beta=True, beta_uncertainty_zero=True,
                  W_and_residual_uncertainty_retained=True, original_training_priors_preserved=True,
                  all_conditional_components_checked=len(flat), posterior_density_checks=len(density_indices),
                  full_covariance_checked=True, mixture_interval_probabilities_checked=True, scores_checked=True,
                  **{key: report[key] for key in ['diagnostics_ok', 'max_rhat', 'min_bulk_ess',
                                                 'min_tail_ess', 'divergences']})
    write_json(folder/'validation.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='figure/single_function_30_hmc')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        validate(args.root)
