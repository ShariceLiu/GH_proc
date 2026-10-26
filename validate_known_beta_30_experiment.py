"""Independent checks of the known-beta experiment and its Gaussian mixture."""
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import linalg, special, stats
from threadpoolctl import threadpool_limits

from GH_zero_mean_hmc import KNOWN_BETA_MODEL, coordinates, prior_center_from_training
from GH_proc_w_betaprior_hmc import chain_diagnostics
from hmc_1d_split_experiments import write_json
from hmc_known_beta_30_experiment import prepare
from summarize_1d_training_comparison import validate_mixture


def validate(root='figure/single_function_30_hmc'):
    root = Path(root)
    data, folder = prepare(root)
    for name, expected in json.loads((folder/'source_provenance.json').read_text()).items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected
    train, test = data['train_ids'], data['test_ids']
    x, y, beta = data['x'], data['y'][train], data['beta']
    assert_array_equal(np.sort(np.r_[train, test]), np.arange(len(x)))
    assert not np.intersect1d(train, test).size
    case = folder/KNOWN_BETA_MODEL
    report = json.loads((case/'report.json').read_text())
    assert_allclose(report['prior_center'], prior_center_from_training(y))
    assert_allclose(report['prior_sd'], 2.)
    with np.load(case/report['selected_samples_file']) as samples:
        q, z = samples['log_samples'], samples['effective_log_samples']
        logp = samples['log_density'].reshape(-1)
        divergent = samples['divergent']
        assert_array_equal(samples['beta_train'], beta[train])
    assert q.shape[:2] == (report['settings']['chains'], report['settings']['draws'])
    assert q.shape[-1] == 4
    assert_allclose(coordinates(q, KNOWN_BETA_MODEL), z, atol=1e-12)
    for values, key in [(q, 'diagnostics'), (z, 'effective_diagnostics')]:
        computed = chain_diagnostics(values)
        for name in computed:
            assert_allclose(computed[name], report[key][name], atol=1e-10)
    assert int(divergent.sum()) == report['divergences']
    with np.load(case/'predictions.npz') as saved:
        p = {k: saved[k] for k in saved.files}
    validate_mixture(p, 'gp')
    flat = q.reshape(-1, 4)
    assert int(p['draws_used']) == len(flat)
    assert_allclose(p['weights'], 1./len(flat))
    assert_array_equal(p['beta_known'], beta)
    W = np.exp(flat[:, -1])
    assert_allclose(p['W_mean'], W.mean())
    assert_allclose(p['W_var'], W.var())
    assert_allclose(p['effect_mean'], W.mean()*beta)
    assert_allclose(p['effect_cov'], W.var()*np.outer(beta, beta))
    effective_noise = np.exp(z.reshape(-1, 4)[:, 1])
    assert_allclose(p['component_y_variances']-p['component_f_variances'],
                    np.broadcast_to(effective_noise[:, None], p['component_y_variances'].shape), atol=1e-12)
    assert_allclose(p['y_cov']-p['f_cov'], effective_noise.mean()*np.eye(len(x)), atol=1e-11)
    # Reconstruct every conditional mean/variance and the full averaged covariance
    # using block conditioning with a nonzero known mean. No prediction helper.
    distances = (x[:, None]-x[None, :])**2
    within = np.zeros_like(p['f_cov'])
    density_indices = {c*q.shape[1]+j for c in range(q.shape[0])
                       for j in np.linspace(0, q.shape[1]-1, 8, dtype=int)}
    for index, values in enumerate(np.exp(flat)):
        v, k, w, scale = values
        K = scale*v*np.exp(-.5*w*distances)
        C = K[np.ix_(train, train)]+scale*k*np.eye(len(train))
        cross = K[:, train]
        residual = y-scale*beta[train]
        L = linalg.cholesky(C, lower=True, check_finite=False)
        expected_mean = scale*beta+cross@linalg.cho_solve((L, True), residual, check_finite=False)
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
    result = dict(validation_passed=True, previous_results_unchanged=True, same_30_151_split=True,
                  true_beta_supplied_at_training_and_prediction_inputs=True,
                  W_and_residual_parameters_inferred=True, all_conditional_components_checked=len(flat),
                  posterior_density_checks=len(density_indices), full_covariance_checked=True,
                  mixture_interval_probabilities_checked=True, scores_checked=True,
                  **{key: report[key] for key in ['diagnostics_ok', 'max_rhat', 'min_bulk_ess',
                                                 'min_tail_ess', 'divergences']})
    write_json(folder/'validation.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        validate()
