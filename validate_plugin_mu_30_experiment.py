"""Independent checks of all fixed-estimated-mu experiments."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from scipy import linalg, special, stats
from threadpoolctl import threadpool_limits

from GH_zero_mean_hmc import PLUGIN_MU_MODELS, coordinates, prior_center_from_training
from GH_proc_w_betaprior_hmc import chain_diagnostics
from hmc_1d_split_experiments import write_json
from hmc_plugin_mu_30_experiment import SOURCES
from summarize_1d_training_comparison import validate_mixture


def validate_case(root, model):
    root = Path(root)
    case, source = root/'plugin_mu'/model, root/'sparse'
    for name, expected in json.loads((case/'source_provenance.json').read_text()).items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected
    with np.load(case/'dataset.npz') as saved:
        data = {k: saved[k] for k in saved.files}
    with np.load(source/'dataset.npz') as original:
        for key in original.files:
            # Some source experiments also save E_prior[W]*beta_true as
            # mu_fixed. This derived field is deliberately replaced by the
            # inferred mu here, independently reconstructed below.
            if key == 'mu_fixed':
                continue
            assert_array_equal(data[key], original[key])
    train, test = data['train_ids'], data['test_ids']
    x, y, mu = data['x'], data['y'][train], data['mu_fixed']
    assert_array_equal(np.sort(np.r_[train, test]), np.arange(len(x)))
    assert not np.intersect1d(train, test).size
    source_report = json.loads((source/f'{SOURCES[model]}_report.json').read_text())
    with np.load(source/source_report['selected_samples_file']) as saved:
        source_q = saved['log_samples']
    distances = (x[:, None]-x[None, :])**2
    expected_mu = np.zeros(len(x))
    # Reconstruct the entire training-data mean estimate without the estimator.
    for values in np.exp(source_q.reshape(-1, source_q.shape[-1])):
        v, a, k, w, mw = values[:5]
        W = 1. if len(values) == 5 else values[5]
        K = v*np.exp(-.5*w*distances[np.ix_(train, train)])
        Sigma = a*np.exp(-.5*mw*distances[:, train])
        C = Sigma[train]+W*(K+k*np.eye(len(train)))
        expected_mu += Sigma@np.linalg.solve(C, y)/(source_q.shape[0]*source_q.shape[1])
    assert_allclose(mu, expected_mu, atol=1e-10)
    if SOURCES[model] != 'gp':
        with np.load(source/f'{SOURCES[model]}_predictions.npz') as original:
            assert_allclose(mu, original['mu_mean'], atol=1e-11)
    report = json.loads((case/'report.json').read_text())
    assert_allclose(report['prior_center'], prior_center_from_training(y))
    assert_allclose(report['prior_center'], np.asarray(source_report['prior_center'])[[0, 2, 3]])
    assert_allclose(report['prior_sd'], 2.)
    with np.load(case/report['selected_samples_file']) as samples:
        q, z = samples['log_samples'], samples['effective_log_samples']
        logp = samples['log_density'].reshape(-1)
        divergent = samples['divergent']
        assert_array_equal(samples['mu_train'], mu[train])
    dimension = 3 if model == 'gp_plugin_mu' else 4
    assert q.shape == (report['settings']['chains'], report['settings']['draws'], dimension)
    assert_allclose(coordinates(q, model), z, atol=1e-12)
    for values, key in [(q, 'diagnostics'), (z, 'effective_diagnostics')]:
        computed = chain_diagnostics(values)
        for name in computed:
            assert_allclose(computed[name], report[key][name], atol=1e-10)
    assert int(divergent.sum()) == report['divergences']
    assert report['diagnostics_ok'], 'HMC diagnostics need attention'
    with np.load(case/'predictions.npz') as saved:
        p = {k: saved[k] for k in saved.files}
    validate_mixture(p, 'gp')
    flat = q.reshape(-1, dimension)
    assert int(p['draws_used']) == len(flat)
    assert_allclose(p['weights'], 1./len(flat))
    assert_array_equal(p['mu_fixed'], mu)
    assert_array_equal(p['mu_mean'], mu)
    for key in ['mu_var', 'mu_cov', 'effect_cov']:
        assert_allclose(p[key], 0.)
    W = np.ones(len(flat)) if dimension == 3 else np.exp(flat[:, -1])
    assert_allclose(p['W_mean'], W.mean())
    assert_allclose(p['W_var'], W.var())
    noise = W*np.exp(flat[:, 1])
    assert_allclose(p['component_y_variances']-p['component_f_variances'],
                    np.broadcast_to(noise[:, None], p['component_y_variances'].shape), atol=1e-12)
    assert_allclose(p['y_cov']-p['f_cov'], noise.mean()*np.eye(len(x)), atol=1e-11)
    within = np.zeros_like(p['f_cov'])
    density_indices = {c*q.shape[1]+j for c in range(q.shape[0])
                       for j in np.linspace(0, q.shape[1]-1, 8, dtype=int)}
    for index, values in enumerate(np.exp(flat)):
        v, k, w = values[:3]
        scale = W[index]
        K = scale*v*np.exp(-.5*w*distances)
        C = K[np.ix_(train, train)]+scale*k*np.eye(len(train))
        cross = K[:, train]
        L = linalg.cholesky(C, lower=True, check_finite=False)
        m = mu+cross@linalg.cho_solve((L, True), y-mu[train], check_finite=False)
        covariance = K-cross@linalg.cho_solve((L, True), cross.T, check_finite=False)
        assert_allclose(p['component_means'][index], m, atol=1e-10)
        assert_allclose(p['component_f_variances'][index], np.diag(covariance), atol=1e-10)
        within += covariance/len(flat)
        if index in density_indices:
            density = stats.multivariate_normal.logpdf(y, mean=mu[train], cov=C)
            density += stats.norm.logpdf(flat[index, :3], report['prior_center'], report['prior_sd']).sum()
            if model == 'stp_plugin_mu':
                density += stats.invgamma.logpdf(scale, 2.5, scale=1.5)+np.log(scale)
            elif model == 'ghp_plugin_mu':
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
    result = dict(model=model, validation_passed=True, previous_results_unchanged=True,
                  same_30_151_split=True, mu_hat_reconstructed_from_training_y_only=True,
                  source_components_checked=source_q.shape[0]*source_q.shape[1],
                  mu_uncertainty_zero=True, original_remaining_parameter_priors_preserved=True,
                  all_conditional_components_checked=len(flat), posterior_density_checks=len(density_indices),
                  full_covariance_checked=True, mixture_intervals_and_scores_checked=True,
                  **{key: report[key] for key in ['diagnostics_ok', 'max_rhat', 'min_bulk_ess',
                                                 'min_tail_ess', 'divergences']})
    write_json(case/'validation.json', result)
    print(json.dumps(result, indent=2), flush=True)
    return result


def validate(root='figure/single_function_30_hmc'):
    root = Path(root)
    results = [validate_case(root, model) for model in PLUGIN_MU_MODELS]
    previous = pd.read_csv(root/'scores_with_plugin_beta.csv')
    combined = pd.read_csv(root/'scores_with_plugin_mu.csv')
    pd.testing.assert_frame_equal(previous, combined.iloc[:len(previous)][previous.columns].reset_index(drop=True))
    result = dict(validation_passed=True, all_prior_scores_preserved=True, cases=results)
    write_json(root/'plugin_mu'/'validation.json', result)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='figure/single_function_30_hmc')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        validate(args.root)
