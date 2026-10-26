"""Independent validation of the added zero-mean fits and saved predictions."""
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import linalg, special, stats
from threadpoolctl import threadpool_limits

from GH_zero_mean_hmc import MODELS, coordinates, prior_center_from_training
from GH_proc_w_betaprior_hmc import chain_diagnostics
from hmc_1d_split_experiments import write_json
from hmc_zero_mean_30_experiment import prepare
from summarize_1d_training_comparison import validate_mixture


def validate(root='figure/single_function_30_hmc'):
    root = Path(root)
    data, folder = prepare(root)
    hashes = json.loads((folder/'source_provenance.json').read_text())
    for name, expected in hashes.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected
    train, test = data['train_ids'], data['test_ids']
    assert_array_equal(np.sort(np.r_[train, test]), np.arange(len(data['x'])))
    assert not np.intersect1d(train, test).size
    x, y = data['x'][train], data['y'][train]
    rows = []
    for model in MODELS:
        model_folder = folder/model
        report = json.loads((model_folder/'report.json').read_text())
        assert_allclose(report['prior_center'], prior_center_from_training(y))
        assert_allclose(report['prior_sd'], 2.)
        with np.load(model_folder/report['selected_samples_file']) as saved:
            q, z = saved['log_samples'], saved['effective_log_samples']
            logp = saved['log_density'].reshape(-1)
            divergent = saved['divergent']
        assert q.shape[:2] == (report['settings']['chains'], report['settings']['draws'])
        assert_allclose(coordinates(q, model), z, atol=1e-12)
        for values, name in [(q, 'diagnostics'), (z, 'effective_diagnostics')]:
            diagnostic = chain_diagnostics(values)
            for key in diagnostic:
                assert_allclose(diagnostic[key], report[name][key], atol=1e-10)
        assert int(np.sum(divergent)) == report['divergences']
        with np.load(model_folder/'predictions.npz') as saved:
            p = {key: saved[key] for key in saved.files}
        # Same Gaussian-mixture representation as the previous GP prediction files.
        validate_mixture(p, 'gp')
        flat = q.reshape(-1, q.shape[-1])
        assert p['draws_used'] == len(flat)
        assert p['component_means'].shape == (len(flat), len(data['x']))
        assert_allclose(p['weights'], 1./len(flat))
        effective = np.exp(z.reshape(-1, z.shape[-1]))
        assert_allclose(p['component_y_variances']-p['component_f_variances'],
                        np.broadcast_to(effective[:, 1, None], p['component_y_variances'].shape), atol=1e-12)
        assert_allclose(p['y_cov']-p['f_cov'], effective[:, 1].mean()*np.eye(len(data['x'])), atol=1e-11)
        for target in ['f', 'y']:
            means = p['component_means'][:, test]
            variances = p[f'component_{target}_variances'][:, test]
            truth = data[target][test]
            density = -.5*(np.log(2*np.pi*variances)+(truth-means)**2/variances)
            nlpd = -np.mean(special.logsumexp(density, axis=0)-np.log(len(flat)))
            score = report['scores']
            assert_allclose(nlpd, score[f'{target}_mean_marginal_nlpd'], atol=1e-11)
            assert_allclose(np.sqrt(np.mean((p[f'{target}_mean'][test]-truth)**2)), score[f'{target}_rmse'])
            ci = p[f'{target}_ci'][test]
            assert_allclose(np.mean((truth >= ci[:, 0]) & (truth <= ci[:, 1])), score[f'{target}_coverage_95'])
            assert_allclose(np.mean(ci[:, 1]-ci[:, 0]), score[f'{target}_mean_interval_width'])
        # Direct full block Gaussian conditioning; no prediction helper is reused.
        indices = [c*q.shape[1]+j for c in range(q.shape[0])
                   for j in np.linspace(0, q.shape[1]-1, 8, dtype=int)]
        within_f = np.zeros_like(p['f_cov'])
        distances = (data['x'][:, None]-data['x'][None, :])**2
        for index in indices:
            values = np.exp(flat[index])
            v, k, w = values[:3]
            W = 1. if model == 'gp_zero' else values[-1]
            K = W*v*np.exp(-.5*w*distances)
            C = K[np.ix_(train, train)]+W*k*np.eye(len(train))
            cross = K[:, train]
            L = linalg.cholesky(C, lower=True)
            expected_mean = cross@linalg.cho_solve((L, True), y)
            expected_cov = K-cross@linalg.cho_solve((L, True), cross.T)
            assert_allclose(p['component_means'][index], expected_mean, atol=1e-10)
            assert_allclose(p['component_f_variances'][index], np.diag(expected_cov), atol=1e-10)
            density = stats.multivariate_normal.logpdf(y, cov=C)
            density += stats.norm.logpdf(flat[index, :3], report['prior_center'], report['prior_sd']).sum()
            if model == 'stp_zero':
                density += stats.invgamma.logpdf(W, 2.5, scale=1.5)+np.log(W)
            elif model == 'ghp_zero':
                density += stats.geninvgauss.logpdf(W, 1., 1.)+np.log(W)
            assert_allclose(density, logp[index], atol=1e-9)
        # Recompute the entire within-component covariance using independent solves.
        for values in np.exp(flat):
            v, k, w = values[:3]
            W = 1. if model == 'gp_zero' else values[-1]
            K = W*v*np.exp(-.5*w*distances)
            C = K[np.ix_(train, train)]+W*k*np.eye(len(train))
            cross = K[:, train]
            L = linalg.cholesky(C, lower=True, check_finite=False)
            within_f += (K-cross@linalg.cho_solve((L, True), cross.T, check_finite=False))/len(flat)
        assert_allclose(p['f_within_cov'], within_f, atol=1e-10)
        mean = p['weights']@p['component_means']
        centered = p['component_means']-mean
        assert_allclose(p['f_between_cov'], (centered.T*p['weights'])@centered, atol=1e-11)
        row = dict(model=model, validation_passed=True, draws=len(flat), conditional_checks=len(indices),
                   **{key: report[key] for key in ['max_rhat', 'min_bulk_ess', 'min_tail_ess',
                                                  'divergences', 'diagnostics_ok']})
        rows.append(row)
        print(json.dumps(row), flush=True)
    result = dict(validation_passed=True, original_source_files_unchanged=True,
                  exact_same_function_and_30_151_split=True, remaining_priors_match_original_residual_priors=True,
                  checks=['posterior density and coordinate transformation', 'HMC diagnostics',
                          'conditional Gaussian means and variances', 'full covariance law of total variance',
                          '95% mixture interval CDF probabilities', 'held-out f/y RMSE and NLPD'],
                  all_diagnostics_ok=all(row['diagnostics_ok'] for row in rows), models=rows)
    write_json(folder/'validation.json', result)
    return result


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        validate()
