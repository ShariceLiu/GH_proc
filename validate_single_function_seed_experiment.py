"""Validate a fresh single-function repetition, independently of prediction code."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import linalg, special, stats
from threadpoolctl import threadpool_limits

from fit_beta_mu_1d_experiment import simulate_1d
from GH_proc_w_betaprior_hmc import chain_diagnostics
from GH_proc_joint_hmc import effective_coordinates
from GH_zero_mean_hmc import coordinates, prior_center_from_training as one_center
from hmc_1d_split_experiments import make_split, prior_center_from_training as two_center, write_json
from hmc_single_function_seed_experiment import MODELS, load_data, paths
from summarize_1d_training_comparison import validate_mixture


def validate(root):
    root = Path(root)
    config = json.loads((root/'configuration.json').read_text())
    provenance = json.loads((root/'source_provenance.json').read_text())
    for name, digest in provenance['previous_experiment'].items():
        assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
    assert hashlib.sha256((root/'sparse'/'dataset.npz').read_bytes()).hexdigest() == provenance['new_dataset_sha256']
    data = load_data(root)
    with threadpool_limits(limits=1):
        reproduced = make_split(simulate_1d(seed=config['data_seed']), 'sparse', sparse_points=30)
    for key in ['x', 'y', 'f', 'beta', 'W', 'effect', 'train_ids', 'test_ids']:
        assert_array_equal(data[key], reproduced[key])
    assert_allclose(data['mu_fixed'], stats.geninvgauss.mean(1., 1.)*data['beta'], atol=1e-14)
    train, test = data['train_ids'], data['test_ids']
    assert len(train) == 30 and len(test) == 151
    assert_array_equal(np.sort(np.r_[train, test]), np.arange(len(data['x'])))
    x, y = data['x'], data['y'][train]
    distances = (x[:, None]-x[None, :])**2
    eye_train = np.eye(len(train))
    results = []
    for model in MODELS:
        folder, report_path, prediction_path = paths(root, model)
        report = json.loads(report_path.read_text())
        assert report['diagnostics_ok']
        center = two_center(y) if model in MODELS[:4] else one_center(y)
        assert_allclose(report['prior_center'], center, atol=1e-14)
        assert_allclose(report['prior_sd'], 2.)
        with np.load(folder/report['selected_samples_file']) as saved:
            q = saved['log_samples']
            logp = saved['log_density'].reshape(-1)
            div = saved['divergent']
            diagnostics = chain_diagnostics(q)
            for key, values in diagnostics.items():
                assert_allclose(values, report['diagnostics'][key], atol=1e-10)
            if model != 'gp':
                z = saved['effective_log_samples']
                expected_z = effective_coordinates(q, model) if model in MODELS[1:4] else coordinates(q, model)
                assert_allclose(z, expected_z, atol=1e-12)
                effective_diag = chain_diagnostics(z)
                for key, values in effective_diag.items():
                    assert_allclose(values, report['effective_diagnostics'][key], atol=1e-10)
            if model == 'ghp_known_beta':
                assert_array_equal(saved['beta_train'], data['beta'][train])
            if model == 'ghp_fixed_mu':
                assert_array_equal(saved['mu_train'], data['mu_fixed'][train])
        assert int(div.sum()) == report['scores']['divergences'] == 0
        assert q.shape[0] == config['hmc']['chains']
        flat = q.reshape(-1, q.shape[-1])
        with np.load(prediction_path) as saved:
            p = {key: saved[key] for key in saved.files}
        mixture_model = model if model in MODELS[1:4] else 'gp'
        validate_mixture(p, mixture_model)
        component_mean_key = 'component_means' if mixture_model == 'gp' else 'component_f_means'
        component_means = p[component_mean_key]
        assert component_means.shape == (len(flat), len(x))
        assert_allclose(p['weights'], 1./len(flat), atol=1e-15)
        if model in MODELS[1:4]:
            assert_allclose(component_means, p['component_y_means'])
        within = np.zeros((len(x), len(x)))
        noises = np.empty(len(flat))
        density_checks = {c*q.shape[1]+j for c in range(q.shape[0])
                          for j in np.linspace(0, q.shape[1]-1, 8, dtype=int)}
        for index, values in enumerate(np.exp(flat)):
            mean = np.zeros(len(x))
            if model in MODELS[:4]:
                v, a, k, w, latent_w = values[:5]
                W = 1. if model == 'gp' else values[-1]
                power = 2 if model == 'beta_gp' else 0
                K = W*v*np.exp(-.5*w*distances)+W**power*a*np.exp(-.5*latent_w*distances)
            else:
                v, k, w = values[:3]
                W = 1. if model == 'gp_zero' else values[-1]
                K = W*v*np.exp(-.5*w*distances)
                if model == 'ghp_known_beta':
                    mean = W*data['beta']
                elif model == 'ghp_fixed_mu':
                    mean = data['mu_fixed']
            noises[index] = W*k
            C = K[np.ix_(train, train)]+noises[index]*eye_train
            cross = K[:, train]
            L = linalg.cholesky(C, lower=True, check_finite=False)
            expected_mean = mean+cross@linalg.cho_solve((L, True), y-mean[train], check_finite=False)
            expected_cov = K-cross@linalg.cho_solve((L, True), cross.T, check_finite=False)
            assert_allclose(component_means[index], expected_mean, atol=2e-9)
            assert_allclose(p['component_f_variances'][index], np.diag(expected_cov), atol=2e-9)
            within += expected_cov/len(flat)
            if index in density_checks:
                density = stats.multivariate_normal.logpdf(y, mean=mean[train], cov=C)
                density += stats.norm.logpdf(flat[index, :len(center)], center, 2.).sum()
                if model in ['tp_mu_gp', 'stp_zero']:
                    density += stats.invgamma.logpdf(W, 2.5, scale=1.5)+np.log(W)
                elif model not in ['gp', 'gp_zero']:
                    density += stats.geninvgauss.logpdf(W, 1., 1.)+np.log(W)
                assert_allclose(density, logp[index], atol=1e-8)
        assert_allclose(p['f_within_cov'], within, atol=2e-9)
        centered = component_means-p['f_mean']
        between = centered.T@centered/len(flat)
        assert_allclose(p['f_between_cov'], between, atol=1e-10)
        assert_allclose(p['f_cov'], within+between, atol=2e-9)
        assert_allclose(p['component_y_variances']-p['component_f_variances'],
                        np.broadcast_to(noises[:, None], p['component_y_variances'].shape), atol=1e-10)
        assert_allclose(p['y_cov']-p['f_cov'], noises.mean()*np.eye(len(x)), atol=1e-10)
        for target in ['f', 'y']:
            truth, means = data[target][test], component_means[:, test]
            var = p[f'component_{target}_variances'][:, test]
            density = -.5*(np.log(2*np.pi*var)+(truth-means)**2/var)
            score = report['scores']
            nlpd = -np.mean(special.logsumexp(density, axis=0)-np.log(len(flat)))
            assert_allclose(nlpd, score[f'{target}_mean_marginal_nlpd'], atol=1e-10)
            assert_allclose(np.sqrt(np.mean((p[f'{target}_mean'][test]-truth)**2)), score[f'{target}_rmse'], atol=1e-12)
            ci = p[f'{target}_ci'][test]
            assert_allclose(np.mean((truth >= ci[:, 0]) & (truth <= ci[:, 1])), score[f'{target}_coverage_95'])
            assert_allclose(np.mean(ci[:, 1]-ci[:, 0]), score[f'{target}_mean_interval_width'])
        result = dict(model=model, validation_passed=True, all_components_checked=len(flat),
                      density_checks=len(density_checks), **{key: report['scores'][key] for key in
                      ['max_rhat', 'min_bulk_ess', 'min_tail_ess', 'divergences', 'diagnostics_ok']})
        results.append(result)
        print(json.dumps(result), flush=True)
    result = dict(validation_passed=True, data_seed=config['data_seed'],
                  generation_reproduced_exactly=True, split_matches_previous_seed=True,
                  previous_experiment_unchanged=True, training_only_priors_verified=True,
                  full_covariances_and_mixture_quantiles_verified=True, scores_verified=True,
                  models=results)
    write_json(root/'validation.json', result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('figure/single_function_30_hmc_seed6'))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        validate(args.root)
