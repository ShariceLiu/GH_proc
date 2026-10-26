"""Independently validate the saved Airfoil split, posterior and predictions."""
import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
import pandas as pd
from scipy import linalg, special, stats
from threadpoolctl import threadpool_limits

from GH_tabular_hmc import MODELS, ZERO_MODELS, has_mixing, coordinates, default_center
from GH_proc_w_betaprior_hmc import chain_diagnostics
from hmc_airfoil_experiment import write_json
from load_real_datasets import load_datasets


def validate(root='figure/airfoil_single_function_hmc', *, models=MODELS, output_path=None, model_root=None):
    root = Path(root)
    with np.load(root/'dataset.npz') as saved:
        d = {k: saved[k] for k in saved.files}
    X, Y = load_datasets(['airfoil'])['airfoil']
    assert_array_equal(d['x'], X.to_numpy())
    assert_array_equal(d['y'], Y.to_numpy())
    train, test = d['train_ids'], d['test_ids']
    assert_array_equal(np.sort(np.r_[train, test]), np.arange(len(Y)))
    assert len(np.intersect1d(train, test)) == 0
    order = np.random.default_rng(int(d['split_seed'])).permutation(len(Y))
    assert_array_equal(train, np.sort(order[:len(train)]))
    assert_allclose(d['x_mean'], d['x'][train].mean(axis=0), atol=1e-12)
    assert_allclose(d['x_scale'], d['x'][train].std(axis=0), atol=1e-12)
    assert_allclose(d['y_mean'], d['y'][train].mean(), atol=1e-12)
    assert_allclose(d['y_scale'], d['y'][train].std(), atol=1e-12)
    assert_allclose(d['x_scaled'], (d['x']-d['x_mean'])/d['x_scale'])
    assert_allclose(d['y_scaled'], (d['y']-d['y_mean'])/d['y_scale'])
    x, y = d['x_scaled'][train], d['y_scaled'][train]
    results = []
    for model in models:
        folder = (root if model_root is None else Path(model_root))/model
        report = json.loads((folder/'report.json').read_text())
        with np.load(folder/report['selected_samples_file']) as saved:
            q, z = saved['log_samples'], saved['effective_log_samples']
            divergent = saved['divergent']
            saved_density = saved['log_density'].reshape(-1)
        assert q.shape[:2] == (report['settings']['chains'], report['settings']['draws'])
        assert_allclose(coordinates(q, model), z, atol=1e-12)
        assert_allclose(report['prior_center'], default_center(y, x.shape[1], model))
        for actual, key in [(q, 'diagnostics'), (z, 'effective_diagnostics')]:
            diagnostic = chain_diagnostics(actual)
            for name in diagnostic:
                assert_allclose(diagnostic[name], report[key][name], atol=1e-10)
        assert int(divergent.sum()) == report['divergences']
        with np.load(folder/'predictions.npz') as saved:
            p = {k: saved[k] for k in saved.files}
        assert_array_equal(p['test_ids'], test)
        means, fv, yv = p['component_means'], p['component_f_variances'], p['component_y_variances']
        flat = q.reshape(-1, q.shape[-1])
        assert means.shape == (len(flat), len(test))
        assert np.all(np.isfinite(means)) and np.all(fv >= 0) and np.all(yv > 0)
        assert_allclose(p['weights'], np.full(len(flat), 1./len(flat)))
        effective = np.exp(z.reshape(-1, z.shape[-1]))
        noise_index = 1 if model in ZERO_MODELS else 2
        assert_allclose(yv-fv, np.broadcast_to(effective[:, noise_index, None]*d['y_scale']**2, yv.shape), atol=1e-11)
        mean = means.mean(axis=0)
        for name, v in [('f', fv), ('y', yv)]:
            assert_allclose(p[f'{name}_mean'], mean, atol=1e-11)
            assert_allclose(p[f'{name}_var'], v.mean(axis=0)+means.var(axis=0), atol=1e-10)
            assert_allclose(p[f'{name}_within_var'], v.mean(axis=0), atol=1e-10)
            assert_allclose(p[f'{name}_between_var'], means.var(axis=0), atol=1e-10)
            for start in range(0, len(test), 128):
                section = slice(start, start+128)
                for side, probability in enumerate([.025, .975]):
                    cdf = special.ndtr((p[f'{name}_ci'][section, side]-means[:, section])/np.sqrt(v[:, section])).mean(axis=0)
                    assert_allclose(cdf, probability, atol=2e-9)
        logp = special.logsumexp(-.5*(np.log(2*np.pi*yv)+(d['y'][test]-means)**2/yv), axis=0)-np.log(len(flat))
        assert_allclose(report['y_mean_marginal_nlpd'], -logp.mean(), atol=1e-11)
        # The density transformation must include the target scale's Jacobian.
        vs = yv/d['y_scale']**2
        standardized_logp = special.logsumexp(-.5*(np.log(2*np.pi*vs)+
                                 (d['y_scaled'][test]-(means-d['y_mean'])/d['y_scale'])**2/vs), axis=0)-np.log(len(flat))
        assert_allclose(logp, standardized_logp-np.log(d['y_scale']), atol=1e-10)
        assert_allclose(report['y_rmse'], np.sqrt(np.mean((mean-d['y'][test])**2)))
        ci = p['y_ci']
        assert_allclose(report['y_coverage_95'], np.mean((d['y'][test] >= ci[:, 0]) & (d['y'][test] <= ci[:, 1])))
        assert_allclose(report['y_mean_interval_width'], np.mean(ci[:, 1]-ci[:, 0]))
        # Independent full block-Gaussian calculations at multiple draws from every chain.
        test_positions = np.unique(np.linspace(0, len(test)-1, 17, dtype=int))
        xp = d['x_scaled'][test[test_positions]]
        allx = np.vstack([x, xp])
        distances = (allx[:, None, :]-allx[None, :, :])**2
        indices = [c*q.shape[1]+j for c in range(q.shape[0])
                   for j in np.linspace(0, q.shape[1]-1, 6, dtype=int)]
        for index in indices:
            physical = np.exp(flat[index])
            W = physical[-1] if has_mixing(model) else 1.
            power = 2 if model == 'beta_gp' else 0
            dimension = x.shape[1]
            if model in ZERO_MODELS:
                size = 2+dimension
                kernel = W*physical[0]*np.exp(-.5*np.sum(distances*physical[2:size], axis=-1))
            else:
                size = 3+2*dimension
                kernel = W*physical[0]*np.exp(-.5*np.sum(distances*physical[3:3+dimension], axis=-1))
                kernel += W**power*physical[1]*np.exp(-.5*np.sum(distances*physical[3+dimension:size], axis=-1))
            C = kernel[:len(train), :len(train)]+W*physical[noise_index]*np.eye(len(train))
            cross = kernel[len(train):, :len(train)]
            L = linalg.cholesky(C, lower=True)
            cm = cross@linalg.cho_solve((L, True), y)
            cv = kernel[len(train):, len(train):]-cross@linalg.cho_solve((L, True), cross.T)
            assert_allclose(means[index, test_positions], d['y_mean']+d['y_scale']*cm, atol=2e-8)
            assert_allclose(fv[index, test_positions], d['y_scale']**2*np.diag(cv), atol=2e-8)
            # Independent normalized posterior, including the log-W Jacobian.
            density = -.5*len(train)*np.log(2*np.pi)-np.log(L.diagonal()).sum()-.5*y@linalg.cho_solve((L, True), y)
            density += stats.norm.logpdf(flat[index, :size], report['prior_center'], report['prior_sd']).sum()
            if model in ('tp_mu_gp', 'stp_zero'):
                density += stats.invgamma.logpdf(W, 2.5, scale=1.5)+np.log(W)
            elif has_mixing(model):
                density += stats.geninvgauss.logpdf(W, 1., 1.)+np.log(W)
            assert_allclose(density, saved_density[index], atol=2e-8)
        result = dict(model=model, draws=len(flat), test_points=len(test),
                      independent_block_checks=len(indices), diagnostics_ok=report['diagnostics_ok'],
                      max_rhat=report['max_rhat'], min_bulk_ess=report['min_bulk_ess'],
                      min_tail_ess=report['min_tail_ess'], divergences=report['divergences'],
                      validation_passed=True)
        results.append(result)
        print(json.dumps(result), flush=True)
    output = dict(validation_passed=True, all_hmc_diagnostics_ok=all(r['diagnostics_ok'] for r in results),
                  split_and_training_only_preprocessing_checked=True, scores_in_original_units=True,
                  mixture_quantiles_checked=True, posterior_densities_checked=True, models=results)
    write_json(root/'validation.json' if output_path is None else output_path, output)
    return output


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        validate()
