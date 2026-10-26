"""Validate completed independent-function regression artifacts without refitting."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import special
from threadpoolctl import threadpool_limits

from hmc_1d_split_experiments import MODELS, prior_center_from_training, write_json
from hmc_gp_tp_1d_experiment import score_predictions
from summarize_1d_training_comparison import validate_mixture


def validate(root, *, watch=False):
    with np.load(root/'dataset.npz') as saved:
        data = {k: saved[k] for k in saved.files}
    checks = []
    for i in range(20):
        folder = root/f'function_{i:02d}'
        if watch:
            while True:
                try:
                    ready = all(json.loads((folder/f'{model}_report.json').read_text())['diagnostics_ok']
                                for model in MODELS)
                except (FileNotFoundError, json.JSONDecodeError):
                    ready = False
                if ready:
                    break
                time.sleep(10)
        with np.load(folder/'dataset.npz') as saved:
            local = {k: saved[k] for k in saved.files}
        assert_array_equal(local['x'], data['x'])
        for key in ('f', 'y', 'beta', 'W', 'train_ids', 'test_ids'):
            assert_array_equal(local[key], data[key][i])
        train, test = local['train_ids'], local['test_ids']
        assert len(train) == 30 and len(test) == 151
        assert not set(train)&set(test)
        assert_array_equal(np.sort(np.r_[train, test]), np.arange(181))
        for model in MODELS:
            report = json.loads((folder/f'{model}_report.json').read_text())
            assert report['diagnostics_ok']
            assert report['scores']['max_rhat'] < 1.01
            assert report['scores']['min_bulk_ess'] >= 400
            assert report['scores']['min_tail_ess'] >= 400
            assert report['scores']['divergences'] == 0
            assert_allclose(report['prior_center'], prior_center_from_training(local['y'][train]))
            with np.load(folder/report['selected_samples_file']) as saved:
                q = saved['log_samples'].reshape(-1, saved['log_samples'].shape[-1])
            with np.load(folder/f'{model}_predictions.npz') as saved:
                p = {k: saved[k] for k in saved.files}
            assert len(q) == len(p['weights'])
            assert_allclose(p['weights'], np.full(len(q), 1/len(q)))
            validate_mixture(p, model)
            scores = score_predictions(local, p, previous=model != 'gp')
            for key, value in scores.items():
                assert_allclose(value, report['scores'][key], atol=1e-10)
            # Reconstruct conditional components independently using only this
            # function's training y. Spanning indices cover every retained chain.
            sample_ids = np.unique(np.linspace(0, len(q)-1, 32, dtype=int))
            x = local['x'][train]
            d2 = (x[:, None]-x[None, :])**2
            cross_d2 = (local['x'][:, None]-x[None, :])**2
            for index in sample_ids:
                v, a, k, w, sw = np.exp(q[index, :5])
                W = 1. if model == 'gp' else np.exp(q[index, 5])
                A, B, noise = W*v, W**(2 if model == 'beta_gp' else 0)*a, W*k
                C = A*np.exp(-w*d2/2)+B*np.exp(-sw*d2/2)+noise*np.eye(len(x))
                cross = A*np.exp(-w*cross_d2/2)+B*np.exp(-sw*cross_d2/2)
                solved = np.linalg.solve(C, cross.T)
                mean = solved.T@local['y'][train]
                variance = A+B-np.einsum('ij,ji->i', cross, solved)
                for target in ('f', 'y'):
                    key = 'component_means' if model == 'gp' else f'component_{target}_means'
                    assert_allclose(p[key][index], mean, atol=1e-8, rtol=1e-7)
                    assert_allclose(p[f'component_{target}_variances'][index],
                                    variance+(noise if target == 'y' else 0), atol=1e-8, rtol=1e-6)
            W = np.ones(len(q)) if model == 'gp' else np.exp(q[:, 5])
            noise = np.exp(q[:, 2])*W
            assert_allclose(p['y_cov']-p['f_cov'], np.eye(181)*noise.mean(), atol=1e-8)
            # Independent log-density calculation, evaluated only on held-out truth.
            for target in ('f', 'y'):
                means = p['component_means' if model == 'gp' else f'component_{target}_means'][:, test]
                var = p[f'component_{target}_variances'][:, test]
                logpdf = -.5*(np.log(2*np.pi*var)+(local[target][test]-means)**2/var)
                nlpd = -np.mean(special.logsumexp(logpdf, axis=0)-np.log(len(q)))
                assert_allclose(nlpd, scores[f'{target}_mean_marginal_nlpd'], atol=1e-10)
            checks.append(dict(function_id=i, model=model, posterior_draws=len(q),
                               selected_attempt=report['selected_attempt'], passed=True))
        print(f'Validated function {i+1}/20, all four models', flush=True)
    write_json(root/'validation.json', dict(verified_fits=len(checks), all_passed=True, fits=checks,
               checks='Exact dataset/split identity; training-only prior inputs; sampler diagnostics; '
                      'mixture moments, covariance decomposition and PSD; interval CDF probabilities; '
                      'independent conditional Gaussian reconstruction; f/y noise relation; held-out scores.'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='figure/independent_functions_30_hmc')
    parser.add_argument('--watch', action='store_true', help='Wait for each function to finish before validating')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        validate(Path(args.output_dir), watch=args.watch)
