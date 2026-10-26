"""Independent provenance, conditioning and scoring audit of the Gasoline experiment."""
import argparse
import json
import numpy as np
import pandas as pd
from scipy import linalg, special, stats
from threadpoolctl import threadpool_limits

from gasoline_plugin_experiment import (SOURCE, OUTPUT, SEEDS, CONTEXT_SEED, COUNTS,
    SHARING, MODELS, METRICS, arrays, digest, prepare, aggregate)
from GH_shared_function_map import make_target as shared_target, gig_mean
from GH_multiple_hmc import make_target as independent_target, coordinates
from GH_independent_plugin_map import INTERNAL, native_center, se
from GH_fixed_function import mixing_quadrature, marginal_distribution
from shared_beta_plugin_experiment import write_json


def validate(seeds=SEEDS, *, experiment=None):
    output = OUTPUT if experiment is None else experiment.OUTPUT
    source = SOURCE if experiment is None else experiment.SOURCE
    all_seeds = SEEDS if experiment is None else experiment.SEEDS
    context_seed = CONTEXT_SEED if experiment is None else experiment.CONTEXT_SEED
    data = prepare() if experiment is None else experiment.prepare()
    x, y = data['x'], data['y']; n = len(x)
    if experiment is None:
        raw = arrays(source/'gasoline.npz')
        np.testing.assert_array_equal(y, raw['y'][:, ::4])
        np.testing.assert_array_equal(data['wavelength_nm'], raw['wavelength_nm'][::4])
        np.testing.assert_allclose(x, (data['wavelength_nm']-900)/80, atol=1e-14)
        design_checks = {}
    else:
        design_checks = experiment.validate_design(data, seeds)
    checks, all_scores, all_common = [], [], []
    n_predictions = 0
    for seed in seeds:
        split_folder = output/f'split_{seed}'; split = arrays(split_folder/'split.npz')
        train, test = split['train_rows'], split['test_rows']
        assert len(train) == 40 and len(test) == 20 and not np.intersect1d(train, test).size
        perm = np.random.default_rng(seed).permutation(60)
        np.testing.assert_array_equal(train, np.sort(perm[:40]))
        np.testing.assert_array_equal(test, np.sort(perm[40:]))
        scale = float(split['common_scale']); Y = y[train]/scale; center = split['prior_center']
        np.testing.assert_allclose(scale, np.std(y[train]-y[train].mean(0)), atol=1e-14)
        np.testing.assert_allclose(center, np.log([.5, max(float(np.mean(Y.mean(0)**2)), .5), .01, 1., .04]), atol=1e-14)
        for j, row in enumerate(test):
            expected = np.random.default_rng(np.random.SeedSequence([context_seed, int(row)])).permutation(n)
            np.testing.assert_array_equal(split['context_orders'][j], expected)
        for sharing in SHARING:
            for model in MODELS:
                folder = split_folder/sharing/model
                fit = json.loads((folder/'fit.json').read_text()); q = np.asarray(fit['log_parameters'])
                assert fit['dataset_sha256'] == digest(output/'dataset.npz')
                assert fit['split_sha256'] == digest(split_folder/'split.npz')
                assert fit['max_hessian_eigenvalue'] < 0
                if sharing == 'shared':
                    target, _ = shared_target(Y, x, model, prior_center=center)
                    value, gradient = target(q)
                else:
                    np.testing.assert_array_equal(fit['native_prior_center'], native_center(center, model))
                    # Check the faster shared-grid likelihood against direct per-curve Cholesky.
                    target = independent_target([(x, Y)], model=INTERNAL[model], prior_center=native_center(center, model), backend='dense')
                    value, gradient = target(coordinates(q, INTERNAL[model]))
                    other = independent_target([(x, Y)], model=INTERNAL[model], prior_center=native_center(center, model), backend='shared_grid')
                    fast_value, fast_gradient = other(coordinates(q, INTERNAL[model]))
                    np.testing.assert_allclose(value, fast_value, atol=1e-5, rtol=0)
                    np.testing.assert_allclose(gradient, fast_gradient, atol=1e-4, rtol=1e-4)
                np.testing.assert_allclose(value, fit['log_posterior'], atol=1e-5, rtol=0)
                assert max(abs(gradient)) < 1e-3
                v, a, k, w, sw = np.exp(q[:5]); K = se(x, x, v, w); Sigma = se(x, x, a, sw)
                W = np.ones(40) if model == 'gp' else np.exp(q[5:])
                np.testing.assert_allclose(W, fit['W_training_mode'], atol=1e-12)
                if sharing == 'shared':
                    c = W/gig_mean() if model == 'ghp_beta' else np.ones(40)
                    t = np.sum(c*c/W); b = np.sum((c/W)[:, None]*Y, axis=0)
                    h = Sigma@linalg.solve(K+k*np.eye(n)+t*Sigma, b, assume_a='pos')
                    shared_fn = h/gig_mean() if model == 'ghp_beta' else h
                    np.testing.assert_allclose(shared_fn, fit['function'], atol=1e-7)
                fit_hash = digest(folder/'fit.json')
                max_fn_error, max_mean_error = 0., 0.
                for count in COUNTS:
                    for j, row in enumerate(test):
                        stem = folder/f'context_{count}'/f'curve_{row+1:02d}'
                        p = arrays(stem.with_suffix('.npz')); meta = json.loads(stem.with_suffix('.json').read_text())
                        assert meta['fit_sha256'] == fit_hash
                        ids, held = p['context_ids'], p['test_ids']
                        np.testing.assert_array_equal(ids, np.sort(split['context_orders'][j, :count]))
                        np.testing.assert_array_equal(held, np.setdiff1d(np.arange(n), ids))
                        A = K[np.ix_(ids, ids)]+k*np.eye(count); yc = y[row, ids]/scale
                        if sharing == 'shared':
                            fn = shared_fn
                        else:
                            mode = meta['context_mode']; Wc = mode['W_for_function']
                            coef = Wc if model == 'ghp_beta' else 1.
                            C = Wc*A+coef**2*Sigma[np.ix_(ids, ids)]
                            fn = coef*Sigma[:, ids]@linalg.solve(C, yc, assume_a='pos')
                            if model != 'gp':
                                assert mode['max_abs_gradient'] < 1e-3 and mode['second_derivative'] < 0
                        np.testing.assert_allclose(fn, p['function_estimate_scaled'], atol=1e-7)
                        max_fn_error = max(max_fn_error, float(max(abs(fn-p['function_estimate_scaled']))))
                        B = linalg.solve(A, K[ids, :], assume_a='pos').T
                        mu = np.zeros(n) if model == 'ghp_beta' else fn
                        beta = fn if model == 'ghp_beta' else np.zeros(n)
                        residual = yc-mu[ids]; location = mu+B@residual; effect = beta-B@beta[ids]
                        S = K-B@K[ids, :]
                        np.testing.assert_allclose(p['location'], scale*location, atol=1e-8)
                        np.testing.assert_allclose(p['conditional_beta'], scale*effect, atol=1e-8)
                        if model in ('ghp_mu', 'ghp_beta'):
                            chi = 1+residual@linalg.solve(A, residual, assume_a='pos')
                            psi = 1+beta[ids]@linalg.solve(A, beta[ids], assume_a='pos')
                            np.testing.assert_allclose([p['W_lambda'], p['W_chi'], p['W_psi']], [1-count/2, chi, psi], rtol=1e-7, atol=1e-8)
                        elif model == 'stp':
                            np.testing.assert_allclose(p['W_shape'], (5+count)/2)
                            np.testing.assert_allclose(p['W_scale'], (3+residual@linalg.solve(A, residual, assume_a='pos'))/2, rtol=1e-7)
                        mean = scale*(location+p['W_mean']*effect)
                        variance = scale**2*(p['W_mean']*(np.diag(S)+k)+p['W_var']*effect**2)
                        np.testing.assert_allclose(mean, p['y_mean'], atol=1e-8)
                        np.testing.assert_allclose(variance, p['y_var'], atol=1e-8)
                        max_mean_error = max(max_mean_error, float(max(abs(mean-p['y_mean']))))
                        for name, noise in (('f', 0.), ('y', k)):
                            np.testing.assert_allclose(p[name+'_scale_var'], scale**2*(np.diag(S)+noise), atol=1e-8)
                            p[name+'_scale_cov'] = np.diag(p[name+'_scale_var'])
                        p['mixing'] = str(p['mixing']); dist = marginal_distribution(p, 'y')
                        np.testing.assert_allclose(-dist.logpdf(y[row])[held], p['y_nlpd'], atol=1e-6)
                        np.testing.assert_allclose(dist.cdf(p['y_ci'][:, 0]), .025, atol=1e-7)
                        np.testing.assert_allclose(dist.cdf(p['y_ci'][:, 1]), .975, atol=1e-7)
                        if j == 0:
                            nodes, weights, _ = mixing_quadrature(p)
                            logp = special.logsumexp(np.log(weights[:, None])+stats.norm.logpdf(y[row],
                                scale*(location+nodes[:, None]*effect),
                                scale*np.sqrt(nodes[:, None]*(np.diag(S)+k))), axis=0)
                            np.testing.assert_allclose(-logp[held], p['y_nlpd'], atol=1e-5, rtol=1e-7)
                        for score_name, targets in (('scores', held), ('common_scores', np.setdiff1d(np.arange(n), split['context_orders'][j, :30]))):
                            ci = p['y_ci'][targets]; truth = y[row, targets]; pos = np.searchsorted(held, targets)
                            actual = dict(y_rmse=np.sqrt(np.mean((truth-mean[targets])**2)), y_nlpd=p['y_nlpd'][pos].mean(),
                                y_crps=p['y_crps'][pos].mean(), y_coverage_95=np.mean((truth >= ci[:, 0]) & (truth <= ci[:, 1])),
                                y_interval_width=np.mean(ci[:, 1]-ci[:, 0]))
                            for key, val in actual.items():
                                np.testing.assert_allclose(val, meta[score_name][key], atol=1e-9)
                        assert (p['y_var'] > 0).all() and (p['y_crps'] >= 0).all()
                        for key, array in p.items():
                            if isinstance(array, np.ndarray) and np.issubdtype(array.dtype, np.number):
                                assert np.isfinite(array).all(), key
                        all_scores.append(meta['scores']); all_common.append(meta['common_scores'])
                        n_predictions += 1
                checks.append(dict(seed=seed, sharing=sharing, model=model, max_abs_gradient=float(max(abs(gradient))),
                    max_hessian_eigenvalue=fit['max_hessian_eigenvalue'], max_function_error=max_fn_error, max_mean_error=max_mean_error))
                print(f'Validated {seed} {sharing} {model}', flush=True)
    assert n_predictions == 320*len(seeds) and len(checks) == 8*len(seeds)
    if tuple(seeds) != all_seeds:
        print(f'Validated {len(checks)} fits and {n_predictions} predictions (partial run).')
        return
    for prefix, rows in (('', all_scores), ('common_', all_common)):
        scores, summary = aggregate(pd.DataFrame(rows), seeds=all_seeds)
        keys = ['seed', 'sharing', 'model', 'context_count']
        saved = pd.read_csv(output/f'{prefix}split_scores.csv').set_index(keys).sort_index()
        np.testing.assert_allclose(scores.set_index(keys).sort_index()[list(METRICS)], saved[list(METRICS)], atol=1e-9)
        keys = keys[1:]; columns = [m+s for m in METRICS for s in ('_mean', '_sd')]
        saved = pd.read_csv(output/f'{prefix}scores.csv').set_index(keys).sort_index()
        np.testing.assert_allclose(summary.set_index(keys).sort_index()[columns], saved[columns], atol=1e-9)
    paired = pd.read_csv(output/'paired_split_comparison.csv')
    for metric in METRICS:
        np.testing.assert_allclose(paired[metric+'_change'], paired[metric+'_individual']-paired[metric+'_shared'], atol=1e-10)
    size = sum(p.stat().st_size for p in output.rglob('*') if p.is_file())
    write_json(output/'validation.json', dict(all_checks_passed=True, n_fits=len(checks), n_predictions=n_predictions,
        train_only_preprocessing=True, same_splits_and_contexts=True, independent_conditioning_checks=checks,
        output_bytes_before_validation_report=size, source_cache_bytes=sum(p.stat().st_size for p in source.iterdir() if p.is_file()),
        **design_checks))
    print(f'Validated {len(checks)} fits and {n_predictions} predictions; output {size/2**20:.2f} MiB.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, choices=SEEDS)
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        validate(SEEDS if args.seed is None else (args.seed,))
