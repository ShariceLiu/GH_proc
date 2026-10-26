"""Validate saved shared-function Tecator fits, split and prediction units."""
import json
import hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import linalg
from threadpoolctl import threadpool_limits

from GH_shared_function_map import MODELS, make_target, shared_calculation_qr, gig_mean
from GH_fixed_function import marginal_distribution
from tecator_shared_plugin_experiment import DEST, SOURCE, load_data
from shared_beta_plugin_experiment import write_json


def validate(output=DEST, expected_data=None):
    source = load_data(SOURCE) if expected_data is None else expected_data
    data = load_data(output)
    for key in source:
        np.testing.assert_array_equal(source[key], data[key])
    train, test = data['train_rows'], data['test_rows']
    assert not set(train).intersection(test)
    assert (len(train), len(test)) == (147, 31)
    scale = float(data['common_scale'])
    Y, x = data['y'][train]/scale, data['x']
    for ids, held in zip(data['context_ids'], data['evaluation_ids']):
        assert len(ids) == 15 and len(held) == 85
        np.testing.assert_array_equal(np.sort(np.r_[ids,held]), np.arange(len(x)))
    checks, rows = [], []
    for model in MODELS:
        fit = json.loads((output/model/'fit.json').read_text())
        fit_hash = hashlib.sha256((output/model/'fit.json').read_bytes()).hexdigest()
        q = np.asarray(fit['log_parameters'])
        target, _ = make_target(Y, x, model, prior_center=fit['prior_center'],
                                prior_sd=fit['prior_sd'], backend='qr')
        value, gradient = target(q)
        assert max(abs(gradient)) < 1e-3
        np.testing.assert_allclose(value, fit['log_posterior'], atol=1e-7, rtol=0)
        assert fit['max_hessian_eigenvalue'] < 0
        calc = shared_calculation_qr(Y, x, q, model, xp=x)
        function = calc['h_pred']/gig_mean() if model == 'ghp_beta' else calc['h_pred']
        np.testing.assert_allclose(function*scale, fit['function_original'], atol=1e-10)
        v, _, k, w, _ = np.exp(q[:5]); v *= scale**2; k *= scale**2
        function *= scale
        errors = []
        for j, row in enumerate(test):
            with np.load(output/model/f'curve_{row+1:03d}.npz') as saved:
                p = {key: saved[key] for key in saved.files}
            p['mixing'] = str(p['mixing'])
            ids, held = data['context_ids'][j], data['evaluation_ids'][j]
            np.testing.assert_array_equal(ids, p['context_ids'])
            np.testing.assert_array_equal(held, p['test_ids'])
            kernel = lambda a,b: v*np.exp(-.5*w*(a[:,None]-b[None,:])**2)
            A = kernel(x[ids],x[ids])+k*np.eye(len(ids))
            cross = kernel(x,x[ids])
            mu = np.zeros(len(x)) if model == 'ghp_beta' else function
            beta = function if model == 'ghp_beta' else np.zeros(len(x))
            residual = data['y'][row,ids]-mu[ids]
            B = linalg.solve(A,cross.T,assume_a='pos').T
            location, effect = mu+B@residual, beta-B@beta[ids]
            conditional_cov = kernel(x,x)-B@cross.T+k*np.eye(len(x))
            mean = location+p['W_mean']*effect
            cov = p['W_mean']*conditional_cov+p['W_var']*np.outer(effect,effect)
            np.testing.assert_allclose(mean, p['y_mean'], atol=2e-8, rtol=1e-8)
            np.testing.assert_allclose(cov, p['y_cov'], atol=2e-9, rtol=1e-6)
            assert np.all(p['y_var'] > 0) and np.all(np.isfinite(p['y_ci']))
            dist = marginal_distribution(p,'y')
            truth = data['y'][row,held]
            np.testing.assert_allclose(-dist.logpdf(data['y'][row])[held],p['y_nlpd'],atol=1e-8)
            record = json.loads((output/model/f'curve_{row+1:03d}.json').read_text())
            assert record['fit_sha256'] == fit_hash
            score = record['scores']
            rmse = np.sqrt(np.mean((truth-mean[held])**2))
            coverage = np.mean((truth >= p['y_ci'][held,0]) & (truth <= p['y_ci'][held,1]))
            np.testing.assert_allclose(rmse,score['y_rmse'],atol=2e-8)
            assert coverage == score['y_coverage_95']
            rows.append(score)
            errors.append(float(np.max(abs(mean-p['y_mean']))))
        checks.append(dict(model=model,max_abs_gradient=float(max(abs(gradient))),
                           max_original_unit_mean_error=max(errors),predictions=len(test)))
    per = pd.DataFrame(rows)
    aggregate = pd.read_csv(output/'scores.csv').set_index('model')
    for model in MODELS:
        frame = per[per.model == model]
        np.testing.assert_allclose(np.sqrt(np.mean(frame.y_rmse**2)),aggregate.loc[model,'y_rmse'])
        for metric in ('nlpd','crps','coverage_95','interval_width'):
            np.testing.assert_allclose(frame[f'y_{metric}'].mean(),aggregate.loc[model,f'y_{metric}'])
    report = dict(all_checks_passed=True, exact_source_split=True, curves_per_model=len(test),
                  held_out_points_per_model=len(test)*85, fits=checks)
    write_json(output/'validation.json',report)
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        validate()
