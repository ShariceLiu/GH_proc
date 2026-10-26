"""Describe saved Airfoil predictive mixtures; no fitting or new sampling.

For a Gaussian mixture, its third central moment is
sum_s weight_s * [(mean_s-mean)^3 + 3*(mean_s-mean)*variance_s].
This separates within-input predictive skewness from the distribution of y
pooled over different inputs. Pooled residual skewness is only descriptive;
there is one held-out observation at each input, not repeated conditional data.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import skew
from threadpoolctl import threadpool_limits


def analyze(root=Path('figure/airfoil_single_function_hmc')):
    out = root/'skewness_diagnostics'
    out.mkdir(exist_ok=True)
    with np.load(root/'dataset.npz') as data:
        y = data['y'].copy()
        train, test = data['train_ids'].copy(), data['test_ids'].copy()
    metadata = dict(n_train=len(train), n_test=len(test),
                    observed_y_skewness_all=float(skew(y, bias=False)),
                    observed_y_skewness_training=float(skew(y[train], bias=False)),
                    observed_y_skewness_test=float(skew(y[test], bias=False)),
                    method='Exact moments of all saved Gaussian mixture components; no new HMC',
                    sample_skewness='bias-corrected Fisher-Pearson coefficient',
                    predictive_skewness='third central mixture moment divided by variance**1.5')
    rows, means, stds = [], {}, {}
    pointwise = pd.DataFrame(dict(row_id=test, y=y[test]))
    for model in ('gp', 'tp_mu_gp', 'mu_gp', 'beta_gp'):
        with np.load(root/model/'predictions.npz') as pred:
            np.testing.assert_array_equal(test, pred['test_ids'])
            weights = pred['weights'].copy()
            cm, cv = pred['component_means'], pred['component_y_variances']
            mean = weights@cm
            centered = cm-mean
            var = weights@(cv+centered**2)
            third = weights@(centered**3+3*centered*cv)
            sk = third/var**1.5
            np.testing.assert_allclose(mean, pred['y_mean'], atol=1e-10)
            np.testing.assert_allclose(var, pred['y_var'], atol=1e-10)
            means[model], stds[model] = mean, np.sqrt(var)
            standardized_residual = (y[test]-mean)/np.sqrt(var)
            # Pool the predictive distributions over a uniform choice of test x.
            delta = mean-mean.mean()
            pooled_var = np.mean(var+delta**2)
            pooled_third = np.mean(third+3*delta*var+delta**3)
            row = dict(model=model, predictive_skew_median=float(np.median(sk)),
                       predictive_abs_skew_median=float(np.median(np.abs(sk))),
                       predictive_abs_skew_95pct=float(np.quantile(np.abs(sk), .95)),
                       predictive_skew_min=float(sk.min()), predictive_skew_max=float(sk.max()),
                       pooled_predictive_skew=float(pooled_third/pooled_var**1.5),
                       pooled_standardized_residual_skew=float(skew(standardized_residual, bias=False)),
                       y_rmse=float(np.sqrt(np.mean((y[test]-mean)**2))),
                       lower_95_interval_miss_rate=float(np.mean(y[test]<pred['y_ci'][:, 0])),
                       upper_95_interval_miss_rate=float(np.mean(y[test]>pred['y_ci'][:, 1])))
            rows.append(row)
            pointwise[f'{model}_predictive_skew'] = sk
            pointwise[f'{model}_standardized_residual'] = standardized_residual
    pairs = []
    for model in ('tp_mu_gp', 'mu_gp', 'beta_gp'):
        pairs.append(dict(model=model, reference='gp',
                          rms_mean_difference_db=float(np.sqrt(np.mean((means[model]-means['gp'])**2))),
                          mean_prediction_correlation=float(np.corrcoef(means[model], means['gp'])[0, 1]),
                          rms_sd_difference_db=float(np.sqrt(np.mean((stds[model]-stds['gp'])**2)))))
    pd.DataFrame(rows).to_csv(out/'predictive_skewness.csv', index=False)
    pointwise.to_csv(out/'pointwise.csv', index=False)
    pd.DataFrame(pairs).to_csv(out/'prediction_comparisons.csv', index=False)
    (out/'summary.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(metadata, indent=2))
    print(pd.DataFrame(rows).to_string(index=False))
    print(pd.DataFrame(pairs).to_string(index=False))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        analyze()
