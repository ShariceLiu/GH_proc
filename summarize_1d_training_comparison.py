"""Compare a sparse single-function fit with the saved 91-point fits.

Uses the original 90 held-out points for both training sizes. Run after
hmc_1d_split_experiments.py has produced all four models for the sparse split.
"""

import argparse
import json
from pathlib import Path

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import special
from threadpoolctl import threadpool_limits

from hmc_beta_mu_1d_experiment import load_dataset
from hmc_1d_split_experiments import MODELS, LABELS, COLORS, prior_center_from_training, write_json
from hmc_gp_tp_1d_experiment import score_predictions
from fit_beta_mu_1d_experiment import _save_csv


SOURCES = dict(gp='figure/gp_tp_1d_hmc', tp_mu_gp='figure/tp_mu_1d_joint_hmc',
               mu_gp='figure/beta_vs_mu_1d_joint_hmc', beta_gp='figure/beta_vs_mu_1d_joint_hmc')


def validate_mixture(prediction, model):
    """Verify stored moments and interval probabilities against the components."""
    weights = prediction['weights']
    assert_allclose(weights.sum(), 1.)
    assert np.all(weights >= 0)
    assert_allclose(prediction['f_mean'], prediction['y_mean'])
    for target in ('f', 'y'):
        means = prediction['component_means' if model == 'gp' else f'component_{target}_means']
        variances = prediction[f'component_{target}_variances']
        assert np.all(variances > 0)
        mean = weights@means
        variance = weights@variances+weights@(means-mean)**2
        assert_allclose(prediction[f'{target}_mean'], mean, atol=1e-10)
        assert_allclose(prediction[f'{target}_var'], variance, atol=1e-9)
        cov = prediction[f'{target}_cov']
        assert_allclose(cov, cov.T, atol=1e-10)
        assert np.linalg.eigvalsh(cov).min() > -1e-8
        assert_allclose(cov, prediction[f'{target}_within_cov']+prediction[f'{target}_between_cov'], atol=1e-9)
        ci = prediction[f'{target}_ci']
        for side, probability in ((0, .025), (1, .975)):
            cdf = weights@special.ndtr((ci[:, side]-means)/np.sqrt(variances))
            assert_allclose(cdf, probability, atol=1e-8)


def summarize(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    original = load_dataset()
    folder = root/'sparse'
    with np.load(folder/'dataset.npz') as saved:
        data = {key: saved[key].copy() for key in saved.files}
    for key in ('x', 'y', 'f', 'beta', 'W'):
        assert_array_equal(data[key], original[key])
    for source in set(SOURCES.values()):
        with np.load(Path(source)/'dataset.npz') as saved:
            for key in ('x', 'y', 'f', 'train_ids', 'test_ids'):
                assert_array_equal(saved[key], original[key])
    assert set(data['train_ids']) < set(original['train_ids'])
    assert set(original['test_ids']) <= set(data['test_ids'])
    n_train = len(data['train_ids'])
    rows, summaries, checked = [], {}, []
    for model in MODELS:
        report = json.loads((folder/f'{model}_report.json').read_text())
        if not report['diagnostics_ok']:
            raise ValueError(f'{model}: resolve HMC diagnostics before comparison')
        assert_allclose(report['prior_center'], prior_center_from_training(data['y'][data['train_ids']]))
        for count, path in [(n_train, folder/f'{model}_predictions.npz'),
                            (91, Path(SOURCES[model])/f'{model}_predictions.npz')]:
            with np.load(path) as p:
                validate_mixture(p, model)
                scores = score_predictions(original, p, previous=model != 'gp')
                # Independent density evaluation checks the logarithm is taken after mixing.
                test = original['test_ids']
                for target in ('f', 'y'):
                    means = p['component_means' if model == 'gp' else f'component_{target}_means'][:, test]
                    variances = p[f'component_{target}_variances'][:, test]
                    logpdf = -.5*(np.log(2*np.pi*variances)+(original[target][test]-means)**2/variances)
                    nlpd = -np.mean(special.logsumexp(logpdf+np.log(p['weights'])[:, None], axis=0))
                    assert_allclose(nlpd, scores[f'{target}_mean_marginal_nlpd'], atol=1e-10)
                rows.append(dict(model=model, n_train=count, n_test=len(test), **scores))
                summaries[count, model] = {f'{t}_{s}': p[f'{t}_{s}'].copy()
                                          for t in ('f', 'y') for s in ('mean', 'ci')}
                if count == n_train:
                    new_scores = score_predictions(data, p, previous=model != 'gp')
                    for key, value in new_scores.items():
                        assert_allclose(value, report['scores'][key], atol=1e-10)
            checked.append(str(path))
    _save_csv(root/'comparison_on_original_90.csv', rows)
    fig, axes = plt.subplots(4, 2, figsize=(13, 11), sharex=True, sharey=True, constrained_layout=True)
    for row, model in enumerate(MODELS):
        for col, count in enumerate((n_train, 91)):
            ax = axes[row, col]
            p = summaries[count, model]
            ids = data['train_ids'] if count == n_train else original['train_ids']
            ax.plot(data['x'], data['f'], 'k--', lw=1, label='True f')
            ax.plot(data['x'], p['f_mean'], color=COLORS[model], label='Posterior mean')
            ax.fill_between(data['x'], *p['f_ci'].T, color=COLORS[model], alpha=.22, label='Pointwise 95% interval')
            ax.scatter(data['x'][ids], data['y'][ids], c='black', s=10, alpha=.6, label='Training y')
            ax.set_title(f'{LABELS[model]} | {count} training observations')
            ax.set(xlabel='x', ylabel='Latent f')
            ax.grid(alpha=.2)
    axes[0, 0].legend(fontsize=8)
    fig.savefig(root/'comparison_with_91.png', dpi=155)
    plt.close(fig)
    labels = dict(gp='GP', tp_mu_gp='STP + GP mean', mu_gp=r'GHP ($\mu$)', beta_gp=r'GHP ($\beta$)')
    lines = [r'% Requires \usepackage{booktabs}', r'\begin{table}[htbp]', r'\centering',
             r'\caption{Single-function regression with '+str(n_train)+r' or 91 training observations. '
             r'Both fits are evaluated on the same original 90 held-out points. '
             r'Kernel/noise parameters are inferred separately for each training set.}',
             r'\label{tab:single_function_training_sizes}', r'\begin{tabular}{lrrrrr}', r'\toprule',
             r'Model & Training points & RMSE ($f$) & NLPD ($f$) & RMSE ($y$) & NLPD ($y$) \\', r'\midrule']
    for row in rows:
        lines.append(labels[row['model']]+f' & {row["n_train"]} & '+' & '.join(f'{row[k]:.6f}' for k in
                     ('f_rmse', 'f_mean_marginal_nlpd', 'y_rmse', 'y_mean_marginal_nlpd'))+r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    (root/'comparison_on_original_90.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_json(root/'validation.json', dict(exact_original_realization=True,
               original_model_datasets_and_splits_match=True,
               nested_training_set=True, common_test_points=len(original['test_ids']),
               all_new_hmc_diagnostics_passed=True, prior_centers_use_only_new_training_y=True,
               validated_predictions=checked,
               checks='Mixture means, total variances, covariance decomposition, positive semidefiniteness, '
                      '95% interval CDF probabilities, NLPD from log-sum-exp and all new saved regression scores.'))
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='figure/single_function_30_hmc')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        summarize(Path(args.output_dir))
