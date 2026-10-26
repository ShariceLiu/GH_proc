"""Add f/y CRPS to all nine saved models in the single-function 30/151 case.

No training or new posterior/predictive sampling. Every retained Gaussian
component is used. Original predictions and score tables are left unchanged.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from scipy import integrate, special
from threadpoolctl import threadpool_limits

from hmc_1d_split_experiments import write_json
from hmc_zero_mean_30_experiment import TEX_LABELS
from predictive_crps import gaussian_mixture_crps


def prediction_path(root, model):
    if model in ['gp', 'tp_mu_gp', 'mu_gp', 'beta_gp']:
        return root/'sparse'/f'{model}_predictions.npz'
    if model in ['gp_zero', 'stp_zero', 'ghp_zero']:
        return root/'zero_mean_baselines'/model/'predictions.npz'
    if model == 'ghp_known_beta':
        return root/'known_beta'/model/'predictions.npz'
    if model == 'ghp_fixed_mu':
        return root/'fixed_mu_prior_mean'/model/'predictions.npz'
    raise ValueError(f'Unknown model {model}')


def independent_check(truth, means, variances, weights, expected):
    """Scalar QUADPACK integration over infinite tails, independent of quad_vec."""
    sd = np.sqrt(variances)
    def left(u):
        return float(weights@special.ndtr((truth-u-means)/sd))**2
    def right(u):
        return float(weights@special.ndtr((means-truth-u)/sd))**2
    a, ea = integrate.quad(left, 0., np.inf, epsabs=1e-10, epsrel=1e-10, limit=200)
    b, eb = integrate.quad(right, 0., np.inf, epsabs=1e-10, epsrel=1e-10, limit=200)
    assert_allclose(a+b, expected, atol=1e-8, rtol=1e-8)
    return dict(absolute_difference=abs(a+b-expected), reference_error_estimate=ea+eb)


def run(root):
    root = Path(root)
    folder = root/'crps'
    folder.mkdir(parents=True, exist_ok=True)
    source_csv = root/'scores_with_fixed_mu.csv'
    table = pd.read_csv(source_csv)
    with np.load(root/'sparse'/'dataset.npz') as saved:
        data = {k: saved[k].copy() for k in saved.files}
    test = data['test_ids']
    assert len(test) == 151 and len(data['train_ids']) == 30
    sources = [source_csv, root/'sparse'/'dataset.npz']
    sources.extend(prediction_path(root, m) for m in table['model'])
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    settings = dict(atol=1e-8, rtol=1e-8, tail_sd=10., batch_size=16)
    provenance = dict(source_sha256=hashes, integration_settings=settings, method='full Gaussian-mixture CDF integration')
    previous = folder/'source_provenance.json'
    if previous.exists() and json.loads(previous.read_text()) != provenance:
        raise ValueError('Source predictions/settings changed; use a separate output folder')
    write_json(previous, provenance)
    point_rows, reports = [], {}
    for row_index, row in table.iterrows():
        model = row['model']
        output = folder/f'{model}.npz'
        report_path = folder/f'{model}.json'
        if output.exists() and report_path.exists():
            with np.load(output) as previous_scores:
                point_scores = {k: previous_scores[k].copy() for k in previous_scores.files}
            report = json.loads(report_path.read_text())
            print(f'{model}: reusing validated CRPS', flush=True)
        else:
            started = time.perf_counter()
            report, point_scores = {}, {}
            with np.load(prediction_path(root, model)) as p:
                if 'component_df' in p.files:
                    assert np.all(np.isinf(p['component_df'])), 'Expected Gaussian conditional components'
                weights = p['weights']
                assert_allclose(weights.sum(), 1., atol=1e-12)
                for target in ['f', 'y']:
                    means = p['component_means' if 'component_means' in p.files else f'component_{target}_means'][:, test]
                    variances = p[f'component_{target}_variances'][:, test]
                    truth = data[target][test]
                    # Confirm both the target/split and the saved score conventions.
                    rmse = np.sqrt(np.mean((weights@means-truth)**2))
                    logpdf = -.5*(np.log(2*np.pi*variances)+(truth-means)**2/variances)
                    nlpd = -np.mean(special.logsumexp(logpdf+np.log(weights)[:, None], axis=0))
                    assert_allclose(rmse, row[f'{target}_rmse'], atol=1e-11)
                    assert_allclose(nlpd, row[f'{target}_mean_marginal_nlpd'], atol=1e-10)
                    print(f'{model}: {target} CRPS, {means.shape[0]} components x {len(test)} test points', flush=True)
                    scores, diagnostic = gaussian_mixture_crps(truth, means, variances, weights, **settings)
                    point_scores[f'{target}_crps'] = scores
                    checks = []
                    for point in [0, len(test)//2, len(test)-1]:
                        checks.append(dict(test_index=int(test[point]), **independent_check(
                            truth[point], means[:, point], variances[:, point], weights, scores[point])))
                    report[target] = dict(mean_crps=float(scores.mean()), **diagnostic,
                                           independent_infinite_integral_checks=checks,
                                           original_rmse_and_nlpd_verified=True)
                    print(f'{model}: {target} mean CRPS={scores.mean():.8f}', flush=True)
            report.update(model=model, validation_passed=True, seconds=time.perf_counter()-started)
            np.savez_compressed(output, **point_scores)
            write_json(report_path, report)
        reports[model] = report
        for target in ['f', 'y']:
            table.loc[row_index, f'{target}_crps'] = float(point_scores[f'{target}_crps'].mean())
        for j, index in enumerate(test):
            point_rows.append(dict(model=model, test_index=int(index), x=float(data['x'][index]),
                                   f_true=float(data['f'][index]), y_true=float(data['y'][index]),
                                   f_crps=float(point_scores['f_crps'][j]), y_crps=float(point_scores['y_crps'][j])))
    table.to_csv(root/'scores_with_crps.csv', index=False)
    pd.DataFrame(point_rows).to_csv(folder/'pointwise_crps.csv', index=False)
    labels = dict(TEX_LABELS, ghp_known_beta=r'GHP (known $\beta$)',
                  ghp_fixed_mu=r'GHP ($\mu=E[W]\beta_{\mathrm{true}}$)')
    columns = ['f_rmse', 'f_mean_marginal_nlpd', 'f_crps', 'y_rmse', 'y_mean_marginal_nlpd', 'y_crps']
    seeds = table['data_seed'].dropna().unique() if 'data_seed' in table else []
    seed = int(seeds[0]) if len(seeds) == 1 else None
    seed_note = f' (data seed {seed})' if seed is not None else ''
    table_label = 'tab:single_function_crps'+(f'_seed{seed}' if seed is not None else '')
    lines = [r'% Requires \usepackage{booktabs}', r'\begin{table}[htbp]', r'\centering', r'\small',
             r'\setlength{\tabcolsep}{4pt}',
             r'\caption{Single simulated function'+seed_note+r', 30 training and 151 held-out points. '
             r'All metrics are evaluated on the same held-out points; NLPD and CRPS '
             r'are averages of pointwise scores of the full posterior predictive mixtures. '
             r'Lower is better. GHP fixes $\lambda=\omega=1$ and STP fixes $\nu=5$. '
             r'The last two models receive the true beta function; the final model fixes '
             r'$\mu(x)=E_{p(W)}[W]\beta_{\mathrm{true}}(x)$ and $\beta=0$.}',
             r'\label{'+table_label+'}', r'\begin{tabular}{lrrrrrr}', r'\toprule',
             r' & \multicolumn{3}{c}{Latent $f$} & \multicolumn{3}{c}{Observation $y$} \\',
             r'\cmidrule(lr){2-4} \cmidrule(lr){5-7}',
             r'Model & RMSE & NLPD & CRPS & RMSE & NLPD & CRPS \\', r'\midrule']
    for row in table.to_dict('records'):
        if row['model'] in ['gp_zero', 'ghp_known_beta']:
            lines.append(r'\midrule')
        lines.append(labels[row['model']]+' & '+' & '.join(f'{row[k]:.5f}' for k in columns)+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (root/'comparison_with_crps.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    # The original data, predictions and score table must be byte-for-byte intact.
    for name, expected in hashes.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected
    errors = [max(reports[m][t]['integration_error_estimate']) for m in reports for t in ['f', 'y']]
    differences = [c['absolute_difference'] for m in reports for t in ['f', 'y']
                   for c in reports[m][t]['independent_infinite_integral_checks']]
    summary = dict(n_models=len(table), n_train=30, n_test=151, no_refitting=True,
                   all_saved_components_used=True, original_files_unchanged=True,
                   validation_passed=all(r['validation_passed'] for r in reports.values()),
                   max_integration_error_estimate=max(errors),
                   n_independent_checks=len(differences), max_independent_difference=max(differences),
                   scores=table[['model', *columns]].to_dict('records'))
    write_json(folder/'summary.json', summary)
    print(table[['model', *columns]].to_string(index=False), flush=True)
    print(json.dumps({k: v for k, v in summary.items() if k != 'scores'}, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('figure/single_function_30_hmc'))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        run(args.root)
