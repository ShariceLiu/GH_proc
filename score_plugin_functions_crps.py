"""Add CRPS for four fixed-estimated-function fits and export the 13-row table.

Reuses saved predictions and the original nine CRPS rows; no refitting.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose
from scipy import special
from threadpoolctl import threadpool_limits

from hmc_1d_split_experiments import write_json
from predictive_crps import gaussian_mixture_crps
from score_single_function_crps import independent_check

MODELS = ('gp_plugin_mu', 'stp_plugin_mu', 'ghp_plugin_mu', 'ghp_plugin_beta')
COLUMNS = ['f_rmse', 'f_mean_marginal_nlpd', 'f_crps', 'y_rmse', 'y_mean_marginal_nlpd', 'y_crps']


def score(root, model):
    root = Path(root)
    folder = root/'crps_plugin_functions'
    folder.mkdir(parents=True, exist_ok=True)
    source = root/('plugin_beta' if model == 'ghp_plugin_beta' else 'plugin_mu')/model/'predictions.npz'
    csv = root/'scores_with_plugin_mu.csv'
    dataset = root/'sparse'/'dataset.npz'
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in [source, csv, dataset]}
    settings = dict(atol=1e-8, rtol=1e-8, tail_sd=10., batch_size=16)
    path = folder/f'{model}.json'
    if path.exists():
        saved = json.loads(path.read_text())
        if saved['source_sha256'] != hashes or saved['settings'] != settings or not saved['validation_passed']:
            raise ValueError('Cached sources/settings changed')
        if (folder/f'{model}.npz').exists():
            print(f'{model}: reusing validated CRPS', flush=True)
            return
    row = pd.read_csv(csv).set_index('model').loc[model]
    with np.load(dataset) as data, np.load(source) as p:
        ids = data['test_ids']
        weights = p['weights']
        assert_allclose(weights.sum(), 1., atol=1e-12)
        assert np.all(np.isinf(p['component_df']))
        report, scores = {}, {}
        for target in ['f', 'y']:
            truth, means = data[target][ids], p['component_means'][:, ids]
            variances = p[f'component_{target}_variances'][:, ids]
            assert_allclose(np.sqrt(np.mean((weights@means-truth)**2)), row[f'{target}_rmse'], atol=1e-11)
            logpdf = -.5*(np.log(2*np.pi*variances)+(truth-means)**2/variances)
            nlpd = -np.mean(special.logsumexp(logpdf+np.log(weights)[:, None], axis=0))
            assert_allclose(nlpd, row[f'{target}_mean_marginal_nlpd'], atol=1e-10)
            print(f'{model}: {target}, all {len(weights)} components, {len(ids)} points', flush=True)
            values, diagnostic = gaussian_mixture_crps(truth, means, variances, weights, **settings)
            checks = [dict(test_index=int(ids[j]), **independent_check(
                truth[j], means[:, j], variances[:, j], weights, values[j])) for j in [0, len(ids)//2, len(ids)-1]]
            scores[f'{target}_crps'] = values
            report[target] = dict(mean_crps=float(values.mean()), **diagnostic,
                                  independent_infinite_integral_checks=checks)
            print(f'{model}: {target} CRPS={values.mean():.8f}', flush=True)
    for name, digest in hashes.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
    np.savez_compressed(folder/f'{model}.npz', **scores)
    write_json(path, dict(model=model, source_sha256=hashes, settings=settings, validation_passed=True, **report))


def table(root):
    root = Path(root)
    original = pd.read_csv(root/'scores_with_crps.csv')
    latest = pd.read_csv(root/'scores_with_plugin_mu.csv').set_index('model')
    rows = original.to_dict('records')
    for model in MODELS:
        row = latest.loc[model].to_dict()
        row['model'] = model
        report = json.loads((root/'crps_plugin_functions'/f'{model}.json').read_text())
        assert report['validation_passed']
        for name, digest in report['source_sha256'].items():
            assert hashlib.sha256((root/name).read_bytes()).hexdigest() == digest
        for target in ['f', 'y']:
            row[f'{target}_crps'] = report[target]['mean_crps']
        rows.append(row)
    frame = pd.DataFrame(rows)
    assert len(frame) == 13 and frame['model'].is_unique
    for name in [c for c in COLUMNS if not c.endswith('crps')]:
        assert_allclose(frame[name], latest.loc[frame['model'], name], atol=1e-12)
    frame.to_csv(root/'scores_with_plugin_functions_crps.csv', index=False)
    labels = [r'GP (two SE)', r'STP + GP mean', r'GHP ($\mu$ GP, $\beta=0$)',
              r'GHP ($\mu=0$, $\beta$ GP)', r'GP ($\mu=0$, one SE)',
              r'STP ($\mu=0$, one SE)', r'GHP ($\mu=\beta=0$, one SE)',
              r'GHP ($\mu=0$, known $\beta$)', r'GHP ($\mu=E[W]\beta_{\mathrm{true}}$)',
              r'GP (fixed $\widehat\mu$)', r'STP (fixed $\widehat\mu$)',
              r'GHP ($\mu=\widehat\mu$, $\beta=0$)', r'GHP ($\mu=0$, $\beta=\widehat\beta$)']
    best = frame[COLUMNS].min()
    seeds = frame['data_seed'].dropna().unique() if 'data_seed' in frame else []
    seed = int(seeds[0]) if len(seeds) == 1 else None
    seed_note = f' (data seed {seed})' if seed is not None else ''
    label = 'tab:regression_fixed_functions'+(f'_seed{seed}' if seed is not None else '')
    lines = [r'% Requires \usepackage{booktabs}', r'\begin{table}[htbp]', r'    \centering', r'    \small',
             r'    \setlength{\tabcolsep}{4pt}',
             r'    \caption{Regression on the same 30 training and 151 test points'+seed_note+r'. '
             r'Lower is better; bold indicates the lowest value in each column. '
             r'In the last four rows, $\widehat\mu(x)=E[\mu(x)\mid y_{\mathrm{train}}]$ '
             r'or $\widehat\beta(x)=E[\beta(x)\mid y_{\mathrm{train}}]$ is held fixed '
             r'while residual parameters and applicable $W$ are refitted. '
             r'Only the known-$\beta$ and $E[W]\beta_{\mathrm{true}}$ rows receive the true beta function.}',
             r'    \label{'+label+'}', r'    \begin{tabular}{lrrrrrr}', r'        \toprule',
             r'        & \multicolumn{3}{c}{Latent $f$} & \multicolumn{3}{c}{Observation $y$} \\',
             r'        \cmidrule(lr){2-4} \cmidrule(lr){5-7}',
             r'        Model & RMSE & NLPD & CRPS & RMSE & NLPD & CRPS \\', r'        \midrule']
    for i, row in enumerate(frame.to_dict('records')):
        if i in [4, 7, 9]:
            lines.append(r'        \midrule')
        values = [r'\textbf{'+f'{row[c]:.5f}'+'}' if row[c] == best[c] else f'{row[c]:.5f}' for c in COLUMNS]
        lines.append('        '+labels[i]+' & '+' & '.join(values)+r' \\')
    lines += [r'        \bottomrule', r'    \end{tabular}', r'\end{table}']
    path = root/'comparison_with_plugin_functions_crps.tex'
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(frame[['model', *COLUMNS]].to_string(index=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('figure/single_function_30_hmc'))
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--stage', choices=['score', 'table', 'all'], default='all')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.stage in ['score', 'all']:
            for model in args.models:
                score(args.root, model)
        if args.stage in ['table', 'all']:
            table(args.root)
