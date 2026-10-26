"""Condition on the true saved beta function; learn v_0, k, w and W with HMC.

Uses the exact same 30/151 single-function split. Beta is supplied at training
and prediction inputs, while only training y enters inference. This experiment
has extra information about the generating process and is labelled accordingly.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from GH_zero_mean_hmc import KNOWN_BETA_MODEL, prior_center_from_training
from hmc_1d_split_experiments import write_json, make_split
from hmc_beta_mu_1d_experiment import load_dataset
from hmc_zero_mean_30_experiment import fit_model, TEX_LABELS


def prepare(root):
    root = Path(root)
    with np.load(root/'sparse'/'dataset.npz') as saved:
        data = {k: saved[k].copy() for k in saved.files}
    expected = make_split(load_dataset(), 'sparse', sparse_points=30)
    for key in ['x', 'y', 'f', 'beta', 'W', 'train_ids', 'test_ids']:
        np.testing.assert_array_equal(data[key], expected[key])
    folder = root/'known_beta'
    folder.mkdir(parents=True, exist_ok=True)
    data_path = folder/'dataset.npz'
    if data_path.exists():
        with np.load(data_path) as previous:
            for key in data:
                np.testing.assert_array_equal(data[key], previous[key])
    else:
        np.savez_compressed(data_path, **data)
    sources = [root/'sparse'/'dataset.npz', root/'scores.csv', root/'scores_with_zero_mean.csv']
    sources.extend(root/'sparse'/f'{m}_report.json' for m in ['gp', 'tp_mu_gp', 'mu_gp', 'beta_gp'])
    sources.extend(root/'zero_mean_baselines'/m/'report.json' for m in ['gp_zero', 'stp_zero', 'ghp_zero'])
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    source_path = folder/'source_provenance.json'
    if source_path.exists():
        if json.loads(source_path.read_text()) != hashes:
            raise ValueError('Earlier comparison sources changed; review provenance')
    else:
        write_json(source_path, hashes)
    write_json(folder/'experiment.json', dict(model=KNOWN_BETA_MODEL, n_train=30, n_test=151,
               mu=0., known_beta='Exact saved generating beta at all training/prediction inputs',
               likelihood='N(y_train; W*beta_train, W*(K(v_0,w)+k*I))',
               learned_parameters=['v_0', 'k', 'w', 'W'], fixed_gig=dict(l=1., omega=1.),
               prior_center=prior_center_from_training(data['y'][data['train_ids']]), prior_sd=2.,
               centering=False, scaling=False, beta_covariance_parameters_fitted=False,
               generating_W_for_reference_only=float(data['W']),
               fit_inputs=['x[train_ids]', 'y[train_ids]', 'beta[train_ids]'],
               prediction_side_information='beta at prediction inputs',
               comparison_note='Known-beta model has privileged information unavailable to the other fits'))
    return data, folder


def summarize(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    data, folder = prepare(root)
    report = json.loads((folder/KNOWN_BETA_MODEL/'report.json').read_text())
    previous = pd.read_csv(root/'scores_with_zero_mean.csv')
    new_row = dict(report['scores'], scenario='sparse', n_signal_components=1, component_swap_acceptance=0.)
    combined = pd.concat([previous, pd.DataFrame([new_row])], ignore_index=True)
    combined['known_beta'] = combined['model'] == KNOWN_BETA_MODEL
    combined.to_csv(root/'scores_with_known_beta.csv', index=False)
    columns = ['f_rmse', 'f_mean_marginal_nlpd', 'y_rmse', 'y_mean_marginal_nlpd']
    lines = [r'% Requires \usepackage{booktabs}', r'\begin{table}[htbp]', r'\centering',
             r'\caption{The same simulated function with 30 training and 151 test points. '
             r'Hyperparameters are inferred with HMC. The final row additionally receives '
             r'the true $\beta(x)$ at training and test inputs; $W$ remains inferred. '
             r'GHP fixes $\lambda=\omega=1$ and STP fixes $\nu=5$.}',
             r'\label{tab:known_beta_30}', r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Model & RMSE ($f$) & NLPD ($f$) & RMSE ($y$) & NLPD ($y$) \\', r'\midrule']
    labels = dict(TEX_LABELS, **{KNOWN_BETA_MODEL: r'GHP (known $\beta$, $\mu=0$)'})
    for row in combined.to_dict('records'):
        if row['model'] in ['gp_zero', KNOWN_BETA_MODEL]:
            lines.append(r'\midrule')
        flag = '' if row['diagnostics_ok'] else r'$^{\dagger}$'
        lines.append(labels[row['model']]+flag+' & '+' & '.join(f'{row[k]:.5f}' for k in columns)+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    if not combined['diagnostics_ok'].all():
        lines.append(r'\par\small $\dagger$ HMC diagnostics require investigation.')
    lines.append(r'\end{table}')
    (root/'comparison_with_known_beta.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    cases = [('ghp_zero', root/'zero_mean_baselines'/'ghp_zero', 'Beta fixed to zero'),
             ('beta_gp', root/'sparse', 'Beta inferred with a GP prior'),
             (KNOWN_BETA_MODEL, folder/KNOWN_BETA_MODEL, 'True beta supplied; W inferred')]
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.5), sharex=True, sharey=True, constrained_layout=True)
    w_values = {}
    for col, (model, source, label) in enumerate(cases):
        rp = source/'beta_gp_report.json' if model == 'beta_gp' else source/'report.json'
        this_report = json.loads(rp.read_text())
        pp = source/'beta_gp_predictions.npz' if model == 'beta_gp' else source/'predictions.npz'
        with np.load(pp) as p:
            for row, target in enumerate(['f', 'y']):
                ax = axes[row, col]
                ax.plot(data['x'], data['f'], 'k--', lw=1.1, label='True f')
                ax.plot(data['x'], p[f'{target}_mean'], color=f'C{col}', label='Posterior mean')
                ax.fill_between(data['x'], *p[f'{target}_ci'].T, color=f'C{col}', alpha=.23,
                                label='Pointwise 95% interval')
                ax.scatter(data['x'][data['train_ids']], data['y'][data['train_ids']], c='black', s=12,
                           alpha=.65, label='Training y')
                if target == 'y':
                    ax.scatter(data['x'][data['test_ids']], data['y'][data['test_ids']], c='gray',
                               marker='x', s=10, alpha=.5, label='Test y')
                score = this_report['scores']
                ax.set(title=f'{label}\n{target} RMSE={score[target+"_rmse"]:.4f}; '
                             f'NLPD={score[target+"_mean_marginal_nlpd"]:.4f}', xlabel='x', ylabel=target)
                ax.grid(alpha=.18)
                if col == 0:
                    ax.legend(fontsize=7)
        with np.load(source/this_report['selected_samples_file']) as samples:
            w_values[label] = np.exp(samples['log_samples'][..., -1]).reshape(-1)
    fig.suptitle('Same 30 / 151 split | known beta supplies additional information; kernel, noise and W are learned')
    fig.savefig(folder/'regression.png', dpi=150)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    axes[0].plot(data['x'], data['beta'], color='C2', label='Known generating beta')
    axes[0].scatter(data['x'][data['train_ids']], data['beta'][data['train_ids']], c='black', s=14,
                    label='Training input locations')
    axes[0].set(xlabel='x', ylabel='beta(x)', title='Beta supplied at training and prediction locations')
    axes[0].legend(fontsize=8)
    for i, (label, W) in enumerate(w_values.items()):
        axes[1].hist(W, bins=100, density=True, histtype='step', color=f'C{i}', linewidth=1.2,
                     label=label+f' (mean {W.mean():.3f}, sd {W.std():.3f})')
    axes[1].axvline(float(data['W']), c='black', ls='--', label=f'Generating W={float(data["W"]):.3f}')
    upper = max(float(np.quantile(W, .995)) for W in w_values.values())
    axes[1].set(xlim=(0., upper), xlabel='W', ylabel='Posterior density', title='W remains unknown in every fit')
    axes[1].legend(fontsize=7)
    fig.savefig(folder/'beta_and_W.png', dpi=155)
    plt.close(fig)
    with np.load(folder/KNOWN_BETA_MODEL/report['selected_samples_file']) as saved:
        samples = saved['log_samples']
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), constrained_layout=True)
    for j, ax in enumerate(axes.flat):
        for chain in samples:
            ax.plot(chain[:, j], lw=.4, alpha=.6)
        ax.set_title('log '+report['parameter_names'][j]+f' | R-hat={report["diagnostics"]["rhat"][j]:.4f}')
    fig.savefig(folder/'traces.png', dpi=130)
    plt.close(fig)
    W = w_values['True beta supplied; W inferred']
    write_json(folder/'summary.json', dict(scores=report['scores'], W_mean=W.mean(), W_std=W.std(),
               W_variance=W.var(), W_ci=np.quantile(W, [.025, .975]), generating_W=float(data['W']),
               extra_information='True beta at training and prediction inputs',
               comparison=combined[['model', *columns]].to_dict('records')))
    print(combined[['model', *columns]].to_string(index=False), flush=True)
    print(f'W posterior: mean={W.mean():.6f}, sd={W.std():.6f}; generating W={float(data["W"]):.6f}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('figure/single_function_30_hmc'))
    parser.add_argument('--stage', choices=['prepare', 'fit', 'summarize', 'all'], default='all')
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=2000)
    parser.add_argument('--draws', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=20026)
    parser.add_argument('--target-accept', type=float, default=.99)
    parser.add_argument('--max-retries', type=int, default=1)
    args = parser.parse_args()
    if args.max_retries < 0:
        parser.error('--max-retries must be nonnegative')
    with threadpool_limits(limits=1):
        data, folder = prepare(args.output_dir)
        if args.stage in ('fit', 'all'):
            fit_model(data, folder, KNOWN_BETA_MODEL, chains=args.chains, warmup=args.warmup,
                      draws=args.draws, seed=args.seed, target_accept=args.target_accept, max_retries=args.max_retries)
        if args.stage in ('summarize', 'all'):
            summarize(args.output_dir)


if __name__ == '__main__':
    main()
