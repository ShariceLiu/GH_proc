"""Estimate a GP-prior mean, fix it, then refit GP/STP/GHP residual models.

Reuses the exact 30/151 data split and each model's own full Bayesian samples.
STP denotes an independent GP mean plus an inverse-gamma-mixed residual.
The GP's second labelled SE component is its mean GP. No component ordering
or relabelling is introduced. This is a two-stage plug-in approximation.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import linalg
from threadpoolctl import threadpool_limits

from GH_proc_joint_hmc import _inputs_1d
from GH_zero_mean_hmc import PLUGIN_MU_MODELS, prior_center_from_training
from hmc_1d_split_experiments import write_json
from hmc_zero_mean_30_experiment import fit_model, TEX_LABELS

SOURCES = dict(gp_plugin_mu='gp', stp_plugin_mu='tp_mu_gp', ghp_plugin_mu='mu_gp')
LABELS = dict(gp_plugin_mu='GP', stp_plugin_mu='STP + GP mean', ghp_plugin_mu='GHP mu')
SCORE_COLUMNS = ['f_rmse', 'f_mean_marginal_nlpd', 'y_rmse', 'y_mean_marginal_nlpd']


def posterior_mu_mean(y_train, x_train, x_pred, log_samples, *, source_model):
    """E[mu_pred|y_train] = average Sigma_pred,train C^-1 y_train.

    C=Sigma_train,train+W*(K_train,train+kI), W=1 for the GP.
    Source physical order is v_0, mu_v0, k, w, mu_w, [W], all logged.
    Inputs contain no true mu/beta/f or held-out y.
    """
    y, x = _inputs_1d(y_train, x_train)
    if source_model not in SOURCES.values():
        raise ValueError('Expected gp, tp_mu_gp or mu_gp source model')
    xp, samples = np.asarray(x_pred, dtype=float), np.asarray(log_samples, dtype=float)
    size = 5 if source_model == 'gp' else 6
    if xp.ndim != 1 or not xp.size or not np.all(np.isfinite(xp)):
        raise ValueError('Need finite nonempty 1D prediction locations')
    if samples.ndim != 3 or samples.shape[-1] != size or not samples.size or not np.all(np.isfinite(samples)):
        raise ValueError(f'Need finite (chains,draws,{size}) samples')
    dxx, dpx = (x[:, None]-x[None, :])**2, (xp[:, None]-x[None, :])**2
    chain_means = np.zeros((samples.shape[0], len(xp)))
    for chain, values in enumerate(samples):
        for q in values:
            v, a, k, w, mw = np.exp(q[:5])
            W = 1. if size == 5 else np.exp(q[5])
            K, Sigma = v*np.exp(-.5*w*dxx), a*np.exp(-.5*mw*dxx)
            C = Sigma+W*(K+k*np.eye(len(x)))
            cross = a*np.exp(-.5*mw*dpx)
            L = linalg.cholesky(C, lower=True, check_finite=False)
            chain_means[chain] += cross@linalg.cho_solve((L, True), y, check_finite=False)/len(values)
    return chain_means.mean(axis=0), chain_means


def prepare(root, model):
    root = Path(root)
    source, base = root/'sparse', root/'plugin_mu'
    original_model = SOURCES[model]
    report_path = source/f'{original_model}_report.json'
    report = json.loads(report_path.read_text())
    if not report['scores']['diagnostics_ok']:
        raise ValueError('Source full Bayesian fit must pass convergence checks')
    with np.load(source/'dataset.npz') as saved:
        data = {k: saved[k].copy() for k in saved.files}
    if len(data['train_ids']) != 30 or len(data['test_ids']) != 151:
        raise ValueError('Expected saved 30/151 split')
    train = data['train_ids']
    with np.load(source/report['selected_samples_file']) as saved:
        samples = saved['log_samples']
    mu, chains = posterior_mu_mean(data['y'][train], data['x'][train], data['x'], samples,
                                   source_model=original_model)
    if original_model != 'gp':
        with np.load(source/f'{original_model}_predictions.npz') as saved:
            np.testing.assert_allclose(mu, saved['mu_mean'], atol=1e-11)
    data['mu_fixed'] = mu
    case = base/model
    case.mkdir(parents=True, exist_ok=True)
    files = [source/'dataset.npz', report_path, source/report['selected_samples_file'],
             source/f'{original_model}_predictions.npz', root/'scores_with_plugin_beta.csv']
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    provenance = case/'source_provenance.json'
    if provenance.exists():
        if json.loads(provenance.read_text()) != hashes:
            raise ValueError('Source posterior or data changed; review provenance')
    else:
        write_json(provenance, hashes)
    if (case/'dataset.npz').exists():
        with np.load(case/'dataset.npz') as previous:
            for key in data:
                np.testing.assert_array_equal(data[key], previous[key])
    else:
        np.savez_compressed(case/'dataset.npz', **data)
    np.savez_compressed(case/'mu_estimate.npz', x=data['x'], mu_mean=mu,
                        per_chain_mu_means=chains, source_draws=samples.shape[0]*samples.shape[1])
    write_json(case/'experiment.json', dict(model=model, source_model=original_model, n_train=30, n_test=151,
               estimator='mu_hat=E[mu|y_train] from this model\'s own GP-prior fit',
               mean_component='Second labelled SE component, no relabelling by length scale',
               source_draws=samples.shape[0]*samples.shape[1],
               likelihood='N(y_train; mu_hat, W*(K(v_0,w)+kI)); GP fixes W=1',
               learned_stage2=['v_0', 'k', 'w']+([] if model == 'gp_plugin_mu' else ['W']),
               mixing_prior='none' if model == 'gp_plugin_mu' else (
                   dict(kind='inverse-gamma', nu=5., shape=2.5, scale=1.5) if model == 'stp_plugin_mu'
                   else dict(kind='GIG', l=1., omega=1.)),
               prior_center=prior_center_from_training(data['y'][train]), prior_sd=2.,
               mu_uncertainty_propagated=False, heldout_y_or_truth_used_for_fitting=False,
               method_note='Two-stage plug-in using original priors in stage 2. Same training y is used '
                           'to estimate mu and fit the residual model; not full Bayesian inference.'))
    return data, base


def summarize(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    base = root/'plugin_mu'
    previous = pd.read_csv(root/'scores_with_plugin_beta.csv')
    rows, pairs = [], []
    for model in PLUGIN_MU_MODELS:
        rp = json.loads((base/model/'report.json').read_text())
        row = dict(rp['scores'], scenario='sparse', n_signal_components=1,
                   component_swap_acceptance=0., known_beta=False, uses_true_beta_information=False)
        if 'data_seed' in previous:
            seeds = previous['data_seed'].dropna().unique()
            if len(seeds) != 1:
                raise ValueError('Expected one data seed per experiment')
            row['data_seed'] = int(seeds[0])
        rows.append(row)
        old = previous.loc[previous['model'] == SOURCES[model]].iloc[0].to_dict()
        pairs.extend([dict(old, family=LABELS[model], method='full_bayesian'),
                      dict(row, family=LABELS[model], method='fixed_estimated_mu')])
    # Include the already-completed beta comparison as a useful reference.
    for key, method in [('beta_gp', 'full_bayesian'), ('ghp_plugin_beta', 'fixed_estimated_beta')]:
        row = previous.loc[previous['model'] == key].iloc[0].to_dict()
        pairs.append(dict(row, family='GHP beta', method=method))
    paired = pd.DataFrame(pairs)
    paired.to_csv(base/'paired_scores.csv', index=False)
    combined = pd.concat([previous, pd.DataFrame(rows)], ignore_index=True)
    combined.to_csv(root/'scores_with_plugin_mu.csv', index=False)
    labels = dict(TEX_LABELS, ghp_known_beta=r'GHP (true $\beta$)',
                  ghp_fixed_mu=r'GHP ($\mu=E[W]\beta_{\mathrm{true}}$)',
                  ghp_plugin_beta=r'GHP (fixed estimated $\beta$)',
                  gp_plugin_mu=r'GP (fixed estimated $\mu$)',
                  stp_plugin_mu=r'STP (fixed estimated $\mu$)',
                  ghp_plugin_mu=r'GHP (fixed estimated $\mu$)')
    for frame, path, caption in [
            (combined, root/'comparison_with_plugin_mu.tex', 'All models on the same 30/151 split.'),
            (paired, base/'paired_comparison.tex', 'Full Bayesian versus fixed estimated mean/beta on the same 30/151 split.')]:
        lines = [r'% Requires \usepackage{booktabs}', r'\begin{table}[htbp]', r'\centering',
                 r'\caption{'+caption+r' Plug-in estimates use only training data. Residual kernel/noise '
                 r'parameters and applicable $W$ are refitted with HMC. GHP fixes $\lambda=\omega=1$; '
                 r'STP fixes $\nu=5$.}', r'\begin{tabular}{lrrrr}', r'\toprule',
                 r'Model & RMSE ($f$) & NLPD ($f$) & RMSE ($y$) & NLPD ($y$) \\', r'\midrule']
        for row in frame.to_dict('records'):
            flag = '' if row['diagnostics_ok'] else r'$^{\dagger}$'
            lines.append(labels[row['model']]+flag+' & '+' & '.join(f'{row[k]:.5f}' for k in SCORE_COLUMNS)+r' \\')
        lines += [r'\bottomrule', r'\end{tabular}']
        if not frame['diagnostics_ok'].all():
            lines.append(r'\par\small $\dagger$ HMC diagnostics require investigation.')
        lines.append(r'\end{table}')
        path.write_text('\n'.join(lines)+'\n', encoding='utf-8')

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True, sharey=True, constrained_layout=True)
    mu_fig, mu_ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    for i, model in enumerate(PLUGIN_MU_MODELS):
        with np.load(base/model/'dataset.npz') as saved:
            data = {k: saved[k] for k in saved.files}
        mu_ax.plot(data['x'], data['mu_fixed'], label=LABELS[model])
        old = np.load(root/'sparse'/f'{SOURCES[model]}_predictions.npz')
        new = np.load(base/model/'predictions.npz')
        for j, target in enumerate(['f', 'y']):
            ax = axes[i, j]
            ax.plot(data['x'], data['f'], 'k--', lw=1., label='True f')
            for p, color, label in [(old, 'C0', 'Full Bayesian'), (new, 'C1', 'Fixed estimated mu')]:
                ax.plot(data['x'], p[f'{target}_mean'], color=color, label=label)
                ax.fill_between(data['x'], *p[f'{target}_ci'].T, color=color, alpha=.18)
            train, test = data['train_ids'], data['test_ids']
            ax.scatter(data['x'][train], data['y'][train], c='black', s=10, label='Training y')
            if target == 'y':
                ax.scatter(data['x'][test], data['y'][test], c='gray', marker='x', s=9, alpha=.45, label='Test y')
            ax.set(title=f'{LABELS[model]}: {target}', xlabel='x', ylabel=target)
            ax.grid(alpha=.18)
            if i == 0:
                ax.legend(fontsize=8)
        old.close()
        new.close()
    fig.suptitle('Same 30 / 151 split | bands are pointwise 95% intervals')
    fig.savefig(base/'regression.png', dpi=150)
    plt.close(fig)
    mu_ax.set(xlabel='x', ylabel='Estimated mu(x)', title='Each model estimates mu from its own GP prior and training y')
    mu_ax.legend()
    mu_ax.grid(alpha=.2)
    mu_fig.savefig(base/'mu_estimates.png', dpi=150)
    plt.close(mu_fig)
    fig, axes = plt.subplots(4, 3, figsize=(13, 9), constrained_layout=True)
    for col, model in enumerate(PLUGIN_MU_MODELS):
        rp = json.loads((base/model/'report.json').read_text())
        with np.load(base/model/rp['selected_samples_file']) as saved:
            samples = saved['log_samples']
        for j in range(4):
            ax = axes[j, col]
            if j >= samples.shape[-1]:
                ax.set_visible(False)
                continue
            for chain in samples:
                ax.plot(chain[:, j], lw=.4, alpha=.6)
            ax.set_title(f'{LABELS[model]}: log {rp["parameter_names"][j]} | R-hat={rp["diagnostics"]["rhat"][j]:.4f}', fontsize=9)
    fig.savefig(base/'traces.png', dpi=130)
    plt.close(fig)
    write_json(base/'summary.json', dict(new_scores=rows, paired_comparison=pairs,
               all_diagnostics_ok=all(r['diagnostics_ok'] for r in rows)))
    print(paired[['family', 'method', *SCORE_COLUMNS]].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('figure/single_function_30_hmc'))
    parser.add_argument('--models', nargs='+', choices=PLUGIN_MU_MODELS, default=list(PLUGIN_MU_MODELS))
    parser.add_argument('--stage', choices=['prepare', 'fit', 'summarize', 'all'], default='all')
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=2000)
    parser.add_argument('--draws', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=23026)
    parser.add_argument('--target-accept', type=float, default=.99)
    parser.add_argument('--max-retries', type=int, default=1)
    args = parser.parse_args()
    if args.max_retries < 0:
        parser.error('--max-retries must be nonnegative')
    with threadpool_limits(limits=1):
        if args.stage != 'summarize':
            for model in args.models:
                data, base = prepare(args.output_dir, model)
                if args.stage in ('fit', 'all'):
                    fit_model(data, base, model, chains=args.chains, warmup=args.warmup, draws=args.draws,
                              seed=args.seed, target_accept=args.target_accept, max_retries=args.max_retries)
        if args.stage in ('summarize', 'all'):
            summarize(args.output_dir)


if __name__ == '__main__':
    main()
