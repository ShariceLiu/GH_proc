"""Fix beta to its training-data posterior mean, then refit residual theta,W.

Stage 1 reuses the saved full Bayesian beta-GP hyperparameter samples and
computes beta_hat(x)=E[beta(x)|y_train]. Stage 2 treats that one function as
fixed and samples p(theta_residual,W | y_train; beta_hat) from the original
residual/noise/GIG priors. This is a two-stage plug-in approximation, not the
original full Bayesian posterior. No true beta, f, W or held-out y is used
to calculate beta_hat or in either inference likelihood.
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
from GH_zero_mean_hmc import PLUGIN_BETA_MODEL, prior_center_from_training
from hmc_1d_split_experiments import write_json
from hmc_zero_mean_30_experiment import fit_model, TEX_LABELS


def posterior_beta_mean(y_train, x_train, x_pred, log_samples):
    """Average E[beta_pred|y_train,theta,W]; needs only observed training y.

    Physical sample order: v_0, sigma_v0, k, w, sigma_w, W (all logged).
    This returns E[beta|y], NOT E[W beta|y]/E[W|y].
    """
    y, x = _inputs_1d(y_train, x_train)
    xp = np.asarray(x_pred, dtype=float)
    samples = np.asarray(log_samples, dtype=float)
    if xp.ndim != 1 or not xp.size or not np.all(np.isfinite(xp)):
        raise ValueError('Need finite nonempty 1D prediction locations')
    if samples.ndim != 3 or samples.shape[-1] != 6 or not samples.size or not np.all(np.isfinite(samples)):
        raise ValueError('Need finite samples with shape (chains,draws,6)')
    dxx = (x[:, None]-x[None, :])**2
    dpx = (xp[:, None]-x[None, :])**2
    chain_means = np.zeros((samples.shape[0], len(xp)))
    for chain, values in enumerate(samples):
        for q in values:
            v, sv, k, w, sw, W = np.exp(q)
            K = v*np.exp(-.5*w*dxx)
            Sigma = sv*np.exp(-.5*sw*dxx)
            C = W*K+W**2*Sigma+W*k*np.eye(len(x))
            cross = W*sv*np.exp(-.5*sw*dpx)
            L = linalg.cholesky(C, lower=True, check_finite=False)
            chain_means[chain] += cross@linalg.cho_solve((L, True), y, check_finite=False)/len(values)
    return chain_means.mean(axis=0), chain_means


def prepare(root):
    root = Path(root)
    source = root/'sparse'
    source_report = json.loads((source/'beta_gp_report.json').read_text())
    if not source_report['scores']['diagnostics_ok']:
        raise ValueError('Source full Bayesian fit must pass convergence checks')
    with np.load(source/'dataset.npz') as saved:
        data = {k: saved[k].copy() for k in saved.files}
    if len(data['train_ids']) != 30 or len(data['test_ids']) != 151:
        raise ValueError('Expected saved 30/151 experiment')
    train = data['train_ids']
    with np.load(source/source_report['selected_samples_file']) as saved:
        samples = saved['log_samples']
    beta_hat, chain_means = posterior_beta_mean(data['y'][train], data['x'][train], data['x'], samples)
    with np.load(source/'beta_gp_predictions.npz') as saved:
        np.testing.assert_allclose(beta_hat, saved['beta_mean'], atol=1e-11)
    # Retain truth for plots/scoring only; fit_model's beta input is the estimate.
    data['beta_true'] = data['beta'].copy()
    data['beta'] = beta_hat
    folder = root/'plugin_beta'
    folder.mkdir(parents=True, exist_ok=True)
    files = [source/'dataset.npz', source/'beta_gp_report.json', source/'beta_gp_predictions.npz',
             source/source_report['selected_samples_file'], root/'scores_with_fixed_mu.csv']
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    provenance = folder/'source_provenance.json'
    if provenance.exists():
        if json.loads(provenance.read_text()) != hashes:
            raise ValueError('Source posterior or data changed; review provenance before reusing fit')
    else:
        write_json(provenance, hashes)
    if (folder/'dataset.npz').exists():
        with np.load(folder/'dataset.npz') as previous:
            for key in data:
                np.testing.assert_array_equal(data[key], previous[key])
    else:
        np.savez_compressed(folder/'dataset.npz', **data)
    np.savez_compressed(folder/'beta_estimate.npz', x=data['x'], beta_mean=beta_hat,
                        per_chain_beta_means=chain_means, source_draws=samples.shape[0]*samples.shape[1])
    write_json(folder/'experiment.json', dict(model=PLUGIN_BETA_MODEL, n_train=30, n_test=151,
               estimator='beta_hat(x)=E[beta(x)|y_train], averaging saved full Bayesian HMC states',
               source_samples=str((source/source_report['selected_samples_file']).relative_to(root)),
               source_draws=samples.shape[0]*samples.shape[1],
               likelihood='N(y_train; W*beta_hat_train, W*(K(v_0,w)+k*I))',
               learned_stage2=['v_0', 'k', 'w', 'W'], fixed_mu=0., fixed_gig=dict(l=1., omega=1.),
               prior_center=prior_center_from_training(data['y'][train]), prior_sd=2.,
               beta_uncertainty_propagated=False, W_uncertainty_propagated=True,
               truth_used_for_beta_estimation_or_fitting=False, heldout_y_used_for_fitting=False,
               method_note='Two-stage plug-in: same training data estimate beta and fit residual parameters. '
                           'Not full Bayes and not obtained by subtracting a beta variance from the old prediction. '
                           'Stage 2 uses original priors, not stage-1 posterior as a prior.'))
    return data, folder


def summarize(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    data, folder = prepare(root)
    case = folder/PLUGIN_BETA_MODEL
    report = json.loads((case/'report.json').read_text())
    previous = pd.read_csv(root/'scores_with_fixed_mu.csv')
    new = dict(report['scores'], scenario='sparse', n_signal_components=1,
               component_swap_acceptance=0., known_beta=False, uses_true_beta_information=False)
    if 'data_seed' in previous:
        seeds = previous['data_seed'].dropna().unique()
        if len(seeds) != 1:
            raise ValueError('Expected one data seed per experiment')
        new['data_seed'] = int(seeds[0])
    combined = pd.concat([previous, pd.DataFrame([new])], ignore_index=True)
    combined.to_csv(root/'scores_with_plugin_beta.csv', index=False)
    cols = ['f_rmse', 'f_mean_marginal_nlpd', 'y_rmse', 'y_mean_marginal_nlpd']
    labels = dict(TEX_LABELS, ghp_known_beta=r'GHP (true $\beta$ supplied)',
                  ghp_fixed_mu=r'GHP ($\mu=E[W]\beta_{\mathrm{true}}$)',
                  ghp_plugin_beta=r'GHP (fixed $\widehat\beta=E[\beta\mid y_{\mathrm{train}}]$)')
    lines = [r'% Requires \usepackage{booktabs}', r'\begin{table}[htbp]', r'\centering',
             r'\caption{Same simulated function, 30 training and 151 test points. '
             r'GHP fixes $\lambda=\omega=1$; STP fixes $\nu=5$. The final row estimates '
             r'a single beta function from training data and holds it fixed while refitting '
             r'the residual kernel, noise and $W$ with HMC. Only the true-beta and '
             r'fixed-$\mu$ rows receive the generating beta function.}',
             r'\label{tab:plugin_beta_30}', r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Model & RMSE ($f$) & NLPD ($f$) & RMSE ($y$) & NLPD ($y$) \\', r'\midrule']
    for row in combined.to_dict('records'):
        if row['model'] in ['gp_zero', 'ghp_known_beta', PLUGIN_BETA_MODEL]:
            lines.append(r'\midrule')
        flag = '' if row['diagnostics_ok'] else r'$^{\dagger}$'
        lines.append(labels[row['model']]+flag+' & '+' & '.join(f'{row[k]:.5f}' for k in cols)+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    if not combined['diagnostics_ok'].all():
        lines.append(r'\par\small $\dagger$ HMC diagnostics require investigation.')
    lines.append(r'\end{table}')
    (root/'comparison_with_plugin_beta.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    cases = [('beta_gp', root/'sparse', 'Full Bayesian beta GP'),
             (PLUGIN_BETA_MODEL, case, 'Estimated beta fixed; residual model refitted'),
             ('ghp_known_beta', root/'known_beta'/'ghp_known_beta', 'True beta supplied (extra information)')]
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.5), sharex=True, sharey=True, constrained_layout=True)
    for col, (model, source, title) in enumerate(cases):
        prefix = 'beta_gp_' if model == 'beta_gp' else ''
        rp = json.loads((source/f'{prefix}report.json').read_text())
        with np.load(source/f'{prefix}predictions.npz') as p:
            for row, target in enumerate(['f', 'y']):
                ax = axes[row, col]
                ax.plot(data['x'], data['f'], 'k--', lw=1.1, label='True f')
                ax.plot(data['x'], p[f'{target}_mean'], color=f'C{col}', label='Posterior mean')
                ax.fill_between(data['x'], *p[f'{target}_ci'].T, color=f'C{col}', alpha=.23,
                                label='Pointwise 95% interval')
                train, test = data['train_ids'], data['test_ids']
                ax.scatter(data['x'][train], data['y'][train], c='black', s=12, alpha=.65, label='Training y')
                if target == 'y':
                    ax.scatter(data['x'][test], data['y'][test], c='gray', marker='x', s=10, alpha=.5, label='Test y')
                score = rp['scores']
                ax.set(title=f'{title}\n{target} RMSE={score[target+"_rmse"]:.4f}; '
                             f'NLPD={score[target+"_mean_marginal_nlpd"]:.4f}', xlabel='x', ylabel=target)
                ax.grid(alpha=.18)
                if col == 0:
                    ax.legend(fontsize=7)
    fig.suptitle('Same 30 / 151 split | plug-in beta learned using only training observations')
    fig.savefig(folder/'regression.png', dpi=150)
    plt.close(fig)
    with np.load(root/'sparse'/'beta_gp_predictions.npz') as source:
        fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
        ax.fill_between(data['x'], *source['beta_ci'].T, color='C0', alpha=.2,
                        label='Full Bayesian beta: 95% pointwise interval (discarded in plug-in fit)')
        ax.plot(data['x'], data['beta_true'], 'k--', label='Generating beta: reference only')
        ax.plot(data['x'], data['beta'], color='C1', label='Estimated beta held fixed in stage 2')
        ax.scatter(data['x'][data['train_ids']], data['beta'][data['train_ids']], c='C1', s=18,
                   label='Estimate at training inputs (not observed beta values)')
        ax.set(xlabel='x', ylabel='beta(x)', title='One estimated beta function; no beta uncertainty propagated')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
        fig.savefig(folder/'beta_estimate.png', dpi=160)
        plt.close(fig)
    with np.load(case/report['selected_samples_file']) as saved:
        samples = saved['log_samples']
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), constrained_layout=True)
    for j, ax in enumerate(axes.flat):
        for chain in samples:
            ax.plot(chain[:, j], lw=.4, alpha=.6)
        ax.set_title('log '+report['parameter_names'][j]+f' | R-hat={report["diagnostics"]["rhat"][j]:.4f}')
    fig.savefig(folder/'traces.png', dpi=130)
    plt.close(fig)
    W = np.exp(samples[..., -1]).reshape(-1)
    write_json(folder/'summary.json', dict(scores=report['scores'], posterior_W_mean=W.mean(),
               posterior_W_std=W.std(), posterior_W_ci=np.quantile(W, [.025, .975]),
               beta_estimate_is_training_posterior_mean=True, beta_uncertainty_propagated=False,
               comparison=combined[['model', *cols]].to_dict('records')))
    print(combined[['model', *cols]].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('figure/single_function_30_hmc'))
    parser.add_argument('--stage', choices=['prepare', 'fit', 'summarize', 'all'], default='all')
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=2000)
    parser.add_argument('--draws', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=22026)
    parser.add_argument('--target-accept', type=float, default=.99)
    parser.add_argument('--max-retries', type=int, default=1)
    args = parser.parse_args()
    if args.max_retries < 0:
        parser.error('--max-retries must be nonnegative')
    with threadpool_limits(limits=1):
        data, folder = prepare(args.output_dir)
        if args.stage in ('fit', 'all'):
            fit_model(data, folder, PLUGIN_BETA_MODEL, chains=args.chains, warmup=args.warmup,
                      draws=args.draws, seed=args.seed, target_accept=args.target_accept, max_retries=args.max_retries)
        if args.stage in ('summarize', 'all'):
            summarize(args.output_dir)


if __name__ == '__main__':
    main()
