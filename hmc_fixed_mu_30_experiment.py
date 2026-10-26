"""Compare fixed mu(x)=E_prior[W]*beta_true(x) with the saved 30/151 fits.

The known beta is used solely to construct an externally fixed mean function.
The regression model has beta=0 and learns residual v_0, k, w, W using HMC.
It receives no true W, held-out y, or latent f in its inference likelihood.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special
from threadpoolctl import threadpool_limits

from GH_zero_mean_hmc import FIXED_MU_MODEL, prior_center_from_training
from hmc_1d_split_experiments import write_json
from hmc_zero_mean_30_experiment import fit_model, TEX_LABELS


def gig_prior_mean(l=1., omega=1.):
    """For p(W) proportional to W^(l-1) exp[-omega*(W+1/W)/2]."""
    if not np.isfinite(l) or not np.isfinite(omega) or omega <= 0:
        raise ValueError('Need finite l and positive omega')
    value = special.kve(l+1, omega)/special.kve(l, omega)
    if not np.isfinite(value) or value <= 0:
        raise ValueError('Could not compute a finite positive GIG mean')
    return float(value)


def prepare(root):
    root = Path(root)
    with np.load(root/'sparse'/'dataset.npz') as saved:
        data = {k: saved[k].copy() for k in saved.files}
    if len(data['train_ids']) != 30 or len(data['test_ids']) != 151:
        raise ValueError('Expected the original 30/151 split')
    coefficient = gig_prior_mean()
    data['mu_fixed'] = coefficient*data['beta']
    folder = root/'fixed_mu_prior_mean'
    folder.mkdir(parents=True, exist_ok=True)
    sources = [root/'sparse'/'dataset.npz', root/'scores_with_known_beta.csv',
               root/'known_beta'/'ghp_known_beta'/'report.json']
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    provenance = folder/'source_provenance.json'
    if provenance.exists():
        if json.loads(provenance.read_text()) != hashes:
            raise ValueError('Earlier data or comparisons changed; review provenance')
    else:
        write_json(provenance, hashes)
    dataset = folder/'dataset.npz'
    if dataset.exists():
        with np.load(dataset) as previous:
            for key in data:
                np.testing.assert_array_equal(data[key], previous[key])
    else:
        np.savez_compressed(dataset, **data)
    write_json(folder/'experiment.json', dict(model=FIXED_MU_MODEL, n_train=30, n_test=151,
               mu='E_prior[W] * saved true beta(x)', fixed_mu_multiplier=coefficient, beta=0.,
               mean_source='GIG prior only, not fitted posterior or generating W',
               likelihood='N(y_train; mu_fixed_train, W*(K(v_0,w)+k*I))',
               learned_parameters=['v_0', 'k', 'w', 'W'], fixed_gig=dict(l=1., omega=1.),
               prior_center=prior_center_from_training(data['y'][data['train_ids']]), prior_sd=2.,
               priors_use_original_training_y=True, additional_centering=False, scaling=False,
               generating_W_for_reference_only=float(data['W']),
               fit_inputs=['x[train_ids]', 'y[train_ids]', 'mu_fixed[train_ids]'],
               comparison_note='Receives the true beta function as extra information, just like known-beta fit'))
    return data, folder


def summarize(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    data, folder = prepare(root)
    report = json.loads((folder/FIXED_MU_MODEL/'report.json').read_text())
    previous = pd.read_csv(root/'scores_with_known_beta.csv')
    row = dict(report['scores'], scenario='sparse', n_signal_components=1,
               component_swap_acceptance=0., known_beta=False)
    combined = pd.concat([previous, pd.DataFrame([row])], ignore_index=True)
    combined['uses_true_beta_information'] = combined['model'].isin(['ghp_known_beta', FIXED_MU_MODEL])
    combined.to_csv(root/'scores_with_fixed_mu.csv', index=False)
    columns = ['f_rmse', 'f_mean_marginal_nlpd', 'y_rmse', 'y_mean_marginal_nlpd']
    labels = dict(TEX_LABELS, ghp_known_beta=r'GHP (known $\beta$, $\mu=0$)',
                  ghp_fixed_mu=r'GHP ($\mu=E[W]\beta_{\mathrm{true}}$, $\beta=0$)')
    lines = [r'% Requires \usepackage{booktabs}', r'\begin{table}[htbp]', r'\centering',
             r'\caption{Same simulated function, with 30 training and 151 test points. '
             r'Kernel/noise parameters and applicable $W$ are inferred using HMC. '
             r'GHP fixes $\lambda=\omega=1$; STP fixes $\nu=5$. '
             r'The final two rows receive the true beta function. The last row fixes '
             r'$\mu(x)=E_{p(W)}[W]\beta_{\mathrm{true}}(x)$, with $E_{p(W)}[W]=2.69948$; '
             r'the residual mixing variable $W$ remains inferred.}',
             r'\label{tab:fixed_mu_30}', r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Model & RMSE ($f$) & NLPD ($f$) & RMSE ($y$) & NLPD ($y$) \\', r'\midrule']
    for row in combined.to_dict('records'):
        if row['model'] in ['gp_zero', 'ghp_known_beta']:
            lines.append(r'\midrule')
        flag = '' if row['diagnostics_ok'] else r'$^{\dagger}$'
        lines.append(labels[row['model']]+flag+' & '+' & '.join(f'{row[k]:.5f}' for k in columns)+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    if not combined['diagnostics_ok'].all():
        lines.append(r'\par\small $\dagger$ HMC diagnostics require investigation.')
    lines.append(r'\end{table}')
    (root/'comparison_with_fixed_mu.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')

    cases = [('beta_gp', root/'sparse', 'Beta inferred with GP prior'),
             ('ghp_known_beta', root/'known_beta'/'ghp_known_beta', 'Known beta; mean = W beta'),
             (FIXED_MU_MODEL, folder/FIXED_MU_MODEL, 'Fixed mean = prior E[W] beta')]
    fig, axes = plt.subplots(2, 3, figsize=(16, 7.5), sharex=True, sharey=True, constrained_layout=True)
    posterior_W = {}
    for col, (model, source, title) in enumerate(cases):
        prefix = 'beta_gp_' if model == 'beta_gp' else ''
        this_report = json.loads((source/f'{prefix}report.json').read_text())
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
                score = this_report['scores']
                ax.set(title=f'{title}\n{target} RMSE={score[target+"_rmse"]:.4f}; '
                             f'NLPD={score[target+"_mean_marginal_nlpd"]:.4f}', xlabel='x', ylabel=target)
                ax.grid(alpha=.18)
                if col == 0:
                    ax.legend(fontsize=7)
        with np.load(source/this_report['selected_samples_file']) as samples:
            posterior_W[title] = np.exp(samples['log_samples'][..., -1]).reshape(-1)
    fig.suptitle('Same 30 / 151 split | last two models receive the true beta function')
    fig.savefig(folder/'regression.png', dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), constrained_layout=True)
    axes[0].plot(data['x'], float(data['W'])*data['beta'], 'k--', label='Generating W beta (reference only)')
    axes[0].plot(data['x'], data['mu_fixed'], color='C2', label='Fixed mu = prior E[W] beta')
    for title, W in posterior_W.items():
        if title.startswith('Known'):
            axes[0].plot(data['x'], W.mean()*data['beta'], color='C1', label='Known-beta posterior mean of W beta')
    axes[0].set(xlabel='x', ylabel='Mean contribution', title='Replacing W by its prior mean')
    axes[0].legend(fontsize=8)
    for i, (title, W) in enumerate(posterior_W.items()):
        axes[1].hist(W, bins=100, density=True, histtype='step', color=f'C{i}',
                     label=f'{title}: {W.mean():.3f} +/- {W.std():.3f}')
    axes[1].axvline(float(data['W']), c='black', ls='--', label='Generating W (reference only)')
    axes[1].axvline(gig_prior_mean(), c='C2', ls=':', label='Prior E[W] used to fix mu')
    axes[1].set(xlabel='W', ylabel='Posterior density', title='Residual W remains inferred',
                xlim=(0, max(np.quantile(W, .995) for W in posterior_W.values())))
    axes[1].legend(fontsize=7)
    fig.savefig(folder/'mean_and_W.png', dpi=150)
    plt.close(fig)
    with np.load(folder/FIXED_MU_MODEL/report['selected_samples_file']) as saved:
        samples = saved['log_samples']
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), constrained_layout=True)
    for j, ax in enumerate(axes.flat):
        for chain in samples:
            ax.plot(chain[:, j], lw=.4, alpha=.6)
        ax.set_title('log '+report['parameter_names'][j]+f' | R-hat={report["diagnostics"]["rhat"][j]:.4f}')
    fig.savefig(folder/'traces.png', dpi=130)
    plt.close(fig)
    W = np.exp(samples[..., -1]).reshape(-1)
    write_json(folder/'summary.json', dict(scores=report['scores'], prior_W_mean=gig_prior_mean(),
               posterior_W_mean=W.mean(), posterior_W_std=W.std(), posterior_W_variance=W.var(),
               posterior_W_ci=np.quantile(W, [.025, .975]), generating_W=float(data['W']),
               comparison=combined[['model', *columns]].to_dict('records')))
    print(combined[['model', *columns]].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('figure/single_function_30_hmc'))
    parser.add_argument('--stage', choices=['prepare', 'fit', 'summarize', 'all'], default='all')
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=2000)
    parser.add_argument('--draws', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=21026)
    parser.add_argument('--target-accept', type=float, default=.99)
    parser.add_argument('--max-retries', type=int, default=1)
    args = parser.parse_args()
    if args.max_retries < 0:
        parser.error('--max-retries must be nonnegative')
    with threadpool_limits(limits=1):
        data, folder = prepare(args.output_dir)
        if args.stage in ('fit', 'all'):
            fit_model(data, folder, FIXED_MU_MODEL, chains=args.chains, warmup=args.warmup,
                      draws=args.draws, seed=args.seed, target_accept=args.target_accept, max_retries=args.max_retries)
        if args.stage in ('summarize', 'all'):
            summarize(args.output_dir)


if __name__ == '__main__':
    main()
