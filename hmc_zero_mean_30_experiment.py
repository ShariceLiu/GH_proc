"""Add three single-SE zero-mean baselines to the saved 30-point experiment.

The original four fits are read, not refitted or overwritten. New results go
under figure/single_function_30_hmc/zero_mean_baselines; combined seven-model
tables/plots have *_with_zero_mean names in the original experiment folder.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from GH_zero_mean_hmc import (MODELS, KNOWN_BETA_MODEL, FIXED_MU_MODEL, PLUGIN_BETA_MODEL, BETA_MODELS,
                               ALL_MODELS, MU_MODELS,
                               prior_center_from_training, sample_parameters,
                               predict_mixture, diagnostics_summary)
from GP_TP_hmc import mixture_logpdf
from hmc_1d_split_experiments import write_json, make_split
from hmc_beta_mu_1d_experiment import load_dataset
from hmc_gp_tp_1d_experiment import score_predictions

OLD_MODELS = ('gp', 'tp_mu_gp', 'mu_gp', 'beta_gp')
LABELS = dict(gp='GP (two SE components)', tp_mu_gp='STP + GP mean', mu_gp='GHP (GP mean)',
              beta_gp='GHP (GP beta)', gp_zero='GP (mu=0, one SE)',
              stp_zero='STP (mu=0, one SE)', ghp_zero='GHP (mu=beta=0, one SE)')
TEX_LABELS = dict(gp='GP (two SE)', tp_mu_gp='STP + GP mean', mu_gp=r'GHP ($\mu$ GP)',
                  beta_gp=r'GHP ($\beta$ GP)', gp_zero=r'GP ($\mu=0$, one SE)',
                  stp_zero=r'STP ($\mu=0$, one SE)', ghp_zero=r'GHP ($\mu=\beta=0$, one SE)')


def prepare(root):
    root = Path(root)
    source = root/'sparse'/'dataset.npz'
    with np.load(source) as saved:
        data = {key: saved[key].copy() for key in saved.files}
    expected = make_split(load_dataset(), 'sparse', sparse_points=30)
    for key in ['x', 'y', 'f', 'beta', 'W', 'train_ids', 'test_ids']:
        np.testing.assert_array_equal(data[key], expected[key])
    if len(data['train_ids']) != 30 or len(data['test_ids']) != 151:
        raise ValueError('This extension requires the saved 30/151 split')
    folder = root/'zero_mean_baselines'
    folder.mkdir(parents=True, exist_ok=True)
    new_data = folder/'dataset.npz'
    if new_data.exists():
        with np.load(new_data) as previous:
            for key in data:
                np.testing.assert_array_equal(data[key], previous[key])
    else:
        np.savez_compressed(new_data, **data)
    files = [root/'scores.csv', source]
    files.extend(root/'sparse'/f'{model}_report.json' for model in OLD_MODELS)
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    provenance = folder/'source_provenance.json'
    if provenance.exists():
        if json.loads(provenance.read_text()) != hashes:
            raise ValueError('Original data or fits changed; review comparison provenance')
    else:
        write_json(provenance, hashes)
    write_json(folder/'experiment.json', dict(n_train=30, n_test=151, mu=0., beta=0.,
               kernel='v_0 * exp(-0.5*w*(x-xprime)^2)', n_signal_components=1,
               fixed_priors=dict(GHP=dict(l=1., omega=1.), STP=dict(nu=5., ig_shape=2.5, ig_scale=1.5)),
               prior_center=prior_center_from_training(data['y'][data['train_ids']]), prior_sd=2.,
               prior_convention='Same remaining residual-kernel/noise priors as the original four fits',
               centering=False, scaling=False, model_names=list(MODELS)))
    return data, folder


def fit_model(data, base_folder, model, *, chains=4, warmup=2000, draws=3000,
              seed=19026, target_accept=.99, max_retries=1):
    folder = base_folder/model
    folder.mkdir(parents=True, exist_ok=True)
    report_path = folder/'report.json'
    if report_path.exists():
        old = json.loads(report_path.read_text())
        if old['diagnostics_ok'] and (folder/'predictions.npz').exists():
            print(f'{model}: reusing completed fit', flush=True)
            return old
    train = data['train_ids']
    x, y = data['x'][train], data['y'][train]
    attempts_path = folder/'attempts.json'
    attempts = json.loads(attempts_path.read_text()) if attempts_path.exists() else []
    started = time.perf_counter()
    for retry in range(max_retries+1):
        number = len(attempts)+1
        model_index = ALL_MODELS.index(model)
        settings = dict(chains=chains, warmup=warmup*2**retry, draws=draws*2**retry,
                        target_accept=min(.999, target_accept+.005*retry), leapfrog_steps=(20, 50),
                        seed=seed+100*model_index+10000*(number-1))
        print(f'{model}: attempt {number}, {settings}', flush=True)
        known_beta = dict(beta=data['beta'][train]) if model in BETA_MODELS else {}
        if model in MU_MODELS:
            known_beta = dict(mu=data['mu_fixed'][train])
        fit = sample_parameters(y, x, model=model, **settings, **known_beta)
        diagnostic = diagnostics_summary(fit)
        np.savez_compressed(folder/f'attempt_{number}_samples.npz', parameter_names=fit['parameter_names'],
                            **{key: value for key, value in fit.items() if isinstance(value, np.ndarray)})
        attempt = dict(attempt=number, settings=settings, **diagnostic,
                       diagnostics=fit['diagnostics'], effective_diagnostics=fit['effective_diagnostics'],
                       warmup_divergences=int(np.sum(fit['warmup_divergent'])))
        attempts.append(attempt)
        write_json(attempts_path, attempts)
        print(json.dumps(diagnostic), flush=True)
        if fit['diagnostics_ok']:
            break
    sampling_seconds = time.perf_counter()-started
    print(f'{model}: predicting from all {chains*settings["draws"]} retained states', flush=True)
    known_beta = dict(beta_train=data['beta'][train], beta_pred=data['beta']) if model in BETA_MODELS else {}
    if model in MU_MODELS:
        known_beta = dict(mu_train=data['mu_fixed'][train], mu_pred=data['mu_fixed'])
    prediction = predict_mixture(y, x, data['x'], fit, **known_beta)
    np.savez_compressed(folder/'predictions.npz', **prediction)
    scores = dict(model=model, n_train=30, n_test=151, **score_predictions(data, prediction), **diagnostic)
    chain_scores = []
    for chain in range(chains):
        section = slice(chain*settings['draws'], (chain+1)*settings['draws'])
        means = prediction['component_means'][section][:, data['test_ids']]
        row = dict(chain=chain+1)
        for target in ['f', 'y']:
            variances = prediction[f'component_{target}_variances'][section][:, data['test_ids']]
            truth = data[target][data['test_ids']]
            row[f'{target}_rmse'] = float(np.sqrt(np.mean((means.mean(axis=0)-truth)**2)))
            row[f'{target}_mean_marginal_nlpd'] = float(-mixture_logpdf(truth, means, variances).mean())
        chain_scores.append(row)
    pd.DataFrame(chain_scores).to_csv(folder/'per_chain_scores.csv', index=False)
    pd.DataFrame(fit['summary']).to_csv(folder/'parameters.csv', index=False)
    report = dict(model=model, scores=scores, **diagnostic, attempts=attempts, selected_attempt=number,
                  selected_samples_file=f'attempt_{number}_samples.npz', settings=settings,
                  sampling_seconds=sampling_seconds, total_seconds=time.perf_counter()-started,
                  **{key: fit[key] for key in ['prior_center', 'prior_sd', 'summary', 'parameter_names',
                     'diagnostics', 'effective_diagnostics', 'initialization', 'step_size', 'inverse_mass']})
    write_json(report_path, report)
    print(json.dumps(scores), flush=True)
    return report


def summarize(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    root = Path(root)
    data, new_folder = prepare(root)
    rows, predictions, reports = [], {}, {}
    for model in (*OLD_MODELS, *MODELS):
        old = model in OLD_MODELS
        report_path = root/'sparse'/f'{model}_report.json' if old else new_folder/model/'report.json'
        pred_path = root/'sparse'/f'{model}_predictions.npz' if old else new_folder/model/'predictions.npz'
        if not report_path.exists():
            continue
        report = json.loads(report_path.read_text())
        reports[model] = report
        rows.append(dict(report['scores'], scenario='sparse', n_signal_components=2 if old else 1,
                         component_swap_acceptance=report['scores'].get('component_swap_acceptance', 0.)))
        with np.load(pred_path) as p:
            predictions[model] = {f'{t}_{stat}': p[f'{t}_{stat}'].copy() for t in ['f', 'y'] for stat in ['mean', 'ci']}
    pd.DataFrame(rows).to_csv(root/'scores_with_zero_mean.csv', index=False)
    pd.DataFrame([row for row in rows if row['model'] in MODELS]).to_csv(new_folder/'scores.csv', index=False)
    write_json(new_folder/'summary.json', dict(completed_new_models=sum(r['model'] in MODELS for r in rows),
               all_diagnostics_ok=all(r['diagnostics_ok'] for r in rows), scores=rows))
    columns = ['f_rmse', 'f_mean_marginal_nlpd', 'y_rmse', 'y_mean_marginal_nlpd']
    lines = [r'% Requires \usepackage{booktabs}', r'\begin{table}[htbp]', r'\centering',
             r'\caption{The same simulated function: 30 training and 151 test points. '
             r'Kernel/noise parameters are inferred with HMC. GHP fixes $\lambda=\omega=1$; '
             r'STP fixes $\nu=5$. The added zero-mean models use one SE kernel.}',
             r'\label{tab:single_function_zero_mean}', r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Model & RMSE ($f$) & NLPD ($f$) & RMSE ($y$) & NLPD ($y$) \\', r'\midrule']
    for row in rows:
        model = row['model']
        if model == MODELS[0]:
            lines.append(r'\midrule')
        flag = '' if row['diagnostics_ok'] else r'$^{\dagger}$'
        lines.append(TEX_LABELS[model]+flag+' & '+' & '.join(f'{row[k]:.5f}' for k in columns)+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}']
    if not all(r['diagnostics_ok'] for r in rows):
        lines.append(r'\par\small $\dagger$ HMC diagnostics require investigation.')
    lines.append(r'\end{table}')
    (root/'comparison_with_zero_mean.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    fig, axes = plt.subplots(len(rows), 2, figsize=(13, 2.6*len(rows)), sharex=True, sharey=True,
                              squeeze=False, constrained_layout=True)
    for i, row in enumerate(rows):
        model, p = row['model'], predictions[row['model']]
        for col, target in enumerate(['f', 'y']):
            ax = axes[i, col]
            ax.plot(data['x'], data['f'], 'k--', lw=1., label='True f')
            ax.plot(data['x'], p[f'{target}_mean'], color=f'C{i}', label='Posterior mean')
            ax.fill_between(data['x'], *p[f'{target}_ci'].T, color=f'C{i}', alpha=.22,
                            label='95% pointwise '+('credible' if target == 'f' else 'predictive')+' interval')
            ax.scatter(data['x'][data['train_ids']], data['y'][data['train_ids']], c='black', s=9,
                       alpha=.65, label='Training y')
            if target == 'y':
                ax.scatter(data['x'][data['test_ids']], data['y'][data['test_ids']], c='gray',
                           marker='x', s=8, alpha=.4, label='Held-out y')
            ax.set(title=f'{LABELS[model]} | {target} RMSE={row[target+"_rmse"]:.4f}; '
                         f'NLPD={row[target+"_mean_marginal_nlpd"]:.4f}', ylabel=target)
            ax.grid(alpha=.18)
            if i == 0:
                ax.legend(fontsize=7, ncol=2)
    for ax in axes[-1]:
        ax.set_xlabel('x')
    fig.suptitle('Same saved function and 30 / 151 split | two-SE models and added one-SE baselines')
    fig.savefig(root/'regression_with_zero_mean.png', dpi=150)
    plt.close(fig)
    new_rows = [row for row in rows if row['model'] in MODELS]
    if new_rows:
        fig, axes = plt.subplots(4, len(new_rows), figsize=(4.5*len(new_rows), 9),
                                  squeeze=False, constrained_layout=True)
        for col, row in enumerate(new_rows):
            model, report = row['model'], reports[row['model']]
            with np.load(new_folder/model/report['selected_samples_file']) as saved:
                samples = saved['log_samples']
            for j in range(4):
                ax = axes[j, col]
                if j >= samples.shape[-1]:
                    ax.set_visible(False)
                    continue
                for chain in samples:
                    ax.plot(chain[:, j], alpha=.65, lw=.4)
                ax.set_title(f'{LABELS[model]}: log {report["parameter_names"][j]}\n'
                             f'R-hat={report["diagnostics"]["rhat"][j]:.4f}', fontsize=9)
        fig.savefig(new_folder/'traces.png', dpi=130)
        plt.close(fig)
    print(pd.DataFrame(rows)[['model', *columns]].to_string(index=False), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=Path('figure/single_function_30_hmc'))
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--stage', choices=['prepare', 'fit', 'summarize', 'all'], default='all')
    parser.add_argument('--chains', type=int, default=4)
    parser.add_argument('--warmup', type=int, default=2000)
    parser.add_argument('--draws', type=int, default=3000)
    parser.add_argument('--seed', type=int, default=19026)
    parser.add_argument('--target-accept', type=float, default=.99)
    parser.add_argument('--max-retries', type=int, default=1)
    args = parser.parse_args()
    if args.max_retries < 0:
        parser.error('--max-retries must be nonnegative')
    with threadpool_limits(limits=1):
        data, folder = prepare(args.output_dir)
        if args.stage in ('fit', 'all'):
            for model in args.models:
                fit_model(data, folder, model, chains=args.chains, warmup=args.warmup, draws=args.draws,
                          seed=args.seed, target_accept=args.target_accept, max_retries=args.max_retries)
        if args.stage in ('summarize', 'all'):
            summarize(args.output_dir)


if __name__ == '__main__':
    main()
