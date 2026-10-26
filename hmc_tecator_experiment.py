"""Four-model Bayesian curve regression on Tecator; stages are resumable.

python hmc_tecator_experiment.py --stage all
Training uses all wavelengths of original C+M curves. Use one representative
per identical composition triple (conservative repeat-measurement grouping).
Original T curves whose group occurs in C+M are excluded. E1/E2 are reserved.
The training-only mean curve and common scale are fixed preprocessing estimates.

Training integrates out the Gaussian functions and jointly samples shared
kernel/noise parameters and each training curve's W with HMC. For each test
curve, only its revealed observations update the shared-parameter posterior;
its new W is integrated by checked log-W quadrature. Predictive Gaussian
mixtures provide pointwise intervals and scores at the unrevealed wavelengths.
"""

import argparse
import hashlib
import json
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import special
from threadpoolctl import threadpool_limits

from explore_tecator import load_tecator
from GH_multiple_hmc import MODELS, sample_parameters, context_log_weights, resample_context
from GH_tabular_hmc import predict_mixture, conditional_marginals
from GP_TP_hmc import mixture_logpdf
from predictive_crps import gaussian_mixture_crps
from hmc_gp_tp_1d_experiment import diagnostics_summary
from hmc_multiple_functions_experiment import LABELS, COLORS, write_json


ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT = ROOT/'figure'/'tecator_multiple_functions_hmc'
PRIOR_CENTER = np.log([.5, .5, .01, 1., .04])
PRIOR_SD = 2.


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare(folder, *, context_count=15, location_seed=24026):
    folder.mkdir(parents=True, exist_ok=True)
    raw = load_tecator()
    settings = dict(source_sha256=raw['source_sha256'], context_count=context_count,
                    location_seed=location_seed,
                    grouping='First source record per identical (moisture, fat, protein) triple; '
                             'a conservative grouping proxy, not a verified specimen identifier.',
                    split='Original C+M training, T testing; remove groups shared across splits. '
                          'E1/E2 reserved and not scored in this experiment.',
                    inputs='Wavelength mapped linearly from [850,1050] nm to [0,10].',
                    outputs='Subtract mean of training curves at each wavelength and divide by '
                            'one training-residual standard deviation. No per-curve normalization.',
                    mean_uncertainty='Training common mean/scale are fixed preprocessing estimates.',
                    mixing=dict(ghp_lambda=1., ghp_omega=1., stp_nu=5.),
                    prior_log_center=PRIOR_CENTER.tolist(), prior_log_sd=PRIOR_SD,
                    train_observations_per_curve=100)
    if (folder/'dataset.npz').exists():
        if json.loads((folder/'settings.json').read_text()) != settings:
            raise ValueError('Settings changed: use a new output folder.')
        return
    if not 2 <= context_count < 100:
        raise ValueError('Need between 2 and 99 revealed wavelengths.')
    samples = raw['samples'].copy()
    composition = samples[['moisture_pct', 'fat_pct', 'protein_pct']]
    representative = ~composition.duplicated()
    train = np.flatnonzero(representative & samples.benchmark_split.eq('train'))
    test = np.flatnonzero(representative & samples.benchmark_split.eq('test'))
    if np.intersect1d(samples.spectrum_group_id.iloc[train],
                      samples.spectrum_group_id.iloc[test]).size:
        raise ValueError('Duplicate spectra across train/test.')
    y = raw['spectra']
    mean = y[train].mean(axis=0)
    scale = float(np.std(y[train]-mean))
    rng = np.random.default_rng(location_seed)
    context = np.stack([np.sort(rng.choice(100, context_count, replace=False)) for _ in test])
    evaluation = np.stack([np.setdiff1d(np.arange(100), c) for c in context])
    np.savez_compressed(folder/'dataset.npz', wavelength_nm=raw['wavelength_nm'],
                        x=(raw['wavelength_nm']-850)/20, y=y, residuals=(y-mean)/scale,
                        common_mean=mean, common_scale=scale, train_rows=train, test_rows=test,
                        context_ids=context, evaluation_ids=evaluation)
    samples['experiment_role'] = 'excluded_repeat_group'
    samples.loc[samples.benchmark_split.eq('extrapolation'), 'experiment_role'] = 'reserved_extrapolation'
    samples.loc[train, 'experiment_role'] = 'train'
    samples.loc[test, 'experiment_role'] = 'test'
    samples.to_csv(folder/'sample_roles.csv', index=False)
    locations = [dict(sample_id=int(row+1), channel=int(i+1), wavelength_nm=raw['wavelength_nm'][i])
                 for row, ids in zip(test, context) for i in ids]
    pd.DataFrame(locations).to_csv(folder/'observed_locations.csv', index=False)
    write_json(folder/'settings.json', settings)
    print(f'Prepared {len(train)} training curves x 100 wavelengths; {len(test)} test curves '
          f'x {context_count} observed + {100-context_count} held out.', flush=True)


def load_data(folder):
    with np.load(folder/'dataset.npz', allow_pickle=False) as saved:
        return {key: saved[key].copy() for key in saved.files}


def fit_model(folder, data, model, *, warmup=1500, draws=2000, max_attempts=3):
    destination = folder/model
    destination.mkdir(parents=True, exist_ok=True)
    data_hash = digest(folder/'dataset.npz')
    if (destination/'fit.json').exists():
        old = json.loads((destination/'fit.json').read_text())
        if old['dataset_sha256'] != data_hash:
            raise ValueError('Saved fit belongs to another dataset.')
        if old['diagnostics_ok']:
            print(f'{model}: reusing completed fit.', flush=True)
            return
    attempts_file = destination/'attempts.json'
    attempts = json.loads(attempts_file.read_text()) if attempts_file.exists() else []
    groups = [(data['x'], data['residuals'][data['train_rows']])]
    for retry in range(len(attempts), max_attempts):
        number = retry+1
        started = time.perf_counter()
        settings = dict(warmup=warmup*number, draws=draws*number, chains=4,
                        seed=25026+MODELS.index(model)+1000*retry,
                        target_accept=min(.99, .95+.02*retry),
                        leapfrog_steps=(15, 35) if retry == 0 else (25, 55),
                        backend='shared_grid', prior_center=PRIOR_CENTER,
                        prior_sd=PRIOR_SD, l=1., omega=1., nu=5., chain_workers=1,
                        chain_processes=2, chain_log_dir=str(destination/f'attempt_{number}_chains'))
        print(f'{model}: fitting {len(data["train_rows"])} curves; attempt {number}.', flush=True)
        fit = sample_parameters(groups, model=model, **settings)
        filename = f'attempt_{number}_samples.npz'
        np.savez_compressed(destination/filename, parameter_names=fit['parameter_names'],
                            **{key: value for key, value in fit.items() if isinstance(value, np.ndarray)})
        attempt = dict(attempt=number, settings=settings, seconds=time.perf_counter()-started,
                       warmup_mode_jumps=int(fit['warmup_mode_jump_accepted'].sum()),
                       retained_mode_jumps=int(fit['mode_jump_accepted'].sum()),
                       warmup_divergences=int(fit['warmup_divergent'].sum()), **diagnostics_summary(fit))
        attempts.append(attempt)
        write_json(attempts_file, attempts)
        report = dict(model=model, dataset_sha256=data_hash, selected_samples=filename,
                      samples_sha256=digest(destination/filename), attempts=attempts,
                      training_sample_ids=(data['train_rows']+1).tolist(),
                      mode_transforms=fit['mode_transforms'],
                      **{key: fit[key] for key in ['summary', 'diagnostics', 'effective_diagnostics',
                          'diagnostics_ok', 'prior_center', 'prior_sd', 'initialization']})
        write_json(destination/'fit.json', report)
        pd.DataFrame(fit['summary']).to_csv(destination/'parameters.csv', index=False)
        print(json.dumps(attempt, default=lambda a: a.tolist()), flush=True)
        if fit['diagnostics_ok']:
            return
    raise RuntimeError(f'{model}: HMC diagnostics failed after {max_attempts} attempts.')


def checked_weights(theta, x, y, model):
    """Convergence-check evidence for each theta and mass at log-W boundaries."""
    options = dict(model=model, backend='shared_grid', chunk_size=64)
    nodes = 192
    t, logw = context_log_weights(theta, x, y[None], quadrature_nodes=nodes, **options)
    error = 0.
    if model != 'gp':
        for _ in range(4):
            fine_t, fine = context_log_weights(theta, x, y[None], quadrature_nodes=nodes*2, **options)
            error = float(np.max(np.abs(special.logsumexp(logw, axis=2)
                                        -special.logsumexp(fine, axis=2))))
            t, logw, nodes = fine_t, fine, nodes*2
            if error < 1e-5:
                break
        if error >= 1e-5:
            raise ValueError(f'W quadrature not converged: {error}')
    return t, logw[0], error


def score_prediction(y, prediction, ids):
    means, variance = prediction['component_means'][:, ids], prediction['component_y_variances'][:, ids]
    nlpd = -mixture_logpdf(y[ids], means, variance, weights=prediction['weights'])
    crps, diagnostic = gaussian_mixture_crps(y[ids], means, variance,
                                            prediction['weights'], atol=1e-8, rtol=1e-8)
    ci = prediction['y_ci'][ids]
    se = (prediction['y_mean'][ids]-y[ids])**2
    scores = dict(y_rmse=float(np.sqrt(se.mean())), y_nlpd=float(nlpd.mean()),
                  y_crps=float(crps.mean()),
                  y_coverage_95=float(np.mean((y[ids] >= ci[:, 0]) & (y[ids] <= ci[:, 1]))),
                  y_interval_width=float(np.mean(ci[:, 1]-ci[:, 0])),
                  crps_max_integration_error=float(np.max(diagnostic['integration_error_estimate'])))
    return scores, dict(squared_error=se, nlpd=nlpd, crps=crps)


def predict_model(folder, data, model, *, draws=6000):
    destination = folder/model
    fit = json.loads((destination/'fit.json').read_text())
    if not fit['diagnostics_ok'] or fit['dataset_sha256'] != digest(folder/'dataset.npz'):
        raise ValueError('Need a converged, matching training fit.')
    sample_file = destination/fit['selected_samples']
    if digest(sample_file) != fit['samples_sha256']:
        raise ValueError('Posterior sample file changed.')
    with np.load(sample_file) as saved:
        theta = saved['theta_log_samples'].reshape(-1, 5)
    rows, diagnostics = [], []
    for j, row in enumerate(data['test_rows']):
        sample_id = int(row+1)
        stem = destination/f'sample_{sample_id:03d}'
        record_file = stem.with_suffix('.json')
        if record_file.exists():
            record = json.loads(record_file.read_text())
            if (record['fit_sha256'] != fit['samples_sha256']
                    or record['prediction_draws'] != draws):
                raise ValueError('Prediction cache settings changed; use a new folder.')
            rows.append(record['scores']); diagnostics.append(record['diagnostics'])
            continue
        ids, evaluation = data['context_ids'][j], data['evaluation_ids'][j]
        x, y = data['x'][ids], data['residuals'][row, ids]
        print(f'{model}: test curve {j+1}/{len(data["test_rows"])}, source sample {sample_id}.', flush=True)
        t, logw, error = checked_weights(theta, x, y, model)
        seed = 26026+1000*MODELS.index(model)+sample_id
        q, diag = resample_context(theta, logw, t, model=model, draws=draws, seed=seed)
        diag.update(quadrature_nodes=len(t), max_quadrature_log_error=error, seed=seed)
        if diag['theta_importance_ess'] < max(200., .03*len(theta)) or diag['edge_mass'] > 1e-8:
            raise ValueError(f'Context inference needs refinement: {diag}')
        prediction = predict_mixture(y, x, data['x'], dict(model=model, log_samples=q[None]),
                                     output_scale=float(data['common_scale']), verbose=False)
        prediction['component_means'] += data['common_mean']
        for target in ['f', 'y']:
            prediction[f'{target}_mean'] += data['common_mean']
            prediction[f'{target}_ci'] += data['common_mean'][:, None]
        scores, pointwise = score_prediction(data['y'][row], prediction, evaluation)
        scores = dict(model=model, sample_id=sample_id, n_test=len(evaluation), **scores)
        if model != 'gp':
            posterior = np.exp(logw-special.logsumexp(logw))
            W = np.exp(t)
            wm = float(np.sum(posterior*W))
            diag.update(W_mean=wm, W_std=float(np.sqrt(np.sum(posterior*(W-wm)**2))))
        np.savez_compressed(stem.with_suffix('.npz'), posterior_log_parameters=q,
                            context_ids=ids, evaluation_ids=evaluation,
                            **prediction, **pointwise)
        record = dict(fit_sha256=fit['samples_sha256'], prediction_draws=draws,
                      scores=scores, diagnostics=diag)
        write_json(record_file, record)
        rows.append(scores); diagnostics.append(diag)
        pd.DataFrame(rows).to_csv(destination/'per_function_scores.csv', index=False)
        print(f'  RMSE={scores["y_rmse"]:.6f}, NLPD={scores["y_nlpd"]:.4f}, '
              f'CRPS={scores["y_crps"]:.6f}; theta ESS={diag["theta_importance_ess"]:.0f}', flush=True)
    pd.DataFrame(rows).to_csv(destination/'per_function_scores.csv', index=False)
    write_json(destination/'prediction_diagnostics.json', diagnostics)


def wait_for_fit(folder, model):
    """Continue predictions as soon as the independently running fit passes."""
    path = folder/model/'fit.json'
    print(f'{model}: waiting for a converged training checkpoint.', flush=True)
    started = time.monotonic()
    while time.monotonic()-started < 8*3600:
        if path.exists():
            try:
                report = json.loads(path.read_text())
            except json.JSONDecodeError:
                time.sleep(1.)
                continue
            if report['diagnostics_ok']:
                return
            if len(report['attempts']) >= 3:
                raise RuntimeError(f'{model}: completed fits have not passed diagnostics.')
        time.sleep(10.)
    raise TimeoutError(f'{model}: no converged fit checkpoint arrived.')


def summarize(folder, data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    all_rows, summary, reports = [], [], {}
    for model in MODELS:
        scores = pd.read_csv(folder/model/'per_function_scores.csv')
        if scores.sample_id.tolist() != (data['test_rows']+1).tolist():
            raise ValueError(f'{model}: incomplete test results.')
        all_rows.append(scores)
        report = json.loads((folder/model/'fit.json').read_text())
        diagnostics = json.loads((folder/model/'prediction_diagnostics.json').read_text())
        reports[model] = dict(hmc=report['attempts'][-1],
                             min_context_theta_ess=min(d['theta_importance_ess'] for d in diagnostics),
                             max_quadrature_error=max(d['max_quadrature_log_error'] for d in diagnostics))
        summary.append(dict(model=model, n_test_functions=len(scores),
                            y_rmse=float(np.sqrt(np.mean(scores.y_rmse**2))),
                            y_nlpd=float(scores.y_nlpd.mean()), y_crps=float(scores.y_crps.mean()),
                            y_coverage_95=float(scores.y_coverage_95.mean()),
                            y_interval_width=float(scores.y_interval_width.mean())))
    pd.DataFrame(summary).to_csv(folder/'scores.csv', index=False)
    pd.concat(all_rows).to_csv(folder/'per_function_scores.csv', index=False)
    # Resample whole held-out curves, paired across models, conditional on training.
    rng = np.random.default_rng(27026)
    bootstrap = rng.integers(len(all_rows[0]), size=(5000, len(all_rows[0])))
    paired = []
    for model, scores in zip(MODELS, all_rows):
        for metric in ['y_rmse', 'y_nlpd', 'y_crps']:
            candidate, gp = scores[metric].to_numpy(), all_rows[0][metric].to_numpy()
            if metric == 'y_rmse':
                point = np.sqrt(np.mean(candidate**2))-np.sqrt(np.mean(gp**2))
                delta = np.sqrt(np.mean(candidate[bootstrap]**2, axis=1))-np.sqrt(np.mean(gp[bootstrap]**2, axis=1))
            else:
                point = np.mean(candidate-gp)
                delta = np.mean((candidate-gp)[bootstrap], axis=1)
            paired.append(dict(model=model, metric=metric, difference_from_gp=float(point),
                               lower=float(np.quantile(delta, .025)), upper=float(np.quantile(delta, .975))))
    pd.DataFrame(paired).to_csv(folder/'paired_comparison.csv', index=False)
    tex_labels = ['GP (two SE)', 'STP + GP mean', r'GHP ($\mu$ GP)', r'GHP ($\beta$ GP)']
    lines = [r'\begin{table}[htbp]', r'\centering', r'\small',
             r'\caption{Tecator spectrum reconstruction: '+str(len(data['train_rows']))+
             ' training curves and '+str(len(data['test_rows']))+' held-out curves, with '+
             str(data['context_ids'].shape[1])+r' random observed wavelengths each. '
             r'Scores concern measured absorbance $y$ at the remaining wavelengths. '
             r'All models use a common mean curve and normalization estimated from training data only. '
             r'GHP fixes $\lambda=\omega=1$; '
             r'STP fixes $\nu=5$.}', r'\label{tab:tecator_four_models}',
             r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Model & RMSE ($y$) & NLPD ($y$) & CRPS ($y$) & 95\% coverage \\', r'\midrule']
    for label, row in zip(tex_labels, summary):
        lines.append(f'{label} & {row["y_rmse"]:.6f} & {row["y_nlpd"]:.4f} & '
                     f'{row["y_crps"]:.6f} & {100*row["y_coverage_95"]:.2f}'+r'\% \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (folder/'comparison.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_json(folder/'summary.json', dict(settings=json.loads((folder/'settings.json').read_text()),
               scores=summary, diagnostics=reports, n_train=len(data['train_rows']),
               n_test=len(data['test_rows']), metric_aggregation='Pooled RMSE; mean pointwise NLPD/CRPS/coverage; '
               'equal held-out counts make all curves equally weighted. No latent f truth is available.'))
    wavelength = data['wavelength_nm']
    for page in range((len(data['test_rows'])+3)//4):
        positions = list(range(page*4, min(len(data['test_rows']), (page+1)*4)))
        fig, axes = plt.subplots(len(positions), 4, figsize=(16, 2.8*len(positions)),
                                 squeeze=False, sharex=True, sharey='row', layout='constrained')
        for i, j in enumerate(positions):
            row, observed = data['test_rows'][j], data['context_ids'][j]
            for col, model in enumerate(MODELS):
                ax = axes[i, col]
                with np.load(folder/model/f'sample_{row+1:03d}.npz') as pred:
                    ax.plot(wavelength, data['y'][row], color='0.45', linestyle='--', lw=1,
                            label='Measured spectrum')
                    ax.plot(wavelength, pred['y_mean'], color=COLORS[model], label='Posterior mean')
                    ax.fill_between(wavelength, *pred['y_ci'].T, color=COLORS[model], alpha=.25,
                                    label='95% predictive interval for y')
                ax.scatter(wavelength[observed], data['y'][row, observed], c='k', s=14,
                           zorder=4, label='Revealed observations')
                ax.set_title(LABELS[model], fontsize=11)
                ax.grid(alpha=.2)
                if col == 0:
                    ax.set_ylabel(f'Sample {row+1}\nAbsorbance')
                if i == len(positions)-1:
                    ax.set_xlabel('Wavelength (nm)')
        axes[0, 0].legend(fontsize=6)
        fig.suptitle('Tecator: prediction from '+str(data['context_ids'].shape[1])+' random wavelengths')
        fig.savefig(folder/f'regression_page_{page+1}.png', dpi=155)
        plt.close(fig)
    # First four source-ordered test curves, shown on an error scale so narrow
    # intervals and small differences remain visible. No selection by scores.
    positions = list(range(min(4, len(data['test_rows']))))
    fig, axes = plt.subplots(len(positions), 4, figsize=(16, 2.8*len(positions)),
                             squeeze=False, sharex=True, sharey='row', layout='constrained')
    for i, j in enumerate(positions):
        row, observed = data['test_rows'][j], data['context_ids'][j]
        truth = data['y'][row]
        for col, model in enumerate(MODELS):
            ax = axes[i, col]
            with np.load(folder/model/f'sample_{row+1:03d}.npz') as pred:
                ax.plot(wavelength, pred['y_mean']-truth, color=COLORS[model], label='Prediction error')
                ax.fill_between(wavelength, *(pred['y_ci']-truth[:, None]).T,
                                color=COLORS[model], alpha=.25, label='95% y interval minus measurement')
            ax.axhline(0, color='0.4', linewidth=.8)
            ax.scatter(wavelength[observed], np.zeros(len(observed)), c='k', s=12, zorder=4)
            ax.set_title(LABELS[model], fontsize=11)
            ax.grid(alpha=.2)
            if col == 0:
                ax.set_ylabel(f'Sample {row+1}\nAbsorbance error')
            if i == len(positions)-1:
                ax.set_xlabel('Wavelength (nm)')
    axes[0, 0].legend(fontsize=6)
    fig.suptitle('Prediction errors: first four held-out spectra in source order')
    fig.savefig(folder/'prediction_errors.png', dpi=160)
    plt.close(fig)
    print(pd.DataFrame(summary).to_string(index=False), flush=True)


class Tee:
    def __init__(self, stream, file):
        self.stream, self.file = stream, file
        self.lock = threading.Lock()

    def write(self, text):
        with self.lock:
            self.stream.write(text)
            self.file.write(text)
            self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--stage', choices=['prepare', 'fit', 'predict', 'wait-predict', 'summarize', 'all'], default='all')
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--context-count', type=int, default=15)
    parser.add_argument('--warmup', type=int, default=1500)
    parser.add_argument('--draws', type=int, default=2000)
    parser.add_argument('--prediction-draws', type=int, default=6000)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output/f'{args.stage}_{args.model or "all"}.log'
    original_stdout = sys.stdout
    with log_path.open('a', encoding='utf-8') as log:
        sys.stdout = Tee(original_stdout, log)
        try:
            with threadpool_limits(1):
                prepare(args.output, context_count=args.context_count)
                data = load_data(args.output)
                for model in ([args.model] if args.model else MODELS):
                    if args.stage in ['fit', 'all']:
                        fit_model(args.output, data, model, warmup=args.warmup, draws=args.draws)
                    if args.stage == 'wait-predict':
                        wait_for_fit(args.output, model)
                    if args.stage in ['predict', 'wait-predict', 'all']:
                        predict_model(args.output, data, model, draws=args.prediction_draws)
                if args.stage in ['summarize', 'all']:
                    summarize(args.output, data)
        finally:
            sys.stdout = original_stdout


if __name__ == '__main__':
    main()
