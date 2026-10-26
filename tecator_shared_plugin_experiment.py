"""Shared beta/mu plug-in models on the saved Tecator split and contexts.

The benchmark train/test split is fixed. Seed 24026 selected context wavelengths.
Keep original absorbance levels, scale only: pointwise centering would force
the shared beta conditional mean to zero (sum_j centered y_j = 0).
All fitted functions/hyperparameters are frozen for new curves; integrate W.
"""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from GH_shared_function_map import MODELS, fit_shared_function, shared_calculation, gig_mean
from GH_fixed_function import fixed_function_prediction
from shared_beta_plugin_experiment import write_json, score_prediction, LABELS

SOURCE = Path('figure/tecator_multiple_functions_hmc')
DEST = Path('figure/tecator_shared_plugin_context24026')


def load_data(source):
    with np.load(source/'dataset.npz') as saved:
        return {key: saved[key].copy() for key in saved.files}


def prepare(source, output):
    data = load_data(source)
    settings = json.loads((source/'settings.json').read_text())
    train = data['train_rows']
    scale = float(data['common_scale'])
    np.testing.assert_allclose(scale, np.std(data['y'][train]-data['y'][train].mean(axis=0)))
    Y = data['y'][train]/scale
    # Shared population mean prior reflects the retained level; all models
    # receive the same prior. Residual prior centers match the earlier setup.
    center = np.log([.5, max(float(np.mean(Y.mean(axis=0)**2)), .5), .01, 1., .04])
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
              for p in [source/'dataset.npz', source/'settings.json', source/'scores.csv', source/'sample_roles.csv']}
    report = dict(source=str(source), source_sha256=hashes, context_seed=settings['location_seed'],
                  split=settings['split'], grouping=settings['grouping'],
                  n_train=len(train), n_test=len(data['test_rows']), training_points=len(data['x']),
                  context_count=data['context_ids'].shape[1], held_out_points=data['evaluation_ids'].shape[1],
                  scale=scale, centering='None: retain original absorbance level',
                  prior_center=center, prior_sd=2., GHP_lambda=1., GHP_omega=1., STP_nu=5.,
                  target='observed absorbance, original units; latent f has no ground truth',
                  estimation='shared-function marginalization, log-coordinate theta/W MAP, then fixed conditional function mean',
                  prediction='fixed theta/function; conjugate W posterior per test curve integrated',
                  comparison_limit='Earlier HMC fits used pointwise centering and separate per-curve latent GPs; this is not an isolated ablation of sharing.')
    output.mkdir(parents=True, exist_ok=True)
    if (output/'experiment.json').exists():
        old = json.loads((output/'experiment.json').read_text())
        if old['source_sha256'] != hashes:
            raise ValueError('Source changed; choose a new output directory')
    write_json(output/'experiment.json', report)
    if not (output/'dataset.npz').exists():
        np.savez_compressed(output/'dataset.npz', **data)
    return data, center


def fit_all(output, data, center, starts):
    scale = float(data['common_scale'])
    Y = data['y'][data['train_rows']]/scale
    source_hash = hashlib.sha256((output/'dataset.npz').read_bytes()).hexdigest()
    for model in MODELS:
        folder = output/model
        folder.mkdir(exist_ok=True)
        if (folder/'fit.json').exists():
            fit = json.loads((folder/'fit.json').read_text())
            if fit['dataset_sha256'] != source_hash or fit['starts_requested'] != starts:
                raise ValueError('Saved fit settings differ')
            if fit.get('optimizer_version') == 2:
                print(f'{model}: reusing fit', flush=True)
                continue
            if not (folder/'fit_initial.json').exists():
                write_json(folder/'fit_initial.json', fit)
        print(f'{model}: fitting {len(Y)} curves x {len(data["x"])} points, {starts} starts', flush=True)
        fit = fit_shared_function(Y, data['x'], data['x'], model, starts=starts, seed=24026,
                                  prior_center=center, backend='qr')
        fit.update(dataset_sha256=source_hash, starts_requested=starts,
                   function_original=scale*fit['function'], population_mean_original=scale*fit['population_mean'])
        write_json(folder/'fit.json', fit)
        print(f'{model}: fit complete, max gradient check error={fit["max_gradient_check_error"]:.3g}', flush=True)


def original_units(prediction, scale):
    p = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in prediction.items()}
    for key in ('location', 'conditional_beta', 'f_mean', 'y_mean', 'f_ci', 'y_ci'):
        p[key] *= scale
    for key in ('f_scale_cov', 'y_scale_cov', 'f_cov', 'y_cov', 'f_var', 'y_var'):
        p[key] *= scale**2
    return p


def check_alternative_lengths(output, data, center):
    """A common additional search for shorter shared-function length scales."""
    scale = float(data['common_scale'])
    Y = data['y'][data['train_rows']]/scale
    for model in MODELS:
        folder = output/model
        fit = json.loads((folder/'fit.json').read_text())
        if fit.get('shared_length_search_version') == 1:
            continue
        write_json(folder/'fit_before_length_search.json',fit)
        points = []
        for w in (1.,4.,16.,64.):
            q = np.asarray(fit['log_parameters']).copy(); q[4] = np.log(w)
            points.append(q)
        print(f'{model}: checking four additional shared-function length starts',flush=True)
        alternative = fit_shared_function(Y,data['x'],data['x'],model,starts=4,seed=24026,
                                           prior_center=center,backend='qr',initial_points=points)
        history = dict(previous_log_posterior=fit['log_posterior'],
                       alternative_log_posterior=alternative['log_posterior'],
                       inverse_length_squared_starts=[1.,4.,16.,64.],
                       attempts=alternative['attempts'])
        if alternative['log_posterior'] > fit['log_posterior']+1e-5:
            for key in ('dataset_sha256','starts_requested'):
                alternative[key] = fit[key]
            alternative.update(function_original=scale*alternative['function'],
                               population_mean_original=scale*alternative['population_mean'])
            fit = alternative
        fit.update(shared_length_search_version=1,length_search=history,total_starts=fit['starts_requested']+4)
        write_json(folder/'fit.json',fit)
        print(f'{model}: best log posterior {fit["log_posterior"]:.6f}',flush=True)


def predict_all(output, data):
    scale = float(data['common_scale'])
    all_rows, checks = [], []
    for model in MODELS:
        folder = output/model
        fit = json.loads((folder/'fit.json').read_text())
        fit_hash = hashlib.sha256((folder/'fit.json').read_bytes()).hexdigest()
        v, a, k, w, hw = np.exp(fit['log_parameters'][:5])
        function = np.asarray(fit['function'])
        kind = 'beta' if model == 'ghp_beta' else 'mu'
        mixing = 'gp' if model == 'gp' else 'ig' if model == 'stp' else 'gig'
        rows, validations = [], []
        for j, row_id in enumerate(data['test_rows']):
            path = folder/f'curve_{row_id+1:03d}'
            if path.with_suffix('.json').exists():
                report = json.loads(path.with_suffix('.json').read_text())
                if report.get('fit_sha256') == fit_hash:
                    rows.append(report['scores']); validations.append(report['validation'])
                    continue
            ids, test = data['context_ids'][j], data['evaluation_ids'][j]
            kwargs = {f'{kind}_train': function[ids], f'{kind}_pred': function}
            p = fixed_function_prediction(data['y'][row_id, ids]/scale, data['x'][ids], data['x'],
                                           v0=v, k=k, w=w, mixing=mixing, **kwargs)
            p = original_units(p, scale)
            score, point, validation = score_prediction(p, None, data['y'][row_id], test, check=j == 0)
            score.update(model=model, sample_id=int(row_id+1), n_test=len(test),
                         W_mean=float(p['W_mean']), W_std=float(np.sqrt(p['W_var'])))
            np.savez_compressed(path.with_suffix('.npz'), **p, **point, context_ids=ids, test_ids=test)
            write_json(path.with_suffix('.json'), dict(scores=score, validation=validation, fit_sha256=fit_hash))
            rows.append(score); validations.append(validation)
            if (j+1)%5 == 0 or j+1 == len(data['test_rows']):
                print(f'{model}: prediction {j+1}/{len(data["test_rows"])}', flush=True)
        pd.DataFrame(rows).to_csv(folder/'per_curve_scores.csv', index=False)
        write_json(folder/'validation.json', dict(all_checks_passed=True, curves=validations))
        all_rows.extend(rows); checks.extend(validations)
    pd.DataFrame(all_rows).to_csv(output/'per_curve_scores.csv', index=False)
    write_json(output/'prediction_validation.json', dict(all_checks_passed=True, predictions=len(all_rows), curves=checks))


def summarize(source, output, data):
    all_rows = pd.read_csv(output/'per_curve_scores.csv')
    rows, parameters = [], []
    scale = float(data['common_scale'])
    for model in MODELS:
        frame = all_rows[all_rows.model == model]
        result = dict(model=model, n_test_curves=len(frame), y_rmse=float(np.sqrt(np.mean(frame.y_rmse**2))))
        for metric in ('nlpd', 'crps', 'coverage_95', 'interval_width', 'mean_abs_skewness'):
            result[f'y_{metric}'] = float(frame[f'y_{metric}'].mean())
        rows.append(result)
        fit = json.loads((output/model/'fit.json').read_text())
        v, a, k, w, hw = np.exp(fit['log_parameters'][:5])
        parameters.append(dict(model=model, residual_variance=v*scale**2, noise_variance_coefficient=k*scale**2,
                               residual_inverse_length_squared=w, population_mean_GP_variance=a*scale**2,
                               mean_inverse_length_squared=hw,
                               beta_GP_variance=a*scale**2/gig_mean()**2 if model == 'ghp_beta' else None))
    scores = pd.DataFrame(rows)
    scores.to_csv(output/'scores.csv', index=False)
    pd.DataFrame(parameters).to_csv(output/'parameters.csv', index=False)
    previous = pd.read_csv(source/'scores.csv').rename(columns={'model':'previous_model'})
    aliases = dict(gp='gp', stp='tp_mu_gp', ghp_mu='mu_gp', ghp_beta='beta_gp')
    comparisons = []
    for r in rows:
        old = previous[previous.previous_model == aliases[r['model']]].iloc[0]
        comparisons.append(dict(model=r['model'], previous_method='independent functions, HMC, wavelength-centered',
                                current_method='shared function, plug-in MAP, uncentered',
                                previous_rmse=old.y_rmse, current_rmse=r['y_rmse'],
                                previous_nlpd=old.y_nlpd, current_nlpd=r['y_nlpd'],
                                previous_crps=old.y_crps, current_crps=r['y_crps']))
    pd.DataFrame(comparisons).to_csv(output/'comparison_with_previous.csv', index=False)
    paired, rng = [], np.random.default_rng(24026)
    baseline = all_rows[all_rows.model == 'gp'].sort_values('sample_id')
    for model in MODELS[1:]:
        candidate = all_rows[all_rows.model == model].sort_values('sample_id')
        np.testing.assert_array_equal(candidate.sample_id, baseline.sample_id)
        ids = rng.integers(0, len(baseline), (5000, len(baseline)))
        for metric in ('rmse', 'nlpd', 'crps'):
            a, b = candidate[f'y_{metric}'].to_numpy(), baseline[f'y_{metric}'].to_numpy()
            delta = (np.sqrt(np.mean(a[ids]**2, axis=1))-np.sqrt(np.mean(b[ids]**2, axis=1)) if metric == 'rmse'
                     else np.mean((a-b)[ids], axis=1))
            difference = np.sqrt(np.mean(a*a))-np.sqrt(np.mean(b*b)) if metric == 'rmse' else np.mean(a-b)
            lo, hi = np.quantile(delta, [.025,.975])
            paired.append(dict(model=model, reference='gp', metric=metric, difference=difference, lower=lo, upper=hi))
    pd.DataFrame(paired).to_csv(output/'paired_comparison.csv', index=False)
    results_text = [
        'Tecator shared-function plug-in regression',
        '147 training spectra; 31 held-out spectra; 15 revealed and 85 scored wavelengths per held-out spectrum.',
        'Fixed benchmark split. Seed 24026 selects revealed wavelengths, not train/test spectra.',
        '', scores.to_string(index=False), '',
        'Lower RMSE, NLPD and CRPS are better. Scores are for observed absorbance in original units.',
        'No noise-free f is available. Predictive bands omit fitted-parameter/shared-function uncertainty.',
        '', 'Comparison with the earlier independent-function HMC run:',
        pd.DataFrame(comparisons).to_string(index=False), '',
        'This comparison changes function sharing, estimation, and centering, so it does not isolate one change.',
        'Fixing the shared mu also leaves each baseline with just one residual SE component.',
        'GHP beta additionally represents curve-to-curve changes along the shared spectral shape via W*beta.',
        'The small reported predictive skewness means any advantage here cannot be attributed to skewness alone.',
        '', 'The eight original optimization starts were supplemented by four less-smooth shared-function starts',
        '(inverse squared length scales 1, 4, 16, 64). Selection uses training posterior density only.',
        'See fit.json, fit_before_length_search.json, validation.json and paired_comparison.csv for diagnostics.'
    ]
    (output/'results.txt').write_text('\n'.join(results_text)+'\n',encoding='utf-8')
    lines = [r'\begin{table}[htbp]', r'\centering',
             r'\caption{Tecator shared-function plug-in regression: original benchmark split, 15 observed wavelengths per test curve (selection seed 24026). Scores in original absorbance units.}',
             r'\label{tab:tecator_shared_plugin}', r'\begin{tabular}{lrrrrr}', r'\toprule',
             r'Model & RMSE ($y$) & NLPD ($y$) & CRPS ($y$) & 95\% coverage & Mean width \\', r'\midrule']
    names = dict(gp=r'GP (shared $\hat\mu$)', stp=r'STP (shared $\hat\mu$)',
                 ghp_mu=r'GHP (shared $\hat\mu$)', ghp_beta=r'GHP (shared $\hat\beta$)')
    for r in rows:
        lines.append(names[r['model']]+' & '+' & '.join(f'{r[k]:.5f}' for k in
                     ('y_rmse','y_nlpd','y_crps','y_coverage_95','y_interval_width'))+r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (output/'comparison.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    wavelength = data['wavelength_nm']
    fig, axes = plt.subplots(1, 2, figsize=(11,4))
    axes[0].plot(wavelength, data['common_mean'], 'k--', label='Empirical training mean')
    for model in MODELS:
        fit = json.loads((output/model/'fit.json').read_text())
        axes[0].plot(wavelength, fit['population_mean_original'], label=LABELS[model])
        if model == 'ghp_beta':
            axes[1].plot(wavelength, fit['function_original'], label='Estimated shared beta')
    axes[0].set_title('Estimated population mean spectra'); axes[1].set_title('Shared beta (no known ground truth)')
    axes[0].set_ylabel('Absorbance'); axes[1].set_ylabel('Absorbance per unit W')
    for ax in axes:
        ax.set_xlabel('Wavelength (nm)'); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output/'shared_functions.png', dpi=160); fig.savefig(output/'shared_functions.pdf'); plt.close(fig)
    with PdfPages(output/'regression.pdf') as pdf:
        for start in range(0, len(data['test_rows']), 5):
            subset = data['test_rows'][start:start+5]
            fig, axes = plt.subplots(len(subset), 4, figsize=(16,2.5*len(subset)), squeeze=False, sharex=True, sharey='row')
            for i, row in enumerate(subset):
                for j, model in enumerate(MODELS):
                    with np.load(output/model/f'curve_{row+1:03d}.npz') as p:
                        ax=axes[i,j]; ids=p['context_ids']
                        ax.fill_between(wavelength,*p['y_ci'].T,color=f'C{j}',alpha=.25)
                        ax.plot(wavelength,p['y_mean'],color=f'C{j}',lw=1.3)
                        ax.plot(wavelength,data['y'][row],'k--',lw=1)
                        ax.scatter(wavelength[ids],data['y'][row,ids],s=12,c='black')
                        ax.set_title(f'{LABELS[model]}, sample {row+1}',fontsize=9)
                        if j == 0: ax.set_ylabel('Absorbance')
                        if i == len(subset)-1: ax.set_xlabel('Wavelength (nm)')
            fig.suptitle('Tecator: dashed measured spectrum; dots revealed; shaded 95% observation interval')
            fig.tight_layout(); pdf.savefig(fig)
            if start == 0:
                fig.savefig(output/'regression.png',dpi=140)
            plt.close(fig)
    print(scores.to_string(index=False),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=SOURCE)
    parser.add_argument('--output',type=Path,default=DEST)
    parser.add_argument('--starts',type=int,default=8)
    parser.add_argument('--stage',choices=('all','fit','predict','summarize'),default='all')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        data,center=prepare(args.source,args.output)
        if args.stage in ('all','fit'):
            fit_all(args.output,data,center,args.starts)
            check_alternative_lengths(args.output,data,center)
        if args.stage in ('all','predict'): predict_all(args.output,data)
        if args.stage in ('all','summarize'): summarize(args.source,args.output,data)
