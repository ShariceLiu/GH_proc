"""Gasoline spectra: five paired splits, four plug-in models, shared/individual functions.

Run with the project Python: gasoline_plugin_experiment.py --stage all
An interrupted run resumes from per-start fitting checkpoints and predictions.
No HMC or dense posterior arrays are saved. Fits run sequentially with one BLAS thread.
"""
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from GH_shared_function_map import MODELS, fit_shared_function, gig_mean
from GH_independent_plugin_map import fit_independent, context_function
from GH_fixed_function import fixed_function_prediction, refine_density_quadrature
from shared_beta_plugin_experiment import write_json, score_prediction
from tecator_shared_plugin_experiment import original_units

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'data'/'real_datasets'/'gasoline'
OUTPUT = ROOT/'figure'/'gasoline_plugin_five_splits'
SOURCE_URL = 'https://raw.githubusercontent.com/cran/pls/master/data/gasoline.RData'
SEEDS = (54026, 54027, 54028, 54029, 54030)
CONTEXT_SEED = 64026
COUNTS = (15, 30)
SHARING = ('shared', 'individual')
METRICS = ('y_rmse', 'y_nlpd', 'y_crps', 'y_coverage_95', 'y_interval_width')
NAMES = dict(gp='GP mu', stp='STP + GP mu', ghp_mu='GHP mu', ghp_beta='GHP beta')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def arrays(path):
    with np.load(path, allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def source_data():
    """Preserve the package's matrix-valued R column, without rounding to CSV/JSON."""
    path = SOURCE/'gasoline.RData'
    if not path.exists():
        import requests
        response = requests.get(SOURCE_URL, timeout=60)
        response.raise_for_status()
        SOURCE.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    output = SOURCE/'gasoline.npz'
    if not output.exists():
        import rdata
        original = rdata.read_rda(path, default_encoding='ASCII', constructor_dict={
            'data.frame': lambda obj, attrs: obj, 'AsIs': lambda obj, attrs: obj})['gasoline']
        y = np.asarray(original['NIR'], dtype=float)
        assert y.shape == (60, 401) and np.isfinite(y).all()
        assert len(np.unique(y, axis=0)) == 60
        wavelengths = np.arange(900., 1701., 2.)
        # Check the labels supplied by the source, rather than assume matrix order.
        labels = original['NIR'].coords['dim_1'].values
        np.testing.assert_array_equal([float(str(s).removesuffix(' nm')) for s in labels], wavelengths)
        np.savez_compressed(output, y=y, wavelength_nm=wavelengths,
                            octane=np.asarray(original['octane']))
        write_json(SOURCE/'source.json', dict(url=SOURCE_URL, raw_sha256=digest(path),
            dataset_sha256=digest(output), raw_bytes=path.stat().st_size,
            reference='Kalivas (1997), Two Data Sets of Near Infrared Spectra, CILS 37:255-259',
            documentation='https://cran.r-project.org/web/packages/pls/pls.pdf',
            units='Diffuse reflectance log(1/R); octane is stored but unused.'))
    metadata = json.loads((SOURCE/'source.json').read_text())
    assert digest(path) == metadata['raw_sha256'] and digest(output) == metadata['dataset_sha256']
    return arrays(output)


def prepare():
    raw = source_data()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    selected = np.arange(0, 401, 4)
    data = dict(y=raw['y'][:, selected], wavelength_nm=raw['wavelength_nm'][selected],
                source_channel_ids=selected, x=np.linspace(0., 10., 101))
    dataset = OUTPUT/'dataset.npz'
    if dataset.exists():
        saved = arrays(dataset)
        for key in data:
            np.testing.assert_array_equal(data[key], saved[key])
    else:
        np.savez_compressed(dataset, **data)
    settings = dict(source_sha256=digest(SOURCE/'gasoline.npz'), dataset_sha256=digest(dataset),
        split_seeds=list(SEEDS), context_seed=CONTEXT_SEED, n_train=40, n_test=20,
        channels=101, channel_rule='Every fourth original wavelength, including both endpoints.',
        context_counts=list(COUNTS), models=list(MODELS), function_sharing=list(SHARING),
        GHP_lambda=1., GHP_omega=1., STP_nu=5., prior_log_sd=2.,
        preprocessing='Divide by training-only across-curve residual SD; no centering or per-curve normalization.',
        score_units='Original log(1/R); held-out observed y, not latent f.',
        prediction='Plug-in theta and function; keep posterior uncertainty in fresh W.',
        aggregation='Pooled held-out RMSE per split; mean other scores. Mean +/- sample SD across five splits.',
        optimization='12 shared starts, 24 individual starts; fixed seeds; sequential single-thread fitting.')
    if (OUTPUT/'experiment.json').exists():
        assert json.loads((OUTPUT/'experiment.json').read_text()) == settings
    else:
        write_json(OUTPUT/'experiment.json', settings)
    for seed in SEEDS:
        folder = OUTPUT/f'split_{seed}'
        folder.mkdir(exist_ok=True)
        perm = np.random.default_rng(seed).permutation(60)
        train, test = np.sort(perm[:40]), np.sort(perm[40:])
        orders = np.stack([np.random.default_rng(np.random.SeedSequence([CONTEXT_SEED, int(row)]))
                           .permutation(101) for row in test])
        scale = float(np.std(data['y'][train]-data['y'][train].mean(0)))
        Y = data['y'][train]/scale
        center = np.log([.5, max(float(np.mean(Y.mean(0)**2)), .5), .01, 1., .04])
        split = dict(train_rows=train, test_rows=test, context_orders=orders,
                     common_scale=np.array(scale), prior_center=center)
        path = folder/'split.npz'
        if path.exists():
            saved = arrays(path)
            for key in split:
                np.testing.assert_array_equal(split[key], saved[key])
        else:
            np.savez_compressed(path, **split)
    if not (OUTPUT/'spectra_overview.pdf').exists():
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(data['wavelength_nm'], data['y'].T, lw=.7, alpha=.5)
        ax.set(xlabel='Wavelength (nm)', ylabel='Diffuse reflectance log(1/R)',
               title='Gasoline: all 60 spectra, 101 retained wavelengths')
        fig.tight_layout(); fig.savefig(OUTPUT/'spectra_overview.pdf'); plt.close(fig)
    (OUTPUT/'method.txt').write_text(METHOD, encoding='utf-8')
    return data


def fit_case(data, seed, sharing, model, *, output=OUTPUT):
    split_folder = output/f'split_{seed}'
    split = arrays(split_folder/'split.npz')
    folder = split_folder/sharing/model
    folder.mkdir(parents=True, exist_ok=True)
    provenance = dict(dataset_sha256=digest(output/'dataset.npz'),
                      split_sha256=digest(split_folder/'split.npz'))
    if (folder/'fit.json').exists():
        fit = json.loads((folder/'fit.json').read_text())
        for key in provenance:
            assert fit[key] == provenance[key]
        return folder, split, fit
    Y = data['y'][split['train_rows']]/split['common_scale']
    center = split['prior_center']; x = data['x']
    checkpoint = folder/'optimization_starts.json'
    completed = json.loads(checkpoint.read_text()) if checkpoint.exists() else []
    coordinate_key = 'log_parameters' if sharing == 'shared' else 'effective_log_parameters'

    def progress(record, q):
        completed.append(dict(record=record, **{coordinate_key: q.tolist()}))
        write_json(checkpoint, completed)
        print(f'{seed} {sharing} {model}: start {record["start"]+1}, '
              f'log posterior {record["log_posterior"]:.4f}, gradient {record["max_abs_gradient"]:.2g}', flush=True)

    print(f'Fitting {seed} {sharing} {model}', flush=True)
    started = time.perf_counter()
    if sharing == 'shared':
        initial = center.copy()
        if model != 'gp':
            initial = np.r_[initial, np.full(len(Y), np.log(.6) if model == 'stp' else np.arcsinh(1.))]
        rng = np.random.default_rng(24026)
        points = []
        for sw in (.04, 1., 16., 256., 4096., 65536.):
            for perturb in (False, True):
                q = initial.copy(); q[4] = np.log(sw)
                if perturb:
                    q += rng.normal(0., .4, len(q))
                points.append(q)
        fit = fit_shared_function(Y, x, x, model, starts=len(points), prior_center=center,
            backend='dense', initial_points=points, refine_stationary=True,
            progress=progress, completed=completed)
    else:
        fit = fit_independent(Y, x, model, center, progress=progress, completed=completed,
                              backend='shared_grid')
    fit.update(provenance, sharing=sharing, fitting_seconds_this_invocation=time.perf_counter()-started)
    write_json(folder/'fit.json', fit)
    print(f'Finished {seed} {sharing} {model}: log posterior {fit["log_posterior"]:.5f}', flush=True)
    return folder, split, fit


def predict_case(data, seed, sharing, model, folder, split, fit):
    theta = np.asarray(fit['log_parameters'][:5]); v, a, k, w, sw = np.exp(theta)
    scale = float(split['common_scale']); x = data['x']; n = len(x)
    mixing = 'gp' if model == 'gp' else 'ig' if model == 'stp' else 'gig'
    kind = 'beta' if model == 'ghp_beta' else 'mu'
    fit_hash = digest(folder/'fit.json')
    rows, common_rows = [], []
    for count in COUNTS:
        case = folder/f'context_{count}'; case.mkdir(exist_ok=True)
        for j, row in enumerate(split['test_rows']):
            stem = case/f'curve_{row+1:02d}'
            meta_path, pred_path = stem.with_suffix('.json'), stem.with_suffix('.npz')
            if meta_path.exists() and pred_path.exists():
                saved = json.loads(meta_path.read_text())
                if saved['fit_sha256'] == fit_hash:
                    rows.append(saved['scores']); common_rows.append(saved['common_scores']); continue
            ids = np.sort(split['context_orders'][j, :count]); held = np.setdiff1d(np.arange(n), ids)
            yc = data['y'][row, ids]/scale
            if sharing == 'shared':
                function = np.asarray(fit['function']); mode = {}
            else:
                function, mode = context_function(yc, x[ids], x, theta, model)
            pred = fixed_function_prediction(yc, x[ids], x, v0=v, k=k, w=w, mixing=mixing, gh_backend='quadrature',
                **{f'{kind}_train': function[ids], f'{kind}_pred': function})
            pred = original_units(pred, scale)
            refine_density_quadrature(pred, data['y'][row])
            scores, point, validation = score_prediction(pred, None, data['y'][row], held, check=j == 0)
            identity = dict(seed=seed, sharing=sharing, model=model, context_count=count, curve=int(row+1))
            scores.update(identity, n_test=len(held), W_mean=pred['W_mean'], W_std=float(np.sqrt(pred['W_var'])))
            common = np.setdiff1d(np.arange(n), split['context_orders'][j, :30])
            positions = np.searchsorted(held, common)
            ci, truth = pred['y_ci'][common], data['y'][row, common]
            common_scores = dict(identity, n_test=len(common),
                y_rmse=float(np.sqrt(np.mean((truth-pred['y_mean'][common])**2))),
                y_nlpd=float(point['y_nlpd'][positions].mean()), y_crps=float(point['y_crps'][positions].mean()),
                y_coverage_95=float(np.mean((truth >= ci[:, 0]) & (truth <= ci[:, 1]))),
                y_interval_width=float(np.mean(ci[:, 1]-ci[:, 0])))
            compact = {key: value for key, value in pred.items() if not key.endswith('_cov')}
            compact.update(f_scale_var=np.diag(pred['f_scale_cov']), y_scale_var=np.diag(pred['y_scale_cov']))
            np.savez_compressed(pred_path, **compact, **point, context_ids=ids, test_ids=held,
                                function_estimate_scaled=function)
            write_json(meta_path, dict(scores=scores, common_scores=common_scores, validation=validation,
                                      context_mode=mode, fit_sha256=fit_hash))
            rows.append(scores); common_rows.append(common_scores)
        print(f'{seed} {sharing} {model}: finished {count}-point predictions', flush=True)
    pd.DataFrame(rows).to_csv(folder/'per_curve_scores.csv', index=False)
    pd.DataFrame(common_rows).to_csv(folder/'common_per_curve_scores.csv', index=False)


def aggregate(curves, *, seeds=SEEDS):
    rows = []
    for keys, frame in curves.groupby(['seed', 'sharing', 'model', 'context_count'], sort=False):
        assert len(frame) == 20 and frame.curve.nunique() == 20
        row = dict(zip(('seed', 'sharing', 'model', 'context_count'), keys))
        row['y_rmse'] = float(np.sqrt(np.mean(frame.y_rmse**2)))
        row.update({metric: float(frame[metric].mean()) for metric in METRICS[1:]})
        rows.append(row)
    scores = pd.DataFrame(rows)
    result = []
    for keys, frame in scores.groupby(['sharing', 'model', 'context_count'], sort=False):
        assert len(frame) == len(seeds) and set(frame.seed) == set(seeds)
        row = dict(zip(('sharing', 'model', 'context_count'), keys))
        for metric in METRICS:
            row[metric+'_mean'] = float(frame[metric].mean())
            row[metric+'_sd'] = float(frame[metric].std(ddof=1))
        result.append(row)
    return scores, pd.DataFrame(result)


def summarize(data, *, output=OUTPUT, seeds=SEEDS, caption=None, table_label='gasoline_plugin',
              coordinate_key='wavelength_nm', xlabel='Wavelength (nm)', ylabel='log(1/R)'):
    parameters = []
    for seed in seeds:
        split = arrays(output/f'split_{seed}'/'split.npz'); scale = float(split['common_scale'])
        for sharing in SHARING:
            for model in MODELS:
                fit = json.loads((output/f'split_{seed}'/sharing/model/'fit.json').read_text())
                v, a, k, w, sw = np.exp(fit['log_parameters'][:5])
                native_a = a/gig_mean()**2 if sharing == 'shared' and model == 'ghp_beta' else a
                population_a = native_a*gig_mean()**2 if model == 'ghp_beta' else native_a
                parameters.append(dict(seed=seed, sharing=sharing, model=model,
                    residual_variance=v*scale**2, latent_variance_native=native_a*scale**2,
                    population_effect_variance=population_a*scale**2, noise_coefficient=k*scale**2,
                    residual_inverse_length_squared=w, latent_inverse_length_squared=sw,
                    latent_function='beta' if model == 'ghp_beta' else 'mu',
                    log_posterior=fit['log_posterior'], max_hessian_eigenvalue=fit['max_hessian_eigenvalue'],
                    W_training_mode_mean=float(np.mean(fit['W_training_mode'])),
                    W_training_mode_min=float(np.min(fit['W_training_mode'])), W_training_mode_max=float(np.max(fit['W_training_mode']))))
    pd.DataFrame(parameters).to_csv(output/'parameters.csv', index=False)
    curves = pd.concat([pd.read_csv(output/f'split_{seed}'/sharing/model/'per_curve_scores.csv')
        for seed in seeds for sharing in SHARING for model in MODELS], ignore_index=True)
    common = pd.concat([pd.read_csv(output/f'split_{seed}'/sharing/model/'common_per_curve_scores.csv')
        for seed in seeds for sharing in SHARING for model in MODELS], ignore_index=True)
    assert len(curves) == len(common) == 320*len(seeds)
    for name, frame in (('', curves), ('common_', common)):
        frame.to_csv(output/f'{name}per_curve_scores.csv', index=False)
        scores, summary = aggregate(frame, seeds=seeds)
        scores.to_csv(output/f'{name}split_scores.csv', index=False)
        summary.to_csv(output/f'{name}scores.csv', index=False)
    scores, summary = aggregate(curves, seeds=seeds)
    paired = scores[scores.sharing == 'individual'].merge(scores[scores.sharing == 'shared'],
        on=['seed', 'model', 'context_count'], suffixes=('_individual', '_shared'), validate='one_to_one')
    for metric in METRICS:
        paired[metric+'_change'] = paired[metric+'_individual']-paired[metric+'_shared']
    paired.to_csv(output/'paired_split_comparison.csv', index=False)
    names = dict(gp=r'GP $\mu$', stp=r'STP + GP $\mu$', ghp_mu=r'GHP $\mu$', ghp_beta=r'GHP $\beta$')
    if caption is None:
        caption = (r'Gasoline spectrum reconstruction: mean $\pm$ sample SD across five 40/20 spectrum splits. '
        r'Each spectrum has 101 wavelengths; 15 or 30 are revealed on each test spectrum. '
        r'All scores evaluate held-out observed log(1/R). Parameters and functions are plug-in estimates.')
    lines = [r'\begin{table}[htbp]', r'\centering', r'\small', r'\caption{'+caption+'}',
        r'\label{tab:'+table_label+'}', r'\resizebox{\textwidth}{!}{%', r'\begin{tabular}{rllrrrr}', r'\toprule',
        r'Observed & Model & Function & RMSE & NLPD & CRPS & 95\% coverage (\%) \\', r'\midrule']
    for count in COUNTS:
        for sharing in SHARING:
            for model in MODELS:
                row = summary[(summary.context_count == count) & (summary.sharing == sharing) & (summary.model == model)].iloc[0]
                values = []
                for metric in METRICS[:4]:
                    factor = 100 if metric == 'y_coverage_95' else 1
                    digits = 2 if metric == 'y_coverage_95' else 5
                    values.append(f'${factor*row[metric+"_mean"]:.{digits}f} \\pm {factor*row[metric+"_sd"]:.{digits}f}$')
                lines.append(f'{count} & {names[model]} & {sharing.capitalize()} & '+' & '.join(values)+r' \\')
            lines.append(r'\midrule')
    lines[-1] = r'\bottomrule'; lines += [r'\end{tabular}}', r'\end{table}']
    (output/'comparison.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    plot_examples(data, output=output, seeds=seeds, coordinate_key=coordinate_key, xlabel=xlabel, ylabel=ylabel)
    print(summary.to_string(index=False), flush=True)


def plot_examples(data, *, output=OUTPUT, seeds=SEEDS, coordinate_key='wavelength_nm',
                  xlabel='Wavelength (nm)', ylabel='log(1/R)'):
    """Preselected first two test spectrum IDs in each split, not selected by scores."""
    wavelength = data[coordinate_key]
    with PdfPages(output/'regression.pdf') as pdf:
        for seed in seeds:
            split = arrays(output/f'split_{seed}'/'split.npz')
            for count in COUNTS:
                for row in split['test_rows'][:2]:
                    fig, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True, sharey=True)
                    for ax, model in zip(axes.flat, MODELS):
                        for sharing, color in zip(SHARING, ('C1', 'C0')):
                            p = arrays(output/f'split_{seed}'/sharing/model/f'context_{count}'/f'curve_{row+1:02d}.npz')
                            ax.fill_between(wavelength, *p['y_ci'].T, color=color, alpha=.18)
                            ax.plot(wavelength, p['y_mean'], color=color, lw=1.2, label=sharing.capitalize())
                        ax.plot(wavelength, data['y'][row], 'k--', lw=1., label='Measured')
                        ids = p['context_ids']; ax.scatter(wavelength[ids], data['y'][row, ids], c='k', s=12, label='Revealed')
                        ax.set(title=NAMES[model], xlabel=xlabel, ylabel=ylabel)
                    axes[0, 0].legend(fontsize=8)
                    fig.suptitle(f'Split {seed}, spectrum {row+1}, {count} revealed; 95% observation intervals')
                    fig.tight_layout(); pdf.savefig(fig)
                    if seed == seeds[0] and row == split['test_rows'][0]:
                        fig.savefig(output/f'regression_context{count}.png', dpi=130)
                    plt.close(fig)


METHOD = '''GASOLINE: SHARED AND INDIVIDUAL PLUG-IN FUNCTIONS

We reconstruct spectra, not octane numbers. The pls package contains 60 gasoline
samples measured at 401 wavelengths (900--1700 nm, 2 nm increments). Retain every
fourth channel, giving 101 values in the original log(1/R) units. No samples are
excluded; exact duplicate spectra are checked. Source data and SHA256 are cached.

Five fixed seeds 54026--54030 split whole spectra into 40 training and 20 test.
Training uses all 101 retained wavelengths. Each test spectrum reveals nested
random sets of 15 and 30 wavelengths, leaving 86 and 71 targets respectively.
The context permutation is fixed by seed 64026 and original spectrum ID, shared
by every model and split. No test values select contexts, parameters or models.
Additional common-target scores evaluate both context sizes on the same 71 points.

x = (wavelength-900)/80 lies in [0,10]. Divide y by one training-only scale:
SD(Y_train - wavelength-wise training mean). Do not center spectra or normalize
each one. Use Gaussian priors, SD=2, on five log hyperparameters (v,a,k,w,w_h),
centered on log([0.5, max(mean(training_mean_scaled**2),0.5),0.01,1,0.04]).
Each SE covariance is v*exp(-w*(x-x')**2/2). The latent-function prior is matched
on h=mu or h=E_prior[W]*beta; the native beta variance is a/E_prior[W]**2.

The four models are GP-mu (W=1), STP plus GP-prior mu (W~IG(2.5,1.5)),
GHP-mu and GHP-beta (W~GIG(lambda=1,chi=psi=1)). GHP-beta fixes mu=0;
the other models fix beta=0. The STP variant adds an unscaled mean GP, so its
training marginal is not a pure Student-t process.

SHARED: One mu/beta function is shared by all training curves. Integrate its
GP prior analytically; jointly optimize log(theta) and all training log(W).
This is MAP in log coordinates and includes log-W Jacobians. At the optimum,
fix the conditional GP posterior mean as the function for all test spectra.

INDIVIDUAL: Each curve has an independent mu_j/beta_j with common GP parameters.
Integrate these functions separately; jointly fit common log(theta) and training
log(W). At test time freeze theta, optimize that spectrum's log(W) using only
its context and the marginalized function, then fix the conditional GP mean.
This uses the same two-stage plug-in procedure as the electricity experiment.

After fixing the function, recompute the test W posterior from the ORIGINAL
mixing prior and fixed-function likelihood, and integrate it for prediction.
Do not reuse the first-stage W posterior as a prior. GHP uses conjugate GIG/GH,
STP uses IG/Student-t, GP uses Gaussian conditioning. Intervals are pointwise
equal-tail 95% posterior predictive intervals for observations. Function and
hyperparameter estimation uncertainty is omitted in these plug-in methods.

Multistart MAP uses 12 shared and 24 individual initializations covering several
length scales; individual starts include both component assignments. Analytic
gradients, finite differences, local Hessians, conditioning and predictions are
checked. It is a local optimization search, not a global-maximum guarantee.
The individual shared-grid solver exactly diagonalizes the common covariances;
it is not a low-rank approximation. All fits are sequential and BLAS uses one thread.
GH intervals use vectorized Gaussian-mixture CDF inversion. Quadrature tails and
resolution are refined to match analytic GH log densities and converged interval
probabilities, including observations far into the predictive tails. This is
numerical refinement only; no statistical parameters are changed using targets.

Evaluation has no latent-f truth: use only held-out observed y. RMSE is pooled
over held-out points within a split; NLPD, CRPS, coverage and interval width are
pointwise averages. Report mean +/- sample SD of the FIVE SPLIT-LEVEL scores.
Splits overlap, so SD is descriptive variation, not an independent-replicate SE.
Compare four models under each sharing choice using identical splits/contexts.

Only compact prediction arrays, fits, scores and selected example figures are
saved. No dense posterior covariance matrices, HMC draws or quadrature particles
are stored. Regression examples are the first two test spectrum IDs per split.

Reproduce: python gasoline_plugin_experiment.py --stage all
Resume one case: --seed 54026 --sharing shared --model gp --stage run
Summarize: --stage summarize
Validate: python validate_gasoline_plugin.py
Dependencies: NumPy, SciPy, pandas, matplotlib, threadpoolctl, requests; rdata
(tested with 1.1.0) is needed only to convert the source RData file on first use.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('all', 'prepare', 'run', 'summarize'), default='all')
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--sharing', choices=SHARING)
    parser.add_argument('--model', choices=MODELS)
    args = parser.parse_args()
    data = prepare()
    if args.stage in ('all', 'run'):
        for seed in (SEEDS if args.seed is None else (args.seed,)):
            for sharing in (SHARING if args.sharing is None else (args.sharing,)):
                for model in (MODELS if args.model is None else (args.model,)):
                    folder, split, fit = fit_case(data, seed, sharing, model)
                    predict_case(data, seed, sharing, model, folder, split, fit)
    if args.stage in ('all', 'summarize'):
        summarize(data)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
