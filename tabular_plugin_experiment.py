"""Four plug-in models on Concrete, Power and Wine; compact, resumable outputs.

python tabular_plugin_experiment.py --dataset concrete
python tabular_plugin_experiment.py --dataset power
python tabular_plugin_experiment.py --dataset wine
python tabular_plugin_experiment.py --stage summarize
"""
import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '1'
import argparse
import hashlib
import json
from pathlib import Path
import time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import special, stats
from threadpoolctl import threadpool_limits

from GH_tabular_hmc import MODELS, ZERO_MODELS
from GH_tabular_plugin_map import fit_map, PluginPredictor
from GH_fixed_function import mixing_quadrature, marginal_distribution, StableGHMarginals, refine_density_quadrature
from load_real_datasets import load_datasets
from shared_beta_plugin_experiment import write_json, student_crps
from predictive_crps import gaussian_mixture_crps, gaussian_mixture_crps_closed_form

ROOT = Path(__file__).resolve().parent
OUTPUT = ROOT/'figure'/'tabular_single_function_plugin'
LABELS = dict(gp='GP (mu)', tp_mu_gp='STP (mu)', mu_gp='GHP (mu)', beta_gp='GHP (beta)')
TEX_NAMES = dict(gp=r'GP ($\hat\mu$)', tp_mu_gp=r'STP ($\hat\mu$)',
                 mu_gp=r'GHP ($\hat\mu$)', beta_gp=r'GHP ($\hat\beta$)')


def load_source(dataset):
    """Read existing sources without downloading or modifying their caches."""
    if dataset == 'wine':
        path = ROOT.parent/'data'/'winequality'/'winequality-white.csv'
        frame = pd.read_csv(path, sep=';')
        X, target = frame.drop(columns='quality'), frame['quality']
        hashes = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()}
    else:
        X, target = load_datasets([dataset])[dataset]
        hashes = {name: hashlib.sha256((ROOT/'data'/'real_datasets'/dataset/name).read_bytes()).hexdigest()
                  for name in ('features.csv', 'target.csv')}
    return X, target, hashes


def prepare(dataset, *, n_train=300, split_seed=17026, output=OUTPUT, starts=24,
            numerical_backend=None):
    """Remove exact duplicate rows; split whole identical-X groups before scaling."""
    X, target, source_hashes = load_source(dataset)
    raw_x, raw_y = X.to_numpy(float), target.to_numpy(float)
    if not np.isfinite(raw_x).all() or not np.isfinite(raw_y).all():
        raise ValueError('Explicit missing-value processing required')
    table = X.copy()
    table['__target__'] = raw_y
    retained = np.flatnonzero(~table.duplicated().to_numpy())
    x, y = raw_x[retained], raw_y[retained]
    _, groups = np.unique(x, axis=0, return_inverse=True)
    order = np.random.default_rng(split_seed).permutation(np.unique(groups))
    train = []
    for group in order:
        ids = np.flatnonzero(groups == group)
        if len(train)+len(ids) <= n_train:
            train.extend(ids.tolist())
        if len(train) == n_train:
            break
    if len(train) != n_train:
        raise ValueError('Cannot obtain requested training size without separating equal-X rows')
    train = np.sort(train)
    test = np.setdiff1d(np.arange(len(y)), train)
    assert not np.intersect1d(groups[train], groups[test]).size
    x_mean, x_scale = x[train].mean(0), x[train].std(0)
    y_mean, y_scale = float(y[train].mean()), float(y[train].std())
    if np.any(x_scale <= 0) or y_scale <= 0:
        raise ValueError('Constant training feature or target')
    data = dict(x=x, y=y, train_ids=train, test_ids=test, source_row_ids=retained,
                input_group_ids=groups, x_mean=x_mean, x_scale=x_scale,
                y_mean=np.array(y_mean), y_scale=np.array(y_scale),
                x_scaled=(x-x_mean)/x_scale, y_scaled=(y-y_mean)/y_scale,
                feature_names=np.asarray(X.columns, dtype=str))
    folder = Path(output)/dataset
    folder.mkdir(parents=True, exist_ok=True)
    if (folder/'dataset.npz').exists():
        with np.load(folder/'dataset.npz') as saved:
            assert set(saved.files) == set(data)
            for key in data:
                np.testing.assert_array_equal(saved[key], data[key])
    else:
        np.savez_compressed(folder/'dataset.npz', **data)
    config = dict(dataset=dataset, raw_rows=len(raw_y), removed_exact_duplicates=len(raw_y)-len(y),
                  retained_rows=len(y), n_features=x.shape[1], n_train=len(train), n_test=len(test),
                  split_seed=split_seed, optimization_seed=505, starts=starts,
                  split=f'Shuffle identical-X groups; add whole groups until exactly {n_train} training rows. Remaining groups are test.',
                  standardization='Training rows only; fixed target intercept equals training y mean.',
                  source_sha256=source_hashes,
                  dataset_sha256=hashlib.sha256((folder/'dataset.npz').read_bytes()).hexdigest(),
                  fixed_priors=dict(GHP_lambda=1., GHP_omega=1., STP_nu=5., log_parameter_prior_sd=2.),
                  models=list(MODELS), kernel='Residual ARD SE plus independent mu/beta ARD SE prior; all amplitudes, noise, and length scales learned.',
                  fitting='Collapsed log-coordinate MAP in theta,W, followed by conditional posterior function mean.',
                  prediction='Fix theta and mu/beta; recompute and integrate W under the fixed-function model.',
                  scores='Held-out observed y only, original units. Latent f is unobserved.',
                  storage='Keep dataset, MAP fits, pointwise predictions/scores; no HMC samples or dense test covariances.')
    if numerical_backend is not None:
        config['numerical_backend'] = numerical_backend
    if (folder/'experiment.json').exists():
        assert json.loads((folder/'experiment.json').read_text()) == config
    write_json(folder/'experiment.json', config)
    if dataset == 'wine':
        method = WINE_METHOD.replace('Select 300 training rows', f'Select {len(train)} training rows')
        method = method.replace('300 train and 3661 test', f'{len(train)} train and {len(test)} test')
        method = method.replace('Use 24 L-BFGS-B starts', f'Use {starts} L-BFGS-B starts')
        (folder/'method.txt').write_text(method, encoding='utf-8')
    print(f'{dataset}: {len(train)} train, {len(test)} test, {x.shape[1]} inputs; '
          f'{len(raw_y)-len(y)} exact duplicate rows removed', flush=True)
    return folder, data, config


def batch_scores(p, truth, *, check=False):
    """Evaluate exact GH/t/Normal marginals, with independent quadrature checks."""
    dist = StableGHMarginals(p, 'y') if p['mixing'] == 'gig' else marginal_distribution(p, 'y')
    logpdf = dist.logpdf(truth)
    mean, variance, ci = p['y_mean'], p['y_var'], p['y_ci']
    np.testing.assert_allclose(dist.mean(), mean, atol=1e-8, rtol=1e-9)
    np.testing.assert_allclose(dist.var(), variance, atol=1e-8, rtol=1e-9)
    np.testing.assert_allclose(dist.cdf(ci[:, 0]), .025, atol=1e-8)
    np.testing.assert_allclose(dist.cdf(ci[:, 1]), .975, atol=1e-8)
    validation = {}
    if p['mixing'] == 'gig':
        W, weights, diagnostics = mixing_quadrature(p)
        np.testing.assert_allclose(weights@W, p['W_mean'], rtol=1e-9)
        np.testing.assert_allclose(weights@(W-p['W_mean'])**2, p['W_var'], atol=1e-10, rtol=1e-8)
        means = p['location']+W[:, None]*p['conditional_beta']
        variances = W[:, None]*np.diag(p['y_scale_cov'])
        mixture_logpdf = special.logsumexp(np.log(weights[:, None])+stats.norm.logpdf(truth, means, np.sqrt(variances)), axis=0)
        if not np.allclose(logpdf, mixture_logpdf, atol=3e-6, rtol=1e-7):
            # Posterior-moment convergence can miss tiny W-tail contributions
            # important to an extreme observation's density. Extend/refine
            # the independent numerical check, keeping the predictive law.
            refine_density_quadrature(p, truth)
            W, weights, diagnostics = mixing_quadrature(p)
            means = p['location']+W[:, None]*p['conditional_beta']
            variances = W[:, None]*np.diag(p['y_scale_cov'])
            positive = weights > 0
            mixture_logpdf = special.logsumexp(np.log(weights[positive, None])+
                stats.norm.logpdf(truth, means[positive], np.sqrt(variances[positive])), axis=0)
            dist = StableGHMarginals(p, 'y')
            ci = p['y_ci']
            validation['density_quadrature_refined'] = True
        np.testing.assert_allclose(logpdf, mixture_logpdf, atol=3e-6, rtol=1e-7)
        crps, info = gaussian_mixture_crps(truth, means, variances, weights, atol=1e-8, rtol=1e-8)
        validation.update(quadrature=diagnostics,
                          max_logpdf_difference=float(max(abs(logpdf-mixture_logpdf))),
                          max_crps_error_estimate=float(max(info['integration_error_estimate'])))
        if check:
            ids = [0, len(truth)//2, len(truth)-1]
            reference = gaussian_mixture_crps_closed_form(truth[ids], means[:, ids], variances[:, ids], weights)
            np.testing.assert_allclose(crps[ids], reference, atol=2e-7, rtol=1e-8)
    elif p['mixing'] == 'ig':
        df = 2*p['W_shape']
        scale = np.sqrt(p['W_scale']/p['W_shape']*np.diag(p['y_scale_cov']))
        crps = student_crps(truth, mean, scale, df)
        if check:
            W, weights, _ = mixing_quadrature(p)
            ids = [0, len(truth)//2, len(truth)-1]
            means = np.broadcast_to(mean[ids], (len(W), len(ids)))
            variances = W[:, None]*np.diag(p['y_scale_cov'])[ids]
            reference, _ = gaussian_mixture_crps(truth[ids], means, variances, weights)
            np.testing.assert_allclose(crps[ids], reference, atol=2e-7, rtol=1e-8)
    else:
        crps = gaussian_mixture_crps_closed_form(truth, mean[None, :], variance[None, :])
    return dict(nlpd=-logpdf, crps=crps, pit=dist.cdf(truth),
                covered=(truth >= ci[:, 0]) & (truth <= ci[:, 1])), validation


def run_model(folder, data, config, model, *, initial_points=()):
    case = folder/model
    case.mkdir(exist_ok=True)
    if (case/'report.json').exists():
        report = json.loads((case/'report.json').read_text())
        assert report['dataset_sha256'] == config['dataset_sha256']
        assert report['fit_sha256'] == hashlib.sha256((case/'fit.json').read_bytes()).hexdigest()
        assert (case/'predictions.npz').exists()
        return report
    train, test = data['train_ids'], data['test_ids']
    x, y = data['x_scaled'][train], data['y_scaled'][train]
    start_time = time.perf_counter()
    if (case/'fit.json').exists():
        fit = json.loads((case/'fit.json').read_text())
        assert fit['dataset_sha256'] == config['dataset_sha256']
    else:
        checkpoint = case/'optimization_starts.json'
        completed = json.loads(checkpoint.read_text()) if checkpoint.exists() else []
        def progress(record, z):
            completed.append(dict(record=record, effective_log_parameters=z))
            write_json(checkpoint, completed)
            print(f'{folder.name} {model}: start {record["start"]+1}/{config["starts"]}, '
                  f'logp={record["log_posterior"]:.6f}, grad={record["max_abs_gradient"]:.2g}, '
                  f'accepted={record["converged"]}', flush=True)
        fitting = fit_map
        if config.get('learn_mixing', False):
            from GH_tabular_learned_plugin_map import fit_learned_map
            fitting = fit_learned_map
        fit = fitting(y, x, model, progress=progress, completed=list(completed),
                      starts=config['starts'], seed=config['optimization_seed'],
                      backend=config.get('numerical_backend', 'dense'), initial_points=initial_points)
        fit['dataset_sha256'] = config['dataset_sha256']
        write_json(case/'fit.json', fit)
    predictor = PluginPredictor(y, x, fit['log_parameters'], model,
                                **fit.get('mixing_hyperparameters', {}))
    compact, checks = [], []
    # 64-point batches bound memory even for all 9,227 held-out power rows.
    for start in range(0, len(test), 64):
        ids = test[start:start+64]
        p = predictor.predict(data['x_scaled'][ids], y_mean=float(data['y_mean']), y_scale=float(data['y_scale']))
        point, check = batch_scores(p, data['y'][ids], check=start == 0)
        saved = {key: p[key] for key in ('location', 'conditional_beta', 'f_mean', 'y_mean', 'f_var', 'y_var', 'f_ci', 'y_ci')}
        saved.update(f_scale_var=np.diag(p['f_scale_cov']), y_scale_var=np.diag(p['y_scale_cov']), **point)
        compact.append(saved)
        checks.append(check)
        if start == 0 or (start//64+1) % 20 == 0:
            print(f'{folder.name} {model}: scored {min(start+64,len(test))}/{len(test)} test rows', flush=True)
    combined = {key: np.concatenate([batch[key] for batch in compact]) for key in compact[0]}
    combined['test_ids'] = test
    combined['function_train_scaled'] = predictor.function_train
    combined['function_test_scaled'] = np.concatenate([predictor.function(data['x_scaled'][test[s:s+256]]) for s in range(0,len(test),256)])
    np.savez_compressed(case/'predictions.npz', **combined)
    report = dict(dataset=config['dataset'], model=model, n_train=len(train), n_test=len(test),
                  y_rmse=float(np.sqrt(np.mean((data['y'][test]-combined['y_mean'])**2))),
                  y_nlpd=float(combined['nlpd'].mean()), y_crps=float(combined['crps'].mean()),
                  y_coverage_95=float(combined['covered'].mean()),
                  y_interval_width=float(np.mean(combined['y_ci'][:, 1]-combined['y_ci'][:, 0])),
                  W_joint_map=predictor.W_map,
                  prediction_W_posterior=predictor.mixing_parameters,
                  max_abs_gradient=fit['max_abs_gradient'], max_hessian_eigenvalue=fit['max_hessian_eigenvalue'],
                  dataset_sha256=config['dataset_sha256'],
                  fit_sha256=hashlib.sha256((case/'fit.json').read_bytes()).hexdigest(),
                  predictions_sha256=hashlib.sha256((case/'predictions.npz').read_bytes()).hexdigest(),
                  validation_passed=True, batch_checks=checks, elapsed_seconds=time.perf_counter()-start_time)
    if model not in ZERO_MODELS:
        report['W_used_for_function_estimate'] = predictor.W_map
    if 'mixing_hyperparameters' in fit:
        report['mixing_hyperparameters'] = fit['mixing_hyperparameters']
    write_json(case/'report.json', report)
    print(f'{folder.name} {model}: RMSE={report["y_rmse"]:.5f}, NLPD={report["y_nlpd"]:.5f}, CRPS={report["y_crps"]:.5f}', flush=True)
    return report


def summarize(datasets=('concrete', 'power'), *, destination=OUTPUT, output=OUTPUT):
    reports = []
    for dataset in datasets:
        folder = Path(output)/dataset
        for model in MODELS:
            file = folder/model/'report.json'
            if file.exists():
                reports.append(json.loads(file.read_text()))
    if not reports:
        return
    keys = ['dataset', 'model', 'n_train', 'n_test', 'y_rmse', 'y_nlpd', 'y_crps', 'y_coverage_95', 'y_interval_width']
    table = pd.DataFrame([{key: item[key] for key in keys} for item in reports])
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    table.to_csv(destination/'scores.csv', index=False)
    caption = (r'Single-function plug-in regression, split seed 17026: 300 training observations per dataset. '
               r'Scores use held-out observed $y$ in original units. All models fix the estimated function '
               r'and hyperparameters and integrate mixing uncertainty where applicable.')
    if tuple(datasets) == ('wine',):
        counts = json.loads((Path(output)/'wine'/'experiment.json').read_text())
        caption = (f'White-wine quality regression, split seed {counts["split_seed"]}: '
                   f'{counts["n_train"]} training and {counts["n_test"]} test observations '
                   r'after removing 937 exact duplicate records. Quality is treated as continuous. '
                   r'Scores evaluate held-out observed $y$. Hyperparameters and $\mu$ or $\beta$ are plug-in '
                   r'estimates; mixing uncertainty is integrated. GHP fixes $\lambda=\omega=1$, and STP fixes $\nu=5$.')
    lines = [r'\begin{table}[htbp]', r'\centering',
        r'\caption{'+caption+'}',
        r'\label{tab:'+('wine_plugin' if tuple(datasets) == ('wine',) else 'tabular_plugin')+'}', r'\begin{tabular}{llrrrr}', r'\toprule',
        r'Dataset & Model & RMSE & NLPD & CRPS & 95\% coverage (\%) \\', r'\midrule']
    for index, dataset in enumerate(datasets):
        rows = table[table.dataset == dataset]
        for row in rows.itertuples():
            cells = [dataset.capitalize(), TEX_NAMES[row.model], f'{row.y_rmse:.5f}', f'{row.y_nlpd:.5f}', f'{row.y_crps:.5f}', f'{100*row.y_coverage_95:.2f}']
            lines.append(' & '.join(cells)+r' \\')
        if index < len(datasets)-1:
            lines.append(r'\midrule')
        if len(rows) != 4:
            continue
        with np.load(Path(output)/dataset/'dataset.npz') as saved:
            truth = saved['y'][saved['test_ids']]
        fig, axes = plt.subplots(2, 4, figsize=(15, 7))
        for j, model in enumerate(MODELS):
            with np.load(Path(output)/dataset/model/'predictions.npz') as p:
                axes[0,j].scatter(truth, p['y_mean'], s=4, alpha=.4)
                bounds = [min(truth.min(),p['y_mean'].min()), max(truth.max(),p['y_mean'].max())]
                axes[0,j].plot(bounds,bounds,'k--',lw=1)
                axes[0,j].set(title=LABELS[model],xlabel='Observed target',ylabel='Predicted mean')
                axes[1,j].hist(p['pit'],bins=np.linspace(0,1,11),density=True,alpha=.75)
                axes[1,j].axhline(1,color='k',ls='--',lw=1)
                axes[1,j].set(xlabel='Predictive CDF at observation',ylabel='Density',xlim=(0,1))
        fig.suptitle(f'{dataset.capitalize()}: held-out predictions and calibration')
        fig.tight_layout()
        fig.savefig(Path(output)/dataset/'regression.png', dpi=150)
        fig.savefig(Path(output)/dataset/'regression.pdf')
        plt.close(fig)
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (destination/'comparison.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(table.to_string(index=False), flush=True)


WINE_METHOD = '''WHITE WINE: FOUR-MODEL SINGLE-FUNCTION PLUG-IN REGRESSION

Source and task
Read the same local winequality-white.csv used by wine.py and the earlier
beta-prior Wine experiment. Its SHA256 is recorded in experiment.json.
Each row is a wine sample with 11 physicochemical inputs and a quality score.
Treat quality as a continuous response. Model ONE function of 11 inputs, with
one global W per mixing model; rows are not independent function realizations.
NLPD is a continuous predictive density score, not a probability of an integer
rating. There is no known latent f, so only observed-y scores are evaluated.

Split and preprocessing
Use the Concrete/Power tabular protocol unchanged: remove exact (X,y)
duplicates, keeping the first occurrence; keep identical-X records in the same
split. Seed 17026 shuffles unique-X groups. Select 300 training rows; all other
groups are test. The source has 4898 rows, 937 exact duplicates, and 3961 retained
rows: 300 train and 3661 test. No identical inputs cross the split.
Standardize inputs and response using training means and population SDs only.
The training response mean is a fixed intercept in original quality units.
In particular, GHP-beta has mu=0 on the CENTERED response scale. This is a new
split/protocol, so its scores are not directly comparable to the earlier
192/48 Wine experiment with lambda=-0.5 and beta uncertainty integrated out.

Models, on the standardized response scale
GP-mu:    y = mu + e; W=1.
STP-mu:   y = mu + sqrt(W)e; W~IG(2.5,1.5), fixed nu=5.
GHP-mu:   y = mu + sqrt(W)e; W~GIG(lambda=1,chi=psi=1).
GHP-beta: y = W beta + sqrt(W)e; same fixed GIG prior.
e~N(0,K+kI). mu or beta has an independent zero-mean GP prior with covariance
Sigma. K and Sigma are separate ARD SE kernels, each with an amplitude and 11
inverse squared length scales. All 25 kernel/noise parameters are learned.
No model learns both a nonconstant mu and beta. STP plus a GP-prior mu is not
a pure Student-t process after mu is marginalized during training.

Estimation and prediction
Marginalize the Gaussian functions for the training likelihood, and optimize
the joint posterior in log(theta), log(W) (GP omits W). Log(theta) priors are
independent Normals with SD=2 and center
log([0.5*s,0.5*s,0.1*s,11 entries of 1/11,11 entries of 0.25/11]),
where s is the variance of standardized training y. Include log-W Jacobians.
Use 24 L-BFGS-B starts, seed 505, alternating the two kernel assignments.
Select the best stationary candidate, with finite-difference gradient and
negative local-Hessian checks. Global optimality is not guaranteed.

At the selected theta_hat,W_hat, set one function to its conditional GP mean:
  mu_hat(*) = Sigma(*,X) C^-1 y, C=W_hat(K+kI)+Sigma;
  beta_hat(*) = W_hat Sigma(*,X) C^-1 y, C=W_hat(K+kI)+W_hat^2 Sigma.
Freeze theta and this function. Recompute p(W|y,theta_hat,function_hat) from
the ORIGINAL mixing prior and the fixed-function training likelihood, then
integrate W for regression. This prediction W posterior differs from the
collapsed MAP W used to estimate the function. This is a two-stage plug-in
method; it omits function and hyperparameter uncertainty and is not full Bayes.
No test targets are used in parameter or function estimation.

Evaluation and storage
Report RMSE, mean pointwise NLPD, mean CRPS, equal-tail 95% observation interval
coverage and average interval width, in original quality units. This is one
train/test split, so no across-split SD or standard error is reported.
Use analytic Normal/Student-t/GH densities and conditional moments; check
mixture quadrature, quantiles and selected CRPS values independently.
Predict in batches of 64. Save pointwise summaries, function estimates and
fits, with no dense test covariance, quadrature particles or HMC samples.
The plots show observed versus predicted scores and predictive CDF values;
integer-valued outcomes can produce structure in that CDF histogram.

Results: figure/tabular_single_function_plugin/wine/
Run: python tabular_plugin_experiment.py --dataset wine
Resume one model: add --model gp (or tp_mu_gp, mu_gp, beta_gp)
Summarize: python tabular_plugin_experiment.py --stage summarize --dataset wine
Validate: python validate_tabular_plugin_experiment.py --dataset wine
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', choices=['concrete','power','wine'])
    parser.add_argument('--model', choices=MODELS)
    parser.add_argument('--stage', choices=['all','prepare','summarize'], default='all')
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        if args.stage == 'summarize':
            if args.dataset:
                summarize((args.dataset,), destination=OUTPUT/args.dataset)
            else:
                summarize()
            return
        if args.dataset is None:
            parser.error('--dataset is required except for --stage summarize')
        folder,data,config = prepare(args.dataset)
        if args.stage == 'prepare':
            return
        for model in ([args.model] if args.model else MODELS):
            run_model(folder,data,config,model)
        if args.dataset == 'wine':
            summarize(('wine',), destination=folder)
        else:
            summarize()


if __name__ == '__main__':
    main()
