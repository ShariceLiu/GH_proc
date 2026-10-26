"""Train shared GP/GHP/GP+TP kernels on independent GHP-beta functions.

Default: 30 training functions x 31 observations; 20 new functions with five
revealed observations each. Same generating parameters as the earlier 1D run.
Run python hmc_multiple_functions_experiment.py; use --stage for checkpoints.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import special
from threadpoolctl import threadpool_limits

from GH_proc_w_betaprior import Multivariate_GH_1d
from GH_multiple_hmc import sample_parameters, context_log_weights, resample_context, MODELS
from GH_proc_joint_hmc import predict_joint_mixture
from GP_TP_hmc import predict_baseline_mixture
from fit_beta_mu_1d_experiment import _json_default, _save_csv
from hmc_gp_tp_1d_experiment import score_predictions, diagnostics_summary


LABELS = {'gp': 'GP', 'tp_mu_gp': 'STP + GP mean', 'mu_gp': 'GHP (mu)', 'beta_gp': 'GHP (beta)'}
COLORS = dict(zip(MODELS, ['C2', 'C3', 'C1', 'C0']))
TRUTH = dict(l=1., omega=1., mu=0., v_0=.2, w=2., k=.05, sigma_v0=1., sigma_w=.15)


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=_json_default), encoding='utf-8')


def prepare(folder, *, seed=6026, n_train=30, n_test=20):
    """Generate full coherent realizations once; never select/reject by outcomes."""
    folder.mkdir(parents=True, exist_ok=True)
    settings = dict(seed=seed, n_train=n_train, n_test=n_test, parameters=TRUTH,
                    n_grid=121, domain=[0., 10.], train_stride=4, context_count=5)
    if (folder/'dataset.npz').exists():
        if json.loads((folder/'generation.json').read_text()) != settings:
            raise ValueError('Use a new output directory for different generating settings')
        return
    if n_train < 2 or n_test < 2:
        raise ValueError('At least two training and two test functions are needed')
    x = np.linspace(0., 10., 121)
    children = np.random.SeedSequence(seed).spawn(n_train+n_test)
    f, y, beta, W = [], [], [], []
    for child in children:
        fs, ys, bs, ws = Multivariate_GH_1d(x, **TRUTH, beta_prior=True,
                                           return_latents=True, seed=np.random.default_rng(child))
        f.append(fs); y.append(ys); beta.append(bs); W.append(ws)
    train_ids, context_ids = np.arange(0, len(x), 4), np.arange(0, len(x), 30)
    np.savez_compressed(folder/'dataset.npz', x=x, f=f, y=y, beta=beta, W=W,
                        train_function_ids=np.arange(n_train),
                        test_function_ids=np.arange(n_train, n_train+n_test),
                        train_ids=train_ids, context_ids=context_ids,
                        test_ids=np.setdiff1d(np.arange(len(x)), context_ids))
    write_json(folder/'generation.json', settings)


def load_data(folder):
    with np.load(folder/'dataset.npz') as data:
        return {key: data[key].copy() for key in data.files}


def fit_model(folder, data, model, *, warmup=1500, draws=2000, seed=7026,
              target_accept=.95, max_retries=1):
    model_folder = folder/model
    model_folder.mkdir(parents=True, exist_ok=True)
    ids = data['train_ids']
    groups = [(data['x'][ids], data['y'][data['train_function_ids']][:, ids])]
    attempts_file = model_folder/'attempts.json'
    attempts = json.loads(attempts_file.read_text()) if attempts_file.exists() else []
    offset = len(attempts)
    started = time.perf_counter()
    for retry in range(max_retries+1):
        number = offset+retry+1
        settings = dict(warmup=warmup*2**retry, draws=draws*2**retry,
                        seed=seed+MODELS.index(model)+10000*(number-1),
                        target_accept=min(.999, target_accept+.04*retry),
                        chains=4, leapfrog_steps=(15, 35))
        print(f'{model}: joint fit of {len(groups[0][1])} training functions, attempt {number}', flush=True)
        fit = sample_parameters(groups, model=model, **settings)
        np.savez_compressed(model_folder/f'attempt_{number}_samples.npz',
                            parameter_names=fit['parameter_names'],
                            **{key: value for key, value in fit.items() if isinstance(value, np.ndarray)})
        attempt = dict(attempt=number, settings=settings, **diagnostics_summary(fit),
                       warmup_divergences=int(fit['warmup_divergent'].sum()))
        attempts.append(attempt)
        write_json(attempts_file, attempts)
        print(json.dumps(attempt, default=_json_default), flush=True)
        if fit['diagnostics_ok']:
            break
    report = dict(model=model, selected_samples=f'attempt_{number}_samples.npz', attempts=attempts,
                  fixed_mixing=dict(l=1., omega=1.) if model in ('mu_gp', 'beta_gp') else
                  (dict(nu=5., ig_shape=2.5, ig_scale=1.5) if model == 'tp_mu_gp' else None),
                  training_function_ids=data['train_function_ids'], training_point_ids=ids,
                  seconds=time.perf_counter()-started,
                  **{key: fit[key] for key in ['parameter_names', 'prior_center', 'prior_sd',
                     'diagnostics', 'effective_diagnostics', 'diagnostics_ok', 'summary', 'initialization']})
    write_json(model_folder/'fit.json', report)
    _save_csv(model_folder/'parameters.csv', fit['summary'])
    return report


def predict_model(folder, data, model, *, prediction_draws=6000, seed=8026):
    """Independent new-function conditioning with quadrature-checked weights.

    Each new function's five observations update p(theta|training data)
    separately; contexts are not pooled together. Training W samples are
    discarded here. A fresh W has the fixed model prior before its context.
    """
    model_folder = folder/model
    report = json.loads((model_folder/'fit.json').read_text())
    if not report['diagnostics_ok']:
        raise ValueError(f'{model}: resolve training HMC diagnostics before prediction')
    with np.load(model_folder/report['selected_samples']) as saved:
        theta = saved['theta_log_samples'].reshape(-1, 5)
    context_ids, test_functions = data['context_ids'], data['test_function_ids']
    contexts = data['y'][test_functions][:, context_ids]
    print(f'{model}: updating posterior separately from each new function context', flush=True)
    nodes = 192
    t, logw = context_log_weights(theta, data['x'][context_ids], contexts, model=model, quadrature_nodes=nodes)
    quad_error = 0.
    if model != 'gp':
        # Compare both evidence and theta weights, not only a single likelihood.
        for _ in range(3):
            fine_t, fine_logw = context_log_weights(theta, data['x'][context_ids], contexts,
                                                    model=model, quadrature_nodes=nodes*2)
            marginal = special.logsumexp(logw, axis=2)
            fine_marginal = special.logsumexp(fine_logw, axis=2)
            quad_error = float(np.max(np.abs(marginal-fine_marginal)))
            t, logw, nodes = fine_t, fine_logw, nodes*2
            if quad_error < 1e-5:
                break
        if quad_error >= 1e-5:
            raise ValueError(f'New-function W quadrature has not converged: {quad_error}')
    rows, inference = [], []
    for j, function in enumerate(test_functions):
        q, diagnostics = resample_context(theta, logw[j], t, model=model,
                                          draws=prediction_draws, seed=seed+100*MODELS.index(model)+j)
        diagnostics.update(function_id=int(function), quadrature_nodes=len(t),
                           max_quadrature_log_error=quad_error)
        if diagnostics['theta_importance_ess'] < max(200., .03*len(theta)) or diagnostics['edge_mass'] > 1e-8:
            raise ValueError(f'{model}, function {function}: context update needs further inference: {diagnostics}')
        print(f'{model}: test function {j+1}/{len(test_functions)}, theta weight ESS={diagnostics["theta_importance_ess"]:.0f}', flush=True)
        fit = dict(model=model, log_samples=q[None], nu=None)
        predict = predict_baseline_mixture if model == 'gp' else predict_joint_mixture
        p = predict(contexts[j], data['x'][context_ids], data['x'], fit)
        local = dict(f=data['f'][function], y=data['y'][function], test_ids=data['test_ids'])
        row = dict(model=model, function_id=int(function), true_W=float(data['W'][function]),
                   **score_predictions(local, p, previous=model != 'gp'))
        if model != 'gp':
            ws = np.exp(q[:, 5])
            diagnostics.update(W_mean=float(ws.mean()), W_std=float(ws.std()),
                               W_lower=float(np.quantile(ws, .025)), W_upper=float(np.quantile(ws, .975)))
        # Parameter draws reconstruct every Gaussian component, avoiding large
        # duplicated mean/variance arrays on disk for 80 regressions.
        np.savez_compressed(model_folder/f'function_{function:02d}_prediction.npz',
                            posterior_log_parameters=q, weights=p['weights'],
                            **{key: value for key, value in p.items()
                               if not key.startswith('component_') and key != 'weights'})
        rows.append(row)
        inference.append(diagnostics)
        _save_csv(model_folder/'per_function_scores.csv', rows)
        write_json(model_folder/'prediction_diagnostics.json', inference)
    return rows


def aggregate(rows):
    """Pooled RMSE and pointwise scores, with each test function equally weighted."""
    result = {}
    for target in ['f', 'y']:
        rmses = np.array([r[f'{target}_rmse'] for r in rows])
        result[f'{target}_rmse'] = float(np.sqrt(np.mean(rmses**2)))
        result[f'{target}_mean_function_rmse'] = float(rmses.mean())
        for metric in ['mean_marginal_nlpd', 'coverage_95', 'mean_interval_width']:
            result[f'{target}_{metric}'] = float(np.mean([r[f'{target}_{metric}'] for r in rows]))
    return result


def summarize(folder, data):
    import csv
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows, reports, model_rows, predictions = [], {}, {}, {}
    for model in MODELS:
        model_folder = folder/model
        if not (model_folder/'prediction_diagnostics.json').exists():
            continue
        with (model_folder/'per_function_scores.csv').open() as stream:
            scores = [{key: (value if key == 'model' else float(value)) for key, value in row.items()}
                      for row in csv.DictReader(stream)]
        if len(scores) != len(data['test_function_ids']):
            continue
        fit = json.loads((model_folder/'fit.json').read_text())
        inference = json.loads((model_folder/'prediction_diagnostics.json').read_text())
        for row, diagnostic in zip(scores, inference):
            if int(row['function_id']) != diagnostic['function_id']:
                raise ValueError('Context diagnostics and score function IDs disagree')
            row['context_joint_nlpd_per_point'] = -diagnostic['log_context_evidence']/len(data['context_ids'])
        reports[model] = dict(fit=fit, prediction=inference)
        model_rows[model] = scores
        row = dict(model=model, n_test_functions=len(scores), **aggregate(scores),
                   context_joint_nlpd_per_point=float(np.mean([r['context_joint_nlpd_per_point'] for r in scores])),
                   max_rhat=max(fit['diagnostics']['rhat']), min_bulk_ess=min(fit['diagnostics']['ess_bulk']),
                   min_tail_ess=min(fit['diagnostics']['ess_tail']),
                   divergences=fit['attempts'][-1]['divergences'], diagnostics_ok=fit['diagnostics_ok'],
                   min_theta_importance_ess=min(d['theta_importance_ess'] for d in inference))
        rows.append(row)
        predictions[model] = {}
        for function in data['test_function_ids']:
            with np.load(model_folder/f'function_{function:02d}_prediction.npz') as saved:
                predictions[model][int(function)] = {key: saved[key].copy() for key in ['f_mean', 'f_ci', 'y_mean', 'y_ci']}
    if not rows:
        return
    _save_csv(folder/'scores.csv', rows)
    _save_csv(folder/'per_function_scores.csv', [r for model in model_rows for r in model_rows[model]])
    parameters = [dict(model=model, **row) for model in reports for row in reports[model]['fit']['summary']]
    _save_csv(folder/'parameters.csv', parameters)
    _save_csv(folder/'shared_parameters.csv', [dict(model=model,
              generating_value=TRUTH.get(row['parameter']) if model == 'beta_gp' else None, **row)
              for model in reports for row in reports[model]['fit']['summary'][:5]])
    paired = []
    if 'gp' in model_rows:
        rng = np.random.default_rng(9026)
        n = len(model_rows['gp'])
        ids = rng.integers(0, n, size=(5000, n))
        for model, candidate in model_rows.items():
            for target in ['f', 'y']:
                for metric in ['rmse', 'mean_marginal_nlpd']:
                    a = np.array([r[f'{target}_{metric}'] for r in candidate])
                    b = np.array([r[f'{target}_{metric}'] for r in model_rows['gp']])
                    if metric == 'rmse':
                        delta = np.sqrt((a[ids]**2).mean(axis=1))-np.sqrt((b[ids]**2).mean(axis=1))
                        point = np.sqrt(np.mean(a*a))-np.sqrt(np.mean(b*b))
                    else:
                        delta = (a[ids]-b[ids]).mean(axis=1)
                        point = (a-b).mean()
                    paired.append(dict(model=model, target=target, metric=metric, difference_from_gp=float(point),
                                       lower=float(np.quantile(delta, .025)), upper=float(np.quantile(delta, .975))))
            context_delta = np.array([r['context_joint_nlpd_per_point'] for r in candidate])-np.array(
                [r['context_joint_nlpd_per_point'] for r in model_rows['gp']])
            boot = context_delta[ids].mean(axis=1)
            paired.append(dict(model=model, target='context_y', metric='joint_nlpd_per_point',
                               difference_from_gp=float(context_delta.mean()),
                               lower=float(np.quantile(boot, .025)), upper=float(np.quantile(boot, .975))))
        _save_csv(folder/'paired_comparison.csv', paired)
    # Every test function is shown in generation order, not selected by scores.
    functions = data['test_function_ids']
    for page in range((len(functions)+4)//5):
        subset = functions[page*5:(page+1)*5]
        fig, axes = plt.subplots(len(subset), 4, figsize=(16, 2.6*len(subset)),
                                 squeeze=False, sharex=True, sharey='row', constrained_layout=True)
        for j, function in enumerate(subset):
            for col, model in enumerate(MODELS):
                ax = axes[j, col]
                if model not in predictions:
                    ax.set_visible(False)
                    continue
                p = predictions[model][int(function)]
                ax.plot(data['x'], data['f'][function], 'k--', lw=1, label='True f')
                ax.plot(data['x'], p['f_mean'], color=COLORS[model], label='Posterior mean')
                ax.fill_between(data['x'], *p['f_ci'].T, alpha=.22, color=COLORS[model], label='Pointwise 95% interval')
                ids = data['context_ids']
                ax.scatter(data['x'][ids], data['y'][function, ids], s=17, c='black', label='Five observed y')
                ax.grid(alpha=.2)
                ax.set_title(LABELS[model], fontsize=10)
                if col == 0:
                    ax.set_ylabel(f'Test {int(function-functions[0])+1}, W={data["W"][function]:.2f}\nf')
                if j == len(subset)-1:
                    ax.set_xlabel('x')
        axes[0, 0].legend(fontsize=6)
        fig.suptitle(f'Shared parameters learned from {len(data["train_function_ids"])} independent functions '
                     f'| {len(data["context_ids"])} observations of each new function')
        fig.savefig(folder/f'regression_page_{page+1}.png', dpi=145)
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for model, scores in model_rows.items():
        ws = [r['true_W'] for r in scores]
        axes[0].scatter(ws, [r['f_rmse'] for r in scores], label=LABELS[model], color=COLORS[model], alpha=.7)
        axes[1].scatter(ws, [r['y_mean_marginal_nlpd'] for r in scores], label=LABELS[model], color=COLORS[model], alpha=.7)
    for ax in axes:
        ax.set_xlabel('Generating W of held-out function')
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    axes[0].set_ylabel('Latent f RMSE')
    axes[1].set_ylabel('Noisy y marginal NLPD')
    fig.savefig(folder/'performance_by_W.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(6, 4, figsize=(16, 12), constrained_layout=True)
    for col, model in enumerate(MODELS):
        if model not in reports:
            for ax in axes[:, col]:
                ax.set_visible(False)
            continue
        fit = reports[model]['fit']
        with np.load(folder/model/fit['selected_samples']) as saved:
            q = saved['log_samples']
        for j in range(6):
            if j == 5 and model == 'gp':
                axes[j, col].set_visible(False)
                continue
            values = q[..., j] if j < 5 else q[..., 5:].mean(axis=-1)
            for chain in values:
                axes[j, col].plot(chain, lw=.35, alpha=.6)
            axes[j, col].set_ylabel('log '+fit['parameter_names'][j] if j < 5 else 'mean log W', fontsize=8)
            axes[j, col].set_title(LABELS[model], fontsize=9)
    fig.savefig(folder/'traces.png', dpi=140)
    plt.close(fig)
    labels_tex = dict(gp='GP', tp_mu_gp='STP + GP mean', mu_gp=r'GHP ($\mu$)', beta_gp=r'GHP ($\beta$)')
    lines = [r'% Requires \usepackage{booktabs}', r'\begin{table}[htbp]', r'\centering',
             r'\caption{Prediction of new GHP-beta functions after learning shared '
             r'hyperparameters from independent training functions. Five observations '
             r'are revealed per new function. RMSE is pooled over held-out points; '
             r'NLPD is the mean pointwise negative log predictive density.}',
             r'\label{tab:multiple_function_regression}', r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Model & RMSE ($f$) & NLPD ($f$) & RMSE ($y$) & NLPD ($y$) \\', r'\midrule']
    for row in rows:
        lines.append(labels_tex[row['model']]+' & '+' & '.join(f'{row[key]:.6f}' for key in
                     ['f_rmse', 'f_mean_marginal_nlpd', 'y_rmse', 'y_mean_marginal_nlpd'])+r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    (folder/'comparison_table.tex').write_text('\n'.join(lines)+'\n')
    write_json(folder/'report.json', dict(generation=json.loads((folder/'generation.json').read_text()),
               scores=rows, paired_comparison=paired, models=reports,
               inference='Shared theta and independent per-training-function W via HMC. New functions '
               'condition separately on five context observations, updating theta by importance weights '
               'and integrating a fresh W by checked log-W quadrature. Conditional Gaussian predictions '
               'are averaged over stratified draws of the updated joint posterior.',
               scope=f'One training ensemble and {len(data["test_function_ids"])} independent held-out functions; bootstrap resamples '
               'test functions conditional on the fitted ensemble, not independent training experiments.'))
    (folder/'README.md').write_text(
        '# Multiple independent functions generated by GHP beta\n\n'
        'Run `python hmc_multiple_functions_experiment.py`. Stages: `--stage prepare`, '
        '`--stage fit --models beta_gp`, `--stage predict --models beta_gp`, '
        '`--stage summarize`. Use a new output folder for new simulation settings.\n\n'
        'Default generation: 30 training functions and 20 independent test functions, '
        '121 grid points on [0,10]. Generating lambda=omega=1, mu=0, v_0=.2, w=2, '
        'k=.05, sigma_v0=1, sigma_w=.15. Each realization independently draws W, '
        'beta and a residual GP; f=W*beta+residual. Noise variance is W*k. '
        'The original generator is called once per full realization, then split.\n\n'
        'All training functions share five inferred kernel/noise parameters within '
        'each model; the latent functions and W are independent across realizations. '
        '31 observations per training function. GIG lambda=omega=1 and TP nu=5 '
        '(IG shape=2.5, scale=1.5) are fixed. STP means TP residual + independent GP mean. '
        'Log-parameter priors are Normal(sd=2), with centers based only on observed '
        'training-function data. Generating kernel/noise parameters and latent '
        'truth are not supplied to the fitter.\n\n'
        'New functions reveal y at x=0,2.5,5,7.5,10. Each context updates the '
        'training theta posterior separately; contexts from other test functions '
        'are not pooled. A fresh W is integrated numerically, with doubled-order '
        'quadrature checks and importance-weight ESS checks. Stratified posterior '
        'resampling uses independent jitter per stratum, followed by analytic '
        'conditional Gaussian moments and actual mixture quantiles. The remaining '
        '116 points per function are used only for evaluation.\n\n'
        '- scores.csv, comparison_table.tex: pooled errors, marginal NLPD, coverage, widths.\n'
        '- per_function_scores.csv, paired_comparison.csv: scores and paired bootstrap '
        'intervals versus GP; resampling units are whole test functions.\n'
        '- regression_page_*.png: every test function, in generation order.\n'
        '- performance_by_W.png, traces.png: scale-dependent errors and HMC traces.\n'
        '- parameters.csv: shared theta and training-function W posterior summaries.\n'
        '- shared_parameters.csv: only the five shared hyperparameters; generating '
        'values are shown for the correctly specified beta model.\n'
        '- Model folders retain every HMC attempt, metadata and prediction diagnostics.\n'
        '- function_*_prediction.npz: posterior summaries, covariances and equally '
        'weighted log(theta,W) draws. These reconstruct all conditional Gaussian '
        'components using the saved context. GP draws have five coordinates.\n\n'
        'The additional context_joint_nlpd_per_point score evaluates the joint density '
        'of the five new-function observations BEFORE conditioning on them, given '
        'only the training ensemble. It is distinct from regression NLPD after '
        'observing that context.\n\n'
        'Results describe this generated ensemble. Bootstrap uncertainty conditions '
        'on the fitted training ensemble; it does not replace repeating training '
        'ensembles or testing other generating models.\n', encoding='utf-8')
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', default='figure/multiple_functions_ghp_beta')
    parser.add_argument('--stage', choices=['all', 'prepare', 'fit', 'predict', 'summarize'], default='all')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--n-train', type=int, default=30)
    parser.add_argument('--n-test', type=int, default=20)
    parser.add_argument('--simulation-seed', type=int, default=6026)
    parser.add_argument('--sampler-seed', type=int, default=7026)
    parser.add_argument('--warmup', type=int, default=1500)
    parser.add_argument('--draws', type=int, default=2000)
    parser.add_argument('--target-accept', type=float, default=.95)
    parser.add_argument('--max-retries', type=int, default=1)
    parser.add_argument('--prediction-draws', type=int, default=6000)
    args = parser.parse_args()
    folder = Path(args.output_dir)
    with threadpool_limits(limits=1):
        if args.stage in ('all', 'prepare'):
            prepare(folder, seed=args.simulation_seed, n_train=args.n_train, n_test=args.n_test)
        if args.stage != 'prepare':
            data = load_data(folder)
            if args.stage in ('all', 'fit'):
                for model in args.models:
                    fit_model(folder, data, model, warmup=args.warmup, draws=args.draws,
                              seed=args.sampler_seed, target_accept=args.target_accept, max_retries=args.max_retries)
            if args.stage in ('all', 'predict'):
                for model in args.models:
                    predict_model(folder, data, model, prediction_draws=args.prediction_draws)
            if args.stage in ('all', 'summarize'):
                summarize(folder, data)
