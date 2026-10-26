"""Seed-5 MAP/full-Bayes comparison learning GIG lambda/omega and STP nu.

Run: python learned_mixing_1d_experiment.py
Resumes completed MAP fits, HMC attempts and prediction summaries. No thinning;
all retained states, including rejections, contribute to prediction. Output
does not store large component prediction arrays: they regenerate from samples.
"""
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
from scipy import optimize, special, stats
from threadpoolctl import threadpool_limits

from GH_learned_mixing import LearnedMixingTarget, MODELS
from GH_proc_joint_hmc import effective_coordinates, physical_coordinates
from GH_proc_w_betaprior import _mixture_intervals
from GH_proc_w_betaprior_hmc import sample_hmc, chain_diagnostics
from GH_zero_tabular_bayes import mixture_crps, independent_crps


ROOT = Path('figure/single_function_30_learned_mixing_seed5')
SOURCE = Path('figure/single_function_30_map_fixed_functions_seed5/dataset.npz')
LABELS = {'beta_gp': r'GHP $\beta$', 'tp_mu_gp': r'STP + GP $\mu$'}


def write_json(path, data):
    def convert(x):
        if isinstance(x, np.ndarray): return x.tolist()
        if isinstance(x, np.generic): return x.item()
        raise TypeError(type(x).__name__)
    path.write_text(json.dumps(data, indent=2, default=convert, allow_nan=False), encoding='utf-8')


def prepare(root=ROOT):
    root.mkdir(parents=True, exist_ok=True)
    source_bytes = SOURCE.read_bytes()
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    dest = root/'dataset.npz'
    if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() != source_hash:
        raise RuntimeError('Existing dataset differs from requested source')
    if not dest.exists(): dest.write_bytes(source_bytes)
    with np.load(dest) as d:
        data = {k: d[k].copy() for k in d.files}
    train, test = data['train_ids'], data['test_ids']
    assert len(train) == 30 and len(test) == 151 and not np.intersect1d(train, test).size
    assert len(np.union1d(train, test)) == len(data['x']) == 181
    write_json(root/'configuration.json', dict(source=str(SOURCE), dataset_sha256=source_hash,
        seed=5, n_train=30, n_test=151, models=list(MODELS),
        generating_parameters=dict(l=1., omega=1., mu=0., v_0=.2, w=2., k=.05, sigma_v0=1., sigma_w=.15, W=float(data['W'])),
        priors=dict(log_kernel_sd=2., log_kernel_centers='log([.5*s2,.5*s2,.1*s2,1,.25]); s2=training variance',
                    lambda_normal_mean=1., lambda_normal_sd=2., log_omega_normal_mean=0., log_omega_normal_sd=1.,
                    log_nu_minus_two_normal_mean=float(np.log(3.)), log_nu_minus_two_normal_sd=1.),
        map_measure='d log(theta) d lambda d log(omega), or d log(theta) d log(nu-2); W integrated',
        prediction='Both methods integrate W and latent GP uncertainty; full Bayes additionally integrates hyperparameters',
        prior_note='Kernel prior centers are training-dependent empirical settings, matching previous experiments; mixing priors are fixed externally.'))
    (root/'method.txt').write_text(
        'Same saved seed-5 GHP-beta function: 30 training points, 151 held out.\n'
        'GHP beta: y|W ~ N(0, W(K1+kI)+W^2 K_beta).\n'
        'STP + GP mean: y|W ~ N(0, W(K1+kI)+K_mu).\n'
        'Both SE amplitudes, both inverse squared length scales and noise are learned.\n'
        'GHP also learns lambda and omega; STP also learns nu>2.\n'
        'Priors: independent Normal log-kernel priors, SD=2, centers from training variance;\n'
        'lambda~N(1,4), log(omega)~N(0,1), log(nu-2)~N(log(3),1).\n'
        'MAP: multistart maximization of the marginal posterior in these coordinates,\n'
        'integrating W by refined Gauss-Legendre quadrature on t=log(W).\n'
        'Full Bayes: four Metropolis-corrected HMC chains for (theta,W,mixing parameters),\n'
        'with latent GPs analytically integrated out. Effective log amplitudes reduce\n'
        'correlations; their log-coordinate Jacobian is one. p(log W) includes +log W.\n'
        'Proper mixing normalizers are retained. All gradients are analytic except\n'
        'the Bessel order derivative (symmetric five-point derivative, checked numerically).\n'
        'Kernel-component swaps have a full posterior Metropolis correction.\n'
        'MAP prediction integrates W conditional on fitted hyperparameters. Full Bayes\n'
        'prediction averages all retained conditional Gaussians, including rejected states.\n'
        'Neither method fixes beta/mu to a point estimate: their uncertainty is retained.\n'
        'Intervals: pointwise 2.5/97.5 percentiles of the Gaussian mixture.\n'
        'Scores: held-out f and y RMSE, pointwise NLPD, CRPS, coverage and interval width.\n'
        'Test values and generating parameters never enter fitting or prior selection.\n'
        'MAP is coordinate-dependent. Hyperparameter posterior uncertainty may remain\n'
        'large with only one independent W; compare prior/posterior plots.\n'
        'Reference normalizers: https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.geninvgauss.html\n'
        'https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.invgamma.html\n', encoding='utf-8')
    return data


def fit_map(target, folder, starts=16):
    path = folder/'map.json'
    if path.exists(): return json.loads(path.read_text())
    rng = np.random.default_rng(5261+target.hdim)
    initial = np.r_[target.center, target.initial_h()]
    bounds = [(-15., 15.)]*5+([(-15., 15.), (-8., 8.)] if target.hdim == 2 else [(-8., 8.)])
    results, records = [], []
    for i in range(starts):
        guess = initial.copy()
        if i >= 2: guess += rng.normal(0, .85, len(guess))
        if i % 2:
            guess[[0, 1]] = guess[[1, 0]]
            guess[[3, 4]] = guess[[4, 3]]
        def objective(u):
            val, grad = target.marginal(u)
            return -val, -grad
        fit = optimize.minimize(objective, guess, jac=True, method='L-BFGS-B', bounds=bounds,
                                options=dict(maxiter=1200, ftol=1e-12, gtol=2e-6, maxls=35))
        val, grad = target.marginal(fit.x)
        good = bool(fit.success and np.isfinite(val) and np.max(abs(grad)) < 2e-4)
        records.append(dict(start=i, converged=good, log_posterior=float(val),
                            max_gradient=float(max(abs(grad))), message=str(fit.message)))
        if good: results.append((val, fit.x, i))
        print(f'{target.model} MAP start {i+1}/{starts}: posterior={val:.6f}, converged={good}', flush=True)
    if not results: raise RuntimeError('No converged MAP starts')
    value, u, index = max(results, key=lambda r:r[0])
    assert all(lo+1e-3 < v < hi-1e-3 for v,(lo,hi) in zip(u,bounds))
    offsets = np.eye(len(u))*1e-4
    numerical = np.array([(target.marginal(u+d)[0]-target.marginal(u-d)[0])/2e-4 for d in offsets])
    np.testing.assert_allclose(numerical, target.marginal(u)[1], atol=3e-6)
    H = np.column_stack([(target.marginal(u+d)[1]-target.marginal(u-d)[1])/2e-4 for d in offsets])
    eigenvalue = np.linalg.eigvalsh((H+H.T)/2).max()
    if eigenvalue >= 0: raise RuntimeError('MAP is not a strict local maximum')
    refined = target.marginal(u, tol=1e-11, tail=50.)[0]
    np.testing.assert_allclose(refined, value, atol=1e-7, rtol=0)
    t, weights, quad, _ = target.quadrature(u, tol=1e-11, tail=50.)
    W = np.exp(t)
    report = dict(model=target.model, u=u, log_posterior=value, selected_start=index,
                  prior_center=target.center, prior_sd=target.sd, attempts=records,
                  gradient_error=float(max(abs(numerical-target.marginal(u)[1]))),
                  max_hessian_eigenvalue=float(eigenvalue), refinement_change=float(abs(refined-value)),
                  quadrature=quad, W_mean=float(weights@W), W_std=float(np.sqrt(weights@(W-weights@W)**2)))
    write_json(path, report)
    return json.loads(path.read_text())


def diagnostics(target, fit):
    z = fit['samples']
    # Check both computational coordinates and original physical log coordinates.
    q = np.concatenate([physical_coordinates(z[..., :6], target.model), z[..., 6:]], -1)
    d1, d2 = chain_diagnostics(z), chain_diagnostics(q)
    r = dict(max_rhat=float(max(d1['rhat'].max(), d2['rhat'].max())),
             min_bulk_ess=float(min(d1['ess_bulk'].min(), d2['ess_bulk'].min())),
             min_tail_ess=float(min(d1['ess_tail'].min(), d2['ess_tail'].min())),
             divergences=int(fit['divergent'].sum()), acceptance=float(fit['acceptance'].mean()),
             swap_acceptance=float(fit['swap_accepted'].mean()), effective=d1, physical=d2)
    r['diagnostics_ok'] = r['max_rhat'] < 1.01 and min(r['min_bulk_ess'], r['min_tail_ess']) >= 400 and r['divergences'] == 0
    return r


def fit_bayes(target, folder, map_fit, *, warmup=1500, draws=3000):
    report_path = folder/'sampling_report.json'
    if report_path.exists():
        report = json.loads(report_path.read_text())
        if report['diagnostics_ok']:
            with np.load(folder/report['selected_samples']) as d: z = d['samples'].copy()
            return z, report
    u = np.asarray(map_fit['u'])
    t, weights, _, _ = target.quadrature(u)
    base = np.r_[effective_coordinates(np.r_[u[:5], weights@t], target.model), u[5:]]
    rng = np.random.default_rng(7251+target.hdim)
    modes = []
    for i in range(4):
        guess = base+rng.normal(0, .6, target.dimension)
        if i%2:
            guess[[0, 1]] = guess[[1, 0]]
            guess[[3, 4]] = guess[[4, 3]]
        def objective(z):
            v, g = target.joint(z)
            return -v, -g
        result = optimize.minimize(objective, guess, jac=True, method='L-BFGS-B', options=dict(maxiter=600, ftol=1e-11))
        modes.append(result.x if np.isfinite(target.joint(result.x)[0]) else guess)
    starts = np.stack(modes)+rng.normal(0, .4, (4,target.dimension))
    for attempt in range(4):
        name = f'attempt_{attempt+1}_samples.npz'
        existing = folder/name
        if existing.exists():
            with np.load(existing) as d: fit = {k: d[k].copy() for k in d.files}
        else:
            # Long chains do not cure integration errors. Later retries mainly
            # reduce step sizes, retaining enough draws to assess convergence.
            settings = dict(warmup=warmup*(1 if attempt == 0 else 2),
                            draws=draws*(1 if attempt == 0 else 2),
                            seed=20261022+target.hdim+100*attempt,
                            target_accept=(.95,.97,.999,.9997)[attempt],
                            leapfrog_steps=(15,40) if attempt < 2 else (25,60),
                            step_size=.04, swap_pairs=[(0,1),(3,4)])
            print(f'{target.model} HMC attempt {attempt+1}: {settings}', flush=True)
            begun = time.perf_counter()
            fit = sample_hmc(target.joint, starts, verbose=True, **settings)
            fit['elapsed_seconds'] = np.array(time.perf_counter()-begun)
            np.savez_compressed(existing, **{k:v for k,v in fit.items() if isinstance(v,np.ndarray)})
            write_json(folder/f'attempt_{attempt+1}_settings.json', settings)
        report = dict(model=target.model, **diagnostics(target, fit), selected_samples=name,
                      chains=fit['samples'].shape[0], draws_per_chain=fit['samples'].shape[1],
                      elapsed_seconds=float(fit['elapsed_seconds']))
        write_json(report_path, report)
        print(json.dumps({k:report[k] for k in ['model','max_rhat','min_bulk_ess','min_tail_ess','divergences','diagnostics_ok']}), flush=True)
        if report['diagnostics_ok']: return fit['samples'], report
        starts = fit['samples'][:, -1].copy()
    raise RuntimeError('HMC did not pass diagnostics; all attempts saved, predictions not presented as converged')


def prediction(target, data, z, weights, folder, method):
    dest = folder/f'{method}_prediction.npz'
    report_path = folder/f'{method}_scores.json'
    fingerprint = hashlib.sha256(np.ascontiguousarray(z).tobytes()+np.ascontiguousarray(weights).tobytes()).hexdigest()
    if dest.exists() and report_path.exists():
        old = json.loads(report_path.read_text())
        if old['states_sha256'] == fingerprint: return old
    components = target.predict_components(z, data['x'])
    test = data['test_ids']
    result, scores, checks = {}, dict(model=target.model, method=method), {}
    for name in ['f','y','native']:
        means = components['component_native_means'] if name == 'native' else components['component_means']
        variances = components[f'component_{name}_variances']
        mean = weights@means
        within = weights@variances
        between = weights@(means-mean)**2
        ci = _mixture_intervals(means, variances, weights, .95)
        result.update({f'{name}_mean':mean, f'{name}_var':within+between,
                       f'{name}_within_var':within, f'{name}_between_var':between, f'{name}_ci':ci})
        # Quantile check independent of the interval routine's bisection state.
        cdf = np.array([weights@special.ndtr((ci[:,j]-means)/np.sqrt(variances)) for j in range(2)])
        np.testing.assert_allclose(cdf, np.broadcast_to(np.array([.025,.975])[:,None], cdf.shape), atol=2e-10)
        if name == 'native': continue
        truth = data[name][test]
        m, v = means[:,test], variances[:,test]
        logpdf = special.logsumexp(np.log(weights)[:,None]+stats.norm.logpdf(truth,m,np.sqrt(v)),axis=0)
        crps, crps_check = mixture_crps(truth, m, np.sqrt(v), weights, atol=2e-7)
        for j in [0, len(test)//2, len(test)-1]:
            reference, error = independent_crps(truth[j],m[:,j],np.sqrt(v[:,j]),weights)
            np.testing.assert_allclose(crps[j],reference,atol=5e-7,rtol=0)
        result.update({f'{name}_logpdf_test':logpdf,f'{name}_crps_test':crps})
        scores.update({f'{name}_rmse':float(np.sqrt(np.mean((mean[test]-truth)**2))),
                       f'{name}_nlpd':float(-logpdf.mean()),f'{name}_crps':float(crps.mean()),
                       f'{name}_coverage':float(np.mean((truth>=ci[test,0])&(truth<=ci[test,1]))),
                       f'{name}_interval_width':float(np.mean(ci[test,1]-ci[test,0]))})
        checks[name] = crps_check
    np.testing.assert_allclose(result['f_mean'],result['y_mean'])
    np.savez_compressed(dest, **result)
    report = dict(scores=scores, states_sha256=fingerprint, components=len(z), validation=checks)
    write_json(report_path, report)
    print(f'{target.model} {method}: {scores}',flush=True)
    return report


def parameter_table(target, map_fit, samples, folder):
    values = target.physical_samples(samples).reshape(-1,target.dimension)
    u = np.asarray(map_fit['u'])
    map_parameters = list(np.exp(u[:5]))+[map_fit['W_mean']]
    map_parameters += [u[5],np.exp(u[6])] if target.model=='beta_gp' else [2+np.exp(u[5])]
    rows = []
    for i,name in enumerate(target.parameter_names):
        lo,median,hi = np.quantile(values[:,i],[.025,.5,.975])
        rows.append(dict(parameter=name,map_estimate=map_parameters[i],posterior_mean=float(values[:,i].mean()),
                         posterior_sd=float(values[:,i].std()),posterior_median=median,lower_95=lo,upper_95=hi,
                         map_note='conditional posterior mean of W' if name=='W' else 'marginal hyperparameter MAP'))
    pd.DataFrame(rows).to_csv(folder/'parameters.csv',index=False)


def summarize(root, data):
    rows=[]
    fig,axes=plt.subplots(2,2,figsize=(12,8),sharex=True,constrained_layout=True)
    for col,model in enumerate(MODELS):
        folder=root/model
        for row,method in enumerate(['map','full_bayes']):
            report=json.loads((folder/f'{method}_scores.json').read_text())
            scores=report['scores'];rows.append(scores)
            with np.load(folder/f'{method}_prediction.npz') as p:
                ax=axes[row,col];x=data['x'];train=data['train_ids']
                ax.fill_between(x,*p['y_ci'].T,color='steelblue',alpha=.12,label='95% predictive y')
                ax.fill_between(x,*p['f_ci'].T,color='steelblue',alpha=.28,label='95% credible f')
                ax.plot(x,data['f'],'k--',lw=1,label='True f')
                ax.plot(x,p['f_mean'],color='steelblue',label='Predicted mean')
                ax.scatter(x[train],data['y'][train],s=16,c='black',label='Training y',zorder=3)
                ax.set(title=LABELS[model]+(' — MAP' if method=='map' else ' — Full Bayes')+f'\nf RMSE={scores["f_rmse"]:.4f}, NLPD={scores["f_nlpd"]:.4f}',xlabel='x',ylabel='f / y')
                ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8)
    fig.savefig(root/'regression.pdf');fig.savefig(root/'regression.png',dpi=150);plt.close(fig)
    pd.DataFrame(rows).to_csv(root/'comparison.csv',index=False)
    lines=[r'\begin{table}[htbp]',r'\centering',r'\small',
           r'\caption{Seed-5 single-function regression: 30 training and 151 held-out points. GHP learns $\lambda,\omega$; STP learns $\nu$. MAP integrates $W$ and latent GPs; full Bayes also integrates hyperparameter uncertainty.}',
           r'\label{tab:learned_mixing_seed5}',r'\begin{tabular}{llrrrrrr}',r'\toprule',
           r' & & \multicolumn{3}{c}{Latent $f$} & \multicolumn{3}{c}{Observation $y$} \\',
           r'\cmidrule(lr){3-5}\cmidrule(lr){6-8}',r'Model & Inference & RMSE & NLPD & CRPS & RMSE & NLPD & CRPS \\',r'\midrule']
    for r in rows:
        method='MAP' if r['method']=='map' else 'Full Bayes'
        lines.append(LABELS[r['model']]+' & '+method+' & '+' & '.join(f'{r[k]:.5f}' for k in ['f_rmse','f_nlpd','f_crps','y_rmse','y_nlpd','y_crps'])+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    (root/'comparison.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    parameters=pd.concat([pd.read_csv(root/m/'parameters.csv').assign(model=m) for m in MODELS],ignore_index=True)
    parameters.to_csv(root/'parameters.csv',index=False)
    mixing_rows=parameters[parameters['parameter'].isin(['lambda','omega','nu'])]
    lines=[r'\begin{table}[htbp]',r'\centering',
           r'\caption{Learned mixing parameters. Full-Bayes summaries are posterior mean $\pm$ SD and equal-tail 95\% credible intervals. GHP generating values were $\lambda=\omega=1$; STP is a competing model.}',
           r'\begin{tabular}{llrrr}',r'\toprule',r'Model & Parameter & MAP & Full Bayes & 95\% interval \\',r'\midrule']
    for _,r in mixing_rows.iterrows():
        symbol='\\lambda' if r['parameter']=='lambda' else '\\'+r['parameter']
        lines.append(LABELS[r['model']]+f' & ${symbol}$ & {r["map_estimate"]:.3f} & '
                     +f'${r["posterior_mean"]:.3f} \\pm {r["posterior_sd"]:.3f}$ & '
                     +f'$[{r["lower_95"]:.3f}, {r["upper_95"]:.3f}]$'+r' \\')
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    (root/'mixing_parameters.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(1,2,figsize=(11,3.5),sharey=True,constrained_layout=True)
    for ax,method in zip(axes,['map','full_bayes']):
        with np.load(root/'beta_gp'/f'{method}_prediction.npz') as p:
            ax.plot(data['x'],data['beta'],'k--',label='True beta')
            ax.plot(data['x'],p['native_mean'],color='steelblue',label='Posterior mean')
            ax.fill_between(data['x'],*p['native_ci'].T,color='steelblue',alpha=.25,label='Pointwise 95% interval')
        ax.set(title='MAP hyperparameters' if method=='map' else 'Full Bayes',xlabel='x',ylabel=r'$\beta(x)$')
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.savefig(root/'beta_function.pdf');plt.close(fig)
    # New mixing-parameter posterior versus its proper prior.
    fig,axes=plt.subplots(1,3,figsize=(13,3.5),constrained_layout=True)
    panels=[('beta_gp',6,'lambda'),('beta_gp',7,'omega'),('tp_mu_gp',6,'nu')]
    for ax,(model,index,name) in zip(axes,panels):
        folder=root/model;report=json.loads((folder/'sampling_report.json').read_text())
        with np.load(folder/report['selected_samples']) as d: z=d['samples'].copy()
        target=LearnedMixingTarget(data['y'][data['train_ids']],data['x'][data['train_ids']],model)
        values=target.physical_samples(z)[...,index].ravel()
        prior=stats.norm(1,2) if name=='lambda' else stats.lognorm(s=1,scale=1 if name=='omega' else 3,loc=0 if name=='omega' else 2)
        lo=min(np.quantile(values,.002),prior.ppf(.002));hi=max(np.quantile(values,.995),prior.ppf(.995))
        grid=np.linspace(lo,hi,400)
        ax.hist(values,bins=70,range=(lo,hi),density=True,color='steelblue',alpha=.6,label='Posterior')
        ax.plot(grid,prior.pdf(grid),'k--',label='Prior')
        table=pd.read_csv(folder/'parameters.csv').set_index('parameter')
        ax.axvline(table.loc[name,'map_estimate'],c='darkred',label='MAP')
        if name!='nu': ax.axvline(1,c='gray',ls=':',label='Generating value')
        ax.set(xlabel='$'+('\\lambda' if name=='lambda' else '\\'+name)+'$',ylabel='Density')
        ax.legend(fontsize=8)
    fig.savefig(root/'mixing_posteriors.pdf');fig.savefig(root/'mixing_posteriors.png',dpi=150);plt.close(fig)
    for model in MODELS:
        folder=root/model;report=json.loads((folder/'sampling_report.json').read_text())
        with np.load(folder/report['selected_samples']) as d: z=d['samples'].copy()
        target=LearnedMixingTarget(data['y'][data['train_ids']],data['x'][data['train_ids']],model)
        v=target.physical_samples(z)
        fig,axes=plt.subplots(target.dimension,1,figsize=(11,2*target.dimension),constrained_layout=True)
        for i,ax in enumerate(axes):
            for chain in range(len(v)): ax.plot(v[chain,:,i],lw=.4,alpha=.65,label=f'Chain {chain+1}')
            ax.set_ylabel(target.parameter_names[i])
        axes[0].legend(ncol=4);axes[-1].set_xlabel('Retained draw')
        fig.savefig(folder/'traces.png',dpi=100);plt.close(fig)
    old=json.loads(Path('figure/single_function_30_hmc/summary.json').read_text())
    pd.DataFrame([r for r in old['scores'] if r['model'] in MODELS]).to_csv(root/'previous_fixed_mixing_full_bayes.csv',index=False)
    reports=[json.loads((root/m/'sampling_report.json').read_text()) for m in MODELS]
    write_json(root/'validation.json',dict(all_diagnostics_ok=all(r['diagnostics_ok'] for r in reports),
        max_rhat=max(r['max_rhat'] for r in reports),min_bulk_ess=min(r['min_bulk_ess'] for r in reports),
        min_tail_ess=min(r['min_tail_ess'] for r in reports),divergences=sum(r['divergences'] for r in reports),
        exact_saved_dataset=True,test_observations_excluded_from_fitting=True,
        map_gradient_and_curvature_checks=True,map_quadrature_refinement=True,
        mixture_quantiles_checked=True,crps_independently_checked=True))


def run(args):
    root=Path(args.output);data=prepare(root)
    for model in ([args.model] if args.model else MODELS):
        folder=root/model;folder.mkdir(exist_ok=True)
        train=data['train_ids']
        target=LearnedMixingTarget(data['y'][train],data['x'][train],model)
        fit=fit_map(target,folder)
        if args.stage=='map': continue
        samples,report=fit_bayes(target,folder,fit,warmup=args.warmup,draws=args.draws)
        parameter_table(target,fit,samples,folder)
        if args.stage=='sample': continue
        u=np.array(fit['u']);t,weights,_,_=target.quadrature(u,tol=1e-11,tail=50.)
        z=np.array([np.r_[effective_coordinates(np.r_[u[:5],ti],model),u[5:]] for ti in t])
        prediction(target,data,z,weights,folder,'map')
        flat=samples.reshape(-1,target.dimension)
        prediction(target,data,flat,np.full(len(flat),1/len(flat)),folder,'full_bayes')
    if all((root/m/'full_bayes_scores.json').exists() for m in MODELS): summarize(root,data)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=MODELS)
    parser.add_argument('--stage',choices=['map','sample','all'],default='all')
    parser.add_argument('--output',default=str(ROOT))
    parser.add_argument('--warmup',type=int,default=1500)
    parser.add_argument('--draws',type=int,default=3000)
    args=parser.parse_args()
    with threadpool_limits(limits=1): run(args)
