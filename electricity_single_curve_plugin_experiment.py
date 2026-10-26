"""Fit all parameters separately on ONE electricity curve's observed readings.

First previous test day, selected by date rather than performance. Reuse its
nested random 15/30 masks; use no other curve for parameter or scale fitting.
Run: python electricity_single_curve_plugin_experiment.py
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import linalg, stats
from threadpoolctl import threadpool_limits

from GH_independent_plugin_map import fit_independent, MODELS, INTERNAL
from GH_tabular_plugin_map import PluginPredictor
from tabular_plugin_experiment import batch_scores
from shared_beta_plugin_experiment import write_json

SOURCE = Path('figure/electricity_mt001_shared_plugin/dataset.npz')
OUTPUT = Path('figure/electricity_mt001_single_curve_plugin')
COUNTS = (15,30)
LABELS = dict(gp=r'GP ($\hat\mu$)',stp=r'STP ($\hat\mu$)',
              ghp_mu=r'GHP ($\hat\mu$)',ghp_beta=r'GHP ($\hat\beta$)')


def prepare():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with np.load(SOURCE) as source:
        row = int(source['test_rows'][0])
        data = dict(x=source['x'].copy(),time_hours=source['time_hours'].copy(),
                    y=source['y'][row].copy(),date=source['dates'][row].copy(),
                    context_order=source['context_orders'][0].copy())
    assert str(data['date'])=='2014-03-06'
    assert len(data['y'])==96
    np.testing.assert_array_equal(np.sort(data['context_order']),np.arange(96))
    path = OUTPUT/'dataset.npz'
    if path.exists():
        with np.load(path) as saved:
            for key in data: np.testing.assert_array_equal(saved[key],data[key])
    else:
        np.savez_compressed(path,**data)
    write_json(OUTPUT/'experiment.json',dict(customer='MT_001',date=str(data['date']),
        source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        selection='First previous test day, selected before fitting or examining scores',
        context_counts=COUNTS,context_seed=44026,points_per_curve=96,
        models=MODELS,GHP_lambda=1.,GHP_omega=1.,STP_nu=5.,starts=24,
        training='Only this curve and its selected observed points; refit kernels/noise/W per context',
        preprocessing='Divide y by SD of this context only; no centering or log transform; fixed x grid in [0,10]',
        prior='Independent Normal log-kernel priors, SD2; center from context only, see case configuration',
        inference='Joint log-coordinate MAP with latent GPs integrated; conditional mean function fixed for prediction',
        prediction='Recompute and integrate W conditional on fixed theta and function; no theta/function uncertainty',
        scoring='Observed y in kW at unrevealed points; latent f unknown; interpolation, not forecasting'))
    (OUTPUT/'method.txt').write_text(
        'ONE CURVE, NO PARAMETER SHARING BETWEEN DAYS\n'
        'Customer MT_001, 2014-03-06: first saved test day, selected by date.\n'
        'Use the earlier nested random 15/30 context masks (seed 44026).\n'
        'Fit GP-mu, STP-mu, GHP-mu and GHP-beta separately at each count.\n'
        'No other day supplies observations, hyperparameters, scaling, or prior centers.\n'
        'Input is the saved deterministic time coordinate [0,10]. Response is not\n'
        'centered; divide by SD of the current observed context (ddof=0).\n'
        'Prior center = log([0.5,max(mean(y_scaled^2),0.5),0.01,1,0.04]);\n'
        'independent log-parameter Normal priors have SD2. For beta subtract\n'
        '2 log(E_prior[W]) from the latent variance log center, as in the earlier\n'
        'electricity experiment, matching the prior on the population mean effect.\n'
        'This context-derived scaling/prior replaces the old multi-day scaling\n'
        'and empirical prior, which would leak information from other days here.\n'
        'Both SE variance amplitudes, both inverse squared length scales and\n'
        'noise are learned. Mixing parameters are fixed: lambda=omega=1, nu=5.\n'
        'During joint MAP fit, integrate both GPs analytically. Fit log theta and\n'
        'log W, including the log-W Jacobian, using 24 starts. GP fixes W=1.\n'
        'At fitted theta,W compute the conditional GP posterior mean function.\n'
        'Freeze theta and this function, then recompute and integrate W under\n'
        'the fixed-function likelihood, as in the earlier plug-in experiments.\n'
        'Hyperparameter and function-estimation uncertainty are omitted.\n'
        'Score held-out observed y only: 81 points for 15 observations, 66 for 30.\n'
        'Also save scores on the common 66 held-out points for both counts.\n'
        'Pointwise equal-tail 95% predictive intervals include observation noise.\n'
        'Fits check gradients and Hessian. Validate Gaussian function conditioning,\n'
        'mixing updates, residual prediction, densities, quantiles and CRPS.\n'
        'A single date and observation mask do not establish a model ranking.\n',encoding='utf-8')
    return data


def validate_prediction(data,ids,fit,predictor,full,scale):
    """Independent dense formulas; do not use predictor kernels or solves."""
    x,y = data['x'][ids],data['y'][ids]/scale
    q=np.asarray(fit['log_parameters']);v,a,k,w,sw=np.exp(q[:5])
    W=1. if fit['model']=='gp' else np.exp(q[5])
    def kernel(left,right,amp,weight):
        return amp*np.exp(-.5*weight*(left[:,None]-right[None,:])**2)
    K=kernel(x,x,v,w);H=kernel(x,x,a,sw);A=K+k*np.eye(len(ids))
    is_beta=fit['model']=='ghp_beta'
    C=W*A+(W**2 if is_beta else 1.)*H
    alpha=linalg.solve(C,y,assume_a='pos')
    fn=(W if is_beta else 1.)*kernel(data['x'],x,a,sw)@alpha
    np.testing.assert_allclose(fn,predictor.function(data['x'][:,None]),atol=1e-9)
    mu=np.zeros(len(data['x'])) if is_beta else fn
    beta=fn if is_beta else np.zeros(len(data['x']))
    B=linalg.solve(A,kernel(x,data['x'],v,w),assume_a='pos').T
    residual=y-mu[ids]
    location=mu+B@residual
    effect=beta-B@beta[ids]
    residual_var=v-np.sum(B*kernel(data['x'],x,v,w),axis=1)
    mix=predictor.mixing_parameters
    if mix['mixing']=='gig':
        p=1-len(ids)/2;chi=1+residual@linalg.solve(A,residual,assume_a='pos')
        psi=1+beta[ids]@linalg.solve(A,beta[ids],assume_a='pos')
        np.testing.assert_allclose([mix['W_lambda'],mix['W_chi'],mix['W_psi']],[p,chi,psi],rtol=1e-9)
        distribution=stats.geninvgauss(p,np.sqrt(chi*psi),scale=np.sqrt(chi/psi))
        np.testing.assert_allclose([mix['W_mean'],mix['W_var']],[distribution.mean(),distribution.var()],rtol=1e-8)
    elif mix['mixing']=='ig':
        shape=(5+len(ids))/2;rate=(3+residual@linalg.solve(A,residual,assume_a='pos'))/2
        np.testing.assert_allclose([mix['W_shape'],mix['W_scale']],[shape,rate],rtol=1e-9)
    np.testing.assert_allclose(full['y_mean'],scale*(location+mix['W_mean']*effect),atol=1e-8)
    np.testing.assert_allclose(full['y_var'],scale**2*(mix['W_mean']*(residual_var+k)+mix['W_var']*effect**2),atol=1e-8)
    return dict(independent_function_conditioning=True,independent_prediction_moments=True,mixing_update=True)


def run_case(data,count,model,*,output=OUTPUT):
    case=Path(output)/f'context_{count}'/model
    case.mkdir(parents=True,exist_ok=True)
    ids=np.sort(data['context_order'][:count])
    held=np.setdiff1d(np.arange(len(data['x'])),ids)
    scale=float(np.std(data['y'][ids]))
    if scale<=0: raise ValueError('Constant observed context')
    y=data['y'][ids]/scale
    center=np.log([.5,max(float(np.mean(y*y)),.5),.01,1.,.04])
    write_json(case/'configuration.json',dict(context_ids=ids,test_ids=held,response_scale=scale,
        population_prior_center=center,scale_source='only observed points of this curve'))
    fitfile=case/'fit.json'
    if fitfile.exists():
        fit=json.loads(fitfile.read_text())
        np.testing.assert_allclose(fit['population_prior_center'],center,atol=1e-12)
    else:
        checkpoint=case/'optimization_starts.json'
        completed=json.loads(checkpoint.read_text()) if checkpoint.exists() else []
        def progress(record,z):
            completed.append(dict(record=record,effective_log_parameters=z))
            write_json(checkpoint,completed)
        fit=fit_independent(y[None,:],data['x'][ids],model,center,
                            progress=progress,completed=list(completed))
        write_json(fitfile,fit)
    predictor=PluginPredictor(y,data['x'][ids,None],fit['log_parameters'],INTERNAL[model])
    full=predictor.predict(data['x'][:,None],y_scale=scale)
    p=predictor.predict(data['x'][held,None],y_scale=scale)
    point,checks=batch_scores(p,data['y'][held],check=True)
    checks.update(validate_prediction(data,ids,fit,predictor,full,scale))
    compact={key:full[key] for key in ('f_mean','f_var','f_ci','y_mean','y_var','y_ci','location','conditional_beta')}
    np.savez_compressed(case/'predictions.npz',**compact,context_ids=ids,test_ids=held,
        function_estimate=scale*predictor.function(data['x'][:,None]),**point)
    def metrics(positions):
        truth=data['y'][held[positions]]
        return dict(y_rmse=float(np.sqrt(np.mean((truth-p['y_mean'][positions])**2))),
            y_nlpd=float(point['nlpd'][positions].mean()),y_crps=float(point['crps'][positions].mean()),
            y_coverage_95=float(point['covered'][positions].mean()),
            y_interval_width=float(np.mean(p['y_ci'][positions,1]-p['y_ci'][positions,0])))
    common=np.setdiff1d(np.arange(len(data['x'])),data['context_order'][:30])
    scores=dict(model=model,n_train=count,n_test=len(held),**metrics(np.arange(len(held))))
    common_scores=dict(model=model,n_train=count,n_test=len(common),**metrics(np.searchsorted(held,common)))
    write_json(case/'report.json',dict(scores=scores,common_test_scores=common_scores,
        W_for_function=predictor.W_map,prediction_W_posterior=predictor.mixing_parameters,
        validation=checks,max_abs_gradient=fit['max_abs_gradient'],
        max_hessian_eigenvalue=fit['max_hessian_eigenvalue']))
    print(f'{count} observations, {model}: RMSE={scores["y_rmse"]:.5f}, NLPD={scores["y_nlpd"]:.5f}, '
          f'coverage={100*scores["y_coverage_95"]:.2f}%',flush=True)
    pars=np.exp(fit['log_parameters'][:5])
    parameter=dict(model=model,n_train=count,v0=pars[0]*scale**2,latent_variance=pars[1]*scale**2,
        k=pars[2]*scale**2,w=pars[3],latent_w=pars[4],W_for_function=predictor.W_map,
        W_prediction_mean=predictor.mixing_parameters['W_mean'],
        W_prediction_std=np.sqrt(predictor.mixing_parameters['W_var']),response_scale=scale)
    return scores,common_scores,parameter


def main():
    data=prepare()
    scores,common,parameters=[],[],[]
    for count in COUNTS:
        for model in MODELS:
            score,same,parameter=run_case(data,count,model)
            scores.append(score);common.append(same);parameters.append(parameter)
    table=pd.DataFrame(scores)
    table.to_csv(OUTPUT/'scores.csv',index=False)
    pd.DataFrame(common).to_csv(OUTPUT/'common_test_scores.csv',index=False)
    pd.DataFrame(parameters).to_csv(OUTPUT/'parameters.csv',index=False)
    lines=[r'\begin{table}[htbp]',r'\centering',r'\small',
        r'\caption{Electricity MT\_001 on 6 March 2014: all kernel/noise parameters and the function estimate are fitted using only this day\textquotesingle s 15 or 30 observed readings. No information is learned from other days. GHP fixes $\lambda=\omega=1$ and STP fixes $\nu=5$. Plug-in functions and MAP hyperparameters are fixed; prediction integrates $W$. Scores concern 81 or 66 held-out observed readings in kW.}',
        r'\label{tab:electricity_single_curve}',r'\begin{tabular}{rlrrrrr}',r'\toprule',
        r'Observed & Model & RMSE & NLPD & CRPS & 95\% coverage (\%) & Width \\',r'\midrule']
    for row in table.itertuples():
        lines.append(f'{row.n_train} & {LABELS[row.model]} & {row.y_rmse:.5f} & {row.y_nlpd:.5f} & {row.y_crps:.5f} & {100*row.y_coverage_95:.2f} & {row.y_interval_width:.4f} '+r'\\')
        if row.model==MODELS[-1] and row.n_train==15:lines.append(r'\midrule')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (OUTPUT/'comparison.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    fig,axes=plt.subplots(4,2,figsize=(13,12),sharex=True,constrained_layout=True)
    for i,model in enumerate(MODELS):
        for j,count in enumerate(COUNTS):
            with np.load(OUTPUT/f'context_{count}'/model/'predictions.npz') as p:
                ax=axes[i,j];t=data['time_hours'];ids=p['context_ids']
                ax.plot(t,data['y'],color='black',lw=1,label='Observed demand (full curve)')
                ax.plot(t,p['y_mean'],color='tab:blue',label='Predictive mean')
                ax.fill_between(t,p['y_ci'][:,0],p['y_ci'][:,1],color='tab:blue',alpha=.2,label='95% predictive interval')
                ax.scatter(t[ids],data['y'][ids],color='tab:red',s=20,zorder=5,label='Training readings')
                ax.set(title=f'{LABELS[model]}, {count} observations',ylabel='Demand (kW)',xlim=(0,24))
                if i==3:ax.set_xlabel('Local time (hours)')
                if i==0 and j==0:ax.legend(fontsize=7)
    fig.suptitle('MT_001, 6 March 2014: fit each model using only this curve')
    fig.savefig(OUTPUT/'regression.pdf');fig.savefig(OUTPUT/'regression.png',dpi=150)
    plt.close(fig)
    write_json(OUTPUT/'validation.json',dict(all_checks_passed=True,n_fits=len(scores),
        selected_date=str(data['date']),training_source='Only observed points of the selected date',
        nested_contexts=True,independent_conditioning_and_mixing_checks=True,
        density_quantile_and_crps_checks=True))
    print(table.to_string(index=False),flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
