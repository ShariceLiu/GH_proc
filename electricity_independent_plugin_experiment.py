"""Independent per-day GP-prior mu/beta on the saved electricity experiment.

python electricity_independent_plugin_experiment.py --stage prepare
python electricity_independent_plugin_experiment.py --model gp
python electricity_independent_plugin_experiment.py --stage summarize
"""
import os
for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[variable]='1'
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

from GH_independent_plugin_map import MODELS, fit_independent, context_function
from GH_fixed_function import fixed_function_prediction
from shared_beta_plugin_experiment import write_json, score_prediction
from tecator_shared_plugin_experiment import original_units

ROOT=Path(__file__).resolve().parent
BASELINE=ROOT/'figure'/'electricity_mt001_shared_plugin'
OUTPUT=ROOT/'figure'/'electricity_mt001_independent_plugin'
COUNTS=(15,30)
METRICS=('y_rmse','y_nlpd','y_crps','y_coverage_95','y_interval_width')
LABELS=dict(gp='GP: separate mu',stp='STP: separate mu',ghp_mu='GHP: separate mu',ghp_beta='GHP: separate beta')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    """Reference the existing dataset; no raw data or dense arrays are copied."""
    with np.load(BASELINE/'dataset.npz') as saved:data={key:saved[key] for key in saved.files}
    previous=json.loads((BASELINE/'experiment.json').read_text())
    settings=dict(baseline_directory=BASELINE.relative_to(ROOT).as_posix(),
                  dataset_sha256=digest(BASELINE/'dataset.npz'),baseline_scores_sha256=digest(BASELINE/'scores.csv'),
                  training_dates=previous['training_dates'],gap_dates=previous['gap_dates'],test_dates=previous['test_dates'],
                  context_counts=COUNTS,context_seed=previous['context_seed'],population_prior_center=previous['prior_center'],
                  prior_sd=2.,GHP_lambda=1.,GHP_omega=1.,STP_nu=5.,
                  training='Independent latent GP functions integrated out; joint MAP in log theta and training log W.',
                  prediction='Freeze theta; context-only log W MAP, conditional latent-function mean, then fix function and integrate fresh W.',
                  uncertainty='Plug-in function and hyperparameters; function-estimation and hyperparameter uncertainty are omitted.',
                  preprocessing=previous['preprocessing'],baseline_comparison='Same observations, contexts, preprocessing and marginal GP priors.')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    normalized=json.loads(json.dumps(settings))
    if (OUTPUT/'experiment.json').exists():assert json.loads((OUTPUT/'experiment.json').read_text())==normalized
    else:write_json(OUTPUT/'experiment.json',settings)
    return data,np.asarray(previous['prior_center'])


def fit_model(data,center,model):
    folder=OUTPUT/model;folder.mkdir(exist_ok=True)
    source_hash=digest(BASELINE/'dataset.npz')
    if (folder/'fit.json').exists():
        fit=json.loads((folder/'fit.json').read_text());assert fit['dataset_sha256']==source_hash
        return fit
    checkpoint=folder/'optimization_starts.json'
    completed=json.loads(checkpoint.read_text()) if checkpoint.exists() else []
    def progress(record,z):
        completed.append(dict(record=record,effective_log_parameters=z.tolist()))
        write_json(checkpoint,completed)
        print(f'{model}: start {record["start"]+1}/24, log posterior {record["log_posterior"]:.5f}, '
              f'max gradient {record["max_abs_gradient"]:.3g}',flush=True)
    started=time.perf_counter()
    fit=fit_independent(data['y'][data['train_rows']]/data['common_scale'],data['x'],model,center,
                        progress=progress,completed=completed)
    fit.update(dataset_sha256=source_hash,fitting_seconds_this_invocation=time.perf_counter()-started)
    write_json(folder/'fit.json',fit)
    print(f'{model}: selected start {fit["selected_start"]+1}; MAP {fit["log_posterior"]:.5f}',flush=True)
    return fit


def predict_model(data,model,fit):
    folder=OUTPUT/model;fit_hash=digest(folder/'fit.json')
    theta=np.asarray(fit['log_parameters'][:5]);v,a,k,w,sw=np.exp(theta)
    scale=float(data['common_scale']);kind='beta' if model=='ghp_beta' else 'mu'
    mixing='gp' if model=='gp' else 'ig' if model=='stp' else 'gig'
    rows=[]
    for count in COUNTS:
        case=folder/f'context_{count}';case.mkdir(exist_ok=True)
        for j,row in enumerate(data['test_rows']):
            stem=case/f'day_{data["dates"][row]}'
            if stem.with_suffix('.json').exists() and stem.with_suffix('.npz').exists():
                record=json.loads(stem.with_suffix('.json').read_text())
                if record['fit_sha256']==fit_hash:rows.append(record['scores']);continue
            ids=np.sort(data['context_orders'][j,:count]);held=np.setdiff1d(np.arange(96),ids)
            function,mode=context_function(data['y'][row,ids]/scale,data['x'][ids],data['x'],theta,model)
            kwargs={f'{kind}_train':function[ids],f'{kind}_pred':function}
            pred=fixed_function_prediction(data['y'][row,ids]/scale,data['x'][ids],data['x'],
                                          v0=v,k=k,w=w,mixing=mixing,**kwargs)
            pred=original_units(pred,scale)
            scores,point,validation=score_prediction(pred,None,data['y'][row],held,check=j==0)
            scores.update(model=model,context_count=count,date=str(data['dates'][row]),n_test=len(held),
                          W_for_function=mode['W_for_function'],W_mean=pred['W_mean'],W_std=float(np.sqrt(pred['W_var'])))
            compact={key:value for key,value in pred.items() if not key.endswith('_cov')}
            compact.update(f_scale_var=np.diag(pred['f_scale_cov']),y_scale_var=np.diag(pred['y_scale_cov']))
            np.savez_compressed(stem.with_suffix('.npz'),**compact,**point,context_ids=ids,test_ids=held,
                                function_estimate_scaled=function,function_estimate_original=scale*function)
            write_json(stem.with_suffix('.json'),dict(scores=scores,validation=validation,context_mode=mode,fit_sha256=fit_hash))
            rows.append(scores)
        print(f'{model}: finished {count}-point contexts',flush=True)
    pd.DataFrame(rows).to_csv(folder/'per_day_scores.csv',index=False)


def aggregate(curves):
    rows=[]
    for (count,model),frame in curves.groupby(['context_count','model'],sort=False):
        assert len(frame)==18
        row=dict(context_count=count,model=model,n_days=len(frame),y_rmse=float(np.sqrt(np.mean(frame.y_rmse**2))))
        row.update({key:float(frame[key].mean()) for key in METRICS[1:]})
        for key in METRICS:
            row[key+'_day_mean']=float(frame[key].mean());row[key+'_day_sd']=float(frame[key].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def summarize(data):
    parameters=[]
    for model in MODELS:
        fit=json.loads((OUTPUT/model/'fit.json').read_text())
        v,a,k,w,sw=np.exp(fit['log_parameters'][:5]);scale=float(data['common_scale'])
        parameters.append(dict(model=model,residual_variance_original=v*scale**2,
            latent_variance_original=a*scale**2,noise_coefficient_original=k*scale**2,
            residual_inverse_length_squared=w,latent_inverse_length_squared=sw,
            input_coordinate='time rescaled to [0,10]',latent_function='beta' if model=='ghp_beta' else 'mu',
            training_W_mode_mean=float(np.mean(fit['W_training_mode'])),
            training_W_mode_min=float(np.min(fit['W_training_mode'])),
            training_W_mode_max=float(np.max(fit['W_training_mode'])),
            max_abs_gradient=fit['max_abs_gradient'],max_hessian_eigenvalue=fit['max_hessian_eigenvalue']))
    pd.DataFrame(parameters).to_csv(OUTPUT/'parameters.csv',index=False)
    curves=pd.concat([pd.read_csv(OUTPUT/model/'per_day_scores.csv') for model in MODELS],ignore_index=True)
    curves.to_csv(OUTPUT/'per_day_scores.csv',index=False)
    summary=aggregate(curves);summary.to_csv(OUTPUT/'scores.csv',index=False)
    baseline=pd.read_csv(BASELINE/'scores.csv')
    comparison=summary.merge(baseline,on=['context_count','model'],suffixes=('_independent','_shared'),validate='one_to_one')
    for key in METRICS:comparison[key+'_change']=comparison[key+'_independent']-comparison[key+'_shared']
    comparison.to_csv(OUTPUT/'comparison_with_shared.csv',index=False)
    compare_shared(data,curves,summary,baseline)
    common=[]
    for model in MODELS:
        for count in COUNTS:
            for j,row in enumerate(data['test_rows']):
                with np.load(OUTPUT/model/f'context_{count}'/f'day_{data["dates"][row]}.npz') as p:
                    ids=np.setdiff1d(np.arange(96),data['context_orders'][j,:30]);pos=np.searchsorted(p['test_ids'],ids)
                    truth=data['y'][row,ids];ci=p['y_ci'][ids]
                    common.append(dict(model=model,context_count=count,date=str(data['dates'][row]),n_test=len(ids),
                        y_rmse=float(np.sqrt(np.mean((truth-p['y_mean'][ids])**2))),y_nlpd=float(p['y_nlpd'][pos].mean()),
                        y_crps=float(p['y_crps'][pos].mean()),y_coverage_95=float(np.mean((truth>=ci[:,0])&(truth<=ci[:,1]))),
                        y_interval_width=float(np.mean(ci[:,1]-ci[:,0]))))
    common=pd.DataFrame(common);common.to_csv(OUTPUT/'common_test_per_day_scores.csv',index=False)
    aggregate(common).to_csv(OUTPUT/'common_test_scores.csv',index=False)
    print(summary[['context_count','model',*METRICS]].to_string(index=False),flush=True)
    names=dict(gp=r'GP ($\hat\mu_j$)',stp=r'STP ($\hat\mu_j$)',ghp_mu=r'GHP ($\hat\mu_j$)',ghp_beta=r'GHP ($\hat\beta_j$)')
    lines=[r'\begin{table}[htbp]',r'\centering',
           r'\caption{Electricity customer MT\_001: independent per-day plug-in functions; shared kernel/noise parameters. Forty training days, five gap days and 18 test days. Scores evaluate held-out observed demand (kW). RMSE is pooled; remaining metrics are averaged.}',
           r'\label{tab:electricity_mt001_independent}',r'\begin{tabular}{rlrrrr}',r'\toprule',
           r'Observed points & Model & RMSE & NLPD & CRPS & 95\% coverage (\%) \\',r'\midrule']
    for count in COUNTS:
        for model in MODELS:
            row=summary[(summary.context_count==count)&(summary.model==model)].iloc[0]
            lines.append(f'{count} & '+names[model]+f' & {row.y_rmse:.5f} & {row.y_nlpd:.5f} & {row.y_crps:.5f} & {100*row.y_coverage_95:.2f}'+r' \\')
        if count==15:lines.append(r'\midrule')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (OUTPUT/'comparison.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    t=data['time_hours']
    with PdfPages(OUTPUT/'regression.pdf') as pdf:
        for count in COUNTS:
            for row in data['test_rows']:
                fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,sharey=True)
                for ax,model in zip(axes.flat,MODELS):
                    relative=Path(model)/f'context_{count}'/f'day_{data["dates"][row]}.npz'
                    with np.load(OUTPUT/relative) as p,np.load(BASELINE/relative) as b:
                        ax.fill_between(t,*b['y_ci'].T,color='C1',alpha=.12,label='Shared: 95% observation interval')
                        ax.fill_between(t,*p['y_ci'].T,alpha=.25,label='Separate: 95% observation interval')
                        ax.plot(t,p['y_mean'],label='Separate: prediction')
                        ax.plot(t,b['y_mean'],color='C1',label='Shared: prediction')
                        ax.plot(t,data['y'][row],'k--',lw=1,label='Observed demand')
                        ids=p['context_ids'];ax.scatter(t[ids],data['y'][row,ids],s=14,c='black',label='Revealed points')
                        ax.set(title=LABELS[model],xlabel='Local time (hours)',ylabel='Demand (kW)')
                axes[0,0].legend(fontsize=7)
                fig.suptitle(f'Day {data["dates"][row]}, {count} observed points; plug-in functions')
                fig.tight_layout();pdf.savefig(fig)
                if row==data['test_rows'][0]:fig.savefig(OUTPUT/f'regression_context{count}.png',dpi=150)
                plt.close(fig)
    with PdfPages(OUTPUT/'estimated_functions.pdf') as pdf:
        for count in COUNTS:
            fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
            for ax,model in zip(axes.flat,MODELS):
                for row in data['test_rows']:
                    with np.load(OUTPUT/model/f'context_{count}'/f'day_{data["dates"][row]}.npz') as p:
                        ax.plot(t,p['function_estimate_original'],label=f'Day {data["dates"][row]}')
                shared=json.loads((BASELINE/model/'fit.json').read_text())
                ax.plot(t,shared['function_original'],'k--',label='Shared estimate')
                ax.set(title=LABELS[model],xlabel='Local time (hours)',ylabel='beta (kW)' if model=='ghp_beta' else 'mu (kW)')
            axes[0,0].legend(fontsize=6,ncol=3);fig.suptitle(f'Estimated functions from {count} context points')
            fig.tight_layout();pdf.savefig(fig);plt.close(fig)
    (OUTPUT/'method.txt').write_text(METHOD,encoding='utf-8')


def compare_shared(data,curves,summary,baseline):
    """Matched scores, paired differences, and a descriptive high-demand split."""
    old=pd.read_csv(BASELINE/'per_day_scores.csv')
    paired=curves.merge(old,on=['context_count','model','date'],suffixes=('_individual','_shared'),validate='one_to_one')
    assert len(paired)==144
    for metric in METRICS:paired[metric+'_change']=paired[metric+'_individual']-paired[metric+'_shared']
    paired['mse_change']=paired.y_rmse_individual**2-paired.y_rmse_shared**2
    paired.to_csv(OUTPUT/'paired_per_day_comparison.csv',index=False)
    rows=[]
    for (count,model),frame in paired.groupby(['context_count','model']):
        record=dict(context_count=count,model=model,n_days=len(frame))
        for metric in METRICS:
            values=frame[metric+'_change'];record[metric+'_change_mean']=float(values.mean())
            record[metric+'_change_sd']=float(values.std(ddof=1))
            record[metric+'_days_lower']=int((values<0).sum())
        rows.append(record)
    pd.DataFrame(rows).to_csv(OUTPUT/'paired_comparison.csv',index=False)
    combined=pd.concat([baseline.assign(function='Shared'),summary.assign(function='Individual')],ignore_index=True)
    combined.to_csv(OUTPUT/'combined_scores.csv',index=False)
    lines=[r'\begin{table}[htbp]',r'\centering',
        r'\caption{Electricity MT\_001: shared versus individual plug-in functions on the same 40 training and 18 test days, with a five-day gap. Scores use identical held-out observed demand (kW). Kernel/noise parameters are refitted for each model.}',
        r'\label{tab:electricity_shared_individual}',r'\begin{tabular}{rllrrrr}',r'\toprule',
        r'Observed & Model & Function & RMSE & NLPD & CRPS & 95\% coverage (\%) \\',r'\midrule']
    names=dict(gp=r'GP $\mu$',stp=r'STP $\mu$',ghp_mu=r'GHP $\mu$',ghp_beta=r'GHP $\beta$')
    for count in COUNTS:
        for model in MODELS:
            for kind in ('Shared','Individual'):
                row=combined[(combined.context_count==count)&(combined.model==model)&(combined.function==kind)].iloc[0]
                lines.append(f'{count} & '+names[model]+f' & {kind} & {row.y_rmse:.5f} & {row.y_nlpd:.5f} & {row.y_crps:.5f} & {100*row.y_coverage_95:.2f}'+r' \\')
        if count==15:lines.append(r'\midrule')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (OUTPUT/'comparison_with_shared.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    threshold=float(data['y'][data['train_rows']].mean(1).max())
    high_dates=data['dates'][data['y'].mean(1)>threshold]
    combined_days=pd.concat([old.assign(function='Shared'),curves.assign(function='Individual')],ignore_index=True)
    combined_days['high_demand']=combined_days.date.isin(high_dates);subgroups=[]
    for (count,model,kind,high),frame in combined_days.groupby(['context_count','model','function','high_demand']):
        subgroups.append(dict(context_count=count,model=model,function=kind,high_demand=bool(high),n_days=len(frame),
            threshold_daily_mean_kw=threshold,y_rmse=float(np.sqrt(np.mean(frame.y_rmse**2))),
            **{key:float(frame[key].mean()) for key in METRICS[1:]}))
    pd.DataFrame(subgroups).to_csv(OUTPUT/'descriptive_day_type_scores.csv',index=False)


METHOD='''INDEPENDENT PER-DAY FUNCTIONS: ELECTRICITY MT_001

Same data as ../electricity_mt001_shared_plugin/dataset.npz (referenced, not copied).
2014 Q1 weekdays: 40 training days, five gap days, 18 test days, 96 points per day.
Identical nested random 15/30 contexts, training-only common scale, no centering,
no per-day normalization. Original-unit scores use kW. Shared baseline untouched.

For GP, STP and GHP-mu each day has its own mu_j~GP(0,Sigma_mu).
For GHP-beta, mu=0 and each day has its own beta_j~GP(0,Sigma_beta).
All latent functions and residual GPs are independent across days; kernel/noise
hyperparameters theta are shared and refitted using only the 40 training days.
There is no fixed common mean curve. GHP fixes GIG(lambda=1,chi=psi=1), STP fixes
IG(shape=2.5,scale=1.5), GP fixes W=1. The STP model adds an unscaled GP-prior
mean; it is not a pure Student-t process.

Use the shared baseline's log hyperparameter prior centers and SD=2. Its common
prior was on h=mu or h=E_prior[W]*beta, so the native beta log variance center is
shifted by -2*log(E_prior[W]). This matches the marginal prior of each function.
Covariance after integrating the function is C_j=W_j*(K+kI)+W_j^r*Sigma,
r=0 for mu and r=2 for beta. Joint log(theta),log(W_train) MAP includes W
Jacobians; the effective log-amplitude transformation has determinant one.
Use 24 multistart initializations, including both component assignments.
Select the best stationary solution found; check gradients and Hessian.
This is a local optimization search, not a global-optimum guarantee.

TEST-DAY ESTIMATION
Freeze fitted theta. Only the current day's 15/30 observed readings enter:
1. Integrate out that day's mu_j/beta_j and find the log-W posterior mode.
2. At that W_hat, compute its native conditional GP posterior mean:
   mu_hat(x*) = Sigma_*c C_c^-1 y_c,
   beta_hat(x*) = W_hat Sigma_*c C_c^-1 y_c.
3. Fix that estimated function. Recompute the W posterior using the original
   mixing prior and the fixed-function likelihood, then integrate W for prediction.
   Do not reuse the collapsed W posterior as a prior.
The W used to obtain the function and the subsequent fixed-function W posterior
have different targets. This is a two-stage plug-in procedure; it omits function
estimation and hyperparameter uncertainty. Context data are reused for estimation,
so these plug-in intervals can be too narrow.

For fixed functions, A=K_cc+kI and residual=y_c-mu_c.
GIG posterior: lambda'=1-n/2, chi'=1+residual'A^-1 residual,
psi'=1+beta_c'A^-1 beta_c. IG posterior: ((5+n)/2,(3+residual'A^-1 residual)/2).
Conditional prediction: N(m+W*b,W*S),
m=mu_*+K_*c A^-1 residual, b=beta_*-K_*c A^-1 beta_c,
S=K_**-K_*c A^-1 K_c*. Observation noise adds W*k to the diagonal.
Integrate W for the GH/Student-t/Normal pointwise marginals and 95% equal-tail bands.

Only observed y has ground truth. RMSE is pooled over held-out points, other
metrics averaged over days. Primary 15/30 scores use 81/66 held-out points;
common_test_scores.csv uses the same 66 points for both contexts.
comparison_with_shared.csv/.tex compare both methods on identical targets.
Per-day scores and paired differences are saved; day-to-day SD is not uncertainty
across independent splits. Adjacent days may be dependent.
Descriptive high-demand subgroup uses a threshold equal to the largest training
daily mean, only for post-fit scoring; complete test values never enter fitting.
All likelihoods allow negative demand; intervals are not clipped.
Compact predictions omit dense covariance matrices. The baseline dataset remains
required to reproduce the experiment. This run makes no new network downloads.
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=MODELS)
    parser.add_argument('--stage',choices=['prepare','all','summarize'],default='all')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        data,center=prepare()
        if args.stage=='prepare':return
        if args.stage=='all':
            for model in ([args.model] if args.model else MODELS):
                fit=fit_model(data,center,model);predict_model(data,model,fit)
        if not args.model or args.stage=='summarize':summarize(data)


if __name__=='__main__':main()
