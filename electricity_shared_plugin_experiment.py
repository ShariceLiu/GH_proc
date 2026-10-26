"""Shared-function MAP comparison on one customer's weekday electricity curves.

python electricity_shared_plugin_experiment.py --stage prepare
python electricity_shared_plugin_experiment.py --model gp
python electricity_shared_plugin_experiment.py --stage summarize
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

from load_electricity_customer import load_data, daily_curves, CACHE, CUSTOMER
from GH_shared_function_map import MODELS, fit_shared_function, gig_mean
from GH_fixed_function import fixed_function_prediction
from shared_beta_plugin_experiment import write_json, score_prediction, LABELS
from tecator_shared_plugin_experiment import original_units

ROOT=Path(__file__).resolve().parent
OUTPUT=ROOT/'figure'/'electricity_mt001_shared_plugin'
COUNTS=(15,30)
CONTEXT_SEED=44026
METRICS=('y_rmse','y_nlpd','y_crps','y_coverage_95','y_interval_width')


def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def prepare():
    source=load_data();dates,curves,quality=daily_curves(source)
    date_strings=dates.astype(str)
    eligible=quality[quality.date.between('2014-01-02','2014-03-31')&quality.weekday.lt(5)]
    # Pre-specified first quarter, weekdays, excluding New Year's Day.
    # Fail instead of silently selecting a different customer/date window by scores.
    if not eligible.complete.all() or eligible.all_zero.any() or eligible.dst_adjusted.any():
        raise ValueError('Pre-specified customer/window fails quality rules; inspect daily_quality.csv')
    rows=np.flatnonzero(np.isin(date_strings,eligible.date))
    dates=date_strings[rows];y=curves[rows]
    if len(y)!=63:raise ValueError(f'Expected 63 eligible weekdays, got {len(y)}')
    train=np.arange(40);gap=np.arange(40,45);test=np.arange(45,63)
    hours=np.arange(96)/4.;x=hours/hours[-1]*10.
    scale=float(np.std(y[train]-y[train].mean(0)))
    if scale<=0:raise ValueError('No training-curve variation')
    Y=y[train]/scale;center=np.log([.5,max(float(np.mean(Y.mean(0)**2)),.5),.01,1.,.04])
    day_ids=dates.astype('datetime64[D]').astype(int)
    orders=np.stack([np.random.default_rng(np.random.SeedSequence([CONTEXT_SEED,int(day_ids[row])])).permutation(96) for row in test])
    data=dict(x=x,time_hours=hours,y=y,dates=dates,day_ids=day_ids,train_rows=train,gap_rows=gap,test_rows=test,
              common_scale=np.array(scale),common_mean=y[train].mean(0),context_orders=orders)
    settings=dict(customer=CUSTOMER,unit='kW',n_train=len(train),n_gap=len(gap),n_test=len(test),points_per_curve=96,
        selection='First source customer; 2014 Q1 weekdays, excluding January 1, chosen before model fitting.',
        calendar_control='Weekdays only within one quarter; not separately adjusted for Monday-Friday or within-quarter trends.',
        timestamps='Group by source local calendar date and recorded times 00:00 to 23:45, without shifting labels.',
        quality_rules='Complete 96-point days; exclude DST-adjusted dates and all-zero days. No clipping or outlier removal.',
        split='First 40 eligible weekdays train, next five gap, last 18 test in source date order.',
        training_dates=dates[train],gap_dates=dates[gap],test_dates=dates[test],
        context_counts=COUNTS,context_seed=CONTEXT_SEED,contexts='Nested random 15/30 points, identical across models.',
        task='Reconstruct each held-out day from scattered revealed readings; not next-day forecasting.',
        preprocessing='Divide by training residual SD about training pointwise mean. No centering, per-day normalization, or log transform.',
        prior_center=center,prior_sd=2.,GHP_lambda=1.,GHP_omega=1.,STP_nu=5.,
        inference='Shared function analytically marginalized for joint log theta/log W training MAP, then conditional GP mean fixed.',
        prediction='Freeze theta and shared function; infer/integrate a fresh W from each test context.',
        source_sha256=digest(CACHE/'series.npz'),scores='Observed electricity demand y, kW; latent f is unknown.')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/'dataset.npz').exists():
        with np.load(OUTPUT/'dataset.npz') as saved:
            for key in data:np.testing.assert_array_equal(saved[key],data[key])
    else:np.savez_compressed(OUTPUT/'dataset.npz',**data)
    write_json(OUTPUT/'experiment.json',settings)
    quality['role']='excluded'
    for role in ('train','gap','test'):quality.loc[quality.date.isin(data['dates'][data[role+'_rows']]),'role']=role
    quality.to_csv(OUTPUT/'daily_quality_and_split.csv',index=False)
    return data,center


def exploration(data):
    fig,axes=plt.subplots(2,2,figsize=(12,7));hours=data['time_hours'];mean=data['common_mean']
    for role,color in [('train','C0'),('gap','C1'),('test','C3')]:
        rows=data[role+'_rows']
        for j,row in enumerate(rows):axes[0,0].plot(hours,data['y'][row],color=color,alpha=.25,label=role if j==0 else None)
        axes[0,1].plot(pd.to_datetime(data['dates'][rows]),data['y'][rows].mean(1),'o-',color=color,label=role)
    axes[0,0].set(xlabel='Local time (hours)',ylabel='Demand (kW)',title='Weekday daily curves');axes[0,0].legend()
    axes[0,1].set(ylabel='Mean demand (kW)',title='Chronological split');axes[0,1].tick_params(axis='x',rotation=25)
    for row in data['train_rows']:axes[1,0].plot(hours,data['y'][row]-mean,alpha=.25)
    axes[1,0].set(xlabel='Local time (hours)',ylabel='Deviation (kW)',title='Training deviations from mean curve')
    axes[1,1].plot(hours,mean,label='Training mean')
    axes[1,1].fill_between(hours,mean-data['y'][data['train_rows']].std(0),mean+data['y'][data['train_rows']].std(0),alpha=.25,label='Across-day SD')
    axes[1,1].set(xlabel='Local time (hours)',ylabel='Demand (kW)',title='Training mean and variation');axes[1,1].legend()
    fig.suptitle('UCI Electricity: MT_001, 2014 Q1 weekdays')
    fig.tight_layout();fig.savefig(OUTPUT/'data_overview.png',dpi=150);fig.savefig(OUTPUT/'data_overview.pdf');plt.close(fig)


def fit_model(data,center,model):
    folder=OUTPUT/model;folder.mkdir(exist_ok=True);source_hash=digest(OUTPUT/'dataset.npz')
    if (folder/'fit.json').exists():
        fit=json.loads((folder/'fit.json').read_text());assert fit['dataset_sha256']==source_hash
        return fit
    Y=data['y'][data['train_rows']]/data['common_scale'];x=data['x']
    rng=np.random.default_rng(24026);base=center.copy()
    if model!='gp':base=np.r_[base,np.full(len(Y),np.log(.6) if model=='stp' else np.arcsinh(1.))]
    points=[]
    for hw in (.04,1.,16.,256.,4096.,65536.):
        for repeat in range(2):
            q=base.copy();q[4]=np.log(hw)
            if repeat:q+=rng.normal(0,.4,len(q))
            points.append(q)
    print(f'{model}: fitting {len(Y)} daily curves, {len(points)} starts',flush=True);started=time.perf_counter()
    fit=fit_shared_function(Y,x,x,model,starts=len(points),seed=24026,prior_center=center,
                            backend='dense',initial_points=points,refine_stationary=True)
    fit.update(dataset_sha256=source_hash,function_original=data['common_scale']*fit['function'],
               population_mean_original=data['common_scale']*fit['population_mean'],
               fitting_seconds=time.perf_counter()-started,initial_points=points)
    write_json(folder/'fit.json',fit)
    print(f'{model}: MAP {fit["log_posterior"]:.5f}; Hessian max {fit["max_hessian_eigenvalue"]:.3g}',flush=True)
    return fit


def predict_model(data,model,fit):
    folder=OUTPUT/model;fit_hash=digest(folder/'fit.json');function=np.asarray(fit['function'])
    v,a,k,w,hw=np.exp(fit['log_parameters'][:5]);scale=float(data['common_scale'])
    kind='beta' if model=='ghp_beta' else 'mu';mixing='gp' if model=='gp' else 'ig' if model=='stp' else 'gig'
    rows=[]
    for count in COUNTS:
        case=folder/f'context_{count}';case.mkdir(exist_ok=True)
        for j,row in enumerate(data['test_rows']):
            stem=case/f'day_{data["dates"][row]}'
            if stem.with_suffix('.json').exists() and stem.with_suffix('.npz').exists():
                record=json.loads(stem.with_suffix('.json').read_text())
                if record['fit_sha256']==fit_hash:rows.append(record['scores']);continue
            ids=np.sort(data['context_orders'][j,:count]);held=np.setdiff1d(np.arange(96),ids)
            pred=fixed_function_prediction(data['y'][row,ids]/scale,data['x'][ids],data['x'],v0=v,k=k,w=w,mixing=mixing,
                                          **{f'{kind}_train':function[ids],f'{kind}_pred':function})
            pred=original_units(pred,scale)
            scores,point,validation=score_prediction(pred,None,data['y'][row],held,check=j==0)
            scores.update(model=model,context_count=count,date=str(data['dates'][row]),n_test=len(held),
                          W_mean=pred['W_mean'],W_std=float(np.sqrt(pred['W_var'])))
            compact={key:value for key,value in pred.items() if not key.endswith('_cov')}
            compact.update(f_scale_var=np.diag(pred['f_scale_cov']),y_scale_var=np.diag(pred['y_scale_cov']))
            np.savez_compressed(stem.with_suffix('.npz'),**compact,**point,context_ids=ids,test_ids=held)
            write_json(stem.with_suffix('.json'),dict(scores=scores,validation=validation,fit_sha256=fit_hash))
            rows.append(scores)
        print(f'{model}: finished {count}-point contexts for {len(data["test_rows"])} test days',flush=True)
    pd.DataFrame(rows).to_csv(folder/'per_day_scores.csv',index=False)


def aggregate(curves):
    result=[]
    for (count,model),frame in curves.groupby(['context_count','model'],sort=False):
        assert len(frame)==18
        row=dict(context_count=count,model=model,n_days=len(frame),y_rmse=float(np.sqrt(np.mean(frame.y_rmse**2))))
        row.update({key:float(frame[key].mean()) for key in METRICS[1:]})
        for key in METRICS:
            row[key+'_day_mean']=float(frame[key].mean());row[key+'_day_sd']=float(frame[key].std(ddof=1))
        result.append(row)
    return pd.DataFrame(result)


def summarize(data):
    curves=pd.concat([pd.read_csv(OUTPUT/model/'per_day_scores.csv') for model in MODELS],ignore_index=True)
    curves.to_csv(OUTPUT/'per_day_scores.csv',index=False);summary=aggregate(curves);summary.to_csv(OUTPUT/'scores.csv',index=False)
    print(summary[['context_count','model',*METRICS]].to_string(index=False),flush=True)
    common=[];parameters=[];paired=[]
    for count in COUNTS:
        base=curves[(curves.model=='gp')&(curves.context_count==count)].set_index('date')
        for model in MODELS:
            part=curves[(curves.model==model)&(curves.context_count==count)].set_index('date').loc[base.index]
            record=dict(context_count=count,model=model)
            for metric in METRICS:
                delta=part[metric]-base[metric];record[metric+'_mean_difference_vs_gp']=float(delta.mean())
                record[metric+'_difference_sd']=float(delta.std(ddof=1))
                record[metric+'_days_lower_than_gp']=int((delta<0).sum())
            paired.append(record)
    pd.DataFrame(paired).to_csv(OUTPUT/'paired_comparison.csv',index=False)
    gp_rows=curves[curves.model=='gp'].set_index(['context_count','date'])
    beta_rows=curves[curves.model=='ghp_beta'].set_index(['context_count','date']).loc[gp_rows.index]
    changes=beta_rows[['y_rmse','y_nlpd','y_crps']]-gp_rows[['y_rmse','y_nlpd','y_crps']]
    changes['mse_change']=beta_rows.y_rmse**2-gp_rows.y_rmse**2
    changes.to_csv(OUTPUT/'ghp_beta_vs_gp_per_day.csv')
    # Descriptive post-fit check; it does not alter training, contexts or primary scores.
    threshold=float(data['y'][data['train_rows']].mean(1).max())
    high_dates=data['dates'][data['y'].mean(1)>threshold]
    grouped=curves.assign(high_demand=curves.date.isin(high_dates));subgroups=[]
    for (count,model,high),frame in grouped.groupby(['context_count','model','high_demand']):
        subgroups.append(dict(context_count=count,model=model,high_demand=bool(high),n_days=len(frame),
            threshold_daily_mean_kw=threshold,y_rmse=float(np.sqrt(np.mean(frame.y_rmse**2))),
            **{key:float(frame[key].mean()) for key in METRICS[1:]}))
    pd.DataFrame(subgroups).to_csv(OUTPUT/'descriptive_day_type_scores.csv',index=False)
    for model in MODELS:
        fit=json.loads((OUTPUT/model/'fit.json').read_text());v,a,k,w,hw=np.exp(fit['log_parameters'][:5]);scale=float(data['common_scale'])
        parameters.append(dict(model=model,residual_variance=v*scale**2,population_function_variance=a*scale**2,
            native_function_variance=a*scale**2/(gig_mean()**2 if model=='ghp_beta' else 1.),noise_coefficient=k*scale**2,
            residual_inverse_length_squared=w,function_inverse_length_squared=hw,
            W_train_min=float(min(fit['W_training_mode'])),W_train_max=float(max(fit['W_training_mode'])),
            log_posterior=fit['log_posterior'],max_hessian_eigenvalue=fit['max_hessian_eigenvalue']))
        for count in COUNTS:
            for j,row in enumerate(data['test_rows']):
                with np.load(OUTPUT/model/f'context_{count}'/f'day_{data["dates"][row]}.npz') as p:
                    ids=np.setdiff1d(np.arange(96),data['context_orders'][j,:30]);pos=np.searchsorted(p['test_ids'],ids)
                    truth=data['y'][row,ids];ci=p['y_ci'][ids]
                    common.append(dict(model=model,context_count=count,date=str(data['dates'][row]),n_test=len(ids),
                        y_rmse=float(np.sqrt(np.mean((truth-p['y_mean'][ids])**2))),y_nlpd=float(p['y_nlpd'][pos].mean()),
                        y_crps=float(p['y_crps'][pos].mean()),y_coverage_95=float(np.mean((truth>=ci[:,0])&(truth<=ci[:,1]))),
                        y_interval_width=float(np.mean(ci[:,1]-ci[:,0]))))
    pd.DataFrame(parameters).to_csv(OUTPUT/'parameters.csv',index=False)
    common=pd.DataFrame(common);common.to_csv(OUTPUT/'common_test_per_day_scores.csv',index=False)
    aggregate(common).to_csv(OUTPUT/'common_test_scores.csv',index=False)
    lines=[r'\begin{table}[htbp]',r'\centering',
        r'\caption{UCI Electricity customer MT\_001: 2014 Q1 weekdays, 40 training days, five gap days and 18 test days. Shared plug-in functions and MAP hyperparameters. Held-out demand is in kW; RMSE is pooled, remaining scores averaged.}',
        r'\label{tab:electricity_shared_plugin}',r'\begin{tabular}{rlrrrr}',r'\toprule',
        r'Observed points & Model & RMSE & NLPD & CRPS & 95\% coverage (\%) \\',r'\midrule']
    names=dict(gp=r'GP ($\hat\mu$)',stp=r'STP ($\hat\mu$)',ghp_mu=r'GHP ($\hat\mu$)',ghp_beta=r'GHP ($\hat\beta$)')
    for count in COUNTS:
        for model in MODELS:
            row=summary[(summary.context_count==count)&(summary.model==model)].iloc[0]
            lines.append(f'{count} & '+names[model]+f' & {row.y_rmse:.5f} & {row.y_nlpd:.5f} & {row.y_crps:.5f} & {100*row.y_coverage_95:.2f}'+r' \\')
        if count==15:lines.append(r'\midrule')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (OUTPUT/'comparison.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    hours=data['time_hours']
    with PdfPages(OUTPUT/'regression.pdf') as pdf:
        for count in COUNTS:
            for row in data['test_rows']:
                fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,sharey=True)
                for ax,model in zip(axes.flat,MODELS):
                    with np.load(OUTPUT/model/f'context_{count}'/f'day_{data["dates"][row]}.npz') as p:
                        ax.fill_between(hours,*p['y_ci'].T,alpha=.25,label='95% observation interval')
                        ax.plot(hours,p['y_mean'],label='Predicted mean');ax.plot(hours,data['y'][row],'k--',lw=1,label='Observed demand')
                        ids=p['context_ids'];ax.scatter(hours[ids],data['y'][row,ids],s=14,c='black',label='Revealed points')
                        ax.set(title=LABELS[model],xlabel='Local time (hours)',ylabel='Demand (kW)')
                axes[0,0].legend(fontsize=8);fig.suptitle(f'MT_001: {data["dates"][row]}, {count} revealed readings')
                fig.tight_layout();pdf.savefig(fig)
                if row==data['test_rows'][0]:fig.savefig(OUTPUT/f'regression_context{count}.png',dpi=150)
                plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].plot(hours,data['common_mean'],'k--',label='Empirical training mean')
    for model in MODELS:
        fit=json.loads((OUTPUT/model/'fit.json').read_text())
        axes[0].plot(hours,fit['population_mean_original'],label=LABELS[model])
        axes[1].plot(hours,np.asarray(fit['population_mean_original'])-data['common_mean'],label=LABELS[model])
    axes[0].set(xlabel='Local time (hours)',ylabel='Demand (kW)',title='Learned population mean functions');axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Local time (hours)',ylabel='Difference (kW)',title='Difference from empirical mean')
    fig.tight_layout();fig.savefig(OUTPUT/'shared_functions.png',dpi=150);fig.savefig(OUTPUT/'shared_functions.pdf');plt.close(fig)
    (OUTPUT/'method.txt').write_text(METHOD,encoding='utf-8')


METHOD='''ELECTRICITY: ONE CUSTOMER, SHARED PLUG-IN FUNCTIONS

Source: UCI ElectricityLoadDiagrams20112014, Trindade (2015), DOI 10.24432/C58C86.
Customer MT_001 was selected as the first source column before any model fitting.
Daily groups use recorded Portuguese local calendar dates, 00:00--23:45. No
timestamp shift, resampling, outlier deletion, or demand log transformation.
Use 2014 Q1 weekdays excluding January 1. Complete 96-point, nonzero days only;
DST-adjusted dates are excluded. First 40 eligible days train, next five gap,
remaining 18 test. No shuffling of days. This limits calendar heterogeneity but
does not remove Monday-Friday effects, seasonal trends, or serial dependence.
Reconstruction from scattered readings is the task, not next-day forecasting.

Scale every reading by a single training-derived residual SD around the empirical
training mean curve. No centering and no per-day scaling; amplitudes are retained.
The empirical mean sets a GP amplitude prior center; it is not subtracted or fixed.
Inputs span [0,10] over the 96 recorded quarter-hour labels.

Training models: Y_j = c_j h + sqrt(W_j) e_j, e_j~N(0,K+kI), shared h~GP(0,Sigma).
GP: W=c=1. STP/GHP-mu: c=1, h=mu. GHP-beta: c=W/E_prior[W], h=E_prior[W]*beta.
This gives the same prior on the population-mean function h in all models.
Both kernels are SE: variance*exp(-inverse_length_squared*(x-x')^2/2).
The STP baseline adds an unscaled GP mean; it is not a pure Student-t process.
Fixed priors: GHP GIG(lambda=1,chi=psi=1); STP IG(shape=2.5,scale=1.5), nu=5.
Five independent Normal priors on log kernel/noise parameters, SD=2. Centers are
log([0.5, max(mean(training_mean_scaled^2),0.5), 0.01, 1, 0.04]).

Integrate out shared h analytically, then optimize log theta and training log W
jointly (12 starts, multiple length scales). Includes the log-W Jacobians. This
MAP depends on the chosen log coordinates. Check the stationary solution by
finite differences and local Hessian; no guarantee of a global optimum.
At the fit, set h_hat=Sigma*(A+t*Sigma)^(-1)*b, A=K+kI,
t=sum(c_j^2/W_j), b=sum(c_j Y_j/W_j). beta_hat=h_hat/E_prior[W], mu_hat=h_hat.

For each test day reveal nested random 15/30 readings. Freeze theta and the shared
function. Use only that context to infer its new W; no training W is reused.
For fixed mu,beta: A_c=K_cc+kI, r=y_c-mu_c. The GIG posterior has
lambda'=1-n/2, chi'=1+r'A_c^(-1)r, psi'=1+beta_c'A_c^(-1)beta_c.
For STP: IG((5+n)/2,(3+r'A_c^(-1)r)/2). GP fixes W=1.
Conditional prediction is N(m+W*b,W*S), with standard residual-GP conditioning;
integrate W to obtain GH/Student-t/Normal marginal densities and equal-tail 95%
intervals. Noise adds W*k to observation variance. Hyperparameter and shared
function estimation uncertainty are omitted, as in the previous plug-in runs.

Evaluate observed y only; latent f is unknown. RMSE is pooled across held-out
points, NLPD/CRPS/coverage/interval width averaged over days. Primary scores use
81/66 held-out points for 15/30 contexts; common_test_scores.csv compares the
same 66 evaluation points for both contexts. Per-day SD is not a split-to-split
uncertainty estimate; adjacent days may remain dependent. No iid significance
test is claimed. Paired differences versus GP are retained.
descriptive_day_type_scores.csv is a post-fit diagnostic separating test days
whose average demand exceeds the largest training-day average. This diagnostic
uses the complete test outcomes for scoring only; it never enters prediction.
All four likelihoods have support on the real line, so they allow negative demand.
The validation script saves negative_demand_probability.csv to quantify this
limitation; the intervals have not been clipped to hide it.

Keep only compact marginal predictions, a single-customer source cache and small
metadata. No large source ZIP, complete 370-customer table or dense covariance
arrays are retained on disk. See experiment.json and validation.json for checks.
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--model',choices=MODELS)
    parser.add_argument('--stage',choices=['prepare','all','summarize'],default='all');args=parser.parse_args()
    with threadpool_limits(limits=1):
        data,center=prepare()
        if args.stage=='prepare':exploration(data);return
        if args.stage=='all':
            for model in ([args.model] if args.model else MODELS):
                fit=fit_model(data,center,model);predict_model(data,model,fit)
        if not args.model or args.stage=='summarize':summarize(data)


if __name__=='__main__':main()
