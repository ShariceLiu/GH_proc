"""Small condition-controlled FS1 pilot with four shared-function MAP models.

python hydraulic_shared_plugin_experiment.py --stage prepare
python hydraulic_shared_plugin_experiment.py --model gp
python hydraulic_shared_plugin_experiment.py --stage summarize
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

from load_hydraulic_fs1 import load_data, CACHE, PROFILE_NAMES
from GH_shared_function_map import MODELS, make_target, fit_shared_function, gig_mean
from GH_fixed_function import fixed_function_prediction
from shared_beta_plugin_experiment import write_json, score_prediction, LABELS
from tecator_shared_plugin_experiment import original_units

ROOT=Path(__file__).resolve().parent
OUTPUT=ROOT/'figure'/'hydraulic_fs1_shared_plugin'
CONTEXT_SEED=34026
COUNTS=(15,30)
METRICS=('y_rmse','y_nlpd','y_crps','y_coverage_95','y_interval_width')


def prepare():
    source,counts=load_data()
    # Condition selection uses profile counts only, never prediction errors.
    largest=counts.iloc[0]
    condition=largest[PROFILE_NAMES[:4]].to_numpy(dtype=int)
    ids=np.flatnonzero(np.all(source['profile'][:,:4]==condition,axis=1)&(source['profile'][:,4]==0))
    if len(ids)!=19 or not np.array_equal(condition,[100,100,0,90]):
        raise ValueError('Source population changed: inspect counts and define a new protocol')
    train,gap,test=ids[:12],ids[12:14],ids[14:]
    t=source['time_seconds'];x=(t-t[0])/(t[-1]-t[0])*10
    y=source['y']
    scale=float(np.std(y[train]-y[train].mean(axis=0)))
    Y=y[train]/scale
    center=np.log([.5,max(float(np.mean(Y.mean(axis=0)**2)),.5),.01,1.,.04])
    orders=np.stack([np.random.default_rng(np.random.SeedSequence([CONTEXT_SEED,int(row)])).permutation(100) for row in test])
    data=dict(x=x,time_seconds=t,y=y,profile=source['profile'],cycle_ids=source['cycle_ids'],
              eligible_rows=ids,train_rows=train,gap_rows=gap,test_rows=test,
              common_scale=np.array(scale),common_mean=y[train].mean(axis=0),context_orders=orders)
    settings=dict(sensor='FS1',unit='litres/minute',n_train=12,n_gap=2,n_test=5,points_per_curve=100,
                  condition=dict(zip(PROFILE_NAMES[:4],map(int,condition))),stable_flag=0,
                  selection='Largest exactly matched stable condition group, selected solely by profile counts.',
                  limitation='Only 19 cycles; preliminary controlled pilot, not a robust model-ranking study.',
                  training_cycles=source['cycle_ids'][train],gap_cycles=source['cycle_ids'][gap],test_cycles=source['cycle_ids'][test],
                  split='First 12 cycles train, next 2 excluded as gap, last 5 test, preserving source cycle order.',
                  aggregation='Arithmetic averages of 6 non-overlapping raw readings; 600 points -> 100.',
                  context_counts=COUNTS,context_seed=CONTEXT_SEED,contexts='Nested random subsets, fixed across models.',
                  preprocessing='Training-only common residual scale; no curve-wise normalization or centering.',
                  prior_center=center,prior_sd=2.,GHP_lambda=1.,GHP_omega=1.,STP_nu=5.,
                  inference='Shared GP marginalized for log(theta),log(W_train) MAP; plug in shared conditional mean.',
                  prediction='Freeze kernel/noise and shared function; infer/integrate fresh W from each test context only.',
                  source_sha256=hashlib.sha256((CACHE/'fs1.npz').read_bytes()).hexdigest(),
                  scores='Held-out aggregated observed flow y, original units; no latent-f ground truth.')
    OUTPUT.mkdir(parents=True,exist_ok=True)
    if (OUTPUT/'dataset.npz').exists():
        with np.load(OUTPUT/'dataset.npz') as saved:
            for key in data:np.testing.assert_array_equal(saved[key],data[key])
        assert json.loads((OUTPUT/'experiment.json').read_text())==json.loads(json.dumps(settings,default=lambda a:a.tolist()))
    else:
        np.savez_compressed(OUTPUT/'dataset.npz',**data)
        write_json(OUTPUT/'experiment.json',settings)
    roles=pd.DataFrame(source['profile'],columns=PROFILE_NAMES)
    roles.insert(0,'cycle_id',source['cycle_ids']);roles['role']='excluded_other_condition_or_unstable'
    roles.loc[train,'role']='train';roles.loc[gap,'role']='gap';roles.loc[test,'role']='test'
    roles.to_csv(OUTPUT/'cycle_roles.csv',index=False)
    counts.to_csv(OUTPUT/'stable_condition_counts.csv',index=False)
    return data,center


def exploration(data):
    t=data['time_seconds'];fig,axes=plt.subplots(2,2,figsize=(12,7))
    for key,color in [('train','C0'),('gap','C1'),('test','C3')]:
        rows=data[key+'_rows']
        for i,row in enumerate(rows):axes[0,0].plot(t,data['y'][row],color=color,alpha=.35,label=key if i==0 else None)
        axes[0,1].plot(data['cycle_ids'][rows],data['y'][rows].mean(axis=1),'o-',color=color,label=key)
    axes[0,0].set(xlabel='Time within cycle (s)',ylabel='FS1 flow (L/min)',title='Exactly matched stable cycles')
    axes[0,0].legend()
    axes[0,1].set(xlabel='Source cycle number',ylabel='Cycle mean flow (L/min)',title='Ordered split with two-cycle gap')
    axes[0,1].legend()
    mean=data['y'][data['train_rows']].mean(0)
    for row in data['train_rows']:axes[1,0].plot(t,data['y'][row]-mean,alpha=.5)
    axes[1,0].set(xlabel='Time within cycle (s)',ylabel='Deviation (L/min)',title='Training deviations from training mean')
    axes[1,1].plot(t,data['y'][data['train_rows']].std(0))
    axes[1,1].set(xlabel='Time within cycle (s)',ylabel='Across-training-cycle SD (L/min)',title='Variation across the cycle')
    fig.tight_layout();fig.savefig(OUTPUT/'data_overview.png',dpi=150);fig.savefig(OUTPUT/'data_overview.pdf');plt.close(fig)


def fit_model(data,center,model):
    folder=OUTPUT/model;folder.mkdir(exist_ok=True)
    source_hash=hashlib.sha256((OUTPUT/'dataset.npz').read_bytes()).hexdigest()
    if (folder/'fit.json').exists():
        fit=json.loads((folder/'fit.json').read_text());assert fit['dataset_sha256']==source_hash
        return fit
    Y=data['y'][data['train_rows']]/data['common_scale'];x=data['x']
    # Dense exact SE algebra has no Taylor-factor length-scale limitation.
    # Short shared length scales are essential for this switching-cycle mean.
    rng=np.random.default_rng(24026);base=center.copy()
    if model!='gp':base=np.r_[base,np.full(len(Y),np.log(.6) if model=='stp' else np.arcsinh(1.))]
    points=[]
    for hw in (.04,1.,16.,256.,4096.,65536.):
        for repeat in range(2):
            q=base.copy();q[4]=np.log(hw)
            if repeat:q+=rng.normal(0,.4,len(q))
            points.append(q)
    print(f'{model}: fitting {len(Y)} curves with {len(points)} starts',flush=True)
    started=time.perf_counter()
    fit=fit_shared_function(Y,x,x,model,starts=len(points),seed=24026,prior_center=center,
                            backend='dense',initial_points=points,refine_stationary=True)
    fit.update(dataset_sha256=source_hash,function_original=data['common_scale']*fit['function'],
               population_mean_original=data['common_scale']*fit['population_mean'],
               fitting_seconds=time.perf_counter()-started,initial_points=points)
    write_json(folder/'fit.json',fit)
    print(f'{model}: MAP={fit["log_posterior"]:.6f}, Hessian max={fit["max_hessian_eigenvalue"]:.3g}',flush=True)
    return fit


def predict_model(data,model,fit):
    folder=OUTPUT/model
    fit_hash=hashlib.sha256((folder/'fit.json').read_bytes()).hexdigest()
    function=np.asarray(fit['function']);v,a,k,w,hw=np.exp(fit['log_parameters'][:5])
    scale=float(data['common_scale']);kind='beta' if model=='ghp_beta' else 'mu'
    mixing='gp' if model=='gp' else 'ig' if model=='stp' else 'gig'
    rows=[]
    for count in COUNTS:
        case=folder/f'context_{count}';case.mkdir(exist_ok=True)
        for j,row in enumerate(data['test_rows']):
            stem=case/f'cycle_{data["cycle_ids"][row]}'
            if stem.with_suffix('.json').exists():
                record=json.loads(stem.with_suffix('.json').read_text())
                if record['fit_sha256']==fit_hash and stem.with_suffix('.npz').exists():
                    rows.append(record['scores']);continue
            ids=np.sort(data['context_orders'][j,:count]);held=np.setdiff1d(np.arange(100),ids)
            kwargs={f'{kind}_train':function[ids],f'{kind}_pred':function}
            pred=fixed_function_prediction(data['y'][row,ids]/scale,data['x'][ids],data['x'],
                                          v0=v,k=k,w=w,mixing=mixing,**kwargs)
            pred=original_units(pred,scale)
            scores,point,validation=score_prediction(pred,None,data['y'][row],held,check=j==0)
            scores.update(model=model,context_count=count,cycle_id=int(data['cycle_ids'][row]),
                          n_test=len(held),W_mean=pred['W_mean'],W_std=float(np.sqrt(pred['W_var'])))
            # Preserve marginals only; full 100x100 covariance is reproducible
            # from dataset.npz and fit.json and is not needed for these scores.
            compact={key:value for key,value in pred.items() if not key.endswith('_cov')}
            compact.update(f_scale_var=np.diag(pred['f_scale_cov']),y_scale_var=np.diag(pred['y_scale_cov']))
            np.savez_compressed(stem.with_suffix('.npz'),**compact,**point,context_ids=ids,test_ids=held)
            write_json(stem.with_suffix('.json'),dict(scores=scores,validation=validation,fit_sha256=fit_hash))
            rows.append(scores)
        print(f'{model}: finished {count}-point contexts',flush=True)
    pd.DataFrame(rows).to_csv(folder/'per_curve_scores.csv',index=False)


def summarize(data):
    frames=[]
    for model in MODELS:
        path=OUTPUT/model/'per_curve_scores.csv'
        if path.exists():frames.append(pd.read_csv(path))
    if not frames:return
    curves=pd.concat(frames,ignore_index=True);curves.to_csv(OUTPUT/'per_curve_scores.csv',index=False)
    summaries=[]
    for (count,model),frame in curves.groupby(['context_count','model'],sort=False):
        assert len(frame)==5
        row=dict(context_count=count,model=model,n_curves=len(frame),y_rmse=float(np.sqrt(np.mean(frame.y_rmse**2))))
        row.update({key:float(frame[key].mean()) for key in METRICS[1:]})
        # Explicitly distinguish mean per-curve RMSE from pooled point RMSE.
        for key in METRICS:
            row[key+'_curve_mean']=float(frame[key].mean());row[key+'_curve_sd']=float(frame[key].std(ddof=1))
        summaries.append(row)
    summary=pd.DataFrame(summaries)
    summary.to_csv(OUTPUT/'scores.csv',index=False)
    print(summary[['context_count','model',*METRICS]].to_string(index=False),flush=True)
    lines=[r'\begin{table}[htbp]',r'\centering',
           r'\caption{Hydraulic FS1 controlled pilot: 12 training cycles, two excluded gap cycles, five test cycles. Scores use held-out six-reading averages in L/min. RMSE is pooled over test points; other scores are averages.}',
           r'\label{tab:hydraulic_fs1_plugin}',r'\begin{tabular}{rlrrrr}',r'\toprule',
           r'Observed points & Model & RMSE & NLPD & CRPS & 95\% coverage (\%) \\',r'\midrule']
    names=dict(gp=r'GP ($\hat\mu$)',stp=r'STP ($\hat\mu$)',ghp_mu=r'GHP ($\hat\mu$)',ghp_beta=r'GHP ($\hat\beta$)')
    for count in COUNTS:
        for model in MODELS:
            selected=summary[(summary.context_count==count)&(summary.model==model)]
            if selected.empty:continue
            row=selected.iloc[0]
            lines.append(f'{count} & '+names[model]+f' & {row.y_rmse:.5f} & {row.y_nlpd:.5f} & {row.y_crps:.5f} & {100*row.y_coverage_95:.2f}'+r' \\')
        if count==15:lines.append(r'\midrule')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (OUTPUT/'comparison.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if len(summary)!=8:return
    t=data['time_seconds']
    with PdfPages(OUTPUT/'regression.pdf') as pdf:
        for count in COUNTS:
            for row in data['test_rows']:
                fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,sharey=True)
                for ax,model in zip(axes.flat,MODELS):
                    with np.load(OUTPUT/model/f'context_{count}'/f'cycle_{data["cycle_ids"][row]}.npz') as p:
                        ax.fill_between(t,*p['y_ci'].T,alpha=.25)
                        ax.plot(t,p['y_mean'],label='Predicted mean')
                        ax.plot(t,data['y'][row],'k--',lw=1,label='Measured bin averages')
                        ids=p['context_ids'];ax.scatter(t[ids],data['y'][row,ids],s=14,c='black',label='Revealed points')
                        ax.set(title=LABELS[model],xlabel='Time within cycle (s)',ylabel='FS1 flow (L/min)')
                axes[0,0].legend(fontsize=8)
                fig.suptitle(f'Cycle {data["cycle_ids"][row]}, {count} observed points; pointwise 95% observation intervals')
                fig.tight_layout();pdf.savefig(fig)
                if row==data['test_rows'][0]:fig.savefig(OUTPUT/f'regression_context{count}.png',dpi=150)
                plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,4))
    axes[0].plot(t,data['common_mean'],'k--',label='Empirical training mean')
    for model in MODELS:
        fit=json.loads((OUTPUT/model/'fit.json').read_text())
        axes[0].plot(t,fit['population_mean_original'],label=LABELS[model])
        axes[1].plot(t,np.asarray(fit['population_mean_original'])-data['common_mean'],label=LABELS[model])
    axes[0].set(xlabel='Time within cycle (s)',ylabel='Flow (L/min)',title='Learned population means')
    axes[0].legend(fontsize=8)
    axes[1].set(xlabel='Time within cycle (s)',ylabel='Difference (L/min)',title='Difference from empirical training mean')
    fig.tight_layout();fig.savefig(OUTPUT/'shared_functions.png',dpi=150);fig.savefig(OUTPUT/'shared_functions.pdf');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=MODELS)
    parser.add_argument('--stage',choices=['prepare','all','summarize'],default='all')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        data,center=prepare()
        if args.stage=='prepare':exploration(data);return
        if args.stage=='all':
            for model in ([args.model] if args.model else MODELS):
                fit=fit_model(data,center,model);predict_model(data,model,fit)
        summarize(data)


if __name__=='__main__':main()
