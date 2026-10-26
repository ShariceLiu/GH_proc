"""GHP with both shared mu and shared beta on the five saved Tecator splits."""
import os
for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[variable]='1'
import argparse
from concurrent.futures import ProcessPoolExecutor,as_completed
from contextlib import redirect_stdout,redirect_stderr
import hashlib
import json
from pathlib import Path
import time
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy import linalg
from threadpoolctl import threadpool_limits

from GH_shared_mu_beta_map import fit_shared_mu_beta,shared_mu_beta_calculation,make_target
from GH_shared_function_map import gig_mean
from GH_fixed_function import fixed_function_prediction
from shared_beta_plugin_experiment import write_json,score_prediction
from tecator_shared_plugin_experiment import load_data,original_units
from tecator_shared_plugin_five_splits import OUTPUT as BASE,SEEDS,METRICS,TEX_NAMES,build_split

OUTPUT=BASE.parent/'tecator_shared_mu_beta_five_splits'
MODEL='ghp_mu_beta'


def prepare(seed):
    source=BASE/f'split_{seed}';folder=OUTPUT/f'split_{seed}';folder.mkdir(parents=True,exist_ok=True)
    data=load_data(source);expected,old_center,_,_=build_split(seed)
    for key in expected:np.testing.assert_array_equal(data[key],expected[key])
    # Split the prior population-mean variance budget between independent GPs.
    center=old_center[[0,1,1,2,3,4,4]].copy();center[1:3]-=np.log(2)
    settings=dict(model=MODEL,split_seed=seed,source=str(source),
                  dataset_sha256=hashlib.sha256((source/'dataset.npz').read_bytes()).hexdigest(),
                  prior_center=center,prior_sd=2.,lambda_fixed=1.,omega_fixed=1.,
                  function_priors='Independent zero-mean SE GPs on mu and h_beta=E[W]*beta; half the old population-mean amplitude center each.',
                  estimation='Integrate both shared GPs; log-coordinate theta/W MAP; plug in joint conditional means.',
                  starts=12,initialization='Own-split training MAPs and both W-level orientations; no held-out values used.')
    if (folder/'dataset.npz').exists():
        saved=load_data(folder)
        for key in data:np.testing.assert_array_equal(saved[key],data[key])
        old=json.loads((folder/'settings.json').read_text())
        if old!=json.loads(json.dumps(settings,default=lambda a:a.tolist())):raise ValueError('Saved settings changed')
    else:
        np.savez_compressed(folder/'dataset.npz',**data);write_json(folder/'settings.json',settings)
    return folder,data,center,source


def initial_points(source,center,data):
    b=np.asarray(json.loads((source/'ghp_beta'/'fit.json').read_text())['log_parameters'])
    m=np.asarray(json.loads((source/'ghp_mu'/'fit.json').read_text())['log_parameters'])
    qb=np.r_[b[0],center[1],b[1],b[2],b[3],center[5],b[4],b[5:]]
    level=data['y'][data['train_rows']].mean(axis=1);level=(level-level.mean())/level.std()
    qm=np.r_[m[0],m[1],center[2],m[2],m[3],m[4],np.log(4.),np.log(gig_mean())+.3*level]
    points=[qb.copy(),qm.copy()]
    for original in (qb,qm):
        reflected=original.copy();reflected[7:]=2*np.mean(original[7:])-original[7:]
        points.append(reflected)
    rng=np.random.default_rng(24026)
    for original in (qb,qm):
        for _ in range(2):
            q=original.copy();q[:7]+=rng.normal(0,.6,7);q[7:]+=rng.normal(0,.1,len(level));points.append(q)
    for wm,wb in ((1.,1.),(4.,16.),(16.,4.),(16.,16.)):
        q=qb.copy();q[5:7]=np.log([wm,wb]);points.append(q)
    return points


def fit_model(folder,data,center,source):
    if (folder/'fit.json').exists():return json.loads((folder/'fit.json').read_text())
    checkpoint=folder/'optimization_starts.json'
    completed=json.loads(checkpoint.read_text()) if checkpoint.exists() else []
    def progress(record,q):
        completed.append(dict(record=record,log_parameters=q.tolist()))
        write_json(checkpoint,completed)
        print(f'Start {record["start"]+1}/12: log posterior={record["log_posterior"]:.6f}, '
              f'gradient={record["max_abs_gradient"]:.3g}, accepted={record["success"]}',flush=True)
    fit=fit_shared_mu_beta(data['y'][data['train_rows']]/data['common_scale'],data['x'],center,
                           initial_points(source,center,data),progress=progress,completed=list(completed))
    scale=float(data['common_scale'])
    fit.update(dataset_sha256=hashlib.sha256((folder/'dataset.npz').read_bytes()).hexdigest(),
               mu_original=scale*fit['mu'],beta_original=scale*fit['beta'],
               population_mean_original=scale*(fit['mu']+gig_mean()*fit['beta']))
    write_json(folder/'fit.json',fit);return fit


def predict_and_validate(folder,data,fit,center,seed):
    scale=float(data['common_scale']);q=np.asarray(fit['log_parameters']);v,am,ab,k,w,wm,wb=np.exp(q[:7])
    mu,beta=np.asarray(fit['mu']),np.asarray(fit['beta'])
    target=make_target(data['y'][data['train_rows']]/scale,data['x'],center)
    value,gradient=target(q);assert max(abs(gradient))<1e-3 and fit['max_hessian_eigenvalue']<0
    np.testing.assert_allclose(value,fit['log_posterior'],atol=1e-6,rtol=0)
    calc=shared_mu_beta_calculation(data['y'][data['train_rows']]/scale,data['x'],q,posterior=True)
    np.testing.assert_allclose(mu,calc['mu'],atol=1e-10);np.testing.assert_allclose(beta,calc['beta'],atol=1e-10)
    fit_hash=hashlib.sha256((folder/'fit.json').read_bytes()).hexdigest();rows=[];checks=[]
    for j,row in enumerate(data['test_rows']):
        stem=folder/f'curve_{row+1:03d}';ids=data['context_ids'][j];test=data['evaluation_ids'][j]
        if stem.with_suffix('.json').exists():
            record=json.loads(stem.with_suffix('.json').read_text())
            if record['fit_sha256']==fit_hash:
                rows.append(record['scores']);checks.append(record['validation']);continue
        p=fixed_function_prediction(data['y'][row,ids]/scale,data['x'][ids],data['x'],
                                    v0=v,k=k,w=w,mu_train=mu[ids],mu_pred=mu,
                                    beta_train=beta[ids],beta_pred=beta,mixing='gig')
        p=original_units(p,scale)
        scores,point,validation=score_prediction(p,None,data['y'][row],test,check=j==0)
        # Independent Gaussian conditioning in original units checks scaling
        # and use of both functions, not a beta-only prediction by accident.
        x=data['x'];kernel=lambda a,b:v*scale**2*np.exp(-.5*w*(a[:,None]-b[None,:])**2)
        A=kernel(x[ids],x[ids])+k*scale**2*np.eye(len(ids));cross=kernel(x,x[ids])
        B=linalg.solve(A,cross.T,assume_a='pos').T
        location=scale*mu+B@(data['y'][row,ids]-scale*mu[ids]);effect=scale*(beta-B@beta[ids])
        mean=location+p['W_mean']*effect
        covariance=p['W_mean']*(kernel(x,x)-B@cross.T+k*scale**2*np.eye(len(x)))+p['W_var']*np.outer(effect,effect)
        np.testing.assert_allclose(mean,p['y_mean'],atol=2e-8,rtol=1e-8)
        np.testing.assert_allclose(covariance,p['y_cov'],atol=2e-9,rtol=1e-6)
        scores.update(model=MODEL,split_seed=seed,sample_id=int(row+1),n_test=len(test),
                       W_mean=float(p['W_mean']),W_std=float(np.sqrt(p['W_var'])))
        np.savez_compressed(stem.with_suffix('.npz'),**p,**point,context_ids=ids,test_ids=test)
        write_json(stem.with_suffix('.json'),dict(scores=scores,validation=validation,fit_sha256=fit_hash))
        rows.append(scores);checks.append(validation)
        if (j+1)%5==0 or j+1==31:print(f'Prediction {j+1}/31',flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(folder/'per_curve_scores.csv',index=False)
    scores=dict(split_seed=seed,model=MODEL,y_rmse=float(np.sqrt(np.mean(frame.y_rmse**2))))
    scores.update({key:float(frame[key].mean()) for key in METRICS[1:]})
    pd.DataFrame([scores]).to_csv(folder/'scores.csv',index=False)
    write_json(folder/'validation.json',dict(all_checks_passed=True,predictions=len(rows),
               exact_saved_split=True,max_abs_gradient=float(max(abs(gradient))),curves=checks))


def plots(folder,data,fit,source):
    wave=data['wavelength_nm'];fig,axes=plt.subplots(1,3,figsize=(14,4))
    axes[0].plot(wave,fit['mu_original']);axes[0].set_title('Shared mu estimate')
    axes[1].plot(wave,fit['beta_original']);axes[1].set_title('Shared beta estimate')
    axes[2].plot(wave,data['common_mean'],'k--',label='Empirical training mean')
    axes[2].plot(wave,fit['population_mean_original'],label='mu + E[W] beta');axes[2].legend();axes[2].set_title('Population mean')
    for ax in axes:ax.set_xlabel('Wavelength (nm)');ax.set_ylabel('Absorbance')
    fig.tight_layout();fig.savefig(folder/'shared_functions.png',dpi=150);fig.savefig(folder/'shared_functions.pdf');plt.close(fig)
    with PdfPages(folder/'regression.pdf') as pdf:
        for start in range(0,31,5):
            subset=data['test_rows'][start:start+5];fig,axes=plt.subplots(len(subset),2,figsize=(11,2.5*len(subset)),squeeze=False,sharex=True,sharey='row')
            for i,row in enumerate(subset):
                for j,(base,label) in enumerate(((source/'ghp_beta','Shared beta only'),(folder,'Shared mu + beta'))):
                    with np.load(base/f'curve_{row+1:03d}.npz') as p:
                        ax=axes[i,j];ids=p['context_ids'];ax.fill_between(wave,*p['y_ci'].T,alpha=.25,color=f'C{j}')
                        ax.plot(wave,p['y_mean'],color=f'C{j}');ax.plot(wave,data['y'][row],'k--',lw=1)
                        ax.scatter(wave[ids],data['y'][row,ids],s=12,c='black');ax.set_title(f'{label}: sample {row+1}')
                        if j==0:ax.set_ylabel('Absorbance')
                        if i==len(subset)-1:ax.set_xlabel('Wavelength (nm)')
            fig.tight_layout();pdf.savefig(fig)
            if start==0:fig.savefig(folder/'regression.png',dpi=120)
            plt.close(fig)


def run_split(seed):
    folder,data,center,source=prepare(seed)
    with (folder/'run.log').open('a',encoding='utf-8',buffering=1) as log,redirect_stdout(log),redirect_stderr(log),threadpool_limits(1):
        try:
            started=time.perf_counter();print(f'Starting split {seed}',flush=True)
            fit=fit_model(folder,data,center,source)
            predict_and_validate(folder,data,fit,center,seed);plots(folder,data,fit,source)
            write_json(folder/'timing_latest.json',dict(seconds=time.perf_counter()-started))
            print('COMPLETE',seed,flush=True)
        except Exception:traceback.print_exc();raise
    return seed


def aggregate():
    frames=[pd.read_csv(BASE/'split_scores.csv')]
    for seed in SEEDS:
        folder=OUTPUT/f'split_{seed}';assert json.loads((folder/'validation.json').read_text())['all_checks_passed']
        frames.append(pd.read_csv(folder/'scores.csv'))
    scores=pd.concat(frames,ignore_index=True);scores.to_csv(OUTPUT/'split_scores.csv',index=False)
    names={**TEX_NAMES,MODEL:r'GHP (shared $\hat\mu,\hat\beta$)'}
    records=[];display=[]
    lines=[r'\begin{table}[htbp]',r'\centering',r'\caption{Tecator plug-in regression: mean $\pm$ sample standard deviation across the same five train/test splits. Scores are for observed $y$ in original absorbance units.}',
           r'\label{tab:tecator_shared_mu_beta}',r'\begin{tabular}{lrrrr}',r'\toprule',r'Model & RMSE & NLPD & CRPS & 95\% coverage (\%) \\',r'\midrule']
    for model in names:
        frame=scores[scores.model==model];assert len(frame)==5
        record=dict(model=model)
        for key in METRICS:record[key+'_mean']=float(frame[key].mean());record[key+'_std']=float(frame[key].std(ddof=1))
        records.append(record);cells=[];shown=dict(model=model)
        for key in ('y_rmse','y_nlpd','y_crps','y_coverage_95'):
            scale=100 if key=='y_coverage_95' else 1;precision=2 if scale==100 else 5
            mean,std=record[key+'_mean']*scale,record[key+'_std']*scale
            cells.append(f'${mean:.{precision}f} \\pm {std:.{precision}f}$');shown[key]=f'{mean:.{precision}f} +/- {std:.{precision}f}'
        display.append(shown);lines.append(names[model]+' & '+' & '.join(cells)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    pd.DataFrame(records).to_csv(OUTPUT/'summary.csv',index=False)
    (OUTPUT/'comparison.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    text=pd.DataFrame(display).to_string(index=False);(OUTPUT/'results.txt').write_text(text+'\n',encoding='utf-8');print(text,flush=True)
    new=scores[scores.model==MODEL].set_index('split_seed');old=scores[scores.model=='ghp_beta'].set_index('split_seed')
    paired=new[['y_rmse','y_nlpd','y_crps']]-old[['y_rmse','y_nlpd','y_crps']]
    paired.to_csv(OUTPUT/'difference_from_beta_only.csv')
    fig,axes=plt.subplots(1,3,figsize=(11,4))
    for ax,key in zip(axes,('y_rmse','y_nlpd','y_crps')):
        for seed in SEEDS:ax.plot([0,1],[old.loc[seed,key],new.loc[seed,key]],'o-',alpha=.7)
        ax.set_xticks([0,1],['Shared beta','Shared mu + beta'])
        ax.set_title(key.removeprefix('y_').upper());ax.grid(axis='y',alpha=.2)
    fig.suptitle('Paired comparison on the same five splits; lower is better')
    fig.tight_layout();fig.savefig(OUTPUT/'comparison.png',dpi=160);fig.savefig(OUTPUT/'comparison.pdf');plt.close(fig)
    decomposition=[]
    for seed in SEEDS:
        folder=OUTPUT/f'split_{seed}';fit=json.loads((folder/'fit.json').read_text());data=load_data(folder)
        mu_var=np.diag(fit['mu_cov']);beta_var=np.diag(fit['beta_cov']);cross=np.diag(fit['mu_beta_cov'])
        valid=(mu_var>0)&(beta_var>0)
        corr=cross[valid]/np.sqrt(mu_var[valid]*beta_var[valid])
        decomposition.append(dict(split_seed=seed,
            W_vs_spectrum_level_correlation=float(np.corrcoef(fit['W'],data['y'][data['train_rows']].mean(axis=1))[0,1]),
            beta_mean_original=float(np.mean(fit['beta_original'])),
            mu_mean_original=float(np.mean(fit['mu_original'])),
            mean_conditional_mu_beta_correlation=float(np.mean(corr)),
            min_conditional_mu_beta_correlation=float(np.min(corr)),
            max_conditional_mu_beta_correlation=float(np.max(corr)),
            nonpositive_covariance_diagonals=int(np.sum(~valid))))
    pd.DataFrame(decomposition).to_csv(OUTPUT/'decomposition_diagnostics.csv',index=False)
    write_json(OUTPUT/'validation.json',dict(all_checks_passed=True,new_fits=5,new_curve_predictions=155,
               same_five_splits=True,baseline_source=str(BASE),std_ddof=1))


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--seed',type=int,choices=SEEDS)
    parser.add_argument('--workers',type=int,default=2);parser.add_argument('--stage',choices=('all','prepare','summarize'),default='all')
    args=parser.parse_args();OUTPUT.mkdir(parents=True,exist_ok=True);seeds=(args.seed,) if args.seed else SEEDS
    for seed in seeds:prepare(seed)
    if args.stage=='prepare':return
    if args.stage=='all':
        if args.seed or args.workers==1:
            for seed in seeds:run_split(seed)
        else:
            errors=[]
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures={pool.submit(run_split,s):s for s in seeds}
                for future in as_completed(futures):
                    try:future.result();print(f'Split {futures[future]} complete',flush=True)
                    except Exception as e:errors.append(futures[future]);print(f'Split {futures[future]} failed: {e}',flush=True)
            if errors:raise RuntimeError(f'Incomplete splits {errors}; resume using --seed')
    if all((OUTPUT/f'split_{seed}'/'validation.json').exists() for seed in SEEDS):aggregate()

if __name__=='__main__':main()
