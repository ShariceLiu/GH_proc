"""Independent per-cycle GP-prior mu/beta on the saved Hydraulic FS1 pilot.

python hydraulic_independent_plugin_experiment.py --stage prepare
python hydraulic_independent_plugin_experiment.py --model gp
python hydraulic_independent_plugin_experiment.py --stage summarize
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
BASELINE=ROOT/'figure'/'hydraulic_fs1_shared_plugin'
OUTPUT=ROOT/'figure'/'hydraulic_fs1_independent_plugin'
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
                  training_cycles=previous['training_cycles'],gap_cycles=previous['gap_cycles'],test_cycles=previous['test_cycles'],
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
            stem=case/f'cycle_{data["cycle_ids"][row]}'
            if stem.with_suffix('.json').exists() and stem.with_suffix('.npz').exists():
                record=json.loads(stem.with_suffix('.json').read_text())
                if record['fit_sha256']==fit_hash:rows.append(record['scores']);continue
            ids=np.sort(data['context_orders'][j,:count]);held=np.setdiff1d(np.arange(100),ids)
            function,mode=context_function(data['y'][row,ids]/scale,data['x'][ids],data['x'],theta,model)
            kwargs={f'{kind}_train':function[ids],f'{kind}_pred':function}
            pred=fixed_function_prediction(data['y'][row,ids]/scale,data['x'][ids],data['x'],
                                          v0=v,k=k,w=w,mixing=mixing,**kwargs)
            pred=original_units(pred,scale)
            scores,point,validation=score_prediction(pred,None,data['y'][row],held,check=j==0)
            scores.update(model=model,context_count=count,cycle_id=int(data['cycle_ids'][row]),n_test=len(held),
                          W_for_function=mode['W_for_function'],W_mean=pred['W_mean'],W_std=float(np.sqrt(pred['W_var'])))
            compact={key:value for key,value in pred.items() if not key.endswith('_cov')}
            compact.update(f_scale_var=np.diag(pred['f_scale_cov']),y_scale_var=np.diag(pred['y_scale_cov']))
            np.savez_compressed(stem.with_suffix('.npz'),**compact,**point,context_ids=ids,test_ids=held,
                                function_estimate_scaled=function,function_estimate_original=scale*function)
            write_json(stem.with_suffix('.json'),dict(scores=scores,validation=validation,context_mode=mode,fit_sha256=fit_hash))
            rows.append(scores)
        print(f'{model}: finished {count}-point contexts',flush=True)
    pd.DataFrame(rows).to_csv(folder/'per_curve_scores.csv',index=False)


def aggregate(curves):
    rows=[]
    for (count,model),frame in curves.groupby(['context_count','model'],sort=False):
        assert len(frame)==5
        row=dict(context_count=count,model=model,n_curves=len(frame),y_rmse=float(np.sqrt(np.mean(frame.y_rmse**2))))
        row.update({key:float(frame[key].mean()) for key in METRICS[1:]})
        for key in METRICS:
            row[key+'_curve_mean']=float(frame[key].mean());row[key+'_curve_sd']=float(frame[key].std(ddof=1))
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
    curves=pd.concat([pd.read_csv(OUTPUT/model/'per_curve_scores.csv') for model in MODELS],ignore_index=True)
    curves.to_csv(OUTPUT/'per_curve_scores.csv',index=False)
    summary=aggregate(curves);summary.to_csv(OUTPUT/'scores.csv',index=False)
    baseline=pd.read_csv(BASELINE/'scores.csv')
    comparison=summary.merge(baseline,on=['context_count','model'],suffixes=('_independent','_shared'),validate='one_to_one')
    for key in METRICS:comparison[key+'_change']=comparison[key+'_independent']-comparison[key+'_shared']
    comparison.to_csv(OUTPUT/'comparison_with_shared.csv',index=False)
    common=[]
    for model in MODELS:
        for count in COUNTS:
            for j,row in enumerate(data['test_rows']):
                with np.load(OUTPUT/model/f'context_{count}'/f'cycle_{data["cycle_ids"][row]}.npz') as p:
                    ids=np.setdiff1d(np.arange(100),data['context_orders'][j,:30]);pos=np.searchsorted(p['test_ids'],ids)
                    truth=data['y'][row,ids];ci=p['y_ci'][ids]
                    common.append(dict(model=model,context_count=count,cycle_id=int(data['cycle_ids'][row]),n_test=len(ids),
                        y_rmse=float(np.sqrt(np.mean((truth-p['y_mean'][ids])**2))),y_nlpd=float(p['y_nlpd'][pos].mean()),
                        y_crps=float(p['y_crps'][pos].mean()),y_coverage_95=float(np.mean((truth>=ci[:,0])&(truth<=ci[:,1]))),
                        y_interval_width=float(np.mean(ci[:,1]-ci[:,0]))))
    common=pd.DataFrame(common);common.to_csv(OUTPUT/'common_test_per_curve_scores.csv',index=False)
    aggregate(common).to_csv(OUTPUT/'common_test_scores.csv',index=False)
    print(summary[['context_count','model',*METRICS]].to_string(index=False),flush=True)
    names=dict(gp=r'GP ($\hat\mu_j$)',stp=r'STP ($\hat\mu_j$)',ghp_mu=r'GHP ($\hat\mu_j$)',ghp_beta=r'GHP ($\hat\beta_j$)')
    lines=[r'\begin{table}[htbp]',r'\centering',
           r'\caption{Hydraulic FS1: independent per-cycle plug-in functions; shared kernel/noise parameters. Twelve training cycles, two gap cycles and five test cycles. Scores evaluate held-out observed flow (L/min). RMSE is pooled; remaining metrics are averaged.}',
           r'\label{tab:hydraulic_fs1_independent}',r'\begin{tabular}{rlrrrr}',r'\toprule',
           r'Observed points & Model & RMSE & NLPD & CRPS & 95\% coverage (\%) \\',r'\midrule']
    for count in COUNTS:
        for model in MODELS:
            row=summary[(summary.context_count==count)&(summary.model==model)].iloc[0]
            lines.append(f'{count} & '+names[model]+f' & {row.y_rmse:.5f} & {row.y_nlpd:.5f} & {row.y_crps:.5f} & {100*row.y_coverage_95:.2f}'+r' \\')
        if count==15:lines.append(r'\midrule')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (OUTPUT/'comparison.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    t=data['time_seconds']
    with PdfPages(OUTPUT/'regression.pdf') as pdf:
        for count in COUNTS:
            for row in data['test_rows']:
                fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True,sharey=True)
                for ax,model in zip(axes.flat,MODELS):
                    relative=Path(model)/f'context_{count}'/f'cycle_{data["cycle_ids"][row]}.npz'
                    with np.load(OUTPUT/relative) as p,np.load(BASELINE/relative) as b:
                        ax.fill_between(t,*p['y_ci'].T,alpha=.25,label='Separate: 95% observation interval')
                        ax.plot(t,p['y_mean'],label='Separate: prediction')
                        ax.plot(t,b['y_mean'],color='C1',label='Shared: prediction')
                        ax.plot(t,data['y'][row],'k--',lw=1,label='Measured bin averages')
                        ids=p['context_ids'];ax.scatter(t[ids],data['y'][row,ids],s=14,c='black',label='Revealed points')
                        ax.set(title=LABELS[model],xlabel='Time within cycle (s)',ylabel='FS1 flow (L/min)')
                axes[0,0].legend(fontsize=7)
                fig.suptitle(f'Cycle {data["cycle_ids"][row]}, {count} observed points; plug-in functions')
                fig.tight_layout();pdf.savefig(fig)
                if row==data['test_rows'][0]:fig.savefig(OUTPUT/f'regression_context{count}.png',dpi=150)
                plt.close(fig)
    with PdfPages(OUTPUT/'estimated_functions.pdf') as pdf:
        for count in COUNTS:
            fig,axes=plt.subplots(2,2,figsize=(12,7),sharex=True)
            for ax,model in zip(axes.flat,MODELS):
                for row in data['test_rows']:
                    with np.load(OUTPUT/model/f'context_{count}'/f'cycle_{data["cycle_ids"][row]}.npz') as p:
                        ax.plot(t,p['function_estimate_original'],label=f'Cycle {data["cycle_ids"][row]}')
                shared=json.loads((BASELINE/model/'fit.json').read_text())
                ax.plot(t,shared['function_original'],'k--',label='Shared estimate')
                ax.set(title=LABELS[model],xlabel='Time within cycle (s)',ylabel='beta (L/min)' if model=='ghp_beta' else 'mu (L/min)')
            axes[0,0].legend(fontsize=8);fig.suptitle(f'Estimated functions from {count} context points')
            fig.tight_layout();pdf.savefig(fig);plt.close(fig)
    (OUTPUT/'method.txt').write_text(METHOD,encoding='utf-8')


METHOD='''INDEPENDENT FUNCTIONS: HYDRAULIC FS1 PILOT

Data and split
Use ../hydraulic_fs1_shared_plugin/dataset.npz unchanged (hash in experiment.json).
The same 19 stable, exactly condition-matched cycles: 12 train, 2 gap, 5 test.
Six adjacent readings are averaged into each of 100 bins. Training-derived common
scaling, no centering or per-curve normalization. Same nested random 15/30 contexts.

Models
GP: y_j = mu_j + g_j + epsilon_j.
STP/GHP mu: y_j = mu_j + sqrt(W_j)(g_j + epsilon_j).
GHP beta: y_j = W_j beta_j + sqrt(W_j)(g_j + epsilon_j).
g_j ~ GP(0,K); epsilon_j ~ N(0,kI). Independent latent mu_j/beta_j per cycle,
including previously unseen test cycles; all kernel/noise hyperparameters shared.
K and latent Sigma are SE kernels, v exp(-w (x-x')^2/2).
GP fixes W=1. STP fixes IG(shape=2.5, scale=1.5); GHP fixes GIG(lambda=1,chi=psi=1).
The STP baseline has an additional unscaled GP mean; it is not a pure Student-t process.

Priors and fitting
Match the previous shared model's marginal GP prior: h_j=mu_j, or h_j=c0 beta_j
where c0=E_prior[W]. Hence the beta variance log-prior center shifts by -2 log(c0).
Five independent Normal(log parameter; training-derived center, SD=2) priors.
Marginalize each latent GP separately. Covariance C_j=W_j(K+kI)+W_j^r Sigma,
r=0 for mu, r=2 for beta. There are NO latent cross-cycle covariance blocks.
Maximize the posterior density in LOG theta and LOG W_train coordinates; log-W
Jacobians are included. This is not MAP with respect to the linear-parameter measure.
The effective amplitude log-coordinate transform has determinant one.
24 starts cover short/long length scales and both component assignments. Select
the largest stationary posterior found, with gradient and local Hessian checks;
global optimality is not guaranteed.

Test-time plug-in procedure
Freeze shared theta. For each fresh cycle use ONLY its context observations y_c.
1. With its latent GP integrated out, maximize p(log W | y_c, theta).
2. At that W_hat, compute the conditional GP mean on the prediction grid:
   mu_hat = Sigma_*c C_c^-1 y_c;
   beta_hat = W_hat Sigma_*c C_c^-1 y_c.
3. Fix that function, then recompute p(W | y_c, theta, function_hat) from the
   ORIGINAL W prior and the fixed-function likelihood. Do not multiply the
   context-mode posterior into this density (which would count context twice).
4. Integrate this W for prediction. GIG posterior lambda=1-n/2,
   chi=1+(y_c-mu_c)' A^-1 (y_c-mu_c), psi=1+beta_c' A^-1 beta_c,
   A=K_cc+kI. IG posterior shape=(5+n)/2, scale=(3+residual' A^-1 residual)/2.
   Conditional prediction is N(m+W b, W S), with the usual residual GP
   conditioning m=mu_*+K_*c A^-1(y_c-mu_c), b=beta_*-K_*c A^-1 beta_c,
   S=K_**-K_*c A^-1 K_c*. Observation covariance adds W k I.

Interpretation
Each test curve estimates its own shape from sparse context; it cannot borrow the
training curves' learned shared shape. The training curves inform theta only.
This is a two-stage empirical-Bayes plug-in procedure, NOT full Bayesian inference.
Function estimation and hyperparameter uncertainty are omitted. Reusing the
context to estimate a fixed function can understate predictive uncertainty.
The context W_hat used for estimating the function and the subsequent conditional
W posterior have different targets and need not agree.

Scores and storage
Only observed bin-average y has ground truth. RMSE is pooled over held-out points;
NLPD/CRPS/coverage/interval width are averaged over equally-sized test curves.
Bands are equal-tail pointwise 95% observation intervals. Negative NLPD is valid
for continuous densities and depends on units. The primary 15/30 scores use 85/70
held-out points, respectively; common_test_scores.csv compares the SAME 70 points.
comparison_with_shared.csv compares identical held-out points and contexts with
the saved shared-function run. Per-curve CSVs retain between-curve variation.
Five adjacent test cycles are a small pilot, not independent evidence for ranking
models across hydraulic operating conditions. Compact NPZs retain marginal
predictions, estimated functions and scores, not dense covariance/sample arrays.
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
