"""Eight models x MAP/HMC/slice, with independent resumable BO trajectories."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from scipy.stats import qmc
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from multi_model_bo import Model,ModelAcquisition,infer,MODELS,LABELS,PRIOR_CENTER,PRIOR_SD
from gh_bo_core import objective
from run_pilot import ROOT,initial_state,write_json,reference_minimum,save_history

METHODS=('map','hmc','slice')


def clean(value):
    if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
    if isinstance(value,list):return [clean(v) for v in value]
    if isinstance(value,float) and not np.isfinite(value):return None
    return value


def summarize(folder,config):
    rows=[];fig,axes=plt.subplots(1,3,figsize=(15,4),sharey=True)
    for ax,method in zip(axes,METHODS):
        for name in MODELS:
            cp=folder/name/method/'checkpoint.json'
            if not cp.exists():continue
            state=json.loads(cp.read_text());y=np.asarray(state['y']);u=np.asarray(state['u'])
            regret=np.minimum.accumulate(y)[4:]-config['reference']['value']
            ax.plot(np.arange(len(regret)),regret,'o-',label=LABELS[name],markersize=3)
            rows.append(dict(model=name,method=method,completed_bo_steps=state['completed_steps'],
                best_y=float(y.min()),best_x=float(5+5*u[y.argmin()]),simple_regret=float(regret[-1]),
                inference_seconds=sum(t['inference_seconds'] for t in state['timings']),
                acquisition_seconds=sum(t['acquisition_seconds'] for t in state['timings'])))
        ax.set(title=method.upper(),xlabel='Completed BO steps');ax.grid(alpha=.2)
        if ax.lines:ax.legend(fontsize=7)
    axes[0].set_ylabel('Simple regret');fig.suptitle(f"Seed {config['seed']}: model comparison")
    fig.tight_layout();fig.savefig(folder/'comparison.png',dpi=160);plt.close(fig)
    if rows:
        with (folder/'summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--steps',type=int,default=5)
    p.add_argument('--models',nargs='+',choices=MODELS,default=list(MODELS))
    p.add_argument('--methods',nargs='+',choices=METHODS,default=list(METHODS))
    p.add_argument('--warmup',type=int,default=256);p.add_argument('--samples',type=int,default=512)
    p.add_argument('--chains',type=int,default=2);p.add_argument('--map-starts',type=int,default=8)
    p.add_argument('--target-accept',type=float,default=.95);p.add_argument('--acquisition-draws',type=int,default=64)
    p.add_argument('--output',type=Path,default=ROOT/'bo_figure'/'eight_model_comparison')
    args=p.parse_args()
    if args.seed<0 or args.steps<0 or args.warmup<50 or args.samples<8 or args.chains<2 or args.map_starts<2 or args.acquisition_draws<1 or not 0<args.target_accept<1:
        p.error('Need nonnegative seed/steps, warmup>=50, samples>=8, chains>=2, map-starts>=2, acquisition-draws>=1, 0<target-accept<1')
    settings={k:getattr(args,k) for k in ['seed','warmup','samples','chains','map_starts','target_accept','acquisition_draws']}
    names=['bo_scripts/run_model_comparison.py','bo_scripts/multi_model_bo.py','bo_scripts/beta_gp_bo.py',
           'bo_scripts/gh_bo_core.py','bo_scripts/run_pilot.py','GH_proc_joint_hmc.py','GP_TP_hmc.py',
           'GH_proc_w_betaprior_hmc.py','GH_proc_w_betaprior.py','GH_proc_w_betaprior_fit.py','GH_proc_w_betaprior_multid.py']
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
    folder=args.output/f'seed_{args.seed:03d}';folder.mkdir(exist_ok=True,parents=True);path=folder/'experiment.json'
    if path.exists():
        config=json.loads(path.read_text())
        if config['settings']!=settings or config['source_sha256']!=hashes:
            raise ValueError('Changed inference settings or source: use a new --output folder')
    else:
        x=np.sort(qmc.LatinHypercube(d=1,seed=args.seed).random(5).ravel());y=objective(x)
        config=dict(seed=args.seed,initial_u=x.tolist(),initial_y=y.tolist(),output_center=float(y.mean()),
            output_scale=float(max(y.std(),1e-8)),settings=settings,source_sha256=hashes,reference=reference_minimum(),
            models=list(MODELS),methods=list(METHODS),lambda_fixed=1.,omega_fixed=1.,nu_fixed=5.,kernel='SE',
            prior_center_two_kernels=PRIOR_CENTER.tolist(),prior_sd_log=PRIOR_SD,
            constant_beta_priors='log(v0,k,w) ~ N(log(.5,1e-4,25), 2^2); mu~N(0,2^2), beta~N(0,1)',
            constant_beta_model='f=mu+W*beta+sqrt(W)*g; infer both scalar mu and beta',
            varying_beta_model='f=W*beta(x)+sqrt(W)*g(x); independent zero-mean beta GP; mu=0',
            mu_models='f=mu(x)+sqrt(W)*g(x); independent zero-mean mu GP; beta=0; W=1 for GP',
            zero_models='gp_zero,stp_zero,ghp_zero: mu=beta=0, one residual SE kernel; log(v0,k,w) prior medians (.5,1e-4,25), SD 2. GP W=1, STP IG(2.5,1.5), GHP GIG(1,1,1).',
            map='Joint log(theta),log(W) MAP (no W for GP); fix conditional mean function, recompute W posterior. Constants remain fixed at their joint mode.',
            mcmc='GP latent functions analytically integrated; jointly sample theta,W (GP omits W). For constant beta and zero models, EI integrates W given theta.',
            comparison_note='MAP drops GP-function uncertainty as in regression plug-in experiments. This comparison is not solely a change of hyperparameter estimator.',
            legacy_note='Constant-beta model uses SE and inferred noise with joint-W MAP; it is NOT numerically identical to the older Matern/marginal-MAP pilot.',
            noise='Infer k under the same log-prior; no noise added to objective; conditional noise W*k (GP: k).',
            scaling='Fixed initial-data centering/scaling; zero mean statements refer to standardized coordinates.',
            acquisition='Latent f EI, xi=0; 96 log-W nodes when needed; 1001 grid + local refinement; exclude duplicates.',
            sampler='Adaptive HMC (not NUTS), randomized 12-30 leapfrog steps. Slice uses stepping-out/shrinkage. Both exchange kernel labels with posterior correction where applicable.',
            resume='Total --steps budget; independent per-model/method states; per-update deterministic seeds; fresh adaptation.')
        write_json(path,config)
    write_json(folder/'requested_budget.json',dict(steps=args.steps,models=args.models,methods=args.methods))
    for name in args.models:
        for method in args.methods:
            dest=folder/name/method;dest.mkdir(parents=True,exist_ok=True);cp=dest/'checkpoint.json'
            state=json.loads(cp.read_text()) if cp.exists() else initial_state(config)
            if len(state['u'])!=5+state['completed_steps']:raise ValueError('Inconsistent checkpoint')
            for step in range(state['completed_steps']+1,args.steps+1):
                start=time.perf_counter();y=(np.asarray(state['y'])-config['output_center'])/config['output_scale']
                model=Model(state['u'],y,name)
                seed=args.seed*1000000+MODELS.index(name)*10000+step*100+METHODS.index(method)*10
                chains,diag,extra=infer(model,method,seed,state['chain_endpoints'],settings)
                fit_time=time.perf_counter()-start;pool=chains.reshape(-1,chains.shape[-1])
                ids=np.random.default_rng(seed+50).choice(len(pool),min(args.acquisition_draws,len(pool)),replace=False)
                draws=pool[ids];acq=ModelAcquisition(model,draws,method)
                (u,ei),grid,values=acq.maximize();acq_time=time.perf_counter()-start-fit_time
                stepdir=dest/f'step_{step:03d}';stepdir.mkdir(exist_ok=True)
                np.savez_compressed(stepdir/'inference.npz',sampling_chains=chains,physical_chains=model.physical(chains),
                    parameter_names=model.names,acquisition_draws=draws,train_u=state['u'],train_y=state['y'],**extra)
                np.savez_compressed(stepdir/'acquisition.npz',u=grid,ei=values,selected_u=u)
                write_json(stepdir/'diagnostics.json',clean(diag))
                timing=dict(step=step,inference_seed=seed,inference_seconds=fit_time,acquisition_seconds=acq_time,
                            selected_u=u,observed_y=float(objective(u)),ei_standardized=ei)
                write_json(stepdir/'selection.json',timing)
                state['u'].append(u);state['y'].append(timing['observed_y']);state['completed_steps']=step
                state['chain_endpoints']=chains[:,-1,:].tolist();state['timings'].append(timing)
                write_json(cp,state);save_history(dest,state,config['reference'])
                fig,ax=plt.subplots(figsize=(7,3));ax.plot(5+5*grid,values);ax.axvline(5+5*u,color='red')
                ax.set(xlabel='x',ylabel='EI (standardized)',title=f'{LABELS[name]} / {method} / step {step}')
                fig.tight_layout();fig.savefig(stepdir/'acquisition.png',dpi=130);plt.close(fig)
                print(f'{name}/{method} step {step}: best={min(state["y"]):.6f}; fit {fit_time:.2f}s, EI {acq_time:.2f}s',flush=True)
                if method!='map':print('  diagnostics_ok:',diag['diagnostics_ok'],'max Rhat:',max(diag['rhat']),flush=True)
            summarize(folder,config)
    print(f'Results: {folder}',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
