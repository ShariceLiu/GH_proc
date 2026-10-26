"""Resumable varying-beta BO. --steps is TOTAL BO steps after five initial points."""
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
from beta_gp_bo import infer,BetaAcquisition,PRIOR_CENTER,PRIOR_SD,PARAMETERS
from gh_bo_core import objective
from run_pilot import ROOT,initial_state,write_json,reference_minimum,save_history


def summarize(folder,config):
    fig,axes=plt.subplots(1,2,figsize=(11,4));rows=[]
    grid=np.linspace(0,1,1500)
    axes[0].plot(5+5*grid,objective(grid),'k-',alpha=.4,label='Objective')
    axes[0].scatter(5+5*np.asarray(config['initial_u']),config['initial_y'],color='black',marker='x',label='Initial')
    for method in ['map','hmc','slice']:
        path=folder/method/'checkpoint.json'
        if not path.exists():continue
        state=json.loads(path.read_text());y=np.asarray(state['y']);u=np.asarray(state['u'])
        regret=np.minimum.accumulate(y)[4:]-config['reference']['value']
        axes[0].scatter(5+5*u[5:],y[5:],label=method)
        axes[1].plot(np.arange(len(regret)),regret,'o-',label=method)
        rows.append(dict(method=method,completed_bo_steps=state['completed_steps'],best_y=float(y.min()),
                         best_x=float(5+5*u[y.argmin()]),simple_regret=float(regret[-1]),
                         inference_seconds=sum(t['inference_seconds'] for t in state['timings']),
                         acquisition_seconds=sum(t['acquisition_seconds'] for t in state['timings'])))
    axes[0].set(xlabel='x',ylabel='f(x)',title='GP-prior beta: BO queries')
    axes[1].set(xlabel='Completed BO steps',ylabel='Simple regret',title=f"Seed {config['seed']}")
    for ax in axes:ax.legend();ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(folder/'comparison.png',dpi=160);plt.close(fig)
    if rows:
        with (folder/'summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=rows[0]);writer.writeheader();writer.writerows(rows)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--steps',type=int,default=5)
    parser.add_argument('--methods',nargs='+',choices=['map','hmc','slice'],default=['map','hmc','slice'])
    parser.add_argument('--warmup',type=int,default=256)
    parser.add_argument('--samples',type=int,default=512)
    parser.add_argument('--chains',type=int,default=2)
    parser.add_argument('--map-starts',type=int,default=8)
    parser.add_argument('--target-accept',type=float,default=.95)
    parser.add_argument('--acquisition-draws',type=int,default=64)
    parser.add_argument('--output',type=Path,default=ROOT/'bo_figure'/'gh_beta_gp_pilot')
    args=parser.parse_args()
    if args.seed<0 or args.steps<0 or args.warmup<50 or args.samples<8 or args.chains<2 or args.map_starts<2 or args.acquisition_draws<1 or not 0<args.target_accept<1:
        parser.error('Invalid sampler/budget settings: warmup>=50, samples>=8, chains>=2, map-starts>=2 required')
    folder=args.output/f'seed_{args.seed:03d}';folder.mkdir(parents=True,exist_ok=True)
    # Include transitive local scientific dependencies so resumes cannot mix implementations.
    names=['bo_scripts/run_beta_gp.py','bo_scripts/beta_gp_bo.py','bo_scripts/gh_bo_core.py','bo_scripts/run_pilot.py',
           'GH_proc_joint_hmc.py','GH_proc_w_betaprior_hmc.py','GH_proc_w_betaprior.py',
           'GH_proc_w_betaprior_fit.py','GH_proc_w_betaprior_multid.py']
    hashes={n:hashlib.sha256((ROOT/n).read_bytes()).hexdigest() for n in names}
    settings={key:getattr(args,key) for key in ['seed','warmup','samples','chains','map_starts','target_accept','acquisition_draws']}
    path=folder/'experiment.json'
    if path.exists():
        config=json.loads(path.read_text())
        if config['settings']!=settings or config['source_sha256']!=hashes:
            raise ValueError('Resume requires identical inference settings and source. Use --output for a different experiment.')
    else:
        initial=np.sort(qmc.LatinHypercube(d=1,seed=args.seed).random(5).ravel());y=objective(initial)
        config=dict(schema_version=1,seed=args.seed,initial_u=initial.tolist(),initial_y=y.tolist(),
                    output_center=float(y.mean()),output_scale=float(max(y.std(),1e-8)),
                    model='f=W beta + sqrt(W) g; beta and g independent zero-mean SE GPs',
                    mu_fixed=0,lambda_fixed=1,omega_fixed=1,
                    parameter_names=PARAMETERS,prior_center=PRIOR_CENTER.tolist(),prior_sd=PRIOR_SD,
                    settings=settings,source_sha256=hashes,reference=reference_minimum(),
                    observation_noise='Deterministic objective; infer W-scaled noise k as in reference model, with log-prior median 1e-4 in standardized units.',
                    map='Joint log(theta),log(W) mode with GPs integrated; fix conditional beta mean, recompute posterior W.',
                    hmc='Existing adaptive diagonal-metric HMC, randomized 12-30 leapfrog steps and component exchanges; not NUTS.',
                    slice='Coordinate-wise stepping-out/shrinkage plus posterior-corrected component exchanges.',
                    normalization='Center/scale fixed from initial observations; mu=0 is in standardized coordinates.',
                    prediction='Latent f EI, xi=0; 1001 candidate grid plus local refinement; duplicate exclusion 1e-5 in u.',
                    resume='Per-step seeds and effective-coordinate chain endpoints saved; fresh adaptation on each new dataset.',
                    pilot_note='Short pilot chains; inspect saved rank-normalized Rhat and bulk/tail ESS before drawing conclusions.')
        write_json(path,config)
    write_json(folder/'requested_budget.json',dict(steps=args.steps,methods=args.methods))
    for method in args.methods:
        dest=folder/method;dest.mkdir(exist_ok=True);cp=dest/'checkpoint.json'
        state=json.loads(cp.read_text()) if cp.exists() else initial_state(config)
        if len(state['u'])!=5+state['completed_steps']:raise ValueError('Invalid checkpoint')
        for step in range(state['completed_steps']+1,args.steps+1):
            start=time.perf_counter()
            y=(np.asarray(state['y'])-config['output_center'])/config['output_scale']
            seed=args.seed*100000+step*100+{'map':0,'hmc':10,'slice':20}[method]
            samples,diagnostic,extra=infer(state['u'],y,method,seed,state['chain_endpoints'],args.warmup,args.samples,args.chains,args.map_starts,args.target_accept)
            elapsed=time.perf_counter()-start
            pool=samples.reshape(-1,6)
            ids=np.random.default_rng(seed+50).choice(len(pool),min(args.acquisition_draws,len(pool)),replace=False)
            draws=pool[ids]
            acquisition=BetaAcquisition(state['u'],y,draws,method)
            (u,ei),grid,values=acquisition.maximize()
            acq_time=time.perf_counter()-start-elapsed
            stepdir=dest/f'step_{step:03d}';stepdir.mkdir(exist_ok=True)
            np.savez_compressed(stepdir/'inference.npz',effective_chains=samples,acquisition_draws=draws,train_u=state['u'],train_y=state['y'],**extra)
            np.savez_compressed(stepdir/'acquisition.npz',u=grid,ei=values,selected_u=u)
            # Convert nonfinite diagnostics to null rather than invalid JSON.
            def clean(value):
                if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
                if isinstance(value,list):return [clean(v) for v in value]
                if isinstance(value,float) and not np.isfinite(value):return None
                return value
            write_json(stepdir/'diagnostics.json',clean(diagnostic))
            new_y=float(objective(u))
            timing=dict(step=step,inference_seed=seed,inference_seconds=elapsed,acquisition_seconds=acq_time,
                        selected_u=u,observed_y=new_y,ei_standardized=ei)
            write_json(stepdir/'selection.json',timing)
            state['u'].append(u);state['y'].append(new_y);state['completed_steps']=step
            state['chain_endpoints']=samples[:,-1,:].tolist();state['timings'].append(timing)
            write_json(cp,state);save_history(dest,state,config['reference'])
            fig,ax=plt.subplots(figsize=(7,3));ax.plot(5+5*grid,values);ax.axvline(5+5*u,color='red')
            ax.set(xlabel='x',ylabel='EI (standardized)',title=f'{method}: step {step}');fig.tight_layout()
            fig.savefig(stepdir/'acquisition.png',dpi=130);plt.close(fig)
            print(f'{method} step {step}: x={5+5*u:.6f}, y={new_y:.6f}, best={min(state["y"]):.6f}; fit {elapsed:.2f}s, EI {acq_time:.2f}s',flush=True)
            if method!='map':print('  diagnostics_ok:',diagnostic['diagnostics_ok'],'max Rhat:',max(diagnostic['rhat']),flush=True)
        summarize(folder,config)
    print(f'Results: {folder}',flush=True)


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
