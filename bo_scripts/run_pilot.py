"""Resumable first-stage BO pilot. --steps specifies total completed BO steps.

Example: python bo_scripts/run_pilot.py --seed 0 --steps 5
Continue: python bo_scripts/run_pilot.py --seed 0 --steps 10
"""
import argparse
import csv
import hashlib
import json
import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
from pathlib import Path
import time
import numpy as np
from scipy import optimize
from scipy.stats import qmc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gh_bo_core import Posterior,Acquisition,objective,PRIOR_MEAN,PRIOR_SD
from inference import map_fit,sample

ROOT=Path(__file__).resolve().parents[1]


def initial_state(config):
    return {'completed_steps':0,'u':list(config['initial_u']),'y':list(config['initial_y']),'chain_endpoints':None,'timings':[]}


def write_json(path,data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf-8')
    os.replace(temp,path)


def reference_minimum():
    grid=np.linspace(0,1,20001); y=objective(grid)
    indices=np.flatnonzero((y[1:-1]<y[:-2])&(y[1:-1]<y[2:]))+1
    options=[(0.,float(y[0])),(1.,float(y[-1]))]
    for i in indices:
        fit=optimize.minimize_scalar(objective,bounds=(grid[i-1],grid[i+1]),method='bounded',options={'xatol':1e-14})
        options.append((float(fit.x),float(fit.fun)))
    u,value=min(options,key=lambda pair:pair[1])
    return {'u':u,'x':5+5*u,'value':value,'used_only_for_scoring':True}


def save_history(folder,state,reference):
    ys=np.asarray(state['y']); xs=np.asarray(state['u'])
    best=np.minimum.accumulate(ys)
    with (folder/'observations.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f); writer.writerow(['evaluation','bo_step','u','x','y','best_y','simple_regret'])
        for i,(u,y,b) in enumerate(zip(xs,ys,best)):
            writer.writerow([i+1,max(0,i-4),u,5+5*u,y,b,b-reference['value']])


def summary(folder,config):
    records=[]
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    grid=np.linspace(0,1,1500)
    axes[0].plot(5+5*grid,objective(grid),'k-',alpha=.45,label='Objective')
    for method in ['map','nuts','slice']:
        path=folder/method/'checkpoint.json'
        if not path.exists(): continue
        state=json.loads(path.read_text())
        y=np.asarray(state['y']); u=np.asarray(state['u']); steps=state['completed_steps']
        regret=np.minimum.accumulate(y)[4:]-config['reference']['value']
        axes[1].plot(np.arange(steps+1),regret,marker='o',label=method)
        axes[0].scatter(5+5*u[5:],y[5:],label=method,alpha=.8)
        records.append({'method':method,'completed_bo_steps':steps,'best_y':float(y.min()),'best_x':float(5+5*u[y.argmin()]),'simple_regret':float(regret[-1]),'inference_seconds':sum(r['inference_seconds'] for r in state['timings']),'acquisition_seconds':sum(r['acquisition_seconds'] for r in state['timings'])})
    axes[0].set(xlabel='x',ylabel='f(x)',title='BO queries (initial points omitted)')
    axes[1].set(xlabel='Completed BO steps',ylabel='Simple regret',title='Seed 0 pilot' if config['seed']==0 else f"Seed {config['seed']} pilot")
    for ax in axes: ax.legend(); ax.grid(alpha=.2)
    fig.tight_layout(); fig.savefig(folder/'comparison.png',dpi=160); plt.close(fig)
    if records:
        with (folder/'summary.csv').open('w',newline='') as f:
            writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    return records


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--steps',type=int,default=5)
    parser.add_argument('--methods',nargs='+',choices=['map','nuts','slice'],default=['map','nuts','slice'])
    parser.add_argument('--warmup',type=int,default=128)
    parser.add_argument('--samples',type=int,default=256)
    parser.add_argument('--chains',type=int,default=2)
    parser.add_argument('--output',type=Path,default=ROOT/'bo_figure'/'gh_fixed_mixing_pilot')
    args=parser.parse_args()
    if args.steps<0 or args.chains<2 or args.samples<4: parser.error('Need steps>=0, chains>=2, samples>=4')
    folder=args.output/f'seed_{args.seed:03d}';folder.mkdir(parents=True,exist_ok=True)
    hashes={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest() for name in ['gh_bo_core.py','inference.py','run_pilot.py']}
    config_path=folder/'experiment.json'
    if config_path.exists():
        config=json.loads(config_path.read_text())
        for key in ['warmup','samples','chains']:
            if config[key]!=getattr(args,key): raise ValueError(f'Resume requires original {key}={config[key]}')
        if config['source_sha256']!=hashes: raise ValueError('Source changed since checkpoint; use a new output folder to avoid mixing implementations')
    else:
        initial=np.sort(qmc.LatinHypercube(d=1,seed=args.seed).random(5).ravel())
        y=objective(initial)
        config={'schema_version':1,'seed':args.seed,'initial_u':initial.tolist(),'initial_y':y.tolist(),'output_center':float(y.mean()),'output_scale':float(max(y.std(),1e-8)),'lambda':1.,'omega':1.,'nugget':1e-6,'parameter_order':['mu','beta','log_v0','log_lengthscale'],'prior_mean':PRIOR_MEAN.tolist(),'prior_sd':PRIOR_SD.tolist(),'warmup':args.warmup,'samples':args.samples,'chains':args.chains,'acquisition_draws':32,'quadrature_nodes':96,'grid_points':1001,'xi':0.,'reference':reference_minimum(),'source_sha256':hashes,'resume_semantics':'--steps is the total BO budget, not additional steps. Per-step deterministic seeds; chain endpoints warm-start fresh adaptation. MAP is the mode in the listed transformed coordinates.','pilot_note':'Short chains for feasibility; evaluate diagnostics before scientific conclusions.'}
        write_json(config_path,config)
    write_json(folder/'requested_budget.json',{'total_bo_steps':args.steps,'methods':args.methods})
    for method in args.methods:
        dest=folder/method;dest.mkdir(exist_ok=True)
        checkpoint=dest/'checkpoint.json'
        state=json.loads(checkpoint.read_text()) if checkpoint.exists() else initial_state(config)
        save_history(dest,state,config['reference'])
        for step in range(state['completed_steps']+1,args.steps+1):
            start=time.perf_counter()
            y=(np.asarray(state['y'])-config['output_center'])/config['output_scale']
            posterior=Posterior(state['u'],y)
            update_seed=args.seed*100000+step*100+{'map':0,'nuts':10,'slice':20}[method]
            if method=='map':
                draws,diag=map_fit(posterior,np.random.default_rng(update_seed),state['chain_endpoints'])
                chains=draws[None,:,:];endpoints=draws
            else:
                chains,diag=sample(posterior,method,update_seed,state['chain_endpoints'],args.warmup,args.samples,args.chains)
                pool=chains.reshape(-1,4)
                indices=np.random.default_rng(update_seed+50).choice(len(pool),min(32,len(pool)),replace=False)
                draws=pool[indices];endpoints=chains[:,-1,:]
            inference_seconds=time.perf_counter()-start
            acquisition=Acquisition(posterior,draws)
            (u,ei),grid,values=acquisition.maximize()
            acquisition_seconds=time.perf_counter()-start-inference_seconds
            new_y=float(objective(u))
            iteration=dest/f'step_{step:03d}';iteration.mkdir(exist_ok=True)
            np.savez_compressed(iteration/'inference.npz',chains=chains,acquisition_draws=draws,train_u=state['u'],train_y=state['y'])
            np.savez_compressed(iteration/'acquisition.npz',u=grid,ei=values,selected_u=u)
            write_json(iteration/'diagnostics.json',diag)
            timing={'step':step,'inference_seed':update_seed,'inference_seconds':inference_seconds,'acquisition_seconds':acquisition_seconds,'selected_u':u,'selected_ei_standardized':ei,'observed_y':new_y}
            write_json(iteration/'selection.json',timing)
            state['u'].append(u);state['y'].append(new_y)
            state['completed_steps']=step;state['chain_endpoints']=endpoints.tolist();state['timings'].append(timing)
            write_json(checkpoint,state)
            save_history(dest,state,config['reference'])
            print(f'{method} step {step}: x={5+5*u:.6f}, y={new_y:.6f}, best={min(state["y"]):.6f}, inference={inference_seconds:.2f}s, EI={acquisition_seconds:.2f}s',flush=True)
            if method!='map': print('  Rhat:',diag['split_rhat'],'ESS:',diag['ess'],flush=True)
        summary(folder,config)
    print(json.dumps(summary(folder,config),indent=2),flush=True)


if __name__=='__main__': main()
