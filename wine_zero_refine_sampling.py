"""Refine slow Wine MH proposals; previous chains become discarded warmup.

The target/data/priors are unchanged. The fitted proposal is frozen before
new production draws, and independent/local moves each have exact MH tests.
Run: python wine_zero_refine_sampling.py --model gp_zero (or stp_zero).
The main experiment runner automatically resumes this stage once configured.
"""
import os
for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[variable]='1'
import argparse
import json
import time
import numpy as np
from scipy import linalg
from sklearn.mixture import GaussianMixture
from threadpoolctl import threadpool_limits
from GH_zero_tabular_bayes import ZeroPosterior,Proposal,sample_chain
from wine_zero_full_bayes_experiment import (OUTPUT,SOURCE,SETTINGS,MODELS,
    prepare,digest,diagnose,write_json)


def sample_refined(data,model):
    folder=OUTPUT/model
    report_path=folder/'sampling_report.json'
    if report_path.exists():
        report=json.loads(report_path.read_text())
        assert report['diagnostics_ok'] and report['samples_sha256']==digest(folder/'posterior_samples.npz')
        return report
    started=time.perf_counter()
    config_path=folder/'refinement.json'
    if not config_path.exists():
        mode=json.loads((folder/'initialization.json').read_text())
        center=np.array(mode['q_mode']);cov=np.array(mode['covariance'])
        L=linalg.cholesky(cov,lower=True)
        paths=[folder/f'chain{i+1}.npz' for i in range(4)]
        warmup=[];starts=[]
        for path in paths:
            with np.load(path) as saved:
                warmup.append(saved['q'][SETTINGS['frozen_warmup']:].copy())
                starts.append(saved['q'][-1].copy())
        pool=np.concatenate(warmup)
        whitened=linalg.solve_triangular(L,(pool-center).T,lower=True).T
        mixture=GaussianMixture(n_components=3,covariance_type='full',reg_covar=.05,
            n_init=3,max_iter=300,random_state=6421).fit(whitened)
        if not mixture.converged_:raise RuntimeError('Proposal fit did not converge')
        components=[dict(weight=float(.85*weight),mean=(center+L@mean).tolist(),
            shape=(L@shape@L.T).tolist(),df=None)
            for weight,mean,shape in zip(mixture.weights_,mixture.means_,mixture.covariances_)]
        components.append(dict(weight=.15,mean=center.tolist(),shape=(2*cov).tolist(),df=5.))
        empirical=.9*np.cov(pool.T)+.1*cov
        config=dict(reason='Initial independence chains mixed slowly in length-scale tails; discarded as extended warmup.',
            old_chain_hashes={p.name:digest(p) for p in paths},old_retained_lengths=[len(a) for a in warmup],
            components=components,local_probability=.2,local_covariance=(2.38**2/len(center)*empirical).tolist(),
            starts=np.array(starts),frozen=True,new_warmup=300,initial_draws=1000,extension_draws=1000,max_draws=6000,
            sampler='Fixed mixture of independence and symmetric random-walk Metropolis-Hastings kernels',
            seed=SETTINGS['seed']+900000+10000*MODELS.index(model),
            target_unchanged=True,dataset_sha256=digest(SOURCE/'dataset.npz'))
        write_json(config_path,config)
    config=json.loads(config_path.read_text())
    for name,expected in config['old_chain_hashes'].items():assert digest(folder/name)==expected
    assert config['dataset_sha256']==digest(SOURCE/'dataset.npz')
    proposal=Proposal(config['components'])
    train=data['train_ids']
    target=ZeroPosterior(data['y_scaled'][train],data['x_scaled'][train],model)
    identity=digest(config_path)+model+digest(OUTPUT/'experiment.json')
    burn=config['new_warmup'];draws=config['initial_draws']
    paths=[folder/f'refined_chain{i+1}.npz' for i in range(4)]
    for path in paths:
        if path.exists():
            with np.load(path) as saved:retained=max(0,len(saved['q'])-burn)
            draws=max(draws,int(np.ceil(retained/config['extension_draws']))*config['extension_draws'])
    while True:
        chains=[sample_chain(target.evaluate,proposal,np.array(config['starts'][i]),count=burn+draws,
            seed=config['seed']+100*i,checkpoint=path,identity=identity,label=f'{model} refined chain {i+1}',
            local_probability=config['local_probability'],local_covariance=config['local_covariance'])
            for i,path in enumerate(paths)]
        samples=np.stack([a['q'][burn:] for a in chains])
        quadratic=np.stack([a['quadratic'][burn:] for a in chains])
        summary,diagnostics,additional,names,W=diagnose(samples,quadratic,model,len(train))
        print(model+' refined: '+json.dumps(summary),flush=True)
        write_json(folder/'refined_diagnostic_progress.json',dict(draws_per_chain=draws,**summary))
        if summary['diagnostics_ok']:break
        if draws>=config['max_draws']:raise RuntimeError('Refined sampling diagnostics need further work')
        draws+=config['extension_draws']
    accepted=np.stack([a['accepted'][burn:] for a in chains])
    logp=np.stack([a['logp'][burn:] for a in chains])
    parameter_names=['v_0','k']+[f'w[{j}]' for j in range(target.d)]
    np.savez_compressed(folder/'posterior_samples.npz',log_theta=samples,quadratic=quadratic,
        W_samples=W,log_density=logp,accepted=accepted,parameter_names=np.array(parameter_names))
    report=dict(model=model,**summary,diagnostics=diagnostics,derived_diagnostics=additional,
        derived_parameter_names=names,chains=4,draws_per_chain=draws,total_retained_draws=4*draws,
        warmup_removed=True,warmup_per_refined_chain=burn,initial_chains_discarded_as_warmup=True,
        acceptance_per_chain=accepted.mean(axis=1),sampler=config['sampler'],
        exact_posterior_correction=True,W_integrated=model=='stp_zero',
        samples_sha256=digest(folder/'posterior_samples.npz'),proposal_sha256=digest(config_path),
        dataset_sha256=digest(SOURCE/'dataset.npz'),elapsed_seconds=time.perf_counter()-started)
    write_json(report_path,report)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=MODELS,required=True)
    args=parser.parse_args()
    with threadpool_limits(limits=SETTINGS['threads']):sample_refined(prepare(),args.model)
