"""Full Bayesian GP/STP, zero functions, on the saved 60% Wine split.

python wine_zero_full_bayes_experiment.py --stage sample --model gp_zero
python wine_zero_full_bayes_experiment.py --stage sample --model stp_zero
python wine_zero_full_bayes_experiment.py --stage all

Exact likelihoods, MH posterior sampling, no covariance approximation.
Sampling checkpoints include RNG state and can be resumed or extended.
"""
import os
for variable in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[variable]='1'
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd
from scipy import linalg,optimize,stats,special
from threadpoolctl import threadpool_limits

from GH_zero_tabular_bayes import (ZeroPosterior,Proposal,sample_chain,mixture_logpdf,
    mixture_cdf,mixture_quantile,mixture_crps,independent_crps)
from GH_proc_w_betaprior_hmc import chain_diagnostics
from shared_beta_plugin_experiment import write_json

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'figure'/'tabular_single_function_plugin_train60'/'wine'
OUTPUT=ROOT/'figure'/'wine_train60_zero_full_bayes'
MODELS=('gp_zero','stp_zero')
SETTINGS=dict(chains=4,pilot=200,frozen_warmup=150,initial_draws=750,
    extension_draws=750,max_draws=4500,seed=22026,prior_sd=2.,nu=5.,
    rhat_limit=1.01,min_ess=400,threads=4)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def prepare():
    OUTPUT.mkdir(exist_ok=True)
    with np.load(SOURCE/'dataset.npz') as saved:
        data={k:saved[k].copy() for k in saved.files}
    config=json.loads((SOURCE/'experiment.json').read_text())
    assert len(data['train_ids'])==2376 and len(data['test_ids'])==1585
    assert data['x'].shape==(3961,11)
    assert config['dataset_sha256']==digest(SOURCE/'dataset.npz')
    train,test=data['train_ids'],data['test_ids']
    np.testing.assert_array_equal(np.sort(np.r_[train,test]),np.arange(len(data['y'])))
    assert not np.intersect1d(data['input_group_ids'][train],data['input_group_ids'][test]).size
    np.testing.assert_allclose(data['x_scaled'],(data['x']-data['x'][train].mean(0))/data['x'][train].std(0))
    np.testing.assert_allclose(data['y_scaled'],(data['y']-data['y'][train].mean())/data['y'][train].std())
    paths=['dataset.npz','experiment.json','scores_with_zero_mean.csv','comparison_with_zero_mean.tex']
    paths += [str(p.relative_to(SOURCE)) for p in SOURCE.rglob('*')
              if p.name in ('fit.json','report.json','predictions.npz','validation.json')]
    hashes={p:digest(SOURCE/p) for p in paths}
    provenance=OUTPUT/'source_provenance.json'
    if provenance.exists():
        assert json.loads(provenance.read_text())==hashes,'Previous source results changed'
    else:write_json(provenance,hashes)
    experiment=dict(dataset='wine',source=str(SOURCE.relative_to(ROOT)),
        dataset_sha256=config['dataset_sha256'],n_train=2376,n_test=1585,split_seed=config['split_seed'],
        models=list(MODELS),settings=SETTINGS,
        mean='mu=beta=0 after fixed training-only centering; original-unit intercept retained.',
        covariance='One ARD SE kernel and iid noise k; 13 kernel/noise parameters.',
        prior='Independent N(default_center,2^2) on physical log kernel/noise parameters; same as MAP baselines.',
        sampler='Independence Metropolis-Hastings with warmup-fitted defensive Normal/Student-t proposals and exact posterior correction; not HMC.',
        W='GP fixes W=1; STP fixes nu=5, W~IG(2.5,1.5), analytically marginalized in training and prediction.',
        prediction='All retained draws, including repeated MH states. Gaussian mixture for GP; Student-t mixture with df=2381 for STP.',
        storage='No dataset copy, dense test covariance or saved prediction-component arrays. Compact samples and pointwise predictions only.')
    path=OUTPUT/'experiment.json'
    if path.exists():assert json.loads(path.read_text())==experiment
    else:write_json(path,experiment)
    (OUTPUT/'method.txt').write_text(METHOD,encoding='utf-8')
    print('Wine full Bayes: 2376 train / 1585 test; same source and preprocessing.',flush=True)
    return data


def find_mode(target,folder,model):
    path=folder/'initialization.json'
    if path.exists():return json.loads(path.read_text())
    old=json.loads((SOURCE/'zero_mean_baselines'/model/'fit.json').read_text())
    q=np.asarray(old['log_parameters'][:target.d+2])
    def objective(z):
        value,gradient=target.target(z)
        return -value,-gradient
    if np.max(abs(target.target(q)[1]))>=1e-5:
        result=optimize.minimize(objective,q,jac=True,method='L-BFGS-B',
            options=dict(maxiter=1500,gtol=1e-8,ftol=1e-14,maxls=40))
        q=result.x
        for _ in range(3):
            if np.max(abs(target.target(q)[1]))<1e-5:break
            result=optimize.minimize(objective,q,jac=True,method='L-BFGS-B',
                options=dict(maxiter=500,gtol=1e-9,ftol=1e-15,maxls=40))
            q=result.x
    value,gradient=target.target(q)
    if max(abs(gradient))>=1e-4:raise RuntimeError('Initial posterior mode not stationary')
    h=1e-4
    differences=[(target.target(q+e*h),target.target(q-e*h)) for e in np.eye(len(q))]
    numerical=np.array([(a[0]-b[0])/(2*h) for a,b in differences])
    hessian=np.column_stack([(a[1]-b[1])/(2*h) for a,b in differences])
    hessian=(hessian+hessian.T)/2
    np.testing.assert_allclose(numerical,gradient,atol=2e-5,rtol=1e-4)
    if np.linalg.eigvalsh(hessian).max()>=0:raise RuntimeError('Nonnegative mode curvature')
    covariance=linalg.inv(-hessian)
    result=dict(q_mode=q,logp=value,covariance=covariance,prior_center=target.center,
        max_abs_gradient=float(max(abs(gradient))),max_hessian_eigenvalue=float(np.linalg.eigvalsh(hessian).max()),
        gradient_check_max_error=float(max(abs(numerical-gradient))),
        role='Proposal initialization only; posterior inference uses retained MH draws.')
    write_json(path,result)
    return result


def diagnose(samples,quadratic,model,n):
    diagnostics=chain_diagnostics(samples)
    q=samples
    derived_names=[];derived=[]
    if model=='stp_zero':
        rng=np.random.default_rng(86229)
        shape=(SETTINGS['nu']+n)/2
        scale=(SETTINGS['nu']-2+quadratic)/2
        W=scale/rng.gamma(shape,size=quadratic.shape)
        derived=[np.log(W),q[:,:,0]+np.log(W),q[:,:,1]+np.log(W)]
        derived_names=['log_W','log_Wv','log_Wk']
    else:W=np.ones(quadratic.shape)
    additional=chain_diagnostics(np.stack(derived,axis=-1)) if derived else None
    all_diagnostics=[diagnostics]+([additional] if additional is not None else [])
    summary=dict(max_rhat=max(float(np.max(d['rhat'])) for d in all_diagnostics),
        min_bulk_ess=min(float(np.min(d['ess_bulk'])) for d in all_diagnostics),
        min_tail_ess=min(float(np.min(d['ess_tail'])) for d in all_diagnostics))
    summary['diagnostics_ok']=bool(summary['max_rhat']<SETTINGS['rhat_limit'] and
        min(summary['min_bulk_ess'],summary['min_tail_ess'])>=SETTINGS['min_ess'])
    return summary,diagnostics,additional,derived_names,W


def sample_model(data,model):
    folder=OUTPUT/model;folder.mkdir(exist_ok=True)
    if (folder/'refinement.json').exists():
        from wine_zero_refine_sampling import sample_refined
        return sample_refined(data,model)
    sampling_report=folder/'sampling_report.json'
    if sampling_report.exists():
        report=json.loads(sampling_report.read_text())
        if report['diagnostics_ok']:
            assert report['samples_sha256']==digest(folder/'posterior_samples.npz')
            print(model+': reusing validated posterior samples',flush=True)
            return report
    train=data['train_ids'];start_time=time.perf_counter()
    target=ZeroPosterior(data['y_scaled'][train],data['x_scaled'][train],model)
    mode=find_mode(target,folder,model)
    qmode,cov=np.asarray(mode['q_mode']),np.asarray(mode['covariance'])
    pilot_proposal=Proposal([dict(weight=1.,mean=qmode.tolist(),shape=cov.tolist(),df=7.)])
    pilot=[]
    mode_hash=digest(folder/'initialization.json')
    identity_base=digest(OUTPUT/'experiment.json')+model+mode_hash
    model_index=MODELS.index(model)
    for chain in range(SETTINGS['chains']):
        seed=SETTINGS['seed']+10000*model_index+100*chain
        rng=np.random.default_rng(seed+700000)
        initial=qmode+2*linalg.cholesky(cov,lower=True)@rng.normal(size=len(qmode))
        pilot.append(sample_chain(target.evaluate,pilot_proposal,initial,count=SETTINGS['pilot'],seed=seed,
            checkpoint=folder/f'pilot_chain{chain+1}.npz',identity=identity_base+'pilot',
            label=f'{model} pilot chain {chain+1}'))
    proposal_path=folder/'proposal.json'
    if proposal_path.exists():
        components=json.loads(proposal_path.read_text())['components']
    else:
        pool=np.concatenate([a['q'][SETTINGS['pilot']//2:] for a in pilot])
        center=pool.mean(0)
        empirical=np.cov(pool.T)
        proposal_cov=1.1*(.9*empirical+.1*cov)
        components=[dict(weight=.8,mean=center.tolist(),shape=proposal_cov.tolist(),df=None),
                    dict(weight=.2,mean=qmode.tolist(),shape=(2*cov).tolist(),df=5.)]
        write_json(proposal_path,dict(components=components,pilot_draws_for_fit=len(pool),
            frozen_before_retention=True,pilot_acceptance=[float(a['accepted'].mean()) for a in pilot]))
    proposal=Proposal(components)
    identity=identity_base+digest(proposal_path)
    draws=SETTINGS['initial_draws']
    for chain in range(SETTINGS['chains']):
        path=folder/f'chain{chain+1}.npz'
        if path.exists():
            with np.load(path) as saved:
                retained=max(0,len(saved['q'])-SETTINGS['frozen_warmup'])
            draws=max(draws,int(np.ceil(retained/SETTINGS['extension_draws']))*SETTINGS['extension_draws'])
    while True:
        chains=[]
        for chain in range(SETTINGS['chains']):
            seed=SETTINGS['seed']+10000*model_index+100*chain+500000
            chains.append(sample_chain(target.evaluate,proposal,pilot[chain]['q'][-1],
                count=SETTINGS['frozen_warmup']+draws,seed=seed,
                checkpoint=folder/f'chain{chain+1}.npz',identity=identity,
                label=f'{model} chain {chain+1}'))
        burn=SETTINGS['frozen_warmup']
        samples=np.stack([a['q'][burn:] for a in chains])
        quadratic=np.stack([a['quadratic'][burn:] for a in chains])
        summary,diagnostics,additional,names,W=diagnose(samples,quadratic,model,len(train))
        print(model+': '+json.dumps(summary),flush=True)
        if summary['diagnostics_ok']:break
        if draws>=SETTINGS['max_draws']:
            write_json(folder/'diagnostics_incomplete.json',dict(draws=draws,**summary))
            raise RuntimeError(f'{model}: sampling diagnostics need further work')
        draws+=SETTINGS['extension_draws']
    accepted=np.stack([a['accepted'][burn:] for a in chains])
    logp=np.stack([a['logp'][burn:] for a in chains])
    parameter_names=['v_0','k']+[f'w[{j}]' for j in range(target.d)]
    np.savez_compressed(folder/'posterior_samples.npz',log_theta=samples,quadratic=quadratic,
        W_samples=W,log_density=logp,accepted=accepted,parameter_names=np.array(parameter_names))
    report=dict(model=model,**summary,diagnostics=diagnostics,derived_diagnostics=additional,
        derived_parameter_names=names,chains=SETTINGS['chains'],draws_per_chain=draws,
        total_retained_draws=samples.shape[0]*samples.shape[1],warmup_removed=True,
        acceptance_per_chain=accepted.mean(axis=1),sampler='Independence Metropolis-Hastings',
        exact_posterior_correction=True,W_integrated=model=='stp_zero',
        samples_sha256=digest(folder/'posterior_samples.npz'),proposal_sha256=digest(proposal_path),
        dataset_sha256=digest(SOURCE/'dataset.npz'),elapsed_seconds=time.perf_counter()-start_time)
    write_json(sampling_report,report)
    return report


def summarize_parameters(folder,data,samples,quadratic,W,report):
    names=['v_0','k']+[f'w[{j}]' for j in range(data['x'].shape[1])]
    arrays=np.exp(samples)
    diagnostics=report['diagnostics']
    if folder.name=='stp_zero':
        arrays=np.concatenate([arrays,W[:,:,None]],axis=2);names+=['W']
        diagnostics={k:list(v)+[report['derived_diagnostics'][k][0]] for k,v in diagnostics.items()}
    factors=np.r_[np.full(2,float(data['y_scale'])**2),1/data['x_scale']**2,
                   [1.] if folder.name=='stp_zero' else []]
    flat=arrays.reshape(-1,arrays.shape[-1]);rows=[]
    for j,name in enumerate(names):
        lower,median,upper=np.quantile(flat[:,j],[.025,.5,.975])
        mean=float(flat[:,j].mean());variance=float(flat[:,j].var())
        if name=='W':
            shape=(SETTINGS['nu']+len(data['train_ids']))/2
            rate=(SETTINGS['nu']-2+quadratic)/2
            means=rate/(shape-1)
            mean=float(means.mean())
            variance=float(np.mean(rate**2/((shape-1)**2*(shape-2))+(means-mean)**2))
        rows.append(dict(parameter=name,mean=mean,variance=variance,std=np.sqrt(variance),
                         lower=float(lower),median=float(median),upper=float(upper),
                         **{k:float(v[j]) for k,v in diagnostics.items()}))
    pd.DataFrame(rows).to_csv(folder/'parameters_standardized.csv',index=False)
    original=[]
    for row,factor in zip(rows,factors):
        converted=dict(row)
        for name in ('mean','std','lower','median','upper'):converted[name]*=factor
        converted['variance']*=factor**2
        original.append(converted)
    pd.DataFrame(original).to_csv(folder/'parameters_original_units.csv',index=False)


def check_conditionals(data,model,unique,component_mean,component_f,component_y,positions):
    """Rebuild selected dense conditioning systems without the fitted helper."""
    train=data['train_ids'];test=data['test_ids'][positions]
    x=data['x_scaled'][train];xp=data['x_scaled'][test];y=data['y_scaled'][train]
    def kernel(left,right,v,w):
        distances=np.zeros((len(left),len(right)))
        for j,value in enumerate(w):distances+=value*(left[:,j,None]-right[None,:,j])**2
        return v*np.exp(-.5*distances)
    checks=[]
    for i in np.unique(np.linspace(0,len(unique)-1,3,dtype=int)):
        v,k,*weights=np.exp(unique[i]);w=np.array(weights)
        A=kernel(x,x,v,w)+k*np.eye(len(x))
        cross=kernel(xp,x,v,w)
        solved=linalg.solve(A,cross.T,assume_a='pos')
        alpha=linalg.solve(A,y,assume_a='pos')
        mean=data['y_mean']+data['y_scale']*(cross@alpha)
        residual=v-np.sum(cross*solved.T,axis=1)
        factor=1. if model=='gp_zero' else (SETTINGS['nu']-2+y@alpha)/(SETTINGS['nu']+len(y))
        np.testing.assert_allclose(mean,component_mean[i,positions],atol=2e-8)
        np.testing.assert_allclose(data['y_scale']**2*factor*residual,component_f[i,positions],atol=2e-8)
        np.testing.assert_allclose(data['y_scale']**2*factor*(residual+k),component_y[i,positions],atol=2e-8)
        checks.append(dict(component=int(i),test_positions=positions.tolist(),passed=True))
    return checks


def predict_model(data,model):
    folder=OUTPUT/model
    fit=json.loads((folder/'sampling_report.json').read_text())
    assert fit['diagnostics_ok'] and fit['samples_sha256']==digest(folder/'posterior_samples.npz')
    if (folder/'report.json').exists():
        report=json.loads((folder/'report.json').read_text())
        assert report['samples_sha256']==fit['samples_sha256']
        assert report['predictions_sha256']==digest(folder/'predictions.npz')
        work=folder/'prediction_work.npz'
        if work.exists():
            assert work.resolve().parent==folder.resolve()
            work.unlink()
        print(model+': reusing validated predictions',flush=True)
        return report
    started=time.perf_counter()
    with np.load(folder/'posterior_samples.npz') as archive:
        samples=archive['log_theta'];quadratic=archive['quadratic'];W=archive['W_samples'];logp=archive['log_density']
    summarize_parameters(folder,data,samples,quadratic,W,fit)
    flat=samples.reshape(-1,samples.shape[-1])
    unique,first,inverse,counts=np.unique(flat,axis=0,return_index=True,return_inverse=True,return_counts=True)
    weights=counts/len(flat)
    assert counts.sum()==len(flat)
    n,m=len(unique),len(data['test_ids'])
    means,fscale,yscale=(np.empty((n,m)) for _ in range(3))
    checkpoint=folder/'prediction_work.npz'
    start=0
    if checkpoint.exists():
        with np.load(checkpoint) as saved:
            assert str(saved['samples_sha256'])==fit['samples_sha256']
            start=len(saved['means'])
            means[:start],fscale[:start],yscale[:start]=saved['means'],saved['fscale'],saved['yscale']
    train=data['train_ids'];test=data['test_ids'];truth=data['y'][test]
    target=ZeroPosterior(data['y_scaled'][train],data['x_scaled'][train],model)
    for index in range(start,n):
        p=target.predict(unique[index],data['x_scaled'][test],
                         output_mean=float(data['y_mean']),output_scale=float(data['y_scale']),batch_size=len(test))
        np.testing.assert_allclose(p['quadratic'],quadratic.ravel()[first[index]],rtol=1e-9,atol=1e-8)
        np.testing.assert_allclose(p['logp'],logp.ravel()[first[index]],rtol=0,atol=1e-7)
        means[index],fscale[index],yscale[index]=p['mean'],p['f_scale2'],p['y_scale2']
        if (index+1)%100==0 or index+1==n:
            temp=folder/'prediction_work.tmp.npz'
            np.savez_compressed(temp,samples_sha256=np.array(fit['samples_sha256']),
                means=means[:index+1],fscale=fscale[:index+1],yscale=yscale[:index+1])
            temp.replace(checkpoint)
            print(f'{model}: conditional predictions {index+1}/{n} unique states ({len(flat)} retained draws)',flush=True)
    del target
    df=None if model=='gp_zero' else SETTINGS['nu']+len(train)
    variance_factor=1. if df is None else df/(df-2)
    mean=weights@means
    between=weights@(means-mean)**2
    output=dict(test_ids=test,f_mean=mean,y_mean=mean,
        f_within_var=variance_factor*(weights@fscale),y_within_var=variance_factor*(weights@yscale),
        between_var=between)
    output['f_var']=output['f_within_var']+between
    output['y_var']=output['y_within_var']+between
    for component,key in ((fscale,'f'),(yscale,'y')):
        independently=weights@(variance_factor*component+means**2)-mean**2
        np.testing.assert_allclose(output[key+'_var'],independently,atol=2e-12,rtol=1e-10)
    output.update(f_ci=np.empty((m,2)),y_ci=np.empty((m,2)),nlpd=np.empty(m),crps=np.empty(m),pit=np.empty(m))
    chainweights=np.array([np.bincount(chain,minlength=n)/len(chain) for chain in inverse.reshape(samples.shape[:2])])
    chainmean=chainweights@means
    chainlogp=np.empty((len(chainweights),m));integration=[]
    for start in range(0,m,16):
        sl=slice(start,start+16);cm=means[:,sl];cs=np.sqrt(yscale[:,sl])
        for name,component in (('f',fscale),('y',yscale)):
            sd=np.sqrt(component[:,sl])
            output[name+'_ci'][sl]=np.column_stack([mixture_quantile(p,cm,sd,weights,df) for p in (.025,.975)])
            ci=output[name+'_ci'][sl]
            np.testing.assert_allclose(mixture_cdf(ci[:,0],cm,sd,weights,df),.025,atol=1e-10)
            np.testing.assert_allclose(mixture_cdf(ci[:,1],cm,sd,weights,df),.975,atol=1e-10)
        output['nlpd'][sl]=-mixture_logpdf(truth[sl],cm,cs,weights,df)
        output['pit'][sl]=mixture_cdf(truth[sl],cm,cs,weights,df)
        output['crps'][sl],info=mixture_crps(truth[sl],cm,cs,weights,df)
        integration.append(info)
        for chain,cw in enumerate(chainweights):
            chainlogp[chain,sl]=mixture_logpdf(truth[sl],cm,cs,cw,df)
        if start==0 or (start//16+1)%20==0:
            print(f'{model}: scored {min(start+16,m)}/{m} held-out observations',flush=True)
    positions=np.unique(np.linspace(0,m-1,7,dtype=int))
    conditioning=check_conditionals(data,model,unique,means,fscale,yscale,positions)
    checks=[]
    for j in positions:
        reference,error=independent_crps(truth[j],means[:,j],np.sqrt(yscale[:,j]),weights,df)
        np.testing.assert_allclose(reference,output['crps'][j],atol=5e-8,rtol=1e-8)
        # Independent density calls and a direct law-of-total-variance check.
        density=(stats.norm.logpdf(truth[j],means[:,j],np.sqrt(yscale[:,j])) if df is None else
                 stats.t.logpdf(truth[j],df,loc=means[:,j],scale=np.sqrt(yscale[:,j])))
        reference_logp=special.logsumexp(np.log(weights)+density)
        np.testing.assert_allclose(-reference_logp,output['nlpd'][j],atol=1e-10)
        checks.append(dict(test_position=int(j),crps_error=abs(reference-output['crps'][j]),adaptive_error=error))
    output['covered']=(truth>=output['y_ci'][:,0])&(truth<=output['y_ci'][:,1])
    for key,value in output.items():assert np.isfinite(value).all(),key
    np.savez_compressed(folder/'predictions.npz',**output)
    scores=dict(y_rmse=float(np.sqrt(np.mean((truth-mean)**2))),y_nlpd=float(output['nlpd'].mean()),
        y_crps=float(output['crps'].mean()),y_coverage_95=float(output['covered'].mean()),
        y_interval_width=float(np.mean(output['y_ci'][:,1]-output['y_ci'][:,0])))
    per_chain=[dict(chain=i+1,y_rmse=float(np.sqrt(np.mean((truth-cm)**2))),
                    y_nlpd=float(-cp.mean())) for i,(cm,cp) in enumerate(zip(chainmean,chainlogp))]
    pd.DataFrame(per_chain).to_csv(folder/'per_chain_scores.csv',index=False)
    report=dict(model=model,**scores,**{key:fit[key] for key in ('max_rhat','min_bulk_ess','min_tail_ess','diagnostics_ok',
        'chains','draws_per_chain','total_retained_draws','samples_sha256','acceptance_per_chain','dataset_sha256')},
        n_train=len(train),n_test=len(test),unique_prediction_states=n,all_retained_draws_used=True,
        component_distribution='Normal' if df is None else 'Student-t',component_df=df,
        validation_passed=True,conditioning_checks=conditioning,score_checks=checks,crps_integration=integration,
        per_chain_scores=per_chain,predictions_sha256=digest(folder/'predictions.npz'),
        prediction_seconds=time.perf_counter()-started)
    write_json(folder/'report.json',report)
    # Only this reconstructible work file is removed; retained samples remain.
    assert checkpoint.resolve().parent==folder.resolve() and checkpoint.name=='prediction_work.npz'
    checkpoint.unlink()
    print(model+': '+json.dumps(scores),flush=True)
    return report


def validate_saved_results(data):
    """Recheck saved arrays, diagnostic aggregates, observation noise and scores."""
    results=[]
    for model in MODELS:
        folder=OUTPUT/model
        report=json.loads((folder/'report.json').read_text())
        sampling=json.loads((folder/'sampling_report.json').read_text())
        assert report['samples_sha256']==sampling['samples_sha256']==digest(folder/'posterior_samples.npz')
        assert report['predictions_sha256']==digest(folder/'predictions.npz')
        assert report['dataset_sha256']==digest(SOURCE/'dataset.npz')
        with np.load(folder/'posterior_samples.npz') as s:
            q=s['log_theta'];Q=s['quadratic'];W=s['W_samples']
        diagnostics,_,_,_,expected_W=diagnose(q,Q,model,len(data['train_ids']))
        assert diagnostics['diagnostics_ok']
        for key in ('max_rhat','min_bulk_ess','min_tail_ess'):
            np.testing.assert_allclose(diagnostics[key],report[key],rtol=1e-12,atol=1e-12)
        np.testing.assert_array_equal(W,expected_W)
        assert report['total_retained_draws']==q.shape[0]*q.shape[1]
        unique=np.unique(q.reshape(-1,q.shape[-1]),axis=0)
        assert report['unique_prediction_states']==len(unique)
        with np.load(folder/'predictions.npz') as p:
            np.testing.assert_array_equal(p['test_ids'],data['test_ids'])
            truth=data['y'][p['test_ids']]
            np.testing.assert_array_equal(p['f_mean'],p['y_mean'])
            for name in ('f','y'):
                np.testing.assert_allclose(p[name+'_var'],p[name+'_within_var']+p['between_var'],atol=1e-12)
                assert (p[name+'_var']>0).all() and (p[name+'_ci'][:,1]>p[name+'_ci'][:,0]).all()
            conditional_EW=1. if model=='gp_zero' else (SETTINGS['nu']-2+Q)/(SETTINGS['nu']+len(data['train_ids'])-2)
            noise=float(np.mean(conditional_EW*np.exp(q[:,:,1])))*float(data['y_scale'])**2
            np.testing.assert_allclose(p['y_var']-p['f_var'],noise,atol=1e-10,rtol=1e-10)
            actual=dict(y_rmse=float(np.sqrt(np.mean((truth-p['y_mean'])**2))),
                y_nlpd=float(p['nlpd'].mean()),y_crps=float(p['crps'].mean()),
                y_coverage_95=float(np.mean((truth>=p['y_ci'][:,0])&(truth<=p['y_ci'][:,1]))),
                y_interval_width=float(np.mean(p['y_ci'][:,1]-p['y_ci'][:,0])))
            for key,value in actual.items():np.testing.assert_allclose(report[key],value,atol=1e-12)
        results.append(dict(model=model,checks_passed=True,posterior_mean_noise_variance=noise))
    return results


def summarize(data):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    validation=validate_saved_results(data)
    results=[];comparison=[]
    for model in MODELS:
        report=json.loads((OUTPUT/model/'report.json').read_text())
        assert report['diagnostics_ok'] and report['validation_passed']
        assert report['predictions_sha256']==digest(OUTPUT/model/'predictions.npz')
        results.append(report)
        for method,folder in [('MAP',SOURCE/'zero_mean_baselines'),('Full Bayes',OUTPUT)]:
            row=json.loads((folder/model/'report.json').read_text())
            comparison.append(dict(model=model,method=method,**{k:row[k] for k in
                ('y_rmse','y_nlpd','y_crps','y_coverage_95','y_interval_width')}))
    table=pd.DataFrame(comparison)
    table.to_csv(OUTPUT/'comparison.csv',index=False)
    keys=['model','n_train','n_test','y_rmse','y_nlpd','y_crps','y_coverage_95','y_interval_width',
          'max_rhat','min_bulk_ess','min_tail_ess','draws_per_chain']
    pd.DataFrame([{k:r[k] for k in keys} for r in results]).to_csv(OUTPUT/'scores.csv',index=False)
    lines=[r'\begin{table}[htbp]',r'\centering',r'\small',
        r'\caption{Wine regression with $\mu=\beta=0$ on the centered response: the same 2376 training and 1585 test rows. '
        r'One ARD SE kernel; STP fixes $\nu=5$. Full Bayesian predictions integrate kernel/noise uncertainty using '
        r'Metropolis--Hastings samples and integrate $W$ analytically. Scores concern observed $y$ in original units.}',
        r'\label{tab:wine_zero_full_bayes}',r'\begin{tabular}{llrrrr}',r'\toprule',
        r'Model & Inference & RMSE & NLPD & CRPS & 95\% coverage (\%) \\',r'\midrule']
    for row in comparison:
        lines.append(' & '.join(['GP' if row['model']=='gp_zero' else 'STP',row['method'],
            f'{row["y_rmse"]:.5f}',f'{row["y_nlpd"]:.5f}',f'{row["y_crps"]:.5f}',
            f'{100*row["y_coverage_95"]:.2f}'])+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (OUTPUT/'comparison.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    truth=data['y'][data['test_ids']]
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for j,model in enumerate(MODELS):
        with np.load(OUTPUT/model/'predictions.npz') as p:
            axes[0,j].scatter(truth,p['y_mean'],s=7,alpha=.4)
            limits=[min(truth.min(),p['y_mean'].min()),max(truth.max(),p['y_mean'].max())]
            axes[0,j].plot(limits,limits,'k--',lw=1)
            axes[0,j].set(title=('GP' if j==0 else 'STP')+' full Bayes',xlabel='Observed quality',ylabel='Predictive mean')
            order=np.argsort(p['y_mean']);xx=np.arange(len(order))
            axes[1,j].fill_between(xx,p['y_ci'][order,0],p['y_ci'][order,1],alpha=.25,label='95% predictive interval')
            axes[1,j].plot(xx,p['y_mean'][order],lw=1,label='Predictive mean')
            axes[1,j].scatter(xx,truth[order],s=3,c='black',alpha=.2)
            axes[1,j].set(xlabel='Test rows ordered by prediction',ylabel='Quality')
            axes[1,j].legend(fontsize=8)
        with np.load(OUTPUT/model/'posterior_samples.npz') as s:
            traces=s['log_theta'];names=s['parameter_names'].tolist()
            if model=='stp_zero':traces=np.concatenate([traces,np.log(s['W_samples'])[:,:,None]],axis=-1);names+=['W']
        tf,ta=plt.subplots(4,4,figsize=(14,10),constrained_layout=True)
        for i,ax in enumerate(ta.flat):
            if i>=len(names):ax.set_visible(False);continue
            for chain in traces:ax.plot(chain[:,i],lw=.4,alpha=.65)
            ax.set_title('log '+names[i],fontsize=9)
        tf.suptitle(model+': four retained posterior chains')
        tf.savefig(OUTPUT/model/'traces.png',dpi=125);plt.close(tf)
    fig.savefig(OUTPUT/'regression.png',dpi=140);fig.savefig(OUTPUT/'regression.pdf');plt.close(fig)
    hashes=json.loads((OUTPUT/'source_provenance.json').read_text())
    for path,expected in hashes.items():assert digest(SOURCE/path)==expected
    write_json(OUTPUT/'validation.json',dict(all_checks_passed=True,models=2,checks=validation,old_results_unchanged=True,
        exact_same_split=True,all_retained_draws_used=True,max_rhat=max(r['max_rhat'] for r in results),
        min_bulk_ess=min(r['min_bulk_ess'] for r in results),min_tail_ess=min(r['min_tail_ess'] for r in results),
        result_size_MiB=sum(p.stat().st_size for p in OUTPUT.rglob('*') if p.is_file())/2**20))
    print(table.to_string(index=False),flush=True)


METHOD='''WINE: FULL BAYESIAN ZERO-FUNCTION GP AND STP

Use the EXACT saved 2376 training / 1585 test split from the 60% Wine MAP
comparison (seed 17026). Exact duplicate rows were removed before splitting.
Retain training-only input/target standardization and the fixed original-unit
intercept 5.8657407407407405. mu=beta=0 refers to this centered response.
There is one 11-dimensional function, one ARD SE kernel, and one global W
in STP. All 13 kernel/noise parameters are uncertain. Priors on their physical
logs are independent Normal(center,2^2), identical to the MAP comparison.
Center=log([0.5,0.1,eleven entries of 1/11]) for variance-one training y.
The intercept, preprocessing constants and STP nu=5 remain fixed.

GP: y|theta ~ Normal(0,A), A=K_theta+kI, W=1.
STP: W~InverseGamma(nu/2,(nu-2)/2); y|theta,W~Normal(0,W A).
Integrate W exactly: y|theta is multivariate t with df=nu and shape
(nu-2)/nu*A. Its covariance is A. The sampling target is the normalized
marginal likelihood times the Gaussian priors on log(theta). No extra log
Jacobian is needed for parameters whose prior is specified on those logs.

Use four Metropolis-Hastings chains. This run is NOT HMC. The exact posterior
density corrects every proposal, so the MAP/Hessian only initialize sampling;
the retained distribution is not a Laplace approximation. Each chain first
runs 200 pilot iterations with a Student-t proposal based on posterior
curvature. Fit a defensive mixture of a pilot-based Normal (weight 0.8) and
a broader curvature-based t_5 (weight 0.2), freeze it, discard another 150
iterations, and retain at least 750 iterations per chain. Rejected states
are retained with their full multiplicities. Extend all chains in blocks
of 750 if needed for rank-normalized split/folded R-hat <1.01 and bulk/tail
ESS >=400, including STP W and effective amplitudes. Checkpoints preserve
the RNG state. No test targets enter estimation, tuning or diagnostics.

For each retained theta, condition analytically on training y. GP prediction
is Gaussian. In STP, W|theta,y ~ IG((nu+n)/2,(nu-2+y^T A^-1 y)/2), and the
conditional prediction after integrating W is Student-t with df=nu+n=2381.
Mix these conditional laws across all retained hyperparameter draws. Use
their conditional variances plus between-mean variance. Compute exact
mixture densities and numerically invert their CDFs for 95% equal-tail
intervals. CRPS uses converged quadrature, independently checked at selected
test rows. No Gaussian approximation replaces the Student-t components.

Scores concern observed y in original quality units: RMSE, mean pointwise
NLPD, CRPS, 95% coverage, width. Quality is treated as continuous. Latent f
is unknown, so no latent-truth scores are reported. Retain compact samples,
pointwise predictions and summaries; prediction components are temporary.
Compare directly with the previous zero-function MAP rows on identical data.
This is a single train/test split, not an across-split uncertainty estimate.

Run/resume: python wine_zero_full_bayes_experiment.py --stage all
Sample only: add --stage sample --model gp_zero (or stp_zero)

SAMPLER REFINEMENT FOR THIS RUN
The initial independence chains mixed slowly in length-scale tails. All those
draws were subsequently treated as additional warmup, not prediction samples.
wine_zero_refine_sampling.py fits three regularized Normal proposal components
to them, adds a defensive Student-t component, and freezes the proposal. Each
iteration chooses an independence MH move (80%) or a symmetric preconditioned
random-walk MH move (20%). Both moves target exactly the same posterior.
Discard 300 new warmup iterations per chain; retain at least 1000, extending
in blocks of 1000 until the same R-hat/ESS checks pass. refinement.json records
the frozen configuration and hashes of discarded warmup. All final prediction
samples come from refined_chain*.npz; sampling_report.json identifies them.
The original experiment.json documents the initial sampler configuration;
refinement.json and sampling_report.json document the final one. The original
chains remain available for audit. --stage all resumes the refined sampler.
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('prepare','sample','predict','summarize','all'),default='all')
    parser.add_argument('--model',choices=MODELS)
    args=parser.parse_args()
    data=prepare()
    if args.stage=='prepare':return
    for model in ([args.model] if args.model else MODELS):
        if args.stage in ('sample','all'):sample_model(data,model)
        if args.stage in ('predict','all'):predict_model(data,model)
    if args.stage in ('summarize','all'):summarize(data)


if __name__=='__main__':
    with threadpool_limits(limits=SETTINGS['threads']):main()
