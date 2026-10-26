"""Independent data, fitting and prediction checks for the FS1 pilot."""
import hashlib
import json
import numpy as np
import pandas as pd
from scipy import linalg, special, stats
from threadpoolctl import threadpool_limits

from load_hydraulic_fs1 import CACHE
from hydraulic_shared_plugin_experiment import OUTPUT, COUNTS, CONTEXT_SEED, METRICS
from GH_shared_function_map import MODELS, make_target, gig_mean
from GH_fixed_function import mixing_quadrature, marginal_distribution
from shared_beta_plugin_experiment import write_json


def validate():
    with np.load(CACHE/'fs1.npz') as saved:source={key:saved[key] for key in saved.files}
    with np.load(OUTPUT/'dataset.npz') as saved:data={key:saved[key] for key in saved.files}
    np.testing.assert_array_equal(source['raw_fs1'].reshape(2205,100,6).mean(2),data['y'])
    np.testing.assert_array_equal(source['profile'],data['profile'])
    selected=np.flatnonzero(np.all(source['profile']==[100,100,0,90,0],axis=1))
    np.testing.assert_array_equal(data['eligible_rows'],selected)
    np.testing.assert_array_equal(data['train_rows'],selected[:12])
    np.testing.assert_array_equal(data['gap_rows'],selected[12:14])
    np.testing.assert_array_equal(data['test_rows'],selected[14:])
    np.testing.assert_array_equal(data['cycle_ids'][selected],np.arange(2187,2206))
    train,test=data['train_rows'],data['test_rows'];x=data['x'];scale=float(data['common_scale'])
    np.testing.assert_allclose(scale,np.std(data['y'][train]-data['y'][train].mean(0)),atol=1e-15)
    Y=data['y'][train]/scale
    source_hash=hashlib.sha256((OUTPUT/'dataset.npz').read_bytes()).hexdigest()
    checks=[];rows=[];common=[]
    for model in MODELS:
        folder=OUTPUT/model;fit=json.loads((folder/'fit.json').read_text())
        assert fit['dataset_sha256']==source_hash
        fit_hash=hashlib.sha256((folder/'fit.json').read_bytes()).hexdigest()
        q=np.asarray(fit['log_parameters']);v,a,k,w,hw=np.exp(q[:5])
        target,_=make_target(Y,x,model,prior_center=fit['prior_center'],backend='dense')
        value,gradient=target(q)
        np.testing.assert_allclose(value,fit['log_posterior'],atol=1e-7,rtol=0)
        assert max(abs(gradient))<1e-3 and fit['max_hessian_eigenvalue']<0
        W=np.ones(len(Y)) if model=='gp' else np.exp(q[5:])
        c=W/gig_mean() if model=='ghp_beta' else np.ones(len(Y))
        kernel=lambda v,w:v*np.exp(-.5*w*(x[:,None]-x[None,:])**2)
        A=kernel(v,w)+k*np.eye(len(x));S=kernel(a,hw)
        t=np.sum(c*c/W);b=np.sum((c/W)[:,None]*Y,axis=0)
        population=S@linalg.solve(A+t*S,b,assume_a='pos')
        fn=population/gig_mean() if model=='ghp_beta' else population
        np.testing.assert_allclose(fn,fit['function'],atol=2e-8,rtol=1e-8)
        fn*=scale;v*=scale**2;k*=scale**2
        full_kernel=kernel(v,w)
        model_errors=[]
        for count in COUNTS:
            for j,row in enumerate(test):
                stem=folder/f'context_{count}'/f'cycle_{data["cycle_ids"][row]}'
                with np.load(stem.with_suffix('.npz')) as saved:p={key:saved[key] for key in saved.files}
                p['mixing']=str(p['mixing'])
                for key,array in p.items():
                    if isinstance(array,np.ndarray) and np.issubdtype(array.dtype,np.number):assert np.isfinite(array).all(),key
                order=np.random.default_rng(np.random.SeedSequence([CONTEXT_SEED,int(row)])).permutation(100)
                ids=np.sort(order[:count]);held=np.setdiff1d(np.arange(100),ids)
                np.testing.assert_array_equal(p['context_ids'],ids);np.testing.assert_array_equal(p['test_ids'],held)
                residual_cov=full_kernel[np.ix_(ids,ids)]+k*np.eye(count)
                cross=full_kernel[:,ids]
                B=linalg.solve(residual_cov,cross.T,assume_a='pos').T
                mu=np.zeros(100) if model=='ghp_beta' else fn
                beta=fn if model=='ghp_beta' else np.zeros(100)
                residual=data['y'][row,ids]-mu[ids]
                location=mu+B@residual;effect=beta-B@beta[ids]
                Sfull=full_kernel-B@cross.T
                np.testing.assert_allclose(location,p['location'],atol=2e-8)
                np.testing.assert_allclose(effect,p['conditional_beta'],atol=2e-8)
                if p['mixing']=='gig':
                    np.testing.assert_allclose(p['W_lambda'],1-count/2)
                    np.testing.assert_allclose(p['W_chi'],1+residual@linalg.solve(residual_cov,residual,assume_a='pos'),rtol=1e-8)
                    np.testing.assert_allclose(p['W_psi'],1+beta[ids]@linalg.solve(residual_cov,beta[ids],assume_a='pos'),rtol=1e-8)
                elif p['mixing']=='ig':
                    np.testing.assert_allclose(p['W_shape'],(5+count)/2)
                    np.testing.assert_allclose(p['W_scale'],(3+residual@linalg.solve(residual_cov,residual,assume_a='pos'))/2,rtol=1e-8)
                mean=location+p['W_mean']*effect
                variance=p['W_mean']*(np.diag(Sfull)+k)+p['W_var']*effect**2
                np.testing.assert_allclose(mean,p['y_mean'],atol=2e-8)
                np.testing.assert_allclose(variance,p['y_var'],atol=2e-9,rtol=1e-7)
                for name,noise in [('f',0.),('y',k)]:
                    np.testing.assert_allclose(np.diag(Sfull)+noise,p[f'{name}_scale_var'],atol=2e-9)
                    p[f'{name}_scale_cov']=np.diag(p[f'{name}_scale_var'])
                dist=marginal_distribution(p,'y')
                np.testing.assert_allclose(-dist.logpdf(data['y'][row])[held],p['y_nlpd'],atol=2e-7,rtol=1e-8)
                np.testing.assert_allclose(dist.cdf(p['y_ci'][:,0]),.025,atol=1e-7)
                np.testing.assert_allclose(dist.cdf(p['y_ci'][:,1]),.975,atol=1e-7)
                nodes,weights,_=mixing_quadrature(p)
                means=location+nodes[:,None]*effect
                vars_=nodes[:,None]*(np.diag(Sfull)+k)
                logp=special.logsumexp(np.log(weights[:,None])+stats.norm.logpdf(data['y'][row],means,np.sqrt(vars_)),axis=0)
                np.testing.assert_allclose(-logp[held],p['y_nlpd'],atol=3e-6,rtol=1e-7)
                record=json.loads(stem.with_suffix('.json').read_text());assert record['fit_sha256']==fit_hash
                score=record['scores'];truth=data['y'][row,held];ci=p['y_ci'][held]
                actual=dict(y_rmse=np.sqrt(np.mean((truth-mean[held])**2)),y_nlpd=p['y_nlpd'].mean(),
                            y_crps=p['y_crps'].mean(),y_coverage_95=np.mean((truth>=ci[:,0])&(truth<=ci[:,1])),
                            y_interval_width=np.mean(ci[:,1]-ci[:,0]))
                for key,val in actual.items():np.testing.assert_allclose(val,score[key],atol=2e-8)
                rows.append(score);model_errors.append(float(max(abs(mean-p['y_mean']))))
                # Compare 15 and 30 contexts on the SAME 70 evaluation points.
                common_ids=np.setdiff1d(np.arange(100),order[:30]);positions=np.searchsorted(held,common_ids)
                truth=data['y'][row,common_ids];ci=p['y_ci'][common_ids]
                common.append(dict(model=model,context_count=count,cycle_id=int(data['cycle_ids'][row]),
                     n_test=len(common_ids),y_rmse=float(np.sqrt(np.mean((truth-mean[common_ids])**2))),
                     y_nlpd=float(p['y_nlpd'][positions].mean()),y_crps=float(p['y_crps'][positions].mean()),
                     y_coverage_95=float(np.mean((truth>=ci[:,0])&(truth<=ci[:,1]))),
                     y_interval_width=float(np.mean(ci[:,1]-ci[:,0]))))
        checks.append(dict(model=model,max_abs_gradient=float(max(abs(gradient))),
                           max_prediction_error=max(model_errors),max_hessian_eigenvalue=fit['max_hessian_eigenvalue']))
    saved=pd.read_csv(OUTPUT/'scores.csv').set_index(['context_count','model'])
    per=pd.DataFrame(rows)
    for (count,model),frame in per.groupby(['context_count','model']):
        np.testing.assert_allclose(np.sqrt(np.mean(frame.y_rmse**2)),saved.loc[(count,model),'y_rmse'])
        for key in METRICS[1:]:np.testing.assert_allclose(frame[key].mean(),saved.loc[(count,model),key])
    common=pd.DataFrame(common);common.to_csv(OUTPUT/'common_test_per_curve_scores.csv',index=False)
    summary=[]
    for (count,model),frame in common.groupby(['context_count','model']):
        summary.append(dict(context_count=count,model=model,y_rmse=float(np.sqrt(np.mean(frame.y_rmse**2))),
                            **{key:float(frame[key].mean()) for key in METRICS[1:]}))
    pd.DataFrame(summary).to_csv(OUTPUT/'common_test_scores.csv',index=False)
    write_json(OUTPUT/'validation.json',dict(all_checks_passed=True,n_models=4,n_predictions=40,
               exact_six_reading_aggregation=True,condition_group_controlled=True,ordered_split_with_gap=True,
               training_only_scale=True,independent_conditioning_checks=checks))
    print('Validated four fits and 40 predictions; common-70-point context comparison saved.')


if __name__=='__main__':
    with threadpool_limits(limits=1):validate()
