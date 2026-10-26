"""Audit source mapping, split, shared MAP functions and all electricity predictions."""
import json
import numpy as np
import pandas as pd
from scipy import linalg, special, stats
from threadpoolctl import threadpool_limits
from electricity_shared_plugin_experiment import OUTPUT, COUNTS, CONTEXT_SEED, METRICS, digest, aggregate
from load_electricity_customer import CACHE, daily_curves
from GH_shared_function_map import MODELS, make_target, gig_mean
from GH_fixed_function import mixing_quadrature, marginal_distribution
from shared_beta_plugin_experiment import write_json


def validate():
    with np.load(CACHE/'series.npz') as saved:source={key:saved[key] for key in saved.files}
    with np.load(OUTPUT/'dataset.npz') as saved:data={key:saved[key] for key in saved.files}
    settings=json.loads((OUTPUT/'experiment.json').read_text())
    assert settings['source_sha256']==digest(CACHE/'series.npz')
    stamps=pd.DatetimeIndex(source['timestamps'])
    expected=pd.bdate_range('2014-01-02','2014-03-31').strftime('%Y-%m-%d').to_numpy()
    np.testing.assert_array_equal(data['dates'],expected)
    for i,date in enumerate(data['dates']):
        ids=np.flatnonzero(stamps.normalize()==pd.Timestamp(str(date)))
        assert len(ids)==96
        np.testing.assert_array_equal(stamps[ids].hour+stamps[ids].minute/60,data['time_hours'])
        np.testing.assert_array_equal(source['y'][ids],data['y'][i])
        assert data['y'][i].max()>0
    _,_,quality=daily_curves(source)
    for year in range(2011,2015):
        for month in (3,10):
            sunday=pd.date_range(f'{year}-{month:02d}-01',periods=31).to_series()
            sunday=sunday[(sunday.dt.month==month)&(sunday.dt.dayofweek==6)].iloc[-1]
            assert bool(quality.loc[quality.date==str(sunday.date()),'dst_adjusted'].iloc[0])
    np.testing.assert_array_equal(data['train_rows'],np.arange(40))
    np.testing.assert_array_equal(data['gap_rows'],np.arange(40,45))
    np.testing.assert_array_equal(data['test_rows'],np.arange(45,63))
    x=data['x'];scale=float(data['common_scale']);train=data['train_rows'];Y=data['y'][train]/scale
    np.testing.assert_allclose(scale,np.std(data['y'][train]-data['y'][train].mean(0)),atol=1e-14)
    center=np.log([.5,max(float(np.mean(Y.mean(0)**2)),.5),.01,1.,.04])
    np.testing.assert_allclose(center,settings['prior_center'],atol=1e-14)
    checks=[];scores=[];negative=[]
    for model in MODELS:
        folder=OUTPUT/model;fit=json.loads((folder/'fit.json').read_text());fit_hash=digest(folder/'fit.json')
        assert fit['dataset_sha256']==digest(OUTPUT/'dataset.npz')
        np.testing.assert_array_equal(center,fit['prior_center'])
        q=np.asarray(fit['log_parameters']);v,a,k,w,hw=np.exp(q[:5])
        target,_=make_target(Y,x,model,prior_center=center,backend='dense');value,gradient=target(q)
        np.testing.assert_allclose(value,fit['log_posterior'],atol=1e-7,rtol=0)
        assert max(abs(gradient))<1e-3 and fit['max_hessian_eigenvalue']<1e-5
        W=np.ones(len(Y)) if model=='gp' else np.exp(q[5:]);c=W/gig_mean() if model=='ghp_beta' else np.ones(len(Y))
        kernel=lambda v,w:v*np.exp(-.5*w*(x[:,None]-x[None,:])**2)
        A=kernel(v,w)+k*np.eye(len(x));Sigma=kernel(a,hw)
        t=np.sum(c*c/W);b=np.sum((c/W)[:,None]*Y,axis=0)
        h=Sigma@linalg.solve(A+t*Sigma,b,assume_a='pos')
        fn=h/gig_mean() if model=='ghp_beta' else h
        np.testing.assert_allclose(fn,fit['function'],atol=2e-8)
        fn*=scale;v*=scale**2;k*=scale**2;K=kernel(v,w);max_error=0.
        for count in COUNTS:
            for j,row in enumerate(data['test_rows']):
                stem=folder/f'context_{count}'/f'day_{data["dates"][row]}'
                with np.load(stem.with_suffix('.npz')) as saved:p={key:saved[key] for key in saved.files}
                p['mixing']=str(p['mixing'])
                for key,array in p.items():
                    if isinstance(array,np.ndarray) and np.issubdtype(array.dtype,np.number):assert np.isfinite(array).all(),key
                record=json.loads(stem.with_suffix('.json').read_text());assert record['fit_sha256']==fit_hash
                order=np.random.default_rng(np.random.SeedSequence([CONTEXT_SEED,int(data['day_ids'][row])])).permutation(96)
                ids=np.sort(order[:count]);held=np.setdiff1d(np.arange(96),ids)
                np.testing.assert_array_equal(ids,p['context_ids']);np.testing.assert_array_equal(held,p['test_ids'])
                np.testing.assert_array_equal(order,data['context_orders'][j])
                A=K[np.ix_(ids,ids)]+k*np.eye(count);B=linalg.solve(A,K[ids,:],assume_a='pos').T
                mu=np.zeros(96) if model=='ghp_beta' else fn;beta=fn if model=='ghp_beta' else np.zeros(96)
                residual=data['y'][row,ids]-mu[ids];location=mu+B@residual;effect=beta-B@beta[ids]
                S=K-B@K[ids,:]
                np.testing.assert_allclose(location,p['location'],atol=2e-8)
                np.testing.assert_allclose(effect,p['conditional_beta'],atol=2e-8)
                if p['mixing']=='gig':
                    np.testing.assert_allclose(p['W_lambda'],1-count/2)
                    np.testing.assert_allclose(p['W_chi'],1+residual@linalg.solve(A,residual,assume_a='pos'),rtol=1e-8)
                    np.testing.assert_allclose(p['W_psi'],1+beta[ids]@linalg.solve(A,beta[ids],assume_a='pos'),rtol=1e-8)
                elif p['mixing']=='ig':
                    np.testing.assert_allclose(p['W_shape'],(5+count)/2)
                    np.testing.assert_allclose(p['W_scale'],(3+residual@linalg.solve(A,residual,assume_a='pos'))/2,rtol=1e-8)
                mean=location+p['W_mean']*effect
                variance=p['W_mean']*(np.diag(S)+k)+p['W_var']*effect**2
                np.testing.assert_allclose(mean,p['y_mean'],atol=2e-8)
                np.testing.assert_allclose(variance,p['y_var'],atol=2e-8,rtol=1e-7)
                max_error=max(max_error,float(max(abs(mean-p['y_mean']))))
                for name,noise in [('f',0.),('y',k)]:
                    np.testing.assert_allclose(p[name+'_scale_var'],np.diag(S)+noise,atol=2e-8)
                    p[name+'_scale_cov']=np.diag(p[name+'_scale_var'])
                dist=marginal_distribution(p,'y')
                np.testing.assert_allclose(-dist.logpdf(data['y'][row])[held],p['y_nlpd'],atol=2e-7,rtol=1e-8)
                np.testing.assert_allclose(dist.cdf(p['y_ci'][:,0]),.025,atol=1e-7)
                np.testing.assert_allclose(dist.cdf(p['y_ci'][:,1]),.975,atol=1e-7)
                nodes,weights,_=mixing_quadrature(p)
                logp=special.logsumexp(np.log(weights[:,None])+stats.norm.logpdf(data['y'][row],
                    location+nodes[:,None]*effect,np.sqrt(nodes[:,None]*(np.diag(S)+k))),axis=0)
                np.testing.assert_allclose(-logp[held],p['y_nlpd'],atol=4e-6,rtol=1e-7)
                truth=data['y'][row,held];ci=p['y_ci'][held]
                actual=dict(y_rmse=np.sqrt(np.mean((truth-mean[held])**2)),y_nlpd=p['y_nlpd'].mean(),
                    y_crps=p['y_crps'].mean(),y_coverage_95=np.mean((truth>=ci[:,0])&(truth<=ci[:,1])),
                    y_interval_width=np.mean(ci[:,1]-ci[:,0]))
                for key,val in actual.items():np.testing.assert_allclose(val,record['scores'][key],atol=1e-8)
                scores.append(record['scores'])
                negative.append(dict(model=model,context_count=count,date=str(data['dates'][row]),
                                     mean_predicted_negative_probability=float(dist.cdf(np.zeros(96))[held].mean())))
        checks.append(dict(model=model,max_abs_gradient=float(max(abs(gradient))),max_prediction_error=max_error,
                           max_hessian_eigenvalue=fit['max_hessian_eigenvalue']))
    summary=aggregate(pd.DataFrame(scores)).set_index(['context_count','model'])
    saved=pd.read_csv(OUTPUT/'scores.csv').set_index(['context_count','model'])
    np.testing.assert_allclose(summary[list(METRICS)],saved[list(METRICS)],atol=1e-8)
    pd.DataFrame(negative).to_csv(OUTPUT/'negative_demand_probability.csv',index=False)
    write_json(OUTPUT/'validation.json',dict(all_checks_passed=True,n_models=4,n_predictions=len(scores),
        exact_source_mapping=True,calendar_and_DST_checks=True,chronological_split_with_gap=True,
        training_only_preprocessing=True,identical_nested_contexts=True,independent_conditioning_checks=checks))
    print(f'Validated four fits and {len(scores)} predictions against source data and independent conditioning.')


if __name__=='__main__':
    with threadpool_limits(limits=1):validate()
