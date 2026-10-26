"""Audit independent FS1 fits, context-only function estimation and predictions."""
import json
import numpy as np
import pandas as pd
from scipy import linalg, special, stats
from threadpoolctl import threadpool_limits
from hydraulic_independent_plugin_experiment import BASELINE, OUTPUT, COUNTS, METRICS, digest, prepare, aggregate
from GH_independent_plugin_map import MODELS, INTERNAL, context_function, native_center, se
from GH_multiple_hmc import make_target, coordinates
from GH_fixed_function import mixing_quadrature, marginal_distribution
from shared_beta_plugin_experiment import write_json


def validate():
    data,center=prepare()
    settings=json.loads((OUTPUT/'experiment.json').read_text())
    assert digest(BASELINE/'dataset.npz')==settings['dataset_sha256']
    assert digest(BASELINE/'scores.csv')==settings['baseline_scores_sha256']
    scale=float(data['common_scale']);x=data['x'];Y=data['y'][data['train_rows']]/scale
    assert not np.intersect1d(data['train_rows'],data['test_rows']).size
    np.testing.assert_array_equal(data['cycle_ids'][data['train_rows']],np.arange(2187,2199))
    np.testing.assert_array_equal(data['cycle_ids'][data['test_rows']],np.arange(2201,2206))
    checks=[];scores=[]
    for model in MODELS:
        folder=OUTPUT/model;fit=json.loads((folder/'fit.json').read_text());q=np.asarray(fit['log_parameters'])
        assert fit['dataset_sha256']==settings['dataset_sha256']
        np.testing.assert_array_equal(fit['native_prior_center'],native_center(center,model))
        target=make_target([(x,Y)],model=INTERNAL[model],prior_center=fit['native_prior_center'])
        value,gradient=target(coordinates(q,INTERNAL[model]))
        np.testing.assert_allclose(value,fit['log_posterior'],atol=1e-8)
        assert max(abs(gradient))<1e-3 and fit['max_hessian_eigenvalue']<0
        v,a,k,w,sw=np.exp(q[:5]);K=se(x,x,v,w);Sigma=se(x,x,a,sw)
        fit_hash=digest(folder/'fit.json');max_fn_error=0.;max_prediction_error=0.
        for count in COUNTS:
            for j,row in enumerate(data['test_rows']):
                stem=folder/f'context_{count}'/f'cycle_{data["cycle_ids"][row]}'
                with np.load(stem.with_suffix('.npz')) as saved:p={key:saved[key] for key in saved.files}
                record=json.loads(stem.with_suffix('.json').read_text());assert record['fit_sha256']==fit_hash
                with np.load(BASELINE/model/f'context_{count}'/stem.with_suffix('.npz').name) as old:
                    np.testing.assert_array_equal(p['context_ids'],old['context_ids'])
                    np.testing.assert_array_equal(p['test_ids'],old['test_ids'])
                ids=p['context_ids'];held=p['test_ids'];y=data['y'][row,ids]/scale
                np.testing.assert_array_equal(ids,np.sort(data['context_orders'][j,:count]))
                fn,mode=context_function(y,x[ids],x,q[:5],model)
                np.testing.assert_allclose(fn,p['function_estimate_scaled'],atol=1e-10)
                np.testing.assert_allclose(mode['W_for_function'],record['context_mode']['W_for_function'])
                W=mode['W_for_function'];coef=W if model=='ghp_beta' else 1.
                A=K[np.ix_(ids,ids)]+k*np.eye(count)
                C=W*A+coef**2*Sigma[np.ix_(ids,ids)]
                direct=coef*Sigma[:,ids]@linalg.solve(C,y,assume_a='pos')
                np.testing.assert_allclose(fn,direct,atol=1e-8)
                max_fn_error=max(max_fn_error,float(max(abs(fn-direct))))
                # Function uncertainty is intentionally removed in plug-in prediction.
                B=linalg.solve(A,K[ids,:],assume_a='pos').T
                mu=np.zeros(100) if model=='ghp_beta' else direct
                beta=direct if model=='ghp_beta' else np.zeros(100)
                residual=y-mu[ids];location=mu+B@residual;effect=beta-B@beta[ids]
                S=K-B@K[ids,:]
                np.testing.assert_allclose(p['location'],location*scale,atol=1e-8)
                np.testing.assert_allclose(p['conditional_beta'],effect*scale,atol=1e-8)
                p['mixing']=str(p['mixing'])
                if model in ('ghp_mu','ghp_beta'):
                    np.testing.assert_allclose(p['W_lambda'],1-count/2)
                    np.testing.assert_allclose(p['W_chi'],1+residual@linalg.solve(A,residual,assume_a='pos'),rtol=1e-9)
                    np.testing.assert_allclose(p['W_psi'],1+beta[ids]@linalg.solve(A,beta[ids],assume_a='pos'),rtol=1e-9)
                elif model=='stp':
                    np.testing.assert_allclose(p['W_shape'],(5+count)/2)
                    np.testing.assert_allclose(p['W_scale'],(3+residual@linalg.solve(A,residual,assume_a='pos'))/2,rtol=1e-9)
                mean=scale*(location+p['W_mean']*effect)
                variance=scale**2*(p['W_mean']*(np.diag(S)+k)+p['W_var']*effect**2)
                np.testing.assert_allclose(mean,p['y_mean'],atol=1e-8)
                np.testing.assert_allclose(variance,p['y_var'],atol=1e-8)
                max_prediction_error=max(max_prediction_error,float(max(abs(mean-p['y_mean']))))
                for name,noise in [('f',0.),('y',k)]:
                    np.testing.assert_allclose(p[name+'_scale_var'],scale**2*(np.diag(S)+noise),atol=1e-8)
                    p[name+'_scale_cov']=np.diag(p[name+'_scale_var'])
                dist=marginal_distribution(p,'y')
                np.testing.assert_allclose(-dist.logpdf(data['y'][row])[held],p['y_nlpd'],atol=1e-7)
                np.testing.assert_allclose(dist.cdf(p['y_ci'][:,0]),.025,atol=1e-7)
                np.testing.assert_allclose(dist.cdf(p['y_ci'][:,1]),.975,atol=1e-7)
                nodes,weights,_=mixing_quadrature(p)
                logp=special.logsumexp(np.log(weights[:,None])+stats.norm.logpdf(data['y'][row],
                    scale*(location+nodes[:,None]*effect),scale*np.sqrt(nodes[:,None]*(np.diag(S)+k))),axis=0)
                np.testing.assert_allclose(-logp[held],p['y_nlpd'],atol=5e-6,rtol=1e-7)
                truth=data['y'][row,held];ci=p['y_ci'][held]
                actual=dict(y_rmse=np.sqrt(np.mean((truth-mean[held])**2)),y_nlpd=p['y_nlpd'].mean(),
                    y_crps=p['y_crps'].mean(),y_coverage_95=np.mean((truth>=ci[:,0])&(truth<=ci[:,1])),
                    y_interval_width=np.mean(ci[:,1]-ci[:,0]))
                for key,val in actual.items():np.testing.assert_allclose(val,record['scores'][key],atol=1e-8)
                for key,array in p.items():
                    if isinstance(array,np.ndarray) and np.issubdtype(array.dtype,np.number):assert np.isfinite(array).all(),key
                scores.append(record['scores'])
        checks.append(dict(model=model,max_abs_gradient=float(max(abs(gradient))),max_function_error=max_fn_error,
                           max_prediction_error=max_prediction_error,max_hessian_eigenvalue=fit['max_hessian_eigenvalue']))
    summary=aggregate(pd.DataFrame(scores)).set_index(['context_count','model'])
    saved=pd.read_csv(OUTPUT/'scores.csv').set_index(['context_count','model'])
    np.testing.assert_allclose(summary[list(METRICS)],saved[list(METRICS)],atol=1e-8)
    write_json(OUTPUT/'validation.json',dict(all_checks_passed=True,n_models=4,n_predictions=40,
               baseline_dataset_and_scores_unchanged=True,same_contexts_and_heldout_points=True,
               frozen_training_hyperparameters=True,context_only_function_inference=True,
               independent_conditioning_checks=checks))
    print('Validated four independent-function fits and all 40 predictions.')


if __name__=='__main__':
    with threadpool_limits(limits=1):validate()
