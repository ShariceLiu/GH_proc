"""Audit saved split, fit, scaling, predictions and metrics independently."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import linalg, special, stats
from threadpoolctl import threadpool_limits

from GH_tabular_hmc import MODELS, ZERO_MODELS, has_mixing, make_target, coordinates, default_center
from GH_fixed_function import mixing_quadrature
from shared_beta_plugin_experiment import write_json
from tabular_plugin_experiment import OUTPUT, load_source


def validate(datasets=('concrete', 'power'), *, destination=OUTPUT, models=MODELS, subfolder=None, output=OUTPUT):
    results = []
    for dataset in datasets:
        root = Path(output)/dataset
        model_root = root if subfolder is None else root/subfolder
        config = json.loads((model_root/'experiment.json').read_text())
        with np.load(root/'dataset.npz') as saved:
            d = {key: saved[key] for key in saved.files}
        X,Y,hashes = load_source(dataset)
        assert config['source_sha256'] == hashes
        raw = X.copy(); raw['__target__'] = Y.to_numpy()
        retained = np.flatnonzero(~raw.duplicated().to_numpy())
        np.testing.assert_array_equal(d['source_row_ids'],retained)
        np.testing.assert_array_equal(d['x'],X.to_numpy()[retained])
        np.testing.assert_array_equal(d['y'],Y.to_numpy()[retained])
        train,test = d['train_ids'],d['test_ids']
        np.testing.assert_array_equal(np.sort(np.r_[train,test]),np.arange(len(retained)))
        assert len(train)==config['n_train'] and len(test)==config['n_test']
        _,groups = np.unique(d['x'], axis=0, return_inverse=True)
        np.testing.assert_array_equal(d['input_group_ids'],groups)
        assert not np.intersect1d(groups[train],groups[test]).size
        expected_train = []
        for group in np.random.default_rng(config['split_seed']).permutation(np.unique(groups)):
            ids = np.flatnonzero(groups == group)
            if len(expected_train)+len(ids) <= config['n_train']:
                expected_train.extend(ids.tolist())
            if len(expected_train) == config['n_train']:
                break
        np.testing.assert_array_equal(train,np.sort(expected_train))
        for key, actual in [('x_mean',d['x'][train].mean(0)),('x_scale',d['x'][train].std(0)),
                            ('y_mean',d['y'][train].mean()),('y_scale',d['y'][train].std())]:
            np.testing.assert_allclose(d[key],actual,atol=1e-13)
        np.testing.assert_allclose(d['x_scaled'],(d['x']-d['x_mean'])/d['x_scale'])
        np.testing.assert_allclose(d['y_scaled'],(d['y']-d['y_mean'])/d['y_scale'])
        assert config['dataset_sha256']==hashlib.sha256((root/'dataset.npz').read_bytes()).hexdigest()
        positions = np.unique(np.linspace(0,len(test)-1,17,dtype=int))
        x,y = d['x_scaled'][train],d['y_scaled'][train]
        xp = d['x_scaled'][test[positions]]
        for model in models:
            folder = model_root/model
            fit=json.loads((folder/'fit.json').read_text())
            report=json.loads((folder/'report.json').read_text())
            assert report['fit_sha256']==hashlib.sha256((folder/'fit.json').read_bytes()).hexdigest()
            assert report['predictions_sha256']==hashlib.sha256((folder/'predictions.npz').read_bytes()).hexdigest()
            assert fit['dataset_sha256']==report['dataset_sha256']==config['dataset_sha256']
            with np.load(folder/'predictions.npz') as saved:
                p={key:saved[key] for key in saved.files}
            for key,value in p.items():
                assert np.isfinite(value).all(),(dataset,model,key)
            np.testing.assert_array_equal(p['test_ids'],test)
            q=np.asarray(fit['log_parameters']);z=coordinates(q,model)
            if 'mixing_hyperparameters' in fit:
                from GH_tabular_learned_plugin_map import LearnedPluginTarget
                target=LearnedPluginTarget(y,x,model,prior_center=default_center(y,x.shape[1],model),
                                           backend=fit.get('numerical_backend','dense'))
                z=np.asarray(fit['optimizer_coordinates'])
                np.testing.assert_allclose(z[:len(q)],coordinates(q,model),atol=1e-12)
                h=z[len(q):]
                expected_mix=(dict(nu=float(2+np.exp(h[0]))) if model=='tp_mu_gp'
                              else dict(lam=float(h[0]),omega=float(np.exp(h[1]))))
                assert expected_mix==fit['mixing_hyperparameters']==report['mixing_hyperparameters']
            else:
                target=make_target(y,x,model=model,prior_center=default_center(y,x.shape[1],model),
                                   backend=fit.get('numerical_backend','dense'))
            logp,gradient=target(z)
            del target
            np.testing.assert_allclose(logp,fit['log_posterior'],atol=1e-8)
            assert max(abs(gradient))<1e-4 and fit['max_hessian_eigenvalue']<0
            # Independent dense Gaussian conditioning for the fitted latent GP.
            n,D=x.shape
            W_map=np.exp(q[-1]) if has_mixing(model) else 1.
            def kernel(left,right,v,w):
                distance=np.zeros((len(left),len(right)))
                for j,weight in enumerate(w):
                    distance += weight*(left[:,j,None]-right[None,:,j])**2
                return v*np.exp(-.5*distance)
            if model in ZERO_MODELS:
                v,k=np.exp(q[:2]);w1=np.exp(q[2:2+D])
                A=kernel(x,x,v,w1)+k*np.eye(n)
                fn_train,fn_test=np.zeros(n),np.zeros(len(xp))
                assert len(q)==2+D+has_mixing(model)
                np.testing.assert_array_equal(p['conditional_beta'],np.zeros(len(test)))
                np.testing.assert_array_equal(p['function_test_scaled'],np.zeros(len(test)))
                np.testing.assert_allclose(report['W_joint_map'],W_map)
            else:
                v,a,k=np.exp(q[:3])
                w1,w2=np.exp(q[3:3+D]),np.exp(q[3+D:3+2*D])
                A=kernel(x,x,v,w1)+k*np.eye(n)
                sigma=kernel(x,x,a,w2)
                C=W_map*A+(W_map**2 if model=='beta_gp' else 1.)*sigma
                alpha=linalg.solve(C,y,assume_a='pos')
                factor=W_map if model=='beta_gp' else 1.
                fn_train=factor*sigma@alpha
                fn_test=factor*kernel(xp,x,a,w2)@alpha
            np.testing.assert_allclose(fn_train,p['function_train_scaled'],atol=1e-9)
            np.testing.assert_allclose(fn_test,p['function_test_scaled'][positions],atol=1e-9)
            mu1,mu2=(np.zeros(n),np.zeros(len(xp))) if model=='beta_gp' else (fn_train,fn_test)
            b1,b2=(fn_train,fn_test) if model=='beta_gp' else (np.zeros(n),np.zeros(len(xp)))
            B=linalg.solve(A,kernel(x,xp,v,w1),assume_a='pos').T
            location=mu2+B@(y-mu1)
            beta=b2-B@b1
            S=kernel(xp,xp,v,w1)-B@kernel(x,xp,v,w1)
            mix=report['prediction_W_posterior'];ew,varw=mix['W_mean'],mix['W_var']
            np.testing.assert_allclose(p['location'][positions],d['y_mean']+d['y_scale']*location,atol=2e-8)
            np.testing.assert_allclose(p['conditional_beta'][positions],d['y_scale']*beta,atol=2e-8)
            for name in ('f','y'):
                np.testing.assert_allclose(p[name+'_mean'],p['location']+ew*p['conditional_beta'],atol=1e-10)
            for name,noise in [('f',0.),('y',k)]:
                np.testing.assert_allclose(p[f'{name}_var'][positions],d['y_scale']**2*(ew*(np.diag(S)+noise)+varw*beta**2),atol=2e-7,rtol=1e-8)
            if mix['mixing']=='gig':
                learned=fit.get('mixing_hyperparameters',dict(lam=1.,omega=1.))
                np.testing.assert_allclose(mix['W_lambda'],learned['lam']-n/2)
                np.testing.assert_allclose(mix['W_chi'],learned['omega']+(y-mu1)@linalg.solve(A,y-mu1,assume_a='pos'),rtol=1e-9)
                np.testing.assert_allclose(mix['W_psi'],learned['omega']+b1@linalg.solve(A,b1,assume_a='pos'),rtol=1e-9)
            elif mix['mixing']=='ig':
                nu=fit.get('mixing_hyperparameters',dict(nu=5.))['nu']
                np.testing.assert_allclose(mix['W_shape'],(nu+n)/2)
                np.testing.assert_allclose(mix['W_scale'],(nu-2+(y-mu1)@linalg.solve(A,y-mu1,assume_a='pos'))/2,rtol=1e-9)
            nodes,weights,_=mixing_quadrature(mix)
            np.testing.assert_allclose(weights@nodes,ew,rtol=1e-8)
            np.testing.assert_allclose(weights@(nodes-ew)**2,varw,atol=1e-10,rtol=1e-8)
            means=location+nodes[:,None]*beta
            variance=nodes[:,None]*(np.diag(S)+k)
            standardized_logp=special.logsumexp(np.log(weights[:,None])+stats.norm.logpdf(d['y_scaled'][test[positions]],means,np.sqrt(variance)),axis=0)
            np.testing.assert_allclose(p['nlpd'][positions],-standardized_logp+np.log(d['y_scale']),atol=3e-6,rtol=1e-7)
            for side,probability in [(0,.025),(1,.975)]:
                endpoint=(p['y_ci'][positions,side]-d['y_mean'])/d['y_scale']
                np.testing.assert_allclose(weights@special.ndtr((endpoint-means)/np.sqrt(variance)),probability,atol=2e-8)
            actual=dict(y_rmse=float(np.sqrt(np.mean((p['y_mean']-d['y'][test])**2))),
                        y_nlpd=float(np.mean(p['nlpd'])),y_crps=float(np.mean(p['crps'])),
                        y_coverage_95=float(np.mean((d['y'][test]>=p['y_ci'][:,0])&(d['y'][test]<=p['y_ci'][:,1]))))
            for key,value in actual.items():
                np.testing.assert_allclose(report[key],value,rtol=1e-12,atol=1e-12)
            results.append(dict(dataset=dataset,model=model,checks_passed=True,**actual))
            print(f'Validated {dataset} {model}',flush=True)
    saved_scores = pd.read_csv(Path(destination)/'scores.csv').set_index(['dataset','model'])
    for result in results:
        for metric in ('y_rmse','y_nlpd','y_crps','y_coverage_95'):
            np.testing.assert_allclose(saved_scores.loc[(result['dataset'],result['model']),metric], result[metric],atol=1e-12)
    write_json(Path(destination)/'validation.json',dict(all_checks_passed=True,models=len(results),checks=results))
    print(f'Validated all {len(results)} fits: split integrity, training-only scaling, saved hashes, independent conditioning, predictive densities/quantiles and score aggregation.')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=('concrete','power','wine'))
    parser.add_argument('--zero',action='store_true',help='Validate the single-SE zero-function baselines.')
    args=parser.parse_args()
    with threadpool_limits(limits=1):
        if args.dataset:
            if args.zero:
                validate((args.dataset,),destination=OUTPUT/args.dataset/'zero_mean_baselines',
                         models=ZERO_MODELS,subfolder='zero_mean_baselines')
            else:
                validate((args.dataset,),destination=OUTPUT/args.dataset)
        elif args.zero:
            parser.error('--zero requires --dataset')
        else:
            validate()
