"""Check the ARD plug-in extension against independent dense conditioning."""
import unittest
import numpy as np
from scipy import linalg
from threadpoolctl import threadpool_limits

from GH_tabular_hmc import MODELS, ZERO_MODELS, ALL_MODELS, has_mixing, make_target, coordinates, default_center
from GH_tabular_plugin_map import PluginPredictor, se_kernel
from GH_fixed_function import fixed_function_prediction


class TabularPluginTests(unittest.TestCase):
    def test_extreme_gh_density_check_refines_mixing_tails(self):
        from tabular_plugin_experiment import batch_scores
        from GH_fixed_function import StableGHMarginals
        x=np.linspace(0,1,8)
        p=fixed_function_prediction(np.zeros(8),x,np.array([1.8]),
                                    v0=.5,k=.1,w=1.,mixing='gig')
        truth=np.array([200.])
        expected=-StableGHMarginals(p,'y').logpdf(truth)
        scores,checks=batch_scores(p,truth,check=True)
        self.assertTrue(checks.get('density_quadrature_refined',False))
        np.testing.assert_allclose(scores['nlpd'],expected,atol=1e-10)
        self.assertTrue(np.isfinite(scores['crps']).all())
        self.assertLess(checks['max_logpdf_difference'],1e-6)

    def test_condensed_target_matches_dense(self):
        rng = np.random.default_rng(884)
        x,y = rng.normal(size=(19,3)),rng.normal(size=19)
        x[4] = x[2]  # Include repeated inputs with positive noise.
        for model in ALL_MODELS:
            center = default_center(y,3,model)
            q = np.r_[center+rng.normal(0,.4,len(center)), [np.log(1.4)] if has_mixing(model) else []]
            z = coordinates(q,model)
            dense = make_target(y,x,model=model,prior_center=center)
            compact = make_target(y,x,model=model,prior_center=center,backend='condensed')
            np.testing.assert_allclose(compact(z)[0],dense(z)[0],atol=1e-11)
            np.testing.assert_allclose(compact(z)[1],dense(z)[1],atol=1e-10)

    def test_ard_function_mean_and_prediction(self):
        rng = np.random.default_rng(9)
        x, xp, y = rng.normal(size=(7,2)), rng.normal(size=(4,2)), rng.normal(size=7)
        parameters = [.7,1.2,.15,.8,2.1,.3,1.4]
        for model in MODELS:
            W = 1. if model == 'gp' else 1.6
            q = np.log(parameters+([] if model == 'gp' else [W]))
            predictor = PluginPredictor(y,x,q,model)
            p = predictor.predict(xp,y_mean=5.,y_scale=2.)
            # Dense kernels built independently of the kernel helper.
            kernel = lambda a,b,v,w: v*np.exp(-.5*np.sum((a[:,None,:]-b[None,:,:])**2*w,axis=2))
            A = kernel(x,x,.7,np.array([.8,2.1]))+.15*np.eye(7)
            sigma = kernel(x,x,1.2,np.array([.3,1.4]))
            cross_sigma = kernel(xp,x,1.2,np.array([.3,1.4]))
            r = 2 if model == 'beta_gp' else 0
            C = W*A+W**r*sigma
            latent_cross = (W if model == 'beta_gp' else 1.)*cross_sigma
            estimate = latent_cross@np.linalg.solve(C,y)
            np.testing.assert_allclose(predictor.function(xp),estimate,atol=1e-12)
            train_estimate = (W if model == 'beta_gp' else 1.)*sigma@np.linalg.solve(C,y)
            np.testing.assert_allclose(train_estimate,predictor.function_train,atol=1e-12)
            cross = kernel(xp,x,.7,np.array([.8,2.1]))
            S = kernel(xp,xp,.7,np.array([.8,2.1]))-cross@np.linalg.solve(A,cross.T)
            mu1, mu2 = (np.zeros(7),np.zeros(4)) if model == 'beta_gp' else (train_estimate,estimate)
            b1, b2 = (train_estimate,estimate) if model == 'beta_gp' else (np.zeros(7),np.zeros(4))
            m = mu2+cross@np.linalg.solve(A,y-mu1)
            b = b2-cross@np.linalg.solve(A,b1)
            np.testing.assert_allclose(p['location'],5+2*m,atol=1e-12)
            np.testing.assert_allclose(p['conditional_beta'],2*b,atol=1e-12)
            expected = 4*(p['W_mean']*np.diag(S)+p['W_var']*b*b)
            np.testing.assert_allclose(p['f_var'],expected,atol=1e-12)
            np.testing.assert_allclose(p['y_var']-p['f_var'],4*p['W_mean']*.15,atol=1e-12)
            # Verify the full log posterior gradient, including transformed priors.
            target = make_target(y,x,model=model,prior_center=default_center(y,2))
            z = coordinates(q,model)
            h = 1e-5
            numerical = np.array([(target(z+e*h)[0]-target(z-e*h)[0])/(2*h) for e in np.eye(len(z))])
            np.testing.assert_allclose(target(z)[1],numerical,atol=2e-8,rtol=2e-7)

    def test_one_dimensional_matches_previous_implementation(self):
        x = np.linspace(0,2,8)
        xp = np.array([.1,.8,2.4])
        y = np.sin(x)+.2*np.cos(4*x)
        for model in ALL_MODELS:
            parameters = [.5,.12,1.3] if model in ZERO_MODELS else [.5,1.,.12,1.3,.3]
            q = np.log(parameters+([1.4] if has_mixing(model) else []))
            predictor = PluginPredictor(y,x[:,None],q,model)
            p = predictor.predict(xp[:,None])
            extra = dict(beta_train=predictor.function_train,beta_pred=predictor.function(xp[:,None])) if model == 'beta_gp' else dict(mu_train=predictor.function_train,mu_pred=predictor.function(xp[:,None]))
            old = fixed_function_prediction(y,x,xp,v0=.5,k=.12,w=1.3,mixing=predictor.mixing,**extra)
            for key in ('W_mean','W_var','location','conditional_beta','f_mean','f_var','y_var','f_ci','y_ci'):
                np.testing.assert_allclose(p[key],old[key],atol=2e-8,rtol=1e-8)

    def test_zero_ard_conditional_mean_and_mixing(self):
        rng = np.random.default_rng(321)
        x, xp, y = rng.normal(size=(8,2)), rng.normal(size=(5,2)), rng.normal(size=8)
        kernel = lambda a,b: .7*np.exp(-.5*np.sum((a[:,None]-b[None,:])**2*[.8,1.6],axis=2))
        A = kernel(x,x)+.15*np.eye(len(x))
        cross = kernel(xp,x)
        m = cross@np.linalg.solve(A,y)
        s = np.diag(kernel(xp,xp)-cross@np.linalg.solve(A,cross.T))
        quadratic = y@np.linalg.solve(A,y)
        for model in ZERO_MODELS:
            q = np.log([.7,.15,.8,1.6]+([1.4] if has_mixing(model) else []))
            predictor = PluginPredictor(y,x,q,model)
            p = predictor.predict(xp,y_mean=6.,y_scale=.8)
            np.testing.assert_array_equal(predictor.function_train,np.zeros(len(x)))
            np.testing.assert_array_equal(predictor.function(xp),np.zeros(len(xp)))
            np.testing.assert_array_equal(p['conditional_beta'],np.zeros(len(xp)))
            np.testing.assert_allclose(p['y_mean'],6.+.8*m,atol=1e-12)
            np.testing.assert_allclose(p['y_var'],.8**2*p['W_mean']*(s+.15),atol=1e-12)
            if model == 'ghp_zero':
                np.testing.assert_allclose([p['W_lambda'],p['W_chi'],p['W_psi']],[1-len(x)/2,1+quadratic,1])
            elif model == 'stp_zero':
                np.testing.assert_allclose([p['W_shape'],p['W_scale']],[(5+len(x))/2,(3+quadratic)/2])
            if has_mixing(model):
                changed = q.copy(); changed[-1] = np.log(4.)
                other = PluginPredictor(y,x,changed,model).predict(xp,y_mean=6.,y_scale=.8)
                # Prediction integrates W from its original prior; it is not fixed at the joint MAP W.
                for key in ('y_mean','y_var','y_ci'):
                    np.testing.assert_allclose(p[key],other[key],atol=1e-12)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        unittest.main()
