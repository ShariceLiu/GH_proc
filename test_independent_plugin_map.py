"""Numerical checks against joint Gaussian conditioning and normalized densities."""
import unittest
import numpy as np
from scipy import linalg, stats
from threadpoolctl import threadpool_limits
from GH_independent_plugin_map import MODELS, INTERNAL, native_center, context_function, se
from GH_multiple_hmc import make_target, coordinates
from GH_shared_function_map import gig_mean


class IndependentFunctions(unittest.TestCase):
    def setUp(self):
        self.x=np.linspace(0,3,7)
        self.Y=np.random.default_rng(5).normal(size=(3,7))
        self.center=np.log([.7,1.2,.08,1.3,.2])

    def test_independent_block_likelihood_and_gradient(self):
        for model in MODELS:
            with self.subTest(model=model):
                center=native_center(self.center,model)
                q=center+np.array([.1,-.15,.05,.2,-.1])
                W=np.ones(3) if model=='gp' else np.array([.5,1.2,2.3])
                if model!='gp':q=np.r_[q,np.log(W)]
                z=coordinates(q,INTERNAL[model]);np.testing.assert_allclose(coordinates(z,INTERNAL[model],inverse=True),q)
                v,a,k,w,sw=np.exp(q[:5]);K=se(self.x,self.x,v,w)+k*np.eye(7);S=se(self.x,self.x,a,sw)
                power=2 if model=='ghp_beta' else 0
                C=linalg.block_diag(*[u*K+u**power*S for u in W])
                expected=stats.multivariate_normal.logpdf(self.Y.ravel(),cov=C)+stats.norm.logpdf(q[:5],center,2.).sum()
                if model!='gp':
                    dist=stats.invgamma(2.5,scale=1.5) if model=='stp' else stats.geninvgauss(1.,1.)
                    expected+=np.sum(dist.logpdf(W)+np.log(W))
                target=make_target([(self.x,self.Y)],model=INTERNAL[model],prior_center=center)
                value,gradient=target(z);np.testing.assert_allclose(value,expected,atol=1e-10)
                fast=make_target([(self.x,self.Y)],model=INTERNAL[model],prior_center=center,backend='shared_grid')
                np.testing.assert_allclose(fast(z)[0],value,atol=1e-10)
                np.testing.assert_allclose(fast(z)[1],gradient,atol=1e-9)
                steps=np.eye(len(z))*1e-5
                numerical=np.array([(target(z+d)[0]-target(z-d)[0])/2e-5 for d in steps])
                np.testing.assert_allclose(gradient,numerical,rtol=1e-5,atol=1e-7)
                # Correlated shared functions would instead have nonzero off-diagonal blocks.
                np.testing.assert_array_equal(C[:7,7:],0.)

    def test_function_mean_and_context_mode(self):
        xp=np.linspace(-.2,3.2,11);y=self.Y[0]
        for model in MODELS:
            with self.subTest(model=model):
                theta=native_center(self.center,model);v,a,k,w,sw=np.exp(theta)
                fn,report=context_function(y,self.x,xp,theta,model);W=report['W_for_function']
                K=se(self.x,self.x,v,w)+k*np.eye(len(y));S=se(self.x,self.x,a,sw)
                multiplier=W if model=='ghp_beta' else 1.
                C=W*K+multiplier**2*S
                cross=multiplier*se(xp,self.x,a,sw)
                expected=cross@np.linalg.solve(C,y)
                np.testing.assert_allclose(fn,expected,atol=1e-11)
                if model!='gp':
                    dist=stats.invgamma(2.5,scale=1.5) if model=='stp' else stats.geninvgauss(1.,1.)
                    def objective(t):
                        W=np.exp(t);r=2 if model=='ghp_beta' else 0
                        return stats.multivariate_normal.logpdf(y,cov=W*K+W**r*S)+dist.logpdf(W)+t
                    t=report['log_W_for_function']
                    np.testing.assert_allclose(report['log_posterior_at_mode'],objective(t),atol=1e-10)
                    grid=np.linspace(-10,10,201)
                    self.assertGreaterEqual(objective(t)+1e-8,max(map(objective,grid)))
                # Adding prediction locations cannot affect the context estimate.
                extra,_=context_function(y,self.x,np.r_[xp,15.],theta,model)
                np.testing.assert_array_equal(extra[:len(xp)],fn)

    def test_comparable_prior(self):
        center=native_center(self.center,'ghp_beta')
        np.testing.assert_allclose(np.exp(center[1])*gig_mean()**2,np.exp(self.center[1]))
        np.testing.assert_array_equal(native_center(self.center,'ghp_mu'),self.center)


if __name__=='__main__':
    with threadpool_limits(limits=1):unittest.main()
