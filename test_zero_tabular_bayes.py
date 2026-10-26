import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy import stats,integrate
from threadpoolctl import threadpool_limits

from GH_zero_tabular_bayes import (ZeroPosterior,Proposal,sample_chain,mixture_crps,
    independent_crps,mixture_logpdf,mixture_quantile,mixture_cdf)
from GH_tabular_hmc import make_target
from predictive_crps import gaussian_mixture_crps_closed_form
from shared_beta_plugin_experiment import student_crps


class BayesTests(unittest.TestCase):
    def test_normalized_targets_gradients_and_conditioning(self):
        rng=np.random.default_rng(52)
        x=rng.normal(size=(9,2));y=rng.normal(size=9);xp=rng.normal(size=(4,2))
        q=np.log([.8,.12,.7,1.3])
        K=lambda a,b:.8*np.exp(-.5*np.sum((a[:,None]-b[None,:])**2*[.7,1.3],axis=2))
        A=K(x,x)+.12*np.eye(len(x));Q=y@np.linalg.solve(A,y)
        for model in ('gp_zero','stp_zero'):
            target=ZeroPosterior(y,x,model)
            p=target.evaluate(q,gradient=True)
            prior=stats.norm.logpdf(q,target.center,2.).sum()
            likelihood=(stats.multivariate_normal.logpdf(y,cov=A) if model=='gp_zero' else
                        stats.multivariate_t.logpdf(y,shape=3/5*A,df=5))
            np.testing.assert_allclose(p['logp'],likelihood+prior,atol=1e-11)
            h=1e-5
            numerical=[(target.evaluate(q+e*h)['logp']-target.evaluate(q-e*h)['logp'])/(2*h) for e in np.eye(len(q))]
            np.testing.assert_allclose(p['gradient'],numerical,atol=1e-7,rtol=1e-6)
            pred=target.predict(q,xp)
            cross=K(xp,x);mean=cross@np.linalg.solve(A,y)
            variance=np.diag(K(xp,xp)-cross@np.linalg.solve(A,cross.T))+.12
            scale=1. if model=='gp_zero' else (3+Q)/(5+len(y))
            np.testing.assert_allclose(pred['mean'],mean,atol=1e-12)
            np.testing.assert_allclose(pred['y_scale2'],scale*variance,atol=1e-12)
            if model=='gp_zero':
                reference=make_target(y,x,model=model,prior_center=target.center,backend='condensed')
                np.testing.assert_allclose(reference(q)[0],p['logp'],atol=1e-12)
                np.testing.assert_allclose(reference(q)[1],p['gradient'],atol=1e-12)
            else:
                shape,rate=pred['W_shape'],pred['W_scale']
                for value in [-.3,1.4]:
                    numerical=integrate.quad(lambda t:np.exp(stats.invgamma.logpdf(np.exp(t),shape,scale=rate)+t+
                        stats.norm.logpdf(value,mean[0],np.sqrt(np.exp(t)*variance[0]))),-12,12,epsabs=1e-12)[0]
                    expected=stats.t.pdf(value,2*shape,loc=mean[0],scale=np.sqrt(pred['y_scale2'][0]))
                    np.testing.assert_allclose(numerical,expected,rtol=1e-10)

    def test_proposal_density_and_mh_resume(self):
        mean=np.array([.3,-1.2]);cov=np.array([[1.,.75],[.75,2.]])
        components=[dict(weight=.8,mean=[0.,0.],shape=(1.2*cov).tolist(),df=None),
                    dict(weight=.2,mean=[.1,-.2],shape=(2*cov).tolist(),df=5)]
        proposal=Proposal(components)
        for q in [np.zeros(2),np.array([3.,-4.])]:
            expected=.8*stats.multivariate_normal.pdf(q,mean=[0.,0.],cov=1.2*cov)+.2*stats.multivariate_t.pdf(q,loc=[.1,-.2],shape=2*cov,df=5)
            np.testing.assert_allclose(proposal.logpdf(q),np.log(expected),atol=1e-13)
        normal=stats.multivariate_normal(mean,cov)
        target=lambda q:dict(logp=float(normal.logpdf(q)),quadratic=float(q@q))
        direct=sample_chain(target,proposal,mean,count=400,seed=712)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'checkpoint.npz'
            sample_chain(target,proposal,mean,count=120,seed=712,checkpoint=checkpoint,identity='test')
            resumed=sample_chain(target,proposal,mean,count=400,seed=712,checkpoint=checkpoint,identity='test')
            for key in direct: np.testing.assert_array_equal(resumed[key],direct[key])
        full=sample_chain(target,proposal,mean,count=16000,seed=27)
        samples=full['q'][1000:]
        np.testing.assert_allclose(samples.mean(0),mean,atol=.06)
        np.testing.assert_allclose(np.cov(samples.T),cov,atol=.12)
        rejected=~full['accepted'][1:]
        np.testing.assert_array_equal(full['q'][1:][rejected],full['q'][:-1][rejected])
        # A fixed mixture of local and independent MH kernels must target the
        # same correlated Gaussian, including after an interrupted checkpoint.
        options=dict(local_covariance=.8*cov,local_probability=.3)
        hybrid=sample_chain(target,proposal,mean,count=16000,seed=28,**options)
        np.testing.assert_allclose(hybrid['q'][1000:].mean(0),mean,atol=.07)
        np.testing.assert_allclose(np.cov(hybrid['q'][1000:].T),cov,atol=.15)
        direct=sample_chain(target,proposal,mean,count=400,seed=713,**options)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint=Path(directory)/'hybrid.npz'
            sample_chain(target,proposal,mean,count=120,seed=713,checkpoint=checkpoint,identity='hybrid',**options)
            resumed=sample_chain(target,proposal,mean,count=400,seed=713,checkpoint=checkpoint,identity='hybrid',**options)
            for key in direct:np.testing.assert_array_equal(resumed[key],direct[key])

    def test_mixture_scores_and_quantiles(self):
        means=np.array([[-.5,.2],[.7,1.3],[1.1,-.3]])
        scales=np.array([[.8,.3],[.5,1.1],[.9,.7]])
        weights=np.array([.2,.3,.5]);truth=np.array([-.7,1.8])
        for df in (None,9.,2381.):
            score,info=mixture_crps(truth,means,scales,weights,df)
            for j in range(2):
                reference,_=independent_crps(truth[j],means[:,j],scales[:,j],weights,df)
                np.testing.assert_allclose(score[j],reference,atol=3e-8)
            for probability in (.025,.975):
                endpoint=mixture_quantile(probability,means,scales,weights,df)
                np.testing.assert_allclose(mixture_cdf(endpoint,means,scales,weights,df),probability,atol=1e-12)
            density=(stats.norm.pdf(truth,means,scales) if df is None else stats.t.pdf(truth,df,loc=means,scale=scales))
            np.testing.assert_allclose(mixture_logpdf(truth,means,scales,weights,df),np.log(weights@density),atol=1e-11)
            if df is None:
                np.testing.assert_allclose(score,gaussian_mixture_crps_closed_form(truth,means,scales**2,weights),atol=3e-8)
            else:
                single,_=mixture_crps(truth,means[:1],scales[:1],np.ones(1),df)
                np.testing.assert_allclose(single,student_crps(truth,means[0],scales[0],df),atol=3e-8)


if __name__=='__main__':
    with threadpool_limits(limits=1):unittest.main()
