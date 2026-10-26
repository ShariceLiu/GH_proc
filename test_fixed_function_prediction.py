"""Independent checks of conjugacy, GH scaling, and observation uncertainty."""
import unittest

import numpy as np
from scipy import linalg, special, stats
from threadpoolctl import threadpool_limits

from GH_fixed_function import fixed_function_prediction, marginal_distribution, mixing_quadrature, StableGHMarginals, refine_density_quadrature


class FixedFunctionPredictionTests(unittest.TestCase):
    def test_large_negative_gig_order(self):
        from scipy import integrate
        from GH_fixed_function import gig_moments, log_scaled_bessel_k
        for p,chi,psi in [(-1187.,2400.,1.),(-1187.5,2400.,3.),(-1187.,2400.,3000.)]:
            center=.5*np.log(chi/psi)+np.arcsinh(p/np.sqrt(chi*psi))
            def logu(t):
                return p*t-.5*(chi*np.exp(-t)+psi*np.exp(t))
            top=logu(center)
            integral=integrate.quad(lambda t:np.exp(logu(t)-top),center-2,center+2,
                                    epsabs=1e-13,epsrel=1e-12)[0]
            lognorm=top+np.log(integral)
            analytic=np.log(2)+.5*p*np.log(chi/psi)+log_scaled_bessel_k(p,np.sqrt(chi*psi))-np.sqrt(chi*psi)
            np.testing.assert_allclose(analytic,lognorm,atol=3e-10,rtol=0)
            ew,varw=gig_moments(p,chi,psi)
            mean=integrate.quad(lambda t:np.exp(t+logu(t)-lognorm),center-2,center+2,
                                epsabs=1e-12,epsrel=1e-12)[0]
            variance=integrate.quad(lambda t:(np.exp(t)-mean)**2*np.exp(logu(t)-lognorm),
                                    center-2,center+2,epsabs=1e-13,epsrel=1e-12)[0]
            np.testing.assert_allclose([ew,varw],[mean,variance],rtol=1e-9,atol=1e-12)
            prediction=dict(mixing='gig',W_lambda=p,W_chi=chi,W_psi=psi,W_mean=ew,W_var=varw,
                location=np.array([.1,-.3]),conditional_beta=np.array([0.,1.5]),
                y_scale_cov=np.diag([.4,.2]))
            prediction['y_mean']=prediction['location']+ew*prediction['conditional_beta']
            prediction['y_var']=ew*np.diag(prediction['y_scale_cov'])+varw*prediction['conditional_beta']**2
            nodes,weights,_=mixing_quadrature(prediction)
            dist=StableGHMarginals(prediction,'y')
            for point in (dist.mean(),dist.mean()+3*np.sqrt(dist.var())):
                logdensity=special.logsumexp(np.log(weights[:,None])+stats.norm.logpdf(point,
                    prediction['location']+nodes[:,None]*prediction['conditional_beta'],
                    np.sqrt(nodes[:,None]*np.diag(prediction['y_scale_cov']))),axis=0)
                np.testing.assert_allclose(dist.logpdf(point),logdensity,atol=1e-8,rtol=0)
            np.testing.assert_allclose(dist.cdf(dist.ppf(.975)),.975,atol=1e-10)

    def setUp(self):
        self.x = np.array([0., .4, 1.1, 1.8])
        self.xp = np.array([.2, .9, 2.3])
        self.y = np.array([.3, -.2, .8, 1.4])
        self.mu1, self.mu2 = .1+.2*self.x, .1+.2*self.xp
        self.beta1, self.beta2 = -.3+.7*self.x, -.3+.7*self.xp

    def test_gh_conjugacy_and_mixture(self):
        p = fixed_function_prediction(self.y, self.x, self.xp, v0=.8, k=.12, w=1.3,
                    mu_train=self.mu1, mu_pred=self.mu2, beta_train=self.beta1, beta_pred=self.beta2,
                    lam=.7, omega=1.4)
        W, weights, _ = mixing_quadrature(p)
        np.testing.assert_allclose(weights@W, p['W_mean'], rtol=1e-9)
        np.testing.assert_allclose(weights@(W-p['W_mean'])**2, p['W_var'], rtol=1e-9)
        A = .8*np.exp(-.65*(self.x[:, None]-self.x)**2)+.12*np.eye(len(self.x))
        b = np.sqrt(p['W_chi']*p['W_psi'])
        posterior = stats.geninvgauss(p['W_lambda'], b, scale=np.sqrt(p['W_chi']/p['W_psi']))
        log_ratios = []
        for value in [.2, .9, 3.]:
            likelihood = stats.multivariate_normal.logpdf(self.y, self.mu1+value*self.beta1, value*A)
            prior = stats.geninvgauss.logpdf(value, .7, 1.4)
            log_ratios.append(likelihood+prior-posterior.logpdf(value))
        np.testing.assert_allclose(log_ratios, np.repeat(log_ratios[0], 3), atol=1e-10)
        cross = .8*np.exp(-.65*(self.xp[:, None]-self.x)**2)
        Kpp = .8*np.exp(-.65*(self.xp[:, None]-self.xp)**2)
        for value in [.2, 3.]:
            mean = self.mu2+value*self.beta2+cross@linalg.solve(A, self.y-self.mu1-value*self.beta1)
            covariance = value*(Kpp-cross@linalg.solve(A, cross.T))
            np.testing.assert_allclose(mean, p['location']+value*p['conditional_beta'], atol=1e-12)
            np.testing.assert_allclose(covariance, value*p['f_scale_cov'], atol=1e-12)
        for name in ('f', 'y'):
            dist = marginal_distribution(p, name)
            means = p['location']+W[:, None]*p['conditional_beta']
            variance = W[:, None]*np.diag(p[f'{name}_scale_cov'])
            points = np.array([-.5, .4, 2.])
            logpdf = special.logsumexp(np.log(weights[:, None])+stats.norm.logpdf(points, means, np.sqrt(variance)), axis=0)
            np.testing.assert_allclose(logpdf, dist.logpdf(points), atol=1e-8)
            np.testing.assert_allclose(dist.mean(), p[f'{name}_mean'], atol=1e-10)
            np.testing.assert_allclose(dist.var(), p[f'{name}_var'], atol=1e-10)
        np.testing.assert_allclose(p['y_cov']-p['f_cov'], .12*p['W_mean']*np.eye(3), atol=1e-12)

    def test_student_t_and_gp_limits(self):
        for mixing in ('ig', 'gp'):
            p = fixed_function_prediction(self.y, self.x, self.xp, v0=.8, k=.12, w=1.3,
                    mu_train=self.mu1, mu_pred=self.mu2, mixing=mixing)
            W, weights, _ = mixing_quadrature(p)
            dist = marginal_distribution(p)
            np.testing.assert_allclose(weights@W, p['W_mean'], rtol=1e-9)
            np.testing.assert_allclose(dist.mean(), p['f_mean'], atol=1e-12)
            np.testing.assert_allclose(dist.var(), p['f_var'], atol=1e-12)
            np.testing.assert_allclose(dist.cdf(p['f_ci'][:, 0]), .025, atol=1e-10)
            if mixing == 'ig':
                self.assertEqual(p['W_shape'], 4.5)
                self.assertGreater(p['W_var'], 0.)
            else:
                self.assertEqual(p['W_var'], 0.)

    def test_stable_gh_matches_scipy_and_handles_large_bessel_argument(self):
        from scipy import integrate
        ordinary = fixed_function_prediction(self.y,self.x,self.xp,v0=.8,k=.12,w=1.3,
                                              beta_train=self.beta1,beta_pred=self.beta2)
        stable, reference = StableGHMarginals(ordinary,'y'), marginal_distribution(ordinary,'y')
        points = np.array([-.5,.4,2.])
        np.testing.assert_allclose(stable.logpdf(points),reference.logpdf(points),atol=1e-10)
        np.testing.assert_allclose(stable.cdf(points),reference.cdf(points),atol=1e-10)
        np.testing.assert_allclose(stable.ppf(.975),reference.ppf(.975),atol=1e-9)
        np.testing.assert_allclose(stable.stats(),reference.stats(moments='s'),atol=1e-9)
        beta1,beta2 = 10+5*self.x,10+5*self.xp
        p = fixed_function_prediction(2*beta1,self.x,self.xp,v0=.05,k=.001,w=1.3,
                                      beta_train=beta1,beta_pred=beta2)
        dist = marginal_distribution(p,'y')
        self.assertIsInstance(dist,StableGHMarginals)
        self.assertGreater(np.sqrt(p['W_chi']*p['W_psi']),500)
        W,weights,_ = mixing_quadrature(p)
        means = p['location']+W[:,None]*p['conditional_beta']
        variances = W[:,None]*np.diag(p['y_scale_cov'])
        np.testing.assert_allclose(weights@means,p['y_mean'],rtol=1e-10)
        np.testing.assert_allclose(weights@(variances+(means-p['y_mean'])**2),p['y_var'],rtol=1e-8)
        points = p['y_mean']+np.sqrt(p['y_var'])*np.array([-3.,0.,4.])
        mixture = special.logsumexp(np.log(weights[:,None])+stats.norm.logpdf(points,means,np.sqrt(variances)),axis=0)
        np.testing.assert_allclose(dist.logpdf(points),mixture,atol=1e-8)
        sd = np.sqrt(p['y_var'])
        for j in range(3):
            mass = integrate.quad(lambda z: np.exp(dist.logpdf(p['y_mean']+z*sd)[j])*sd[j],-12,12,epsabs=1e-10)[0]
            self.assertAlmostEqual(mass,1.,places=8)
        np.testing.assert_allclose(dist.cdf(p['y_ci'][:,0]),.025,atol=1e-10)
        np.testing.assert_allclose(dist.cdf(p['y_ci'][:,1]),.975,atol=1e-10)

    def test_gh_skewness_before_individual_bessel_underflow(self):
        import mpmath as mp
        p=fixed_function_prediction(self.y,self.x,self.xp,v0=.8,k=.12,w=1.3,
                                     beta_train=self.beta1,beta_pred=self.beta2,omega=350.)
        argument=np.sqrt(p['W_chi']*p['W_psi'])
        self.assertTrue(100<argument<500)
        dist=marginal_distribution(p,'y');self.assertIsInstance(dist,StableGHMarginals)
        with mp.workdps(70):
            chi,psi,order=map(mp.mpf,[p['W_chi'],p['W_psi'],p['W_lambda']])
            z=mp.sqrt(chi*psi)
            moments=[(chi/psi)**(mp.mpf(j)/2)*mp.besselk(order+j,z)/mp.besselk(order,z) for j in (1,2,3)]
            m1,m2,m3=moments;variance=m2-m1*m1;central3=m3-3*m1*m2+2*m1**3
            expected=[]
            for b,c in zip(p['conditional_beta'],np.diag(p['y_scale_cov'])):
                b,c=mp.mpf(b),mp.mpf(c)
                expected.append(float((b**3*central3+3*b*c*variance)/(c*m1+b*b*variance)**mp.mpf('1.5')))
        np.testing.assert_allclose(dist.stats(moments='s'),expected,atol=1e-10,rtol=1e-9)

    def test_vectorized_intervals_and_remote_density_quadrature(self):
        for beta in (False, True):
            kwargs = dict(beta_train=self.beta1, beta_pred=self.beta2) if beta else {}
            p = fixed_function_prediction(self.y, self.x, self.xp, v0=.8, k=.12, w=1.3,
                                          gh_backend='quadrature', **kwargs)
            reference = dict(p); reference.pop('gh_backend')
            scipy_dist = marginal_distribution(reference, 'y')
            np.testing.assert_allclose(p['y_ci'], np.column_stack([scipy_dist.ppf(.025), scipy_dist.ppf(.975)]), atol=1e-8)
            truth = p['y_mean']+np.sqrt(p['y_var'])*np.array([-50., 0., 70.])
            refine_density_quadrature(p, truth)
            W, weights, _ = mixing_quadrature(p)
            lp = special.logsumexp(np.log(weights[:, None])+stats.norm.logpdf(truth,
                p['location']+W[:, None]*p['conditional_beta'], np.sqrt(W[:, None]*np.diag(p['y_scale_cov']))), axis=0)
            np.testing.assert_allclose(lp, scipy_dist.logpdf(truth), atol=1e-7, rtol=1e-9)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        unittest.main()
