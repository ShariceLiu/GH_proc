"""Independent mathematical checks for the single-SE zero-mean baselines."""
import unittest
import numpy as np
from numpy.testing import assert_allclose
from scipy import integrate, stats
from threadpoolctl import threadpool_limits

from GH_zero_mean_hmc import (MODELS, coordinates, make_target,
                               conditional_prediction, predict_mixture)


class ZeroMeanTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([0., .4, 1.2])
        self.y = np.array([.6, -.2, .9])
        self.xp = np.array([.2, .8])
        self.q = np.log([.7, .12, 1.1, 1.7])
        self.center = np.array([-.4, -2., .2])
        self.sd = np.array([1., 1.4, 1.6])

    def test_density_and_gradient_all_models(self):
        v, k, w, W = np.exp(self.q)
        base = v*np.exp(-.5*w*(self.x[:, None]-self.x[None, :])**2)+k*np.eye(len(self.x))
        for model in MODELS:
            with self.subTest(model=model):
                q = self.q[:3] if model == 'gp_zero' else self.q
                z = coordinates(q, model)
                assert_allclose(coordinates(z, model, inverse=True), q)
                target = make_target(self.y, self.x, model=model, prior_center=self.center,
                                     prior_sd=self.sd, l=.7, omega=1.3, nu=5.)
                covariance = base if model == 'gp_zero' else W*base
                expected = stats.multivariate_normal.logpdf(self.y, cov=covariance)
                expected += stats.norm.logpdf(q[:3], self.center, self.sd).sum()
                if model == 'stp_zero':
                    expected += stats.invgamma.logpdf(W, 2.5, scale=1.5)+np.log(W)
                elif model == 'ghp_zero':
                    expected += stats.geninvgauss.logpdf(W, .7, 1.3)+np.log(W)
                assert_allclose(target(z)[0], expected, atol=1e-12)
                delta = 1e-5*np.eye(len(q))
                numerical = [(target(z+d)[0]-target(z-d)[0])/2e-5 for d in delta]
                assert_allclose(target(z)[1], numerical, rtol=1e-6, atol=1e-7)

    def test_stp_marginal_likelihood_matches_exact_student_t(self):
        target = make_target(self.y, self.x, model='stp_zero', prior_center=self.center, prior_sd=self.sd)
        log_prior = stats.norm.logpdf(self.q[:3], self.center, self.sd).sum()
        v, k, w = np.exp(self.q[:3])
        C = v*np.exp(-.5*w*(self.x[:, None]-self.x[None, :])**2)+k*np.eye(len(self.x))
        expected = stats.multivariate_t.logpdf(self.y, loc=np.zeros(3), shape=3./5.*C, df=5.)
        def integrand(t):
            z = coordinates(np.r_[self.q[:3], t], 'stp_zero')
            return np.exp(target(z)[0]-log_prior)
        probability, _ = integrate.quad(integrand, -15., 20., epsabs=1e-12, epsrel=1e-10)
        assert_allclose(np.log(probability), expected, atol=1e-10)

    def test_gaussian_conditionals_and_W_scaling(self):
        v, k, w, W = np.exp(self.q)
        all_x = np.r_[self.x, self.xp]
        K = v*np.exp(-.5*w*(all_x[:, None]-all_x[None, :])**2)
        for model in MODELS:
            q = self.q[:3] if model == 'gp_zero' else self.q
            factor = 1. if model == 'gp_zero' else W
            T = factor*K
            C = T[:3, :3]+factor*k*np.eye(3)
            mean = T[3:, :3]@np.linalg.solve(C, self.y)
            covariance = T[3:, 3:]-T[3:, :3]@np.linalg.solve(C, T[:3, 3:])
            p = conditional_prediction(self.y, self.x, self.xp, q, model=model)
            assert_allclose(p['f_mean'], mean, atol=1e-12)
            assert_allclose(p['f_cov'], covariance, atol=1e-12)
            assert_allclose(p['y_cov'], covariance+factor*k*np.eye(2), atol=1e-12)
            zero = conditional_prediction(np.zeros(3), self.x, self.xp, q, model=model)
            assert_allclose(zero['f_mean'], 0., atol=1e-14)
        p1 = conditional_prediction(self.y, self.x, self.xp, self.q, model='ghp_zero')
        q2 = self.q.copy()
        q2[-1] += np.log(2.)
        p2 = conditional_prediction(self.y, self.x, self.xp, q2, model='ghp_zero')
        assert_allclose(p1['f_mean'], p2['f_mean'], atol=1e-12)
        assert_allclose(2.*p1['f_cov'], p2['f_cov'], atol=1e-12)

    def test_mixture_moments_and_quantiles(self):
        q2 = self.q+np.array([.3, -.6, .2, .5])
        fit = dict(model='ghp_zero', log_samples=np.array([[self.q], [q2]]))
        p = predict_mixture(self.y, self.x, self.xp, fit, interval_batch_size=1)
        components = [conditional_prediction(self.y, self.x, self.xp, q, model='ghp_zero')
                      for q in [self.q, q2]]
        mean = sum(c['f_mean'] for c in components)/2
        assert_allclose(p['f_mean'], mean)
        for name in ['f', 'y']:
            covariance = sum(c[f'{name}_cov']+np.outer(c['f_mean']-mean, c['f_mean']-mean)
                             for c in components)/2
            assert_allclose(p[f'{name}_cov'], covariance)
            for side, probability in enumerate([.025, .975]):
                cdf = stats.norm.cdf((p[f'{name}_ci'][:, side]-p['component_means'])
                                     /np.sqrt(p[f'component_{name}_variances'])).mean(axis=0)
                assert_allclose(cdf, probability, atol=1e-10)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        unittest.main()
