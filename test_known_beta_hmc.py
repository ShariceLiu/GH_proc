"""Known-beta likelihood/gradient checks independent of the implementation."""
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy import stats
from threadpoolctl import threadpool_limits

from GH_zero_mean_hmc import (KNOWN_BETA_MODEL, coordinates, make_target,
                               conditional_prediction, predict_mixture)


class KnownBetaTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([0., .4, 1.2])
        self.y = np.array([.6, -.2, .9])
        self.beta = np.array([.5, -.2, .3])
        self.xp = np.array([.2, .8])
        self.bp = np.array([-.1, .7])
        self.q = np.log([.7, .12, 1.1, 1.7])
        self.center = np.array([-.4, -2., .2])
        self.sd = np.array([1., 1.4, 1.6])

    def test_density_and_gradient_include_W_dependent_mean(self):
        v, k, w, W = np.exp(self.q)
        C = W*(v*np.exp(-.5*w*(self.x[:, None]-self.x[None, :])**2)+k*np.eye(3))
        target = make_target(self.y, self.x, model=KNOWN_BETA_MODEL, beta=self.beta,
                             prior_center=self.center, prior_sd=self.sd, l=.7, omega=1.3)
        z = coordinates(self.q, KNOWN_BETA_MODEL)
        expected = (stats.multivariate_normal.logpdf(self.y, mean=W*self.beta, cov=C)
                    +stats.norm.logpdf(self.q[:3], self.center, self.sd).sum()
                    +stats.geninvgauss.logpdf(W, .7, 1.3)+np.log(W))
        assert_allclose(target(z)[0], expected, atol=1e-12)
        delta = np.eye(4)*1e-5
        numeric = [(target(z+d)[0]-target(z-d)[0])/2e-5 for d in delta]
        assert_allclose(target(z)[1], numeric, atol=1e-7, rtol=1e-6)

    def test_nonzero_mean_gaussian_conditioning(self):
        v, k, w, W = np.exp(self.q)
        all_x = np.r_[self.x, self.xp]
        K = W*v*np.exp(-.5*w*(all_x[:, None]-all_x[None, :])**2)
        C = K[:3, :3]+W*k*np.eye(3)
        expected = W*self.bp+K[3:, :3]@np.linalg.solve(C, self.y-W*self.beta)
        covariance = K[3:, 3:]-K[3:, :3]@np.linalg.solve(C, K[:3, 3:])
        p = conditional_prediction(self.y, self.x, self.xp, self.q, model=KNOWN_BETA_MODEL,
                                    beta_train=self.beta, beta_pred=self.bp)
        assert_allclose(p['f_mean'], expected, atol=1e-12)
        assert_allclose(p['f_cov'], covariance, atol=1e-12)
        assert_allclose(p['y_cov'], covariance+W*k*np.eye(2), atol=1e-12)
        at_mean = conditional_prediction(W*self.beta, self.x, self.xp, self.q, model=KNOWN_BETA_MODEL,
                                          beta_train=self.beta, beta_pred=self.bp)
        assert_allclose(at_mean['f_mean'], W*self.bp, atol=1e-12)

    def test_zero_beta_reduces_to_existing_GHP_baseline(self):
        target = make_target(self.y, self.x, model=KNOWN_BETA_MODEL, beta=np.zeros(3), prior_center=self.center)
        old = make_target(self.y, self.x, model='ghp_zero', prior_center=self.center)
        z = coordinates(self.q, KNOWN_BETA_MODEL)
        assert_allclose(target(z)[0], old(z)[0], atol=1e-12)
        assert_allclose(target(z)[1], old(z)[1], atol=1e-12)
        new_p = conditional_prediction(self.y, self.x, self.xp, self.q, model=KNOWN_BETA_MODEL,
                                       beta_train=np.zeros(3), beta_pred=np.zeros(2))
        old_p = conditional_prediction(self.y, self.x, self.xp, self.q, model='ghp_zero')
        for key in ['f_mean', 'f_cov', 'y_cov']:
            assert_allclose(new_p[key], old_p[key], atol=1e-12)

    def test_mixture_includes_W_uncertainty_and_validates_beta(self):
        q2 = self.q+np.array([.3, -.4, .2, .5])
        fit = dict(model=KNOWN_BETA_MODEL, log_samples=np.array([[self.q], [q2]]))
        p = predict_mixture(self.y, self.x, self.xp, fit, beta_train=self.beta, beta_pred=self.bp)
        cs = [conditional_prediction(self.y, self.x, self.xp, q, model=KNOWN_BETA_MODEL,
                                     beta_train=self.beta, beta_pred=self.bp) for q in [self.q, q2]]
        mean = sum(c['f_mean'] for c in cs)/2
        covariance = sum(c['f_cov']+np.outer(c['f_mean']-mean, c['f_mean']-mean) for c in cs)/2
        assert_allclose(p['f_mean'], mean)
        assert_allclose(p['f_cov'], covariance)
        W = np.exp([self.q[-1], q2[-1]])
        assert_allclose(p['effect_mean'], W.mean()*self.bp)
        assert_allclose(p['effect_cov'], W.var()*np.outer(self.bp, self.bp))
        for side, probability in enumerate([.025, .975]):
            cdf = stats.norm.cdf((p['f_ci'][:, side]-p['component_means'])
                                 /np.sqrt(p['component_f_variances'])).mean(axis=0)
            assert_allclose(cdf, probability, atol=1e-10)
        with self.assertRaises(ValueError):
            make_target(self.y, self.x, model=KNOWN_BETA_MODEL, prior_center=self.center)
        with self.assertRaises(ValueError):
            conditional_prediction(self.y, self.x, self.xp, self.q, model=KNOWN_BETA_MODEL,
                                    beta_train=self.beta, beta_pred=[.2])


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        unittest.main()
