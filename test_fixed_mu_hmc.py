"""Independent density, derivative and conditioning checks for fixed mu."""
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy import stats
from threadpoolctl import threadpool_limits

from GH_zero_mean_hmc import (FIXED_MU_MODEL, KNOWN_BETA_MODEL, coordinates, make_target,
                              conditional_prediction, predict_mixture)
from hmc_fixed_mu_30_experiment import gig_prior_mean


class FixedMuTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([0., .4, 1.2])
        self.y = np.array([.6, -.2, .9])
        self.mu = np.array([.5, -.2, .3])
        self.xp = np.array([.2, .8])
        self.mp = np.array([-.1, .7])
        self.q = np.log([.7, .12, 1.1, 1.7])
        self.center = np.array([-.4, -2., .2])

    def test_gig_mean_uses_prior_only(self):
        for l, omega in [(1., 1.), (-.5, 1.), (.3, 2.)]:
            assert_allclose(gig_prior_mean(l, omega), stats.geninvgauss.mean(l, omega), rtol=1e-12)

    def test_normalized_density_gradient_and_translation(self):
        v, k, w, W = np.exp(self.q)
        C = W*(v*np.exp(-.5*w*(self.x[:, None]-self.x[None, :])**2)+k*np.eye(3))
        target = make_target(self.y, self.x, model=FIXED_MU_MODEL, mu=self.mu,
                             prior_center=self.center, l=.7, omega=1.3)
        z = coordinates(self.q, FIXED_MU_MODEL)
        expected = (stats.multivariate_normal.logpdf(self.y, mean=self.mu, cov=C)
                    +stats.norm.logpdf(self.q[:3], self.center, 2.).sum()
                    +stats.geninvgauss.logpdf(W, .7, 1.3)+np.log(W))
        assert_allclose(target(z)[0], expected, atol=1e-12)
        delta = np.eye(4)*1e-5
        numeric = [(target(z+d)[0]-target(z-d)[0])/2e-5 for d in delta]
        assert_allclose(target(z)[1], numeric, atol=1e-7, rtol=1e-6)
        translated = make_target(self.y-self.mu, self.x, model='ghp_zero',
                                 prior_center=self.center, l=.7, omega=1.3)
        assert_allclose(target(z)[0], translated(z)[0], atol=1e-12)
        assert_allclose(target(z)[1], translated(z)[1], atol=1e-12)

    def test_conditioning_and_exact_beta_equivalence_at_one_W(self):
        v, k, w, W = np.exp(self.q)
        all_x = np.r_[self.x, self.xp]
        K = W*v*np.exp(-.5*w*(all_x[:, None]-all_x[None, :])**2)
        C = K[:3, :3]+W*k*np.eye(3)
        expected = self.mp+K[3:, :3]@np.linalg.solve(C, self.y-self.mu)
        covariance = K[3:, 3:]-K[3:, :3]@np.linalg.solve(C, K[:3, 3:])
        p = conditional_prediction(self.y, self.x, self.xp, self.q, model=FIXED_MU_MODEL,
                                    mu_train=self.mu, mu_pred=self.mp)
        assert_allclose(p['f_mean'], expected, atol=1e-12)
        assert_allclose(p['f_cov'], covariance, atol=1e-12)
        assert_allclose(p['y_cov'], covariance+W*k*np.eye(2), atol=1e-12)
        beta_case = conditional_prediction(self.y, self.x, self.xp, self.q, model=KNOWN_BETA_MODEL,
                                            beta_train=self.mu/W, beta_pred=self.mp/W)
        for key in p:
            assert_allclose(p[key], beta_case[key], atol=1e-12)

    def test_mixture_and_input_validation(self):
        q2 = self.q+np.array([.3, -.4, .2, .5])
        fit = dict(model=FIXED_MU_MODEL, log_samples=np.array([[self.q], [q2]]))
        p = predict_mixture(self.y, self.x, self.xp, fit, mu_train=self.mu, mu_pred=self.mp)
        cs = [conditional_prediction(self.y, self.x, self.xp, q, model=FIXED_MU_MODEL,
                                     mu_train=self.mu, mu_pred=self.mp) for q in [self.q, q2]]
        mean = sum(c['f_mean'] for c in cs)/2
        covariance = sum(c['f_cov']+np.outer(c['f_mean']-mean, c['f_mean']-mean) for c in cs)/2
        assert_allclose(p['f_mean'], mean)
        assert_allclose(p['f_cov'], covariance)
        assert_allclose(p['effect_mean'], self.mp)
        assert_allclose(p['effect_cov'], 0.)
        with self.assertRaises(ValueError):
            make_target(self.y, self.x, model=FIXED_MU_MODEL, prior_center=self.center)
        with self.assertRaises(ValueError):
            make_target(self.y, self.x, model=FIXED_MU_MODEL, mu=self.mu, beta=self.mu,
                         prior_center=self.center)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        unittest.main()
