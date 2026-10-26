"""Density, gradient, conditioning and 1D checks for the one-SE ARD models."""
import unittest
import numpy as np
from numpy.testing import assert_allclose
from scipy import stats
from threadpoolctl import threadpool_limits

from GH_tabular_hmc import ZERO_MODELS, coordinates, make_target, conditional_marginals, default_center, sample_parameters
from GH_zero_mean_hmc import make_target as old_target, conditional_prediction as old_prediction


class TabularZeroTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[0., .2], [.4, -.8], [1.2, .6], [.7, .9]])
        self.xp = np.array([[.3, .1], [.8, -.4]])
        self.y = np.array([.6, -.2, .9, -.3])
        self.q = np.log([.7, .12, 1.1, .5, 1.7])
        self.center = np.array([-.5, -2., .3, -.4])
        self.sd = np.array([.8, 1.1, 1.5, 2.])

    def test_normalized_density_gradient_and_conditioning(self):
        for model in ZERO_MODELS:
            q = self.q[:-1] if model == 'gp_zero' else self.q
            v, k, w0, w1 = np.exp(q[:4])
            W = 1. if model == 'gp_zero' else np.exp(q[-1])
            all_x = np.vstack([self.x, self.xp])
            delta = (all_x[:, None]-all_x[None, :])**2
            T = W*v*np.exp(-.5*np.sum(delta*[w0, w1], axis=-1))
            C = T[:4, :4]+W*k*np.eye(4)
            z = coordinates(q, model)
            assert_allclose(coordinates(z, model, inverse=True), q, atol=1e-14)
            target = make_target(self.y, self.x, model=model, prior_center=self.center,
                                 prior_sd=self.sd, l=.7, omega=1.3)
            expected = stats.multivariate_normal.logpdf(self.y, cov=C)
            expected += stats.norm.logpdf(q[:4], self.center, self.sd).sum()
            if model == 'stp_zero':
                expected += stats.invgamma.logpdf(W, 2.5, scale=1.5)+np.log(W)
            elif model == 'ghp_zero':
                expected += stats.geninvgauss.logpdf(W, .7, 1.3)+np.log(W)
            assert_allclose(target(z)[0], expected, atol=1e-11)
            delta = np.eye(len(z))*1e-5
            numerical = [(target(z+d)[0]-target(z-d)[0])/2e-5 for d in delta]
            assert_allclose(target(z)[1], numerical, atol=1e-7, rtol=2e-6)
            pred = conditional_marginals(self.y, self.x, self.xp, q, model=model)
            cross = T[4:, :4]
            assert_allclose(pred['mean'], cross@np.linalg.solve(C, self.y), atol=1e-12)
            variance = np.diag(T[4:, 4:]-cross@np.linalg.solve(C, cross.T))
            assert_allclose(pred['f_variance'], variance, atol=1e-12)
            assert_allclose(pred['y_variance'], variance+W*k, atol=1e-12)

    def test_1d_matches_previous_zero_models(self):
        for model in ZERO_MODELS:
            q = self.q[[0, 1, 2]] if model == 'gp_zero' else self.q[[0, 1, 2, 4]]
            center = self.center[:3]
            target = make_target(self.y, self.x[:, 0], model=model, prior_center=center)
            old = old_target(self.y, self.x[:, 0], model=model, prior_center=center)
            z = coordinates(q, model)
            assert_allclose(target(z)[0], old(z)[0], atol=1e-11)
            assert_allclose(target(z)[1], old(z)[1], atol=1e-11)
            p = conditional_marginals(self.y, self.x[:, 0], self.xp[:, 0], q, model=model)
            previous = old_prediction(self.y, self.x[:, 0], self.xp[:, 0], q, model=model)
            assert_allclose(p['mean'], previous['f_mean'], atol=1e-12)
            assert_allclose(p['y_variance'], np.diag(previous['y_cov']), atol=1e-12)

    def test_priors_keep_existing_residual_and_noise_parameters(self):
        original = default_center(self.y, 2)
        for model in ZERO_MODELS:
            assert_allclose(default_center(self.y, 2, model), original[[0, 2, 3, 4]])

    def test_one_kernel_sampler_disables_component_swaps(self):
        for model in ZERO_MODELS:
            fit = sample_parameters(self.y, self.x, model=model, chains=2, warmup=50, draws=20,
                                    leapfrog_steps=(2, 3), verbose=False, seed=987)
            self.assertEqual(fit['log_samples'].shape, (2, 20, 4+(model != 'gp_zero')))
            self.assertFalse(fit['swap_accepted'].any())


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        unittest.main()
