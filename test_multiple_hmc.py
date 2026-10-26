"""Independent checks of hierarchical densities and new-function inference."""

import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy import stats, special, integrate

from GH_multiple_hmc import make_target, coordinates, context_log_weights, resample_context, MODELS
from GH_proc_joint_hmc import make_joint_target


class MultipleTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([0., .7, 1.3])
        self.y = np.array([[.3, -.6, .1], [1.2, .7, -.5]])
        self.theta = np.log([.3, .9, .08, 1.4, .2])
        self.center = np.array([-.5, 0., -2., .2, -.9])

    def test_density_gradient_and_jacobian_all_models(self):
        groups = [(self.x, self.y), (np.array([.1, .9]), np.array([[.2, .8]]))]
        for model in MODELS:
            with self.subTest(model=model):
                q = self.theta if model == 'gp' else np.r_[self.theta, np.log([.6, 1.3, 2.1])]
                z = coordinates(q, model)
                assert_allclose(coordinates(z, model, inverse=True), q)
                target = make_target(groups, model=model, prior_center=self.center)
                expected = stats.norm.logpdf(self.theta, self.center, 2.).sum()
                v, a, k, w, sw = np.exp(self.theta)
                offset = 0
                for x, ys in groups:
                    d2 = (x[:, None]-x[None, :])**2
                    K, S = v*np.exp(-w*d2/2), a*np.exp(-sw*d2/2)
                    for y in ys:
                        W = 1. if model == 'gp' else np.exp(q[5+offset])
                        C = W*(K+k*np.eye(len(x)))+W**(2 if model == 'beta_gp' else 0)*S
                        expected += stats.multivariate_normal.logpdf(y, cov=C)
                        if model != 'gp':
                            expected += (stats.invgamma.logpdf(W, 2.5, scale=1.5) if model == 'tp_mu_gp'
                                         else stats.geninvgauss.logpdf(W, 1., 1.))+np.log(W)
                        offset += 1
                assert_allclose(target(z)[0], expected, atol=1e-10)
                numeric = [(target(z+d)[0]-target(z-d)[0])/2e-5 for d in np.eye(len(z))*1e-5]
                assert_allclose(target(z)[1], numeric, rtol=2e-6, atol=1e-7)

    def test_one_realization_matches_existing_target(self):
        q = np.r_[self.theta, np.log(1.7)]
        for model in MODELS[1:]:
            z = coordinates(q, model)
            new = make_target([(self.x, self.y[:1])], model=model, prior_center=self.center)
            old = make_joint_target(self.y[0], self.x, model=model, prior_center=self.center)
            assert_allclose(new(z)[0], old(z)[0], atol=1e-11)
            assert_allclose(new(z)[1], old(z)[1], atol=1e-11)

    def test_shared_grid_backend_matches_dense_density_and_gradient(self):
        rng = np.random.default_rng(917)
        x = np.linspace(0., 10., 27)
        groups = [(x, rng.normal(size=(8, len(x)))), (self.x, self.y)]
        for model in MODELS[1:]:
            for noise in (.08, 1e-4):
                q = np.r_[self.theta.copy(), np.linspace(-1.7, 1.3, 10)]
                q[2] = np.log(noise)
                z = coordinates(q, model)
                dense = make_target(groups, model=model, prior_center=self.center)
                spectral = make_target(groups, model=model, prior_center=self.center,
                                       backend='shared_grid')
                assert_allclose(spectral(z)[0], dense(z)[0], rtol=2e-8, atol=1e-7)
                assert_allclose(spectral(z)[1], dense(z)[1], rtol=3e-7, atol=1e-5)
                if noise == .08:
                    numeric = [(spectral(z+d)[0]-spectral(z-d)[0])/2e-5
                               for d in np.eye(len(z))*1e-5]
                    assert_allclose(spectral(z)[1], numeric, rtol=3e-5, atol=2e-5)

    def test_context_quadrature_against_independent_scalar_integration(self):
        d2 = (self.x[:, None]-self.x[None, :])**2
        v, a, k, w, sw = np.exp(self.theta)
        K = v*np.exp(-w*d2/2)+k*np.eye(len(self.x))
        S = a*np.exp(-sw*d2/2)
        for model in MODELS:
            t, logw = context_log_weights(self.theta[None], self.x, self.y, model=model)
            spectral_t, spectral_logw = context_log_weights(
                self.theta[None], self.x, self.y, model=model, backend='shared_grid')
            assert_allclose(spectral_t, t)
            assert_allclose(spectral_logw, logw, rtol=2e-10, atol=1e-9)
            for j in range(2):
                if model == 'gp':
                    expected = stats.multivariate_normal.pdf(self.y[j], cov=K+S)
                else:
                    def density(t):
                        W = np.exp(t)
                        logprior = stats.invgamma.logpdf(W, 2.5, scale=1.5) if model == 'tp_mu_gp' else stats.geninvgauss.logpdf(W, 1., 1.)
                        C = W*K+W**(2 if model == 'beta_gp' else 0)*S
                        return np.exp(stats.multivariate_normal.logpdf(self.y[j], cov=C)+logprior+t)
                    expected, _ = integrate.quad(density, -10., 14., epsabs=1e-11, epsrel=1e-9)
                assert_allclose(np.exp(special.logsumexp(logw[j])), expected, rtol=1e-8)

    def test_smooth_low_noise_covariances_in_both_component_orders(self):
        x = np.linspace(0., 10., 100)
        y = np.array([.4*np.sin(x/3)+.01*np.cos(2*x), .3*np.cos(x/4)])
        for model in MODELS[1:]:
            for swapped in [False, True]:
                theta = np.log([.0005, .5, 1e-9, 6., .1])
                if swapped:
                    theta[[0, 1, 3, 4]] = theta[[1, 0, 4, 3]]
                z = coordinates(np.r_[theta, np.log([.4, 3.])], model)
                dense = make_target([(x, y)], model=model, prior_center=self.center)
                spectral = make_target([(x, y)], model=model, prior_center=self.center,
                                       backend='shared_grid')
                direct_value, direct_grad = dense(z)
                value, grad = spectral(z)
                self.assertTrue(np.isfinite(value))
                assert_allclose(value, direct_value, rtol=2e-6, atol=.002)
                assert_allclose(grad, direct_grad, rtol=.002, atol=.02)

    def test_resampling_updates_theta_and_preserves_scale_variation(self):
        theta = np.repeat(self.theta[None], 1000, axis=0)
        logw = np.log(np.tile([.2, .8], (1000, 1))/1000)
        q, diagnostics = resample_context(theta, logw, np.log([1., 4.]), model='mu_gp', draws=10000, seed=7)
        assert_allclose(np.exp(q[:, 5]).mean(), 3.4, atol=.02)
        self.assertGreater(np.std(q[:, 5]), .4)
        assert_allclose(diagnostics['theta_importance_ess'], 1000.)
        # A single function's context cannot use observations from another test function.
        other = self.y.copy()
        other[1] *= 100
        theta = np.vstack([self.theta, self.theta+.2])
        _, first = context_log_weights(theta, self.x, self.y, model='gp')
        _, second = context_log_weights(theta, self.x, other, model='gp')
        assert_allclose(first[0], second[0])


if __name__ == '__main__':
    unittest.main()
