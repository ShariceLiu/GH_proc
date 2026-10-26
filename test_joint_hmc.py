"""Independent mathematical checks for joint-W HMC and Gaussian regression."""

import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy import stats

from GH_proc_joint_hmc import (effective_coordinates, physical_coordinates,
                               make_joint_target, conditional_prediction, predict_joint_mixture)
from GH_proc_w_betaprior_hmc import sample_hmc


class TestJointHMC(unittest.TestCase):
    def setUp(self):
        self.x = np.array([0., 0.4, 1.2])
        self.y = np.array([0.6, -0.2, 0.9])
        self.q = np.log([0.7, 1.3, 0.12, 1.1, 0.3, 1.7])
        self.center = np.array([-0.4, 0.1, -2., 0.2, -1.2])
        self.sd = np.array([1., 1.2, 1.4, 1.6, 1.8])

    def test_target_matches_direct_density_including_log_W_jacobian(self):
        distances = (self.x[:, None]-self.x[None, :])**2
        v, a, k, w, sw, W = np.exp(self.q)
        for model, power in [('beta_gp', 2), ('mu_gp', 0)]:
            with self.subTest(model=model):
                target = make_joint_target(self.y, self.x, model=model, prior_center=self.center,
                                           prior_sd=self.sd, l=-0.5, omega=1.3)
                covariance = W*(v*np.exp(-0.5*w*distances)+k*np.eye(3))+W**power*a*np.exp(-0.5*sw*distances)
                expected = (stats.multivariate_normal.logpdf(self.y, cov=covariance)
                            +stats.norm.logpdf(self.q[:5], self.center, self.sd).sum()
                            +stats.geninvgauss.logpdf(W, -0.5, 1.3)+np.log(W))
                z = effective_coordinates(self.q, model)
                assert_allclose(physical_coordinates(z, model), self.q, atol=1e-14)
                assert_allclose(target(z)[0], expected, atol=1e-12)

    def test_analytic_gradient_for_both_models(self):
        for model in ['beta_gp', 'mu_gp']:
            target = make_joint_target(self.y, self.x, model=model,
                                       prior_center=self.center, prior_sd=self.sd)
            z = effective_coordinates(self.q, model)
            perturbations = np.eye(6)*1e-5
            numeric = [(target(z+d)[0]-target(z-d)[0])/2e-5 for d in perturbations]
            assert_allclose(target(z)[1], numeric, rtol=1e-6, atol=1e-7)

    def test_conditional_gaussian_direct_block_and_model_equivalence(self):
        xp = np.array([0.2, 0.8])
        all_x = np.r_[self.x, xp]
        distances = (all_x[:, None]-all_x[None, :])**2
        v, a, k, w, sw, W = np.exp(self.q)
        for model, power in [('beta_gp', 2), ('mu_gp', 0)]:
            H = W**power*a*np.exp(-0.5*sw*distances)
            T = W*v*np.exp(-0.5*w*distances)+H
            D = T[:3, :3]+W*k*np.eye(3)
            prediction = conditional_prediction(self.y, self.x, xp, self.q, model=model)
            for name, cross, prior in [('f', T[3:, :3], T[3:, 3:]), ('effect', H[3:, :3], H[3:, 3:])]:
                assert_allclose(prediction[f'{name}_mean'], cross@np.linalg.solve(D, self.y))
                assert_allclose(prediction[f'{name}_cov'], prior-cross@np.linalg.solve(D, cross.T))
        beta = conditional_prediction(self.y, self.x, xp, self.q, model='beta_gp')
        mu_q = self.q.copy()
        mu_q[1] += 2*mu_q[5]
        mu = conditional_prediction(self.y, self.x, xp, mu_q, model='mu_gp')
        assert_allclose(beta['f_mean'], mu['f_mean'])
        assert_allclose(beta['f_cov'], mu['f_cov'])

    def test_mixture_total_covariance_noise_and_quantiles(self):
        xp = np.array([0.1, 1.1])
        q2 = self.q+np.array([0.3, -0.6, 0.4, 0.2, 0.6, -0.5])
        for model in ['beta_gp', 'mu_gp']:
            p = predict_joint_mixture(self.y, self.x, xp,
                                      dict(model=model, log_samples=np.array([[self.q], [q2]])))
            components = [conditional_prediction(self.y, self.x, xp, q, model=model) for q in [self.q, q2]]
            mean = sum(c['f_mean'] for c in components)/2
            covariance = sum(c['f_cov']+np.outer(c['f_mean']-mean, c['f_mean']-mean) for c in components)/2
            assert_allclose(p['f_mean'], mean)
            assert_allclose(p['f_cov'], covariance)
            assert_allclose(p['f_cov'], p['f_within_cov']+p['f_between_cov'])
            assert_allclose(p['y_cov']-p['f_cov'], sum(c['noise_variance'] for c in components)/2*np.eye(2), atol=1e-13)
            for side, probability in enumerate([0.025, 0.975]):
                cdf = stats.norm.cdf((p['f_ci'][:, side]-p['component_f_means'])
                                     /np.sqrt(p['component_f_variances'])).mean(axis=0)
                assert_allclose(cdf, probability, atol=1e-10)
            if model == 'beta_gp':
                native_mean = sum(c['effect_mean']/c['W'] for c in components)/2
                assert_allclose(p['beta_mean'], native_mean)

    def test_metropolis_swaps_preserve_asymmetric_gaussian_target(self):
        mean = np.array([-0.7, 0.9])
        covariance = np.array([[1.2, 0.3], [0.3, 0.7]])
        precision = np.linalg.inv(covariance)
        def target(q):
            centered = q-mean
            return -0.5*centered@precision@centered, -precision@centered
        fit = sample_hmc(target, [[-2., 2.], [2., -2.], [1., 1.], [-1., -1.]],
                         warmup=300, draws=1500, seed=42, leapfrog_steps=(5, 15),
                         swap_pairs=[(0, 1)])
        values = fit['samples'].reshape(-1, 2)
        assert_allclose(values.mean(axis=0), mean, atol=0.08)
        assert_allclose(np.cov(values.T), covariance, atol=0.10)
        self.assertTrue(np.any(fit['swap_accepted']))
        self.assertTrue(np.any(~fit['swap_accepted']))


if __name__ == '__main__':
    unittest.main()
