"""Independent checks of HMC mechanics, gradients and posterior averaging."""

import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy import special

from GH_proc_w_betaprior import beta_f_inf
from GH_proc_w_betaprior_fit import _pack, _unpack
from GH_proc_w_betaprior_hmc import (
    _kernel_target, _leapfrog, chain_diagnostics, predict_hyperparameter_mixture, sample_hmc,
)


class HMCTests(unittest.TestCase):
    def setUp(self):
        self.covariance = np.array([[1., 0.65], [0.65, 2.]])
        self.precision = np.linalg.inv(self.covariance)

    def gaussian(self, q):
        return -0.5*q @ self.precision @ q, -self.precision @ q

    def test_leapfrog_reversibility_and_energy_accuracy(self):
        q, p, metric = np.array([1.2, -0.8]), np.array([0.7, 0.4]), np.array([0.8, 1.7])
        lp, gradient = self.gaussian(q)
        q1, p1, lp1, g1 = _leapfrog(self.gaussian, q, p, gradient, 0.1, 20, metric)
        q2, p2, _, _ = _leapfrog(self.gaussian, q1, -p1, g1, 0.1, 20, metric)
        assert_allclose(q2, q, atol=1e-12)
        assert_allclose(p2, -p, atol=1e-12)
        _, p3, lp3, _ = _leapfrog(self.gaussian, q, p, gradient, 0.05, 40, metric)
        h0 = -lp + 0.5*np.sum(metric*p**2)
        err1 = abs(-lp1+0.5*np.sum(metric*p1**2)-h0)
        err2 = abs(-lp3+0.5*np.sum(metric*p3**2)-h0)
        self.assertLess(err2, err1/3)

    def test_gaussian_sampling_recovers_mean_and_covariance(self):
        fit = sample_hmc(self.gaussian, [[-3., 2.], [3., -2.], [2., 2.], [-2., -2.]],
                         draws=1200, warmup=300, seed=19, leapfrog_steps=(5, 15))
        flat = fit['samples'].reshape(-1, 2)
        assert_allclose(flat.mean(axis=0), 0, atol=0.12)
        assert_allclose(np.cov(flat.T), self.covariance, atol=0.16)
        self.assertLess(fit['diagnostics']['rhat'].max(), 1.02)
        self.assertGreater(fit['diagnostics']['ess_bulk'].min(), 400)
        self.assertEqual(fit['samples'].shape, (4, 1200, 2))
        self.assertTrue(np.any(~fit['acceptance']))
        self.assertFalse(np.any(fit['divergent']))

    def test_diagnostics_detect_location_and_scale_disagreement(self):
        rng = np.random.default_rng(34)
        iid = rng.normal(size=(4, 1000, 2))
        good = chain_diagnostics(iid)
        self.assertLess(good['rhat'].max(), 1.01)
        self.assertGreater(good['ess_bulk'].min(), 2500)
        iid[0, :, 0] += 3
        iid[0, :, 1] *= 5
        bad = chain_diagnostics(iid)
        self.assertGreater(bad['rhat'].min(), 1.1)

    def test_threaded_chains_reproduce_serial_chains(self):
        options = dict(draws=100, warmup=60, seed=981, leapfrog_steps=(2, 5))
        starts = [[-2., 1.], [2., -1.]]
        serial = sample_hmc(self.gaussian, starts, **options)
        parallel = sample_hmc(self.gaussian, starts, chain_workers=2, **options)
        assert_allclose(serial['samples'], parallel['samples'], atol=0, rtol=0)
        assert_allclose(serial['acceptance_probability'], parallel['acceptance_probability'], atol=0, rtol=0)

    def test_inverse_paired_mode_moves_recover_unequal_mixture_weights(self):
        centers = np.array([[-8., 0.], [0., 8.]])
        def target(q):
            delta = q-centers
            values = np.log([.7, .3])-.5*np.sum(delta**2, axis=1)
            norm = special.logsumexp(values)
            return norm, -np.exp(values-norm)@delta
        fit = sample_hmc(target, [centers[0], centers[1], centers[0]+.1, centers[1]-.1],
                         warmup=200, draws=1200, seed=118, leapfrog_steps=(3, 8),
                         mode_transforms=[([1, 0], [0., 16.])], chain_workers=2)
        right = fit['samples'].sum(axis=-1) > 0
        assert_allclose(right.mean(), .3, atol=.04)
        self.assertGreater(int(fit['mode_jump_accepted'].sum()), 200)
        self.assertLess(fit['diagnostics']['rhat'].max(), 1.03)

    def test_hyperparameter_target_gradient_including_prior(self):
        x, y = np.array([[0., 0.], [0.3, -1.], [1.2, 0.8]]), np.array([1., -0.3, 0.4])
        center, sd = np.zeros(7), np.linspace(0.8, 2., 7)
        target = _kernel_target(y, x, 2, -0.5, 1., 0., center, sd, (-10, 10), dict(quad_tol=1e-9))
        q = np.array([-0.3, 0.4, -2., -0.4, 0.7, -1.3, -0.9])
        analytic = target(q)[1]
        perturbations = 1e-5 * np.eye(len(q))
        numeric = np.array([(target(q+delta)[0]-target(q-delta)[0])/2e-5
                            for delta in perturbations])
        assert_allclose(analytic, numeric, rtol=1e-5, atol=1e-7)
        self.assertEqual(target(q+30)[0], -np.inf)

    def test_prediction_averages_hyperparameter_and_W_uncertainty(self):
        y, x, xp = np.array([0.5, -0.4]), np.array([0., 1.]), np.array([0.4, 1.8])
        params = dict(v_0=0.6, sigma_v0=0.9, k=0.1, w=1., sigma_w=0.2, mu=0.)
        q1 = _pack(params, 1, False)
        q2 = q1 + np.array([0.7, -0.5, 0.4, 0.5, -0.6])
        fit = dict(log_samples=np.array([[q1, q1], [q2, q2]]), dimension=1,
                   fixed_params=dict(l=-0.5, omega=1., mu=0.), map_fit=dict(quad_options={}))
        combined = predict_hyperparameter_mixture(y, x, xp, fit, max_draws=4)
        individuals = [beta_f_inf(y, x, xp, **_unpack(q, 1, -0.5, 1., 0., False)) for q in (q1, q2)]
        for name in ('beta', 'f', 'y'):
            mean = sum(p[f'{name}_mean'] for p in individuals)/2
            var = sum(p[f'{name}_var'] + (p[f'{name}_mean']-mean)**2 for p in individuals)/2
            assert_allclose(combined[f'{name}_mean'], mean, atol=1e-12)
            assert_allclose(combined[f'{name}_var'], var, atol=1e-12)
            interval = combined[f'{name}_ci']
            self.assertTrue(np.all(interval[:, 1] > interval[:, 0]))
        self.assertAlmostEqual(combined['weights'].sum(), 1.)


if __name__ == '__main__':
    unittest.main()
