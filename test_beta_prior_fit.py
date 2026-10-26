"""Checks of normalized evidence, its gradient, fitting, and predictive scoring."""

import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy import integrate, stats

from GH_proc_w_betaprior import beta_f_inf, log_marginal_likelihood
from GH_proc_w_betaprior_fit import compare_fixed_omega, fit_kernel_parameters, score_predictions
from fit_beta_prior_experiment import standardize_split


class MarginalLikelihoodTests(unittest.TestCase):
    def test_evidence_against_direct_W_integral_across_GIG_parameters(self):
        # Comparisons across omega/l would fail if their normalization were omitted.
        for lam, omega in ((-0.5, 0.5), (1., 2.), (0., 5.)):
            with self.subTest(lam=lam, omega=omega):
                parameters = dict(l=lam, omega=omega, mu=-0.2, v_0=0.7, sigma_v0=1.2, k=0.15)
                value = log_marginal_likelihood([1.6], [0.], **parameters, quad_tol=1e-9)
                integral, _ = integrate.quad(
                    lambda W: stats.geninvgauss.pdf(W, lam, omega) *
                    stats.norm.pdf(1.6, -0.2, np.sqrt(W*(0.7+0.15)+W**2*1.2)),
                    0, np.inf, epsabs=1e-12, epsrel=1e-10)
                assert_allclose(value, np.log(integral), rtol=1e-8, atol=1e-9)

    def test_analytic_gradient_against_finite_differences(self):
        xs = np.array([[0., 0.], [0.5, 1.], [1.5, -0.3]])
        ys = np.array([0.2, 1.1, -0.5])
        theta = np.array([-0.4, -0.2, -2., -0.7, 0.6, -1., 0.2, 0.1])

        def parameters(t):
            return dict(v_0=np.exp(t[0]), sigma_v0=np.exp(t[1]), k=np.exp(t[2]),
                        w=np.exp(t[3:5]), sigma_w=np.exp(t[5:7]), mu=t[7], quad_tol=1e-9)

        _, g = log_marginal_likelihood(ys, xs, **parameters(theta), return_gradient=True)
        analytic = np.r_[g['log_v_0'], g['log_sigma_v0'], g['log_k'],
                         g['log_w'], g['log_sigma_w'], g['mu']]
        finite_difference = []
        for i in range(len(theta)):
            plus, minus = theta.copy(), theta.copy()
            plus[i] += 1e-5
            minus[i] -= 1e-5
            finite_difference.append((log_marginal_likelihood(ys, xs, **parameters(plus)) -
                                      log_marginal_likelihood(ys, xs, **parameters(minus)))/2e-5)
        assert_allclose(analytic, finite_difference, rtol=1e-5, atol=1e-7)

    def test_noisy_predictive_density_matches_evidence_ratio(self):
        xs, ys = np.array([0., 0.7]), np.array([0.2, -0.3])
        xp, yp = np.array([1.1]), np.array([0.6])
        parameters = dict(l=-0.5, omega=2, mu=0, v_0=0.6, sigma_v0=0.8, k=0.2)
        posterior = beta_f_inf(ys, xs, xp, **parameters)
        metrics = score_predictions(yp, posterior)
        log_predictive = (log_marginal_likelihood(np.r_[ys, yp], np.r_[xs, xp], **parameters) -
                          log_marginal_likelihood(ys, xs, **parameters))
        assert_allclose(metrics['mean_marginal_nlpd'], -log_predictive, atol=1e-7)
        ew = posterior['weights'] @ posterior['W']
        assert_allclose(posterior['y_var'] - posterior['f_var'], ew*parameters['k'], atol=1e-12)
        assert_allclose(posterior['log_marginal_likelihood'],
                        log_marginal_likelihood(ys, xs, **parameters), atol=1e-12)


class FittingTests(unittest.TestCase):
    def test_MAP_fit_improves_objective_and_keeps_GIG_fixed(self):
        xs = np.linspace(-2, 2, 12)
        ys = np.sin(1.5*xs) + 0.08*np.random.default_rng(1).normal(size=12)
        fit = fit_kernel_parameters(ys, xs, l=-0.5, omega=2, n_restarts=2, maxiter=150,
                                    fit_mu=True, seed=4)
        self.assertTrue(fit['converged'], fit['message'])
        self.assertLess(fit['objective'], fit['initial_objective'] - 0.01)
        self.assertEqual(fit['params']['l'], -0.5)
        self.assertEqual(fit['params']['omega'], 2)
        self.assertTrue(np.all(np.r_[fit['params']['w'], fit['params']['sigma_w'],
                                    fit['params']['v_0'], fit['params']['sigma_v0'], fit['params']['k']] > 0))

    def test_validation_labels_do_not_affect_fit_or_beta(self):
        xs = np.linspace(-2, 2, 10)
        ys = np.sin(xs)
        xv = np.array([-0.8, 0.2, 1.8])
        yv = np.sin(xv)
        options = dict(omegas=(1, 2), fit_options=dict(n_restarts=1, maxiter=150), verbose=False)
        first = compare_fixed_omega(ys, xs, yv, xv, **options)
        second = compare_fixed_omega(ys, xs, yv+100, xv, **options)
        self.assertEqual(first['baseline_omega'], 1)
        for r1, r2 in zip(first['records'], second['records']):
            self.assertTrue(r1['fit']['converged'])
            assert_allclose(r1['posterior']['beta_mean'], r2['posterior']['beta_mean'], atol=1e-12)
            self.assertGreater(r2['metrics']['rmse'], r1['metrics']['rmse']+90)
        self.assertEqual(first['records'][0]['beta_mean_shift_rmse'], 0)

    def test_scaler_uses_training_only(self):
        xs = np.array([[0., 2.], [2., 2.], [4., 2.], [100., -50.]])
        ys = np.array([1., 3., 5., 1000.])
        xt, yt, _, _, scaler = standardize_split(xs, ys, [0, 1, 2], [3])
        assert_allclose(xt.mean(axis=0), 0, atol=1e-12)
        assert_allclose(yt.mean(), 0, atol=1e-12)
        assert_allclose(scaler['x_mean'], [2., 2.])
        self.assertEqual(scaler['y_mean'], 3.)
        self.assertEqual(scaler['x_scale'][1], 1.)

    def test_cannot_override_fixed_GIG_through_initial_values(self):
        with self.assertRaises(ValueError):
            fit_kernel_parameters([0., 1.], [0., 1.], initial={'omega': 3})
        with self.assertRaises(ValueError):
            compare_fixed_omega([0., 1.], [0., 1.], [0.5], [0.5], omegas=[1, 1])


if __name__ == '__main__':
    unittest.main()
