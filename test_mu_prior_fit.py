"""Checks of the GP-mean likelihood gradients and shared MAP fitting."""

import unittest
import numpy as np
from numpy.testing import assert_allclose

from GH_proc_mu_prior import mu_f_inf, mu_log_marginal_likelihood
from GH_proc_mu_prior_fit import fit_mu_kernel_parameters
from GH_proc_w_betaprior_fit import fit_kernel_parameters


class MuFittingTests(unittest.TestCase):
    def test_analytic_gradient_including_constant_mean(self):
        for x in (np.array([0., 0.4, 1.2]), np.array([[0., 0.], [0.4, -0.8], [1.2, 0.5]])):
            dimension = 1 if x.ndim == 1 else x.shape[1]
            y = np.array([0.3, -0.4, 0.9])
            z = np.r_[[-0.3, 0.2, -1.9], np.linspace(-0.6, 0.2, dimension),
                      np.linspace(-1.2, -0.4, dimension), 0.15]

            def evaluate(vector, gradient=False):
                p = np.exp(vector[:-1])
                return mu_log_marginal_likelihood(
                    y, x, l=1, omega=1, v_0=p[0], mu_v0=p[1], k=p[2],
                    w=p[3:3+dimension], mu_w=p[3+dimension:], mu0=vector[-1],
                    return_gradient=gradient, quad_tol=1e-9)

            _, g = evaluate(z, True)
            analytic = np.r_[g['log_v_0'], g['log_mu_v0'], g['log_k'],
                             g['log_w'], g['log_mu_w'], g['mu0']]
            numeric = [(evaluate(z+delta)-evaluate(z-delta))/2e-5 for delta in 1e-5*np.eye(len(z))]
            assert_allclose(analytic, numeric, atol=1e-7, rtol=1e-5)

    def test_MAP_learns_mean_kernel_and_returns_prediction_parameters(self):
        x = np.linspace(-2, 2, 16)
        y = np.sin(1.7*x)+0.6*np.cos(0.5*x)+0.06*np.random.default_rng(8).normal(size=len(x))
        fit = fit_mu_kernel_parameters(y, x, l=1, omega=1, seed=3, n_restarts=2)
        self.assertTrue(fit['converged'], fit['message'])
        self.assertLess(fit['objective'], fit['initial_objective']-0.1)
        params = fit['params']
        self.assertEqual((params['l'], params['omega'], params['mu0']), (1., 1., 0.))
        self.assertNotAlmostEqual(params['mu_v0'], 0.5*np.var(y), places=4)
        self.assertGreater(np.max(np.abs(params['mu_w']-0.25)), 1e-4)
        vector = np.log(np.r_[params['v_0'], params['mu_v0'], params['k'], params['w'], params['mu_w']])
        penalty = 0.5*np.sum(((vector-fit['prior']['vector_center'])/fit['prior']['vector_sd'])**2)
        value = mu_log_marginal_likelihood(y, x, **params, **fit['quad_options'])
        assert_allclose(fit['objective'], -value+penalty, atol=1e-8)
        p = mu_f_inf(y, x, [0.3, 1.8], **params)
        self.assertTrue(np.all(np.isfinite(p['y_mean'])))
        self.assertTrue(np.all(p['y_ci'][:, 1] > p['y_ci'][:, 0]))

    def test_models_use_the_same_log_parameter_priors(self):
        x = np.linspace(-2, 2, 10)
        y = np.cos(1.3*x)+0.1*x
        common = dict(l=1, omega=1, n_restarts=1, seed=4)
        beta = fit_kernel_parameters(y, x, **common)
        mu = fit_mu_kernel_parameters(y, x, **common)
        assert_allclose(beta['prior']['vector_center'], mu['prior']['vector_center'])
        assert_allclose(beta['prior']['vector_sd'], mu['prior']['vector_sd'])
        self.assertFalse(mu['prior']['fit_mu0'])

    def test_fixed_GIG_and_GP_prior_mean_cannot_be_overridden_by_initial(self):
        for initial in ({'omega': 2.}, {'mu0': 3.}, {'sigma_v0': 1.}):
            with self.assertRaises(ValueError):
                fit_mu_kernel_parameters([0., 1.], [0., 1.], initial=initial)


if __name__ == '__main__':
    unittest.main()
