"""Check GP-prior mean estimation, fixed-mean densities and uncertainty."""
import unittest
import numpy as np
from numpy.testing import assert_allclose
from scipy import stats
from threadpoolctl import threadpool_limits
from GH_zero_mean_hmc import (PLUGIN_MU_MODELS, coordinates, make_target, conditional_prediction, predict_mixture)
from hmc_plugin_mu_30_experiment import posterior_mu_mean, SOURCES


class PluginMuTests(unittest.TestCase):
    def test_estimate_uses_unscaled_independent_mean_GP(self):
        x, xp, y = np.array([0., .4, 1.2]), np.array([.2, .8]), np.array([.6, -.2, .9])
        for source in SOURCES.values():
            values = np.array([[.7, .8, .12, 1.1, .3, .6], [.4, 1.2, .15, 2., .4, 2.3]])
            if source == 'gp':
                values = values[:, :5]
            expected = []
            for q in values:
                v, a, k, w, mw = q[:5]
                W = q[5] if len(q) == 6 else 1.
                d = (x[:, None]-x[None, :])**2
                cross = a*np.exp(-.5*mw*(xp[:, None]-x[None, :])**2)
                C = a*np.exp(-.5*mw*d)+W*(v*np.exp(-.5*w*d)+k*np.eye(3))
                expected.append(cross@np.linalg.solve(C, y))
            mean, chains = posterior_mu_mean(y, x, xp, np.log(values[:, None, :]), source_model=source)
            assert_allclose(chains, expected, atol=1e-12)
            assert_allclose(mean, np.mean(expected, axis=0), atol=1e-12)

    def test_all_families_density_gradient_conditioning_and_mixture(self):
        x, xp, y = np.array([0., .4, 1.2]), np.array([.2, .8]), np.array([.6, -.2, .9])
        mu, mp, center = np.array([.5, -.2, .3]), np.array([-.1, .7]), np.array([-.4, -2., .2])
        for model in PLUGIN_MU_MODELS:
            with self.subTest(model=model):
                q = np.log([.7, .12, 1.1] if model == 'gp_plugin_mu' else [.7, .12, 1.1, 1.7])
                v, k, w = np.exp(q[:3])
                W = 1. if len(q) == 3 else np.exp(q[3])
                all_x = np.r_[x, xp]
                K = W*v*np.exp(-.5*w*(all_x[:, None]-all_x[None, :])**2)
                C = K[:3, :3]+W*k*np.eye(3)
                target = make_target(y, x, model=model, mu=mu, prior_center=center)
                z = coordinates(q, model)
                expected = stats.multivariate_normal.logpdf(y, mu, C)+stats.norm.logpdf(q[:3], center, 2.).sum()
                if model == 'stp_plugin_mu':
                    expected += stats.invgamma.logpdf(W, 2.5, scale=1.5)+np.log(W)
                elif model == 'ghp_plugin_mu':
                    expected += stats.geninvgauss.logpdf(W, 1., 1.)+np.log(W)
                assert_allclose(target(z)[0], expected, atol=1e-12)
                delta = np.eye(len(q))*1e-5
                numeric = [(target(z+d)[0]-target(z-d)[0])/2e-5 for d in delta]
                assert_allclose(target(z)[1], numeric, atol=1e-7, rtol=1e-6)
                p = conditional_prediction(y, x, xp, q, model=model, mu_train=mu, mu_pred=mp)
                assert_allclose(p['f_mean'], mp+K[3:, :3]@np.linalg.solve(C, y-mu), atol=1e-12)
                assert_allclose(p['f_cov'], K[3:, 3:]-K[3:, :3]@np.linalg.solve(C, K[:3, 3:]), atol=1e-12)
                fit = dict(model=model, log_samples=np.array([[q], [q+.2]]))
                p = predict_mixture(y, x, xp, fit, mu_train=mu, mu_pred=mp)
                assert_allclose(p['mu_var'], 0.)
                assert_allclose(p['mu_cov'], 0.)
                assert_allclose(p['mu_mean'], mp)
                assert_allclose(p['effect_cov'], 0.)
                if len(q) == 3:
                    assert_allclose([p['W_mean'], p['W_var']], [1., 0.])
                else:
                    self.assertGreater(p['W_var'], 0.)
                for side, probability in enumerate([.025, .975]):
                    cdf = stats.norm.cdf((p['f_ci'][:, side]-p['component_means'])/np.sqrt(p['component_f_variances'])).mean(axis=0)
                    assert_allclose(cdf, probability, atol=1e-10)
                with self.assertRaises(ValueError):
                    make_target(y, x, model=model, prior_center=center)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        unittest.main()
