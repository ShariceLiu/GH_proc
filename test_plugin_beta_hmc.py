"""Tests for training-only beta estimation and fixed-beta prediction."""
import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy import stats
from threadpoolctl import threadpool_limits

from GH_zero_mean_hmc import PLUGIN_BETA_MODEL, coordinates, make_target, conditional_prediction, predict_mixture
from hmc_plugin_beta_30_experiment import posterior_beta_mean


class PluginBetaTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([0., .4, 1.2])
        self.y = np.array([.6, -.2, .9])
        self.xp = np.array([.2, .8])
        self.source = np.log(np.array([[[.7, .8, .12, 1.1, .3, .6]],
                                      [[.4, 1.2, .15, 2., .4, 2.3]]]))

    def test_beta_mean_from_joint_gaussian_cross_covariance(self):
        all_x = np.r_[self.x, self.xp]
        distances = (all_x[:, None]-all_x[None, :])**2
        conditional, effect = [], []
        for v, sv, k, w, sw, W in np.exp(self.source.reshape(-1, 6)):
            K = v*np.exp(-.5*w*distances)
            Sigma = sv*np.exp(-.5*sw*distances)
            C = W*K[:3, :3]+W**2*Sigma[:3, :3]+W*k*np.eye(3)
            conditional.append(W*Sigma[3:, :3]@np.linalg.solve(C, self.y))
            effect.append(W*conditional[-1])
        actual, chains = posterior_beta_mean(self.y, self.x, self.xp, self.source)
        assert_allclose(chains, conditional, atol=1e-12)
        assert_allclose(actual, np.mean(conditional, axis=0), atol=1e-12)
        # E[beta] must not be computed as E[W beta]/E[W].
        wrong_ratio = np.mean(effect, axis=0)/np.exp(self.source[..., -1]).mean()
        self.assertGreater(np.max(np.abs(actual-wrong_ratio)), 1e-3)

    def test_estimated_beta_fixed_density_gradient_and_prediction(self):
        beta, _ = posterior_beta_mean(self.y, self.x, np.r_[self.x, self.xp], self.source)
        bt, bp = beta[:3], beta[3:]
        q = np.log([.7, .12, 1.1, 1.7])
        center = np.array([-.4, -2., .2])
        v, k, w, W = np.exp(q)
        all_x = np.r_[self.x, self.xp]
        K = W*v*np.exp(-.5*w*(all_x[:, None]-all_x[None, :])**2)
        C = K[:3, :3]+W*k*np.eye(3)
        target = make_target(self.y, self.x, model=PLUGIN_BETA_MODEL, beta=bt, prior_center=center)
        z = coordinates(q, PLUGIN_BETA_MODEL)
        expected = (stats.multivariate_normal.logpdf(self.y, W*bt, C)
                    +stats.norm.logpdf(q[:3], center, 2.).sum()
                    +stats.geninvgauss.logpdf(W, 1., 1.)+np.log(W))
        assert_allclose(target(z)[0], expected, atol=1e-12)
        delta = np.eye(4)*1e-5
        numeric = [(target(z+d)[0]-target(z-d)[0])/2e-5 for d in delta]
        assert_allclose(target(z)[1], numeric, atol=1e-7, rtol=1e-6)
        p = conditional_prediction(self.y, self.x, self.xp, q, model=PLUGIN_BETA_MODEL,
                                    beta_train=bt, beta_pred=bp)
        assert_allclose(p['f_mean'], W*bp+K[3:, :3]@np.linalg.solve(C, self.y-W*bt), atol=1e-12)
        assert_allclose(p['f_cov'], K[3:, 3:]-K[3:, :3]@np.linalg.solve(C, K[:3, 3:]), atol=1e-12)
        fit = dict(model=PLUGIN_BETA_MODEL, log_samples=np.array([[q], [q+np.array([.2, -.1, .3, .4])]]))
        p = predict_mixture(self.y, self.x, self.xp, fit, beta_train=bt, beta_pred=bp)
        assert_allclose(p['beta_var'], 0.)
        assert_allclose(p['beta_cov'], 0.)
        assert_allclose(p['beta_fixed'], bp)
        self.assertGreater(p['W_var'], 0.)
        assert_allclose(p['effect_cov'], p['W_var']*np.outer(bp, bp))


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        unittest.main()
