"""Independent density/gradient and regression checks for the GP+TP model."""

import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy import integrate, stats

from GH_proc_joint_hmc import (make_joint_target, effective_coordinates,
                               conditional_prediction, predict_joint_mixture)


class TestTPMuHMC(unittest.TestCase):
    def setUp(self):
        self.x = np.array([0., .4, 1.2])
        self.y = np.array([.6, -.2, .9])
        self.q = np.log([.7, 1.3, .12, 1.1, .3, 1.7])
        self.center = np.array([-.4, .1, -2., .2, -1.2])
        self.nu = 5.

    def target(self, nu=5.):
        return make_joint_target(self.y, self.x, model='tp_mu_gp',
                                 prior_center=self.center, prior_sd=2., nu=nu)

    def test_normalized_density_and_jacobian(self):
        v, a, k, w, sw, W = np.exp(self.q)
        d2 = (self.x[:, None]-self.x[None, :])**2
        C = a*np.exp(-sw*d2/2)+W*(v*np.exp(-w*d2/2)+k*np.eye(3))
        expected = (stats.multivariate_normal.logpdf(self.y, cov=C)
                    +stats.norm.logpdf(self.q[:5], self.center, 2.).sum()
                    +stats.invgamma.logpdf(W, a=2.5, scale=1.5)+np.log(W))
        assert_allclose(self.target()(effective_coordinates(self.q, 'tp_mu_gp'))[0], expected, atol=1e-12)

    def test_gradient(self):
        target = self.target()
        for shift in [0., -1., 1.]:
            z = effective_coordinates(self.q, 'tp_mu_gp')+shift
            numeric = [(target(z+d)[0]-target(z-d)[0])/2e-5 for d in np.eye(6)*1e-5]
            assert_allclose(target(z)[1], numeric, rtol=2e-6, atol=1e-7)

    def test_conditional_blocks_and_mixture(self):
        xp = np.array([.2, .8])
        q2 = self.q+np.array([.3, -.5, .2, .4, -.6, -.4])
        p = predict_joint_mixture(self.y, self.x, xp,
                                  dict(model='tp_mu_gp', log_samples=np.array([[self.q], [q2]])))
        conditional = []
        for q in [self.q, q2]:
            v, a, k, w, sw, W = np.exp(q)
            all_x = np.r_[self.x, xp]
            d2 = (all_x[:, None]-all_x[None, :])**2
            H = a*np.exp(-sw*d2/2)
            T = H+W*v*np.exp(-w*d2/2)
            C = T[:3, :3]+W*k*np.eye(3)
            c = conditional_prediction(self.y, self.x, xp, q, model='tp_mu_gp')
            for name, prior in [('f', T), ('effect', H)]:
                cross = prior[3:, :3]
                assert_allclose(c[name+'_mean'], cross@np.linalg.solve(C, self.y))
                assert_allclose(c[name+'_cov'], prior[3:, 3:]-cross@np.linalg.solve(C, cross.T))
            conditional.append(c)
        m = sum(c['f_mean'] for c in conditional)/2
        C = sum(c['f_cov']+np.outer(c['f_mean']-m, c['f_mean']-m) for c in conditional)/2
        assert_allclose(p['f_cov'], C)
        assert_allclose(p['mu_mean'], p['effect_mean'])
        assert_allclose(p['y_cov']-p['f_cov'], np.eye(2)*np.exp([self.q[2]+self.q[5], q2[2]+q2[5]]).mean(), atol=1e-12)
        for name in ['f', 'mu', 'y']:
            for side, prob in enumerate([.025, .975]):
                assert_allclose(stats.norm.cdf((p[name+'_ci'][:, side]-p['component_'+name+'_means'])
                                               /np.sqrt(p['component_'+name+'_variances'])).mean(axis=0), prob, atol=1e-10)

    def test_zero_mean_kernel_recovers_student_t_likelihood(self):
        # Integrate the NEW target over physical log W with K2 effectively zero.
        # The result must match a single-SE Student-t likelihood independently.
        q = self.q.copy()
        q[1] = -40.
        prior = stats.norm.logpdf(q[:5], self.center, 2.).sum()
        target = self.target()
        def density(t):
            point = np.r_[q[:5], t]
            return np.exp(target(effective_coordinates(point, 'tp_mu_gp'))[0]-prior)
        value, error = integrate.quad(density, -15, 25, epsabs=1e-11, epsrel=1e-9)
        v, _, k, w, _, _ = np.exp(q)
        C = v*np.exp(-w*(self.x[:, None]-self.x[None, :])**2/2)+k*np.eye(3)
        expected = stats.multivariate_t.pdf(self.y, shape=C*(self.nu-2)/self.nu, df=self.nu)
        assert_allclose(value, expected, rtol=1e-8, atol=1e-11)

    def test_invalid_nu(self):
        for nu in [0., 2., np.nan, np.inf]:
            with self.assertRaises(ValueError):
                self.target(nu)


if __name__ == '__main__':
    unittest.main()
