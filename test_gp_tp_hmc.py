"""Independent density, gradient, conditional and scoring checks for GP/TP."""

import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy import integrate, stats

from GP_TP_hmc import (make_baseline_target, two_se_kernel, conditional_prediction,
                       predict_baseline_mixture, mixture_intervals, mixture_logpdf)


class BaselineTests(unittest.TestCase):
    def setUp(self):
        self.x = np.array([0., .4, 1.3])
        self.y = np.array([.6, -.3, 1.1])
        self.xp = np.array([.2, 1.8])
        self.q = np.log([.7, 1.2, .13, 1.4, .25])
        self.center = np.array([-.4, .1, -2., .3, -1.1])
        self.sd = np.array([1., 1.2, 1.4, 1.6, 1.8])
        self.nu = 5.

    def target(self, model, nu=5.):
        return make_baseline_target(self.y, self.x, model=model, nu=nu,
                                     prior_center=self.center, prior_sd=self.sd)

    def covariance(self):
        return two_se_kernel(self.x, self.x, self.q)+np.exp(self.q[2])*np.eye(3)

    def test_normalized_likelihood_matches_scipy(self):
        prior = stats.norm.logpdf(self.q, self.center, self.sd).sum()
        C = self.covariance()
        assert_allclose(self.target('gp')(self.q)[0]-prior,
                        stats.multivariate_normal.logpdf(self.y, cov=C), atol=1e-12)
        assert_allclose(self.target('tp')(self.q)[0]-prior,
                        stats.multivariate_t.logpdf(self.y, shape=C*(self.nu-2)/self.nu, df=self.nu), atol=1e-12)

    def test_student_likelihood_matches_independent_scale_integration(self):
        C = self.covariance()
        def integrand(t):
            W = np.exp(t)
            return np.exp(stats.multivariate_normal.logpdf(self.y, cov=W*C)
                          +stats.invgamma.logpdf(W, a=self.nu/2, scale=(self.nu-2)/2)+t)
        integrated, error = integrate.quad(integrand, -12, 14, epsabs=1e-12, epsrel=1e-10)
        loglike = self.target('tp')(self.q)[0]-stats.norm.logpdf(self.q, self.center, self.sd).sum()
        assert_allclose(np.log(integrated), loglike, atol=1e-9)

    def test_analytic_gradients(self):
        for model in ['gp', 'tp']:
            target = self.target(model)
            perturbations = np.eye(5)*1e-5
            numerical = [(target(self.q+d)[0]-target(self.q-d)[0])/2e-5 for d in perturbations]
            assert_allclose(target(self.q)[1], numerical, rtol=1e-6, atol=1e-7)

    def test_gp_conditional_against_direct_block_formula(self):
        p = conditional_prediction(self.y, self.x, self.xp, self.q, model='gp')
        cross = two_se_kernel(self.xp, self.x, self.q)
        covariance = two_se_kernel(self.xp, self.xp, self.q)-cross@np.linalg.solve(self.covariance(), cross.T)
        assert_allclose(p['f_mean'], cross@np.linalg.solve(self.covariance(), self.y))
        assert_allclose(p['f_cov'], covariance, atol=1e-13)
        assert_allclose(p['y_cov']-p['f_cov'], np.exp(self.q[2])*np.eye(2), atol=1e-13)

    def test_tp_conditionals_match_joint_over_marginal_density(self):
        p = conditional_prediction(self.y, self.x, self.xp, self.q, model='tp', nu=self.nu)
        test_values = np.array([.1, .7])
        locations = np.r_[self.x, self.xp]
        K = two_se_kernel(locations, locations, self.q)
        noise = np.exp(self.q[2])
        log_marginal = stats.multivariate_t.logpdf(self.y, shape=self.covariance()*(self.nu-2)/self.nu, df=self.nu)
        for target in ['f', 'y']:
            joint = K+np.diag([noise]*3+([0., 0.] if target == 'f' else [noise]*2))
            expected = stats.multivariate_t.logpdf(np.r_[self.y, test_values],
                                                  shape=joint*(self.nu-2)/self.nu, df=self.nu)-log_marginal
            actual = stats.multivariate_t.logpdf(test_values, loc=p[f'{target}_mean'],
                                                 shape=p[f'{target}_cov']*(p['df']-2)/p['df'], df=p['df'])
            assert_allclose(actual, expected, atol=1e-10)
        expected_mean, expected_var = stats.invgamma.stats(a=p['W_shape'], scale=p['W_scale'], moments='mv')
        assert_allclose([p['W_mean'], p['W_var']], [expected_mean, expected_var])

    def test_tp_gaussian_limit(self):
        gp = conditional_prediction(self.y, self.x, self.xp, self.q, model='gp')
        tp = conditional_prediction(self.y, self.x, self.xp, self.q, model='tp', nu=1e8)
        assert_allclose(tp['f_mean'], gp['f_mean'], atol=1e-12)
        assert_allclose(tp['f_cov'], gp['f_cov'], rtol=1e-6)

    def test_mixture_includes_within_between_covariance_and_correct_t_quantiles(self):
        q2 = self.q+np.array([.7, -.3, .6, -.4, .8])
        for model in ['gp', 'tp']:
            fit = dict(model=model, nu=self.nu, log_samples=np.array([[self.q], [q2]]))
            p = predict_baseline_mixture(self.y, self.x, self.xp, fit)
            parts = [conditional_prediction(self.y, self.x, self.xp, q, model=model, nu=self.nu) for q in [self.q, q2]]
            mean = sum(part['f_mean'] for part in parts)/2
            covariance = sum(part['f_cov']+np.outer(part['f_mean']-mean, part['f_mean']-mean) for part in parts)/2
            assert_allclose(p['f_cov'], covariance)
            assert_allclose(p['f_cov'], p['f_within_cov']+p['f_between_cov'])
            assert_allclose(p['f_mean'], p['y_mean'])
            for side, probability in enumerate([.025, .975]):
                v = p['component_f_variances']
                if model == 'gp':
                    cdf = stats.norm.cdf(p['f_ci'][:, side], loc=p['component_means'], scale=np.sqrt(v))
                else:
                    df = p['component_df']
                    cdf = stats.t.cdf(p['f_ci'][:, side], df, loc=p['component_means'], scale=np.sqrt(v*(df-2)/df))
                assert_allclose(cdf.mean(axis=0), probability, atol=1e-10)
            if model == 'tp':
                w_mean = sum(part['W_mean'] for part in parts)/2
                w_var = sum(part['W_var']+(part['W_mean']-w_mean)**2 for part in parts)/2
                assert_allclose([p['W_mean'], p['W_var']], [w_mean, w_var])

    def test_mixture_logpdf_uses_density_average_and_covariance_parameterization(self):
        means = np.array([[-1., 0.], [2., 1.]])
        variances = np.array([[.3, 1.], [2., .7]])
        values = np.array([.2, -.3])
        weights = np.array([.3, .7])
        for df in [np.inf, 8.]:
            if np.isinf(df):
                pdf = stats.norm.pdf(values, loc=means, scale=np.sqrt(variances))
            else:
                pdf = stats.t.pdf(values, df, loc=means, scale=np.sqrt(variances*(df-2)/df))
            assert_allclose(mixture_logpdf(values, means, variances, df=df, weights=weights), np.log(weights@pdf))

    def test_invalid_df_is_rejected(self):
        with self.assertRaises(ValueError):
            self.target('tp', nu=2.)
        with self.assertRaises(ValueError):
            self.target('unknown')


if __name__ == '__main__':
    unittest.main()
