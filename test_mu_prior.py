"""Independent checks for a GP mean independent of the shared GIG variable."""

import unittest
import numpy as np
from numpy.testing import assert_allclose
from scipy import integrate, linalg, stats

from GH_proc_w_betaprior import _se, beta_f_inf
from GH_proc_w_betaprior_fit import score_predictions
from GH_proc_mu_prior import beta_effect_posterior, mu_f_inf, mu_log_marginal_likelihood
from compare_beta_mu_gp import gig_second_moment


class MuPriorTests(unittest.TestCase):
    def test_prior_covariance_matching_uses_the_GIG_second_moment(self):
        for lam, omega in [(-0.5, 1.), (1., 1.), (0.4, 2.)]:
            direct, _ = integrate.quad(lambda W: W**2*stats.geninvgauss.pdf(W, lam, omega),
                                       0, np.inf, epsabs=1e-10, epsrel=1e-10)
            assert_allclose(gig_second_moment(lam, omega), direct, rtol=1e-9)

    def test_conditionals_against_direct_block_gaussian_and_W_density(self):
        x = np.array([[0., 0.], [0.4, -0.7], [1.1, 0.5]])
        xp = np.array([[0.2, 0.6], [1.4, -0.2]])
        y = np.array([0.1, 1., -0.4])
        params = dict(l=-0.5, omega=1.3, w=[0.7, 1.8], v_0=0.6, k=0.15,
                      mu0=0.3, mu_v0=1.1, mu_w=[0.2, 0.9])
        posterior = mu_f_inf(y, x, xp, **params)
        K = _se(x, x, params['v_0'], np.array(params['w']))
        M = _se(x, x, params['mu_v0'], np.array(params['mu_w']))
        Kpp = _se(xp, xp, params['v_0'], np.array(params['w']))
        Mpp = _se(xp, xp, params['mu_v0'], np.array(params['mu_w']))
        Kpt = _se(xp, x, params['v_0'], np.array(params['w']))
        Mpt = _se(xp, x, params['mu_v0'], np.array(params['mu_w']))
        lo, hi = posterior['quadrature']['log_w_bounds']
        _, qw = np.polynomial.legendre.leggauss(len(posterior['W']))
        log_mass = []
        for i, W in enumerate(posterior['W']):
            D = M + W*(K + params['k']*np.eye(len(x)))
            cross = np.vstack((Mpt, Mpt+W*Kpt))
            prior = np.block([[Mpp, Mpp], [Mpp, Mpp+W*Kpp]])
            direct_mean = params['mu0'] + cross @ linalg.solve(D, y-params['mu0'], assume_a='pos')
            direct_cov = prior - cross @ linalg.solve(D, cross.T, assume_a='pos')
            assert_allclose(posterior['component_means'][i], direct_mean, atol=1e-9)
            assert_allclose(posterior['component_covariances'][i], direct_cov, atol=1e-9)
            log_mass.append(stats.multivariate_normal.logpdf(y, mean=np.full(len(x), params['mu0']), cov=D)
                            + stats.geninvgauss.logpdf(W, params['l'], params['omega'])
                            + np.log(W) + np.log(qw[i]) + np.log((hi-lo)/2))
        from scipy.special import logsumexp
        assert_allclose(posterior['weights'], np.exp(log_mass-logsumexp(log_mass)), atol=1e-10)
        assert_allclose(posterior['log_marginal_likelihood'], logsumexp(log_mass), atol=1e-9)
        # Pointwise quantiles invert the mixture CDF, including observation noise.
        for name, section in [('mu', slice(0, 2)), ('f', slice(2, 4)), ('y', slice(2, 4))]:
            means = posterior['component_means'][:, section]
            variance = (posterior['component_y_variances'] if name == 'y' else
                        np.diagonal(posterior['component_covariances'], axis1=1, axis2=2)[:, section])
            for column, probability in enumerate((0.025, 0.975)):
                cdf = posterior['weights'] @ stats.norm.cdf(
                    posterior[f'{name}_ci'][:, column], loc=means, scale=np.sqrt(variance))
                assert_allclose(cdf, probability, atol=1e-9)

    def test_scalar_evidence_against_independent_W_integral(self):
        params = dict(l=0.7, omega=1.2, mu0=-0.2, mu_v0=0.8, v_0=0.4, k=0.15)
        actual = mu_log_marginal_likelihood([1.1], [0.], **params, quad_tol=1e-9)
        value, _ = integrate.quad(
            lambda W: stats.geninvgauss.pdf(W, params['l'], params['omega'])*
            stats.norm.pdf(1.1, params['mu0'], np.sqrt(params['mu_v0']+W*(params['v_0']+params['k']))),
            0, np.inf, epsabs=1e-12, epsrel=1e-10)
        assert_allclose(actual, np.log(value), atol=1e-8)

    def test_zero_mean_GP_recovers_existing_zero_beta_model(self):
        y, x, xp = [0.1, -0.6], [0., 0.7], [0.2, 1.2]
        common = dict(l=-0.5, omega=1, w=0.7, v_0=0.9, k=0.2)
        a = beta_f_inf(y, x, xp, **common, mu=0.4, sigma_v0=0)
        b = mu_f_inf(y, x, xp, **common, mu0=0.4, mu_v0=0)
        for key in ['log_marginal_likelihood', 'f_mean', 'f_cov', 'f_ci', 'y_ci']:
            assert_allclose(a[key], b[key], atol=1e-8)

    def test_predictive_density_matches_joint_evidence_ratio(self):
        y, x, xp, yp = np.array([0.3, -0.6]), np.array([0., 0.7]), np.array([1.1]), np.array([0.8])
        params = dict(l=1, omega=1, v_0=0.6, k=0.2, mu_v0=0.7, mu_w=0.3)
        posterior = mu_f_inf(y, x, xp, **params)
        log_predictive = (mu_log_marginal_likelihood(np.r_[y, yp], np.r_[x, xp], **params)
                          - mu_log_marginal_likelihood(y, x, **params))
        assert_allclose(score_predictions(yp, posterior)['mean_marginal_nlpd'], -log_predictive, atol=1e-7)

    def test_effect_moments_include_W_beta_dependence(self):
        # Unequal W and conditional beta means make E[W beta] != E[W]E[beta].
        posterior = dict(W=np.array([1., 3.]), weights=np.array([0.25, 0.75]),
                         component_means=np.array([[2., 0.], [-1., 0.]]),
                         component_covariances=np.array([np.diag([0.4, 1.]), np.diag([0.2, 1.])]),
                         credible_mass=0.95)
        p = beta_effect_posterior(posterior)
        assert_allclose(p['mean'], [-1.75])
        assert_allclose(p['var'], [0.25*0.4+0.75*9*0.2 + 0.25*(2+1.75)**2+0.75*(-3+1.75)**2])

    def test_fixed_W_reparameterization_has_identical_joint_covariance(self):
        # With mu|W ~ GP(0,W^2 Sigma_beta), the [W beta,f,y] distribution
        # is identical to [mu,f,y] before and after Gaussian conditioning.
        x = np.array([0., 0.4, 1.])
        W, k = 1.8, 0.1
        S, K = _se(x, x, 0.7, np.array([0.3])), _se(x, x, 0.4, np.array([1.2]))
        prior_beta_f = np.block([[S, W*S], [W*S, W**2*S+W*K]])
        transform = np.diag(np.r_[np.full(3, W), np.ones(3)])
        M = W**2*S
        prior_mu_f = np.block([[M, M], [M, M+W*K]])
        assert_allclose(transform @ prior_beta_f @ transform, prior_mu_f)
        cross_beta_y = np.vstack((W*S, W**2*S+W*K))
        cross_mu_y = np.vstack((M, M+W*K))
        D = M+W*(K+k*np.eye(3))
        assert_allclose(transform @ cross_beta_y, cross_mu_y)
        assert_allclose(transform @ (prior_beta_f-cross_beta_y @ np.linalg.solve(D, cross_beta_y.T)) @ transform,
                        prior_mu_f-cross_mu_y @ np.linalg.solve(D, cross_mu_y.T), atol=1e-12)


if __name__ == '__main__':
    unittest.main()
