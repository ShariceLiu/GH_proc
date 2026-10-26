"""Independent checks for ARD marginalization, gradients and spectral prediction."""
import unittest
import numpy as np
from scipy import integrate, linalg, stats
from threadpoolctl import threadpool_limits

from GH_learned_mixing import LearnedMixingTarget, mixing_prior
from GH_learned_mixing_ard import ARDLearnedMixingMAP
from GH_tabular_hmc import conditional_marginals, make_target


class ARDLearnedMixingTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(928)
        self.x, self.y, self.xp = rng.normal(size=(9, 3)), rng.normal(size=9), rng.normal(size=(4, 3))

    def test_one_dimensional_agreement(self):
        for model in ('beta_gp', 'tp_mu_gp'):
            old = LearnedMixingTarget(self.y, self.x[:, 0], model)
            new = ARDLearnedMixingMAP(self.y, self.x[:, :1], model)
            u = np.r_[new.center, new.initial_h()]+.1
            np.testing.assert_allclose(new.marginal(u)[0], old.marginal(u)[0], atol=1e-10)
            np.testing.assert_allclose(new.marginal(u)[1], old.marginal(u)[1], atol=1e-9)

    def test_ard_gradients_and_independent_integral(self):
        for model in ('beta_gp', 'tp_mu_gp', 'mu_gp'):
            target = ARDLearnedMixingMAP(self.y, self.x, model)
            u = np.r_[target.center, target.initial_h()]+np.linspace(-.4, .5, target.qdim+target.hdim)
            value, grad = target.marginal(u)
            eps = np.eye(len(u))*2e-5
            numerical = [(target.marginal(u+d)[0]-target.marginal(u-d)[0])/4e-5 for d in eps]
            np.testing.assert_allclose(grad, numerical, atol=3e-6, rtol=2e-5)
            q, h = u[:target.qdim], u[target.qdim:]
            params = np.exp(q)
            dx = self.x[:, None, :]-self.x[None, :, :]
            K = params[0]*np.exp(-.5*((dx**2)*params[3:6]).sum(2))
            H = params[1]*np.exp(-.5*((dx**2)*params[6:9]).sum(2))
            family = 'beta_gp' if model in ('beta_gp', 'mu_gp') else model
            priors = target.kernel_prior(q)[0]+mixing_prior(h, family)[0]
            def density(t):
                W = np.exp(t)
                covariance = W*(K+params[2]*np.eye(len(self.y)))+W**target.power*H
                lp = stats.multivariate_normal.logpdf(self.y, cov=covariance)
                if model in ('beta_gp', 'mu_gp'):
                    lp += stats.geninvgauss.logpdf(W, h[0], np.exp(h[1]))+t
                else:
                    nu = 2+np.exp(h[0])
                    lp += stats.invgamma.logpdf(W, nu/2, scale=(nu-2)/2)+t
                return np.exp(lp+priors-value)
            integral, _ = integrate.quad(density, -15, 15, epsabs=1e-10, epsrel=1e-10)
            self.assertAlmostEqual(integral, 1., places=8)

    def test_gp_matches_direct_cholesky_target(self):
        target = ARDLearnedMixingMAP(self.y, self.x, 'gp')
        direct = make_target(self.y, self.x, model='gp', prior_center=target.center)
        rng = np.random.default_rng(292)
        for _ in range(3):
            q = target.center+rng.normal(0, .6, target.qdim)
            value, grad = target.marginal(q)
            reference, reference_grad = direct(q)
            np.testing.assert_allclose(value, reference, atol=1e-10)
            np.testing.assert_allclose(grad, reference_grad, atol=1e-10)
            eps = np.eye(len(q))*2e-5
            numerical = [(target.marginal(q+d)[0]-target.marginal(q-d)[0])/4e-5 for d in eps]
            np.testing.assert_allclose(grad, numerical, atol=3e-6, rtol=2e-5)
        self.assertEqual(len(target.initial_h()), 0)
        self.assertEqual(len(target.parameter_names), target.qdim)

    def test_spectral_prediction_against_dense_conditioning(self):
        for model in ('gp', 'beta_gp', 'tp_mu_gp', 'mu_gp'):
            target = ARDLearnedMixingMAP(self.y, self.x, model)
            u = np.r_[target.center, target.initial_h()]
            result = target.predict_components(u, self.xp)
            indices = np.unique(np.searchsorted(np.cumsum(result['weights']), [.01, .1, .5, .9, .99]))
            for i in indices:
                t = result['log_W'][i]
                pars = np.r_[u[:target.qdim], t] if target.has_mixing else u
                reference = conditional_marginals(self.y, self.x, self.xp, pars, model=model)
                np.testing.assert_allclose(result['component_means'][i], reference['mean'], atol=1e-10)
                np.testing.assert_allclose(result['component_f_variances'][i], reference['f_variance'], atol=1e-10)
                np.testing.assert_allclose(result['component_y_variances'][i], reference['y_variance'], atol=1e-10)
                K, H = target.kernels(u[:target.qdim], target.dist, target.eye.shape)
                from GH_tabular_hmc import squared_distances
                _, Hp = target.kernels(u[:target.qdim], squared_distances(self.xp, self.x).reshape(3, -1), (4, 9))
                W = np.exp(t)
                factor = W if model == 'beta_gp' else 1.
                covariance = W*(K+np.exp(u[2])*target.eye)+W**target.power*H
                expected_mean = factor*Hp@linalg.solve(covariance, self.y, assume_a='pos')
                expected_var = np.exp(u[1])-factor**2*np.sum(Hp*linalg.solve(covariance, Hp.T, assume_a='pos').T, axis=1)
                np.testing.assert_allclose(result['component_native_means'][i], expected_mean, atol=1e-10)
                np.testing.assert_allclose(result['component_native_variances'][i], expected_var, atol=1e-10)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        unittest.main()
