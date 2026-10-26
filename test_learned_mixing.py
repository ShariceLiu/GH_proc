"""Independent density, derivative, marginalization and conditioning checks."""
import unittest
import numpy as np
from scipy import integrate, stats
from threadpoolctl import threadpool_limits

from GH_learned_mixing import LearnedMixingTarget, mixing_log_density, mixing_prior
from GH_proc_joint_hmc import make_joint_target, physical_coordinates, conditional_prediction


class LearnedMixingTests(unittest.TestCase):
    def setUp(self):
        self.x = np.linspace(0, 3, 8)
        self.y = np.array([.2, .4, -.1, -.5, -.4, .1, .6, .3])

    def test_mixing_densities_and_normalization(self):
        for model, h in [('beta_gp', [1.2, -.4]), ('beta_gp', [-2., .7]), ('tp_mu_gp', [np.log(.7)])]:
            t = np.linspace(-3, 3, 11)
            density = mixing_log_density(t, h, model)
            if model == 'beta_gp':
                expected = stats.geninvgauss.logpdf(np.exp(t), h[0], np.exp(h[1]))+t
            else:
                nu = 2+np.exp(h[0])
                expected = stats.invgamma.logpdf(np.exp(t), nu/2, scale=(nu-2)/2)+t
            np.testing.assert_allclose(density, expected, atol=1e-11)
            integral, _ = integrate.quad(lambda s: np.exp(mixing_log_density(s, h, model)), -40, 40, epsabs=1e-10)
            self.assertAlmostEqual(integral, 1., places=9)

    def test_joint_target_and_gradients(self):
        rng = np.random.default_rng(22)
        for model in ['beta_gp', 'tp_mu_gp']:
            target = LearnedMixingTarget(self.y, self.x, model)
            for _ in range(4):
                z = target.initial()+rng.normal(0, .6, target.dimension)
                args = dict(l=z[6], omega=np.exp(z[7])) if model == 'beta_gp' else dict(nu=2+np.exp(z[6]))
                old = make_joint_target(self.y, self.x, model=model, prior_center=target.center, **args)
                self.assertAlmostEqual(target.joint(z)[0], old(z[:6])[0]+mixing_prior(z[6:], model)[0], places=9)
                eps = np.eye(len(z))*1e-5
                numerical = np.array([(target.joint(z+e)[0]-target.joint(z-e)[0])/2e-5 for e in eps])
                np.testing.assert_allclose(target.joint(z)[1], numerical, atol=2e-6, rtol=2e-5)

    def test_marginal_integral_and_gradient(self):
        for model in ['beta_gp', 'tp_mu_gp']:
            target = LearnedMixingTarget(self.y, self.x, model)
            u = np.r_[target.center, target.initial_h()]
            value, grad = target.marginal(u, tol=1e-10, tail=48.)
            from GH_proc_joint_hmc import effective_coordinates
            def density(t):
                z = np.r_[effective_coordinates(np.r_[u[:5], t], model), u[5:]]
                return np.exp(target.joint(z)[0]-value)
            integral, _ = integrate.quad(density, -20, 20, epsabs=1e-10, epsrel=1e-10)
            self.assertAlmostEqual(integral, 1., places=8)
            eps = np.eye(len(u))*2e-5
            numerical = np.array([(target.marginal(u+e)[0]-target.marginal(u-e)[0])/4e-5 for e in eps])
            np.testing.assert_allclose(grad, numerical, atol=3e-6, rtol=2e-5)

    def test_conditional_predictions(self):
        for model in ['beta_gp', 'tp_mu_gp']:
            target = LearnedMixingTarget(self.y, self.x, model)
            z = target.initial()
            xp = np.linspace(-.2, 3.2, 15)
            new = target.predict_components(z, xp)
            old = conditional_prediction(self.y, self.x, xp, physical_coordinates(z[:6], model), model=model)
            np.testing.assert_allclose(new['component_means'][0], old['f_mean'], atol=1e-12)
            np.testing.assert_allclose(new['component_f_variances'][0], np.diag(old['f_cov']), atol=1e-12)
            np.testing.assert_allclose(new['component_y_variances'][0], np.diag(old['f_cov'])+old['noise_variance'], atol=1e-12)
            factor = np.exp(z[5]) if model == 'beta_gp' else 1.
            np.testing.assert_allclose(new['component_native_means'][0], old['effect_mean']/factor, atol=1e-12)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        unittest.main()
