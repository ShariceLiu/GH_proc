"""Check learned-mixing joint posterior and plug-in mixing updates."""
import unittest
import numpy as np
from scipy import linalg, stats
from threadpoolctl import threadpool_limits

from GH_tabular_learned_plugin_map import LearnedPluginTarget, MODELS
from GH_tabular_hmc import coordinates
from GH_tabular_plugin_map import PluginPredictor


class LearnedPluginTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(712)
        self.x, self.y = rng.normal(size=(8, 2)), rng.normal(size=8)

    def test_density_and_gradient_against_independent_gaussian(self):
        rng = np.random.default_rng(913)
        for model in MODELS:
            target = LearnedPluginTarget(self.y, self.x, model)
            for _ in range(3):
                z = target.initial()+rng.normal(0, .6, target.dimension)
                q = coordinates(z[:target.qdim+1], model, inverse=True)
                h = z[target.qdim+1:]
                v, a, k = np.exp(q[:3])
                W = np.exp(q[-1])
                dx = self.x[:, None, :]-self.x[None, :, :]
                K = v*np.exp(-.5*((dx**2)*np.exp(q[3:5])).sum(-1))
                H = a*np.exp(-.5*((dx**2)*np.exp(q[5:7])).sum(-1))
                cov = W*(K+k*np.eye(len(self.y)))+(W**2 if model=='beta_gp' else 1.)*H
                reference = stats.multivariate_normal.logpdf(self.y, cov=cov)
                reference += stats.norm.logpdf(q[:-1], target.center, 2.).sum()
                if model == 'tp_mu_gp':
                    nu = 2+np.exp(h[0])
                    reference += stats.invgamma.logpdf(W, nu/2, scale=(nu-2)/2)+q[-1]
                    reference += stats.norm.logpdf(h[0], np.log(3.), 1.)
                else:
                    reference += stats.geninvgauss.logpdf(W, h[0], np.exp(h[1]))+q[-1]
                    reference += stats.norm.logpdf(h[0], 1., 2.)+stats.norm.logpdf(h[1], 0., 1.)
                value, gradient = target(z)
                np.testing.assert_allclose(value, reference, atol=1e-10, rtol=0)
                eps = np.eye(len(z))*2e-5
                numerical = [(target(z+e)[0]-target(z-e)[0])/4e-5 for e in eps]
                np.testing.assert_allclose(gradient, numerical, atol=3e-6, rtol=2e-5)

    def test_fitted_mixing_parameters_reach_prediction(self):
        for model in MODELS:
            target = LearnedPluginTarget(self.y, self.x, model)
            q = coordinates(target.initial()[:target.qdim+1], model, inverse=True)
            params = dict(nu=8.2) if model=='tp_mu_gp' else dict(lam=-.8, omega=.7)
            predictor = PluginPredictor(self.y, self.x, q, model, **params)
            mix = predictor.mixing_parameters
            mu = np.zeros(len(self.y)) if model=='beta_gp' else predictor.function_train
            beta = predictor.function_train if model=='beta_gp' else np.zeros(len(self.y))
            residual = self.y-mu
            quad = residual@linalg.solve(predictor.A, residual, assume_a='pos')
            if model=='tp_mu_gp':
                self.assertAlmostEqual(mix['W_shape'], (8.2+len(self.y))/2)
                self.assertAlmostEqual(mix['W_scale'], (8.2-2+quad)/2)
            else:
                self.assertAlmostEqual(mix['W_lambda'], -.8-len(self.y)/2)
                self.assertAlmostEqual(mix['W_chi'], .7+quad)
                self.assertAlmostEqual(mix['W_psi'], .7+beta@linalg.solve(predictor.A,beta,assume_a='pos'))


if __name__=='__main__':
    with threadpool_limits(limits=1):
        unittest.main()
