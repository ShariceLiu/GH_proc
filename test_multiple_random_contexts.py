"""Check nested evaluation design and denser-context quadrature independently."""

import unittest

import numpy as np
from numpy.testing import assert_allclose, assert_array_equal
from scipy import integrate, special, stats
from threadpoolctl import threadpool_limits

from GH_multiple_hmc import MODELS
from hmc_multiple_random_contexts import random_locations, checked_weights


class RandomContextTests(unittest.TestCase):
    def test_nested_disjoint_reproducible_design(self):
        small, large, test = random_locations(121, 20)
        again = random_locations(121, 20)
        for a, b in zip((small, large, test), again):
            assert_array_equal(a, b)
        self.assertEqual(len(np.unique(large, axis=0)), 20)
        for a, b, c in zip(small, large, test):
            self.assertEqual(len(a), 5)
            self.assertEqual(len(b), 15)
            self.assertEqual(len(c), 106)
            self.assertTrue(set(a) < set(b))
            self.assertFalse(set(a) & set(c))
            self.assertFalse(set(b) & set(c))
            assert_array_equal(np.sort(np.r_[b, c]), np.arange(121))
        with self.assertRaises(ValueError):
            random_locations(15, 20)

    def test_fifteen_point_context_against_adaptive_integration(self):
        rng = np.random.default_rng(17)
        x = np.sort(rng.uniform(0, 10, 15))
        y = np.sin(x)+rng.normal(0, .2, len(x))
        theta = np.log([[.2, 1., .05, 2., .15], [.4, .8, .1, 1.5, .3]])
        d2 = (x[:, None]-x[None, :])**2
        with threadpool_limits(limits=1):
            for model in MODELS:
                _, logw, error = checked_weights(theta, x, y, model)
                self.assertLess(error, 1e-5)
                for index, (v, a, k, w, sw) in enumerate(np.exp(theta)):
                    K = v*np.exp(-w*d2/2)+k*np.eye(len(x))
                    S = a*np.exp(-sw*d2/2)
                    if model == 'gp':
                        expected = stats.multivariate_normal.logpdf(y, cov=K+S)
                    else:
                        def log_integrand(t):
                            W = np.exp(t)
                            prior = (stats.invgamma.logpdf(W, 2.5, scale=1.5) if model == 'tp_mu_gp'
                                     else stats.geninvgauss.logpdf(W, 1., 1.))
                            C = W*K+W**(2 if model == 'beta_gp' else 0)*S
                            return stats.multivariate_normal.logpdf(y, cov=C)+prior+t
                        shift = max(log_integrand(t) for t in np.linspace(-5, 5, 101))
                        mass, _ = integrate.quad(lambda t: np.exp(log_integrand(t)-shift),
                                                 -10, 14, epsabs=1e-10, epsrel=1e-9)
                        expected = np.log(mass)+shift
                    assert_allclose(special.logsumexp(logw[index])+np.log(len(theta)), expected, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
