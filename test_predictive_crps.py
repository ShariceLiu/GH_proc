"""Compare CDF integration to independent analytic CRPS special cases."""
import unittest
import numpy as np
from numpy.testing import assert_allclose
from scipy import stats
from threadpoolctl import threadpool_limits

from predictive_crps import gaussian_mixture_crps, gaussian_mixture_crps_closed_form


class CRPSTests(unittest.TestCase):
    def test_single_normal(self):
        truth = np.array([-8., -.5, 0., 2., 12.])
        mean, sd = .3, 1.2
        z = (truth-mean)/sd
        expected = sd*(z*(2*stats.norm.cdf(z)-1)+2*stats.norm.pdf(z)-1/np.sqrt(np.pi))
        actual, _ = gaussian_mixture_crps(truth, np.full((1, 5), mean), np.full((1, 5), sd**2))
        assert_allclose(actual, expected, atol=1e-10)

    def test_documented_weighted_mixture_example(self):
        # scoringrules.crps_mixnorm documentation example.
        args = (0., [.1, -.3, 1.], np.square([.4, 2.1, .7]), [.1, .2, .7])
        numerical, _ = gaussian_mixture_crps(*args)
        assert_allclose(numerical, .46806866729387275, atol=1e-10)
        assert_allclose(numerical, gaussian_mixture_crps_closed_form(*args), atol=1e-10)

    def test_multimodal_mixture_and_scale_translation(self):
        rng = np.random.default_rng(123)
        means = rng.normal(0, 3, (20, 7))
        variances = np.exp(rng.normal(-1., 2., (20, 7)))
        weights = rng.uniform(size=20)
        weights[0] = 0.
        truth = rng.normal(size=7)
        actual, d = gaussian_mixture_crps(truth, means, variances, weights, batch_size=3)
        exact = gaussian_mixture_crps_closed_form(truth, means, variances, weights)
        assert_allclose(actual, exact, atol=1e-8)
        scaled, _ = gaussian_mixture_crps(3*truth+7, 3*means+7, 9*variances, weights)
        assert_allclose(scaled, 3*actual, atol=1e-8)
        self.assertTrue(d['all_components_used'])

    def test_mixture_is_not_average_component_CRPS(self):
        score, _ = gaussian_mixture_crps(0., [-2., 2.], [1., 1.])
        component = np.mean([gaussian_mixture_crps_closed_form(0., [m], [1.])[0] for m in [-2., 2.]])
        self.assertGreater(component-score[0], .1)

    def test_invalid_inputs(self):
        for weights in [[-1., 2.], [0., 0.], [1.], [np.nan, 1.]]:
            with self.assertRaises(ValueError):
                gaussian_mixture_crps(0., [0., 1.], [1., 1.], weights)
        with self.assertRaises(ValueError):
            gaussian_mixture_crps(0., [0.], [0.])

    def test_pointwise_tolerance_for_mixed_scales(self):
        score, diagnostic = gaussian_mixture_crps([0., 0.], [[0., 0.]], [[1e8, 1.]])
        expected = np.array([1e4, 1.])*(np.sqrt(2.)-1.)/np.sqrt(np.pi)
        assert_allclose(score, expected, atol=1e-8, rtol=1e-12)
        self.assertTrue(np.all(diagnostic['integration_error_estimate'] <= 1e-8+1e-8*score))

    def test_truth_far_outside_predictive_support(self):
        truth = np.array([-1e4, 1e4])
        score, _ = gaussian_mixture_crps(truth, [[0., 0.]], [[1., 1.]])
        assert_allclose(score, np.abs(truth)-1/np.sqrt(np.pi), atol=1e-9)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        unittest.main()
