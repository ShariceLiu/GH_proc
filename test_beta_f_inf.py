"""Numerical checks against direct Gaussian conditioning and independent integrals."""

import unittest

import numpy as np
from numpy.testing import assert_allclose
from scipy import integrate, special, stats

from GH_proc_w_betaprior import Multivariate_GH, Multivariate_GH_1d, beta_f_inf


def se(x, y, variance, w):
    return variance * np.exp(-w * np.subtract.outer(x, y)**2 / 2)


class BetaPosteriorTests(unittest.TestCase):
    def test_scalar_against_direct_integration_in_W(self):
        # Integrating over W directly independently checks the log-W Jacobian.
        y, mu, v, sigma, k, lam, omega = 1.6, -0.2, 0.7, 1.2, 0.15, -0.5, 1.3
        result = beta_f_inf([y], [0], mu=mu, v_0=v, sigma_v0=sigma,
                            k=k, l=lam, omega=omega, quad_tol=1e-9)

        def integrand(W):
            T = v + k + W*sigma
            density = stats.geninvgauss.pdf(W, lam, omega) * stats.norm.pdf(y, mu, np.sqrt(W*T))
            mean = np.array([sigma*(y-mu)/T, mu + (W*sigma+v)*(y-mu)/T])
            cov = np.array([[sigma*(v+k)/T, W*sigma*k/T],
                            [W*sigma*k/T, W*(W*sigma+v)*k/T]])
            return density * np.r_[1, W, W**2, mean, (cov + np.outer(mean, mean)).ravel()]

        integral, _ = integrate.quad_vec(integrand, 0, np.inf, epsabs=1e-11, epsrel=1e-10)
        moments = integral / integral[0]
        assert_allclose(result['weights'] @ result['W'], moments[1], rtol=1e-7)
        assert_allclose(result['weights'] @ result['W']**2, moments[2], rtol=1e-7)
        assert_allclose(result['joint_mean'], moments[3:5], rtol=1e-7)
        assert_allclose(result['joint_cov'], moments[5:].reshape(2, 2) - np.outer(moments[3:5], moments[3:5]), rtol=1e-7)

    def test_zero_beta_prior_matches_conjugate_GIG_GP(self):
        xs = np.array([0., 0.6, 1.5])
        ys = np.array([0.1, 1.4, -0.2])
        mu, v, w, k, lam, omega = 0.3, 0.8, 1.2, 0.2, 0.5, 1.7
        result = beta_f_inf(ys, xs, mu=mu, v_0=v, w=w, k=k,
                            l=lam, omega=omega, sigma_v0=0)
        K = se(xs, xs, v, w)
        C = K + k*np.eye(xs.size)
        residual = ys - mu
        p = lam - xs.size/2
        a, b = omega, omega + residual @ np.linalg.solve(C, residual)
        ew = np.sqrt(b/a) * special.kv(p+1, np.sqrt(a*b)) / special.kv(p, np.sqrt(a*b))
        assert_allclose(result['weights'] @ result['W'], ew, rtol=1e-7)
        assert_allclose(result['beta_mean'], 0, atol=1e-12)
        assert_allclose(result['beta_ci'], 0, atol=1e-12)
        assert_allclose(result['f_mean'], mu + K @ np.linalg.solve(C, residual), atol=1e-10)
        assert_allclose(result['f_cov'], ew * (K - K @ np.linalg.solve(C, K)), atol=1e-9)

    def test_joint_components_at_new_inputs_against_block_conditioning(self):
        # Equal numbers of different train/test inputs catch spurious cross-kernel jitter.
        xs, xp, ys = np.array([0., 1.]), np.array([0.3, 1.4]), np.array([0.2, 1.1])
        v, w, sigma, sw, k, mu = 0.6, 2., 0.8, 0.3, 0.15, -0.1
        result = beta_f_inf(ys, xs, xp, v_0=v, w=w, sigma_v0=sigma, sigma_w=sw, k=k, mu=mu)
        for i in (0, len(result['W'])//2, -1):
            W = result['W'][i]
            K, S = se(xs, xs, v, w), se(xs, xs, sigma, sw)
            Kp, Sp = se(xp, xp, v, w), se(xp, xp, sigma, sw)
            Kx, Sx = se(xp, xs, v, w), se(xp, xs, sigma, sw)
            D = W*(K + k*np.eye(xs.size)) + W**2*S
            B = np.vstack([W*Sx, W*Kx + W**2*Sx])
            Q = np.block([[Sp, W*Sp], [W*Sp, W*Kp + W**2*Sp]])
            mean = np.r_[np.zeros(xp.size), np.full(xp.size, mu)] + B @ np.linalg.solve(D, ys-mu)
            cov = Q - B @ np.linalg.solve(D, B.T)
            assert_allclose(result['component_means'][i], mean, rtol=1e-8, atol=1e-9)
            assert_allclose(result['component_covariances'][i], cov, rtol=1e-8, atol=1e-9)
        self.assertGreaterEqual(np.linalg.eigvalsh(result['joint_cov']).min(), -1e-10)

    def test_refined_quadrature_and_credible_interval_probabilities(self):
        xs = np.linspace(0, 4, 7)
        args = dict(mu=0, k=0.15, sigma_w=0.3, v_0=0.4)
        _, ys, _, _ = Multivariate_GH_1d(xs, **args, beta_prior=True, return_latents=True, seed=3)
        result = beta_f_inf(ys, xs, **args)
        refined = beta_f_inf(ys, xs, **args, n_particles=128, quad_tol=1e-9, tail_log_drop=45)
        for key in ('joint_mean', 'joint_cov', 'beta_ci', 'f_ci'):
            assert_allclose(result[key], refined[key], rtol=2e-6, atol=1e-7)
        self.assertAlmostEqual(result['weights'].sum(), 1)
        self.assertTrue(np.all(result['weights'] >= 0))
        stds = np.sqrt(np.diagonal(result['component_covariances'], axis1=1, axis2=2))
        bands = np.vstack([result['beta_ci'], result['f_ci']])
        for column, probability in ((0, 0.025), (1, 0.975)):
            cdf = special.ndtr((bands[:, column] - result['component_means']) / stds)
            assert_allclose(result['weights'] @ cdf, probability, atol=1e-10)
        # Confirm that between-W uncertainty is included, not just conditional variance.
        within = np.einsum('p,pij->ij', result['weights'], result['component_covariances'])
        between = result['joint_cov'] - within
        self.assertGreater(np.trace(between), 1e-4)
        self.assertGreaterEqual(np.linalg.eigvalsh(between).min(), -1e-10)

    def test_input_validation_and_deterministic_beta_option(self):
        xs = np.array([0., 1.])
        _, _, beta, W = Multivariate_GH_1d(xs, a_b=2, b_b=-1, return_latents=True, seed=1)
        assert_allclose(beta, [2, 1])
        self.assertGreater(W, 0)
        for kwargs in ({'k': 0}, {'sigma_v0': -1}, {'credible_mass': 1},
                       {'n_particles': 4}, {'log_w_bounds': (2, 1)}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                beta_f_inf([1., 2.], xs, **kwargs)
        with self.assertRaises(ValueError):
            beta_f_inf([1.], xs)


class MultidimensionalBetaPosteriorTests(unittest.TestCase):
    def test_1d_column_inputs_and_original_generator_agree(self):
        xs = np.array([0., 0.7, 1.8])
        kwargs = dict(w=1.2, sigma_w=0.3, beta_prior=True, return_latents=True, seed=4)
        original = Multivariate_GH_1d(xs, **kwargs)
        general = Multivariate_GH(xs[:, None], **kwargs)
        for old, new in zip(original, general):
            assert_allclose(old, new, atol=1e-12)
        first = beta_f_inf(original[1], xs, w=1.2, sigma_w=0.3)
        second = beta_f_inf(original[1], xs[:, None], w=[1.2], sigma_w=[0.3])
        for key in ('W', 'weights', 'joint_mean', 'joint_cov', 'beta_ci', 'f_ci'):
            assert_allclose(first[key], second[key], atol=1e-12)
        self.assertEqual(first['x_pred'].shape, (3,))
        self.assertEqual(second['x_pred'].shape, (3, 1))

    def test_2d_components_and_weights_against_direct_gaussian_likelihood(self):
        xs = np.array([[0., 0.], [0.7, 1.4], [1.5, -0.3]])
        xp = np.array([[0.3, 0.8], [1.8, 0.5]])
        ys = np.array([0.2, 1.1, -0.5])
        w, sw = np.array([0.4, 2.]), np.array([1.3, 0.15])
        v, sigma, k, mu, lam, omega = 0.6, 0.8, 0.15, -0.1, -0.5, 1.3
        result = beta_f_inf(ys, xs, xp, v_0=v, w=w, sigma_v0=sigma, sigma_w=sw,
                            k=k, mu=mu, l=lam, omega=omega)

        def kernel(a, b, amplitude, weights):
            # Independent construction as a product of two 1D SE kernels.
            return amplitude * se(a[:, 0], b[:, 0], 1, weights[0]) * se(a[:, 1], b[:, 1], 1, weights[1])

        K, S = kernel(xs, xs, v, w), kernel(xs, xs, sigma, sw)
        Kp, Sp = kernel(xp, xp, v, w), kernel(xp, xp, sigma, sw)
        Kx, Sx = kernel(xp, xs, v, w), kernel(xp, xs, sigma, sw)
        direct_log_likelihood = []
        for i, W in enumerate(result['W']):
            D = W*(K + k*np.eye(len(xs))) + W**2*S
            direct_log_likelihood.append(stats.multivariate_normal.logpdf(ys, mean=np.full(len(xs), mu), cov=D))
            if i not in (0, len(result['W'])//2, len(result['W'])-1):
                continue
            B = np.vstack([W*Sx, W*Kx + W**2*Sx])
            Q = np.block([[Sp, W*Sp], [W*Sp, W*Kp + W**2*Sp]])
            mean = np.r_[np.zeros(len(xp)), np.full(len(xp), mu)] + B @ np.linalg.solve(D, ys-mu)
            cov = Q - B @ np.linalg.solve(D, B.T)
            assert_allclose(result['component_means'][i], mean, rtol=1e-8, atol=1e-9)
            assert_allclose(result['component_covariances'][i], cov, rtol=1e-8, atol=1e-9)
        _, integration_weights = special.roots_legendre(len(result['W']))
        log_weights = (np.array(direct_log_likelihood) +
                       stats.geninvgauss.logpdf(result['W'], lam, omega) +
                       result['log_W'] + np.log(integration_weights))
        expected = np.exp(log_weights - special.logsumexp(log_weights))
        assert_allclose(result['weights'], expected, rtol=1e-9, atol=1e-12)
        self.assertEqual(result['component_means'].shape[1], 2*len(xp))
        self.assertGreaterEqual(np.linalg.eigvalsh(result['joint_cov']).min(), -1e-10)

    def test_3d_axis_permutation_preserves_inference(self):
        xs = np.random.default_rng(3).normal(size=(6, 3))
        xp = np.random.default_rng(4).normal(size=(4, 3))
        ys = np.array([0.3, -1.2, 0.7, 0., 1.8, -0.6])
        w, sw = np.array([0.2, 1.4, 3.]), np.array([0.1, 0.5, 1.5])
        order = [2, 0, 1]
        first = beta_f_inf(ys, xs, xp, w=w, sigma_w=sw)
        second = beta_f_inf(ys, xs[:, order], xp[:, order], w=w[order], sigma_w=sw[order])
        for key in ('weights', 'joint_mean', 'joint_cov', 'beta_ci', 'f_ci'):
            assert_allclose(first[key], second[key], rtol=1e-8, atol=1e-10)

    def test_shared_kernel_weights_and_irrelevant_constant_coordinates(self):
        xs = np.array([0., 0.6, 1.8])
        ys = np.array([0.1, -0.3, 0.8])
        expanded = np.column_stack([xs, np.full_like(xs, 2), np.zeros_like(xs)])
        first = beta_f_inf(ys, xs, w=1.4, sigma_w=0.3)
        second = beta_f_inf(ys, expanded, w=1.4, sigma_w=0.3)
        for key in ('W', 'weights', 'joint_mean', 'joint_cov'):
            assert_allclose(first[key], second[key], atol=1e-10)

    def test_linear_beta_generation_in_3d(self):
        xs = np.random.default_rng(6).normal(size=(8, 3))
        slopes = np.array([0.3, -1.2, 0.1])
        f, y, beta, W = Multivariate_GH(xs, v_0=0, a_b=0.25, b_b=slopes, mu=0.4,
                                       beta_prior=False, return_latents=True, seed=2)
        assert_allclose(beta, 0.25 + xs @ slopes, atol=1e-12)
        assert_allclose(f, 0.4 + W*beta, atol=1e-12)
        self.assertEqual(y.shape, (8,))
        self.assertTrue(np.all(np.isfinite(y)))
        self.assertGreater(W, 0)

    def test_multidimensional_validation(self):
        xs, ys = np.ones((3, 2)), np.ones(3)
        for kwargs in ({'w': [1, 2, 3]}, {'sigma_w': [1]}, {'w': [1, 0]},
                       {'x_pred': np.ones((2, 3))}, {'x_pred': np.array([1., 2.])}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                beta_f_inf(ys, xs, **kwargs)
        with self.assertRaises(ValueError):
            beta_f_inf(ys, xs.T)  # feature-by-sample arrays need an explicit transpose
        with self.assertRaises(ValueError):
            Multivariate_GH(xs, beta_prior=False, b_b=[1, 2, 3])


if __name__ == '__main__':
    unittest.main()
