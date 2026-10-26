"""Dense joint-Gaussian reference for shared-function likelihood/gradients."""
import unittest

import numpy as np
from scipy import linalg, stats
from threadpoolctl import threadpool_limits

from GH_shared_function_map import shared_calculation, shared_calculation_qr, make_target, gig_mean, MODELS
from GH_fixed_function import fixed_function_prediction
from shared_beta_plugin_experiment import matched_gp_prediction, student_crps


class SharedFunctionTests(unittest.TestCase):
    def test_qr_with_tiny_noise_against_high_precision_joint_gaussian(self):
        import mpmath as mp
        x = np.linspace(0, 1, 8)
        Y = np.array([2+.2*x, 3+.3*x+1e-5*np.sin(3*x), 4+.4*x])
        q = np.log([.3, 20., 1e-10, .4, .02, 1.1, 2., 3.])
        result = shared_calculation_qr(Y, x, q, 'ghp_beta', xp=x)
        with mp.workdps(60):
            def reference(logs):
                v, a, k, w, hw, *W = [mp.exp(z) for z in logs]
                c = [z/mp.mpf(gig_mean()) for z in W]
                n = len(x)
                def entry(i, j):
                    row_i, col_i = divmod(i, n)
                    row_j, col_j = divmod(j, n)
                    d2 = (mp.mpf(x[col_i])-mp.mpf(x[col_j]))**2
                    value = c[row_i]*c[row_j]*a*mp.exp(-hw*d2/2)
                    if row_i == row_j:
                        value += W[row_i]*(v*mp.exp(-w*d2/2)+(k if col_i == col_j else 0))
                    return value
                C = mp.matrix([[entry(i, j) for j in range(Y.size)] for i in range(Y.size)])
                L = mp.cholesky(C)
                y = mp.matrix(Y.ravel().tolist())
                return -(Y.size*mp.log(2*mp.pi)+2*sum(mp.log(L[i,i]) for i in range(Y.size))
                         +(y.T*mp.cholesky_solve(C,y))[0])/2
            logs = list(map(mp.mpf, q))
            value = reference(logs)
            self.assertAlmostEqual(result['log_likelihood'], float(value), places=6)
            numerical = []
            for j in range(len(q)):
                upper, lower = logs.copy(), logs.copy()
                upper[j] += mp.mpf('1e-6'); lower[j] -= mp.mpf('1e-6')
                numerical.append(float((reference(upper)-reference(lower))/mp.mpf('2e-6')))
            np.testing.assert_allclose(result['gradient'], numerical, rtol=2e-6, atol=2e-6)

    def test_dense_likelihood_conditional_mean_and_gradients(self):
        x = np.array([0., .6, 1.7, 2.8])
        Y = np.array([[.3, -.2, .8, 1.1], [-.1, .5, .9, .6], [.8, .3, 1.4, 1.2]])
        J, n = Y.shape
        for model in MODELS:
            with self.subTest(model=model):
                q = np.log([.7, 1.2, .09, 1.4, .3])
                W = np.ones(J)
                if model != 'gp':
                    W = np.array([.6, 1.4, 2.1])
                    q = np.r_[q, np.log(W)]
                calc = shared_calculation(Y, x, q, model, xp=x)
                qr = shared_calculation_qr(Y, x, q, model, xp=x)
                for key in ('log_likelihood', 'gradient', 'h_pred', 'h_cov'):
                    np.testing.assert_allclose(qr[key], calc[key], atol=1e-10, rtol=1e-10)
                xp = np.array([-1., 4., 6.])
                np.testing.assert_allclose(shared_calculation_qr(Y,x,q,model,xp=xp)['h_pred'],
                                           shared_calculation(Y,x,q,model,xp=xp)['h_pred'],atol=1e-10)
                c = W/gig_mean() if model == 'ghp_beta' else np.ones(J)
                d2 = (x[:, None]-x)**2
                A = .7*np.exp(-.7*d2)+.09*np.eye(n)
                S = 1.2*np.exp(-.15*d2)
                C = np.kron(np.diag(W), A)+np.kron(np.outer(c, c), S)
                reference = stats.multivariate_normal.logpdf(Y.ravel(), cov=C)
                self.assertAlmostEqual(calc['log_likelihood'], reference, places=10)
                cross = np.hstack([cj*S for cj in c])
                expected = cross@linalg.solve(C, Y.ravel())
                np.testing.assert_allclose(calc['h_pred'], expected, atol=1e-11)
                np.testing.assert_allclose(calc['h_cov'], S-cross@linalg.solve(C, cross.T), atol=1e-11)
                target, _ = make_target(Y, x, model)
                delta = 1e-5*np.eye(len(q))
                numeric = np.array([(target(q+d)[0]-target(q-d)[0])/2e-5 for d in delta])
                np.testing.assert_allclose(numeric, target(q)[1], atol=2e-7, rtol=2e-6)

    def test_empty_context_is_prior(self):
        xp = np.array([.2, .9, 2.3])
        beta = np.array([.5, .8, 1.])
        p = fixed_function_prediction([], [], xp, v0=.2, k=.05, w=2., beta_pred=beta)
        np.testing.assert_allclose(p['f_mean'], gig_mean()*beta, atol=1e-12)
        self.assertAlmostEqual(p['W_lambda'], 1.)
        self.assertAlmostEqual(p['W_chi'], 1.)
        self.assertAlmostEqual(p['W_psi'], 1.)
        for mixing in ('gp', 'ig'):
            p = fixed_function_prediction([], [], xp, v0=.2, k=.05, w=2., mixing=mixing, mu_pred=beta)
            np.testing.assert_allclose(p['f_mean'], beta, atol=1e-12)

    def test_oracle_gp_has_same_first_two_prior_moments(self):
        x = np.linspace(0, 3, 9)
        beta = .5+.3*np.sin(x)
        gp = matched_gp_prediction(x, [], np.array([], dtype=int), beta)
        gh = fixed_function_prediction([], [], x, v0=.2, k=.05, w=2., beta_pred=beta)
        for name in ('f', 'y'):
            np.testing.assert_allclose(gp[f'{name}_mean'], gh[f'{name}_mean'], atol=1e-12)
            np.testing.assert_allclose(gp[f'{name}_cov'], gh[f'{name}_cov'], atol=1e-12)

    def test_student_crps_against_cdf_integral(self):
        from scipy import integrate
        for df, mean, scale, truth in [(5., 1., .7, -2.), (10., -1., 2., .5), (20., 0., .3, 0.)]:
            dist = stats.t(df, loc=mean, scale=scale)
            expected = (integrate.quad(lambda u: dist.cdf(u)**2, -np.inf, truth)[0]
                        +integrate.quad(lambda u: dist.sf(u)**2, truth, np.inf)[0])
            self.assertAlmostEqual(student_crps(truth, mean, scale, df), expected, places=9)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        unittest.main()
