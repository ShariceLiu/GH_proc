"""Independent density, gradient, prediction and 1D compatibility checks."""
import unittest
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
from numpy.testing import assert_allclose
from scipy import stats
from threadpoolctl import threadpool_limits

from GH_tabular_hmc import (MODELS, coordinates, make_target, conditional_marginals,
                            predict_mixture)
from GH_proc_joint_hmc import make_joint_target, conditional_prediction
from GP_TP_hmc import make_baseline_target
from hmc_airfoil_experiment import fit_model, write_json


class TabularChecks(unittest.TestCase):
    def setUp(self):
        self.x = np.array([[0., .2], [.4, -.8], [1.2, .6], [.7, .9]])
        self.xp = np.array([[.3, .1], [.8, -.4]])
        self.y = np.array([.6, -.2, .9, -.3])
        self.q = np.log([.7, 1.3, .12, 1.1, .5, .3, 1.2, 1.7])
        self.center = np.linspace(-1., .5, 7)
        self.sd = np.linspace(.7, 1.8, 7)

    def direct(self, model):
        v, a, k, w0, w1, s0, s1, W = np.exp(self.q)
        if model == 'gp':
            W = 1.
        x = np.vstack([self.x, self.xp])
        delta = (x[:, None, :]-x[None, :, :])**2
        T = W*v*np.exp(-.5*np.sum(delta*[w0, w1], axis=2))
        T += W**(2 if model == 'beta_gp' else 0)*a*np.exp(-.5*np.sum(delta*[s0, s1], axis=2))
        C = T[:4, :4]+W*k*np.eye(4)
        return T, C, W*k

    def test_density_gradients_and_prediction_all_models(self):
        for model in MODELS:
            with self.subTest(model=model):
                q = self.q[:-1] if model == 'gp' else self.q
                z = coordinates(q, model)
                assert_allclose(coordinates(z, model, inverse=True), q, atol=1e-14)
                target = make_target(self.y, self.x, model=model, prior_center=self.center,
                                     prior_sd=self.sd, l=1.2, omega=.8, nu=5.)
                T, C, noise = self.direct(model)
                expected = stats.multivariate_normal.logpdf(self.y, cov=C)
                expected += stats.norm.logpdf(q[:7], self.center, self.sd).sum()
                if model != 'gp':
                    W = np.exp(q[-1])
                    if model == 'tp_mu_gp':
                        expected += stats.invgamma.logpdf(W, a=2.5, scale=1.5)+q[-1]
                    else:
                        expected += stats.geninvgauss.logpdf(W, 1.2, .8)+q[-1]
                assert_allclose(target(z)[0], expected, atol=1e-11)
                perturbations = np.eye(len(q))*1e-5
                numeric = [(target(z+p)[0]-target(z-p)[0])/2e-5 for p in perturbations]
                assert_allclose(target(z)[1], numeric, rtol=2e-6, atol=1e-7)
                pred = conditional_marginals(self.y, self.x, self.xp, q, model=model)
                cross = T[4:, :4]
                assert_allclose(pred['mean'], cross@np.linalg.solve(C, self.y))
                covariance = T[4:, 4:]-cross@np.linalg.solve(C, cross.T)
                assert_allclose(pred['f_variance'], np.diag(covariance), atol=1e-12)
                assert_allclose(pred['y_variance'], np.diag(covariance)+noise, atol=1e-12)

    def test_one_dimensional_matches_existing_models(self):
        x = self.x[:, 0]
        q = self.q[[0, 1, 2, 3, 5, 7]]
        center, sd = self.center[:5], self.sd[:5]
        for model in MODELS:
            qi = q[:-1] if model == 'gp' else q
            target = make_target(self.y, x, model=model, prior_center=center, prior_sd=sd)
            previous = make_baseline_target if model == 'gp' else make_joint_target
            old = previous(self.y, x, model=model, prior_center=center, prior_sd=sd)
            z = coordinates(qi, model)
            assert_allclose(target(z)[0], old(z)[0], atol=1e-11)
            assert_allclose(target(z)[1], old(z)[1], atol=1e-11)
            if model != 'gp':
                p = conditional_marginals(self.y, x, self.xp[:, 0], qi, model=model)
                old_p = conditional_prediction(self.y, x, self.xp[:, 0], qi, model=model)
                assert_allclose(p['mean'], old_p['f_mean'])
                assert_allclose(p['f_variance'], np.diag(old_p['f_cov']), atol=1e-12)

    def test_original_units_mixture_variance_and_quantiles(self):
        q2 = self.q+np.linspace(-.3, .5, len(self.q))
        fit = dict(model='beta_gp', log_samples=np.array([[self.q], [q2]]))
        p = predict_mixture(self.y, self.x, self.xp, fit, output_mean=120., output_scale=7., verbose=False,
                            interval_batch_size=1)
        components = [conditional_marginals(self.y, self.x, self.xp, q, model='beta_gp')
                      for q in [self.q, q2]]
        means = np.array([120.+7.*c['mean'] for c in components])
        assert_allclose(p['y_mean'], means.mean(axis=0))
        for name in ['f', 'y']:
            variances = np.array([49.*c[f'{name}_variance'] for c in components])
            assert_allclose(p[f'{name}_var'], variances.mean(axis=0)+means.var(axis=0))
            for side, probability in enumerate([.025, .975]):
                cdf = stats.norm.cdf((p[f'{name}_ci'][:, side]-means)/np.sqrt(variances)).mean(axis=0)
                assert_allclose(cdf, probability, atol=1e-10)

    def test_resume_uses_saved_samples_without_resampling(self):
        # Synthetic metadata tests recovery mechanics, not convergence quality.
        rng = np.random.default_rng(12)
        x, y = rng.normal(size=(7, 5)), rng.normal(size=7)
        data = dict(x_scaled=x, y_scaled=y, y=10.+2.*y,
                    train_ids=np.arange(4), test_ids=np.arange(4, 7),
                    y_mean=10., y_scale=2., x_scale=np.ones(5))
        center = np.log(np.r_[.5, .5, .1, np.full(5, .2), np.full(5, .05)])
        samples = center+rng.normal(0., .05, size=(2, 8, 13))
        diagnostic = dict(rhat=np.ones(13), ess_bulk=np.full(13, 500.), ess_tail=np.full(13, 450.))
        attempt = dict(attempt=1, settings=dict(chains=2, draws=8), diagnostics_ok=True,
                       max_rhat=1., min_bulk_ess=500., min_tail_ess=450., divergences=0,
                       acceptance=.95, swap_acceptance=.8, diagnostics=diagnostic,
                       effective_diagnostics=diagnostic)
        from GH_tabular_hmc import parameter_names
        with tempfile.TemporaryDirectory(prefix='airfoil_resume_test_') as temp:
            folder = Path(temp)/'gp'
            folder.mkdir()
            write_json(folder/'attempts.json', [attempt])
            np.savez(folder/'attempt_1_samples.npz', log_samples=samples,
                     parameter_names=parameter_names(5, 'gp'), prior_center=center,
                     prior_sd=np.full(13, 2.))
            with patch('hmc_airfoil_experiment.sample_parameters') as sampler:
                result = fit_model(temp, data, 'gp')
                sampler.assert_not_called()
            self.assertTrue(result['resumed_predictions'])
            self.assertTrue((folder/'predictions.npz').exists())
            self.assertTrue((folder/'parameters_original_units.csv').exists())


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        unittest.main()
