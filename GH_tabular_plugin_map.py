"""ARD MAP fits and fixed-function or zero-function GP/STP/GHP predictions.

Gaussian functions are marginalized for log-coordinate MAP, then replaced by
their conditional means. Prediction recomputes and integrates W conditional
on that fixed function. Pointwise predictions are computed in small batches;
no test-by-test covariance or HMC component arrays need to be saved.
"""
import time
import numpy as np
from scipy import linalg, optimize
from scipy.spatial.distance import cdist

from GH_tabular_hmc import (ALL_MODELS, ZERO_MODELS, make_target, default_center, coordinates,
                            has_mixing, parameter_names)
from GH_fixed_function import marginal_distribution, StableGHMarginals, gig_moments


def se_kernel(x, xp, variance, inverse_lengths):
    x, xp = np.asarray(x), np.asarray(xp)
    scale = np.sqrt(np.asarray(inverse_lengths))
    distances = cdist(x*scale, xp*scale, metric='sqeuclidean')
    return variance * np.exp(-.5 * distances)


def fit_map(y, x, model, *, starts=24, seed=505, progress=None, completed=(),
            backend='dense', initial_points=()):
    """Select the best stationary log-posterior mode using training data only."""
    if model not in ALL_MODELS or starts < 2:
        raise ValueError('Need a supported model and >=2 starts')
    single = model in ZERO_MODELS
    d = x.shape[1]
    center = default_center(y, d, model)
    target = make_target(y, x, model=model, prior_center=center, backend=backend)
    initial_points = [np.asarray(z, dtype=float) for z in initial_points]
    base = center.copy()
    if has_mixing(model):
        logw = np.log(.6) if model in ('tp_mu_gp', 'stp_zero') else np.arcsinh(1.)
        base = coordinates(np.r_[base, logw], model)
    rng = np.random.default_rng(seed)
    attempts, solutions = [], []
    previous = {item['record']['start']: item for item in completed}
    for index in range(starts):
        guess = base.copy()
        if index >= (1 if single else 2):
            guess += rng.normal(0, 1.5, len(base))
        if index % 2 and not single:
            guess[[0, 1]] = guess[[1, 0]]
            first, second = guess[3:3+d].copy(), guess[3+d:3+2*d].copy()
            guess[3:3+d], guess[3+d:3+2*d] = second, first
        if index < len(initial_points):
            guess = initial_points[index].copy()
            if guess.shape != base.shape or not np.isfinite(target(guess)[0]):
                raise ValueError('Invalid supplied effective-log starting point')
        if index in previous:
            item = previous[index]
            z = np.asarray(item['effective_log_parameters'])
            record = item['record']
            np.testing.assert_allclose(target(z)[0], record['log_posterior'], atol=1e-7)
        else:
            last_update = time.monotonic()
            evaluations = 0
            def objective(z):
                nonlocal last_update, evaluations
                value, gradient = target(z)
                evaluations += 1
                if progress and time.monotonic()-last_update > 45:
                    print(f'{model}: start {index+1}/{starts}, evaluation {evaluations}, '
                          f'logp={value:.6f}, gradient={np.max(abs(gradient)):.2g}', flush=True)
                    last_update = time.monotonic()
                return -value, -gradient
            result = optimize.minimize(objective, guess, jac=True, method='L-BFGS-B',
                options=dict(maxiter=2000, ftol=1e-14, gtol=1e-8, maxls=40, maxcor=30))
            z = result.x
            # Refine near-stationary modes if the relative objective stopping
            # criterion triggers before the gradient is sufficiently small.
            for _ in range(4):
                value, gradient = target(z)
                if np.max(abs(gradient)) < 1e-5:
                    break
                result = optimize.minimize(objective, z, jac=True, method='L-BFGS-B',
                    options=dict(maxiter=500, ftol=1e-15, gtol=1e-9, maxls=50))
                z = result.x
            value, gradient = target(z)
            record = dict(start=index, log_posterior=float(value),
                          max_abs_gradient=float(max(abs(gradient))),
                          evaluations=evaluations,
                          converged=bool(np.isfinite(value) and max(abs(gradient)) < 1e-4),
                          optimizer_message=str(result.message))
            if progress:
                progress(record, z)
        attempts.append(record)
        if record['converged']:
            value, gradient = target(z)
            assert max(abs(gradient)) < 1e-4
            solutions.append((value, z, index))
    if not solutions:
        raise RuntimeError(f'{model}: none of {starts} starts reached a stationary mode')
    value, z, index = max(solutions, key=lambda item: item[0])
    h = 1e-4
    offsets = np.eye(len(z))*h
    differences = [(target(z+e), target(z-e)) for e in offsets]
    numerical = np.array([(plus[0]-minus[0])/(2*h) for plus,minus in differences])
    hessian = np.column_stack([(plus[1]-minus[1])/(2*h) for plus,minus in differences])
    eigmax = float(np.linalg.eigvalsh((hessian+hessian.T)/2).max())
    np.testing.assert_allclose(numerical, target(z)[1], atol=2e-5, rtol=1e-4)
    if eigmax >= 0:
        raise RuntimeError('Selected stationary point is not a strict local maximum')
    q = coordinates(z, model, inverse=True)
    return dict(model=model, log_parameters=q, effective_log_parameters=z, numerical_backend=backend,
                parameter_names=parameter_names(d, model), prior_center=center, prior_sd=2.,
                log_posterior=float(value), selected_start=index, attempts=attempts,
                max_abs_gradient=float(max(abs(target(z)[1]))),
                max_hessian_eigenvalue=eigmax,
                gradient_check_max_error=float(max(abs(numerical-target(z)[1]))))


class PluginPredictor:
    """Fit no new parameters: condition on the supplied log-coordinate MAP."""
    def __init__(self, y, x, log_parameters, model, *, lam=1., omega=1., nu=5.):
        self.y, self.x, self.model = np.asarray(y), np.asarray(x), model
        self.d = x.shape[1]
        self.single = model in ZERO_MODELS
        q = np.asarray(log_parameters)
        size = 2+self.d if self.single else 3+2*self.d
        if model not in ALL_MODELS or q.shape != (size+has_mixing(model),):
            raise ValueError('Invalid model or parameter shape')
        physical = np.exp(q)
        if self.single:
            self.v, self.k = physical[:2]
            self.w = physical[2:2+self.d]
        else:
            self.v, self.a, self.k = physical[:3]
            self.w, self.w_latent = physical[3:3+self.d], physical[3+self.d:3+2*self.d]
        self.W_map = float(physical[-1]) if has_mixing(model) else 1.
        r = 2 if model == 'beta_gp' else 0
        self.A = se_kernel(x, x, self.v, self.w)+self.k*np.eye(len(x))
        if self.single:
            self.function_train = np.zeros(len(x))
        else:
            sigma = se_kernel(x, x, self.a, self.w_latent)
            C = self.W_map*self.A + self.W_map**r*sigma
            self.alpha = linalg.cho_solve((linalg.cholesky(C, lower=True), True), y)
            self.function_coefficient = self.W_map if model == 'beta_gp' else 1.
            self.function_train = self.function_coefficient*sigma@self.alpha
        self.L = linalg.cholesky(self.A, lower=True)
        mu = np.zeros(len(x)) if model == 'beta_gp' else self.function_train
        beta = self.function_train if model == 'beta_gp' else np.zeros(len(x))
        self.ar = linalg.cho_solve((self.L, True), y-mu)
        self.ab = linalg.cho_solve((self.L, True), beta)
        self.mixing = 'gp' if not has_mixing(model) else ('ig' if model in ('tp_mu_gp', 'stp_zero') else 'gig')
        self.mixing_parameters = dict(mixing=self.mixing)
        if self.mixing == 'gig':
            p, chi, psi = lam-len(x)/2, omega+(y-mu)@self.ar, omega+beta@self.ab
            ew, varw = gig_moments(p, chi, psi)
            self.mixing_parameters.update(W_lambda=p, W_chi=float(chi), W_psi=float(psi))
        elif self.mixing == 'ig':
            shape, scale = (nu+len(x))/2, (nu-2+(y-mu)@self.ar)/2
            ew, varw = scale/(shape-1), scale**2/((shape-1)**2*(shape-2))
            self.mixing_parameters.update(W_shape=shape, W_scale=float(scale))
        else:
            ew, varw = 1., 0.
        if not np.isfinite([ew, varw]).all() or ew <= 0 or varw < 0:
            raise FloatingPointError('Invalid posterior W moments')
        self.mixing_parameters.update(W_mean=float(ew), W_var=float(varw))

    def function(self, xp):
        if self.single:
            return np.zeros(len(xp))
        return self.function_coefficient*se_kernel(xp, self.x, self.a, self.w_latent)@self.alpha

    def predict(self, xp, *, y_mean=0., y_scale=1.):
        """Exact pointwise moments, compact batch covariance placeholders.

        Diagonal matrices below are interfaces to existing marginal scoring;
        they deliberately do not represent the joint predictive covariance.
        These temporary matrices are never saved as covariance estimates.
        """
        function = self.function(xp)
        mu = np.zeros(len(xp)) if self.model == 'beta_gp' else function
        beta = function if self.model == 'beta_gp' else np.zeros(len(xp))
        cross = se_kernel(xp, self.x, self.v, self.w)
        solved = linalg.solve_triangular(self.L, cross.T, lower=True)
        variance = self.v-np.sum(solved**2, axis=0)
        if np.any(variance <= 0):
            raise FloatingPointError('Nonpositive conditional residual variance')
        p = dict(self.mixing_parameters,
                 location=y_mean+y_scale*(mu+cross@self.ar),
                 conditional_beta=y_scale*(beta-cross@self.ab))
        for name, noise in [('f', 0.), ('y', self.k)]:
            c = y_scale**2*(variance+noise)
            p[f'{name}_scale_cov'] = np.diag(c)
            p[f'{name}_mean'] = p['location']+p['W_mean']*p['conditional_beta']
            p[f'{name}_var'] = p['W_mean']*c+p['W_var']*p['conditional_beta']**2
            dist = StableGHMarginals(p, name) if self.mixing == 'gig' else marginal_distribution(p, name)
            p[f'{name}_ci'] = np.column_stack([dist.ppf(.025), dist.ppf(.975)])
        return p
