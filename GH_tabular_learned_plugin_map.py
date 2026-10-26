"""Joint MAP for the existing fixed-function plug-in experiment, learning mixing.

Optimize log(theta), log W and lambda/log omega (GHP), or log(nu-2)
(STP). Gaussian processes are integrated out during fitting. The effective
log-amplitude transform has determinant one. Prediction is subsequently
handled by PluginPredictor, with the fitted mixing hyperparameters supplied.
"""
import time
import numpy as np
from scipy import optimize

from GH_learned_mixing import mixing_prior, mixing_log_density
from GH_tabular_hmc import inputs, make_target, default_center, coordinates, parameter_names

MODELS = ('tp_mu_gp', 'mu_gp', 'beta_gp')


class LearnedPluginTarget:
    def __init__(self, y, x, model, *, prior_center=None, prior_sd=2., backend='dense'):
        y, x = inputs(y, x)
        if model not in MODELS:
            raise ValueError('Expected STP-mu, GHP-mu or GHP-beta')
        self.model, self.d = model, x.shape[1]
        self.family = 'tp_mu_gp' if model == 'tp_mu_gp' else 'beta_gp'
        self.qdim = 3+2*self.d
        self.hdim = 1 if model == 'tp_mu_gp' else 2
        self.dimension = self.qdim+1+self.hdim
        self.center = default_center(y, self.d, model) if prior_center is None else np.asarray(prior_center)
        self.prior_sd = prior_sd
        self.h0 = np.array([np.log(3.)]) if self.hdim == 1 else np.array([1., 0.])
        self.base = make_target(y, x, model=model, prior_center=self.center, prior_sd=prior_sd, backend=backend)

    def initial(self):
        t = np.log(.6) if self.hdim == 1 else np.arcsinh(1.)
        return np.r_[coordinates(np.r_[self.center, t], self.model), self.h0]

    def __call__(self, z):
        z = np.asarray(z, float)
        if z.shape != (self.dimension,) or not np.isfinite(z).all():
            return -np.inf, np.zeros(self.dimension)
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                # Reuse the established ARD Gaussian likelihood and kernel
                # priors, replacing only its original fixed mixing density.
                value, base_grad = self.base(z[:self.qdim+1])
                if not np.isfinite(value):
                    return -np.inf, np.zeros(self.dimension)
                t, h = z[self.qdim], z[self.qdim+1:]
                old, old_t, _ = mixing_log_density(t, self.h0, self.family, True)
                new, new_t, new_h = mixing_log_density(t, h, self.family, True)
                prior, prior_grad = mixing_prior(h, self.family)
                value += new-old+prior
                base_grad[-1] += new_t-old_t
                grad = np.r_[base_grad, new_h+prior_grad]
                if np.isfinite(value) and np.isfinite(grad).all():
                    return float(value), grad
        except (ArithmeticError, ValueError, np.linalg.LinAlgError):
            pass
        return -np.inf, np.zeros(self.dimension)


def fit_learned_map(y, x, model, *, starts=24, seed=505, progress=None, completed=(),
                    backend='dense', initial_points=()):
    if starts < 2:
        raise ValueError('Need at least two starts')
    target = LearnedPluginTarget(y, x, model, backend=backend)
    initial_points = [np.asarray(z) for z in initial_points]
    bounds = [(-20., 20.)]*3+[(-15., 15.)]*(2*target.d)+[(-10., 10.)]
    bounds += [(-8., 8.)] if target.hdim == 1 else [(-15., 15.), (-8., 8.)]
    rng = np.random.default_rng(seed)
    previous = {item['record']['start']:item for item in completed}
    attempts, solutions = [], []
    for index in range(starts):
        guess = target.initial()
        if index >= 2:
            guess[:target.qdim+1] += rng.normal(0, 1.5, target.qdim+1)
            guess[target.qdim+1:] += rng.normal(0, .6, target.hdim)
        if index % 2:
            guess[[0, 1]] = guess[[1, 0]]
            a, b = np.arange(3, 3+target.d), np.arange(3+target.d, target.qdim)
            guess[np.r_[a, b]] = guess[np.r_[b, a]]
        if index < len(initial_points):
            guess = initial_points[index].copy()
        if not np.isfinite(target(guess)[0]):
            raise ValueError('Invalid optimization initial point')
        if index in previous:
            item = previous[index]
            z = np.asarray(item['effective_log_parameters'])
            record = item['record']
            np.testing.assert_allclose(target(z)[0], record['log_posterior'], atol=1e-7, rtol=0)
        else:
            evaluations, last_update = 0, time.monotonic()
            def objective(z):
                nonlocal evaluations, last_update
                value, grad = target(z)
                evaluations += 1
                if time.monotonic()-last_update > 45:
                    print(f'{model}: start {index+1}/{starts}, evaluation {evaluations}, logp={value:.6f}', flush=True)
                    last_update = time.monotonic()
                return -value, -grad
            result = optimize.minimize(objective, guess, jac=True, method='L-BFGS-B', bounds=bounds,
                options=dict(maxiter=2000, ftol=1e-14, gtol=1e-8, maxls=40, maxcor=30))
            z = result.x
            for _ in range(3):
                if max(abs(target(z)[1])) < 1e-5:
                    break
                result = optimize.minimize(objective, z, jac=True, method='L-BFGS-B', bounds=bounds,
                    options=dict(maxiter=500, ftol=1e-15, gtol=1e-9, maxls=50))
                z = result.x
            value, gradient = target(z)
            record = dict(start=index, log_posterior=float(value), max_abs_gradient=float(max(abs(gradient))),
                evaluations=evaluations, converged=bool(np.isfinite(value) and max(abs(gradient)) < 1e-4),
                optimizer_message=str(result.message))
            if progress:
                progress(record, z)
        attempts.append(record)
        if record['converged']:
            value, grad = target(z)
            if max(abs(grad)) >= 1e-4:
                raise AssertionError('Saved MAP candidate is not stationary')
            solutions.append((value, z.copy(), index))
    if not solutions:
        raise RuntimeError('No converged MAP candidate')
    value, z, index = max(solutions, key=lambda r:r[0])
    if not all(lo+1e-3 < v < hi-1e-3 for v, (lo, hi) in zip(z, bounds)):
        raise RuntimeError('Selected MAP touches a numerical bound')
    eps = np.eye(len(z))*1e-4
    differences = [(target(z+e), target(z-e)) for e in eps]
    numerical = np.array([(plus[0]-minus[0])/2e-4 for plus, minus in differences])
    hessian = np.column_stack([(plus[1]-minus[1])/2e-4 for plus, minus in differences])
    max_eigenvalue = float(np.linalg.eigvalsh((hessian+hessian.T)/2).max())
    np.testing.assert_allclose(numerical, target(z)[1], atol=2e-5, rtol=1e-4)
    if max_eigenvalue >= 0:
        raise RuntimeError('Selected stationary point is not a strict local maximum')
    base, h = z[:target.qdim+1], z[target.qdim+1:]
    mixing = dict(nu=float(2+np.exp(h[0]))) if target.hdim == 1 else dict(lam=float(h[0]), omega=float(np.exp(h[1])))
    return dict(model=model, log_parameters=coordinates(base, model, inverse=True),
        effective_log_parameters=base, optimizer_coordinates=z,
        mixing_coordinates=h, mixing_hyperparameters=mixing,
        numerical_backend=backend, parameter_names=parameter_names(target.d, model),
        prior_center=target.center, prior_sd=target.prior_sd, log_posterior=value,
        selected_start=index, attempts=attempts, numerical_bounds=bounds,
        max_abs_gradient=float(max(abs(target(z)[1]))), max_hessian_eigenvalue=max_eigenvalue,
        gradient_check_max_error=float(max(abs(numerical-target(z)[1]))))
