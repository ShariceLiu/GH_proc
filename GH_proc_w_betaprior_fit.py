"""Fit both SE kernels and noise with fixed GIG parameters; compare fixed omega.

MAP here means the mode in LOG positive parameters, with independent Gaussian
priors on those log parameters. Defaults are broad, training-scale priors.
Use log_prior_sd=None for bounded maximum marginal likelihood instead.
Posterior prediction plugs in fitted hyperparameters; bands integrate W,
beta and f but do not integrate hyperparameter uncertainty.
"""

import warnings

import numpy as np
from scipy import optimize, special

from GH_proc_w_betaprior import (
    _check_model, _inputs, _vector, beta_f_inf, log_marginal_likelihood,
)


def _pack(parameters, dimension, fit_mu):
    w, sw = _check_model(-0.5, 1, parameters['w'], parameters['v_0'],
                         parameters['k'], parameters['mu'], parameters['sigma_v0'],
                         parameters['sigma_w'], dimension)
    positive = np.r_[parameters['v_0'], parameters['sigma_v0'], parameters['k'], w, sw]
    if np.any(positive <= 0):
        raise ValueError("Fitted variances and kernel weights must be strictly positive")
    vector = np.log(positive)
    return np.r_[vector, parameters['mu']] if fit_mu else vector


def _unpack(vector, dimension, l, omega, mu, fit_mu):
    positive = np.exp(vector[:3 + 2*dimension])
    return dict(v_0=float(positive[0]), sigma_v0=float(positive[1]), k=float(positive[2]),
                w=positive[3:3+dimension], sigma_w=positive[3+dimension:],
                mu=float(vector[-1]) if fit_mu else float(mu), l=float(l), omega=float(omega))


def fit_kernel_parameters(ys, xs, *, l=-0.5, omega=1, mu=0, fit_mu=False,
                           initial=None, prior_centers=None, log_prior_sd=2.0,
                           n_restarts=3, maxiter=200, seed=0,
                           log_bounds=(-10, 10), quad_options=None, verbose=False):
    """Fit v_0, sigma_v0, k, w[D], sigma_w[D], optionally mu; fix l and omega.

    xs is (N,D) or (N,) and ys is (N,). Standardizing inputs and centering/
    scaling outputs using training statistics is recommended. No preprocessing
    is performed here, so returned params can be passed directly to beta_f_inf
    with these same inputs/outputs.

    The default starting/prior centers use training variance s2: v_0=sigma_v0
    =0.5*s2, k=0.1*s2, w[d]=1/D, sigma_w[d]=0.25/D. These training-dependent
    centers are an empirical choice, not externally specified prior knowledge.
    initial and prior_centers independently override any fitted parameters.
    Changing initial does not change the prior. With fit_mu=True the default
    mean center is mean(ys), and its prior std is 5*sqrt(s2).

    log_prior_sd is the std of Gaussian priors on log positive parameters;
    None disables all hyperparameter priors (bounded MLE). log_bounds are
    computational bounds on each log positive parameter. Multiple independent
    starts use L-BFGS-B with analytic, quadrature-averaged likelihood gradients.

    Returns params, log_marginal_likelihood, objective, initial_objective,
    converged, restart diagnostics, prior settings, and boundary_parameters.
    Nonconverged runs are reported; no convergence is inferred from mere exit.
    """
    return _fit_kernel_parameters(
        ys, xs, likelihood=log_marginal_likelihood, l=l, omega=omega,
        mu=mu, fit_mu=fit_mu, initial=initial, prior_centers=prior_centers,
        log_prior_sd=log_prior_sd, n_restarts=n_restarts, maxiter=maxiter,
        seed=seed, log_bounds=log_bounds, quad_options=quad_options, verbose=verbose)


def _fit_kernel_parameters(ys, xs, *, likelihood, l, omega, mu, fit_mu,
                            initial, prior_centers, log_prior_sd, n_restarts,
                            maxiter, seed, log_bounds, quad_options, verbose):
    """Shared MAP optimizer; likelihood adapter uses the beta parameter names.

    Both model wrappers use identical priors, initialization, bounds and
    optimizer settings. Only the marginalized likelihood differs.
    """
    xs, ys = _inputs(xs, 'xs'), _vector(ys, 'ys')
    if len(xs) != len(ys):
        raise ValueError("xs must have one row per observation in ys")
    dimension = 1 if xs.ndim == 1 else xs.shape[1]
    _check_model(l, omega, 1, 1, 1, mu, 1, 1, dimension)
    if not isinstance(n_restarts, (int, np.integer)) or n_restarts < 1:
        raise ValueError("n_restarts must be a positive integer")
    if not isinstance(maxiter, (int, np.integer)) or maxiter < 1:
        raise ValueError("maxiter must be a positive integer")
    if log_prior_sd is not None and (not np.isfinite(log_prior_sd) or log_prior_sd <= 0):
        raise ValueError("log_prior_sd must be positive or None")
    log_bounds = np.asarray(log_bounds, dtype=float)
    if log_bounds.shape != (2,) or not np.all(np.isfinite(log_bounds)) or log_bounds[0] >= log_bounds[1]:
        raise ValueError("log_bounds must be two finite increasing values")
    quadrature = dict(quad_tol=1e-7)
    if quad_options:
        allowed = {'n_particles', 'max_particles', 'quad_tol', 'tail_log_drop', 'log_w_bounds'}
        if set(quad_options) - allowed:
            raise ValueError("quad_options may only set numerical quadrature controls")
        quadrature.update(quad_options)
    scale2 = float(np.var(ys))
    scale2 = scale2 if scale2 > 1e-12 else 1.0
    defaults = dict(v_0=0.5*scale2, sigma_v0=0.5*scale2, k=0.1*scale2,
                    w=np.full(dimension, 1/dimension), sigma_w=np.full(dimension, 0.25/dimension),
                    mu=float(np.mean(ys)) if fit_mu else float(mu))
    allowed = {'v_0', 'sigma_v0', 'k', 'w', 'sigma_w'} | ({'mu'} if fit_mu else set())
    for name, supplied in (('initial', initial), ('prior_centers', prior_centers)):
        if supplied is not None and set(supplied) - allowed:
            raise ValueError(f"{name} may only override fitted parameters: {sorted(allowed)}")
    start = _pack({**defaults, **(initial or {})}, dimension, fit_mu)
    center = _pack({**defaults, **(prior_centers or {})}, dimension, fit_mu)
    n_positive = 3 + 2*dimension
    if np.any(start[:n_positive] < log_bounds[0]) or np.any(start[:n_positive] > log_bounds[1]):
        raise ValueError("Initial parameters are outside log_bounds")
    sd = np.full(start.size, log_prior_sd if log_prior_sd is not None else 1.)
    if fit_mu:
        sd[-1] = 5*np.sqrt(scale2)

    def objective(vector):
        parameters = _unpack(vector, dimension, l, omega, mu, fit_mu)
        log_like, gradient = likelihood(ys, xs, **parameters,
                                        return_gradient=True, **quadrature)
        derivative = np.r_[gradient['log_v_0'], gradient['log_sigma_v0'],
                           gradient['log_k'], gradient['log_w'], gradient['log_sigma_w']]
        if fit_mu:
            derivative = np.r_[derivative, gradient['mu']]
        penalty, penalty_gradient = 0., np.zeros_like(vector)
        if log_prior_sd is not None:
            standardized = (vector - center) / sd
            penalty = 0.5 * np.dot(standardized, standardized)
            penalty_gradient = standardized / sd
        return -log_like + penalty, -derivative + penalty_gradient

    initial_objective = float(objective(start)[0])
    bounds = [tuple(log_bounds)]*n_positive + ([(None, None)] if fit_mu else [])
    names = ['log_v_0', 'log_sigma_v0', 'log_k'] + [f'log_w[{d}]' for d in range(dimension)]
    names += [f'log_sigma_w[{d}]' for d in range(dimension)]
    rng = np.random.default_rng(seed)
    results, restarts = [], []
    for restart in range(n_restarts):
        guess = start.copy()
        if restart:
            guess[:n_positive] = np.clip(guess[:n_positive] + rng.normal(0, 0.8, n_positive), *log_bounds)
            if fit_mu:
                guess[-1] += rng.normal(0, 0.25*np.sqrt(scale2))
        result = optimize.minimize(objective, guess, method='L-BFGS-B', jac=True,
                                   bounds=bounds, options=dict(maxiter=maxiter, ftol=1e-9, gtol=1e-5, maxls=30))
        results.append(result)
        diagnostics = dict(restart=restart, objective=float(result.fun), converged=bool(result.success),
                           iterations=int(result.nit), evaluations=int(result.nfev), message=str(result.message))
        restarts.append(diagnostics)
        if verbose:
            print(f"omega={omega:g}, restart={restart+1}: objective={result.fun:.4f}, "
                  f"converged={result.success}, iterations={result.nit}", flush=True)
    finite = [r for r in results if np.isfinite(r.fun) and np.all(np.isfinite(r.x))]
    if not finite:
        raise RuntimeError("No restart produced finite fitted parameters")
    converged = [r for r in finite if r.success]
    best = min(converged or finite, key=lambda r: r.fun)
    if not best.success:
        warnings.warn(f"No fit converged for omega={omega}: {best.message}", RuntimeWarning)
    fitted = _unpack(best.x, dimension, l, omega, mu, fit_mu)
    on_boundary = np.isclose(best.x[:n_positive], log_bounds[0], atol=1e-4) | np.isclose(best.x[:n_positive], log_bounds[1], atol=1e-4)
    return dict(params=fitted, log_marginal_likelihood=likelihood(ys, xs, **fitted, **quadrature),
                objective=float(best.fun), initial_objective=initial_objective,
                converged=bool(best.success), message=str(best.message), restarts=restarts,
                boundary_parameters=[name for name, hit in zip(names, on_boundary) if hit],
                prior=dict(log_prior_sd=log_prior_sd, vector_center=center, vector_sd=sd,
                           fit_mu=fit_mu), quad_options=quadrature)


def score_predictions(ys, posterior):
    """RMSE, MAE, mean marginal NLPD and pointwise coverage for noisy held-out y."""
    ys = _vector(ys, 'ys')
    if ys.shape != posterior['y_mean'].shape:
        raise ValueError("ys must match the prediction locations")
    m = len(ys)
    means = posterior['component_means'][:, m:]
    variances = posterior['component_y_variances']
    log_weights = np.full_like(posterior['weights'], -np.inf)
    np.log(posterior['weights'], out=log_weights, where=posterior['weights'] > 0)
    component_logpdf = -0.5*(np.log(2*np.pi*variances) + (ys-means)**2/variances)
    logpdf = special.logsumexp(log_weights[:, None] + component_logpdf, axis=0)
    error = posterior['y_mean'] - ys
    intervals = posterior['y_ci']
    return dict(rmse=float(np.sqrt(np.mean(error**2))), mse=float(np.mean(error**2)),
                mae=float(np.mean(np.abs(error))), mean_marginal_nlpd=float(-np.mean(logpdf)),
                interval_coverage=float(np.mean((ys >= intervals[:, 0]) & (ys <= intervals[:, 1]))),
                mean_interval_width=float(np.mean(intervals[:, 1]-intervals[:, 0])))


def compare_fixed_omega(ys, xs, y_valid, x_valid, *, omegas=(0.5, 1, 2, 5),
                         l=-0.5, fit_options=None, verbose=True):
    """Refit on the same training data for each fixed omega; evaluate on validation.

    Validation labels never enter fitting. This is sensitivity analysis: if
    results are used to choose omega, these are validation scores, not an
    independent final test score. The caller controls the split/preprocessing.
    Returns records (fit, posterior, metrics), baseline_omega, and beta-mean
    changes relative to omega=1 (or the first supplied omega if 1 is absent).
    """
    x_valid, y_valid = _inputs(x_valid, 'x_valid'), _vector(y_valid, 'y_valid')
    xs, ys = _inputs(xs, 'xs'), _vector(ys, 'ys')
    if len(x_valid) != len(y_valid) or len(xs) != len(ys):
        raise ValueError("Each input array must have one row per response")
    d = 1 if xs.ndim == 1 else xs.shape[1]
    if (1 if x_valid.ndim == 1 else x_valid.shape[1]) != d:
        raise ValueError("Training and validation inputs must have the same dimension")
    omegas = np.asarray(omegas, dtype=float)
    if omegas.ndim != 1 or not omegas.size or not np.all(np.isfinite(omegas)) or np.any(omegas <= 0):
        raise ValueError("omegas must be a nonempty sequence of positive finite values")
    if len(np.unique(omegas)) != len(omegas):
        raise ValueError("omegas must not contain duplicates")
    fit_options = dict(fit_options or {})
    if set(fit_options) & {'l', 'omega', 'verbose'}:
        raise ValueError("Set l/omegas/verbose on compare_fixed_omega, not in fit_options")
    records = []
    for omega in omegas:
        fit = fit_kernel_parameters(ys, xs, l=l, omega=float(omega), verbose=verbose, **fit_options)
        posterior = beta_f_inf(ys, xs, x_pred=x_valid, **fit['params'], **fit['quad_options'])
        metrics = score_predictions(y_valid, posterior)
        records.append(dict(omega=float(omega), fit=fit, posterior=posterior, metrics=metrics))
        if verbose:
            print(f"Validation: RMSE={metrics['rmse']:.4f}, marginal NLPD="
                  f"{metrics['mean_marginal_nlpd']:.4f}, coverage={metrics['interval_coverage']:.3f}", flush=True)
    baseline = next((r for r in records if r['omega'] == 1), records[0])
    for record in records:
        difference = record['posterior']['beta_mean'] - baseline['posterior']['beta_mean']
        record['beta_mean_shift_rmse'] = float(np.sqrt(np.mean(difference**2)))
    return dict(records=records, baseline_omega=baseline['omega'])
