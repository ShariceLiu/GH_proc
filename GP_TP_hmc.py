"""Gaussian and Student-t process baselines with a sum of two SE kernels.

Parameter order: [v1, v2, k, w1, w2]; kernel weights w are inverse length^2.
GP: f~GP(0,K), y=f+epsilon, epsilon~N(0,kI).
TP: W~InvGamma(nu/2,(nu-2)/2), f|W~GP(0,W*K), epsilon|W~N(0,W*kI).
The inverse-gamma second parameter is scale: p(W) proportional to
W**(-alpha-1)*exp(-scale/W). Signal and noise share one W per realization.

The TP uses the covariance parameterization in Shah, Wilson & Ghahramani
(2014), sections 4.2, 4.3, 4.6 and supplementary likelihood derivatives:
https://proceedings.mlr.press/v33/shah14.pdf
W and the functions are marginalized analytically. HMC samples all five log
hyperparameters with proper Normal priors. No MAP plug-in or W quadrature.

For the alternative K2+W*K1 model (independent GP mean plus TP residual),
use GH_proc_joint_hmc with model='tp_mu_gp' and hmc_tp_mu_1d_experiment.py.
That model samples W explicitly and has Gaussian conditional predictions.
"""

import warnings

import numpy as np
from scipy import linalg, optimize, special

from GH_proc_joint_hmc import _inputs_1d
from GH_proc_w_betaprior import _mixture_intervals
from GH_proc_w_betaprior_hmc import sample_hmc


PARAMETER_NAMES = ['v1', 'v2', 'k', 'w1', 'w2']


def _validate_model(model, nu):
    if model not in ('gp', 'tp'):
        raise ValueError('model must be gp or tp')
    if model == 'tp' and (not np.isfinite(nu) or nu <= 2):
        raise ValueError('This covariance-parameterized TP requires finite nu > 2')


def two_se_kernel(x, xp, log_parameters):
    v1, v2, _, w1, w2 = np.exp(log_parameters)
    distances = (np.asarray(x)[:, None]-np.asarray(xp)[None, :])**2
    return v1*np.exp(-0.5*w1*distances)+v2*np.exp(-0.5*w2*distances)


def make_baseline_target(y, x, *, model, nu=5., prior_center, prior_sd=2.):
    """Return (log posterior, analytic gradient) in ORIGINAL log parameters.

    The likelihood is normalized; the posterior is known up to its evidence.
    Normal priors are specified directly on logs, so no extra theta Jacobian
    is added. The TP likelihood already integrates its inverse-gamma W prior.
    """
    _validate_model(model, nu)
    y, x = _inputs_1d(y, x)
    center = np.asarray(prior_center, dtype=float)
    sd = np.broadcast_to(np.asarray(prior_sd, dtype=float), (5,)).copy()
    if center.shape != (5,) or not np.all(np.isfinite(center)):
        raise ValueError('prior_center must contain five finite log means')
    if np.any(sd <= 0) or not np.all(np.isfinite(sd)):
        raise ValueError('prior_sd must be positive and finite')
    n = len(y)
    distances = (x[:, None]-x[None, :])**2
    eye = np.eye(n)
    prior_constant = -np.log(sd).sum()-2.5*np.log(2*np.pi)
    normal_constant = -0.5*n*np.log(2*np.pi)
    tp_constant = (special.gammaln((nu+n)/2)-special.gammaln(nu/2)
                   -0.5*n*np.log((nu-2)*np.pi)) if model == 'tp' else 0.

    def target(q):
        q = np.asarray(q, dtype=float)
        if q.shape != (5,) or not np.all(np.isfinite(q)):
            return -np.inf, np.zeros(5)
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                v1, v2, k, w1, w2 = np.exp(q)
                first = v1*np.exp(-0.5*w1*distances)
                second = v2*np.exp(-0.5*w2*distances)
                C = first+second+k*eye
                L = linalg.cholesky(C, lower=True, check_finite=False)
                alpha = linalg.cho_solve((L, True), y, check_finite=False)
                quadratic = y@alpha
                half_logdet = np.log(L.diagonal()).sum()
                if model == 'gp':
                    loglike = normal_constant-half_logdet-0.5*quadratic
                    multiplier = 1.
                else:
                    loglike = tp_constant-half_logdet-0.5*(nu+n)*np.log1p(quadratic/(nu-2))
                    multiplier = (nu+n)/(nu-2+quadratic)
                score = 0.5*(multiplier*np.outer(alpha, alpha)
                             -linalg.cho_solve((L, True), eye, check_finite=False))
                gradient = np.array([np.sum(score*first), np.sum(score*second), k*np.trace(score),
                                     np.sum(score*first*(-0.5*w1*distances)),
                                     np.sum(score*second*(-0.5*w2*distances))])
                standardized = (q-center)/sd
                value = loglike+prior_constant-0.5*standardized@standardized
                gradient -= standardized/sd
                if np.isfinite(value) and np.all(np.isfinite(gradient)):
                    return float(value), gradient
        except (np.linalg.LinAlgError, FloatingPointError, OverflowError):
            pass
        return -np.inf, np.zeros(5)

    return target


def sample_baseline_parameters(y, x, *, model, nu=5., prior_center=None, prior_sd=2.,
                               chains=4, warmup=1000, draws=1500, seed=3026,
                               target_accept=0.9, leapfrog_steps=(20, 50), verbose=False):
    """HMC of both SE amplitudes/weights and noise, fixing the TP's nu.

    Default empirical log-prior centers match the earlier beta/mu experiments:
    log([.5*var(y), .5*var(y), .1*var(y), 1, .25]). Only training y is used.
    Supply prior_center for external priors. Normal log priors are untruncated.
    Optimization only initializes dispersed chains. Metropolis kernel exchanges
    preserve the full posterior and avoid trapping chains in label assignments.
    """
    y, x = _inputs_1d(y, x)
    if not isinstance(chains, (int, np.integer)) or chains < 2:
        raise ValueError('Use at least two chains')
    scale = float(np.var(y))
    scale = scale if scale > 1e-12 else 1.
    center = np.log([.5*scale, .5*scale, .1*scale, 1., .25]) if prior_center is None else np.asarray(prior_center)
    sd = np.broadcast_to(np.asarray(prior_sd, dtype=float), (5,)).copy()
    target = make_baseline_target(y, x, model=model, nu=nu, prior_center=center, prior_sd=sd)
    rng = np.random.default_rng(seed)
    modes, initialization = [], []
    for chain in range(chains):
        guess = center+rng.normal(0., .7, 5)
        if chain % 2:
            guess = guess[[1, 0, 2, 4, 3]]
        def objective(q):
            value, gradient = target(q)
            return -value, -gradient
        result = optimize.minimize(objective, guess, jac=True, method='L-BFGS-B',
                                   options=dict(maxiter=400, ftol=1e-10))
        mode = result.x if np.isfinite(target(result.x)[0]) else guess
        modes.append(mode)
        initialization.append(dict(success=bool(result.success), log_density=target(mode)[0], message=str(result.message)))
    starts = np.stack(modes)+rng.normal(0., .35, (chains, 5))
    fit = sample_hmc(target, starts, draws=draws, warmup=warmup, seed=seed,
                     step_size=.05, target_accept=target_accept, leapfrog_steps=leapfrog_steps,
                     swap_pairs=[(0, 1), (3, 4)], verbose=verbose)
    fit['log_samples'] = fit.pop('samples')
    fit['warmup_log_samples'] = fit.pop('warmup_samples')
    fit['parameter_samples'] = np.exp(fit['log_samples'])
    fit.update(model=model, nu=nu if model == 'tp' else None, prior_center=center,
               prior_sd=sd, parameter_names=PARAMETER_NAMES, initialization=initialization, starts=starts)
    flat = fit['parameter_samples'].reshape(-1, 5)
    quantiles = np.quantile(flat, [.025, .5, .975], axis=0)
    fit['summary'] = [dict(parameter=name, mean=float(flat[:, j].mean()), std=float(flat[:, j].std()),
                           variance=float(flat[:, j].var()), median=float(quantiles[1, j]),
                           lower=float(quantiles[0, j]), upper=float(quantiles[2, j]),
                           **{key: float(value[j]) for key, value in fit['diagnostics'].items()})
                      for j, name in enumerate(PARAMETER_NAMES)]
    diagnostics = fit['diagnostics']
    fit['diagnostics_ok'] = bool(np.max(diagnostics['rhat']) < 1.01
                               and np.min(diagnostics['ess_bulk']) >= 400
                               and np.min(diagnostics['ess_tail']) >= 400
                               and not np.any(fit['divergent']))
    if not fit['diagnostics_ok']:
        warnings.warn(f'{model}: inspect HMC diagnostics before interpreting summaries', RuntimeWarning)
    return fit


def conditional_prediction(y, x, x_pred, log_parameters, *, model, nu=5.):
    """Exact Gaussian/Student-t prediction conditional on theta, integrating W.

    Returns COVARIANCE matrices, not SciPy t shape matrices. For TP, df=nu+n
    and shape=((df-2)/df)*covariance. f and fresh y share the same mean.
    W|theta,y is InvGamma((nu+n)/2,(nu-2+y'C^-1 y)/2).
    """
    _validate_model(model, nu)
    y, x = _inputs_1d(y, x)
    xp, q = np.asarray(x_pred, dtype=float), np.asarray(log_parameters, dtype=float)
    if xp.ndim != 1 or not xp.size or not np.all(np.isfinite(xp)):
        raise ValueError('x_pred must be a nonempty finite 1D array')
    if q.shape != (5,) or not np.all(np.isfinite(q)):
        raise ValueError('log_parameters must contain five finite values')
    k = float(np.exp(q[2]))
    C = two_se_kernel(x, x, q)+k*np.eye(len(x))
    cross = two_se_kernel(xp, x, q)
    prior = two_se_kernel(xp, xp, q)
    L = linalg.cholesky(C, lower=True, check_finite=False)
    alpha = linalg.cho_solve((L, True), y, check_finite=False)
    solved = linalg.solve_triangular(L, cross.T, lower=True, check_finite=False)
    base = prior-solved.T@solved
    base = (base+base.T)/2
    mean = cross@alpha
    result = dict(f_mean=mean, y_mean=mean.copy(), df=np.inf)
    multiplier = 1.
    if model == 'tp':
        shape = .5*(nu+len(y))
        scale = .5*(nu-2+y@alpha)
        multiplier = scale/(shape-1)
        result.update(df=nu+len(y), W_shape=shape, W_scale=scale,
                      W_mean=multiplier,
                      W_var=scale**2/((shape-1)**2*(shape-2)) if shape > 2 else np.inf)
    result.update(f_cov=multiplier*base,
                  y_cov=multiplier*(base+k*np.eye(len(xp))))
    return result


def mixture_logpdf(values, means, variances, *, df=np.inf, weights=None):
    """Pointwise log predictive density, using variances rather than t scales."""
    means, variances = np.asarray(means), np.asarray(variances)
    values = np.asarray(values)
    if (means.ndim != 2 or variances.shape != means.shape or values.shape != means.shape[1:]
            or np.any(variances <= 0) or not np.all(np.isfinite(variances))):
        raise ValueError('Need values[N], means[S,N] and finite positive variances[S,N]')
    if weights is None:
        weights = np.full(len(means), 1/len(means))
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (len(means),) or np.any(weights < 0) or not np.isfinite(weights.sum()) or weights.sum() <= 0:
        raise ValueError('Invalid mixture weights')
    log_weights = np.full_like(weights, -np.inf)
    np.log(weights/weights.sum(), out=log_weights, where=weights > 0)
    if np.isposinf(df):
        logpdf = -.5*(np.log(2*np.pi*variances)+(values-means)**2/variances)
    else:
        if not np.isfinite(df) or df <= 2:
            raise ValueError('Finite df must exceed two for covariance parameterization')
        scale2 = variances*((df-2)/df)
        logpdf = (special.gammaln((df+1)/2)-special.gammaln(df/2)
                  -.5*np.log(df*np.pi*scale2)-.5*(df+1)*np.log1p((values-means)**2/(df*scale2)))
    return special.logsumexp(logpdf+log_weights[:, None], axis=0)


def mixture_intervals(means, variances, weights, *, df=np.inf, credible_mass=.95):
    """Equal-tail marginal mixture quantiles (Student-t or Gaussian)."""
    if not 0 < credible_mass < 1:
        raise ValueError('credible_mass must lie in (0,1)')
    if np.isposinf(df):
        return _mixture_intervals(means, variances, weights, credible_mass)
    if not np.isfinite(df) or df <= 2:
        raise ValueError('Finite df must exceed two')
    scales = np.sqrt(np.maximum(variances, 0)*((df-2)/df))
    intervals = []
    for probability in [(1-credible_mass)/2, (1+credible_mass)/2]:
        quantiles = means+special.stdtrit(df, probability)*scales
        lower, upper = quantiles.min(axis=0), quantiles.max(axis=0)
        for _ in range(55):
            mid = (lower+upper)/2
            z = np.divide(mid-means, scales, out=np.zeros_like(means), where=scales > 0)
            cdf = np.where(scales > 0, special.stdtr(df, z), mid >= means)
            below = weights@cdf < probability
            lower, upper = np.where(below, mid, lower), np.where(below, upper, mid)
        intervals.append((lower+upper)/2)
    return np.column_stack(intervals)


def predict_baseline_mixture(y, x, x_pred, fit, *, credible_mass=.95):
    """Average all conditional predictions; full covariance includes theta uncertainty.

    TP components are Student-t, not Gaussian, because W is integrated exactly.
    W posterior moments also integrate theta via the law of total variance.
    """
    samples = np.asarray(fit['log_samples'])
    if samples.ndim != 3 or samples.shape[-1] != 5 or not samples.size:
        raise ValueError('log_samples must have shape (chains,draws,5)')
    model, nu = fit['model'], fit['nu'] if fit['model'] == 'tp' else 5.
    flat = samples.reshape(-1, 5)
    count, n = len(flat), len(x_pred)
    means = np.empty((count, n))
    variances = {name: np.empty((count, n)) for name in ['f', 'y']}
    within = {name: np.zeros((n, n)) for name in ['f', 'y']}
    w_scale = np.empty(count) if model == 'tp' else None
    for i, q in enumerate(flat):
        p = conditional_prediction(y, x, x_pred, q, model=model, nu=nu)
        means[i] = p['f_mean']
        for name in ['f', 'y']:
            variances[name][i] = np.maximum(np.diag(p[f'{name}_cov']), 0.)
            within[name] += p[f'{name}_cov']/count
        if w_scale is not None:
            w_scale[i] = p['W_scale']
    weights = np.full(count, 1/count)
    mean = weights@means
    centered = means-mean
    between = (centered.T*weights)@centered
    result = dict(weights=weights, draws_used=count, credible_mass=credible_mass,
                  component_df=p['df'], component_means=means)
    for name in ['f', 'y']:
        covariance = within[name]+between
        variance = np.maximum(np.diag(covariance), 0.)
        result.update({f'{name}_mean': mean.copy(), f'{name}_cov': covariance,
                       f'{name}_var': variance, f'{name}_std': np.sqrt(variance),
                       f'{name}_within_cov': within[name], f'{name}_between_cov': between.copy(),
                       f'component_{name}_variances': variances[name],
                       f'{name}_ci': mixture_intervals(means, variances[name], weights,
                                                      df=p['df'], credible_mass=credible_mass)})
    if w_scale is not None:
        shape = p['W_shape']
        w_means = w_scale/(shape-1)
        w_mean = weights@w_means
        w_within = weights@(w_scale**2/((shape-1)**2*(shape-2))) if shape > 2 else np.inf
        w_var = w_within+weights@(w_means-w_mean)**2
        result.update(W_shape=shape, component_W_scales=w_scale, W_mean=w_mean,
                      W_var=w_var, W_std=np.sqrt(w_var))
    return result
