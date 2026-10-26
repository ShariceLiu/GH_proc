"""Two-SE and one-SE regression models with ARD inputs and marginalized GPs.

Physical log parameters: [v1, v2, k, w1[0:D], w2[0:D], (W)].
Each w is an inverse squared length scale. HMC uses effective amplitudes
A=W*v1, B=W**r*v2, N=W*k (GP fixes W=1; r=2 only for beta_gp).
Independent Normal priors are on the physical log kernel/noise parameters;
the transformation between log coordinates has determinant one. Mixing-prior
densities include dW/dlogW. This extends the earlier 1D model, with the same
Metropolis-corrected HMC sampler and whole-component exchange proposals.
The *_zero variants remove the second SE component. Their physical logs are
[v_0, k, w[0:D], (W)], with effective amplitudes W*v_0 and W*k. There are
no component swaps for one-SE models. GP variants fix W=1, STP variants use
the fixed inverse-gamma prior, and GHP variants use the fixed GIG prior.
"""

import numpy as np
from scipy import linalg, optimize, special
from scipy.spatial.distance import pdist, squareform

from GH_proc_w_betaprior import _mixture_intervals
from GH_proc_w_betaprior_hmc import chain_diagnostics, sample_hmc

MODELS = ('gp', 'tp_mu_gp', 'mu_gp', 'beta_gp')
ZERO_MODELS = ('gp_zero', 'stp_zero', 'ghp_zero')
ALL_MODELS = (*MODELS, *ZERO_MODELS)


def has_mixing(model):
    return model not in ('gp', 'gp_zero')


def inputs(y, x):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if (x.ndim != 2 or not x.size or y.shape != (len(x),)
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))):
        raise ValueError('Need finite x[N,D] and y[N]')
    return y, x


def coordinates(values, model, *, inverse=False):
    if model not in ALL_MODELS:
        raise ValueError(f'Unknown model: {model}')
    result = np.array(values, dtype=float, copy=True)
    if has_mixing(model):
        t = result[..., -1].copy() * (-1 if inverse else 1)
        result[..., 0] += t
        if model in ZERO_MODELS:
            result[..., 1] += t
        else:
            result[..., 1] += (2 if model == 'beta_gp' else 0)*t
            result[..., 2] += t
    return result


def parameter_names(dimension, model):
    if model in ZERO_MODELS:
        return ['v_0', 'k']+[f'w[{j}]' for j in range(dimension)]+(['W'] if has_mixing(model) else [])
    latent = 'sigma' if model == 'beta_gp' else 'mu'
    first, second = ('v1', 'v2') if model == 'gp' else ('v_0', f'{latent}_v0')
    names = [first, second, 'k']
    names += [f'w1[{j}]' for j in range(dimension)]
    names += [f'w2[{j}]' for j in range(dimension)]
    return names + ([] if model == 'gp' else ['W'])


def default_center(y, dimension, model='gp'):
    scale = max(float(np.var(y)), 1e-12)
    if model in ZERO_MODELS:
        return np.log(np.r_[.5*scale, .1*scale, np.full(dimension, 1./dimension)])
    return np.log(np.r_[.5*scale, .5*scale, .1*scale,
                        np.full(dimension, 1./dimension),
                        np.full(dimension, .25/dimension)])


def squared_distances(x, xp):
    return np.ascontiguousarray((x.T[:, :, None]-xp.T[:, None, :])**2)


def make_target(y, x, *, model, prior_center, prior_sd=2., l=1., omega=1., nu=5., backend='dense'):
    """Normalized marginal likelihood + log priors, and analytic gradient."""
    y, x = inputs(y, x)
    if model not in ALL_MODELS:
        raise ValueError('Unknown model')
    n, dimension = x.shape
    single = model in ZERO_MODELS
    size = 2+dimension if single else 3+2*dimension
    total = size + has_mixing(model)
    center = np.asarray(prior_center, dtype=float)
    sd = np.broadcast_to(np.asarray(prior_sd, dtype=float), (size,)).copy()
    if center.shape != (size,) or not np.all(np.isfinite(center)):
        raise ValueError('Invalid log prior center')
    if not np.all(np.isfinite(sd)) or np.any(sd <= 0):
        raise ValueError('Invalid log prior standard deviation')
    if model in ('tp_mu_gp', 'stp_zero') and (not np.isfinite(nu) or nu <= 2):
        raise ValueError('Need nu > 2')
    if model in ('mu_gp', 'beta_gp', 'ghp_zero') and (not np.isfinite(l) or not np.isfinite(omega) or omega <= 0):
        raise ValueError('Invalid GIG parameters')
    if backend not in ('dense', 'condensed'):
        raise ValueError('backend must be dense or condensed')
    if backend == 'condensed':
        # Symmetric pair distances need only the strict upper triangle.
        # Construct one dimension at a time to avoid an N*N*D temporary.
        distances = np.empty((dimension, n*(n-1)//2))
        for j in range(dimension):
            distances[j] = pdist(x[:, j:j+1], metric='sqeuclidean')
    else:
        distances = squared_distances(x, x).reshape(dimension, -1)
    diagonal = np.diag_indices(n)

    def component(amplitude, weights):
        values = amplitude*np.exp(-.5*(weights@distances))
        if backend == 'dense':
            return values.reshape(n, n)
        matrix = squareform(values)
        matrix[diagonal] = amplitude
        return matrix

    def length_gradient(weighted_score, weights):
        # Off-diagonal symmetric entries occur twice in the full trace;
        # diagonal distance derivatives are zero.
        values = 2*squareform(weighted_score, checks=False) if backend == 'condensed' else weighted_score.ravel()
        return -.5*weights*(distances@values)
    constant = -.5*(n+size)*np.log(2*np.pi)-np.log(sd).sum()
    if model in ('tp_mu_gp', 'stp_zero'):
        shape, scale = nu/2., (nu-2)/2.
        mixing_constant = shape*np.log(scale)-special.gammaln(shape)
    elif has_mixing(model):
        mixing_constant = -np.log(2.)-np.log(special.kve(l, omega))+omega

    def target(z):
        z = np.asarray(z, dtype=float)
        if z.shape != (total,) or not np.all(np.isfinite(z)):
            return -np.inf, np.zeros(total)
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                pars = np.exp(z)
                if single:
                    a, noise = pars[:2]
                    w1 = pars[2:size]
                else:
                    a, b, noise = pars[:3]
                    w1, w2 = pars[3:3+dimension], pars[3+dimension:size]
                first = component(a, w1)
                if single:
                    covariance = first.copy()
                else:
                    second = component(b, w2)
                    covariance = first+second
                covariance[diagonal] += noise
                L = linalg.cholesky(covariance, lower=True, check_finite=False)
                alpha = linalg.cho_solve((L, True), y, check_finite=False)
                # POTRI forms one triangle of C^-1, saving redundant solves.
                lower_inverse, info = linalg.lapack.dpotri(L, lower=1)
                if info != 0:
                    raise np.linalg.LinAlgError('POTRI failed')
                inverse = lower_inverse+lower_inverse.T
                inverse[diagonal] *= .5
                score = .5*(np.outer(alpha, alpha)-inverse)
                sf = score*first
                gradient = np.empty(total)
                if single:
                    gradient[:2] = [sf.sum(), noise*np.trace(score)]
                    gradient[2:size] = length_gradient(sf, w1)
                else:
                    ss = score*second
                    gradient[:3] = [sf.sum(), ss.sum(), noise*np.trace(score)]
                    gradient[3:3+dimension] = length_gradient(sf, w1)
                    gradient[3+dimension:size] = length_gradient(ss, w2)
                q = coordinates(z, model, inverse=True)
                standardized = (q[:size]-center)/sd
                prior_gradient = -standardized/sd
                gradient[:size] += prior_gradient
                logp = constant-np.log(L.diagonal()).sum()-.5*y@alpha-.5*standardized@standardized
                if has_mixing(model):
                    W, t = pars[-1], z[-1]
                    if model in ('tp_mu_gp', 'stp_zero'):
                        logp += mixing_constant-shape*t-scale/W
                        mixing_gradient = -shape+scale/W
                    else:
                        logp += mixing_constant+l*t-.5*omega*(W+1/W)
                        mixing_gradient = l-.5*omega*(W-1/W)
                    r = 2 if model == 'beta_gp' else 0
                    if single:
                        gradient[-1] = mixing_gradient-prior_gradient[0]-prior_gradient[1]
                    else:
                        gradient[-1] = mixing_gradient-prior_gradient[0]-r*prior_gradient[1]-prior_gradient[2]
                if np.isfinite(logp) and np.all(np.isfinite(gradient)):
                    return float(logp), gradient
        except (np.linalg.LinAlgError, FloatingPointError, OverflowError):
            pass
        return -np.inf, np.zeros(total)

    return target


def diagnostic_summary(fit):
    ds = [fit['diagnostics'], fit['effective_diagnostics']]
    result = dict(max_rhat=max(float(np.max(d['rhat'])) for d in ds),
                  min_bulk_ess=min(float(np.min(d['ess_bulk'])) for d in ds),
                  min_tail_ess=min(float(np.min(d['ess_tail'])) for d in ds),
                  divergences=int(np.sum(fit['divergent'])),
                  acceptance=float(np.mean(fit['acceptance'])),
                  swap_acceptance=float(np.mean(fit['swap_accepted'])))
    result['diagnostics_ok'] = bool(result['max_rhat'] < 1.01 and result['min_bulk_ess'] >= 400
                                    and result['min_tail_ess'] >= 400 and result['divergences'] == 0)
    return result


def sample_parameters(y, x, *, model, prior_center=None, prior_sd=2., l=1., omega=1., nu=5.,
                      chains=4, warmup=1000, draws=1500, seed=18026,
                      target_accept=.95, leapfrog_steps=(12, 30), verbose=True):
    """Sample kernel/noise parameters and, for mixing models, the single W.

Optimization only initializes chains. Every retained state, including rejected
states, contributes to posterior predictions. No test targets enter this fit.
"""
    y, x = inputs(y, x)
    dimension = x.shape[1]
    center = default_center(y, dimension, model) if prior_center is None else np.asarray(prior_center)
    target = make_target(y, x, model=model, prior_center=center, prior_sd=prior_sd,
                         l=l, omega=omega, nu=nu)
    pairs = [] if model in ZERO_MODELS else [(0, 1)] + [(3+j, 3+dimension+j) for j in range(dimension)]
    base = center.copy()
    if has_mixing(model):
        logW = np.log((nu-2)/nu) if model in ('tp_mu_gp', 'stp_zero') else np.arcsinh(l/omega)
        base = coordinates(np.r_[center, logW], model)
    rng = np.random.default_rng(seed)
    starts, initialization = [], []
    for chain in range(chains):
        guess = base+rng.normal(0., .7, len(base))
        if chain % 2:
            for left, right in pairs:
                guess[[left, right]] = guess[[right, left]]
        def objective(z):
            value, gradient = target(z)
            return -value, -gradient
        mode = optimize.minimize(objective, guess, jac=True, method='L-BFGS-B',
                                 options=dict(maxiter=600, ftol=1e-10))
        position = mode.x if np.isfinite(target(mode.x)[0]) else guess
        initialization.append(dict(success=bool(mode.success), log_density=target(position)[0],
                                   message=str(mode.message)))
        for _ in range(1000):
            start = position+rng.normal(0., .35, len(base))
            if np.isfinite(target(start)[0]):
                starts.append(start)
                break
        else:
            raise RuntimeError('Could not initialize finite chain')
    fit = sample_hmc(target, np.array(starts), draws=draws, warmup=warmup, seed=seed,
                     step_size=.03, leapfrog_steps=leapfrog_steps, target_accept=target_accept,
                     swap_pairs=pairs or None, verbose=verbose)
    fit['effective_log_samples'] = fit.pop('samples')
    fit['log_samples'] = coordinates(fit['effective_log_samples'], model, inverse=True)
    fit['warmup_log_samples'] = coordinates(fit.pop('warmup_samples'), model, inverse=True)
    fit['effective_diagnostics'] = fit.pop('diagnostics')
    fit['diagnostics'] = chain_diagnostics(fit['log_samples'])
    fit.update(model=model, dimension=dimension, prior_center=center,
               prior_sd=np.broadcast_to(prior_sd, center.shape).copy(),
               parameter_names=parameter_names(dimension, model), initialization=initialization,
               starts=np.array(starts), l=l, omega=omega, nu=nu)
    fit.update(diagnostic_summary(fit))
    return fit


def conditional_marginals(y, x, xp, log_parameters, *, model, distances=None):
    """Exact pointwise f/y Gaussian moments for one hyperparameter draw.

Only diagonal variances are formed; this is exact for pointwise RMSE/NLPD and
intervals, and avoids constructing a large, unused test-test covariance.
"""
    y, x = inputs(y, x)
    xp = np.asarray(xp, dtype=float)
    if xp.ndim == 1 and x.shape[1] == 1:
        xp = xp[:, None]
    if xp.ndim != 2 or xp.shape[1] != x.shape[1] or not xp.size or not np.all(np.isfinite(xp)):
        raise ValueError('Invalid prediction inputs')
    d, n, m = x.shape[1], len(x), len(xp)
    z = coordinates(log_parameters, model)
    single = model in ZERO_MODELS
    size = 2+d if single else 3+2*d
    if z.shape != (size+has_mixing(model),) or not np.all(np.isfinite(z)):
        raise ValueError('Invalid log parameters')
    if single:
        a, noise = np.exp(z[:2])
        w1 = np.exp(z[2:size])
        b = 0.
    else:
        a, b, noise = np.exp(z[:3])
        w1, w2 = np.exp(z[3:3+d]), np.exp(z[3+d:3+2*d])
    if distances is None:
        distances = (squared_distances(x, x).reshape(d, -1),
                     squared_distances(x, xp).reshape(d, -1))
    dt, dc = distances
    covariance = (a*np.exp(-.5*(w1@dt))).reshape(n, n)
    if not single:
        covariance += (b*np.exp(-.5*(w2@dt))).reshape(n, n)
    covariance[np.diag_indices(n)] += noise
    cross = (a*np.exp(-.5*(w1@dc))).reshape(n, m)
    if not single:
        cross += (b*np.exp(-.5*(w2@dc))).reshape(n, m)
    L = linalg.cholesky(covariance, lower=True, check_finite=False)
    alpha = linalg.cho_solve((L, True), y, check_finite=False)
    solved = linalg.solve_triangular(L, cross, lower=True, check_finite=False)
    variance = a+b-np.einsum('ij,ij->j', solved, solved)
    if np.min(variance) < -1e-8*max(a+b, 1.):
        raise ArithmeticError('Negative conditional variance beyond roundoff')
    variance = np.maximum(variance, 0.)
    return dict(mean=cross.T@alpha, f_variance=variance, y_variance=variance+noise,
                noise_variance=noise)


def predict_mixture(y, x, xp, fit, *, output_mean=0., output_scale=1., credible_mass=.95,
                    verbose=True, interval_batch_size=64):
    """Use ALL retained draws; return Gaussian mixture in original output units."""
    y, x = inputs(y, x)
    xp = np.asarray(xp, dtype=float)
    if xp.ndim == 1:
        xp = xp[:, None]
    if not np.isfinite(output_scale) or output_scale <= 0 or not np.isfinite(output_mean):
        raise ValueError('Invalid output transformation')
    if not isinstance(interval_batch_size, (int, np.integer)) or interval_batch_size < 1:
        raise ValueError('interval_batch_size must be a positive integer')
    samples = fit['log_samples']
    flat = samples.reshape(-1, samples.shape[-1])
    count, m = len(flat), len(xp)
    means, fvar, yvar = (np.empty((count, m)) for _ in range(3))
    d = x.shape[1]
    distances = (squared_distances(x, x).reshape(d, -1),
                 squared_distances(x, xp).reshape(d, -1))
    for index, q in enumerate(flat):
        p = conditional_marginals(y, x, xp, q, model=fit['model'], distances=distances)
        means[index] = output_mean+output_scale*p['mean']
        fvar[index] = output_scale**2*p['f_variance']
        yvar[index] = output_scale**2*p['y_variance']
        if verbose and (index+1) % 500 == 0:
            print(f'Prediction {index+1}/{count}', flush=True)
    weights = np.full(count, 1./count)
    mean = means.mean(axis=0)
    between = np.mean((means-mean)**2, axis=0)
    result = dict(weights=weights, component_means=means, component_f_variances=fvar,
                  component_y_variances=yvar, draws_used=count, credible_mass=credible_mass)
    for name, variances in [('f', fvar), ('y', yvar)]:
        within = variances.mean(axis=0)
        # The same pointwise CDF inversion, batched to bound temporary memory.
        intervals = np.concatenate([
            _mixture_intervals(means[:, start:start+interval_batch_size],
                               variances[:, start:start+interval_batch_size], weights, credible_mass)
            for start in range(0, m, interval_batch_size)], axis=0)
        result.update({f'{name}_mean': mean.copy(), f'{name}_var': within+between,
                       f'{name}_within_var': within, f'{name}_between_var': between,
                       f'{name}_ci': intervals})
    return result
