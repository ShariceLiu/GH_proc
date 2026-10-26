"""Shared-hyperparameter HMC for independent GP/GHP/GP+TP realizations.

Each observation group is (x[N], y[J,N]); groups allow different grids.
The five physical log hyperparameters are shared. Mixing models additionally
have one independent W per function. No latent function values are sampled.
HMC uses amplitudes relative to the geometric mean W to reduce correlations;
this triangular linear transformation of log coordinates has determinant one.
"""

import warnings
from concurrent.futures import ProcessPoolExecutor
from contextlib import redirect_stdout
from pathlib import Path

import numpy as np
from scipy import linalg, optimize, special

from GH_proc_w_betaprior_hmc import sample_hmc, chain_diagnostics


MODELS = ('gp', 'tp_mu_gp', 'mu_gp', 'beta_gp')


def _chain_group_worker(payload):
    """Spawn-safe independent chain group; each process limits BLAS to one thread."""
    from threadpoolctl import threadpool_limits
    groups, target_options, starts, sampler_options, log_path = payload
    with threadpool_limits(1):
        target = make_target(groups, **target_options)
        if log_path is None:
            return sample_hmc(target, starts, **sampler_options)
        path = Path(log_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('w', encoding='utf-8') as stream, redirect_stdout(stream):
            result = sample_hmc(target, starts, **sampler_options)
        np.savez_compressed(path.with_suffix('.npz'),
                            **{key: value for key, value in result.items() if isinstance(value, np.ndarray)})
        return result


def validate_groups(groups):
    result = []
    for x, y in groups:
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        if (x.ndim != 1 or not x.size or y.ndim != 2 or y.shape[1] != len(x)
                or not len(y) or not np.isfinite(x).all() or not np.isfinite(y).all()):
            raise ValueError('Groups must contain finite x[N] and y[J,N] arrays')
        result.append((x, y))
    if not result:
        raise ValueError('At least one observation group is needed')
    return result


def power(model):
    if model not in MODELS:
        raise ValueError('Unknown model')
    return 2 if model == 'beta_gp' else 0


def coordinates(values, model, inverse=False):
    """Physical log theta,W <-> amplitudes based on mean(log W)."""
    q = np.array(values, dtype=float, copy=True)
    r = power(model)
    if model == 'gp':
        if q.shape[-1] != 5:
            raise ValueError('GP requires five log parameters')
    else:
        if q.shape[-1] < 6:
            raise ValueError('Mixing models need at least one W')
        shift = q[..., 5:].mean(axis=-1)*(-1 if inverse else 1)
        q[..., 0] += shift
        q[..., 1] += r*shift
        q[..., 2] += shift
    return q


def dense_grid_likelihood(y, K, S, k, W, r, d2, w, sw):
    """Direct Cholesky fallback for a poorly conditioned shared decomposition."""
    j, n = y.shape
    eye = np.eye(n)
    WK, WS = W[:, None, None]*K, W[:, None, None]**r*S
    noise = W*k
    C = WK+WS+noise[:, None, None]*eye
    L = np.linalg.cholesky(C)
    Linv = np.linalg.solve(L, np.broadcast_to(eye, C.shape))
    inv = np.swapaxes(Linv, -1, -2)@Linv
    alpha = (inv@y[..., None])[..., 0]
    value = (-.5*j*n*np.log(2*np.pi)-np.log(L.diagonal(axis1=-2, axis2=-1)).sum()
             -.5*np.sum(y*alpha))
    score = .5*(alpha[:, :, None]*alpha[:, None, :]-inv)
    g0, g1 = np.sum(score*WK, axis=(1, 2)), np.sum(score*WS, axis=(1, 2))
    g2 = noise*np.trace(score, axis1=-2, axis2=-1)
    grad = np.array([g0.sum(), g1.sum(), g2.sum(),
                     np.sum(score*WK*(-.5*w*d2)), np.sum(score*WS*(-.5*sw*d2))])
    return float(value), grad, g0+r*g1+g2


def shared_grid_likelihood(y, K, S, k, W, r, d2, w, sw):
    """Exact Gaussian likelihood/gradients using one shared spectral basis.

    Whiten by B=K+S+kI. In this basis S has eigenvalues lambda in [0,1]
    and K+kI has eigenvalues 1-lambda, so each function's covariance is
    diagonal with entries W*(1-lambda)+W**r*lambda. This avoids one cubic
    factorization per function; it is not a low-rank kernel approximation.
    """
    j, n = y.shape
    L = linalg.cholesky(K+S+k*np.eye(n), lower=True, check_finite=False)
    # Diagonalize the smaller component to reduce cancellation. Tiny departures
    # of eigenvalues from [0,1] are retained: rounded SE matrices need not be
    # exactly PSD, while their noise-regularized covariance can still be valid.
    complement = K[0, 0]+k < S[0, 0]
    smaller = K+k*np.eye(n) if complement else S
    left = linalg.solve_triangular(L, smaller, lower=True, check_finite=False)
    whitened = linalg.solve_triangular(L, left.T, lower=True, check_finite=False)
    values, vectors = linalg.eigh((whitened+whitened.T)/2, check_finite=False)
    if complement:
        values = 1-values
    P = linalg.solve_triangular(L.T, vectors, lower=False, check_finite=False)
    projected = y@P
    Wr = W**r
    variance = W[:, None]*(1-values)+Wr[:, None]*values
    if np.any(variance <= 0):
        return dense_grid_likelihood(y, K, S, k, W, r, d2, w, sw)
    precision = 1/variance
    quadratic = projected**2*precision
    value = (-.5*j*n*np.log(2*np.pi)-j*np.log(L.diagonal()).sum()
             -.5*np.log(variance).sum()-.5*quadratic.sum())
    alpha = (projected*precision)@P.T

    def score_sum(coefficients):
        diagonal = np.sum(coefficients[:, None]*precision, axis=0)
        return .5*((alpha.T*coefficients)@alpha-(P*diagonal)@P.T)

    score_A, score_S = score_sum(W), score_sum(Wr)
    grad = np.array([np.sum(score_A*K), np.sum(score_S*S), k*np.trace(score_A),
                     np.sum(score_A*K*(-.5*w*d2)), np.sum(score_S*S*(-.5*sw*d2))])
    derivative = W[:, None]*(1-values)+r*Wr[:, None]*values
    grad_t = .5*np.sum((quadratic-1)*precision*derivative, axis=1)
    return float(value), grad, grad_t


def make_target(groups, *, model, prior_center, prior_sd=2., l=1., omega=1., nu=5.,
                backend='dense'):
    """Normalized likelihood and priors, up to evidence, with analytic gradient.

    p(theta,{W_j}|data) = p(theta) product_j p(W_j) N(y_j;0,C_j),
    C_j=W_j*(K+kI)+W_j**r*Sigma. GP uses C=K+Sigma+kI.
    Priors on five log(theta) coordinates are counted ONCE, not per function.
    For each t_j=log W_j the density includes the Jacobian exp(t_j).
    """
    groups = validate_groups(groups)
    if backend not in ('dense', 'shared_grid'):
        raise ValueError('Unknown likelihood backend')
    r = power(model)
    count = sum(len(y) for _, y in groups)
    size = 5 if model == 'gp' else 5+count
    center = np.asarray(prior_center, dtype=float)
    sd = np.broadcast_to(np.asarray(prior_sd, dtype=float), (5,)).copy()
    if center.shape != (5,) or not np.isfinite(center).all() or not np.isfinite(sd).all() or np.any(sd <= 0):
        raise ValueError('Need five finite prior centers and positive prior scales')
    if model == 'tp_mu_gp' and (not np.isfinite(nu) or nu <= 2):
        raise ValueError('nu must exceed two')
    if model in ('beta_gp', 'mu_gp') and (not np.isfinite(l) or not np.isfinite(omega) or omega <= 0):
        raise ValueError('Invalid GIG parameters')
    cached = [(y, (x[:, None]-x[None, :])**2, np.eye(len(x))) for x, y in groups]
    prior_constant = -np.log(sd).sum()-2.5*np.log(2*np.pi)
    mix_constant = (-np.log(2)-np.log(special.kve(l, omega))+omega
                    if model in ('beta_gp', 'mu_gp') else 0.)
    a, b = nu/2, (nu-2)/2
    if model == 'tp_mu_gp':
        mix_constant = a*np.log(b)-special.gammaln(a)

    def target(z):
        z = np.asarray(z, dtype=float)
        if z.shape != (size,) or not np.isfinite(z).all():
            return -np.inf, np.zeros(size)
        try:
            with np.errstate(over='raise', divide='raise', invalid='raise'):
                q = coordinates(z, model, inverse=True)
                v, latent_v, k, w, sw = np.exp(q[:5])
                grad = np.zeros(size)
                value, offset = 0., 0
                for y, d2, eye in cached:
                    j, n = y.shape
                    K = v*np.exp(-.5*w*d2)
                    S = latent_v*np.exp(-.5*sw*d2)
                    if model == 'gp':
                        C = K+S+k*eye
                        L = np.linalg.cholesky(C)
                        Linv = np.linalg.solve(L, eye)
                        inv = Linv.T@Linv
                        alpha = y@inv
                        value += -.5*j*n*np.log(2*np.pi)-j*np.log(L.diagonal()).sum()-.5*np.sum(y*alpha)
                        score = .5*(alpha.T@alpha-j*inv)
                        grad[:5] += [np.sum(score*K), np.sum(score*S), k*np.trace(score),
                                     np.sum(score*K*(-.5*w*d2)), np.sum(score*S*(-.5*sw*d2))]
                    else:
                        W = np.exp(q[5+offset:5+offset+j])
                        if backend == 'shared_grid':
                            local_value, local_grad, local_t = shared_grid_likelihood(
                                y, K, S, k, W, r, d2, w, sw)
                            value += local_value
                            grad[:5] += local_grad
                            grad[5+offset:5+offset+j] = local_t
                            offset += j
                            continue
                        WK = W[:, None, None]*K
                        WS = W[:, None, None]**r*S
                        noise = W*k
                        C = WK+WS+noise[:, None, None]*eye
                        L = np.linalg.cholesky(C)
                        Linv = np.linalg.solve(L, np.broadcast_to(eye, C.shape))
                        inv = np.swapaxes(Linv, -1, -2)@Linv
                        alpha = (inv@y[..., None])[..., 0]
                        value += -.5*j*n*np.log(2*np.pi)-np.log(L.diagonal(axis1=-2, axis2=-1)).sum()-.5*np.sum(y*alpha)
                        score = .5*(alpha[:, :, None]*alpha[:, None, :]-inv)
                        g0 = np.sum(score*WK, axis=(1, 2))
                        g1 = np.sum(score*WS, axis=(1, 2))
                        g2 = noise*np.trace(score, axis1=-2, axis2=-1)
                        grad[:5] += [g0.sum(), g1.sum(), g2.sum(),
                                     np.sum(score*WK*(-.5*w*d2)), np.sum(score*WS*(-.5*sw*d2))]
                        grad[5+offset:5+offset+j] = g0+r*g1+g2
                    offset += j
                standardized = (q[:5]-center)/sd
                value += prior_constant-.5*standardized@standardized
                grad[:5] -= standardized/sd
                if model != 'gp':
                    t = q[5:]
                    W = np.exp(t)
                    if model == 'tp_mu_gp':
                        value += np.sum(mix_constant-a*t-b/W)
                        grad[5:] += -a+b/W
                    else:
                        value += np.sum(mix_constant+l*t-.5*omega*(W+1/W))
                        grad[5:] += l-.5*omega*(W-1/W)
                    grad[5:] -= (grad[0]+r*grad[1]+grad[2])/count
                if np.isfinite(value) and np.isfinite(grad).all():
                    return float(value), grad
        except (np.linalg.LinAlgError, FloatingPointError, OverflowError):
            pass
        return -np.inf, np.zeros(size)
    return target


def sample_parameters(groups, *, model, prior_center=None, prior_sd=2., l=1., omega=1., nu=5.,
                      chains=4, warmup=1500, draws=2000, seed=7026, target_accept=.95,
                      leapfrog_steps=(15, 35), verbose=True, backend='dense', chain_workers=1,
                      chain_processes=1, chain_log_dir=None):
    """Sample shared hyperparameters and all function-specific scales jointly."""
    groups = validate_groups(groups)
    count = sum(len(y) for _, y in groups)
    scale = max(float(np.var(np.concatenate([y.ravel() for _, y in groups]))), 1e-12)
    center = np.log([.5*scale, .5*scale, .1*scale, 1., .25]) if prior_center is None else np.asarray(prior_center)
    sd = np.broadcast_to(np.asarray(prior_sd, dtype=float), (5,)).copy()
    target = make_target(groups, model=model, prior_center=center, prior_sd=sd,
                         l=l, omega=omega, nu=nu, backend=backend)
    rng = np.random.default_rng(seed)
    dimension = 5 if model == 'gp' else 5+count
    initial_t = np.log((nu-2)/nu) if model == 'tp_mu_gp' else np.arcsinh(l/omega)
    modes, initialization = [], []
    for index in range(chains):
        guess = coordinates(np.r_[center, np.full(count, initial_t)] if model != 'gp' else center, model)
        guess += rng.normal(0., .5, dimension)
        if index % 2:
            guess[[0, 1, 3, 4]] = guess[[1, 0, 4, 3]]
        def objective(z):
            value, grad = target(z)
            return -value, -grad
        result = optimize.minimize(objective, guess, jac=True, method='L-BFGS-B',
                                   options=dict(maxiter=600, ftol=1e-10, maxls=50))
        mode = result.x if np.isfinite(target(result.x)[0]) else guess
        refinements = 0
        if backend == 'shared_grid':
            # Restrict each optimization move while finding a usable start.
            # These temporary local boxes are NOT bounds on the HMC posterior.
            for refinements in range(12):
                value, gradient = target(mode)
                if np.max(np.abs(gradient)) < 1.:
                    break
                local = optimize.minimize(objective, mode, jac=True, method='L-BFGS-B',
                                           bounds=list(zip(mode-1., mode+1.)),
                                           options=dict(maxiter=300, ftol=1e-11, maxls=50))
                if np.isfinite(target(local.x)[0]) and target(local.x)[0] > value:
                    mode = local.x
                else:
                    break
        modes.append(mode)
        initialization.append(dict(success=bool(result.success), log_density=target(mode)[0],
                                    max_abs_gradient=float(np.max(np.abs(target(mode)[1]))),
                                    local_refinements=refinements,
                                    message=str(result.message)))
    if backend == 'shared_grid':
        # A line search can report convergence after an invalid trial without
        # finding a stationary point. Do not start HMC far down that steep slope.
        best = int(np.argmax([item['log_density'] for item in initialization]))
        for index, item in enumerate(initialization):
            if (item['max_abs_gradient'] > 10.
                    and item['log_density'] < initialization[best]['log_density']-50.):
                modes[index] = modes[best].copy()
                item['replaced_by_better_initial_mode'] = best
        if verbose:
            print('Initialization:', initialization, flush=True)
    starts = np.stack(modes)+rng.normal(0., .05 if backend == 'shared_grid' else .15,
                                       (chains, dimension))
    mode_transforms = []
    if backend == 'shared_grid' and model != 'gp':
        assignments = np.array([mode[3] > mode[4] for mode in modes])
        if assignments.any() and (~assignments).any():
            indices = [np.flatnonzero(assignments == label) for label in [False, True]]
            chosen = [max(ids, key=lambda i: target(modes[i])[0]) for ids in indices]
            perm = np.arange(dimension)
            perm[[0, 1, 3, 4]] = perm[[1, 0, 4, 3]]
            shift = modes[chosen[1]]-modes[chosen[0]][perm]
            mode_transforms.append((perm, shift))
    sampler_options = dict(warmup=warmup, draws=draws, seed=seed, step_size=.03,
                           target_accept=target_accept, leapfrog_steps=leapfrog_steps,
                           swap_pairs=[(0, 1), (3, 4)], verbose=verbose,
                           mode_transforms=mode_transforms, chain_workers=chain_workers)
    if chain_processes == 1:
        fit = sample_hmc(target, starts, **sampler_options)
    else:
        if not isinstance(chain_processes, int) or not 1 < chain_processes <= chains//2:
            raise ValueError('Each independent chain process needs at least two chains')
        payloads = []
        for group, indices in enumerate(np.array_split(np.arange(chains), chain_processes)):
            options = dict(sampler_options, seed=seed+100000*group, chain_workers=1)
            log_path = None if chain_log_dir is None else str(Path(chain_log_dir)/f'group_{group}.log')
            payloads.append((groups, dict(model=model, prior_center=center, prior_sd=sd,
                            l=l, omega=omega, nu=nu, backend=backend), starts[indices], options, log_path))
        with ProcessPoolExecutor(max_workers=chain_processes) as pool:
            results = list(pool.map(_chain_group_worker, payloads))
        fit = {key: np.concatenate([result[key] for result in results], axis=0)
               for key, value in results[0].items() if isinstance(value, np.ndarray)}
        fit['diagnostics'] = chain_diagnostics(fit['samples'])
    fit['effective_log_samples'] = fit.pop('samples')
    fit['log_samples'] = coordinates(fit['effective_log_samples'], model, inverse=True)
    fit['warmup_log_samples'] = coordinates(fit.pop('warmup_samples'), model, inverse=True)
    fit['effective_diagnostics'] = fit.pop('diagnostics')
    fit['diagnostics'] = chain_diagnostics(fit['log_samples'])
    fit['theta_log_samples'] = fit['log_samples'][..., :5]
    fit['parameter_samples'] = np.exp(fit['log_samples'])
    latent = 'sigma' if model == 'beta_gp' else 'mu'
    names = ['v_0', f'{latent}_v0', 'k', 'w', f'{latent}_w']
    if model != 'gp':
        names += [f'W_{j:02d}' for j in range(count)]
    fit.update(model=model, parameter_names=names, prior_center=center, prior_sd=sd,
               initialization=initialization, l=l, omega=omega, nu=nu, starts=starts,
               likelihood_backend=backend)
    fit['mode_transforms'] = mode_transforms
    flat = fit['parameter_samples'].reshape(-1, dimension)
    quantiles = np.quantile(flat, [.025, .5, .975], axis=0)
    fit['summary'] = [dict(parameter=name, mean=float(flat[:, i].mean()), std=float(flat[:, i].std()),
                           median=float(quantiles[1, i]), lower=float(quantiles[0, i]), upper=float(quantiles[2, i]),
                           **{key: float(values[i]) for key, values in fit['diagnostics'].items()})
                      for i, name in enumerate(names)]
    d = fit['diagnostics']
    fit['diagnostics_ok'] = bool(np.max(d['rhat']) < 1.01 and np.min(d['ess_bulk']) >= 400
                               and np.min(d['ess_tail']) >= 400 and not np.any(fit['divergent']))
    if not fit['diagnostics_ok']:
        warnings.warn(f'{model}: inspect multiple-function sampler diagnostics', RuntimeWarning)
    return fit


def context_log_weights(theta, x, observations, *, model, quadrature_nodes=192,
                       log_w_bounds=(-10., 14.), l=1., omega=1., nu=5., chunk_size=64,
                       backend='dense'):
    """Joint theta,W weights for each NEW function, using only its context.

    theta[S,5] comes from p(theta|training functions). Contexts[J,N] update
    that posterior separately for each new function. W is freshly integrated
    under its prior, not borrowed from a training function. The returned
    log weights have shape (J,S,Q); GP has Q=1. W quadrature includes dt
    weights and dW/dt. All training posterior states have equal initial weight.
    """
    theta = np.asarray(theta, dtype=float)
    if backend not in ('dense', 'shared_grid'):
        raise ValueError('Unknown likelihood backend')
    x, observations = validate_groups([(x, observations)])[0]
    if theta.ndim != 2 or theta.shape[1] != 5 or not len(theta) or not np.isfinite(theta).all():
        raise ValueError('theta must have shape (draws,5)')
    r = power(model)
    if model == 'gp':
        t, prior_weights = np.zeros(1), np.zeros(1)
    else:
        nodes, weights = special.roots_legendre(quadrature_nodes)
        lo, hi = log_w_bounds
        t = lo+(nodes+1)*(hi-lo)/2
        if model == 'tp_mu_gp':
            a, b = nu/2, (nu-2)/2
            logprior = a*np.log(b)-special.gammaln(a)-a*t-b*np.exp(-t)
        else:
            logprior = l*t-.5*omega*(np.exp(t)+np.exp(-t))-np.log(2)-np.log(special.kve(l, omega))+omega
        prior_weights = logprior+np.log(weights*(hi-lo)/2)
    W = np.exp(t)
    d2, eye = (x[:, None]-x[None, :])**2, np.eye(len(x))
    result = np.empty((len(observations), len(theta), len(t)))
    for start in range(0, len(theta), chunk_size):
        stop = min(start+chunk_size, len(theta))
        v, latent_v, k, w, sw = np.exp(theta[start:stop]).T
        K = v[:, None, None]*np.exp(-.5*w[:, None, None]*d2)
        S = latent_v[:, None, None]*np.exp(-.5*sw[:, None, None]*d2)
        if backend == 'shared_grid' and model != 'gp':
            L = np.linalg.cholesky(K+S+k[:, None, None]*eye)
            complement = v+k < latent_v
            smaller = np.where(complement[:, None, None], K+k[:, None, None]*eye, S)
            left = np.linalg.solve(L, smaller)
            white = np.linalg.solve(L, np.swapaxes(left, -1, -2))
            eigenvalues, vectors = np.linalg.eigh((white+np.swapaxes(white, -1, -2))/2)
            eigenvalues = np.where(complement[:, None], 1-eigenvalues, eigenvalues)
            variance = (W[None, :, None]*(1-eigenvalues[:, None])
                        +W[None, :, None]**r*eigenvalues[:, None])
            if np.any(variance <= 0):
                _, direct = context_log_weights(
                    theta[start:stop], x, observations, model=model, quadrature_nodes=quadrature_nodes,
                    log_w_bounds=log_w_bounds, l=l, omega=omega, nu=nu, chunk_size=chunk_size,
                    backend='dense')
                result[:, start:stop, :] = direct+np.log(stop-start)-np.log(len(theta))
                continue
            P = np.linalg.solve(np.swapaxes(L, -1, -2), vectors)
            projected = np.einsum('jn,snk->skj', observations, P)
            loglike = (-.5*len(x)*np.log(2*np.pi)
                       -np.log(L.diagonal(axis1=-2, axis2=-1)).sum(axis=-1)[:, None, None]
                       -.5*np.log(variance).sum(axis=-1)[..., None]
                       -.5*np.einsum('sqn,snj->sqj', 1/variance, projected**2))
            result[:, start:stop, :] = (np.moveaxis(loglike, -1, 0)
                                        +prior_weights[None, None, :]-np.log(len(theta)))
            continue
        C = W[None, :, None, None]*(K[:, None]+k[:, None, None, None]*eye)
        C = C+W[None, :, None, None]**r*S[:, None]
        L = np.linalg.cholesky(C)
        rhs = np.broadcast_to(observations.T, (*C.shape[:-2], len(x), len(observations)))
        whitened = np.linalg.solve(L, rhs)
        loglike = (-.5*len(x)*np.log(2*np.pi)-np.log(L.diagonal(axis1=-2, axis2=-1)).sum(axis=-1)[..., None]
                   -.5*np.sum(whitened**2, axis=-2))
        result[:, start:stop, :] = np.moveaxis(loglike, -1, 0)+prior_weights[None, None, :]-np.log(len(theta))
    return t, result


def resample_context(theta, log_weights, log_w_nodes, *, model, draws=6000, seed=0):
    """Stratified resampling with INDEPENDENT jitter in every CDF stratum.

    One shared random offset can give poor W coverage when neighboring theta
    draws have similar W distributions, so systematic resampling is not used.
    """
    log_weights = np.asarray(log_weights)
    norm = special.logsumexp(log_weights)
    weights = np.exp(log_weights-norm)
    theta_weights = weights.sum(axis=1)
    cdf = np.cumsum(weights.ravel())
    cdf[-1] = 1.
    rng = np.random.default_rng(seed)
    indices = np.searchsorted(cdf, (np.arange(draws)+rng.random(draws))/draws)
    ti, wi = np.divmod(indices, weights.shape[1])
    q = theta[ti].copy()
    if model != 'gp':
        q = np.column_stack([q, log_w_nodes[wi]])
    diagnostics = dict(log_context_evidence=float(norm),
                       theta_importance_ess=float(1/np.sum(theta_weights**2)),
                       max_theta_weight=float(theta_weights.max()),
                       edge_mass=float(weights[:, :2].sum()+weights[:, -2:].sum()) if model != 'gp' else 0.)
    return q, diagnostics
