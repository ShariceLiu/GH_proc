"""HMC for SE/noise hyperparameters, with beta, f and W marginalized out.

Uses NumPy/SciPy only. This is ordinary Metropolis-corrected HMC with random
trajectory lengths, not NUTS. Sampling coordinates are LOG positive parameters.
Independent Gaussian priors on these coordinates need no extra Jacobian.

The target is p(log theta | y) proportional to p(log theta) times
integral N(y; mu, W*(K_theta + k*I) + W**2*Sigma_theta) p(W) dW.
Thus W is integrated by quadrature, while beta/f are integrated analytically;
none of these latent quantities enlarge the HMC state dimension.
"""

import warnings
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy import special, stats

from GH_proc_w_betaprior import (
    _inputs, _mixture_intervals, beta_f_inf, log_marginal_likelihood,
)
from GH_proc_w_betaprior_fit import _pack, _unpack, fit_kernel_parameters


def _leapfrog(target, q, p, gradient, step_size, steps, inverse_mass):
    """Reversible leapfrog integration; target returns (log density, gradient)."""
    q, p = q.copy(), p.copy()
    p += 0.5 * step_size * gradient
    for step in range(steps):
        q += step_size * inverse_mass * p
        logp, gradient = target(q)
        if not np.isfinite(logp) or not np.all(np.isfinite(gradient)):
            return q, p, -np.inf, gradient
        p += (0.5 if step == steps - 1 else 1.) * step_size * gradient
    return q, p, logp, gradient


def _ess(values):
    """Multichain ESS with Geyer's initial positive, monotone paired sequence."""
    chains, draws = values.shape
    centered = values - values.mean(axis=1, keepdims=True)
    n_fft = 1 << (2 * draws - 1).bit_length()
    spectrum = np.fft.rfft(centered, n=n_fft, axis=1)
    autocov = np.fft.irfft(spectrum * spectrum.conj(), n=n_fft, axis=1)[:, :draws] / draws
    within = np.mean(np.var(values, axis=1, ddof=1))
    variance = (draws - 1) / draws * within + np.var(values.mean(axis=1), ddof=1)
    if within <= 0 or variance <= 0:
        return float('nan')
    rho = 1 - (within - autocov.mean(axis=0)) / variance
    rho[0] = 1.
    paired = rho[:2*(draws//2)].reshape(-1, 2).sum(axis=1)
    nonpositive = np.flatnonzero(paired <= 0)
    paired = paired[:nonpositive[0]] if nonpositive.size else paired
    tau = max(1., -1 + 2 * np.minimum.accumulate(paired).sum())
    return float(chains * draws / tau)


def chain_diagnostics(samples):
    """Rank-normalized split/folded R-hat, bulk ESS and 5%/95% tail ESS.

    Input shape (chains, draws, parameters); warm-up must already be removed.
    R-hat near 1 and adequate ESS are checks, not proof of exploring all modes.
    At least two chains and eight retained draws per chain are required.
    """
    samples = np.asarray(samples, dtype=float)
    if (samples.ndim != 3 or samples.shape[0] < 2 or samples.shape[1] < 8
            or not np.all(np.isfinite(samples))):
        raise ValueError('Diagnostics need finite samples with >=2 chains and >=8 draws')
    half = samples.shape[1] // 2
    split = np.concatenate((samples[:, :half], samples[:, -half:]), axis=0)
    rhat, bulk, tail = [], [], []

    def rank_normalize(x):
        ranks = stats.rankdata(x.ravel()).reshape(x.shape)
        return special.ndtri((ranks - 0.375) / (x.size + 0.25))

    def r_hat(x):
        within = np.mean(np.var(x, axis=1, ddof=1))
        if within <= 0:
            return float('inf')
        variance = (half - 1) / half * within + np.var(x.mean(axis=1), ddof=1)
        return float(np.sqrt(variance / within))

    for d in range(samples.shape[2]):
        values = split[:, :, d]
        ranked = rank_normalize(values)
        folded = rank_normalize(np.abs(values - np.median(values)))
        rhat.append(max(r_hat(ranked), r_hat(folded)))
        bulk.append(_ess(ranked))
        lower, upper = np.quantile(values, [0.05, 0.95])
        tail.append(min(_ess((values <= lower).astype(float)),
                        _ess((values <= upper).astype(float))))
    return dict(rhat=np.array(rhat), ess_bulk=np.array(bulk), ess_tail=np.array(tail))


def sample_hmc(target, starts, *, draws=800, warmup=500, seed=0,
               step_size=0.1, leapfrog_steps=(12, 30), target_accept=0.85,
               divergence_threshold=100., verbose=False, swap_pairs=None,
               mode_transforms=None, chain_workers=1):
    """Sample a differentiable log density using diagonal-metric HMC.

    Dual averaging adapts the step size during warm-up. Halfway through warm-up,
    the diagonal inverse mass is set to a regularized coordinate variance and
    dual averaging restarts. Both are frozen before retaining samples. Random
    trajectory lengths and +/-10% step jitter help avoid periodic trajectories.
    Rejections ARE retained; no thinning is performed. Failed trajectories are
    rejected and counted as divergences. Energy error is H(proposed)-H(current).
    Optional disjoint swap_pairs propose a coordinate permutation after each
    HMC step, accepted by the full target density ratio. This involution has
    unit Jacobian and preserves the same posterior; it can mix kernel labels.
    Optional (permutation, shift) mode_transforms propose P*q+b or its inverse
    with equal probability. These volume-preserving, inverse-paired moves get
    a full Metropolis correction. Independent chains may run on worker threads.
    """
    starts = np.asarray(starts, dtype=float)
    if starts.ndim != 2 or not starts.size or not np.all(np.isfinite(starts)):
        raise ValueError('starts must be a finite (chains, parameters) array')
    for name, count, minimum in [('draws', draws, 8), ('warmup', warmup, 50)]:
        if not isinstance(count, (int, np.integer)) or count < minimum:
            raise ValueError(f'{name} must be an integer >= {minimum}')
    if starts.shape[0] < 2:
        raise ValueError('Use at least two independent chains')
    if (len(leapfrog_steps) != 2 or any(not isinstance(s, (int, np.integer)) for s in leapfrog_steps)
            or not 1 <= leapfrog_steps[0] <= leapfrog_steps[1]):
        raise ValueError('leapfrog_steps must contain two increasing positive integers')
    if not np.isfinite(step_size) or step_size <= 0 or not 0 < target_accept < 1:
        raise ValueError('step_size must be positive and 0 < target_accept < 1')
    if not np.isfinite(divergence_threshold) or divergence_threshold <= 0:
        raise ValueError('divergence_threshold must be positive')
    chains, dimension = starts.shape
    if not isinstance(chain_workers, (int, np.integer)) or chain_workers < 1:
        raise ValueError('chain_workers must be a positive integer')
    transforms = []
    for perm, shift in ([] if mode_transforms is None else mode_transforms):
        perm, shift = np.asarray(perm, dtype=int), np.asarray(shift, dtype=float)
        if (perm.shape != (dimension,) or not np.array_equal(np.sort(perm), np.arange(dimension))
                or shift.shape != (dimension,) or not np.isfinite(shift).all()):
            raise ValueError('Invalid affine mode proposal')
        transforms.append((perm, np.argsort(perm), shift))
    permutation = None
    if swap_pairs is not None:
        pairs = np.asarray(swap_pairs)
        if (pairs.ndim != 2 or pairs.shape[1] != 2 or not pairs.size
                or not np.issubdtype(pairs.dtype, np.integer)
                or np.any(pairs < 0) or np.any(pairs >= dimension)
                or np.unique(pairs).size != pairs.size):
            raise ValueError('swap_pairs must contain disjoint valid coordinate pairs')
        permutation = np.arange(dimension)
        for left, right in pairs:
            permutation[left], permutation[right] = right, left
    total = warmup + draws
    trace = np.empty((chains, total, dimension))
    log_density = np.empty((chains, total))
    accepted = np.zeros((chains, total), dtype=bool)
    accept_prob = np.empty((chains, total))
    energy_error = np.empty((chains, total))
    divergent = np.zeros((chains, total), dtype=bool)
    swap_accepted = np.zeros((chains, total), dtype=bool)
    mode_jump_accepted = np.zeros((chains, total), dtype=bool)
    mode_jump_proposed = np.zeros((chains, total), dtype=bool)
    step_sizes, inverse_masses = np.empty(chains), np.empty((chains, dimension))

    def run_chain(pair):
        chain, child_seed = pair
        rng = np.random.default_rng(child_seed)
        q = starts[chain].copy()
        logp, gradient = target(q)
        if not np.isfinite(logp) or not np.all(np.isfinite(gradient)):
            raise ValueError(f'Chain {chain + 1} starts outside the finite target density')
        inverse_mass = np.ones(dimension)
        epsilon = step_size
        adapt_count, h_bar = 0, 0.
        log_eps_bar, adapt_mu = np.log(epsilon), np.log(10 * epsilon)
        for iteration in range(total):
            if iteration == warmup // 2:
                # inverse_mass approximates Cov(q), so momentum has Cov(q)^-1.
                window = trace[chain, warmup//5:iteration]
                # Weak, sample-size-dependent regularization matters when many
                # curves identify length scales far more tightly than amplitudes.
                # A fixed +0.1 floor would erase that scale information.
                shrinkage = 5./(len(window)+5.)
                inverse_mass = np.clip((1-shrinkage)*np.var(window, axis=0, ddof=1)
                                       +shrinkage*1e-3, 1e-8, 25.)
                epsilon = step_size
                adapt_count, h_bar = 0, 0.
                log_eps_bar, adapt_mu = np.log(epsilon), np.log(10 * epsilon)
            if iteration == warmup:
                epsilon = float(np.exp(log_eps_bar))
            p = rng.normal(size=dimension) / np.sqrt(inverse_mass)
            steps = int(rng.integers(leapfrog_steps[0], leapfrog_steps[1] + 1))
            proposal = _leapfrog(target, q, p, gradient, epsilon*rng.uniform(0.9, 1.1),
                                 steps, inverse_mass)
            q_new, p_new, logp_new, grad_new = proposal
            with np.errstate(over='ignore', invalid='ignore'):
                delta = logp - logp_new + 0.5*np.sum(inverse_mass*(p_new**2 - p**2))
            if not np.isfinite(delta):
                delta = np.inf
            probability = float(np.exp(min(0., -delta)))
            if rng.random() < probability:
                q, logp, gradient = q_new, logp_new, grad_new
                accepted[chain, iteration] = True
            if permutation is not None:
                swapped = q[permutation]
                swap_logp, swap_gradient = target(swapped)
                if (np.isfinite(swap_logp) and np.all(np.isfinite(swap_gradient))
                        and np.log(rng.random()) < min(0., swap_logp-logp)):
                    q, logp, gradient = swapped, swap_logp, swap_gradient
                    swap_accepted[chain, iteration] = True
            if transforms and rng.random() < .2:
                perm, inverse, shift = transforms[int(rng.integers(len(transforms)))]
                jumped = q[perm]+shift if rng.random() < .5 else (q-shift)[inverse]
                jump_logp, jump_gradient = target(jumped)
                mode_jump_proposed[chain, iteration] = True
                if (np.isfinite(jump_logp) and np.all(np.isfinite(jump_gradient))
                        and np.log(rng.random()) < min(0., jump_logp-logp)):
                    q, logp, gradient = jumped, jump_logp, jump_gradient
                    mode_jump_accepted[chain, iteration] = True
            trace[chain, iteration] = q
            log_density[chain, iteration] = logp
            accept_prob[chain, iteration] = probability
            energy_error[chain, iteration] = delta
            divergent[chain, iteration] = abs(delta) > divergence_threshold
            if iteration < warmup:
                adapt_count += 1
                eta = 1 / (adapt_count + 10)
                h_bar = (1-eta)*h_bar + eta*(target_accept-probability)
                log_eps = np.clip(adapt_mu - np.sqrt(adapt_count)/0.05*h_bar, -9., 1.)
                weight = adapt_count**(-0.75)
                log_eps_bar = weight*log_eps + (1-weight)*log_eps_bar
                epsilon = float(np.exp(log_eps))
            if verbose and ((iteration+1) % 100 == 0 or iteration+1 == total):
                stage = 'warmup' if iteration < warmup else 'sampling'
                print(f'Chain {chain+1}/{chains}, {stage} {iteration+1}/{total}, '
                      f'step={epsilon:.3g}, recent acceptance='
                      f'{accepted[chain, max(0, iteration-99):iteration+1].mean():.2f}', flush=True)
        step_sizes[chain] = epsilon
        inverse_masses[chain] = inverse_mass

    chain_seeds = list(enumerate(np.random.SeedSequence(seed).spawn(chains)))
    if chain_workers == 1:
        for pair in chain_seeds:
            run_chain(pair)
    else:
        with ThreadPoolExecutor(max_workers=min(chains, chain_workers)) as pool:
            futures = [pool.submit(run_chain, pair) for pair in chain_seeds]
            for future in futures:
                future.result()
    result = dict(samples=trace[:, warmup:], warmup_samples=trace[:, :warmup],
                  log_density=log_density[:, warmup:], acceptance=accepted[:, warmup:],
                  acceptance_probability=accept_prob[:, warmup:], energy_error=energy_error[:, warmup:],
                  divergent=divergent[:, warmup:], warmup_divergent=divergent[:, :warmup],
                  swap_accepted=swap_accepted[:, warmup:],
                  mode_jump_accepted=mode_jump_accepted[:, warmup:],
                  mode_jump_proposed=mode_jump_proposed[:, warmup:],
                  warmup_mode_jump_accepted=mode_jump_accepted[:, :warmup],
                  mass_adaptation_version=np.full(chains, 2, dtype=int),
                  step_size=np.array(step_sizes), inverse_mass=np.array(inverse_masses))
    result['diagnostics'] = chain_diagnostics(result['samples'])
    return result


def _kernel_target(ys, xs, dimension, l, omega, mu, center, sd, bounds, quadrature):
    def target(vector):
        if np.any(vector < bounds[0]) or np.any(vector > bounds[1]):
            return -np.inf, np.zeros_like(vector)
        parameters = _unpack(vector, dimension, l, omega, mu, False)
        try:
            likelihood, gradient = log_marginal_likelihood(
                ys, xs, **parameters, return_gradient=True, **quadrature)
        except (np.linalg.LinAlgError, FloatingPointError, OverflowError, RuntimeError):
            return -np.inf, np.zeros_like(vector)
        derivative = np.r_[gradient['log_v_0'], gradient['log_sigma_v0'], gradient['log_k'],
                           gradient['log_w'], gradient['log_sigma_w']]
        standardized = (vector - center) / sd
        return likelihood - 0.5*standardized @ standardized, derivative - standardized/sd
    return target


def sample_kernel_parameters(ys, xs, *, l=-0.5, omega=1, mu=0,
                             prior_centers=None, log_prior_sd=1.5,
                             initial=None, chains=4, draws=800, warmup=500,
                             seed=0, n_restarts=3, log_bounds=(-10., 10.),
                             quad_options=None, step_size=0.1, leapfrog_steps=(12, 30),
                             target_accept=0.85, verbose=False):
    """Infer v_0, sigma_v0, k, w[D], sigma_w[D], fixing l, omega and mu.

    p(log theta | y) is proportional to the W-integrated likelihood times
    independent Normal priors on log theta, truncated to log_bounds. Priors
    default to the empirical training-scale centers in fit_kernel_parameters;
    supply prior_centers for externally chosen priors. No true simulation
    parameters are used for initialization. MAP initializes dispersed chains.

    Returns log_samples and parameter_samples with shape (chains,draws,3+2D),
    parameter_names, summaries, sampler diagnostics, MAP fit and prior settings.
    w/sigma_w are inverse squared length scales: length = 1/sqrt(weight).
    Metropolis correction removes leapfrog discretization bias. The target
    still has numerical W quadrature error, and finite chains have Monte
    Carlo error; check diagnostics before using the summaries.
    """
    if not isinstance(chains, (int, np.integer)) or chains < 2:
        raise ValueError('chains must be an integer >= 2')
    if log_prior_sd is None:
        raise ValueError('HMC requires a proper log-parameter prior')
    xs = _inputs(xs, 'xs')
    dimension = 1 if xs.ndim == 1 else xs.shape[1]
    fit = fit_kernel_parameters(
        ys, xs, l=l, omega=omega, mu=mu, initial=initial, prior_centers=prior_centers,
        log_prior_sd=log_prior_sd, n_restarts=n_restarts, log_bounds=log_bounds,
        quad_options=quad_options, seed=seed, verbose=verbose)
    center, sd = fit['prior']['vector_center'], fit['prior']['vector_sd']
    target = _kernel_target(ys, xs, dimension, l, omega, mu, center, sd,
                            log_bounds, fit['quad_options'])
    mode = _pack(fit['params'], dimension, False)
    rng = np.random.default_rng(seed)
    starts = np.empty((chains, len(mode)))
    for chain in range(chains):
        for attempt in range(1000):
            start = mode + rng.normal(0, 0.7, size=len(mode))
            if np.isfinite(target(start)[0]):
                starts[chain] = start
                break
        else:
            raise RuntimeError('Could not find finite dispersed chain starts')
    result = sample_hmc(target, starts, draws=draws, warmup=warmup, seed=seed,
                        step_size=step_size, leapfrog_steps=leapfrog_steps,
                        target_accept=target_accept, verbose=verbose)
    result['log_samples'] = result.pop('samples')
    result['parameter_samples'] = np.exp(result['log_samples'])
    names = ['v_0', 'sigma_v0', 'k'] + [f'w[{d}]' for d in range(dimension)]
    names += [f'sigma_w[{d}]' for d in range(dimension)]
    result.update(parameter_names=names, map_fit=fit, prior=fit['prior'],
                  fixed_params=dict(l=l, omega=omega, mu=mu), dimension=dimension,
                  log_bounds=np.asarray(log_bounds), starts=starts)
    flat = result['parameter_samples'].reshape(-1, len(mode))
    quantiles = np.quantile(flat, [0.025, 0.5, 0.975], axis=0)
    result['summary'] = [dict(parameter=name, mean=float(flat[:, d].mean()),
                              median=float(quantiles[1, d]), lower=float(quantiles[0, d]),
                              upper=float(quantiles[2, d]),
                              **{key: float(value[d]) for key, value in result['diagnostics'].items()})
                         for d, name in enumerate(names)]
    if (np.any(result['diagnostics']['rhat'] > 1.01) or np.any(result['divergent'])
            or np.any(result['diagnostics']['ess_bulk'] < 100)
            or np.any(result['diagnostics']['ess_tail'] < 100)):
        warnings.warn('Inspect HMC diagnostics: R-hat > 1.01, low ESS or divergences. '
                      'Posterior summaries may need longer/better-tuned chains.', RuntimeWarning)
    return result


def predict_hyperparameter_mixture(ys, xs, x_pred, fit, *, max_draws=64,
                                   credible_mass=0.95):
    """Pointwise beta/f/y means and intervals integrating theta as well as W.

    Uses an equal, evenly spaced number of retained draws from every chain.
    max_draws is rounded down to a multiple of the chain count. Increase it
    to check Monte Carlo stability. Full joint covariances are not retained.
    """
    if not isinstance(max_draws, (int, np.integer)) or max_draws < fit['log_samples'].shape[0]:
        raise ValueError('max_draws must be an integer at least the chain count')
    chains, draws, _ = fit['log_samples'].shape
    indices = np.linspace(0, draws-1, min(draws, max_draws//chains), dtype=int)
    vectors = fit['log_samples'][:, indices].reshape(-1, fit['log_samples'].shape[-1])
    means, variances, noisy_variances, weights = [], [], [], []
    for vector in vectors:
        parameters = _unpack(vector, fit['dimension'], **fit['fixed_params'], fit_mu=False)
        posterior = beta_f_inf(ys, xs, x_pred, **parameters,
                               **fit['map_fit']['quad_options'], credible_mass=credible_mass)
        means.append(posterior['component_means'])
        variances.append(np.diagonal(posterior['component_covariances'], axis1=1, axis2=2).copy())
        noisy_variances.append(posterior['component_y_variances'])
        weights.append(posterior['weights']/len(vectors))
    means, variances = np.concatenate(means), np.maximum(np.concatenate(variances), 0)
    weights = np.concatenate(weights)
    weights /= weights.sum()
    noisy_variances = np.concatenate(noisy_variances)
    m = means.shape[1] // 2
    result = dict(component_means=means, component_y_variances=noisy_variances,
                  weights=weights, hyperparameter_draws=len(vectors), credible_mass=credible_mass)
    for name, section in [('beta', slice(0, m)), ('f', slice(m, 2*m)), ('y', slice(m, 2*m))]:
        component_mean = means[:, section]
        component_var = noisy_variances if name == 'y' else variances[:, section]
        mean = weights @ component_mean
        var = weights @ (component_var + (component_mean-mean)**2)
        result.update({f'{name}_mean': mean, f'{name}_var': var, f'{name}_std': np.sqrt(var),
                       f'{name}_ci': _mixture_intervals(component_mean, component_var, weights, credible_mass)})
    return result
