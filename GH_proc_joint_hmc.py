"""Joint HMC of theta and W for the 1D beta-GP and independent mu-GP models.

Latent Gaussian processes are integrated analytically, NOT optimized. Positive
parameters are [v_0, latent_v0, k, w, latent_w, W]; w denotes inverse length^2.
The prior is independent Normal on log(theta), and GIG(lambda, omega) on W.
The tp_mu_gp variant uses W~InvGamma(nu/2,(nu-2)/2) instead: an independent
mu~GP(0,K2) plus a Student-t residual, f|W~GP(0,K2+W*K1).
Sampling uses effective log amplitudes [log(W*v_0), log(W**power*latent_v0),
log(W*k), log(w), log(latent_w), log(W)], with power=2 (beta) or 0 (mu).
This linear transformation of log coordinates has determinant one. It reduces
likelihood correlations without changing the original independent priors.
The log-W density includes the dW/dlogW Jacobian. No quadrature is used.
Metropolis-corrected swaps of the two effective SE components allow exploration
of both kernel assignments; no length-scale ordering constraint is imposed.
"""

import warnings

import numpy as np
from scipy import linalg, optimize, special

from GH_proc_w_betaprior import _mixture_intervals
from GH_proc_w_betaprior_hmc import chain_diagnostics, sample_hmc


def _power(model):
    if model not in ('beta_gp', 'mu_gp', 'tp_mu_gp'):
        raise ValueError('model must be beta_gp, mu_gp or tp_mu_gp')
    return 2 if model == 'beta_gp' else 0


def _inputs_1d(y, x):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.ndim == 2 and x.shape[1] == 1:
        x = x[:, 0]
    if (x.ndim != 1 or y.ndim != 1 or x.size != y.size or not x.size
            or not np.all(np.isfinite(x)) or not np.all(np.isfinite(y))):
        raise ValueError('x and y must be matching nonempty finite 1D arrays')
    return y, x


def effective_coordinates(log_parameters, model):
    z = np.array(log_parameters, dtype=float, copy=True)
    if z.shape[-1] != 6:
        raise ValueError('Expected six log parameters')
    t = z[..., 5].copy()
    z[..., 0] += t
    z[..., 1] += _power(model)*t
    z[..., 2] += t
    return z


def physical_coordinates(effective, model):
    q = np.array(effective, dtype=float, copy=True)
    if q.shape[-1] != 6:
        raise ValueError('Expected six log parameters')
    t = q[..., 5].copy()
    q[..., 0] -= t
    q[..., 1] -= _power(model)*t
    q[..., 2] -= t
    return q


def make_joint_target(y, x, *, model, prior_center, prior_sd=2., l=1., omega=1., nu=5.):
    """Normalized log density and analytic gradient in effective coordinates.

    prior_center is the five-dimensional mean of log(theta), prior_sd its
    scalar or five-dimensional standard deviation. Means of both GPs are zero.
    The returned target integrates beta/mu and f, but samples W explicitly.
    beta_gp/mu_gp use GIG(l,omega); tp_mu_gp uses a fixed inverse-gamma with
    shape nu/2 and scale (nu-2)/2, ignoring l and omega. For the latter,
    log p(log W) = constant - (nu/2)*log W - (nu-2)/(2*W).
    """
    y, x = _inputs_1d(y, x)
    power = _power(model)
    center = np.asarray(prior_center, dtype=float)
    sd = np.broadcast_to(np.asarray(prior_sd, dtype=float), (5,)).copy()
    if center.shape != (5,) or not np.all(np.isfinite(center)):
        raise ValueError('prior_center must contain five finite log means')
    if np.any(sd <= 0) or not np.all(np.isfinite(sd)):
        raise ValueError('prior_sd must be positive and finite')
    if model == 'tp_mu_gp' and (not np.isfinite(nu) or nu <= 2):
        raise ValueError('tp_mu_gp requires finite nu > 2')
    if model != 'tp_mu_gp' and (not np.isfinite(l) or not np.isfinite(omega) or omega <= 0):
        raise ValueError('GIG needs finite lambda and positive finite omega')
    distances = (x[:, None]-x[None, :])**2
    identity = np.eye(len(x))
    normal_constant = -0.5*len(x)*np.log(2*np.pi)
    prior_constant = -np.log(sd).sum()-2.5*np.log(2*np.pi)
    # kve(l, omega) = exp(omega)*K_l(omega); avoids Bessel underflow.
    if model == 'tp_mu_gp':
        ig_shape, ig_scale = nu/2, (nu-2)/2
        mixing_constant = ig_shape*np.log(ig_scale)-special.gammaln(ig_shape)
    else:
        mixing_constant = -np.log(2.)-np.log(special.kve(l, omega))+omega
    if not np.isfinite(mixing_constant):
        raise ValueError('Mixing prior normalization could not be evaluated')

    def target(z):
        z = np.asarray(z, dtype=float)
        if z.shape != (6,) or not np.all(np.isfinite(z)):
            return -np.inf, np.zeros(6)
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                a, b, noise, w, sw, W = np.exp(z)
                residual = a*np.exp(-0.5*w*distances)
                latent = b*np.exp(-0.5*sw*distances)
                covariance = residual + latent + noise*identity
                L = linalg.cholesky(covariance, lower=True, check_finite=False)
                alpha = linalg.cho_solve((L, True), y, check_finite=False)
                loglike = normal_constant-np.log(L.diagonal()).sum()-0.5*y@alpha
                score = 0.5*(np.outer(alpha, alpha)
                             - linalg.cho_solve((L, True), identity, check_finite=False))
                gradient = np.array([
                    np.sum(score*residual), np.sum(score*latent), noise*np.trace(score),
                    np.sum(score*residual*(-0.5*w*distances)),
                    np.sum(score*latent*(-0.5*sw*distances)), 0.])
                q = physical_coordinates(z, model)
                standardized = (q[:5]-center)/sd
                prior_grad = -standardized/sd
                gradient[:5] += prior_grad
                # q amplitudes = z amplitudes - [1, power, 1]*log(W).
                if model == 'tp_mu_gp':
                    # p(W)*dW/dlogW: the Jacobian changes -(shape+1)t to -shape*t.
                    log_mixing_t = mixing_constant-ig_shape*z[5]-ig_scale/W
                    mixing_gradient = -ig_shape+ig_scale/W
                else:
                    log_mixing_t = l*z[5]-0.5*omega*(W+1/W)+mixing_constant
                    mixing_gradient = l-0.5*omega*(W-1/W)
                gradient[5] = (-prior_grad[0]-power*prior_grad[1]-prior_grad[2]
                               +mixing_gradient)
                logp = loglike + prior_constant - 0.5*standardized@standardized + log_mixing_t
                if not np.isfinite(logp) or not np.all(np.isfinite(gradient)):
                    return -np.inf, np.zeros(6)
                return float(logp), gradient
        except (np.linalg.LinAlgError, FloatingPointError, OverflowError):
            return -np.inf, np.zeros(6)

    return target


def sample_joint_parameters(y, x, *, model, l=1., omega=1., prior_center=None,
                            prior_sd=2., chains=4, draws=1500, warmup=1000,
                            seed=2026, target_accept=0.9, leapfrog_steps=(20, 50),
                            verbose=False, nu=5.):
    """Sample the joint posterior; optimization is used ONLY to start chains.

    Default log prior centers match the earlier MAP experiment, using training
    response variance. Thus this is Bayesian inference conditional on empirical
    prior settings; supply external prior_center values to avoid that choice.
    Unlike the MAP optimizer's computational bounds, these Normal priors are
    untruncated. No generating parameters or held-out data enter this function.
    Each HMC iteration is followed by a posterior-corrected component swap.
    Set model='tp_mu_gp' for an independent GP mean plus Student-t residual.
    nu is fixed, not sampled; the other models keep their original GIG prior.
    """
    y, x = _inputs_1d(y, x)
    _power(model)
    if not isinstance(chains, (int, np.integer)) or chains < 2:
        raise ValueError('Use at least two chains')
    scale = float(np.var(y))
    scale = scale if scale > 1e-12 else 1.
    center = np.log([0.5*scale, 0.5*scale, 0.1*scale, 1., 0.25]) if prior_center is None else np.asarray(prior_center, dtype=float)
    sd = np.broadcast_to(np.asarray(prior_sd, dtype=float), (5,)).copy()
    target = make_joint_target(y, x, model=model, prior_center=center, prior_sd=sd, l=l, omega=omega, nu=nu)
    rng = np.random.default_rng(seed)
    modes, initialization = [], []
    for index in range(chains):
        initial_log_w = np.log((nu-2)/nu) if model == 'tp_mu_gp' else np.arcsinh(l/omega)
        guess = effective_coordinates(np.r_[center, initial_log_w], model)
        guess += rng.normal(0., 0.7, 6)
        if index % 2:
            # Explore both possible assignments of the two SE components.
            guess[[0, 1]] = guess[[1, 0]]
            guess[[3, 4]] = guess[[4, 3]]
        def objective(z):
            value, grad = target(z)
            return -value, -grad
        result = optimize.minimize(objective, guess, jac=True, method='L-BFGS-B',
                                   options=dict(maxiter=400, ftol=1e-10))
        mode = result.x if np.isfinite(target(result.x)[0]) else guess
        modes.append(mode)
        initialization.append(dict(success=bool(result.success), log_density=target(mode)[0],
                                   message=str(result.message)))
    starts = np.stack(modes) + rng.normal(0., 0.35, (chains, 6))
    result = sample_hmc(target, starts, draws=draws, warmup=warmup, seed=seed,
                        step_size=0.05, target_accept=target_accept,
                        leapfrog_steps=leapfrog_steps, verbose=verbose,
                        swap_pairs=[(0, 1), (3, 4)])
    result['effective_log_samples'] = result.pop('samples')
    result['log_samples'] = physical_coordinates(result['effective_log_samples'], model)
    result['warmup_log_samples'] = physical_coordinates(result.pop('warmup_samples'), model)
    result['parameter_samples'] = np.exp(result['log_samples'])
    result['effective_diagnostics'] = result.pop('diagnostics')
    result['diagnostics'] = chain_diagnostics(result['log_samples'])
    latent = 'sigma' if model == 'beta_gp' else 'mu'
    names = ['v_0', f'{latent}_v0', 'k', 'w', f'{latent}_w', 'W']
    result.update(model=model, parameter_names=names, prior_center=center, prior_sd=sd,
                  l=l if model != 'tp_mu_gp' else None,
                  omega=omega if model != 'tp_mu_gp' else None,
                  nu=nu if model == 'tp_mu_gp' else None,
                  mixing_prior='inverse-gamma' if model == 'tp_mu_gp' else 'GIG',
                  initialization=initialization, starts=starts)
    flat = result['parameter_samples'].reshape(-1, 6)
    quantiles = np.quantile(flat, [0.025, 0.5, 0.975], axis=0)
    result['summary'] = [dict(parameter=name, mean=float(flat[:, d].mean()),
                              std=float(flat[:, d].std()), variance=float(flat[:, d].var()),
                              median=float(quantiles[1, d]), lower=float(quantiles[0, d]),
                              upper=float(quantiles[2, d]),
                              **{key: float(value[d]) for key, value in result['diagnostics'].items()})
                         for d, name in enumerate(names)]
    diagnostics = result['diagnostics']
    result['diagnostics_ok'] = bool(
        np.max(diagnostics['rhat']) < 1.01 and np.min(diagnostics['ess_bulk']) >= 400
        and np.min(diagnostics['ess_tail']) >= 400 and not np.any(result['divergent']))
    if not result['diagnostics_ok']:
        warnings.warn(f'{model}: inspect convergence diagnostics before trusting posterior summaries', RuntimeWarning)
    return result


def conditional_prediction(y, x, x_pred, log_parameters, *, model):
    """Exact conditional Gaussian moments for f and h=W*beta (or h=mu).

    Returns full covariances at prediction locations, conditional on theta,W.
    native latent beta moments follow by dividing h moments by W and W**2.
    """
    y, x = _inputs_1d(y, x)
    xp = np.asarray(x_pred, dtype=float)
    if xp.ndim != 1 or not xp.size or not np.all(np.isfinite(xp)):
        raise ValueError('x_pred must be a nonempty finite 1D array')
    q = np.asarray(log_parameters, dtype=float)
    if q.shape != (6,) or not np.all(np.isfinite(q)):
        raise ValueError('log_parameters must contain six finite values')
    a, b, noise, w, sw, W = np.exp(effective_coordinates(q, model))
    def kernel(u, v, amplitude, weight):
        return amplitude*np.exp(-0.5*weight*(u[:, None]-v[None, :])**2)
    Hxx = kernel(x, x, b, sw)
    Hpx = kernel(xp, x, b, sw)
    Hpp = kernel(xp, xp, b, sw)
    Txx = kernel(x, x, a, w)+Hxx
    Tpx = kernel(xp, x, a, w)+Hpx
    Tpp = kernel(xp, xp, a, w)+Hpp
    L = linalg.cholesky(Txx+noise*np.eye(len(x)), lower=True, check_finite=False)
    alpha = linalg.cho_solve((L, True), y, check_finite=False)
    result = dict(noise_variance=noise, W=W)
    for name, cross, prior in [('f', Tpx, Tpp), ('effect', Hpx, Hpp)]:
        solved = linalg.solve_triangular(L, cross.T, lower=True, check_finite=False)
        covariance = prior-solved.T@solved
        covariance = (covariance+covariance.T)/2
        result[f'{name}_mean'] = cross@alpha
        result[f'{name}_cov'] = covariance
    return result


def predict_joint_mixture(y, x, x_pred, fit, *, credible_mass=0.95):
    """Average all retained conditional Gaussians, including full f covariance.

    Equal weights are used for every retained MCMC state (including repeats
    from rejection). Bands are pointwise equal-tail Gaussian-mixture quantiles,
    not mean +/- 1.96*std. No additional f draws or thinning are needed.
    Stores conditional means and marginal variances so the mixture is reusable.
    """
    if not 0 < credible_mass < 1:
        raise ValueError('credible_mass must lie in (0,1)')
    samples = np.asarray(fit['log_samples'], dtype=float)
    if samples.ndim != 3 or samples.shape[2] != 6 or not samples.size:
        raise ValueError('fit log_samples must have shape (chains,draws,6)')
    flat = samples.reshape(-1, 6)
    model = fit['model']
    native = 'beta' if model == 'beta_gp' else 'mu'
    targets = ['f', 'effect', native, 'y']
    n, count = len(x_pred), len(flat)
    means = {name: np.empty((count, n)) for name in targets}
    variances = {name: np.empty((count, n)) for name in targets}
    within = {name: np.zeros((n, n)) for name in targets}
    for index, q in enumerate(flat):
        conditional = conditional_prediction(y, x, x_pred, q, model=model)
        factor = conditional['W'] if model == 'beta_gp' else 1.
        for name, mean, cov in [
                ('f', conditional['f_mean'], conditional['f_cov']),
                ('effect', conditional['effect_mean'], conditional['effect_cov']),
                (native, conditional['effect_mean']/factor, conditional['effect_cov']/factor**2),
                ('y', conditional['f_mean'], conditional['f_cov']+conditional['noise_variance']*np.eye(n))]:
            means[name][index] = mean
            variances[name][index] = np.maximum(np.diag(cov), 0.)
            within[name] += cov/count
    weights = np.full(count, 1/count)
    result = dict(weights=weights, credible_mass=credible_mass, draws_used=count)
    for name in targets:
        component = means[name]
        mean = weights@component
        centered = component-mean
        between = (centered.T*weights)@centered
        covariance = within[name]+between
        variance = np.maximum(np.diag(covariance), 0.)
        result.update({f'{name}_mean': mean, f'{name}_cov': covariance,
                       f'{name}_var': variance, f'{name}_std': np.sqrt(variance),
                       f'{name}_within_cov': within[name], f'{name}_between_cov': between,
                       f'{name}_ci': _mixture_intervals(component, variances[name], weights, credible_mass),
                       f'component_{name}_means': component,
                       f'component_{name}_variances': variances[name]})
    return result
