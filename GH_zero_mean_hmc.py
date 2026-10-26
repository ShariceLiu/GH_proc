"""Single-SE, zero-mean GP/STP/GHP baselines for the existing 1D experiments.

GP: y|theta ~ N(0, K + kI).
STP: W~IG(nu/2,(nu-2)/2), y|theta,W ~ N(0, W(K+kI)).
GHP: W~GIG(lambda,omega), y|theta,W ~ N(0, W(K+kI)), beta=0.
There is no independent GP mean, second kernel, or fitted constant offset.
An optional ghp_known_beta model uses an externally supplied beta function:
y|theta,W,beta ~ N(W*beta, W(K+kI)); beta is not inferred.
ghp_plugin_beta uses the same conditional model but receives a beta estimate
learned from training observations, rather than the true generating function.
The ghp_fixed_mu variant instead uses N(mu, W(K+kI)), with supplied mu
independent of W. Neither supplied function has fitted GP hyperparameters.
gp_plugin_mu, stp_plugin_mu and ghp_plugin_mu hold a training-data estimate
of mu fixed, using respectively no mixing, inverse-gamma or GIG mixing.

Physical log parameters are [log(v_0), log(k), log(w), (log(W))]. HMC uses
[log(W*v_0), log(W*k), log(w), log(W)] for mixing models. Priors are Normal
on the physical log kernel parameters; the transformation between log
coordinates has determinant one. The mixing prior includes dW/dlog(W).
"""
import numpy as np
from scipy import linalg, optimize, special

from GH_proc_joint_hmc import _inputs_1d
from GH_proc_w_betaprior import _mixture_intervals
from GH_proc_w_betaprior_hmc import chain_diagnostics, sample_hmc

MODELS = ('gp_zero', 'stp_zero', 'ghp_zero')
KNOWN_BETA_MODEL = 'ghp_known_beta'
FIXED_MU_MODEL = 'ghp_fixed_mu'
PLUGIN_BETA_MODEL = 'ghp_plugin_beta'
BETA_MODELS = (KNOWN_BETA_MODEL, PLUGIN_BETA_MODEL)
PLUGIN_MU_MODELS = ('gp_plugin_mu', 'stp_plugin_mu', 'ghp_plugin_mu')
MU_MODELS = (FIXED_MU_MODEL, *PLUGIN_MU_MODELS)
GP_MODELS = ('gp_zero', 'gp_plugin_mu')
STP_MODELS = ('stp_zero', 'stp_plugin_mu')
ALL_MODELS = (*MODELS, KNOWN_BETA_MODEL, FIXED_MU_MODEL, PLUGIN_BETA_MODEL, *PLUGIN_MU_MODELS)
GHP_MODELS = tuple(m for m in ALL_MODELS if m not in (*GP_MODELS, *STP_MODELS))


def _known_beta(beta, size, model):
    if model not in BETA_MODELS:
        if beta is not None:
            raise ValueError('A supplied beta requires ghp_known_beta or ghp_plugin_beta')
        return None
    values = np.asarray(beta, dtype=float)
    if values.shape != (size,) or not np.all(np.isfinite(values)):
        raise ValueError(f'{model} requires {size} finite beta values')
    return values


def _fixed_mu(mu, size, model):
    if model not in MU_MODELS:
        if mu is not None:
            raise ValueError('A supplied mu requires a fixed-mu or plug-in-mu model')
        return None
    values = np.asarray(mu, dtype=float)
    if values.shape != (size,) or not np.all(np.isfinite(values)):
        raise ValueError(f'{model} requires {size} finite mu values')
    return values


def coordinates(values, model, *, inverse=False):
    if model not in ALL_MODELS:
        raise ValueError('Unknown zero-mean model')
    result = np.array(values, dtype=float, copy=True)
    expected = 3 if model in GP_MODELS else 4
    if result.ndim < 1 or result.shape[-1] != expected:
        raise ValueError(f'{model} requires {expected} log parameters')
    if model not in GP_MODELS:
        t = result[..., -1].copy()*(-1 if inverse else 1)
        result[..., 0] += t
        result[..., 1] += t
    return result


def prior_center_from_training(y):
    """Keep the original residual-kernel/noise priors; remove the second GP."""
    scale = float(np.var(y))
    scale = scale if scale > 1e-12 else 1.
    return np.log([.5*scale, .1*scale, 1.])


def make_target(y, x, *, model, prior_center, prior_sd=2., l=1., omega=1., nu=5., beta=None, mu=None):
    y, x = _inputs_1d(y, x)
    if model not in ALL_MODELS:
        raise ValueError('Unknown model')
    beta = _known_beta(beta, len(y), model)
    mu = _fixed_mu(mu, len(y), model)
    if model in STP_MODELS and (not np.isfinite(nu) or nu <= 2):
        raise ValueError('Need nu > 2')
    if model in GHP_MODELS and (not np.isfinite(l) or not np.isfinite(omega) or omega <= 0):
        raise ValueError('Invalid GIG parameters')
    center = np.asarray(prior_center, dtype=float)
    sd = np.broadcast_to(np.asarray(prior_sd, dtype=float), (3,)).copy()
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError('Need three finite log-prior centers')
    if not np.all(np.isfinite(sd)) or np.any(sd <= 0):
        raise ValueError('Invalid prior standard deviations')
    dimension = 3 if model in GP_MODELS else 4
    distances = (x[:, None]-x[None, :])**2
    eye = np.eye(len(y))
    constant = -.5*(len(y)+3)*np.log(2*np.pi)-np.log(sd).sum()
    if model in STP_MODELS:
        shape, scale = nu/2., (nu-2)/2.
        mixing_constant = shape*np.log(scale)-special.gammaln(shape)
    elif model in GHP_MODELS:
        mixing_constant = -np.log(2.)-np.log(special.kve(l, omega))+omega

    def target(z):
        z = np.asarray(z, dtype=float)
        if z.shape != (dimension,) or not np.all(np.isfinite(z)):
            return -np.inf, np.zeros(dimension)
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                a, noise, w = np.exp(z[:3])
                K = a*np.exp(-.5*w*distances)
                C = K+noise*eye
                L = linalg.cholesky(C, lower=True, check_finite=False)
                # Effective covariance holds A=W*v and N=W*k fixed, but the
                # known-beta mean still depends on W and must be differentiated.
                mean = np.exp(z[-1])*beta if beta is not None else mu
                residual = y-mean if mean is not None else y
                alpha = linalg.cho_solve((L, True), residual, check_finite=False)
                score = .5*(np.outer(alpha, alpha)-linalg.cho_solve((L, True), eye, check_finite=False))
                gradient = np.empty(dimension)
                gradient[:3] = [np.sum(score*K), noise*np.trace(score),
                                np.sum(score*K*(-.5*w*distances))]
                q = coordinates(z, model, inverse=True)
                standardized = (q[:3]-center)/sd
                prior_gradient = -standardized/sd
                gradient[:3] += prior_gradient
                logp = constant-np.log(L.diagonal()).sum()-.5*residual@alpha-.5*standardized@standardized
                if model not in GP_MODELS:
                    W, t = np.exp(z[-1]), z[-1]
                    if model in STP_MODELS:
                        logp += mixing_constant-shape*t-scale/W
                        mixing_gradient = -shape+scale/W
                    else:
                        logp += mixing_constant+l*t-.5*omega*(W+1/W)
                        mixing_gradient = l-.5*omega*(W-1/W)
                    gradient[-1] = mixing_gradient-prior_gradient[0]-prior_gradient[1]
                    if beta is not None:
                        gradient[-1] += mean@alpha
                if np.isfinite(logp) and np.all(np.isfinite(gradient)):
                    return float(logp), gradient
        except (np.linalg.LinAlgError, FloatingPointError, OverflowError):
            pass
        return -np.inf, np.zeros(dimension)
    return target


def diagnostics_summary(fit):
    both = [fit['diagnostics'], fit['effective_diagnostics']]
    result = dict(max_rhat=max(float(np.max(d['rhat'])) for d in both),
                  min_bulk_ess=min(float(np.min(d['ess_bulk'])) for d in both),
                  min_tail_ess=min(float(np.min(d['ess_tail'])) for d in both),
                  divergences=int(np.sum(fit['divergent'])),
                  acceptance=float(np.mean(fit['acceptance'])))
    result['diagnostics_ok'] = bool(result['max_rhat'] < 1.01 and result['min_bulk_ess'] >= 400
                                    and result['min_tail_ess'] >= 400 and not result['divergences'])
    return result


def sample_parameters(y, x, *, model, prior_center=None, prior_sd=2., l=1., omega=1., nu=5.,
                      chains=4, warmup=2000, draws=3000, target_accept=.99,
                      leapfrog_steps=(20, 50), seed=19026, verbose=True, beta=None, mu=None):
    """Joint HMC of theta and W; Gaussian processes are marginalized exactly."""
    y, x = _inputs_1d(y, x)
    center = prior_center_from_training(y) if prior_center is None else np.asarray(prior_center)
    sd = np.broadcast_to(prior_sd, (3,)).copy()
    target = make_target(y, x, model=model, prior_center=center, prior_sd=sd,
                         l=l, omega=omega, nu=nu, beta=beta, mu=mu)
    base = center.copy()
    if model not in GP_MODELS:
        t = np.log((nu-2)/nu) if model in STP_MODELS else np.arcsinh(l/omega)
        base = coordinates(np.r_[center, t], model)
    rng = np.random.default_rng(seed)
    starts, initialization = [], []
    for _ in range(chains):
        guess = base+rng.normal(0., .7, len(base))
        def objective(z):
            value, gradient = target(z)
            return -value, -gradient
        mode = optimize.minimize(objective, guess, jac=True, method='L-BFGS-B',
                                 options=dict(maxiter=400, ftol=1e-10))
        position = mode.x if np.isfinite(target(mode.x)[0]) else guess
        initialization.append(dict(success=bool(mode.success), log_density=target(position)[0], message=str(mode.message)))
        for _ in range(1000):
            start = position+rng.normal(0., .35, len(base))
            if np.isfinite(target(start)[0]):
                starts.append(start)
                break
        else:
            raise RuntimeError('Could not initialize finite chain')
    fit = sample_hmc(target, np.array(starts), draws=draws, warmup=warmup, seed=seed,
                     step_size=.05, leapfrog_steps=leapfrog_steps, target_accept=target_accept, verbose=verbose)
    fit['effective_log_samples'] = fit.pop('samples')
    fit['log_samples'] = coordinates(fit['effective_log_samples'], model, inverse=True)
    fit['warmup_log_samples'] = coordinates(fit.pop('warmup_samples'), model, inverse=True)
    fit['effective_diagnostics'] = fit.pop('diagnostics')
    fit['diagnostics'] = chain_diagnostics(fit['log_samples'])
    fit.update(model=model, prior_center=center, prior_sd=sd, l=l, omega=omega, nu=nu,
               parameter_names=['v_0', 'k', 'w']+([] if model in GP_MODELS else ['W']),
               starts=np.array(starts), initialization=initialization)
    if model in BETA_MODELS:
        fit['beta_train'] = np.asarray(beta, dtype=float).copy()
    if model in MU_MODELS:
        fit['mu_train'] = np.asarray(mu, dtype=float).copy()
    fit.update(diagnostics_summary(fit))
    flat = np.exp(fit['log_samples'].reshape(-1, len(base)))
    fit['summary'] = []
    for j, name in enumerate(fit['parameter_names']):
        quantiles = np.quantile(flat[:, j], [.025, .5, .975])
        fit['summary'].append(dict(parameter=name, mean=float(flat[:, j].mean()), std=float(flat[:, j].std()),
                                   variance=float(flat[:, j].var()), lower=float(quantiles[0]),
                                   median=float(quantiles[1]), upper=float(quantiles[2]),
                                   **{key: float(value[j]) for key, value in fit['diagnostics'].items()}))
    return fit


def conditional_prediction(y, x, xp, log_parameters, *, model, beta_train=None, beta_pred=None,
                           mu_train=None, mu_pred=None):
    """Full Gaussian f/y moments given theta and W; GP fixes W=1."""
    y, x = _inputs_1d(y, x)
    xp = np.asarray(xp, dtype=float)
    if xp.ndim != 1 or not xp.size or not np.all(np.isfinite(xp)):
        raise ValueError('Need finite nonempty 1D prediction inputs')
    beta_train = _known_beta(beta_train, len(x), model)
    beta_pred = _known_beta(beta_pred, len(xp), model)
    mu_train = _fixed_mu(mu_train, len(x), model)
    mu_pred = _fixed_mu(mu_pred, len(xp), model)
    q = np.asarray(log_parameters, dtype=float)
    z = coordinates(q, model)
    if z.ndim != 1 or not np.all(np.isfinite(z)):
        raise ValueError('Invalid log parameters')
    a, noise, w = np.exp(z[:3])
    def kernel(u, v):
        return a*np.exp(-.5*w*(u[:, None]-v[None, :])**2)
    L = linalg.cholesky(kernel(x, x)+noise*np.eye(len(x)), lower=True, check_finite=False)
    cross = kernel(xp, x)
    W = np.exp(z[-1]) if model in BETA_MODELS else 1.
    residual = y-W*beta_train if beta_train is not None else y
    if mu_train is not None:
        residual = y-mu_train
    mean = cross@linalg.cho_solve((L, True), residual, check_finite=False)
    if beta_pred is not None:
        mean += W*beta_pred
    if mu_pred is not None:
        mean += mu_pred
    solved = linalg.solve_triangular(L, cross.T, lower=True, check_finite=False)
    covariance = kernel(xp, xp)-solved.T@solved
    covariance = .5*(covariance+covariance.T)
    return dict(f_mean=mean, f_cov=covariance, noise_variance=noise,
                y_mean=mean.copy(), y_cov=covariance+noise*np.eye(len(xp)))


def predict_mixture(y, x, xp, fit, *, credible_mass=.95, interval_batch_size=48,
                    beta_train=None, beta_pred=None, mu_train=None, mu_pred=None):
    """Average all HMC states; bands invert the actual Gaussian-mixture CDF."""
    if not 0 < credible_mass < 1 or interval_batch_size < 1:
        raise ValueError('Invalid interval settings')
    samples = np.asarray(fit['log_samples'])
    expected = 3 if fit['model'] in GP_MODELS else 4
    if samples.ndim != 3 or samples.shape[-1] != expected or not samples.size:
        raise ValueError('Invalid sample array')
    flat = samples.reshape(-1, expected)
    count, n = len(flat), len(xp)
    means = np.empty((count, n))
    variances = {t: np.empty((count, n)) for t in ['f', 'y']}
    within = {t: np.zeros((n, n)) for t in ['f', 'y']}
    for index, q in enumerate(flat):
        p = conditional_prediction(y, x, xp, q, model=fit['model'],
                                    beta_train=beta_train, beta_pred=beta_pred,
                                    mu_train=mu_train, mu_pred=mu_pred)
        means[index] = p['f_mean']
        for name in ['f', 'y']:
            variances[name][index] = np.maximum(np.diag(p[f'{name}_cov']), 0.)
            within[name] += p[f'{name}_cov']/count
    weights = np.full(count, 1./count)
    mean = weights@means
    centered = means-mean
    between = (centered.T*weights)@centered
    result = dict(component_means=means, weights=weights, component_df=np.inf,
                  credible_mass=credible_mass, draws_used=count)
    for name in ['f', 'y']:
        covariance = within[name]+between
        ci = np.concatenate([
            _mixture_intervals(means[:, start:start+interval_batch_size],
                               variances[name][:, start:start+interval_batch_size], weights, credible_mass)
            for start in range(0, n, interval_batch_size)], axis=0)
        result.update({f'{name}_mean': mean.copy(), f'{name}_cov': covariance,
                       f'{name}_var': np.maximum(np.diag(covariance), 0.),
                       f'{name}_std': np.sqrt(np.maximum(np.diag(covariance), 0.)),
                       f'{name}_within_cov': within[name], f'{name}_between_cov': between.copy(),
                       f'{name}_ci': ci, f'component_{name}_variances': variances[name]})
    if fit['model'] in BETA_MODELS:
        beta = np.asarray(beta_pred, dtype=float)
        W = np.exp(flat[:, -1])
        result.update(beta_known=beta, W_mean=W.mean(), W_var=W.var(),
                      effect_mean=W.mean()*beta, effect_cov=W.var()*np.outer(beta, beta))
        if fit['model'] == PLUGIN_BETA_MODEL:
            result.update(beta_fixed=beta.copy(), beta_mean=beta.copy(),
                          beta_var=np.zeros(n), beta_cov=np.zeros((n, n)))
    if fit['model'] in MU_MODELS:
        W = np.ones(count) if fit['model'] in GP_MODELS else np.exp(flat[:, -1])
        result.update(mu_fixed=np.asarray(mu_pred, dtype=float).copy(), W_mean=W.mean(), W_var=W.var(),
                      effect_mean=np.asarray(mu_pred, dtype=float).copy(), effect_cov=np.zeros((n, n)))
        if fit['model'] in PLUGIN_MU_MODELS:
            result.update(mu_mean=np.asarray(mu_pred, dtype=float).copy(),
                          mu_var=np.zeros(n), mu_cov=np.zeros((n, n)))
    return result
