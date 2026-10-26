"""Regression with beta=0 and an independent GP mean mu(x).

mu ~ GP(mu0, M), W ~ GIG(l, omega), independently;
f | mu,W ~ N(mu, W*K), y | f,W ~ N(f, W*k*I).
Thus y | W ~ N(mu0, M + W*(K+kI)). All parameters are fixed here.
Uses the same normalized GIG prior and log-W quadrature as beta_f_inf.
"""

import numpy as np
from scipy import linalg, special

from GH_proc_w_betaprior import (
    _check_model, _inputs, _mixture_intervals, _se, _vector, _w_quadrature,
)


def _training(ys, xs, l, omega, w, v_0, k, mu0, mu_v0, mu_w):
    K, M = _se(xs, xs, v_0, w), _se(xs, xs, mu_v0, mu_w)
    C = K + k*np.eye(len(xs))
    eigenvalues, V = linalg.eigh(M, C)
    eigenvalues = np.maximum(eigenvalues, 0)
    log_det_C = 2*np.log(np.diag(linalg.cholesky(C, lower=True))).sum()
    residual = V.T @ (ys-mu0)
    log_normalizer = np.log(2) + np.log(special.kve(l, omega))
    if not np.isfinite(log_normalizer):
        raise ValueError('GIG normalization is not finite for these l, omega')

    def log_density(t):
        t = np.asarray(t)
        W = np.exp(t)
        denom = W[..., None] + eigenvalues
        likelihood = -0.5*(len(xs)*np.log(2*np.pi) + log_det_C
                            + np.log(denom).sum(axis=-1)
                            + np.sum(residual**2/denom, axis=-1))
        return likelihood + l*t - 2*omega*np.sinh(t/2)**2 - log_normalizer

    return K, M, eigenvalues, V, residual, log_density


def _prepare(ys, xs, l, omega, w, v_0, k, mu0, mu_v0, mu_w,
             n_particles, max_particles, quad_tol, tail_log_drop):
    xs, ys = _inputs(xs, 'xs'), _vector(ys, 'ys')
    if len(xs) != len(ys):
        raise ValueError('xs must have one row per observation')
    dimension = 1 if xs.ndim == 1 else xs.shape[1]
    w, mu_w = _check_model(l, omega, w, v_0, k, mu0, mu_v0, mu_w, dimension)
    if (not isinstance(n_particles, (int, np.integer)) or n_particles < 8
            or not isinstance(max_particles, (int, np.integer)) or max_particles < 2*n_particles):
        raise ValueError('Use integer n_particles >= 8 and max_particles >= 2*n_particles')
    if not np.isfinite(quad_tol) or quad_tol <= 0 or not np.isfinite(tail_log_drop) or tail_log_drop < 10:
        raise ValueError('quad_tol must be positive and tail_log_drop must be at least 10')
    return ys, xs, w, mu_w


def mu_log_marginal_likelihood(ys, xs, *, l=1, omega=1, w=1, v_0=1,
                                k=1, mu0=0, mu_v0=1, mu_w=0.1,
                                return_gradient=False,
                                n_particles=64, max_particles=2048,
                                quad_tol=1e-6, tail_log_drop=35, log_w_bounds=None):
    """Log p(y | parameters), integrating mu, f and W.

    With return_gradient=True, also return analytic, quadrature-averaged
    derivatives in log positive parameters: log_v_0, log_mu_v0, log_k,
    log_w[D], log_mu_w[D]. mu0 differentiates the ordinary constant mean.
    l and omega remain fixed. No finite differences through quadrature.
    """
    ys, xs, w, mu_w = _prepare(ys, xs, l, omega, w, v_0, k, mu0, mu_v0, mu_w,
                               n_particles, max_particles, quad_tol, tail_log_drop)
    K, M, eigenvalues, V, residual, log_density = _training(
        ys, xs, l, omega, w, v_0, k, mu0, mu_v0, mu_w)
    log_W, weights, diagnostics = _w_quadrature(
        log_density, np.arcsinh((l-len(ys)/2)/omega), n_particles, max_particles,
        quad_tol, tail_log_drop, log_w_bounds)
    log_likelihood = diagnostics['log_integral']
    if not return_gradient:
        return log_likelihood
    W = np.exp(log_W)
    denom = W[:, None] + eigenvalues
    alphas = (residual/denom) @ V.T  # rows are D_W^-1 (y-mu0)
    grad_K = 0.5*((alphas.T*(weights*W)) @ alphas
                  - (V*(weights @ (W[:, None]/denom))) @ V.T)
    grad_M = 0.5*((alphas.T*weights) @ alphas
                  - (V*(weights @ (1/denom))) @ V.T)
    points = xs[:, None] if xs.ndim == 1 else xs
    grad_w, grad_mw = np.empty(points.shape[1]), np.empty(points.shape[1])
    for d in range(points.shape[1]):
        distance2 = (points[:, None, d]-points[None, :, d])**2
        grad_w[d] = -0.5*w[d]*np.sum(grad_K*K*distance2)
        grad_mw[d] = -0.5*mu_w[d]*np.sum(grad_M*M*distance2)
    gradient = dict(log_v_0=float(np.sum(grad_K*K)), log_mu_v0=float(np.sum(grad_M*M)),
                    log_k=float(k*np.trace(grad_K)), log_w=grad_w, log_mu_w=grad_mw,
                    mu0=float(np.sum(weights @ alphas)))
    return log_likelihood, gradient


def mu_f_inf(ys, xs, x_pred=None, *, l=1, omega=1, w=1, v_0=1,
             k=1, mu0=0, mu_v0=1, mu_w=0.1,
             n_particles=64, max_particles=2048, quad_tol=1e-6,
             tail_log_drop=35, log_w_bounds=None, credible_mass=0.95):
    """Infer [mu(x_pred), f(x_pred)] with beta=0 and mu independent of W.

    xs: (N,) or (N,D). mu0 is the constant prior mean of the random mu(x).
    M(x,x') = mu_v0 * exp(-sum_d mu_w[d]*(x[d]-x'[d])**2/2).
    mu_w/w are inverse squared length scales. Observation noise is W*k.

    Results parallel beta_f_inf, replacing beta_* keys with mu_* keys.
    component_means/covariances are ordered [mu, f]. Posterior moments and
    pointwise equal-tail intervals integrate W; hyperparameters remain fixed.
    y_* summaries predict new noisy observations. score_predictions() accepts
    this result because its second component block is f in both models.
    """
    ys, xs, w, mu_w = _prepare(ys, xs, l, omega, w, v_0, k, mu0, mu_v0, mu_w,
                               n_particles, max_particles, quad_tol, tail_log_drop)
    x_pred = xs.copy() if x_pred is None else _inputs(x_pred, 'x_pred')
    dimension = 1 if xs.ndim == 1 else xs.shape[1]
    if (1 if x_pred.ndim == 1 else x_pred.shape[1]) != dimension:
        raise ValueError('xs and x_pred must have the same dimension')
    if not 0 < credible_mass < 1:
        raise ValueError('credible_mass must lie between zero and one')
    _, _, eigenvalues, V, residual, log_density = _training(
        ys, xs, l, omega, w, v_0, k, mu0, mu_v0, mu_w)
    log_W, weights, diagnostics = _w_quadrature(
        log_density, np.arcsinh((l-len(ys)/2)/omega), n_particles, max_particles,
        quad_tol, tail_log_drop, log_w_bounds)
    particles = np.exp(log_W)
    m = len(x_pred)
    M_pp, K_pp = _se(x_pred, x_pred, mu_v0, mu_w), _se(x_pred, x_pred, v_0, w)
    M_pV, K_pV = _se(x_pred, xs, mu_v0, mu_w) @ V, _se(x_pred, xs, v_0, w) @ V
    means = np.empty((len(particles), 2*m))
    covariances = np.empty((len(particles), 2*m, 2*m))
    for i, W in enumerate(particles):
        denom = W + eigenvalues
        cross = np.vstack((M_pV, M_pV + W*K_pV))
        means[i] = mu0 + cross @ (residual/denom)
        prior = np.block([[M_pp, M_pp], [M_pp, M_pp + W*K_pp]])
        scaled = cross / np.sqrt(denom)
        covariance = prior - scaled @ scaled.T
        covariances[i] = (covariance + covariance.T)/2
    mean = weights @ means
    centered = means - mean
    covariance = np.einsum('p,pij->ij', weights, covariances) + (centered.T*weights) @ centered
    covariance = (covariance + covariance.T)/2
    intervals = _mixture_intervals(means, np.diagonal(covariances, axis1=1, axis2=2),
                                   weights, credible_mass)
    result = dict(W=particles, log_W=log_W, weights=weights,
                  component_means=means, component_covariances=covariances,
                  joint_mean=mean, joint_cov=covariance, mu_f_cov=covariance[:m, m:],
                  x_pred=x_pred, credible_mass=credible_mass, quadrature=diagnostics,
                  log_marginal_likelihood=diagnostics['log_integral'])
    for name, section in [('mu', slice(0, m)), ('f', slice(m, 2*m))]:
        variance = np.maximum(np.diag(covariance)[section], 0)
        result.update({f'{name}_mean': mean[section], f'{name}_cov': covariance[section, section],
                       f'{name}_var': variance, f'{name}_std': np.sqrt(variance),
                       f'{name}_ci': intervals[section]})
    y_variances = np.maximum(np.diagonal(covariances[:, m:, m:], axis1=1, axis2=2), 0) + particles[:, None]*k
    result.update(component_y_variances=y_variances, y_mean=result['f_mean'].copy(),
                  y_cov=result['f_cov'] + float(weights @ particles)*k*np.eye(m),
                  y_ci=_mixture_intervals(means[:, m:], y_variances, weights, credible_mass))
    result['y_var'] = np.maximum(np.diag(result['y_cov']), 0)
    result['y_std'] = np.sqrt(result['y_var'])
    return result


def beta_effect_posterior(posterior):
    """Summaries of h=W*beta, for comparison against an additive GP mu.

    E[W*beta | y] generally differs from E[W | y]*E[beta | y]. Multiply
    each conditional Gaussian component before averaging its moments.
    Returns mean/cov/var/std/ci and the transformed component means/variances.
    """
    m = posterior['component_means'].shape[1]//2
    W, weights = posterior['W'], posterior['weights']
    means = W[:, None]*posterior['component_means'][:, :m]
    covariance = np.einsum('p,pij->ij', weights*W**2,
                           posterior['component_covariances'][:, :m, :m])
    mean = weights @ means
    centered = means - mean
    covariance += (centered.T*weights) @ centered
    covariance = (covariance + covariance.T)/2
    variances = np.maximum(np.diagonal(posterior['component_covariances'][:, :m, :m], axis1=1, axis2=2), 0)*W[:, None]**2
    variance = np.maximum(np.diag(covariance), 0)
    return dict(mean=mean, cov=covariance, var=variance, std=np.sqrt(variance),
                ci=_mixture_intervals(means, variances, weights, posterior['credible_mass']),
                component_means=means, component_variances=variances)
