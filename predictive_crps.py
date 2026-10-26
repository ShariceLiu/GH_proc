"""Pointwise CRPS for Gaussian predictive mixtures, including HMC mixtures.

CRPS(F,t) = integral_{-inf}^t F(u)^2 du + integral_t^inf (1-F(u))^2 du.
The numerical method retains every component and avoids the O(S^2) pair sum.
The closed-form method is intended for small mixtures and independent checks.
Reference: https://scoringrules.readthedocs.io/en/latest/generated/scoringrules.crps_mixnorm.html
"""
import numpy as np
from scipy import integrate, special


def _inputs(truth, means, variances, weights):
    truth = np.atleast_1d(np.asarray(truth, dtype=float))
    means, variances = np.asarray(means, dtype=float), np.asarray(variances, dtype=float)
    if means.ndim == 1:
        means, variances = means[:, None], variances[:, None]
    if (truth.ndim != 1 or means.ndim != 2 or variances.shape != means.shape
            or means.shape[1] != truth.size or not means.size):
        raise ValueError('Need truth (points,), means/variances (components, points)')
    if not all(np.all(np.isfinite(a)) for a in [truth, means, variances]) or np.any(variances <= 0):
        raise ValueError('Inputs must be finite and component variances strictly positive')
    weights = np.ones(means.shape[0]) if weights is None else np.asarray(weights, dtype=float)
    if (weights.shape != (means.shape[0],) or not np.all(np.isfinite(weights))
            or np.any(weights < 0) or not np.isfinite(weights.sum()) or weights.sum() <= 0):
        raise ValueError('Need finite nonnegative component weights with positive sum')
    weights = weights/weights.sum()
    keep = weights > 0
    return truth, means[keep], variances[keep], weights[keep]


def normal_absolute_moment(delta, variance):
    """E|Z| for Z~Normal(delta, variance)."""
    sd = np.sqrt(variance)
    z = delta/sd
    return delta*special.erf(z/np.sqrt(2.))+sd*np.sqrt(2./np.pi)*np.exp(-.5*z*z)


def gaussian_mixture_crps_closed_form(truth, means, variances, weights=None):
    """Exact finite-mixture formula; quadratic in the number of components."""
    truth, means, variances, weights = _inputs(truth, means, variances, weights)
    first = weights@normal_absolute_moment(truth-means, variances)
    second = np.zeros_like(truth)
    # One point at a time keeps the reference implementation's memory bounded.
    for point in range(truth.size):
        delta = means[:, point, None]-means[None, :, point]
        variance = variances[:, point, None]+variances[None, :, point]
        second[point] = weights@normal_absolute_moment(delta, variance)@weights
    return first-.5*second


def gaussian_mixture_crps(truth, means, variances, weights=None, *, atol=1e-8,
                          rtol=1e-8, tail_sd=10., batch_size=16):
    """Integrate the full mixture CDF; return (pointwise scores, diagnostics).

    Uses adaptive Gauss-Kronrod integration split at the observed truth. Both
    sides are transformed to [0,1]; the survival function uses Phi(-z) directly
    to avoid subtracting nearly equal numbers. Truncation endpoints enclose
    every component's mean +/- tail_sd standard deviations. Truth values beyond
    these endpoints contribute an explicit distance term, avoiding very long
    integration intervals that could hide the narrow CDF transition.
    The separately reported tail bound is conservative. Integration error is
    a numerical estimate, not an HMC or statistical uncertainty estimate.
    """
    truth, means, variances, weights = _inputs(truth, means, variances, weights)
    if (not np.isfinite(atol) or not np.isfinite(rtol) or atol <= 0 or rtol <= 0
            or not np.isfinite(tail_sd) or tail_sd < 6 or batch_size < 1):
        raise ValueError('Invalid integration settings')
    sd = np.sqrt(variances)
    scores, errors, bounds = np.empty_like(truth), np.empty_like(truth), np.empty_like(truth)
    evaluations, refinements = [], 0
    for start in range(0, truth.size, batch_size):
        section = slice(start, start+batch_size)
        m, s, t = means[:, section], sd[:, section], truth[section]
        lower = np.min(m-tail_sd*s, axis=0)
        upper = np.max(m+tail_sd*s, axis=0)
        split = np.clip(t, lower, upper)
        offset = np.abs(t-split)
        left, right = split-lower, upper-split

        def integrand(u):
            cdf = weights@special.ndtr((lower+u*left-m)/s)
            survival = weights@special.ndtr((m-(split+u*right))/s)
            return left*cdf*cdf+right*survival*survival

        value, error, info = integrate.quad_vec(integrand, 0., 1., epsabs=atol, epsrel=rtol,
                                                norm='max', quadrature='gk21', full_output=True)
        neval = int(info.neval)
        if not info.success or not np.all(np.isfinite(value)) or np.any(value < 0):
            raise RuntimeError(f'CRPS integration failed at points {start}: {info.message}')
        # F^2 <= F in the left tail (and likewise survival on the right).
        # Each integrated normal tail is <= sd*phi(tail_sd), a loose upper bound.
        # With exterior truths, |1-F^2| <= 2*(1-F); allow twice each tail bound.
        bound = 4*(weights@s)*np.exp(-.5*tail_sd**2)/np.sqrt(2*np.pi)
        if np.any(error+bound > atol+rtol*np.abs(value+offset)):
            # quad_vec controls a batch norm: a large score can otherwise allow
            # an error estimate above the tolerance of smaller scores in the batch.
            value, error, info = integrate.quad_vec(integrand, 0., 1., epsabs=atol, epsrel=0.,
                                                    norm='max', quadrature='gk21', full_output=True)
            neval += int(info.neval)
            refinements += 1
        if not info.success or np.any(error+bound > atol+rtol*np.abs(value+offset)):
            raise RuntimeError(f'CRPS error estimate exceeds pointwise tolerance at batch {start}')
        scores[section], errors[section], bounds[section] = value+offset, error, bound
        evaluations.append(neval)
    return scores, dict(integration_error_estimate=errors, tail_error_bound=bounds,
                       evaluations_per_batch=evaluations, atol=atol, rtol=rtol, tail_sd=tail_sd,
                       n_components=len(weights), all_components_used=True,
                       absolute_tolerance_refinements=refinements)
