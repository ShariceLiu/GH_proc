"""Inference for a GH mixture with an independent zero-mean GP prior on beta."""

import numpy as np
from functools import lru_cache
from scipy import linalg, optimize, special, stats


def _vector(values, name):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a nonempty, finite 1D array")
    return values


def _inputs(values, name):
    """Accept (N,) for 1D or (N, D) with one input point per row."""
    values = np.asarray(values, dtype=float)
    if (values.ndim not in (1, 2) or not values.size or
            not np.all(np.isfinite(values))):
        raise ValueError(f"{name} must be a nonempty finite array of shape (N,) or (N, D)")
    return values


def _dimension_values(values, dimension, name, *, positive=False):
    values = np.asarray(values, dtype=float)
    if values.ndim == 0:
        values = np.full(dimension, float(values))
    if values.shape != (dimension,) or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a finite scalar or an array of length D={dimension}")
    if positive and np.any(values <= 0):
        raise ValueError(f"{name} must be strictly positive in every dimension")
    return values


def _check_model(l, omega, w, v_0, k, mu, sigma_v0, sigma_w, dimension=1):
    scalars = [l, omega, v_0, k, mu, sigma_v0]
    if any(np.ndim(value) != 0 for value in scalars) or not np.all(np.isfinite(scalars)):
        raise ValueError("l, omega, v_0, k, mu and sigma_v0 must be finite scalars")
    if min(omega, k) <= 0 or min(v_0, sigma_v0) < 0:
        raise ValueError("omega and k must be positive; variances nonnegative")
    w = _dimension_values(w, dimension, "w", positive=True)
    sigma_w = _dimension_values(sigma_w, dimension, "sigma_w", positive=True)
    return w, sigma_w


def _se(xs, ys, variance, inverse_length_sq):
    # No diagonal jitter on cross-covariances, even when len(xs) == len(ys).
    xs = xs[:, None] if xs.ndim == 1 else xs
    ys = ys[:, None] if ys.ndim == 1 else ys
    squared_distance = np.sum(inverse_length_sq * (xs[:, None, :] - ys[None, :, :])**2, axis=-1)
    return variance * np.exp(-0.5 * squared_distance)


def Multivariate_GH(xs, l=1, omega=1, w=1, v_0=1, a_b=1, b_b=0,
                    k=1, mu=-1, *, beta_prior=True, sigma_v0=1,
                    sigma_w=0.1, seed=None, return_latents=False):
    """Generate scalar beta, f and y at N inputs of any dimension D.

    xs: shape (N, D), one input per row; (N,) also works for D=1.
    w, sigma_w: positive scalars (shared across dimensions), or arrays of
        length D. They are inverse squared length scales, not length scales:
        K(x,x') = v_0 * exp(-sum_d w[d]*(x[d]-x'[d])**2 / 2).
        Sigma uses sigma_v0 and sigma_w in the same way.
    beta_prior=True: draw beta ~ GP(0, Sigma), independently of W.
    beta_prior=False: use beta(x) = a_b + sum_d b_b[d]*x[d]; a scalar b_b
        applies the same slope to every dimension. a_b/b_b are ignored in GP mode.

    One scalar W ~ GIG(l, omega) is shared across all input points.
    f | beta,W ~ N(mu + W*beta, W*K); y | f,W ~ N(f, W*k*I).
    Returns (f, y), or (f, y, beta, W) if return_latents=True; f/y/beta
    each have shape (N,). seed accepts an integer or a numpy.random.Generator.
    For a coherent train/test realization, simulate on their concatenated
    inputs once, then split the observations.
    """
    xs = _inputs(xs, "xs")
    points = xs[:, None] if xs.ndim == 1 else xs
    n, dimension = points.shape
    w, sigma_w = _check_model(l, omega, w, v_0, k, mu, sigma_v0, sigma_w, dimension)
    rng = np.random.default_rng(seed)
    W = float(stats.geninvgauss.rvs(p=l, b=omega, random_state=rng))
    if beta_prior:
        beta = rng.multivariate_normal(np.zeros(n), _se(xs, xs, sigma_v0, sigma_w))
    else:
        if np.ndim(a_b) != 0 or not np.isfinite(a_b):
            raise ValueError("a_b must be a finite scalar")
        slopes = _dimension_values(b_b, dimension, "b_b")
        beta = a_b + points @ slopes
    fs = rng.multivariate_normal(mu + W * beta, W * _se(xs, xs, v_0, w))
    ys = fs + np.sqrt(W * k) * rng.standard_normal(n)
    return (fs, ys, beta, W) if return_latents else (fs, ys)


def Multivariate_GH_1d(xs, l=1, omega=1, w=1, v_0=1, a_b=1, b_b=0,
                       k=1, mu=-1, *, beta_prior=False, sigma_v0=1,
                       sigma_w=0.1, seed=None, return_latents=False):
    """Generate f and y using one shared W ~ GIG(l, omega).

    K(x,x') = v_0 exp(-w (x-x')**2 / 2).
    With beta_prior=True, beta ~ GP(0, Sigma), independently of W, where
    Sigma(x,x') = sigma_v0 exp(-sigma_w (x-x')**2 / 2).
    Otherwise the original beta(x) = a_b + b_b*x is used.

    f | beta,W ~ N(mu + W*beta, W*K); y | f,W ~ N(f, W*k*I).
    Returns (f, y), or (f, y, beta, W) if return_latents=True.
    seed accepts an integer or a numpy.random.Generator.
    """
    xs = _vector(xs, "xs")
    return Multivariate_GH(xs, l=l, omega=omega, w=w, v_0=v_0, a_b=a_b, b_b=b_b,
                           k=k, mu=mu, beta_prior=beta_prior, sigma_v0=sigma_v0,
                           sigma_w=sigma_w, seed=seed, return_latents=return_latents)


@lru_cache(maxsize=16)
def _legendre_rule(count):
    """Reuse fixed quadrature rules across likelihood/gradient evaluations."""
    return special.roots_legendre(count)


def _w_quadrature(log_density, center, n_particles, max_particles, quad_tol,
                  tail_log_drop, log_w_bounds):
    """Gauss-Legendre quadrature for the density of t=log(W), including dW/dt."""
    if log_w_bounds is None:
        lo, hi = center - 8.0, center + 8.0
        for _ in range(32):
            grid = np.linspace(lo, hi, max(65, int((hi - lo) * 4) + 1))
            heights = log_density(grid)
            # Refine detected modes so narrow peaks are resolved by quadrature.
            indices = np.flatnonzero((heights[1:-1] >= heights[:-2]) &
                                     (heights[1:-1] >= heights[2:])) + 1
            modes = [optimize.minimize_scalar(
                lambda t: -float(log_density(t)), bounds=(grid[i-1], grid[i+1]),
                method="bounded", options={"xatol": 1e-10}) for i in indices]
            peak = max([float(np.max(heights))] + [-mode.fun for mode in modes])
            left_ok = heights[0] < peak - tail_log_drop
            right_ok = heights[-1] < peak - tail_log_drop
            if left_ok and right_ok and modes:
                significant = [mode.x for mode in modes if -mode.fun > peak - tail_log_drop]
                target = peak - tail_log_drop
                lo = optimize.brentq(lambda t: float(log_density(t)) - target,
                                     lo, min(significant))
                hi = optimize.brentq(lambda t: float(log_density(t)) - target,
                                     max(significant), hi)
                break
            lo -= 8.0 if not left_ok else 0.0
            hi += 8.0 if not right_ok else 0.0
        else:
            raise RuntimeError("Could not bracket p(log W | y); supply log_w_bounds")
    else:
        bounds = np.asarray(log_w_bounds, dtype=float)
        if bounds.shape != (2,) or not np.all(np.isfinite(bounds)) or bounds[0] >= bounds[1]:
            raise ValueError("log_w_bounds must contain two finite, increasing values")
        lo, hi = bounds

    previous = None
    count = n_particles
    while True:
        nodes, quadrature_weights = _legendre_rule(count)
        log_W = (hi + lo) / 2 + (hi - lo) / 2 * nodes
        log_mass = log_density(log_W) + np.log(quadrature_weights) + np.log((hi - lo) / 2)
        log_Z = special.logsumexp(log_mass)
        # Check normalization and W^-1, W, W^2 integrals on successively finer grids.
        log_integrals = np.array([special.logsumexp(log_mass + power * log_W)
                                  for power in (0, -1, 1, 2)])
        error = np.inf if previous is None else np.max(np.abs(log_integrals - previous))
        if np.isfinite(log_Z) and error < quad_tol:
            weights = np.exp(log_mass - log_Z)
            weights /= weights.sum()
            return log_W, weights, {"log_w_bounds": (float(lo), float(hi)),
                                    "n_particles": count, "log_integral_change": float(error),
                                    "quad_tol": quad_tol, "log_integral": float(log_Z)}
        if count >= max_particles:
            raise RuntimeError("W quadrature did not converge; increase max_particles "
                               "or check log_w_bounds")
        previous = log_integrals
        count = min(2 * count, max_particles)


def _mixture_intervals(means, variances, weights, credible_mass):
    """Pointwise equal-tail quantiles, including any zero-variance components."""
    stds = np.sqrt(np.maximum(variances, 0))
    intervals = []
    for probability in ((1 - credible_mass) / 2, (1 + credible_mass) / 2):
        component_quantiles = means + special.ndtri(probability) * stds
        lower = component_quantiles.min(axis=0)
        upper = component_quantiles.max(axis=0)
        for _ in range(60):
            mid = (lower + upper) / 2
            z = np.divide(mid - means, stds, out=np.zeros_like(means), where=stds > 0)
            cdfs = np.where(stds > 0, special.ndtr(z), mid >= means)
            below = weights @ cdfs < probability
            lower = np.where(below, mid, lower)
            upper = np.where(below, upper, mid)
        intervals.append((lower + upper) / 2)
    return np.column_stack(intervals)


def _training_distribution(ys, xs, l, omega, w, v_0, k, mu, sigma_v0, sigma_w):
    """Shared, normalized training likelihood; inputs/parameters already validated."""
    n = len(ys)
    K = _se(xs, xs, v_0, w)
    Sigma = _se(xs, xs, sigma_v0, sigma_w)
    C = K + k * np.eye(n)
    # V.T @ C @ V = I; V.T @ Sigma @ V = diag(eigenvalues).
    eigenvalues, V = linalg.eigh(Sigma, C)
    eigenvalues = np.maximum(eigenvalues, 0)
    log_det_C = 2 * np.log(np.diag(linalg.cholesky(C, lower=True))).sum()
    residual = V.T @ (ys - mu)
    # kve(l, omega) = exp(omega)*K_l(omega). Keep this parameter-dependent
    # normalizer when comparing models with different l or omega.
    log_gig_normalizer = np.log(2) + np.log(special.kve(l, omega))
    if not np.isfinite(log_gig_normalizer):
        raise ValueError("GIG normalization is not finite for these l, omega")

    def log_density(log_W):
        log_W = np.asarray(log_W)
        W = np.exp(log_W)
        denom = 1 + W[..., None] * eigenvalues
        quadratic = np.sum(residual**2 / denom, axis=-1) / W
        log_likelihood = -0.5 * (n * (np.log(2*np.pi) + log_W) + log_det_C
                                 + np.log(denom).sum(axis=-1) + quadratic)
        # Normalized log p(W) + log W (Jacobian), using cosh(t)-1=2*sinh(t/2)^2.
        return (log_likelihood + l * log_W - 2 * omega * np.sinh(log_W / 2)**2
                - log_gig_normalizer)

    return K, Sigma, eigenvalues, V, residual, log_density


def log_marginal_likelihood(ys, xs, *, l=-0.5, omega=1, w=1, v_0=1,
                            k=1, mu=0, sigma_v0=1, sigma_w=0.1,
                            return_gradient=False, n_particles=64,
                            max_particles=2048, quad_tol=1e-6,
                            tail_log_drop=35, log_w_bounds=None):
    """Log p(y | X, parameters), integrating beta, f and W, with GIG normalization.

    Parameter meanings/input shapes match beta_f_inf. If return_gradient=True,
    returns (log_likelihood, gradient_dict). Gradient keys log_v_0,
    log_sigma_v0, log_k, log_w, log_sigma_w differentiate LOG positive
    parameters; mu differentiates the ordinary mean. Vector gradients have
    length D even when a scalar weight is supplied (sum them for a tied weight).
    l and omega are fixed here; their gradients are not returned.

    The gradient averages exact conditional-Gaussian derivatives over p(W|y),
    avoiding finite differences through adaptive quadrature. As with the
    likelihood, it is accurate up to quadrature/tail-truncation error.
    """
    xs, ys = _inputs(xs, "xs"), _vector(ys, "ys")
    if len(xs) != len(ys):
        raise ValueError("xs must have one row per observation in ys")
    dimension = 1 if xs.ndim == 1 else xs.shape[1]
    w, sigma_w = _check_model(l, omega, w, v_0, k, mu, sigma_v0, sigma_w, dimension)
    if (not isinstance(n_particles, (int, np.integer)) or n_particles < 8 or
            not isinstance(max_particles, (int, np.integer)) or max_particles < 2*n_particles):
        raise ValueError("Use integer n_particles >= 8 and max_particles >= 2*n_particles")
    if not np.isfinite(quad_tol) or quad_tol <= 0 or not np.isfinite(tail_log_drop) or tail_log_drop < 10:
        raise ValueError("quad_tol must be positive and tail_log_drop must be at least 10")
    K, Sigma, eigenvalues, V, residual, log_density = _training_distribution(
        ys, xs, l, omega, w, v_0, k, mu, sigma_v0, sigma_w)
    log_W, weights, diagnostics = _w_quadrature(
        log_density, np.arcsinh((l - len(ys)/2) / omega), n_particles, max_particles,
        quad_tol, tail_log_drop, log_w_bounds)
    log_likelihood = diagnostics["log_integral"]
    if not return_gradient:
        return log_likelihood
    W = np.exp(log_W)
    denom = 1 + W[:, None] * eigenvalues
    alphas = (residual / (W[:, None] * denom)) @ V.T  # rows are D_W^-1 (y-mu)
    grad_K = 0.5 * ((alphas.T * (weights * W)) @ alphas
                    - (V * (weights @ (1 / denom))) @ V.T)
    grad_Sigma = 0.5 * ((alphas.T * (weights * W**2)) @ alphas
                        - (V * ((weights * W) @ (1 / denom))) @ V.T)
    points = xs[:, None] if xs.ndim == 1 else xs
    grad_w, grad_sw = np.empty(dimension), np.empty(dimension)
    for d in range(dimension):
        squared_distance = (points[:, None, d] - points[None, :, d])**2
        grad_w[d] = -0.5 * w[d] * np.sum(grad_K * K * squared_distance)
        grad_sw[d] = -0.5 * sigma_w[d] * np.sum(grad_Sigma * Sigma * squared_distance)
    gradient = dict(log_v_0=float(np.sum(grad_K * K)),
                    log_sigma_v0=float(np.sum(grad_Sigma * Sigma)),
                    log_k=float(k * np.trace(grad_K)), log_w=grad_w,
                    log_sigma_w=grad_sw, mu=float(np.sum(weights @ alphas)))
    return log_likelihood, gradient


def beta_f_inf(ys, xs, x_pred=None, *, l=1, omega=1, w=1, v_0=1, k=1,
               mu=-1, sigma_v0=1, sigma_w=0.1, n_particles=64,
               max_particles=2048, quad_tol=1e-6, tail_log_drop=35,
               log_w_bounds=None, credible_mass=0.95):
    """Infer beta(x_pred), f(x_pred) under an independent beta ~ GP(0, Sigma).

    All kernel parameters, mu, k, l and omega are fixed, not fitted. Kernels
    and the W-scaled observation noise match Multivariate_GH(beta_prior=True).
    xs: (N, D), one input point per row, or (N,) for 1D. ys: (N,).
    x_pred: (M, D), or (M,) for 1D; defaults to xs. To use feature-by-sample
    arrays from wine.py/GH_2d.py, transpose them before calling this function.
    w and sigma_w: positive scalars or vectors of length D; both are inverse
    squared SE length scales, so w[d] = 1 / length_scale[d]**2.

    Integrating beta and f gives y | W ~ N(mu, W*(K + k*I) + W**2*Sigma).
    Conditional on W the joint posterior of [beta, f] is Gaussian. Integrate
    these Gaussians with normalized quadrature weights for t=log(W). The
    log density includes the Jacobian +t. Particle count doubles from
    n_particles until quadrature converges (at least two resolutions).
    Automatic bounds truncate tails at tail_log_drop below the peak log
    density; this is a numerical heuristic, not an exact tail-mass bound.
    Explicit log_w_bounds instead conditions the approximation on that range.

    Returns a dict with:
      W, log_W, weights: (P,) quadrature nodes and normalized probability masses.
      component_means: (P, 2*M), ordered [beta(x_pred), f(x_pred)].
      component_covariances: (P, 2*M, 2*M), in the same joint order.
      joint_mean, joint_cov: moments after marginalizing W, including the
          covariance of the conditional means (law of total covariance).
      beta_mean/cov/var/std/ci, f_mean/cov/var/std/ci: marginal summaries.
          ci has shape (M, 2): actual pointwise equal-tail mixture quantiles,
          not mean +/- Gaussian multiples of std or simultaneous bands.
      beta_f_cov: (M, M) posterior cross-covariance of beta and f.
      y_mean/cov/var/std/ci, component_y_variances: predictions for NEW noisy
          observations; add W*k within each mixture component before taking
          mixture quantiles. These include observation noise even at a training x.
      log_marginal_likelihood: fully normalized training log likelihood.
      x_pred, credible_mass, quadrature: locations and integration diagnostics.

    f is the latent function: its bands exclude new observation noise.
    Dense component storage costs O(P*M**2); W nodes are quadrature points,
    not equally weighted Monte Carlo samples.
    """
    xs, ys = _inputs(xs, "xs"), _vector(ys, "ys")
    if xs.shape[0] != ys.size:
        raise ValueError("xs must have one row per observation in ys")
    x_pred = xs.copy() if x_pred is None else _inputs(x_pred, "x_pred")
    dimension = 1 if xs.ndim == 1 else xs.shape[1]
    prediction_dimension = 1 if x_pred.ndim == 1 else x_pred.shape[1]
    if dimension != prediction_dimension:
        raise ValueError("xs and x_pred must have the same input dimension D")
    w, sigma_w = _check_model(l, omega, w, v_0, k, mu, sigma_v0, sigma_w, dimension)
    if (not isinstance(n_particles, (int, np.integer)) or n_particles < 8 or
            not isinstance(max_particles, (int, np.integer)) or max_particles < 2*n_particles):
        raise ValueError("Use integer n_particles >= 8 and max_particles >= 2*n_particles")
    if not np.isfinite(quad_tol) or quad_tol <= 0 or not np.isfinite(tail_log_drop) or tail_log_drop < 10:
        raise ValueError("quad_tol must be positive and tail_log_drop must be at least 10")
    if not 0 < credible_mass < 1:
        raise ValueError("credible_mass must lie between 0 and 1")

    n, m = xs.shape[0], x_pred.shape[0]
    _, _, eigenvalues, V, residual, log_density = _training_distribution(
        ys, xs, l, omega, w, v_0, k, mu, sigma_v0, sigma_w)

    log_W, weights, diagnostics = _w_quadrature(
        log_density, np.arcsinh((l - n/2) / omega), n_particles, max_particles,
        quad_tol, tail_log_drop, log_w_bounds)
    particles = np.exp(log_W)
    Sigma_pp = _se(x_pred, x_pred, sigma_v0, sigma_w)
    K_pp = _se(x_pred, x_pred, v_0, w)
    Sigma_pV = _se(x_pred, xs, sigma_v0, sigma_w) @ V
    K_pV = _se(x_pred, xs, v_0, w) @ V
    means = np.empty((particles.size, 2*m))
    covariances = np.empty((particles.size, 2*m, 2*m))
    for i, W in enumerate(particles):
        denom = 1 + W * eigenvalues
        cross = np.vstack((Sigma_pV, W * Sigma_pV + K_pV))
        means[i] = cross @ (residual / denom)
        means[i, m:] += mu
        prior_cov = np.block([[Sigma_pp, W * Sigma_pp],
                              [W * Sigma_pp, W**2 * Sigma_pp + W * K_pp]])
        scaled_cross = cross * np.sqrt(W / denom)
        covariance = prior_cov - scaled_cross @ scaled_cross.T
        covariances[i] = (covariance + covariance.T) / 2

    mean = weights @ means
    centered = means - mean
    covariance = np.einsum("p,pij->ij", weights, covariances) + (centered.T * weights) @ centered
    covariance = (covariance + covariance.T) / 2
    intervals = _mixture_intervals(means, np.diagonal(covariances, axis1=1, axis2=2),
                                   weights, credible_mass)
    result = {"W": particles, "log_W": log_W, "weights": weights,
              "component_means": means, "component_covariances": covariances,
              "joint_mean": mean, "joint_cov": covariance,
              "beta_f_cov": covariance[:m, m:], "x_pred": x_pred,
              "credible_mass": credible_mass, "quadrature": diagnostics,
              "log_marginal_likelihood": diagnostics["log_integral"]}
    for name, section in (("beta", slice(0, m)), ("f", slice(m, 2*m))):
        result[f"{name}_mean"] = mean[section]
        result[f"{name}_cov"] = covariance[section, section]
        result[f"{name}_var"] = np.maximum(np.diag(covariance)[section], 0)
        result[f"{name}_std"] = np.sqrt(result[f"{name}_var"])
        result[f"{name}_ci"] = intervals[section]
    y_variances = np.maximum(np.diagonal(covariances[:, m:, m:], axis1=1, axis2=2), 0) + particles[:, None] * k
    result["component_y_variances"] = y_variances
    result["y_mean"] = result["f_mean"].copy()
    result["y_cov"] = result["f_cov"] + float(weights @ particles) * k * np.eye(m)
    result["y_var"] = np.maximum(np.diag(result["y_cov"]), 0)
    result["y_std"] = np.sqrt(result["y_var"])
    result["y_ci"] = _mixture_intervals(means[:, m:], y_variances, weights, credible_mass)
    return result


def demo(seed=7):
    """Simulate beta, f, y; infer from a subset; plot both latent functions."""
    import matplotlib.pyplot as plt

    xs = np.linspace(0, 10, 181)
    parameters = dict(l=1, omega=1, v_0=0.2, w=2, k=0.05, mu=0,
                      sigma_v0=1, sigma_w=0.15)
    fs, ys, beta, W = Multivariate_GH_1d(
        xs, **parameters,b_b=0.5, beta_prior=True, return_latents=True, seed=seed)
    train_ids = np.arange(0, xs.size, 2)
    posterior = beta_f_inf(ys[train_ids], xs[train_ids], x_pred=xs, **parameters)
    print(f"True W: {W:.3f}; posterior mean W: {posterior['weights'] @ posterior['W']:.3f}")
    print(f"Quadrature: {posterior['quadrature']}")
    fig, axes = plt.subplots(2, 1, sharex=True, figsize=(9, 7))
    for ax, name, truth in zip(axes, ("beta", "f"), (beta, fs)):
        band = posterior[f"{name}_ci"]
        ax.fill_between(xs, band[:, 0], band[:, 1], alpha=0.25, label="95% pointwise credible interval")
        ax.plot(xs, truth, "k--", label=f"True {name}")
        ax.plot(xs, posterior[f"{name}_mean"], label="Posterior mean")
        ax.set_ylabel(name)
    axes[1].scatter(xs[train_ids], ys[train_ids], s=12, color="tab:red", label="Observed y")
    for ax in axes:
        ax.legend()
    axes[1].set_xlabel("x")
    fig.tight_layout()
    plt.savefig('figure/betaprior/betaf.png')
    plt.show()
    
    return posterior


if __name__ == '__main__':
    demo(seed = 5)
