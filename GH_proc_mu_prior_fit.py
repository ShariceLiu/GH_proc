"""MAP fitting for beta=0 with an independent GP mean mu(x)."""

from GH_proc_mu_prior import mu_log_marginal_likelihood
from GH_proc_w_betaprior_fit import _fit_kernel_parameters


def _likelihood_adapter(ys, xs, *, mu, sigma_v0, sigma_w, return_gradient=False, **kwargs):
    """Translate the shared optimizer's coordinates to the GP-mean model."""
    answer = mu_log_marginal_likelihood(
        ys, xs, mu0=mu, mu_v0=sigma_v0, mu_w=sigma_w,
        return_gradient=return_gradient, **kwargs)
    if not return_gradient:
        return answer
    value, gradient = answer
    names = {'log_mu_v0': 'log_sigma_v0', 'log_mu_w': 'log_sigma_w', 'mu0': 'mu'}
    return value, {names.get(key, key): val for key, val in gradient.items()}


def fit_mu_kernel_parameters(ys, xs, *, l=1, omega=1, mu0=0, fit_mu0=False,
                              initial=None, prior_centers=None, log_prior_sd=2.0,
                              n_restarts=3, maxiter=200, seed=0,
                              log_bounds=(-10, 10), quad_options=None, verbose=False):
    """Fit v_0, w[D], k, mu_v0, mu_w[D] by MAP in log parameters.

    mu(x)~GP(mu0,M), independently of W; beta=0. Both SE covariance matrices
    and noise are learned. l/omega and (by default) the prior mean mu0 remain
    fixed. fit_mu0=True additionally estimates the constant GP prior mean.

    Priors and optimization match fit_kernel_parameters: independent Normal
    priors on log positive parameters, default sd=2, bounded L-BFGS-B and
    multiple starts. Prior centers use training variance s2: v_0=mu_v0=.5*s2,
    k=.1*s2, w[d]=1/D, mu_w[d]=.25/D. These are empirical priors. initial and
    prior_centers accept these native mu-model parameter names independently.
    Inputs/outputs are used in their supplied units without preprocessing.

    Returns params suitable for mu_f_inf(), convergence/restart diagnostics,
    prior settings and quadrature controls. W and both latent GPs are
    marginalized in the fitting likelihood. Predictions plugging in params
    subsequently condition on this fitted hyperparameter vector.
    """
    to_shared = {'mu0': 'mu', 'mu_v0': 'sigma_v0', 'mu_w': 'sigma_w'}

    def translate(values, name):
        if values is None:
            return None
        allowed = {'v_0', 'w', 'k', 'mu_v0', 'mu_w'} | ({'mu0'} if fit_mu0 else set())
        if set(values)-allowed:
            raise ValueError(f'{name} may only override fitted parameters: {sorted(allowed)}')
        return {to_shared.get(key, key): value for key, value in values.items()}

    result = _fit_kernel_parameters(
        ys, xs, likelihood=_likelihood_adapter, l=l, omega=omega, mu=mu0,
        fit_mu=fit_mu0, initial=translate(initial, 'initial'),
        prior_centers=translate(prior_centers, 'prior_centers'),
        log_prior_sd=log_prior_sd, n_restarts=n_restarts, maxiter=maxiter, seed=seed,
        log_bounds=log_bounds, quad_options=quad_options, verbose=verbose)
    native = {value: key for key, value in to_shared.items()}
    result['params'] = {native.get(key, key): value for key, value in result['params'].items()}
    result['boundary_parameters'] = [key.replace('sigma_v0', 'mu_v0').replace('sigma_w', 'mu_w')
                                      for key in result['boundary_parameters']]
    result['prior']['fit_mu0'] = result['prior'].pop('fit_mu')
    result['prior']['vector_order'] = '[log_v_0, log_mu_v0, log_k, log_w[D], log_mu_w[D], optional mu0]'
    return result
