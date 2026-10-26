"""Point estimation of one GP-prior function shared by independent curves.

Y[j] = c[j]*h + sqrt(s[j])*e[j], e[j]~N(0,A), h~N(0,Sigma).
GP: s=c=1; STP/GHP-mu: s=W,c=1; GHP-beta: s=W,c=W/E_prior[W].
Thus h is the population mean in every model and has the SAME GP prior.
The native shared beta is h/E_prior[W]. Gaussian h is integrated exactly
for log-coordinate theta/W MAP, then its conditional mean is plugged in.

Training grids are shared; a determinant identity reduces a (J*n)^3
calculation to two n*n Cholesky factorizations. No low-rank approximation.
"""
import numpy as np
from scipy import linalg, optimize, special

MODELS = ('gp', 'stp', 'ghp_mu', 'ghp_beta')


def se_factor(x, variance, inverse_length, *, extent=None):
    """SE feature factor and its log inverse-length derivative.

    Taylor features of exp(w*x*x') avoid forming an ill-conditioned dense
    covariance. Terms past the Poisson tail are below floating point accuracy.
    """
    z = np.sqrt(inverse_length)*(x-(x.min()+x.max())/2)
    peak = float(np.max(z*z))
    if extent is not None:
        zp = np.sqrt(inverse_length)*(np.asarray(extent)-(x.min()+x.max())/2)
        peak = max(peak, float(np.max(zp*zp)))
    count = int(np.ceil(peak+12*np.sqrt(peak+1)+40))
    if count > 4000:
        raise FloatingPointError('SE feature expansion outside numerical range')
    F = np.empty((len(x), count))
    F[:, 0] = np.sqrt(variance)*np.exp(-.5*z*z)
    for j in range(1, count):
        F[:, j] = F[:, j-1]*z/np.sqrt(j)
    if np.max(z*z) > 1000:
        powers = np.arange(count)[None, :]
        logs = special.xlogy(powers,abs(z[:,None]))-.5*special.gammaln(powers+1)
        F = np.sqrt(variance)*np.exp(-.5*z[:,None]**2+logs)*np.sign(z[:,None])**powers
    derivative = .5*F*(np.arange(count)[None, :]-z[:, None]**2)
    return F, derivative


def shared_calculation_qr(Y, x, q, model, *, lam=1., omega=1., nu=5., xp=None):
    """Same exact likelihood, with SE factors and QR for tiny noise floors."""
    J, n = Y.shape
    v, a, k, w, hw = np.exp(q[:5])
    W = np.ones(J) if model == 'gp' else np.exp(q[5:])
    c = W/gig_mean(lam, omega) if model == 'ghp_beta' else np.ones(J)
    F, D = se_factor(x, v, w)
    G, E = se_factor(x, a, hw, extent=xp)
    t = np.sum(c*c/W)
    b = np.sum((c/W)[:, None]*Y, axis=0)
    base = np.column_stack((F, np.sqrt(k)*np.eye(n)))
    LA = linalg.qr(base.T, mode='economic')[1].T
    LH = linalg.qr(np.column_stack((base, np.sqrt(t)*G)).T, mode='economic')[1].T
    solveA = lambda z: linalg.solve_triangular(LA, z, lower=True)
    solveH = lambda z: linalg.solve_triangular(LH, z, lower=True)
    centered = Y-c[:, None]*(b/t)
    whitened = solveA(centered.T)
    whitened_b = solveH(b)
    alpha = linalg.solve_triangular(LH.T, whitened_b, lower=False)
    FA, DA, GA = solveA(F), solveA(D), solveA(G)
    FH, DH, GH, EH = solveH(F), solveH(D), solveH(G), solveH(E)
    af, ad, ag, ae = whitened_b@FH, whitened_b@DH, whitened_b@GH, whitened_b@EH
    h = G@ag
    pf = whitened.T@FA+(c/t)[:, None]*af
    pd = whitened.T@DA+(c/t)[:, None]*ad
    quadratic = np.sum(whitened**2/W[None, :])+whitened_b@whitened_b/t
    logdet = 2*((J-1)*np.log(abs(LA.diagonal())).sum()+np.log(abs(LH.diagonal())).sum())+n*np.log(W).sum()
    loglike = -.5*(J*n*np.log(2*np.pi)+logdet+quadratic)
    invLA, invLH = solveA(np.eye(n)), solveH(np.eye(n))
    precision_residual = linalg.solve_triangular(LA.T, whitened, lower=False).T+(c/t)[:, None]*alpha
    grad = np.zeros_like(q)
    grad[:5] = [
        .5*(np.sum(pf*pf/W[:, None])-(J-1)*np.sum(FA*FA)-np.sum(FH*FH)),
        .5*(ag@ag-t*np.sum(GH*GH)),
        .5*k*(np.sum(precision_residual**2/W[:, None])-(J-1)*np.sum(invLA**2)-np.sum(invLH**2)),
        np.sum(pf*pd/W[:, None])-(J-1)*np.sum(FA*DA)-np.sum(FH*DH),
        ag@ae-t*np.sum(GH*EH)]
    if model != 'gp':
        trace = np.sum(GH*GH)
        Aalpha_norm = np.sum((LA.T@alpha)**2)
        quad_rows = np.sum(whitened**2, axis=0)+2*c/t*(centered@alpha)+(c/t)**2*Aalpha_norm
        grad[5:] = -.5*n+.5*(quad_rows+c*c*trace)/W
        if model == 'ghp_beta':
            ph = (whitened.T@GA+(c/t)[:, None]*ag)@ag
            grad[5:] += c*(ph-c*trace)/W
    S = G@G.T
    covariance_h = S-t*(G@GH.T)@(GH@G.T)
    result = dict(log_likelihood=float(loglike), gradient=grad, h_train=h,
                  h_cov=(covariance_h+covariance_h.T)/2, W=W, c=c)
    if xp is not None:
        # Training/prediction features must use the identical expansion center.
        z = np.sqrt(hw)*(np.asarray(xp)-(x.min()+x.max())/2)
        cross_features = np.empty((len(z), G.shape[1]))
        cross_features[:, 0] = np.sqrt(a)*np.exp(-.5*z*z)
        for j in range(1, G.shape[1]):
            cross_features[:, j] = cross_features[:, j-1]*z/np.sqrt(j)
        result['h_pred'] = cross_features@ag
    return result


def gig_mean(lam=1., omega=1.):
    return float(special.kve(lam+1, omega)/special.kve(lam, omega))


def shared_calculation(Y, x, q, model, *, lam=1., omega=1., nu=5., xp=None):
    """Collapsed log likelihood, analytic physical-log gradient, h mean.

q = log(v0, mean_GP_variance, k, w, mean_GP_w, [W_1,...,W_J]).
The h prior is already integrated, so joint MAP here does not optimize h.
"""
    J, n = Y.shape
    v, a, k, w, hw = np.exp(q[:5])
    W = np.ones(J) if model == 'gp' else np.exp(q[5:])
    c = W/gig_mean(lam, omega) if model == 'ghp_beta' else np.ones(J)
    d2 = (x[:, None]-x[None, :])**2
    K, S = v*np.exp(-.5*w*d2), a*np.exp(-.5*hw*d2)
    A = K+k*np.eye(n)
    t = np.sum(c*c/W)
    b = np.sum((c/W)[:, None]*Y, axis=0)
    H = A+t*S
    LA, LH = linalg.cholesky(A, lower=True), linalg.cholesky(H, lower=True)
    invA, invH = (linalg.cho_solve((L, True), np.eye(n)) for L in (LA, LH))
    alpha = linalg.cho_solve((LH, True), b)
    h = S@alpha
    # Project in curve space first. Using Y-c*S@alpha and then multiplying
    # by A^-1 amplifies roundoff when H is much larger than the noise floor.
    centered = Y-c[:, None]*(b/t)
    precision_centered = linalg.cho_solve((LA, True), centered.T).T
    precision_residual = precision_centered+(c/t)[:, None]*alpha
    residual = centered+(c/t)[:, None]*(A@alpha)
    quadratic = np.sum(centered*precision_centered/W[:, None])+b@alpha/t
    logdet = 2*((J-1)*np.log(LA.diagonal()).sum()+np.log(LH.diagonal()).sum())+n*np.log(W).sum()
    loglike = -.5*(J*n*np.log(2*np.pi)+logdet+quadratic)
    covariance_h = S-t*S@invH@S
    covariance_h = (covariance_h+covariance_h.T)/2
    # t*A^-1*Cov(h|Y)*A^-1 = A^-1-H^-1. Avoid amplifying
    # cancellation in the tiny posterior covariance when curves are smooth.
    scoreA = .5*((precision_residual.T/W)@precision_residual-(J-1)*invA-invH)
    scoreS = .5*(np.outer(alpha, alpha)-t*invH)
    grad = np.zeros_like(q)
    grad[:5] = [np.sum(scoreA*K), np.sum(scoreS*S), k*np.trace(scoreA),
                 np.sum(scoreA*K*(-.5*w*d2)), np.sum(scoreS*S*(-.5*hw*d2))]
    if model != 'gp':
        trace = np.sum(S*invH)  # tr(A^-1 Cov(h|Y)) = tr(Sigma H^-1)
        grad[5:] = -.5*n+.5*(np.sum(residual*precision_residual, axis=1)+c*c*trace)/W
        if model == 'ghp_beta':
            grad[5:] += c*(precision_residual@h-c*trace)/W
    result = dict(log_likelihood=float(loglike), gradient=grad, h_train=h,
                  h_cov=covariance_h, W=W, c=c)
    if xp is not None:
        cross = a*np.exp(-.5*hw*(np.asarray(xp)[:, None]-x[None, :])**2)
        result['h_pred'] = cross@alpha
    return result


def make_target(Y, x, model, *, prior_center=None, prior_sd=2., lam=1., omega=1., nu=5., backend='dense'):
    Y, x = np.asarray(Y, dtype=float), np.asarray(x, dtype=float)
    if (model not in MODELS or Y.ndim != 2 or x.ndim != 1 or Y.shape[1] != len(x)
            or not Y.size or not np.isfinite(Y).all() or not np.isfinite(x).all()):
        raise ValueError('Need finite Y[curves,points], x[points], and a valid model')
    if min(prior_sd, omega) <= 0 or nu <= 2 or not np.isfinite([prior_sd, lam, omega, nu]).all():
        raise ValueError('Invalid prior parameters')
    scale = max(float(np.var(Y)), 1e-6)
    center = np.log([.5*scale, .5*scale, .1*scale, 1., .25]) if prior_center is None else np.asarray(prior_center)
    if center.shape != (5,) or not np.isfinite(center).all():
        raise ValueError('Need five finite log prior centers')
    expected = 5 if model == 'gp' else 5+len(Y)
    normal = -5*np.log(prior_sd)-2.5*np.log(2*np.pi)
    mixing_normal = -np.log(2)-np.log(special.kve(lam, omega))+omega
    ig_shape, ig_scale = nu/2, (nu-2)/2

    def target(q):
        if np.shape(q) != (expected,) or not np.isfinite(q).all():
            return -np.inf, np.zeros(expected)
        try:
            with np.errstate(over='raise', invalid='raise', divide='raise'):
                calculation = shared_calculation_qr if backend == 'qr' else shared_calculation
                calc = calculation(Y, x, q, model, lam=lam, omega=omega, nu=nu)
                standardized = (q[:5]-center)/prior_sd
                value = calc['log_likelihood']+normal-.5*standardized@standardized
                grad = calc['gradient'].copy()
                grad[:5] -= standardized/prior_sd
                if model != 'gp':
                    W, logs = np.exp(q[5:]), q[5:]
                    if model == 'stp':
                        value += np.sum(ig_shape*np.log(ig_scale)-special.gammaln(ig_shape)-ig_shape*logs-ig_scale/W)
                        grad[5:] += -ig_shape+ig_scale/W
                    else:
                        value += np.sum(mixing_normal+lam*logs-.5*omega*(W+1/W))
                        grad[5:] += lam-.5*omega*(W-1/W)
                if not np.isfinite(value) or not np.isfinite(grad).all():
                    raise FloatingPointError('Nonfinite target')
                return float(value), grad
        except (np.linalg.LinAlgError, FloatingPointError, OverflowError, ValueError):
            return -np.inf, np.zeros(expected)
    return target, center


def fit_shared_function(Y, x, xp, model, *, starts=8, seed=50, lam=1., omega=1., nu=5.,
                        prior_center=None, prior_sd=2., backend='dense', initial_points=None,
                        refine_stationary=False, progress=None, completed=()):
    """Multistart log theta/W MAP then one conditional shared-GP mean."""
    target, center = make_target(Y, x, model, lam=lam, omega=omega, nu=nu,
                                 prior_center=prior_center, prior_sd=prior_sd, backend=backend)
    rng = np.random.default_rng(seed)
    initial = center.copy()
    if model != 'gp':
        t = np.log((nu-2)/nu) if model == 'stp' else np.arcsinh(lam/omega)
        initial = np.r_[initial, np.full(len(Y), t)]
    attempts, candidates = [], []
    previous = {item['record']['start']: item for item in completed}
    for i in range(starts):
        guess = initial.copy()
        if initial_points is not None and i < len(initial_points):
            guess = np.asarray(initial_points[i], dtype=float).copy()
            if guess.shape != initial.shape:
                raise ValueError('Initial point has incorrect number of parameters')
        elif i:
            guess[:5] += rng.normal(0, .9, 5)
            if model != 'gp':
                guess[5:] += rng.normal(0, .4, len(Y))
        if i in previous:
            item = previous[i]
            q = np.asarray(item['log_parameters'])
            record = item['record']
            value, grad = target(q)
            np.testing.assert_allclose(value, record['log_posterior'], atol=1e-6, rtol=0)
            attempts.append(record)
            if record['success']:
                assert np.max(abs(grad)) < 1e-3
                candidates.append((value, q, i))
            continue
        def objective(q):
            value, gradient = target(q)
            return -value, -gradient
        fit = optimize.minimize(objective, guess, method='L-BFGS-B', jac=True,
                                options=dict(maxiter=2500, ftol=1e-13, gtol=2e-6, maxls=40))
        value, grad = target(fit.x)
        restarts = 0
        if backend == 'qr':
            # Invalid far-away trial covariances can stop an unbounded line
            # search prematurely. Restart in a moving neighborhood; these
            # are step restrictions, not bounds on the statistical model.
            for _ in range(20):
                if np.max(abs(grad)) < 1.:
                    break
                local = optimize.minimize(objective, fit.x, method='L-BFGS-B', jac=True,
                                          bounds=list(zip(fit.x-.75, fit.x+.75)),
                                          options=dict(maxiter=1000, ftol=1e-14, gtol=2e-6,
                                                       maxls=50, maxcor=30))
                new_value, new_grad = target(local.x)
                if new_value <= value+1e-7:
                    break
                fit, value, grad = local, new_value, new_grad
                restarts += 1
        refinements = 0
        # With thousands of observations, objective roundoff may trigger ftol
        # before the gradient is small. Near a mode, refine using derivatives.
        if (backend == 'qr' or refine_stationary) and np.max(abs(grad)) < 1.:
            for _ in range(5):
                if np.max(abs(grad)) < 1e-5:
                    break
                displacements = 1e-4*np.eye(len(fit.x))
                H = np.column_stack([(target(fit.x+d)[1]-target(fit.x-d)[1])/2e-4
                                     for d in displacements])
                H = (H+H.T)/2
                if np.linalg.eigvalsh(H).max() >= 0:
                    break
                step = np.linalg.solve(H, -grad)
                proposed = fit.x+step
                new_value, new_grad = target(proposed)
                if new_value < value-1e-5 or np.max(abs(new_grad)) >= np.max(abs(grad)):
                    break
                fit.x, value, grad = proposed, new_value, new_grad
                refinements += 1
        ok = bool(fit.success and np.isfinite(value) and np.max(np.abs(grad)) < 1e-3)
        attempts.append(dict(start=i, success=ok, log_posterior=value,
                             max_abs_gradient=float(np.max(np.abs(grad))), message=str(fit.message),
                             iterations=int(fit.nit), local_restarts=restarts, newton_refinements=refinements))
        if progress:
            progress(attempts[-1], fit.x)
        if ok:
            candidates.append((value, fit.x, i))
    if not candidates:
        raise RuntimeError(f'{model}: no converged optimization: {attempts}')
    value, q, selected = max(candidates, key=lambda row: row[0])
    calculation = shared_calculation_qr if backend == 'qr' else shared_calculation
    calc = calculation(Y, x, q, model, lam=lam, omega=omega, nu=nu, xp=xp)
    # Finite differences independently check the selected stationary point.
    # A higher-order difference at a larger step balances roundoff in large
    # data log likelihoods against truncation near their sharp maximum.
    fd_step = 5e-3 if backend == 'qr' else 1e-4
    steps = fd_step*np.eye(len(q))
    fd = np.array([(-target(q-3*d)[0]+9*target(q-2*d)[0]-45*target(q-d)[0]
                    +45*target(q+d)[0]-9*target(q+2*d)[0]+target(q+3*d)[0])/(60*fd_step)
                   for d in steps])
    np.testing.assert_allclose(fd, target(q)[1], atol=1e-4 if backend == 'qr' else 3e-5, rtol=2e-3)
    steps = 1e-4*np.eye(len(q))
    hessian = np.column_stack([(target(q+d)[1]-target(q-d)[1])/(2e-4) for d in steps])
    maximum_eigenvalue = float(np.linalg.eigvalsh((hessian+hessian.T)/2).max())
    if maximum_eigenvalue > 1e-5:
        raise RuntimeError(f'{model}: selected point is not a local maximum')
    return dict(model=model, log_parameters=q, population_mean=calc['h_pred'],
                function=calc['h_pred']/gig_mean(lam, omega) if model == 'ghp_beta' else calc['h_pred'],
                function_kind='beta' if model == 'ghp_beta' else 'mu',
                prior_center=center, prior_sd=prior_sd, lam=lam, omega=omega, nu=nu,
                numerical_backend=backend,
                optimizer_version=2,
                attempts=attempts, selected_start=selected, log_posterior=value,
                max_hessian_eigenvalue=maximum_eigenvalue,
                max_gradient_check_error=float(np.max(np.abs(fd-target(q)[1]))),
                W_training_mode=calc['W'],
                estimation='MAP in log theta and log W; shared function marginalized, then conditional mean fixed')
