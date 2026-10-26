"""Exact conditional prediction with fixed kernels and fixed beta/mu functions.

GIG convention: p(W) proportional to W**(lambda-1)*exp(-(chi/W+psi*W)/2).
Given W, y has mean mu+W*beta and covariance W*(K+kI).
GHP predictions integrate W analytically; the GP has W=1, and the STP
with fixed mu and beta=0 has inverse-gamma mixing and Student-t prediction.
Numerical quadrature supplies checks, CRPS, and GH quantiles when SciPy's
unscaled Bessel implementation underflows; predictive moments stay analytic.
"""
import numpy as np
from scipy import linalg, special, stats

from GH_proc_joint_hmc import _inputs_1d
from GH_proc_w_betaprior import _w_quadrature


def log_scaled_bessel_k(order, argument):
    """log(exp(x)*K_order(x)), including orders that overflow scipy.kve.

    K is even in its order. Reduce to fractional orders and use the positive
    recurrence K_(v+1)=K_(v-1)+2*v/x*K_v, evaluated with logaddexp.
    """
    x = np.asarray(argument, dtype=float)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        result = np.log(special.kve(order, x))
    if np.all(np.isfinite(result)):
        return result
    if np.any(~np.isfinite(x)) or np.any(x <= 0) or not np.isfinite(order):
        raise ValueError('Need positive finite arguments and finite Bessel order')
    v = abs(float(order))
    integer = int(np.floor(v))
    fraction = v-integer
    low = np.log(special.kve(fraction, x))
    if integer == 0:
        return low
    high = np.log(special.kve(fraction+1, x))
    for j in range(1, integer):
        low, high = high, np.logaddexp(low, np.log(2*(fraction+j)/x)+high)
    return np.where(np.isfinite(result), result, high)


def gig_moments(p, chi, psi):
    """Mean and variance of GIG, with stable ratios at large negative order."""
    argument = np.sqrt(chi*psi)
    with np.errstate(over='ignore', invalid='ignore', divide='ignore'):
        denominator = special.kve(p, argument)
        r1 = special.kve(p+1, argument)/denominator
        r2 = special.kve(p+2, argument)/special.kve(p+1, argument)
    if not np.isfinite([r1,r2]).all():
        if p <= -2:
            # Successive positive-order ratios avoid subtracting huge logs.
            v = -p
            integer = int(np.floor(v))
            fraction = v-integer
            ratio = special.kve(fraction+1,argument)/special.kve(fraction,argument)
            for j in range(1,integer):
                previous, ratio = ratio, 2*(fraction+j)/argument+1/ratio
            r1, r2 = 1/ratio, 1/previous
        else:
            r1 = np.exp(log_scaled_bessel_k(p+1,argument)-log_scaled_bessel_k(p,argument))
            r2 = np.exp(log_scaled_bessel_k(p+2,argument)-log_scaled_bessel_k(p+1,argument))
    mean = np.sqrt(chi/psi)*r1
    variance = chi/psi*r1*(r2-r1)
    return float(mean), float(variance)


class StableGHMarginals:
    """GH marginals when SciPy's unscaled Bessel moments underflow.

    Density uses the analytic GH formula with scaled Bessel functions;
    CDF/quantiles use converged one-dimensional Gaussian-mixture quadrature.
    """
    def __init__(self, prediction, name):
        self.prediction, self.name = prediction, name
        self.W, self.weights, self.diagnostics = mixing_quadrature(prediction)
        self.means = prediction['location']+self.W[:, None]*prediction['conditional_beta']
        self.variances = self.W[:, None]*np.diag(prediction[f'{name}_scale_cov'])
        self.stds = np.sqrt(self.variances)

    def mean(self):
        return self.prediction[f'{self.name}_mean']

    def var(self):
        return self.prediction[f'{self.name}_var']

    def logpdf(self, values):
        p = self.prediction['W_lambda']
        chi, psi = self.prediction['W_chi'], self.prediction['W_psi']
        c = np.diag(self.prediction[f'{self.name}_scale_cov'])
        beta = self.prediction['conditional_beta']
        delta = np.asarray(values)-self.prediction['location']
        Q, R = chi+delta*delta/c, psi+beta*beta/c
        argument, radius = np.sqrt(chi*psi), np.sqrt(Q*R)
        linear = argument+beta*delta/c
        # Rationalize linear-radius to avoid subtracting large similar terms.
        numerator = (np.sqrt(chi)*beta-np.sqrt(psi)*delta)**2/c
        exponent = np.where(linear >= 0, -numerator/(radius+linear), linear-radius)
        return (.5*p*np.log(psi/chi)-.5*np.log(2*np.pi*c)
                +.5*(p-.5)*np.log(Q/R)+log_scaled_bessel_k(p-.5,radius)
                -log_scaled_bessel_k(p,argument)+exponent)

    def cdf(self, values):
        return self.weights@special.ndtr((np.asarray(values)-self.means)/self.stds)

    def pdf(self, values):
        return np.exp(self.logpdf(values))

    def ppf(self, probability):
        quantiles = self.means+special.ndtri(probability)*self.stds
        lo, hi = quantiles.min(axis=0), quantiles.max(axis=0)
        for _ in range(60):
            mid = (lo+hi)/2
            below = self.cdf(mid) < probability
            lo, hi = np.where(below,mid,lo), np.where(below,hi,mid)
        return (lo+hi)/2

    def stats(self, moments='s'):
        if moments != 's':
            raise ValueError('Only skewness is provided; use mean() or var() for moments')
        delta = self.means-self.mean()
        return (self.weights@(delta**3+3*delta*self.variances))/self.var()**1.5


def fixed_function_prediction(y, x, xp, *, v0, k, w, mixing='gig',
                              mu_train=None, mu_pred=None,
                              beta_train=None, beta_pred=None,
                              lam=1., omega=1., nu=5., gh_backend='auto'):
    """Return exact moments and equal-tail 95% marginal predictive intervals.

All function values are supplied point estimates. No uncertainty in them or
the kernel parameters is included. The posterior uncertainty in W IS kept.
f is latent; y predictions refer to fresh independent observation noise.
"""
    # Empty context is meaningful for predicting a wholly new curve.
    y, x = np.asarray(y, dtype=float), np.asarray(x, dtype=float)
    if y.size or x.size:
        y, x = _inputs_1d(y, x)
    elif y.shape != (0,) or x.shape != (0,):
        raise ValueError('Empty context must use two 1D empty arrays')
    xp = np.asarray(xp, dtype=float)
    if xp.ndim != 1 or not xp.size or not np.all(np.isfinite(xp)):
        raise ValueError('Need finite 1D prediction inputs')
    if mixing not in ('gp', 'gig', 'ig'):
        raise ValueError('mixing must be gp, gig or ig')
    if gh_backend not in ('auto', 'quadrature'):
        raise ValueError('gh_backend must be auto or quadrature')
    if not np.all(np.isfinite([v0, k, w])) or min(v0, k, w) <= 0:
        raise ValueError('Kernel variance, noise and inverse length squared must be positive')

    def vector(value, size):
        result = np.zeros(size) if value is None else np.asarray(value, dtype=float)
        if result.shape != (size,) or not np.all(np.isfinite(result)):
            raise ValueError('Function estimates must be matching finite vectors')
        return result

    mu1, mu2 = vector(mu_train, len(x)), vector(mu_pred, len(xp))
    beta1, beta2 = vector(beta_train, len(x)), vector(beta_pred, len(xp))
    if mixing == 'ig' and (np.any(beta1) or np.any(beta2)):
        raise ValueError('The Student-t case requires beta=0')

    def kernel(a, b):
        return v0*np.exp(-.5*w*(a[:, None]-b[None, :])**2)

    A = kernel(x, x)+k*np.eye(len(x))
    cross = kernel(xp, x)
    residual = y-mu1
    if len(x):
        L = linalg.cholesky(A, lower=True)
        ar = linalg.cho_solve((L, True), residual)
        ab = linalg.cho_solve((L, True), beta1)
        solved = linalg.solve_triangular(L, cross.T, lower=True)
    else:
        ar, ab = np.empty(0), np.empty(0)
        solved = np.empty((0, len(xp)))
    m = mu2+cross@ar
    b = beta2-cross@ab
    S = kernel(xp, xp)-solved.T@solved
    S = (S+S.T)/2
    result = dict(mixing=mixing, location=m, conditional_beta=b,
                  f_scale_cov=S, y_scale_cov=S+k*np.eye(len(xp)))
    if mixing == 'gig' and gh_backend == 'quadrature':
        # Vectorized mixture CDF inversion avoids repeated scalar GH integrals.
        # The density and moments remain analytic; quadrature is convergence checked.
        result['gh_backend'] = gh_backend
    if mixing == 'gig':
        if not np.all(np.isfinite([lam, omega])) or omega <= 0:
            raise ValueError('Need finite lambda and positive omega')
        p = lam-len(x)/2
        chi, psi = omega+residual@ar, omega+beta1@ab
        ew, varw = gig_moments(p, chi, psi)
        result.update(W_lambda=p, W_chi=chi, W_psi=psi)
    elif mixing == 'ig':
        if not np.isfinite(nu) or nu <= 2:
            raise ValueError('Need nu > 2')
        shape, scale = (nu+len(x))/2, (nu-2+residual@ar)/2
        if shape <= 2:
            raise ValueError('This routine reports Var(W), requiring posterior shape > 2')
        ew = scale/(shape-1)
        varw = scale**2/((shape-1)**2*(shape-2))
        result.update(W_shape=shape, W_scale=scale)
    else:
        ew, varw = 1., 0.
    if not np.all(np.isfinite([ew, varw])) or ew <= 0 or varw < 0:
        raise FloatingPointError('Mixing moments could not be evaluated accurately')
    result.update(W_mean=float(ew), W_var=float(varw))
    for name in ('f', 'y'):
        covariance = ew*result[f'{name}_scale_cov']+varw*np.outer(b, b)
        result[f'{name}_mean'] = m+ew*b
        result[f'{name}_cov'] = covariance
        result[f'{name}_var'] = np.diag(covariance).copy()
        distribution = marginal_distribution(result, name)
        result[f'{name}_ci'] = np.column_stack([distribution.ppf(.025), distribution.ppf(.975)])
    return result


def marginal_distribution(prediction, name='f'):
    """SciPy vector of pointwise GH, Student-t or Normal distributions.

For a scalar X|W~N(m+bW,cW), the SciPy GH parameters are
scale=sqrt(chi*c), b_scipy=b*sqrt(chi/c), a=sqrt(chi*(psi+b*b/c)).
"""
    m, b = prediction['location'], prediction['conditional_beta']
    c = np.diag(prediction[f'{name}_scale_cov'])
    if np.any(c <= 0):
        raise ValueError('Marginal prediction requires strictly positive residual variance')
    if prediction['mixing'] == 'gig':
        chi, psi = prediction['W_chi'], prediction['W_psi']
        # SciPy's third/fourth moment expressions multiply raw Bessel values
        # and reciprocal powers, which overflow before individual K_nu does.
        if prediction.get('gh_backend', 'auto') == 'quadrature' or np.sqrt(chi*psi) > 100:
            return StableGHMarginals(prediction, name)
        return stats.genhyperbolic(prediction['W_lambda'],
                                  np.sqrt(chi*(psi+b*b/c)), b*np.sqrt(chi/c),
                                  loc=m, scale=np.sqrt(chi*c))
    if prediction['mixing'] == 'ig':
        shape, scale = prediction['W_shape'], prediction['W_scale']
        return stats.t(2*shape, loc=m, scale=np.sqrt(scale/shape*c))
    return stats.norm(loc=m+b, scale=np.sqrt(c))


def mixing_quadrature(prediction):
    """Normalized nodes for CRPS/checks and numerically stable GH quantiles."""
    if prediction['mixing'] == 'gp':
        return np.ones(1), np.ones(1), dict(n_particles=1)
    if prediction['mixing'] == 'gig':
        p, chi, psi = (prediction[f'W_{key}'] for key in ('lambda', 'chi', 'psi'))
        argument = np.sqrt(chi*psi)
        mid = .5*np.log(chi/psi)
        norm = -np.log(2)-log_scaled_bessel_k(p, argument)
        def log_density(t):
            u = t-mid
            return norm+p*u-2*argument*np.sinh(u/2)**2
        center = np.log(np.sqrt(chi/psi))+np.arcsinh(p/argument)
    else:
        a, scale = prediction['W_shape'], prediction['W_scale']
        def log_density(t):
            return a*np.log(scale)-special.gammaln(a)-a*t-scale*np.exp(-t)
        center = np.log(scale/a)
    minimum = int(prediction.get('quadrature_min_particles', 64))
    tail_drop = float(prediction.get('quadrature_tail_log_drop', 60.))
    t, weights, diagnostics = _w_quadrature(log_density, center, minimum, 2048, 1e-11, tail_drop, None)
    return np.exp(t), weights, diagnostics


def refine_density_quadrature(prediction, truth, name='y'):
    """Extend GH integration tails/resolution for an independent density check.

    Moment convergence alone need not resolve a very unlikely observation's
    density: it can depend on W values with tiny posterior mass. This only
    changes numerical integration settings, not the fitted predictive law.
    Analytic GH log densities provide the convergence reference.
    """
    if prediction['mixing'] != 'gig':
        return
    dist = StableGHMarginals(prediction, name)
    truth = np.asarray(truth)
    analytic = dist.logpdf(truth)
    # A units-invariant starting extent, followed by explicit convergence tests.
    drop = max(60., 30.+float(np.max(dist.logpdf(dist.mean())-analytic)))
    for expansion in range(4):
        prediction['quadrature_tail_log_drop'] = drop*2**expansion
        for minimum in (128, 256, 512, 1024):
            prediction['quadrature_min_particles'] = minimum
            W, weights, _ = mixing_quadrature(prediction)
            mean = prediction['location']+W[:, None]*prediction['conditional_beta']
            sd = np.sqrt(W[:, None]*np.diag(prediction[f'{name}_scale_cov']))
            positive = weights > 0
            numerical = special.logsumexp(np.log(weights[positive, None])+
                stats.norm.logpdf(truth, mean[positive], sd[positive]), axis=0)
            density_ok = np.all(np.abs(numerical-analytic) <= 1e-7+1e-9*np.abs(analytic))
            intervals_ok = True
            for target in ('f', 'y'):
                if target+'_ci' not in prediction:
                    continue
                refined = StableGHMarginals(prediction, target)
                old = prediction[target+'_ci']
                discrepancy = max(np.max(abs(refined.cdf(old[:, 0])-.025)),
                                  np.max(abs(refined.cdf(old[:, 1])-.975)))
                intervals_ok &= discrepancy < 1e-8
                prediction[target+'_ci'] = np.column_stack([refined.ppf(.025), refined.ppf(.975)])
            if density_ok and intervals_ok:
                return
    raise RuntimeError('GH density quadrature did not agree with analytic density')
