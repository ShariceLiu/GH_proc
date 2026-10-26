"""Learn GIG/IG parameters as well as the two SE kernels and noise.

GHP beta: C_y(W)=W*(K1+kI)+W**2*K2; STP+GP mean: W*(K1+kI)+K2.
GPs are integrated analytically. MAP integrates W by adaptive log-W
quadrature; HMC samples W jointly in effective log-amplitude coordinates.
Priors are specified on log kernel parameters, lambda, log(omega), and
log(nu-2). Only log(W) needs an additional Jacobian. The effective-coordinate
map is a unit-determinant linear map of these log coordinates.

GIG normalization follows scipy.stats.geninvgauss(p=lambda,b=omega,scale=1).
IG uses scipy.stats.invgamma(a=nu/2,scale=(nu-2)/2), giving E[W]=1.
"""
import numpy as np
from scipy import linalg, special

from GH_proc_w_betaprior import _w_quadrature
from GH_proc_joint_hmc import effective_coordinates, physical_coordinates


MODELS = ('beta_gp', 'tp_mu_gp')


def mixing_prior(h, model):
    """Normalized density and gradient in lambda/log-omega or log(nu-2)."""
    center = np.array([1., 0.]) if model == 'beta_gp' else np.array([np.log(3.)])
    sd = np.array([2., 1.]) if model == 'beta_gp' else np.array([1.])
    z = (np.asarray(h)-center)/sd
    return float(-.5*z@z-np.log(sd).sum()-.5*len(z)*np.log(2*np.pi)), -z/sd


def mixing_log_density(t, h, model, gradient=False):
    """log p(t|h), t=log W, including dW/dt; optional derivatives (t,h)."""
    t = np.asarray(t, dtype=float)
    W = np.exp(t)
    if model == 'beta_gp':
        lam, omega = h[0], np.exp(h[1])
        scaled = special.kve(lam, omega)
        if not np.isfinite(scaled) or scaled <= 0:
            raise FloatingPointError('Nonfinite GIG normalizer')
        # -omega*cosh(t)+omega avoids subtracting large equal terms near t=0.
        value = lam*t-2*omega*np.sinh(t/2)**2-np.log(2.)-np.log(scaled)
        if not gradient:
            return value
        eps = 1e-4*max(1., abs(lam))
        orders = lam+eps*np.array([-2., -1., 1., 2.])
        logs = np.log(special.kve(orders, omega))
        d_order = (logs[0]-8*logs[1]+8*logs[2]-logs[3])/(12*eps)
        ratio = (special.kve(lam-1, omega)+special.kve(lam+1, omega))/scaled
        dh = np.stack((t-d_order, .5*omega*(ratio-W-1/W)), axis=-1)
        return value, lam-.5*omega*(W-1/W), dh
    if model == 'tp_mu_gp':
        excess = np.exp(h[0])
        a, b = 1.+excess/2, excess/2
        value = a*np.log(b)-special.gammaln(a)-a*t-b/W
        if not gradient:
            return value
        dh = (.5*excess*(np.log(b)+a/b-special.digamma(a)-t-1/W))[..., None]
        return value, -a+b/W, dh
    raise ValueError('Unknown model')


class LearnedMixingTarget:
    def __init__(self, y, x, model, prior_center=None, prior_sd=2.):
        self.y, self.x = np.asarray(y, float), np.asarray(x, float)
        if model not in MODELS or self.y.ndim != 1 or self.x.shape != self.y.shape:
            raise ValueError('Expected matching 1D x/y and supported model')
        if not np.isfinite(self.y).all() or not np.isfinite(self.x).all():
            raise ValueError('Nonfinite training data')
        self.model, self.power = model, (2 if model == 'beta_gp' else 0)
        self.hdim = 2 if model == 'beta_gp' else 1
        self.dimension = 6+self.hdim
        scale = max(float(np.var(self.y)), 1e-12)
        self.center = np.log([.5*scale, .5*scale, .1*scale, 1., .25]) if prior_center is None else np.asarray(prior_center)
        self.sd = np.broadcast_to(prior_sd, (5,)).copy()
        self.dist = (self.x[:, None]-self.x)**2
        self.eye = np.eye(len(self.x))
        self.constant = -.5*len(self.x)*np.log(2*np.pi)

    def kernel_prior(self, q):
        z = (q-self.center)/self.sd
        return float(-.5*z@z-np.log(self.sd).sum()-2.5*np.log(2*np.pi)), -z/self.sd

    def initial_h(self):
        return np.array([1., 0.]) if self.model == 'beta_gp' else np.array([np.log(3.)])

    def initial(self):
        t = np.arcsinh(1.) if self.model == 'beta_gp' else np.log(.6)
        return np.r_[effective_coordinates(np.r_[self.center, t], self.model), self.initial_h()]

    def joint(self, z):
        """Normalized log posterior (up to evidence), gradient in effective z."""
        z = np.asarray(z, float)
        if z.shape != (self.dimension,) or not np.isfinite(z).all():
            return -np.inf, np.zeros(self.dimension)
        try:
            with np.errstate(over='raise', divide='raise', invalid='raise'):
                A, B, N, w, sw, W = np.exp(z[:6])
                K = A*np.exp(-.5*w*self.dist)
                H = B*np.exp(-.5*sw*self.dist)
                L = linalg.cholesky(K+H+N*self.eye, lower=True, check_finite=False)
                alpha = linalg.cho_solve((L, True), self.y, check_finite=False)
                like = self.constant-np.log(L.diagonal()).sum()-.5*self.y@alpha
                score = .5*(np.outer(alpha, alpha)-linalg.cho_solve((L, True), self.eye, check_finite=False))
                q = physical_coordinates(z[:6], self.model)
                pk, gk = self.kernel_prior(q[:5])
                ph, gh = mixing_prior(z[6:], self.model)
                pw, gt, gmix = mixing_log_density(z[5], z[6:], self.model, True)
                grad = np.r_[np.sum(score*K), np.sum(score*H), N*np.trace(score),
                             np.sum(score*K*(-.5*w*self.dist)),
                             np.sum(score*H*(-.5*sw*self.dist)),
                             gt-gk[0]-self.power*gk[1]-gk[2], gmix+gh]
                grad[:5] += gk
                value = float(like+pk+ph+pw)
                if not np.isfinite(value) or not np.isfinite(grad).all():
                    raise FloatingPointError('Nonfinite target')
                return value, grad
        except (FloatingPointError, OverflowError, ValueError, np.linalg.LinAlgError):
            return -np.inf, np.zeros(self.dimension)

    def quadrature(self, u, *, tol=1e-8, tail=40.):
        """p(t|y,hyperparameters), using a generalized eigendecomposition."""
        q, h = np.asarray(u[:5]), np.asarray(u[5:])
        v, a, k, w, sw = np.exp(q)
        K = v*np.exp(-.5*w*self.dist)
        H = a*np.exp(-.5*sw*self.dist)
        C = K+k*self.eye
        e, V = linalg.eigh(H, C, check_finite=False)
        e = np.maximum(e, 0.)
        residual = V.T@self.y
        logdet = 2*np.log(linalg.cholesky(C, lower=True).diagonal()).sum()

        def density(t):
            t = np.asarray(t)
            W = np.exp(t)
            d = W[..., None]+W[..., None]**self.power*e
            return (self.constant-.5*logdet-.5*np.log(d).sum(-1)
                    -.5*(residual**2/d).sum(-1)+mixing_log_density(t, h, self.model))
        center = np.arcsinh(h[0]/np.exp(h[1])) if self.model == 'beta_gp' else np.log(np.exp(h[0])/(np.exp(h[0])+2))
        t, weights, info = _w_quadrature(density, center, 48, 1536, tol, tail, None)
        return t, weights, info, (K, H, e, V, residual)

    def marginal(self, u, *, tol=1e-8, tail=40.):
        """Log p(log theta,h|y), W integrated; gradients averaged over W."""
        u = np.asarray(u, float)
        try:
            with np.errstate(over='raise', divide='raise', invalid='raise'):
                t, weights, info, (K, H, e, V, residual) = self.quadrature(u, tol=tol, tail=tail)
                W = np.exp(t)
                d = W[:, None]+W[:, None]**self.power*e
                alpha = (residual/d)@V.T
                scores = []
                for factor in (W, W**self.power):
                    weighted = weights*factor
                    scores.append(.5*((alpha.T*weighted)@alpha-(V*((weighted[:, None]/d).sum(0)))@V.T))
                Cscore, Hscore = scores
                v, a, k, w, sw = np.exp(u[:5])
                g = np.array([np.sum(Cscore*K), np.sum(Hscore*H), k*np.trace(Cscore),
                              np.sum(Cscore*K*(-.5*w*self.dist)), np.sum(Hscore*H*(-.5*sw*self.dist))])
                pk, gk = self.kernel_prior(u[:5])
                ph, gh = mixing_prior(u[5:], self.model)
                _, _, gm = mixing_log_density(t, u[5:], self.model, True)
                value = float(info['log_integral']+pk+ph)
                grad = np.r_[g+gk, weights@gm+gh]
                if not np.isfinite(value) or not np.isfinite(grad).all():
                    raise FloatingPointError('Nonfinite marginal target')
                return value, grad
        except (FloatingPointError, OverflowError, ValueError, RuntimeError, np.linalg.LinAlgError):
            return -np.inf, np.zeros(5+self.hdim)

    def physical_samples(self, z):
        z = np.asarray(z)
        q = physical_coordinates(z[..., :6], self.model)
        kernel = np.exp(q)
        extra = np.stack([z[..., 6], np.exp(z[..., 7])], -1) if self.model == 'beta_gp' else (2+np.exp(z[..., 6]))[..., None]
        return np.concatenate([kernel, extra], -1)

    @property
    def parameter_names(self):
        latent = 'sigma' if self.model == 'beta_gp' else 'mu'
        return ['v_0', latent+'_v0', 'k', 'w', latent+'_w', 'W']+(['lambda', 'omega'] if self.model == 'beta_gp' else ['nu'])

    def predict_components(self, z, xp):
        """Conditional marginal Gaussians; no extra latent-function sampling."""
        z = np.atleast_2d(z)
        xp = np.asarray(xp)
        dist = (xp[:, None]-self.x)**2
        means, variances, effects, effect_vars = [np.empty((len(z), len(xp))) for _ in range(4)]
        noises = np.empty(len(z))
        for i, state in enumerate(z):
            A, B, N, w, sw, W = np.exp(state[:6])
            L = linalg.cholesky(A*np.exp(-.5*w*self.dist)+B*np.exp(-.5*sw*self.dist)+N*self.eye, lower=True, check_finite=False)
            alpha = linalg.cho_solve((L, True), self.y, check_finite=False)
            Hpx = B*np.exp(-.5*sw*dist)
            Tpx = A*np.exp(-.5*w*dist)+Hpx
            solved = linalg.solve_triangular(L, Tpx.T, lower=True, check_finite=False)
            means[i] = Tpx@alpha
            variances[i] = A+B-np.square(solved).sum(0)
            solved_h = linalg.solve_triangular(L, Hpx.T, lower=True, check_finite=False)
            factor = W if self.model == 'beta_gp' else 1.
            effects[i] = Hpx@alpha/factor
            effect_vars[i] = (B-np.square(solved_h).sum(0))/factor**2
            noises[i] = N
        if variances.min() < -1e-7 or effect_vars.min() < -1e-7:
            raise ArithmeticError('Negative conditional variance')
        return dict(component_means=means, component_f_variances=np.maximum(variances, 1e-14),
                    component_y_variances=np.maximum(variances, 1e-14)+noises[:, None],
                    component_native_means=effects, component_native_variances=np.maximum(effect_vars, 1e-14))
