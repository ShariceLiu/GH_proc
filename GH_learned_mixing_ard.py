"""Marginal hyperparameter MAP for two-SE GP, GHP and STP+GP-mean.

The Gaussian processes are integrated analytically; scalar W is integrated by
adaptive quadrature. Priors and the MAP reference measure match the 1D learned
mixing experiment: Normal priors on log kernel parameters, lambda/log omega,
or log(nu-2). Prediction retains W and latent-function uncertainty.
"""
import numpy as np
from scipy import linalg

from GH_learned_mixing import mixing_prior, mixing_log_density
from GH_proc_w_betaprior import _w_quadrature
from GH_tabular_hmc import inputs, default_center, squared_distances, parameter_names

MODELS = ('gp', 'tp_mu_gp', 'mu_gp', 'beta_gp')


class ARDLearnedMixingMAP:
    def __init__(self, y, x, model, prior_center=None, prior_sd=2.):
        self.y, self.x = inputs(y, x)
        if model not in MODELS:
            raise ValueError('Expected gp, tp_mu_gp, mu_gp or beta_gp')
        self.model, self.power = model, 2 if model == 'beta_gp' else 0
        self.has_mixing = model != 'gp'
        # GHP-mu has the same GIG density as GHP-beta, but a W^0 mean kernel.
        self.mixing_model = 'beta_gp' if model in ('mu_gp', 'beta_gp') else model
        self.d = self.x.shape[1]
        self.qdim = 3+2*self.d
        self.hdim = 0 if model == 'gp' else (2 if self.mixing_model == 'beta_gp' else 1)
        self.center = (default_center(self.y, self.d) if prior_center is None
                       else np.asarray(prior_center, float))
        self.sd = np.broadcast_to(prior_sd, (self.qdim,)).copy()
        if (self.center.shape != (self.qdim,) or not np.isfinite(self.center).all()
                or not np.isfinite(self.sd).all() or np.any(self.sd <= 0)):
            raise ValueError('Invalid kernel prior')
        self.dist = squared_distances(self.x, self.x).reshape(self.d, -1)
        self.eye = np.eye(len(self.x))
        self.constant = -.5*len(self.x)*np.log(2*np.pi)

    def initial_h(self):
        if not self.has_mixing:
            return np.empty(0)
        return np.array([1., 0.]) if self.mixing_model == 'beta_gp' else np.array([np.log(3.)])

    @property
    def parameter_names(self):
        if not self.has_mixing:
            return parameter_names(self.d, self.model)
        return parameter_names(self.d, self.model)[:-1]+(['lambda', 'omega']
                    if self.mixing_model == 'beta_gp' else ['nu'])

    def kernel_prior(self, q):
        z = (q-self.center)/self.sd
        return float(-.5*z@z-np.log(self.sd).sum()-.5*self.qdim*np.log(2*np.pi)), -z/self.sd

    def kernels(self, q, distances, shape):
        v, a = np.exp(q[:2])
        w, sw = np.exp(q[3:3+self.d]), np.exp(q[3+self.d:self.qdim])
        return (v*np.exp(-.5*(w@distances))).reshape(shape), (a*np.exp(-.5*(sw@distances))).reshape(shape)

    def quadrature(self, u, *, tol=1e-8, tail=40.):
        u = np.asarray(u, float)
        if u.shape != (self.qdim+self.hdim,) or not np.isfinite(u).all():
            raise ValueError('Invalid hyperparameters')
        q, h = u[:self.qdim], u[self.qdim:]
        K, H = self.kernels(q, self.dist, self.eye.shape)
        C = K+np.exp(q[2])*self.eye
        e, V = linalg.eigh(H, C, check_finite=False)
        if e.min() < -1e-8*max(1., e.max()):
            raise ArithmeticError('Negative generalized eigenvalue beyond roundoff')
        e = np.maximum(e, 0.)
        residual = V.T@self.y
        logdet = 2*np.log(linalg.cholesky(C, lower=True, check_finite=False).diagonal()).sum()

        def likelihood(t):
            W = np.exp(np.asarray(t))
            d = W[..., None]+W[..., None]**self.power*e
            return (self.constant-.5*logdet-.5*np.log(d).sum(-1)
                    -.5*(residual**2/d).sum(-1))
        if not self.has_mixing:
            # A single W=1 component is a computational convention for GP;
            # it has no mixing prior, inferred W, or mixing hyperparameters.
            return (np.array([0.]), np.array([1.]),
                    dict(log_integral=float(likelihood(0.)), n_particles=1, method='exact GP, W=1'),
                    (K, H, e, V, residual))
        def density(t):
            return likelihood(t)+mixing_log_density(t, h, self.mixing_model)
        center = (np.arcsinh(h[0]/np.exp(h[1])) if self.mixing_model == 'beta_gp'
                  else h[0]-np.logaddexp(h[0], np.log(2.)))
        t, weights, info = _w_quadrature(density, center, 48, 1536, tol, tail, None)
        return t, weights, info, (K, H, e, V, residual)

    def marginal(self, u, *, tol=1e-8, tail=40.):
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
                    scores.append(.5*((alpha.T*weighted)@alpha
                        -(V*(weighted[:, None]/d).sum(0))@V.T))
                Cscore, Hscore = scores
                w, sw = np.exp(u[3:3+self.d]), np.exp(u[3+self.d:self.qdim])
                g = np.r_[np.sum(Cscore*K), np.sum(Hscore*H), np.exp(u[2])*np.trace(Cscore),
                          -.5*w*(self.dist@(Cscore*K).ravel()),
                          -.5*sw*(self.dist@(Hscore*H).ravel())]
                pk, gk = self.kernel_prior(u[:self.qdim])
                if self.has_mixing:
                    ph, gh = mixing_prior(u[self.qdim:], self.mixing_model)
                    _, _, gm = mixing_log_density(t, u[self.qdim:], self.mixing_model, True)
                    value, grad = float(info['log_integral']+pk+ph), np.r_[g+gk, weights@gm+gh]
                else:
                    value, grad = float(info['log_integral']+pk), g+gk
                if not np.isfinite(value) or not np.isfinite(grad).all():
                    raise FloatingPointError('Nonfinite posterior')
                return value, grad
        except (ArithmeticError, ValueError, RuntimeError, np.linalg.LinAlgError):
            return -np.inf, np.zeros(self.qdim+self.hdim)

    def predict_components(self, u, xp, *, quadrature=None):
        """Pointwise conditional Gaussians using one eigendecomposition.

        No test-test covariance or repeated training Cholesky is needed. Native
        means/variances describe beta (not W*beta), or the STP's added GP mean.
        """
        xp = np.asarray(xp, float)
        if xp.ndim == 1 and self.d == 1:
            xp = xp[:, None]
        if xp.ndim != 2 or xp.shape[1] != self.d or not len(xp) or not np.isfinite(xp).all():
            raise ValueError('Invalid prediction inputs')
        t, weights, info, (_, _, e, V, residual) = (
            self.quadrature(u, tol=1e-11, tail=50.) if quadrature is None else quadrature)
        Kpx, Hpx = self.kernels(np.asarray(u[:self.qdim]),
            squared_distances(xp, self.x).reshape(self.d, -1), (len(xp), len(self.x)))
        KV, HV = Kpx@V, Hpx@V
        v, a, k = np.exp(u[:3])
        means, variances, native_means, native_vars = [np.empty((len(t), len(xp))) for _ in range(4)]
        for i, W in enumerate(np.exp(t)):
            d = W+W**self.power*e
            projected = W*KV+W**self.power*HV
            means[i] = projected@(residual/d)
            variance = W*v+W**self.power*a-(projected**2/d).sum(1)
            factor = W if self.model == 'beta_gp' else 1.
            native_means[i] = factor*HV@(residual/d)
            native_variance = a-factor**2*(HV**2/d).sum(1)
            if variance.min() < -1e-7*max(1., W*v+W**self.power*a) or native_variance.min() < -1e-7*max(1., a):
                raise ArithmeticError('Negative predictive variance beyond roundoff')
            variances[i], native_vars[i] = np.maximum(variance, 1e-14), np.maximum(native_variance, 1e-14)
        return dict(log_W=t, weights=weights, component_means=means,
            component_f_variances=variances, component_y_variances=variances+np.exp(t)[:, None]*k,
            component_native_means=native_means, component_native_variances=native_vars)
