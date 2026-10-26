"""Small, self-contained GHP BO model; no imports from legacy experiments.

GIG convention: density proportional to w^(lambda-1) exp(-(chi/w+psi*w)/2).
Parameters in sampling/MAP coordinates: [mu, beta, log(v0), log(lengthscale)].
The nugget is inside the W-scaled covariance. All computations use float64.
"""
import math
import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
from functools import lru_cache
import numpy as np
import torch
from scipy import linalg, optimize, special

torch.set_num_threads(1)
DTYPE = torch.float64
PRIOR_MEAN = np.array([0., 0., 0., math.log(.2)])
PRIOR_SD = np.array([2., 1., 1.5, 1.])


def objective(u):
    x = 5 + 5 * np.asarray(u)
    return -(x - 1)**2 * np.sin(3*x + 5/x + 1)


class LogBesselK(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, order):
        value = float(z.detach())
        ctx.save_for_backward(z)
        ctx.order = order
        return z.new_tensor(np.log(special.kve(order, value)) - value)

    @staticmethod
    def backward(ctx, grad):
        z, = ctx.saved_tensors
        value = float(z.detach())
        p = ctx.order
        derivative = -.5*(special.kve(p-1, value)+special.kve(p+1, value))/special.kve(p, value)
        return grad*z.new_tensor(derivative), None


def kernel(x, z, v0, ell):
    r = np.sqrt(5.) * np.abs(np.asarray(x)[:, None]-np.asarray(z)[None, :])/ell
    return v0*(1+r+r*r/3)*np.exp(-r)


class Posterior:
    def __init__(self, x, y, lam=1., omega=1., nugget=1e-6):
        self.x, self.y = np.asarray(x), np.asarray(y)
        self.lam, self.omega, self.nugget = lam, omega, nugget
        self.xt, self.yt = torch.tensor(self.x, dtype=DTYPE), torch.tensor(self.y, dtype=DTYPE)

    def log_prob_torch(self, q):
        mu, beta, lv, le = q.unbind()
        r = math.sqrt(5.)*torch.abs(self.xt[:, None]-self.xt[None, :])/le.exp()
        c = lv.exp()*(1+r+r*r/3)*torch.exp(-r) + self.nugget*torch.eye(len(self.x), dtype=DTYPE)
        L = torch.linalg.cholesky(c)
        residual = self.yt-mu
        b = torch.ones_like(residual)*beta
        sol = torch.cholesky_solve(torch.stack([residual, b], dim=1), L)
        chi = self.omega + residual @ sol[:, 0]
        psi = self.omega + b @ sol[:, 1]
        p = self.lam-len(self.x)/2
        z = torch.sqrt(chi*psi)
        loglik = (-len(self.x)/2*math.log(2*math.pi)-torch.log(L.diag()).sum()
                  + residual @ sol[:, 1] + p/2*(chi.log()-psi.log())
                  + LogBesselK.apply(z, p)
                  - (np.log(special.kve(self.lam, self.omega))-self.omega))
        prior = -.5*(((q-q.new_tensor(PRIOR_MEAN))/q.new_tensor(PRIOR_SD))**2).sum()
        return loglik+prior

    def value_grad(self, q):
        t = torch.tensor(q, dtype=DTYPE, requires_grad=True)
        try:
            lp = self.log_prob_torch(t)
            g, = torch.autograd.grad(lp, t)
            if not torch.isfinite(lp) or not torch.isfinite(g).all():
                raise ValueError('Nonfinite posterior')
            return -float(lp.detach()), -g.detach().numpy()
        except (RuntimeError, ValueError):
            return 1e100, np.zeros(4)

    def components(self, q, candidates):
        mu, beta, lv, le = q
        v0, ell = np.exp(lv), np.exp(le)
        c = kernel(self.x, self.x, v0, ell)+self.nugget*np.eye(len(self.x))
        L = linalg.cholesky(c, lower=True)
        residual, b = self.y-mu, np.full(len(self.x), beta)
        alpha = linalg.cho_solve((L, True), np.stack([residual, b], axis=1))
        cross = kernel(self.x, candidates, v0, ell)
        base = mu+cross.T @ alpha[:, 0]
        skew = beta-cross.T @ alpha[:, 1]
        v = linalg.solve_triangular(L, cross, lower=True)
        variance = np.maximum(v0-(v*v).sum(axis=0), 1e-14)
        chi = self.omega+residual @ alpha[:, 0]
        psi = self.omega+b @ alpha[:, 1]
        return base, skew, variance, self.lam-len(self.x)/2, chi, psi

    def log_prob(self, q):
        try:
            mu, beta, lv, le = q
            if not np.all(np.isfinite(q)) or abs(lv)>700 or abs(le)>350:
                return -np.inf
            c = kernel(self.x, self.x, np.exp(lv), np.exp(le))+self.nugget*np.eye(len(self.x))
            L = linalg.cholesky(c, lower=True)
            residual, b = self.y-mu, np.full(len(self.x), beta)
            a = linalg.cho_solve((L, True), np.stack([residual, b], axis=1))
            chi, psi = self.omega+residual @ a[:, 0], self.omega+b @ a[:, 1]
            p, z = self.lam-len(self.x)/2, np.sqrt(chi*psi)
            lp = (-len(self.x)/2*np.log(2*np.pi)-np.log(L.diagonal()).sum()+residual @ a[:, 1]
                  +p/2*np.log(chi/psi)+np.log(special.kve(p,z))-z
                  -np.log(special.kve(self.lam,self.omega))+self.omega
                  -.5*np.sum(((q-PRIOR_MEAN)/PRIOR_SD)**2))
            return float(lp) if np.isfinite(lp) else -np.inf
        except (ValueError, linalg.LinAlgError, OverflowError):
            return -np.inf


@lru_cache(None)
def legendre(n):
    return np.polynomial.legendre.leggauss(n)


def mixing_quadrature(p, chi, psi, n=96):
    # Integrate t=log W. Its log density is strictly concave.
    root = np.hypot(p, np.sqrt(chi*psi))
    mode_w = (p+root)/psi if p >= 0 else chi/(root-p)
    mode = np.log(mode_w)
    def h(t):
        return p*t-.5*(chi*np.exp(-t)+psi*np.exp(t))
    peak = h(mode)
    width = 1.
    while h(mode-width)>peak-40 or h(mode+width)>peak-40:
        width *= 2
    lo = optimize.brentq(lambda t:h(t)-peak+40, mode-width, mode)
    hi = optimize.brentq(lambda t:h(t)-peak+40, mode, mode+width)
    z, weights = legendre(n)
    t = (lo+hi)/2+(hi-lo)/2*z
    logmass = h(t)+np.log(weights)+np.log((hi-lo)/2)
    weights = np.exp(logmass-special.logsumexp(logmass))
    return np.exp(t), weights


class Acquisition:
    def __init__(self, posterior, draws, nodes=96):
        self.posterior, self.draws, self.nodes = posterior, draws, nodes
        # Training solves and quadrature are cached per parameter sample.
        self.cache = []
        for q in draws:
            mu,beta,lv,le = q
            v0,ell = np.exp(lv),np.exp(le)
            L = linalg.cholesky(kernel(posterior.x,posterior.x,v0,ell)+posterior.nugget*np.eye(len(posterior.x)),lower=True)
            r,b = posterior.y-mu,np.full(len(posterior.x),beta)
            a = linalg.cho_solve((L,True),np.stack([r,b],axis=1))
            w,weights = mixing_quadrature(posterior.lam-len(r)/2,posterior.omega+r@a[:,0],posterior.omega+b@a[:,1],nodes)
            self.cache.append((mu,beta,v0,ell,L,a,w,weights))

    def __call__(self, candidates):
        x = np.atleast_1d(candidates)
        total = np.zeros(len(x))
        for mu,beta,v0,ell,L,a,w,weights in self.cache:
            cross = kernel(self.posterior.x,x,v0,ell)
            base,skew = mu+cross.T@a[:,0],beta-cross.T@a[:,1]
            v = linalg.solve_triangular(L,cross,lower=True)
            variance = np.maximum(v0-(v*v).sum(axis=0),1e-14)
            m = base[None,:]+w[:,None]*skew[None,:]
            sd = np.sqrt(w[:,None]*variance[None,:])
            improvement = self.posterior.y.min()-m
            z = improvement/sd
            ei = improvement*special.ndtr(z)+sd*np.exp(-z*z/2)/np.sqrt(2*np.pi)
            total += weights @ np.maximum(ei,0)
        return total/len(self.cache)

    def maximize(self, ngrid=1001):
        grid = np.linspace(0,1,ngrid)
        values = self(grid)
        valid = np.min(np.abs(grid[:,None]-self.posterior.x[None,:]),axis=1)>1e-5
        scores = np.where(valid,values,-np.inf)
        indices = np.argsort(scores)[-8:]
        options = [(float(grid[i]),float(scores[i])) for i in indices]
        for i in indices:
            fit = optimize.minimize_scalar(lambda u:-self([u])[0],bounds=(grid[max(0,i-1)],grid[min(ngrid-1,i+1)]),method='bounded',options={'xatol':1e-9})
            if np.min(np.abs(fit.x-self.posterior.x))>1e-5:
                options.append((float(fit.x),float(-fit.fun)))
        chosen = max(options,key=lambda a:a[1])
        return chosen,grid,values
