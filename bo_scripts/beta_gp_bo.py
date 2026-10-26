"""Varying-beta BO adapters for the existing joint GP-prior model.

mu=0, lambda=omega=1. Integrating beta gives W*Kf + W^2*Kbeta.
MAP follows map_fixed_function_30_experiment: joint log(theta),log(W)
mode, conditional beta mean plug-in, then recompute the GIG posterior W.
HMC and slice retain beta uncertainty analytically and sample theta,W.
"""
import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scipy import linalg, optimize, special
from GH_proc_joint_hmc import make_joint_target, effective_coordinates, physical_coordinates
from GH_proc_w_betaprior_hmc import sample_hmc, chain_diagnostics
from gh_bo_core import Acquisition, mixing_quadrature

PRIOR_CENTER = np.log([.5, .5, 1e-4, 25., 6.25])
PRIOR_SD = 2.
PARAMETERS = ['v0', 'beta_v0', 'k', 'w', 'beta_w', 'W']


def se(x, xp, amplitude, w):
    return amplitude*np.exp(-.5*w*(np.asarray(x)[:, None]-np.asarray(xp)[None, :])**2)


def target_for(x, y):
    return make_joint_target(y, x, model='beta_gp', prior_center=PRIOR_CENTER,
                             prior_sd=PRIOR_SD, l=1., omega=1.)


def fit_map(target, seed, previous=None, starts=8):
    rng = np.random.default_rng(seed)
    base = effective_coordinates(np.r_[PRIOR_CENTER, np.arcsinh(1.)], 'beta_gp')
    candidates = []
    reports = []
    for i in range(starts):
        q = base.copy() if i < 2 else base+rng.normal(0, 1., 6)
        if previous is not None and i == 0:
            q = np.asarray(previous)[0].copy()
        if i % 2:
            q[[0, 1]] = q[[1, 0]]; q[[3, 4]] = q[[4, 3]]
        def objective(z):
            value, gradient = target(z)
            return -value, -gradient
        fit = optimize.minimize(objective, q, jac=True, method='L-BFGS-B',
                                options={'maxiter':1500, 'ftol':1e-13, 'gtol':1e-7, 'maxls':40})
        value, grad = target(fit.x)
        good = bool(np.isfinite(value) and np.max(np.abs(grad)) < 1e-4)
        reports.append({'start':i, 'stationary':good, 'optimizer_success':bool(fit.success),
                        'log_posterior':float(value) if np.isfinite(value) else None,
                        'max_abs_gradient':float(np.max(np.abs(grad))), 'message':str(fit.message)})
        if good:
            candidates.append((value, fit.x))
    if not candidates:
        raise RuntimeError('No stationary MAP solution; increase --map-starts')
    _, q = max(candidates, key=lambda item:item[0])
    return q, {'starts':reports, 'selected_gradient_norm':float(np.linalg.norm(target(q)[1]))}


def slice_samples(target, starts, seed, warmup, draws):
    chains=[]; swaps=[]
    for initial, child in zip(starts, np.random.SeedSequence(seed).spawn(len(starts))):
        rng=np.random.default_rng(child);q=initial.copy();trace=[];accepted=0
        for iteration in range(warmup+draws):
            for d in rng.permutation(6):
                level=target(q)[0]+np.log(rng.random())
                left=q[d]-rng.random();right=left+1.
                j=int(rng.integers(64));k=63-j
                trial=q.copy()
                while j:
                    trial[d]=left
                    if target(trial)[0]<=level: break
                    left-=1.;j-=1
                while k:
                    trial[d]=right
                    if target(trial)[0]<=level: break
                    right+=1.;k-=1
                for _ in range(10000):
                    value=rng.uniform(left,right);trial=q.copy();trial[d]=value
                    if target(trial)[0]>=level:
                        q=trial;break
                    if value<q[d]:left=value
                    else:right=value
                else:raise RuntimeError('Slice shrinkage failed')
            # Same posterior-corrected component exchange as the existing HMC.
            trial=q[[1,0,2,4,3,5]]
            if np.log(rng.random())<min(0.,target(trial)[0]-target(q)[0]):
                q=trial
                if iteration>=warmup:accepted+=1
            if iteration>=warmup:trace.append(q.copy())
        chains.append(trace);swaps.append(accepted/draws)
    return np.asarray(chains), {'swap_acceptance':swaps}


def infer(x, y, method, seed, previous, warmup, draws, chains, map_starts, target_accept):
    target=target_for(x,y)
    if method=='map':
        q,diag=fit_map(target,seed,previous,starts=map_starts)
        samples=q[None,None,:];extra={}
    else:
        if previous is None:
            q,_=fit_map(target,seed,starts=map_starts)
            rng=np.random.default_rng(seed)
            starts=np.tile(q,(chains,1))+rng.normal(0,.25,(chains,6))
            for i in range(1,chains,2):
                starts[i]=starts[i,[1,0,2,4,3,5]]
        else: starts=np.asarray(previous)
        if method=='hmc':
            result=sample_hmc(target,starts,warmup=warmup,draws=draws,seed=seed,
                              step_size=.05,target_accept=target_accept,
                              leapfrog_steps=(12,30),swap_pairs=[(0,1),(3,4)])
            samples=result['samples']
            extra={k:v for k,v in result.items() if isinstance(v,np.ndarray) and k!='samples'}
            diag={'divergences':int(result['divergent'].sum()),
                  'acceptance':result['acceptance'].mean(axis=1).tolist(),
                  'step_size':result['step_size'].tolist()}
        else:
            samples,diag=slice_samples(target,starts,seed,warmup,draws);extra={}
        physical=physical_coordinates(samples,'beta_gp')
        diagnostics=chain_diagnostics(physical)
        diag.update({k:np.asarray(v).tolist() for k,v in diagnostics.items()})
        diag['diagnostics_ok']=bool(np.max(diagnostics['rhat'])<1.01 and
            np.min(diagnostics['ess_bulk'])>=400 and np.min(diagnostics['ess_tail'])>=400
            and diag.get('divergences',0)==0)
    return samples,diag,extra


class BetaAcquisition:
    maximize=Acquisition.maximize

    def __init__(self,x,y,effective_draws,method,nodes=96):
        self.x=np.asarray(x);self.y=np.asarray(y);self.posterior=self
        self.method=method;self.cache=[]
        for q in physical_coordinates(effective_draws,'beta_gp'):
            v,b,k,w,bw,W=np.exp(q)
            C=W*(se(x,x,v,w)+k*np.eye(len(x)))+W**2*se(x,x,b,bw)
            L=linalg.cholesky(C,lower=True)
            alpha=linalg.cho_solve((L,True),y)
            if method=='map':
                beta_train=W*se(x,x,b,bw)@alpha
                A=se(x,x,v,w)+k*np.eye(len(x))
                residual_L=linalg.cholesky(A,lower=True)
                solves=linalg.cho_solve((residual_L,True),np.column_stack([y,beta_train]))
                chi=1.+self.y@solves[:,0];psi=1.+beta_train@solves[:,1]
                nodes_w,weights=mixing_quadrature(1.-len(x)/2,chi,psi,nodes)
                self.cache.append((v,b,k,w,bw,W,L,alpha,residual_L,solves,nodes_w,weights))
            else:self.cache.append((v,b,k,w,bw,W,L,alpha))

    def components(self,candidates,index=0):
        xp=np.atleast_1d(candidates)
        v,b,k,w,bw,W,L,alpha,*rest=self.cache[index]
        if self.method=='map':
            residual_L,solves,nodes_w,weights=rest
            beta_pred=W*se(xp,self.x,b,bw)@alpha
            cross=se(xp,self.x,v,w)
            base=cross@solves[:,0];skew=beta_pred-cross@solves[:,1]
            solved=linalg.solve_triangular(residual_L,cross.T,lower=True)
            variance=np.maximum(v-np.sum(solved**2,axis=0),1e-14)
            return base[None,:]+nodes_w[:,None]*skew[None,:],nodes_w[:,None]*variance[None,:],weights
        cross=W*se(xp,self.x,v,w)+W**2*se(xp,self.x,b,bw)
        mean=cross@alpha
        solved=linalg.solve_triangular(L,cross.T,lower=True)
        var=np.maximum(W*v+W**2*b-np.sum(solved**2,axis=0),1e-14)
        return mean[None,:],var[None,:],np.ones(1)

    def __call__(self,candidates):
        total=np.zeros(len(np.atleast_1d(candidates)))
        for i in range(len(self.cache)):
            mean,var,weights=self.components(candidates,i)
            sd=np.sqrt(var);delta=self.y.min()-mean;z=delta/sd
            total+=weights@np.maximum(delta*special.ndtr(z)+sd*np.exp(-z*z/2)/np.sqrt(2*np.pi),0.)
        return total/len(self.cache)
