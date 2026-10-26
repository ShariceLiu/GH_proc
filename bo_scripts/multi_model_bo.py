"""Eight BO surrogates with matched SE-kernel conventions and joint-W inference.

MAP uses fixed-function plug-in prediction; HMC/slice retain GP-function
uncertainty. Mixing shapes are fixed: GIG(1,1,1), IG(5/2,3/2).
"""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import numpy as np
from scipy import linalg,optimize,special
from beta_gp_bo import se,PRIOR_CENTER,PRIOR_SD
from GH_proc_joint_hmc import make_joint_target,effective_coordinates,physical_coordinates
from GP_TP_hmc import make_baseline_target
from GH_proc_w_betaprior_hmc import sample_hmc,chain_diagnostics
from gh_bo_core import Acquisition,mixing_quadrature

MODELS=('ghp_const_beta','ghp_beta_gp','ghp_mu_gp','gp_mu_gp','stp_mu_gp',
        'gp_zero','stp_zero','ghp_zero')
JOINT={'ghp_beta_gp':'beta_gp','ghp_mu_gp':'mu_gp','stp_mu_gp':'tp_mu_gp'}
LABELS={'ghp_const_beta':'GHP constant beta','ghp_beta_gp':'GHP varying beta',
        'ghp_mu_gp':'GHP GP mean','gp_mu_gp':'GP GP mean','stp_mu_gp':'STP GP mean',
        'gp_zero':'GP zero mean','stp_zero':'STP zero mean','ghp_zero':'GHP zero mean'}


class Model:
    def __init__(self,x,y,name):
        if name not in MODELS:raise ValueError(name)
        self.x=np.asarray(x);self.y=np.asarray(y);self.name=name
        self.constant=name=='ghp_const_beta';self.gp=name in ('gp_mu_gp','gp_zero')
        self.zero=name in ('gp_zero','stp_zero','ghp_zero')
        self.swaps=None if self.constant or self.zero else [(0,1),(3,4)]
        if self.zero:
            self.base=np.log([.5,1e-4,25.])
            self.names=['log_v0','log_k','log_w']
            if not self.gp:
                self.base=np.r_[self.base,np.log(.6) if name=='stp_zero' else np.arcsinh(1.)]
                self.names+=['log_W']
            self.target=self.zero_target
        elif self.constant:
            self.base=np.array([np.log(.5),np.log(1e-4),np.log(25.),0.,0.,np.arcsinh(1.)])
            self.names=['log_v0','log_k','log_w','mu','beta','log_W']
            self.target=self.constant_target
        else:
            self.names=['log_v0','log_latent_v0','log_k','log_w','log_latent_w']
            if self.gp:
                self.base=PRIOR_CENTER.copy()
                self.target=make_baseline_target(self.y,self.x,model='gp',prior_center=PRIOR_CENTER,prior_sd=PRIOR_SD)
            else:
                kind=JOINT[name]
                self.base=effective_coordinates(np.r_[PRIOR_CENTER,np.log(.6) if name=='stp_mu_gp' else np.arcsinh(1.)],kind)
                self.names+=['log_W']
                self.target=make_joint_target(self.y,self.x,model=kind,prior_center=PRIOR_CENTER,prior_sd=PRIOR_SD,l=1.,omega=1.,nu=5.)

    def physical(self,z):
        return np.asarray(z).copy() if self.constant or self.gp or self.zero else physical_coordinates(z,JOINT[self.name])

    def zero_target(self,q):
        """One residual SE kernel, no mean/skew parameters or latent mean GP."""
        try:
            with np.errstate(over='raise',invalid='raise',divide='raise'):
                v,k,w=np.exp(q[:3]);W=1. if self.gp else np.exp(q[3])
                distance=(self.x[:,None]-self.x[None,:])**2
                signal=W*v*np.exp(-.5*w*distance);noise=W*k
                C=signal+noise*np.eye(len(self.x));L=linalg.cholesky(C,lower=True)
                a=linalg.cho_solve((L,True),self.y)
                score=.5*(np.outer(a,a)-linalg.cho_solve((L,True),np.eye(len(self.y))))
                standardized=(q[:3]-self.base[:3])/2
                lp=(-len(self.y)/2*np.log(2*np.pi)-np.log(L.diagonal()).sum()-.5*self.y@a
                    -.5*standardized@standardized-3*np.log(2*np.sqrt(2*np.pi)))
                grad=np.array([np.sum(score*signal),noise*np.trace(score),
                               np.sum(score*signal*(-.5*w*distance))])-standardized/2
                if not self.gp:
                    if self.name=='stp_zero':
                        lp+=2.5*np.log(1.5)-special.gammaln(2.5)-2.5*q[3]-1.5/W
                        mix_grad=-2.5+1.5/W
                    else:
                        lp+=q[3]-.5*(W+1/W)-np.log(2)-np.log(special.kve(1,1))+1
                        mix_grad=1-.5*(W-1/W)
                    grad=np.r_[grad,np.sum(score*C)+mix_grad]
                if np.isfinite(lp) and np.isfinite(grad).all():return float(lp),grad
        except (ValueError,FloatingPointError,linalg.LinAlgError):pass
        return -np.inf,np.zeros(len(self.base))

    def exchange(self,q):
        trial=q.copy()
        if self.swaps:
            for a,b in self.swaps:trial[[a,b]]=trial[[b,a]]
        return trial

    def constant_target(self,q):
        """Joint log density in [log v,log k,log w,mu,beta,log W]."""
        try:
            with np.errstate(over='raise',invalid='raise',divide='raise'):
                lv,lk,lw,mu,beta,t=q;v,k,w,W=np.exp([lv,lk,lw,t])
                distance=(self.x[:,None]-self.x[None,:])**2
                signal=W*v*np.exp(-.5*w*distance);noise=W*k
                C=signal+noise*np.eye(len(self.x));L=linalg.cholesky(C,lower=True)
                r=self.y-mu-W*beta;a=linalg.cho_solve((L,True),r)
                score=.5*(np.outer(a,a)-linalg.cho_solve((L,True),np.eye(len(r))))
                standardized=(q[:3]-self.base[:3])/2
                lp=(-len(r)/2*np.log(2*np.pi)-np.log(L.diagonal()).sum()-.5*r@a
                    -.5*standardized@standardized-3*np.log(2*np.sqrt(2*np.pi))
                    -.5*(mu/2)**2-np.log(2*np.sqrt(2*np.pi))-.5*beta**2-.5*np.log(2*np.pi)
                    +t-.5*(W+1/W)-np.log(2)-np.log(special.kve(1,1))+1)
                grad=np.array([np.sum(score*signal),noise*np.trace(score),
                    np.sum(score*signal*(-.5*w*distance)),a.sum()-mu/4,
                    W*a.sum()-beta,np.sum(score*C)+W*beta*a.sum()+1-.5*(W-1/W)])
                grad[:3]-=standardized/2
                if np.isfinite(lp) and np.isfinite(grad).all():return float(lp),grad
        except (ValueError,FloatingPointError,linalg.LinAlgError):pass
        return -np.inf,np.zeros(6)


def map_fit(model,seed,previous,starts):
    rng=np.random.default_rng(seed);fits=[];reports=[]
    for i in range(starts):
        q=model.base.copy() if i<2 else model.base+rng.normal(0,1,len(model.base))
        if previous is not None and i==0:q=np.asarray(previous)[0].copy()
        if i%2:q=model.exchange(q)
        def objective(z):
            lp,g=model.target(z);return -lp,-g
        fit=optimize.minimize(objective,q,jac=True,method='L-BFGS-B',
                              options={'maxiter':1500,'ftol':1e-13,'gtol':1e-7,'maxls':40})
        lp,g=model.target(fit.x);norm=float(np.max(np.abs(g)))
        valid=bool(np.isfinite(lp) and norm<1e-4)
        reports.append(dict(start=i,stationary=valid,optimizer_success=bool(fit.success),
                            log_posterior=float(lp) if np.isfinite(lp) else None,max_abs_gradient=norm))
        if valid:fits.append((lp,fit.x))
    if not fits:raise RuntimeError(f'{model.name}: no stationary MAP solution; increase --map-starts')
    _,q=max(fits,key=lambda a:a[0])
    return q,dict(starts=reports,selected_max_abs_gradient=float(np.max(np.abs(model.target(q)[1]))))


def slice_sample(model,starts,seed,warmup,draws):
    result=[]
    for initial,child in zip(starts,np.random.SeedSequence(seed).spawn(len(starts))):
        rng=np.random.default_rng(child);q=initial.copy();trace=[]
        for iteration in range(warmup+draws):
            for d in rng.permutation(len(q)):
                level=model.target(q)[0]+np.log(rng.random());left=q[d]-rng.random();right=left+1
                j=int(rng.integers(64));k=63-j;trial=q.copy()
                while j:
                    trial[d]=left
                    if model.target(trial)[0]<=level:break
                    left-=1;j-=1
                while k:
                    trial[d]=right
                    if model.target(trial)[0]<=level:break
                    right+=1;k-=1
                for _ in range(10000):
                    value=rng.uniform(left,right);trial=q.copy();trial[d]=value
                    if model.target(trial)[0]>=level:q=trial;break
                    if value<q[d]:left=value
                    else:right=value
                else:raise RuntimeError('Slice shrinkage failed')
            if model.swaps:
                trial=model.exchange(q)
                if np.log(rng.random())<min(0.,model.target(trial)[0]-model.target(q)[0]):q=trial
            if iteration>=warmup:trace.append(q.copy())
        result.append(trace)
    return np.asarray(result)


def infer(model,method,seed,previous,settings):
    if method=='map':
        q,diag=map_fit(model,seed,previous,settings['map_starts'])
        return q[None,None,:],diag,{}
    if previous is None:
        q,_=map_fit(model,seed,None,settings['map_starts'])
        starts=q+np.random.default_rng(seed).normal(0,.25,(settings['chains'],len(q)))
        for i in range(1,len(starts),2):starts[i]=model.exchange(starts[i])
    else:starts=np.asarray(previous)
    extra={};diag={}
    if method=='hmc':
        result=sample_hmc(model.target,starts,warmup=settings['warmup'],draws=settings['samples'],seed=seed,
            step_size=.05,target_accept=settings['target_accept'],leapfrog_steps=(12,30),swap_pairs=model.swaps)
        samples=result['samples'];extra={k:v for k,v in result.items() if isinstance(v,np.ndarray) and k!='samples'}
        diag=dict(divergences=int(result['divergent'].sum()),acceptance=result['acceptance'].mean(axis=1).tolist())
    else:samples=slice_sample(model,starts,seed,settings['warmup'],settings['samples'])
    d=chain_diagnostics(model.physical(samples));diag.update({k:v.tolist() for k,v in d.items()})
    diag['diagnostics_ok']=bool(np.max(d['rhat'])<1.01 and np.min(d['ess_bulk'])>=400
                                and np.min(d['ess_tail'])>=400 and diag.get('divergences',0)==0)
    return samples,diag,extra


class ModelAcquisition:
    maximize=Acquisition.maximize

    def __init__(self,model,draws,method,nodes=96):
        self.model=model;self.x=model.x;self.y=model.y;self.posterior=self;self.method=method;self.cache=[]
        for q in model.physical(draws):
            if model.constant or model.zero:
                if model.zero:
                    v,k,w=np.exp(q[:3]);mu=beta=0.
                else:
                    lv,lk,lw,mu,beta,t=q;v,k,w,W=np.exp([lv,lk,lw,t])
                A=se(self.x,self.x,v,w)+k*np.eye(len(self.x));L=linalg.cholesky(A,lower=True)
                solves=linalg.cho_solve((L,True),np.column_stack([self.y-mu,np.full(len(self.x),beta)]))
                # Integrate W conditional on each theta (Rao-Blackwellization).
                if model.name=='gp_zero':ww,weights=np.ones(1),np.ones(1)
                elif model.name=='stp_zero':ww,weights=mixing_quadrature(-(5+len(self.x))/2,3+self.y@solves[:,0],0.,nodes)
                else:ww,weights=mixing_quadrature(1-len(self.x)/2,1+(self.y-mu)@solves[:,0],1+beta*solves[:,1].sum(),nodes)
                self.cache.append(dict(v=v,k=k,w=w,mu=mu,beta=beta,L=L,solves=solves,ww=ww,weights=weights))
                continue
            v,b,k,w,bw=np.exp(q[:5]);W=1. if model.gp else np.exp(q[5])
            factor=W**2 if model.name=='ghp_beta_gp' else 1.
            C=W*(se(self.x,self.x,v,w)+k*np.eye(len(self.x)))+factor*se(self.x,self.x,b,bw)
            L=linalg.cholesky(C,lower=True);alpha=linalg.cho_solve((L,True),self.y)
            cache=dict(v=v,b=b,k=k,w=w,bw=bw,W=W,factor=factor,L=L,alpha=alpha)
            if method=='map':
                native_factor=W if model.name=='ghp_beta_gp' else 1.
                estimated=native_factor*se(self.x,self.x,b,bw)@alpha
                beta=estimated if model.name=='ghp_beta_gp' else np.zeros(len(self.x))
                mu=np.zeros(len(self.x)) if model.name=='ghp_beta_gp' else estimated
                residual=self.y-mu
                Lr=linalg.cholesky(se(self.x,self.x,v,w)+k*np.eye(len(self.x)),lower=True)
                solves=linalg.cho_solve((Lr,True),np.column_stack([residual,beta]))
                if model.gp:ww,weights=np.ones(1),np.ones(1)
                elif model.name=='stp_mu_gp':ww,weights=mixing_quadrature(-(5+len(self.x))/2,3+residual@solves[:,0],0.,nodes)
                else:ww,weights=mixing_quadrature(1-len(self.x)/2,1+residual@solves[:,0],1+beta@solves[:,1],nodes)
                cache.update(Lr=Lr,solves=solves,ww=ww,weights=weights,native_factor=native_factor)
            self.cache.append(cache)

    def components(self,xp,index=0):
        xp=np.atleast_1d(xp);c=self.cache[index];m=self.model
        cross=se(xp,self.x,c['v'],c['w'])
        if m.constant or m.zero or self.method=='map':
            if m.constant or m.zero:
                location=np.full(len(xp),c['mu']);beta=np.full(len(xp),c['beta']);L=c['L']
            else:
                estimated=c['native_factor']*se(xp,self.x,c['b'],c['bw'])@c['alpha']
                location=np.zeros(len(xp)) if m.name=='ghp_beta_gp' else estimated
                beta=estimated if m.name=='ghp_beta_gp' else np.zeros(len(xp));L=c['Lr']
            base=location+cross@c['solves'][:,0];skew=beta-cross@c['solves'][:,1]
            solved=linalg.solve_triangular(L,cross.T,lower=True)
            var=np.maximum(c['v']-(solved**2).sum(axis=0),1e-14)
            return base[None,:]+c['ww'][:,None]*skew[None,:],c['ww'][:,None]*var[None,:],c['weights']
        cross=c['W']*cross+c['factor']*se(xp,self.x,c['b'],c['bw'])
        mean=cross@c['alpha'];solved=linalg.solve_triangular(c['L'],cross.T,lower=True)
        var=np.maximum(c['W']*c['v']+c['factor']*c['b']-(solved**2).sum(axis=0),1e-14)
        return mean[None,:],var[None,:],np.ones(1)

    def __call__(self,xp):
        total=np.zeros(len(np.atleast_1d(xp)))
        for i in range(len(self.cache)):
            mean,var,weights=self.components(xp,i);sd=np.sqrt(var);delta=self.y.min()-mean;z=delta/sd
            total+=weights@np.maximum(delta*special.ndtr(z)+sd*np.exp(-z*z/2)/np.sqrt(2*np.pi),0.)
        return total/len(self.cache)
