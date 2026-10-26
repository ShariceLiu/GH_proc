"""Exact zero-mean GP/STP posterior and memory-bounded Bayesian predictions.

One ARD SE kernel; Gaussian priors on log(v), log(k), log(w_d), SD 2.
For STP, W~IG(nu/2,(nu-2)/2) is integrated analytically, both in training
and prediction. The predictive mixture is Student-t, not a Gaussian fit to it.
"""
import json
import time
from pathlib import Path

import numpy as np
from scipy import linalg, special, stats, integrate
from scipy.spatial.distance import pdist, squareform, cdist

from GH_tabular_hmc import default_center


class ZeroPosterior:
    def __init__(self,y,x,model,*,nu=5.,prior_sd=2.):
        self.y,self.x=np.asarray(y,float),np.asarray(x,float)
        if model not in ('gp_zero','stp_zero') or self.x.ndim!=2 or self.y.shape!=(len(x),):
            raise ValueError('Need supported zero-mean model and x[N,D], y[N]')
        if not np.isfinite(self.x).all() or not np.isfinite(self.y).all() or nu<=2 or prior_sd<=0:
            raise ValueError('Invalid data/prior')
        self.model,self.nu,self.prior_sd=model,float(nu),float(prior_sd)
        self.n,self.d=self.x.shape
        self.center=default_center(y,self.d,model)
        self.distances=np.empty((self.d,self.n*(self.n-1)//2))
        for j in range(self.d):
            self.distances[j]=pdist(self.x[:,j:j+1],metric='sqeuclidean')
        self.diagonal=np.diag_indices(self.n)

    def evaluate(self,q,*,gradient=False,factor=False):
        q=np.asarray(q,float)
        if q.shape!=(self.d+2,) or not np.isfinite(q).all():
            raise ValueError('Expected finite log kernel/noise parameters')
        try:
            with np.errstate(over='raise',invalid='raise',divide='raise'):
                v,k,*weights=np.exp(q)
                w=np.asarray(weights)
                K=squareform(v*np.exp(-.5*(w@self.distances)))
                K[self.diagonal]=v
                A=K.copy(); A[self.diagonal]+=k
                L=linalg.cholesky(A,lower=True,check_finite=False)
                alpha=linalg.cho_solve((L,True),self.y,check_finite=False)
                quadratic=float(self.y@alpha)
                logdet_half=np.log(L.diagonal()).sum()
                if self.model=='gp_zero':
                    likelihood=-.5*self.n*np.log(2*np.pi)-logdet_half-.5*quadratic
                    coefficient=1.
                else:
                    nu=self.nu
                    likelihood=(special.gammaln((nu+self.n)/2)-special.gammaln(nu/2)
                        -.5*self.n*np.log(np.pi*(nu-2))-logdet_half
                        -.5*(nu+self.n)*np.log1p(quadratic/(nu-2)))
                    coefficient=(nu+self.n)/(nu-2+quadratic)
                standardized=(q-self.center)/self.prior_sd
                prior=-.5*standardized@standardized-len(q)*np.log(self.prior_sd*np.sqrt(2*np.pi))
                result=dict(logp=float(likelihood+prior),quadratic=quadratic)
                if gradient:
                    lower,info=linalg.lapack.dpotri(L,lower=1)
                    if info: raise np.linalg.LinAlgError('POTRI failed')
                    inverse=lower+lower.T; inverse[self.diagonal]*=.5
                    score=.5*(coefficient*np.outer(alpha,alpha)-inverse)
                    weighted=score*K
                    deriv=np.r_[weighted.sum(),k*np.trace(score),
                        -.5*w*(self.distances@(2*squareform(weighted,checks=False)))]
                    result['gradient']=deriv-standardized/self.prior_sd
                if factor:
                    result.update(L=L,alpha=alpha)
                return result
        except (FloatingPointError,OverflowError,np.linalg.LinAlgError):
            result=dict(logp=-np.inf,quadratic=np.nan)
            if gradient: result['gradient']=np.zeros(len(q))
            return result

    def target(self,q):
        p=self.evaluate(q,gradient=True)
        return p['logp'],p['gradient']

    def predict(self,q,xp,*,output_mean=0.,output_scale=1.,batch_size=128):
        p=self.evaluate(q,factor=True)
        if not np.isfinite(p['logp']):
            raise FloatingPointError('Nonfinite prediction parameter draw')
        v,k,*weights=np.exp(q); w=np.asarray(weights)
        xp=np.asarray(xp,float)
        mean=np.empty(len(xp)); residual=np.empty(len(xp))
        for start in range(0,len(xp),batch_size):
            sl=slice(start,start+batch_size)
            cross=v*np.exp(-.5*cdist(self.x*np.sqrt(w),xp[sl]*np.sqrt(w),metric='sqeuclidean'))
            solved=linalg.solve_triangular(p['L'],cross,lower=True,check_finite=False)
            mean[sl]=cross.T@p['alpha']
            residual[sl]=v-np.sum(solved**2,axis=0)
        if residual.min()<=0:
            raise FloatingPointError('Nonpositive latent conditional variance')
        if self.model=='stp_zero':
            shape=(self.nu+self.n)/2
            scale=(self.nu-2+p['quadratic'])/2
            factor=scale/shape
        else:
            shape,scale,factor=np.inf,np.inf,1.
        return dict(mean=output_mean+output_scale*mean,
            f_scale2=output_scale**2*factor*residual,
            y_scale2=output_scale**2*factor*(residual+k),
            quadratic=p['quadratic'],logp=p['logp'],W_shape=shape,W_scale=scale)


class Proposal:
    """Independent Normal/Student-t mixture; every move gets an exact MH correction."""
    def __init__(self,components):
        self.components=components
        self.weights=np.array([c['weight'] for c in components],float)
        np.testing.assert_allclose(self.weights.sum(),1.)
        self.prepared=[]
        for c in components:
            mean=np.asarray(c['mean'],float); L=linalg.cholesky(c['shape'],lower=True)
            df=c.get('df'); d=len(mean)
            constant=-np.log(L.diagonal()).sum()
            constant+=(-.5*d*np.log(2*np.pi) if df is None else
                special.gammaln((df+d)/2)-special.gammaln(df/2)-.5*d*np.log(df*np.pi))
            self.prepared.append((mean,L,df,float(constant)))

    def draw(self,rng):
        mean,L,df,_=self.prepared[rng.choice(len(self.weights),p=self.weights)]
        z=rng.normal(size=len(mean))
        if df is not None: z/=np.sqrt(rng.chisquare(df)/df)
        return mean+L@z

    def logpdf(self,q):
        terms=[]
        for weight,(mean,L,df,constant) in zip(self.weights,self.prepared):
            z=linalg.solve_triangular(L,q-mean,lower=True,check_finite=False)
            terms.append(np.log(weight)+constant-(.5*z@z if df is None else
                .5*(df+len(mean))*np.log1p(z@z/df)))
        return float(special.logsumexp(terms))


def sample_chain(target,proposal,start,*,count,seed,checkpoint=None,identity='',label='',progress_every=100,
                 local_covariance=None,local_probability=0.):
    """Fixed-proposal MH; retain rejections and resume RNG exactly.

    No gradients or approximated likelihoods enter the acceptance decision.
    ``count`` may be increased to extend an existing chain without restarting.
    Optionally mix in a symmetric, fixed-covariance local random-walk kernel.
    """
    if not 0<=local_probability<=1 or (local_probability and local_covariance is None):
        raise ValueError('Invalid local-move configuration')
    local_cholesky=None if local_covariance is None else linalg.cholesky(local_covariance,lower=True)
    rng=np.random.default_rng(seed)
    dimension=len(start)
    arrays=dict(q=np.empty((count,dimension)),logp=np.empty(count),quadratic=np.empty(count),
                accepted=np.zeros(count,bool),accept_prob=np.empty(count))
    checkpoint=None if checkpoint is None else Path(checkpoint)
    begin=0
    if checkpoint is not None and checkpoint.exists():
        with np.load(checkpoint,allow_pickle=False) as saved:
            if str(saved['identity'])!=identity: raise ValueError('Checkpoint identity mismatch')
            begin=len(saved['q'])
            if begin>count: raise ValueError('Cannot shorten a saved chain')
            for key in arrays: arrays[key][:begin]=saved[key]
            rng.bit_generator.state=json.loads(str(saved['rng_state']))
        q=arrays['q'][begin-1].copy()
        p=dict(logp=arrays['logp'][begin-1],quadratic=arrays['quadratic'][begin-1])
    else:
        q=np.asarray(start,float).copy();p=target(q)
    if not np.isfinite(p['logp']): raise ValueError('Nonfinite initial state')
    proposal_logp=proposal.logpdf(q)
    last=time.monotonic()
    for i in range(begin,count):
        local=local_probability>0 and rng.random()<local_probability
        candidate=q+local_cholesky@rng.normal(size=dimension) if local else proposal.draw(rng)
        new=target(candidate)
        candidate_logp=proposal.logpdf(candidate)
        # Select a kernel independently of the state. Each kernel separately
        # obeys detailed balance; a symmetric local proposal needs no q ratio.
        log_ratio=new['logp']-p['logp']+(0. if local else proposal_logp-candidate_logp)
        probability=float(np.exp(min(0.,log_ratio))) if not np.isnan(log_ratio) else 0.
        accept=rng.random()<probability
        if accept:
            q,p,proposal_logp=candidate,new,candidate_logp
        arrays['q'][i]=q; arrays['logp'][i]=p['logp']; arrays['quadratic'][i]=p['quadratic']
        arrays['accepted'][i]=accept; arrays['accept_prob'][i]=probability
        if (i+1)%progress_every==0 or i+1==count:
            if checkpoint is not None:
                temporary=checkpoint.with_suffix('.tmp.npz')
                np.savez_compressed(temporary,identity=np.array(identity),
                    rng_state=np.array(json.dumps(rng.bit_generator.state)),
                    **{k:v[:i+1] for k,v in arrays.items()})
                temporary.replace(checkpoint)
            if label:
                print(f'{label}: {i+1}/{count}, acceptance {arrays["accepted"][max(0,i-99):i+1].mean():.2f}',flush=True)
            last=time.monotonic()
        elif label and time.monotonic()-last>45:
            print(f'{label}: {i+1}/{count}',flush=True);last=time.monotonic()
    return arrays


def mixture_logpdf(truth,means,scales,weights,df=None):
    z=(np.asarray(truth)-means)/scales
    if df is None:
        logp=-.5*z*z-.5*np.log(2*np.pi)-np.log(scales)
    else:
        constant=special.gammaln((df+1)/2)-special.gammaln(df/2)-.5*np.log(df*np.pi)
        logp=constant-.5*(df+1)*np.log1p(z*z/df)-np.log(scales)
    positive=np.asarray(weights)>0
    return special.logsumexp(np.log(np.asarray(weights)[positive,None])+logp[positive],axis=0)


def mixture_cdf(values,means,scales,weights,df=None):
    z=(np.asarray(values)-means)/scales
    return weights@(special.ndtr(z) if df is None else special.stdtr(df,z))


def mixture_quantile(probability,means,scales,weights,df=None):
    z=special.ndtri(probability) if df is None else stats.t.ppf(probability,df)
    bounds=means+z*scales
    low,high=bounds.min(0),bounds.max(0)
    for _ in range(50):
        mid=(low+high)/2
        below=mixture_cdf(mid,means,scales,weights,df)<probability
        low,high=np.where(below,mid,low),np.where(below,high,mid)
    return (low+high)/2


def mixture_crps(truth,means,scales,weights,df=None,*,atol=2e-8):
    """E|X-y| - integral F(1-F), with refined Gauss-Legendre rules.

    Supports Normal or common-df Student-t components; no Gaussianization.
    The finite integration tail bound and refinement differences are recorded.
    """
    z=(np.asarray(truth)-means)/scales
    if df is None:
        first=scales*(z*(2*special.ndtr(z)-1)+2*stats.norm.pdf(z))
    else:
        first=scales*(z*(2*special.stdtr(df,z)-1)+2*(df+z*z)/(df-1)*stats.t.pdf(z,df))
    first=weights@first
    tail=10.
    def tail_moment(a):
        if df is None: return stats.norm.pdf(a)-a*stats.norm.sf(a)
        return (df+a*a)/(df-1)*stats.t.pdf(a,df)-a*stats.t.sf(a,df)
    while 2*np.max(weights@scales)*tail_moment(tail)>atol/10:
        tail*=2
        if tail>1e5: raise RuntimeError('Could not bound CRPS integration tails')
    lo=(means-tail*scales).min(0);hi=(means+tail*scales).max(0)
    previous=None
    for count in (32,64,128,256,512):
        nodes,rule=np.polynomial.legendre.leggauss(count)
        integral=np.zeros(len(first))
        for node,weight in zip(nodes,rule):
            cdf=mixture_cdf((lo+hi)/2+(hi-lo)/2*node,means,scales,weights,df)
            integral+=weight*cdf*(1-cdf)
        integral*=(hi-lo)/2
        error=np.full(len(first),np.inf) if previous is None else abs(integral-previous)
        if np.all(error<atol):
            score=first-integral
            if np.any(score<0) or not np.isfinite(score).all(): raise ArithmeticError('Invalid CRPS')
            return score,dict(nodes=count,max_refinement_difference=float(max(error)),
                max_tail_bound=float(2*np.max(weights@scales)*tail_moment(tail)))
        previous=integral
    # Very heavy tails or disparate component scales can require an impractical
    # global rule. The independent adaptive CDF-square formula handles those.
    checked=[independent_crps(truth[j],means[:,j],scales[:,j],weights,df) for j in range(len(first))]
    values,errors=np.array(checked).T
    if np.any(errors>atol): raise RuntimeError('Mixture CRPS integration did not converge')
    return values,dict(nodes=count,adaptive_fallback=True,
        max_refinement_difference=float(max(error)),max_adaptive_error=float(max(errors)),max_tail_bound=0.)


def independent_crps(truth,means,scales,weights,df=None):
    """Independent adaptive integral of F^2 and survival^2 for one test point."""
    def cdf(t):
        z=(t-means)/scales
        return float(weights@(special.ndtr(z) if df is None else special.stdtr(df,z)))
    def survival(t):
        z=(means-t)/scales
        return float(weights@(special.ndtr(z) if df is None else special.stdtr(df,z)))
    left,e1=integrate.quad(lambda t:cdf(t)**2,-np.inf,truth,epsabs=1e-9,epsrel=1e-9)
    right,e2=integrate.quad(lambda t:survival(t)**2,truth,np.inf,epsabs=1e-9,epsrel=1e-9)
    return left+right,e1+e2
