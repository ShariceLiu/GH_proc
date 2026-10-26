"""Independent per-curve latent GP functions with shared MAP hyperparameters.

The prior for h_j is identical to the previous shared h prior, but h_j are
independent across cycles. For beta, h_j = E_prior[W] beta_j, so the native
beta variance prior is shifted by -2 log E_prior[W]. Prediction first obtains
a context-only log-W MAP with h_j marginalized, fixes its conditional GP mean,
then integrates W under the fixed-function conditional likelihood.
"""
import numpy as np
from scipy import linalg, optimize, special

from GH_multiple_hmc import make_target, coordinates
from GH_shared_function_map import gig_mean

MODELS=('gp','stp','ghp_mu','ghp_beta')
INTERNAL=dict(gp='gp',stp='tp_mu_gp',ghp_mu='mu_gp',ghp_beta='beta_gp')


def native_center(population_center,model):
    center=np.array(population_center,dtype=float,copy=True)
    if model=='ghp_beta':center[1]-=2*np.log(gig_mean())
    return center


def fit_independent(Y,x,model,population_center,*,progress=None,completed=(),seed=24026,
                    backend='dense'):
    """Collapsed joint log-coordinate MAP, shared theta and independent W_j."""
    internal=INTERNAL[model];center=native_center(population_center,model)
    target=make_target([(x,Y)],model=internal,prior_center=center,backend=backend)
    base=center.copy()
    if model!='gp':base=np.r_[base,np.full(len(Y),np.log(.6) if model=='stp' else np.arcsinh(1.))]
    rng=np.random.default_rng(seed);points=[]
    for sw in (.04,1.,16.,256.,4096.,65536.):
        for reverse in (False,True):
            for perturb in (False,True):
                q=base.copy();q[4]=np.log(sw)
                if perturb:q+=rng.normal(0,.4,len(q))
                z=coordinates(q,internal)
                if reverse:z[[0,1,3,4]]=z[[1,0,4,3]]
                points.append(z)
    previous={item['record']['start']:item for item in completed}
    attempts=[];candidates=[]
    def objective(z):
        value,gradient=target(z)
        return -value,-gradient
    for index,guess in enumerate(points):
        if index in previous:
            item=previous[index];z=np.asarray(item['effective_log_parameters']);record=item['record']
            np.testing.assert_allclose(target(z)[0],record['log_posterior'],atol=1e-6,rtol=0)
        else:
            result=optimize.minimize(objective,guess,jac=True,method='L-BFGS-B',
                                    options=dict(maxiter=2500,ftol=1e-13,gtol=1e-7,maxls=50,maxcor=30))
            z=result.x;value,gradient=target(z);local_restarts=0
            for _ in range(10):
                if max(abs(gradient))<.1:break
                local=optimize.minimize(objective,z,jac=True,method='L-BFGS-B',bounds=list(zip(z-.75,z+.75)),
                                       options=dict(maxiter=1000,ftol=1e-14,gtol=1e-8,maxls=50,maxcor=30))
                new_value,new_gradient=target(local.x)
                if new_value<=value+1e-8:break
                z,value,gradient=local.x,new_value,new_gradient;local_restarts+=1
            refinements=0
            if max(abs(gradient))<1:
                for _ in range(8):
                    if max(abs(gradient))<1e-5:break
                    h=1e-4;offsets=np.eye(len(z))*h
                    H=np.column_stack([(target(z+d)[1]-target(z-d)[1])/(2*h) for d in offsets]);H=(H+H.T)/2
                    if np.linalg.eigvalsh(H).max()>=0:break
                    step=linalg.solve(H,-gradient,assume_a='sym')
                    proposed=z+step;new_value,new_gradient=target(proposed)
                    if new_value<value-1e-6 or max(abs(new_gradient))>=max(abs(gradient)):break
                    z,value,gradient=proposed,new_value,new_gradient;refinements+=1
            record=dict(start=index,log_posterior=float(value),max_abs_gradient=float(max(abs(gradient))),
                        converged=bool(np.isfinite(value) and max(abs(gradient))<1e-3),
                        local_restarts=local_restarts,newton_refinements=refinements,optimizer_message=str(result.message))
            if progress:progress(record,z)
        attempts.append(record)
        if record['converged']:
            value,gradient=target(z);assert max(abs(gradient))<1e-3
            candidates.append((value,z,index))
    if not candidates:raise RuntimeError(f'{model}: no stationary fit; inspect saved optimization starts')
    value,z,index=max(candidates,key=lambda item:item[0])
    h=1e-4;offsets=np.eye(len(z))*h
    numerical=np.array([(target(z+d)[0]-target(z-d)[0])/(2*h) for d in offsets])
    H=np.column_stack([(target(z+d)[1]-target(z-d)[1])/(2*h) for d in offsets]);H=(H+H.T)/2
    eigmax=float(np.linalg.eigvalsh(H).max())
    np.testing.assert_allclose(numerical,target(z)[1],atol=5e-5,rtol=2e-3)
    if eigmax>=0:raise RuntimeError('Selected fit is not a strict local maximum')
    q=coordinates(z,internal,inverse=True)
    return dict(model=model,internal_model=internal,numerical_backend=backend,log_parameters=q,effective_log_parameters=z,
                native_prior_center=center,population_prior_center=population_center,prior_sd=2.,
                log_posterior=float(value),selected_start=index,attempts=attempts,initial_points=points,
                max_abs_gradient=float(max(abs(target(z)[1]))),max_hessian_eigenvalue=eigmax,
                gradient_check_max_error=float(max(abs(numerical-target(z)[1]))),
                W_training_mode=np.ones(len(Y)) if model=='gp' else np.exp(q[5:]))


def se(x,xp,v,w):
    return v*np.exp(-.5*w*(np.asarray(x)[:,None]-np.asarray(xp)[None,:])**2)


def context_function(y,x,xp,log_theta,model):
    """Infer only a fresh curve's log W and latent GP mean from its context.

    log_theta is frozen. The prior on W is included once, including dW/dlogW.
    The returned function is the native mu or beta, not W beta.
    """
    theta=np.asarray(log_theta);y=np.asarray(y);x=np.asarray(x);xp=np.asarray(xp)
    if theta.shape!=(5,) or y.shape!=x.shape or not np.isfinite(y).all():raise ValueError('Invalid context')
    v,a,k,w,sw=np.exp(theta);n=len(x)
    A=se(x,x,v,w)+k*np.eye(n);S=se(x,x,a,sw)
    power=2 if model=='ghp_beta' else 0
    def target(t):
        W=np.exp(t);C=W*A+W**power*S
        try:
            L=linalg.cholesky(C,lower=True);alpha=linalg.cho_solve((L,True),y)
            value=-.5*n*np.log(2*np.pi)-np.log(L.diagonal()).sum()-.5*y@alpha
            if model=='stp':value+=2.5*np.log(1.5)-special.gammaln(2.5)-2.5*t-1.5/W
            else:value+=-np.log(2)-np.log(special.kve(1.,1.))+1+t-.5*(W+1/W)
            return float(value)
        except (linalg.LinAlgError,FloatingPointError):return -np.inf
    if model=='gp':
        t=0.;report=dict(W_for_function=1.,log_W_for_function=0.,mode_search='GP fixes W=1')
    else:
        lo,hi=-12.,12.
        for _ in range(5):
            grid=np.arange(lo,hi+.125,.25);values=np.array([target(t) for t in grid]);best=float(values.max())
            if values[0]<best-50 and values[-1]<best-50:break
            if values[0]>=best-50:lo-=8.
            if values[-1]>=best-50:hi+=8.
        else:raise RuntimeError('Context log-W mode not enclosed')
        candidates=[]
        for i in range(1,len(grid)-1):
            if values[i]>=values[i-1] and values[i]>=values[i+1]:
                result=optimize.minimize_scalar(lambda t:-target(t),bounds=(grid[i-1],grid[i+1]),
                                                method='bounded',options=dict(xatol=1e-11))
                if result.success:candidates.append((-result.fun,float(result.x)))
        if not candidates:raise RuntimeError('No context log-W mode')
        value,t=max(candidates)
        step=1e-4;gradient=(target(t+step)-target(t-step))/(2*step)
        curvature=(target(t+step)-2*target(t)+target(t-step))/step**2
        if abs(gradient)>1e-3 or curvature>=0:raise RuntimeError('Nonstationary context mode')
        report=dict(W_for_function=float(np.exp(t)),log_W_for_function=t,log_posterior_at_mode=value,
                    mode_search_bounds=[lo,hi],candidate_modes=candidates,
                    max_abs_gradient=float(abs(gradient)),second_derivative=float(curvature))
    W=np.exp(t);C=W*A+W**power*S;L=linalg.cholesky(C,lower=True)
    alpha=linalg.cho_solve((L,True),y)
    fn=(W if model=='ghp_beta' else 1.)*se(xp,x,a,sw)@alpha
    return fn,report
