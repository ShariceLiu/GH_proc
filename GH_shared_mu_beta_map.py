"""Two independent shared GP functions: mu + (W/E[W])*h_beta.

The native beta is h_beta/E[W]. Gaussian functions are integrated exactly.
Weighted curve-space QR reduces J*n observations to a 2*n Gaussian system
plus J-2 residual directions. SE feature QR protects tiny noise variances.
q=log(v, a_mu, a_h_beta, k, w, w_mu, w_beta, W_1,...,W_J).
"""
import numpy as np
from scipy import linalg, optimize, special
from GH_shared_function_map import se_factor, gig_mean


def shared_mu_beta_calculation(Y,x,q,*,lam=1.,omega=1.,posterior=False):
    J,n=Y.shape
    v,am,ab,k,w,wm,wb=np.exp(q[:7]);W=np.exp(q[7:])
    c0=gig_mean(lam,omega);c=W/c0
    F,D=se_factor(x,v,w);Gm,Dm=se_factor(x,am,wm);Gb,Db=se_factor(x,ab,wb)
    design=np.column_stack((1/np.sqrt(W),c/np.sqrt(W)))
    U,R=linalg.qr(design,mode='economic')
    Yw=Y/np.sqrt(W[:,None]); projected=U.T@Yw
    orthogonal=Yw-U@projected
    base=np.column_stack((F,np.sqrt(k)*np.eye(n)))
    LA=linalg.qr(base.T,mode='economic')[1].T
    shared_m=np.vstack([R[i,0]*Gm for i in range(2)])
    shared_b=np.vstack([R[i,1]*Gb for i in range(2)])
    LH=linalg.qr(np.column_stack((linalg.block_diag(base,base),shared_m,shared_b)).T,mode='economic')[1].T
    solveA=lambda b:linalg.solve_triangular(LA,b,lower=True)
    solveH=lambda b:linalg.solve_triangular(LH,b,lower=True)
    E=solveA(orthogonal.T);z=solveH(projected.ravel())
    alpha=linalg.solve_triangular(LH.T,z,lower=False).reshape(2,n)
    Hm,Hb=solveH(shared_m),solveH(shared_b)
    dm,db=solveH(np.vstack([R[i,0]*Dm for i in range(2)])),solveH(np.vstack([R[i,1]*Db for i in range(2)]))
    agm,agb=z@Hm,z@Hb
    FA,DA=solveA(F),solveA(D)
    # Project precision-times-data onto kernel factors without subtracting
    # large inverse covariance products.
    pf=E.T@FA+U@(alpha@F);pd=E.T@DA+U@(alpha@D)
    HF=solveH(linalg.block_diag(F,F));HD=solveH(linalg.block_diag(D,D))
    invLA,invLH=solveA(np.eye(n)),solveH(np.eye(2*n))
    tau=linalg.solve_triangular(LA.T,E,lower=False).T+U@alpha
    grad=np.zeros_like(q)
    grad[:7]=[
        .5*(np.sum(pf**2)-(J-2)*np.sum(FA**2)-np.sum(HF**2)),
        .5*(agm@agm-np.sum(Hm**2)),
        .5*(agb@agb-np.sum(Hb**2)),
        .5*k*(np.sum(tau**2)-(J-2)*np.sum(invLA**2)-np.sum(invLH**2)),
        np.sum(pf*pd)-(J-2)*np.sum(FA*DA)-np.sum(HF*HD),
        agm@(z@dm)-np.sum(Hm*dm),
        agb@(z@db)-np.sum(Hb*db)]
    # W derivative: residual covariance contribution, then mean-loading
    # contribution. This accounts for QR's changing subspace analytically.
    HA=solveH(linalg.block_diag(LA,LA)).reshape(2*n,2,n)
    trace_A=np.einsum('ran,rbn->ab',HA,HA)
    tau_Aroot=E.T+U@(alpha@LA)
    grad[7:]=.5*(np.sum(tau_Aroot**2,axis=1)-n*(1-np.sum(U**2,axis=1))
                  -np.einsum('ja,ab,jb->j',U,trace_A,U))
    B0=np.vstack((Gb,np.zeros_like(Gb)));B1=np.vstack((np.zeros_like(Gb),Gb))
    trace_beta=np.array([np.sum(solveH(B)*Hb) for B in (B0,B1)])
    tauGb=E.T@solveA(Gb)+U@(alpha@Gb)
    grad[7:]+=c/np.sqrt(W)*(tauGb@agb-U@trace_beta)
    logdet=2*((J-2)*np.log(abs(LA.diagonal())).sum()+np.log(abs(LH.diagonal())).sum())+n*np.log(W).sum()
    loglike=-.5*(J*n*np.log(2*np.pi)+logdet+np.sum(E**2)+z@z)
    result=dict(log_likelihood=float(loglike),gradient=grad)
    if posterior:
        result.update(mu=Gm@agm,beta=Gb@agb/c0,W=W)
        cm=Gm@Gm.T-(Gm@Hm.T)@(Hm@Gm.T)
        cb=(Gb@Gb.T-(Gb@Hb.T)@(Hb@Gb.T))/c0**2
        cross=-(Gm@Hm.T)@(Hb@Gb.T)/c0
        result.update(mu_cov=(cm+cm.T)/2,beta_cov=(cb+cb.T)/2,mu_beta_cov=cross)
    return result


def make_target(Y,x,center,*,prior_sd=2.,lam=1.,omega=1.):
    Y,x,center=np.asarray(Y,float),np.asarray(x,float),np.asarray(center,float)
    if Y.ndim!=2 or Y.shape[0]<2 or Y.shape[1]!=len(x) or center.shape!=(7,):
        raise ValueError('Need >=2 curves on a common grid and seven log prior centers')
    constant=-7*np.log(prior_sd)-3.5*np.log(2*np.pi)
    gig_constant=-np.log(2)-np.log(special.kve(lam,omega))+omega
    def target(q):
        if np.shape(q)!=(7+len(Y),) or not np.isfinite(q).all():return -np.inf,np.zeros(7+len(Y))
        try:
            with np.errstate(over='raise',invalid='raise',divide='raise'):
                calc=shared_mu_beta_calculation(Y,x,q,lam=lam,omega=omega)
                W=np.exp(q[7:]);standardized=(q[:7]-center)/prior_sd
                value=calc['log_likelihood']+constant-.5*standardized@standardized
                value+=np.sum(gig_constant+lam*q[7:]-.5*omega*(W+1/W))
                grad=calc['gradient'];grad[:7]-=standardized/prior_sd
                grad[7:]+=lam-.5*omega*(W-1/W)
                if not np.isfinite(value) or not np.isfinite(grad).all():raise FloatingPointError()
                return float(value),grad
        except (ValueError,FloatingPointError,np.linalg.LinAlgError):
            return -np.inf,np.zeros(7+len(Y))
    return target


def fit_shared_mu_beta(Y,x,center,initial_points,*,progress=None,completed=None):
    target=make_target(Y,x,center);attempts=[];candidates=[]
    # Cache combined value/gradient evaluation within scipy's objective call.
    def objective(q):
        value,grad=target(q);return -value,-grad
    for i,guess in enumerate(initial_points):
        if completed is not None and i<len(completed):
            item=completed[i];q=np.asarray(item['log_parameters']);record=item['record']
            value,grad=target(q)
            np.testing.assert_allclose(value,record['log_posterior'],atol=1e-5,rtol=0)
            attempts.append(record)
            if record['success'] and max(abs(grad))<1e-3:candidates.append((value,q,i))
            continue
        fit=optimize.minimize(objective,guess,jac=True,method='L-BFGS-B',
                              options=dict(maxiter=1800,ftol=1e-13,gtol=2e-6,maxls=40,maxcor=30))
        value,grad=target(fit.x);restarts=0
        for _ in range(16):
            if max(abs(grad))<1:break
            local=optimize.minimize(objective,fit.x,jac=True,method='L-BFGS-B',
                                    bounds=list(zip(fit.x-.75,fit.x+.75)),
                                    options=dict(maxiter=800,ftol=1e-14,gtol=2e-6,maxls=50,maxcor=30))
            nv,ng=target(local.x)
            if nv<=value+1e-7:break
            fit,value,grad=local,nv,ng;restarts+=1
        refinements=0
        if max(abs(grad))<2:
            for _ in range(5):
                if max(abs(grad))<1e-5:break
                ds=1e-4*np.eye(len(fit.x))
                H=np.column_stack([(target(fit.x+d)[1]-target(fit.x-d)[1])/2e-4 for d in ds]);H=(H+H.T)/2
                if np.linalg.eigvalsh(H).max()>=0:break
                proposal=fit.x+np.linalg.solve(H,-grad);nv,ng=target(proposal)
                if nv<value-1e-5 or max(abs(ng))>=max(abs(grad)):break
                fit.x,value,grad=proposal,nv,ng;refinements+=1
        ok=bool(np.isfinite(value) and max(abs(grad))<1e-3)
        record=dict(start=i,success=ok,log_posterior=value,max_abs_gradient=float(max(abs(grad))),
                    message=str(fit.message),iterations=int(fit.nit),local_restarts=restarts,newton_refinements=refinements)
        attempts.append(record)
        if progress:progress(record,fit.x)
        if ok:candidates.append((value,fit.x.copy(),i))
    if not candidates:raise RuntimeError(f'No stationary fit: {attempts}')
    value,q,selected=max(candidates,key=lambda c:c[0])
    ds=.005*np.eye(len(q))
    fd=np.array([(-target(q-3*d)[0]+9*target(q-2*d)[0]-45*target(q-d)[0]+45*target(q+d)[0]
                  -9*target(q+2*d)[0]+target(q+3*d)[0])/.3 for d in ds])
    np.testing.assert_allclose(fd,target(q)[1],atol=1e-4,rtol=2e-3)
    ds=1e-4*np.eye(len(q))
    H=np.column_stack([(target(q+d)[1]-target(q-d)[1])/2e-4 for d in ds]);H=(H+H.T)/2
    eig=float(np.linalg.eigvalsh(H).max())
    if eig>=0:raise RuntimeError('Selected stationary point is not a strict local maximum')
    calc=shared_mu_beta_calculation(Y,x,q,posterior=True)
    return dict(model='ghp_mu_beta',log_parameters=q,log_posterior=value,prior_center=center,prior_sd=2.,
                lam=1.,omega=1.,attempts=attempts,selected_start=selected,max_hessian_eigenvalue=eig,
                max_abs_gradient=float(max(abs(target(q)[1]))),max_gradient_check_error=float(max(abs(fd-target(q)[1]))),
                **{key:calc[key] for key in ('mu','beta','mu_cov','beta_cov','mu_beta_cov','W')})
