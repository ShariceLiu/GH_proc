"""MAP and posterior samplers with common priors and saved chain endpoints."""
import sys
from pathlib import Path
import numpy as np
import torch
from scipy import optimize
from gh_bo_core import PRIOR_MEAN, PRIOR_SD, DTYPE

sys.path.insert(0,str(Path(__file__).parent/'_vendor'))


def map_fit(posterior, rng, previous=None, starts=6):
    initial = [PRIOR_MEAN.copy()]
    if previous is not None:
        initial.append(np.asarray(previous)[0])
    initial += [PRIOR_MEAN+rng.normal(size=4)*PRIOR_SD*.5 for _ in range(starts-len(initial))]
    fits = [optimize.minimize(posterior.value_grad,q,jac=True,method='BFGS',options={'maxiter':250,'gtol':1e-5}) for q in initial]
    good = [f for f in fits if np.isfinite(f.fun) and f.fun<1e90]
    if not good:
        raise RuntimeError('All MAP starts failed')
    best = min(good,key=lambda f:f.fun)
    return best.x[None,:], {'starts':[{'objective':float(f.fun),'success':bool(f.success),'gradient_norm':float(np.linalg.norm(f.jac)),'message':str(f.message)} for f in fits], 'selected_gradient_norm':float(np.linalg.norm(best.jac))}


def slice_chain(posterior, initial, rng, warmup, samples):
    q = initial.copy()
    out = []
    widths = np.array([1.,.7,.7,.5])
    evaluations = 0
    for iteration in range(warmup+samples):
        for d in rng.permutation(4):
            threshold = posterior.log_prob(q)+np.log(rng.uniform())
            left = q[d]-widths[d]*rng.uniform()
            right = left+widths[d]
            # Randomly split a finite expansion budget (Neal stepping-out).
            j = int(rng.integers(64)); k = 63-j
            candidate = q.copy()
            while j>0:
                candidate[d]=left; evaluations+=1
                if posterior.log_prob(candidate)<=threshold: break
                left-=widths[d]; j-=1
            while k>0:
                candidate[d]=right; evaluations+=1
                if posterior.log_prob(candidate)<=threshold: break
                right+=widths[d]; k-=1
            for _ in range(10000):
                value = rng.uniform(left,right)
                candidate=q.copy(); candidate[d]=value; evaluations+=1
                if posterior.log_prob(candidate)>=threshold:
                    q=candidate; break
                if value<q[d]: left=value
                else: right=value
            else:
                raise RuntimeError('Slice shrinkage failed')
        if iteration>=warmup: out.append(q.copy())
    return np.asarray(out),evaluations


def sample(posterior, method, seed, previous=None, warmup=128, samples=256, chains=2):
    import pyro
    from pyro.infer.mcmc import MCMC,NUTS
    from pyro.ops.stats import effective_sample_size,split_gelman_rubin
    rng=np.random.default_rng(seed)
    result=[]; details=[]
    for chain in range(chains):
        initial=(np.asarray(previous)[chain].copy() if previous is not None
                 else PRIOR_MEAN+rng.normal(size=4)*PRIOR_SD*.25)
        if method=='slice':
            draws,count=slice_chain(posterior,initial,rng,warmup,samples)
            details.append({'density_evaluations':count})
        else:
            pyro.set_rng_seed(seed+chain)
            nuts=NUTS(potential_fn=lambda state:-posterior.log_prob_torch(state['q']),target_accept_prob=.85,max_tree_depth=7,full_mass=True)
            mcmc=MCMC(nuts,num_samples=samples,warmup_steps=warmup,initial_params={'q':torch.tensor(initial,dtype=DTYPE)},disable_progbar=True)
            mcmc.run()
            draws=mcmc.get_samples()['q'].numpy()
            diagnostic=mcmc.diagnostics()
            details.append({'divergences':len(diagnostic['divergences']['chain 0']), 'acceptance_rate':float(diagnostic['acceptance rate']['chain 0']), 'step_size':float(nuts.step_size)})
        result.append(draws)
    result=np.asarray(result)
    t=torch.tensor(result)
    diag={'chain_details':details,'split_rhat':split_gelman_rubin(t).tolist(),'ess':effective_sample_size(t).tolist(), 'diagnostics_note':'Two-chain short pilot; conventional split R-hat, not rank-normalized. Not a convergence certificate.'}
    return result,diag
