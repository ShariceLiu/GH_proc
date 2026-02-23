import torch
import math
from tqdm import tqdm
from scipy import special
import numpy as np

class LogKveZ(torch.autograd.Function):
    """
    log(kve(nu, z)) with gradients w.r.t. z (nu treated as constant).
    Works for scalar or tensor z. Forward/backward use SciPy on CPU. 
    # TODO: check what is log(kv(nu, z))
    """

    @staticmethod
    def forward(ctx, z: torch.Tensor, nu: float):
        nu = float(nu)

        z_det = z.detach()
        z_shape = z_det.shape

        # Convert to CPU numpy; flatten so SciPy always returns an ndarray
        z_cpu = z_det.to("cpu", dtype=torch.float64).reshape(-1).numpy()

        kve = special.kve(nu, z_cpu)                  # ndarray
        out_cpu = np.log(np.asarray(kve))            # ndarray, never numpy.float64
        out = torch.from_numpy(out_cpu).to(z.device, dtype=z.dtype).reshape(z_shape)

        ctx.save_for_backward(z_det)
        ctx.nu = nu
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        (z_saved,) = ctx.saved_tensors
        nu = ctx.nu

        z_shape = z_saved.shape
        z_cpu = z_saved.to("cpu", dtype=torch.float64).reshape(-1).numpy()

        kve0 = special.kve(nu,       z_cpu)
        kve_m = special.kve(nu - 1., z_cpu)
        kve_p = special.kve(nu + 1., z_cpu)

        # d/dz log(kve) = 1 - 0.5*(kve(nu-1)+kve(nu+1))/kve(nu)
        dlog_dz_cpu = 1.0 - 0.5 * (kve_m + kve_p) / kve0
        dlog_dz = torch.from_numpy(np.asarray(dlog_dz_cpu)).to(
            grad_out.device, dtype=grad_out.dtype
        ).reshape(z_shape)

        grad_z = grad_out * dlog_dz
        return grad_z, None

@torch.no_grad()
def step_out_1d(theta0, L, R, logy, log_target, w, max_steps=50):
    """
    Expand [L, R] in steps of size w until both ends are outside the slice:
        log_target(L) <= logy and log_target(R) <= logy
    """
    # step out left
    j = 0
    while j < max_steps:
        if log_target(L) <= logy:
            break
        L = L - w
        j += 1

    # step out right
    k = 0
    while k < max_steps:
        if log_target(R) <= logy:
            break
        R = R + w
        k += 1

    return L, R

@torch.no_grad()
def slice_sample_1d(theta0, log_target, w, max_steps=50):
    """
    One slice-sampling update with stepping-out + shrinkage.
    """
    device, dtype = theta0.device, theta0.dtype

    # slice threshold
    logp0 = log_target(theta0)
    logy = logp0 + torch.log(torch.rand((), device=device, dtype=dtype))

    # initial random-centered bracket
    v = torch.rand((), device=device, dtype=dtype) * w
    L = theta0 - v
    R = L + w

    # stepping out (optional but useful)
    L, R = step_out_1d(theta0, L, R, logy, log_target, w, max_steps=max_steps)

    # shrinkage
    while True:
        theta1 = L + torch.rand((), device=device, dtype=dtype) * (R - L)
        if log_target(theta1) >= logy:
            return theta1
        # shrink
        if theta1 < theta0:
            L = theta1
        else:
            R = theta1

def SE_cov_torch(xs, ys, v0, w, jitter=1e-7):
    xs = torch.as_tensor(xs)
    ys = torch.as_tensor(ys)
    v0 = torch.as_tensor(v0, dtype=xs.dtype, device=xs.device)
    w = torch.as_tensor(w, dtype=xs.dtype, device=xs.device)

    xs = xs.reshape(-1, 1)
    ys = ys.reshape(-1, 1)

    sqdist = (xs - ys.T).pow(2)
    K = v0 * torch.exp(-0.5 * w * sqdist)

    if jitter is not None and xs.shape[0] == ys.shape[0]:
        K = K + jitter * torch.eye(xs.shape[0], dtype=K.dtype, device=K.device)

    return K

def matern52_ard_cov_torch(xs, ys, theta0=1.0, w=1.0, jitter=1e-7):
    """
    ARD Matérn 5/2 kernel (Snóek et al. style):

      k(x,x') = theta0 * (1 + sqrt(5)*r + 5/3 * r^2) * exp(-sqrt(5)*r)
      r^2(x,x') = sum_d ((x_d - x'_d)^2 / thetad_d^2)

    Args:
      xs: (N,) or (N,D)
      ys: (M,) or (M,D)
      theta0: scalar amplitude (variance scale)
      thetad: scalar lengthscale or (D,) for ARD lengthscales
      jitter: add to diagonal when N==M (intended for K(X,X))

    Returns:
      K: (N,M) kernel matrix (torch)
    """
    thetad = w**(-0.5)
    # thetad = w
    xs = torch.as_tensor(xs)
    ys = torch.as_tensor(ys)

    if xs.ndim == 1:
        xs = xs[:, None]
    if ys.ndim == 1:
        ys = ys[:, None]

    theta0 = torch.as_tensor(theta0, dtype=xs.dtype, device=xs.device)
    thetad = torch.as_tensor(thetad, dtype=xs.dtype, device=xs.device)

    # Ensure ARD lengthscales broadcast across dimensions
    # thetad: (D,) or scalar
    xs_scaled = xs / thetad
    ys_scaled = ys / thetad

    # Pairwise squared distance in scaled space:
    # r2 = ||xs_scaled - ys_scaled||^2
    xs2 = (xs_scaled ** 2).sum(dim=1, keepdim=True)      # (N,1)
    ys2 = (ys_scaled ** 2).sum(dim=1, keepdim=True).T    # (1,M)
    r2 = xs2 + ys2 - 2.0 * (xs_scaled @ ys_scaled.T)     # (N,M)
    r2 = torch.clamp(r2, min=0.0)

    r = torch.sqrt(r2 + 1e-18)  # small epsilon avoids NaN grad at r=0

    sqrt5_r = torch.sqrt(torch.tensor(5.0, dtype=xs.dtype, device=xs.device)) * r
    K = theta0 * (1.0 + sqrt5_r + (5.0 / 3.0) * r2) * torch.exp(-sqrt5_r)

    if jitter is not None and xs.shape[0] == ys.shape[0]:
        K = K + jitter * torch.eye(xs.shape[0], dtype=K.dtype, device=K.device)

    return K

def log_kn_large_order_np(v,x):
    """ assume order is fixed, x grows with order, only return terms with x involved """
    w  = x/abs(v)
    eta = torch.sqrt(1+w**2)+ torch.log(w/(1+torch.sqrt(1+w**2)))
    return - torch.log(1+w**2)/4 - abs(v)*eta

def gh_logprob(x, f, v_0=1.0, w=1.0,  l=1.0, omega=1.0, k=0.05, a_b = 0.0, cov = 'mat'):
    p = x.numel()
    if cov == 'mat':
        K=matern52_ard_cov_torch(x,x, v_0, w)+ torch.eye(p, device=x.device, dtype=x.dtype)*k
    else:  
        K=SE_cov_torch(x,x, v_0, w)+ torch.eye(p, device=x.device, dtype=x.dtype)*k

    beta = torch.ones(p, device=x.device, dtype=x.dtype)*a_b
    mu = torch.zeros(p, device=x.device, dtype=x.dtype)

    # alpha = torch.linalg.solve(K, f - mu) # K^{-1} y
    # deltaf = (f-mu) @ alpha

    Kb = torch.linalg.solve(K, beta)
    bKb = beta @ Kb

    invK = torch.linalg.inv(K)
    deltaf = (f-mu) @ invK @ (f - mu)
    bKb = beta @ invK @ beta


    sign, logdet = torch.linalg.slogdet(K)
    
    nu = float(l - p/2)
    z = torch.sqrt((omega + deltaf)*(omega + bKb)) # .clamp_min(1e-30)
    # logkn = log_kn_large_order_np(nu, z)
    logkn = LogKveZ.apply(z, nu) - z
    # if logkn == 0.0 or not torch.isfinite(logkn):
    #     import pdb;pdb.set_trace()
    #     # extreme fallback: avoid -inf; keeps code running
    #     logkn = torch.finfo(float).tiny
    
    logp = (l - p/2)/2 * ( torch.log(omega + deltaf) - torch.log(omega + bKb) )\
        + logkn \
         + (f-mu) @ invK @ beta \
        - logdet*sign/2 \
        # - (LogKveZ.apply(torch.tensor(omega, dtype=x.dtype).clamp_min(1e-30), float(l)) - z)
        # - p/2* math.log(2*math.pi)
    
    return logp


def g_likelihood(xs, ys, k=0.05, v0=10, w=1, cov = 'mat'):
    n = xs.numel()
    if cov == 'mat':
        K=matern52_ard_cov_torch(xs,xs, v0, w)+ torch.eye(n, device=xs.device, dtype=xs.dtype)*k
    else:
        K=SE_cov_torch(xs,xs, v0, w)+ torch.eye(n, device=xs.device, dtype=xs.dtype)*k
    # jax.debug.print('K is finite: {}, v_0:{}, wls:{}', jnp.all(jnp.isfinite(K)), v_0, wls)

    sign, logdet = torch.linalg.slogdet(K)
    if sign <= 0:
        return torch.tensor(-float("inf"), device=xs.device, dtype=xs.dtype)
    
    # invK = torch.linalg.inv(K)

    mu = torch.zeros_like(ys)
    alpha = torch.linalg.solve(K, ys - mu) # K^{-1} y
    
    beta = (ys-mu) @ alpha


    return - logdet/2 - 0.5* beta #- len(xs)/2* math.log(2*math.pi)



def hmc_sample_v0_w(
    xs, ys,
    k=0.05,
    num_samples=2000,
    burn_in=500,
    step_size=0.02,
    num_leapfrog=25,
    init_v0=10.0,
    init_w=1.0,
    prior_std_logv0=1.0,   # σ_{v0}
    prior_std_logw=1.0,    # σ_{w}
    device=None,
    dtype=torch.float64,
    seed=0,
    cov ='mat'
):
    """
    HMC for GP hyperparameters (v0, w) with prior:
      log(v0) ~ N(0, prior_std_logv0^2)
      log(w)  ~ N(0, prior_std_logw^2)

    Samples are returned in u-space: [log(v0), log(w)].
    """
    torch.manual_seed(seed)

    xs = torch.as_tensor(xs, dtype=dtype, device=device)
    ys = torch.as_tensor(ys, dtype=dtype, device=device).reshape(-1)

    s0 = torch.tensor(prior_std_logv0, dtype=dtype, device=xs.device)
    s1 = torch.tensor(prior_std_logw,  dtype=dtype, device=xs.device)

    def log_prior_u(u_vec):
        # independent Gaussians on u = (log v0, log w); constants can be dropped
        return -0.5 * (u_vec[0] / s0) ** 2 - 0.5 * (u_vec[1] / s1) ** 2

    def log_target_u(u_vec):
        v0 = torch.exp(u_vec[0])
        w  = torch.exp(u_vec[1])
        ll = g_likelihood(xs, ys, k=k, v0=v0, w=w, cov=cov)   # your function
        return ll + log_prior_u(u_vec)

    def logp_and_grad(u_vec):
        u_req = u_vec.clone().detach().requires_grad_(True)
        lp = log_target_u(u_req)
        if not torch.isfinite(lp):
            return lp.detach(), torch.zeros_like(u_req)
        (g,) = torch.autograd.grad(lp, u_req)
        return lp.detach(), g.detach()

    u = torch.tensor([math.log(init_v0), math.log(init_w)], dtype=dtype, device=xs.device)

    samples = []
    accepts = 0
    total_iters = burn_in + num_samples

    for t in tqdm(range(total_iters)):
        p0 = torch.randn_like(u)

        lp0, g0 = logp_and_grad(u)
        if not torch.isfinite(lp0):
            u = u + 0.01 * torch.randn_like(u)
            continue

        u_prop = u.clone()
        p_prop = p0.clone()

        # leapfrog: p += (eps/2) * grad_logp
        p_prop = p_prop + 0.5 * step_size * g0

        ok = True
        lp_prop = None
        g_prop = None

        for i in range(num_leapfrog):
            u_prop = u_prop + step_size * p_prop

            lp_prop, g_prop = logp_and_grad(u_prop)
            if not torch.isfinite(lp_prop):
                ok = False
                break

            if i != num_leapfrog - 1:
                p_prop = p_prop + step_size * g_prop

        if ok:
            p_prop = p_prop + 0.5 * step_size * g_prop
            p_prop = -p_prop  # optional

            H0 = -lp0 + 0.5 * (p0 @ p0)
            H1 = -lp_prop + 0.5 * (p_prop @ p_prop)
            log_alpha = -(H1 - H0)

            if torch.log(torch.rand((), device=xs.device, dtype=dtype)) < log_alpha:
                u = u_prop
                accepts += 1

        if t >= burn_in:
            samples.append(u.clone())

    samples = torch.stack(samples, dim=0) if samples else torch.empty((0, 2), device=xs.device, dtype=dtype)
    accept_rate = accepts / max(1, total_iters)
    return samples, accept_rate

def hmc_sample_v0_w_gh(
    xs, ys,
    k=0.05, l = 1.0, omega = 1.0,
    num_samples=2000,
    burn_in=500,
    step_size=0.02,
    num_leapfrog=25,
    init_v0=10.0,
    init_w=1.0,
    init_beta = 0.0,
    prior_std_logv0=1.0,   # σ_{v0}
    prior_std_logw=1.0,    # σ_{w}
    prior_std_beta=1.0,    # σ_{w}
    device=None,
    dtype=torch.float64,
    seed=0,
    cov ='mat'
):
    """
    HMC for GP hyperparameters (v0, w) with prior:
      log(v0) ~ N(0, prior_std_logv0^2)
      log(w)  ~ N(0, prior_std_logw^2)

    Samples are returned in u-space: [log(v0), log(w)].
    """
    torch.manual_seed(seed)

    xs = torch.as_tensor(xs, dtype=dtype, device=device)
    ys = torch.as_tensor(ys, dtype=dtype, device=device).reshape(-1)

    s0 = torch.tensor(prior_std_logv0, dtype=dtype, device=xs.device)
    s1 = torch.tensor(prior_std_logw,  dtype=dtype, device=xs.device)
    s2 = torch.tensor(prior_std_beta,  dtype=dtype, device=xs.device)

    def log_prior_u(u_vec):
        # independent Gaussians on u = (log v0, log w); constants can be dropped
        return -0.5 * (u_vec[0] / s0) ** 2 - 0.5 * (u_vec[1] / s1) ** 2 - 0.5 * (u_vec[2] / s2) ** 2

    def log_target_u(u_vec):
        v0 = torch.exp(u_vec[0])
        w  = torch.exp(u_vec[1])
        beta = u_vec[2]
        ll = gh_logprob(xs, ys, k=k, v_0=v0, w=w, cov=cov, l=l, omega=omega, a_b=beta)   # your function
        return ll + log_prior_u(u_vec)

    def logp_and_grad(u_vec):
        u_req = u_vec.clone().detach().requires_grad_(True)
        lp = log_target_u(u_req)
        if not torch.isfinite(lp):
            return lp.detach(), torch.zeros_like(u_req)
        (g,) = torch.autograd.grad(lp, u_req)
        return lp.detach(), g.detach()

    u = torch.tensor([math.log(init_v0), math.log(init_w), init_beta], dtype=dtype, device=xs.device)

    samples = []
    accepts = 0
    total_iters = burn_in + num_samples

    for t in tqdm(range(total_iters)):
        p0 = torch.randn_like(u)

        lp0, g0 = logp_and_grad(u)
        if not torch.isfinite(lp0):
            u = u + 0.01 * torch.randn_like(u)
            continue

        u_prop = u.clone()
        p_prop = p0.clone()

        # leapfrog: p += (eps/2) * grad_logp
        p_prop = p_prop + 0.5 * step_size * g0

        ok = True
        lp_prop = None
        g_prop = None

        for i in range(num_leapfrog):
            u_prop = u_prop + step_size * p_prop

            lp_prop, g_prop = logp_and_grad(u_prop)
            if not torch.isfinite(lp_prop):
                ok = False
                break

            if i != num_leapfrog - 1:
                p_prop = p_prop + step_size * g_prop

        if ok:
            p_prop = p_prop + 0.5 * step_size * g_prop
            p_prop = -p_prop  # optional

            H0 = -lp0 + 0.5 * (p0 @ p0)
            H1 = -lp_prop + 0.5 * (p_prop @ p_prop)
            log_alpha = -(H1 - H0)

            if torch.log(torch.rand((), device=xs.device, dtype=dtype)) < log_alpha:
                u = u_prop
                accepts += 1

        if t >= burn_in:
            samples.append(u.clone())

    samples = torch.stack(samples, dim=0) if samples else torch.empty((0, 3), device=xs.device, dtype=dtype)
    accept_rate = accepts / max(1, total_iters)
    return samples, accept_rate


def hmc_sample_v0_w_gh_w_matrix(
    xs, ys,
    k=0.05, l=1.0, omega=1.0,
    num_samples=2000,
    burn_in=500,
    step_size=0.02,
    num_leapfrog=25,
    init_v0=10.0,
    init_w=1.0,
    init_beta=0.0,
    prior_std_logv0=1.0,
    prior_std_logw=1.0,
    prior_std_beta=1.0,
    mass_diag=None,             # <-- diagonal mass matrix entries (length 3). If None, use prior variances.
    device=None,
    dtype=torch.float64,
    seed=0,
    cov='mat',
):
    """
    Samples u = [log(v0), log(w), beta] using HMC with diagonal mass matrix M.

    Prior in u-space:
      log(v0) ~ N(0, prior_std_logv0^2)
      log(w)  ~ N(0, prior_std_logw^2)
      beta    ~ N(0, prior_std_beta^2)

    Returns:
      samples_u: (num_samples, 3) tensor of [log(v0), log(w), beta]
      accept_rate: float
    """
    torch.manual_seed(seed)

    xs = torch.as_tensor(xs, dtype=dtype, device=device)
    ys = torch.as_tensor(ys, dtype=dtype, device=device).reshape(-1)

    # Prior stds on u-space
    s0 = torch.tensor(prior_std_logv0, dtype=dtype, device=xs.device)
    s1 = torch.tensor(prior_std_logw,  dtype=dtype, device=xs.device)
    s2 = torch.tensor(prior_std_beta,  dtype=dtype, device=xs.device)

    def log_prior_u(u_vec):
        # constants dropped
        return -0.5 * (u_vec[0] / s0) ** 2 - 0.5 * (u_vec[1] / s1) ** 2 - 0.5 * (u_vec[2] / s2) ** 2

    def log_target_u(u_vec):
        v0 = torch.exp(u_vec[0])   # positivity
        w  = torch.exp(u_vec[1])   # positivity
        beta = u_vec[2]            # unconstrained
        ll = gh_logprob(xs, ys, k=k, v_0=v0, w=w, cov=cov, l=l, omega=omega, a_b=beta)
        return ll + log_prior_u(u_vec)

    def logp_and_grad(u_vec):
        u_req = u_vec.clone().detach().requires_grad_(True)
        lp = log_target_u(u_req)
        if not torch.isfinite(lp):
            return lp.detach(), torch.zeros_like(u_req)
        (g,) = torch.autograd.grad(lp, u_req)
        return lp.detach(), g.detach()

    # --- Diagonal mass matrix M (preconditioner) ---
    if mass_diag is None:
        # A reasonable default: match parameter scale under the prior (variance)
        mass_diag = torch.tensor([s0**2, s1**2, s2**2], dtype=dtype, device=xs.device).flatten()
    else:
        mass_diag = torch.as_tensor(mass_diag, dtype=dtype, device=xs.device).flatten()

    if mass_diag.numel() != 3:
        raise ValueError("mass_diag must have length 3 for [log(v0), log(w), beta].")
    if torch.any(mass_diag <= 0):
        raise ValueError("mass_diag entries must be positive.")

    inv_mass = 1.0 / mass_diag
    sqrt_mass = torch.sqrt(mass_diag)

    def kinetic(p_vec):
        # K(p) = 0.5 * p^T M^{-1} p for diag M
        return 0.5 * torch.sum((p_vec * p_vec) * inv_mass)

    # initial state
    u = torch.tensor([math.log(init_v0), math.log(init_w), init_beta], dtype=dtype, device=xs.device)

    samples = []
    accepts = 0
    total_iters = burn_in + num_samples

    for t in tqdm(range(total_iters)):
        # p ~ N(0, M) i.e. p_i = sqrt(m_i) * z_i
        p0 = torch.randn_like(u) * sqrt_mass

        lp0, g0 = logp_and_grad(u)
        if not torch.isfinite(lp0):
            u = u + 0.01 * torch.randn_like(u)
            continue

        u_prop = u.clone()
        p_prop = p0.clone()

        # half step momentum: p += (eps/2) * grad_logp
        p_prop = p_prop + 0.5 * step_size * g0

        ok = True
        lp_prop = None
        g_prop = None

        for i in range(num_leapfrog):
            # full step position: u += eps * M^{-1} p
            u_prop = u_prop + step_size * (p_prop * inv_mass)

            lp_prop, g_prop = logp_and_grad(u_prop)
            if not torch.isfinite(lp_prop):
                ok = False
                break

            if i != num_leapfrog - 1:
                p_prop = p_prop + step_size * g_prop

        if ok:
            p_prop = p_prop + 0.5 * step_size * g_prop
            p_prop = -p_prop  # optional but standard

            H0 = -lp0 + kinetic(p0)
            H1 = -lp_prop + kinetic(p_prop)
            log_alpha = -(H1 - H0)

            if torch.log(torch.rand((), device=xs.device, dtype=dtype)) < log_alpha:
                u = u_prop
                accepts += 1

        if t >= burn_in:
            samples.append(u.clone())

    samples = torch.stack(samples, dim=0) if samples else torch.empty((0, 3), device=xs.device, dtype=dtype)
    accept_rate = accepts / max(1, total_iters)
    return samples, accept_rate