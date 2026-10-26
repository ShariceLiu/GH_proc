import numpy as np
import torch
from torch.distributions import Normal
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib import cm

from scipy.special import kve
from scipy import stats
import math

from tools.tools_bay import *

std_normal = Normal(0.0, 1.0)

def tgt_fn(x):
    return -(x - 1) ** 2 * torch.sin(3 * x + 5 / x + 1)
    # return x**2
    # return - torch.sin(x)
    # return  -(x - 1) ** 2 # torch.sin(3*x)

def gig_logpdf(w: torch.Tensor, lam, omega):
    """
    Symmetric GIG with chi=psi=omega:
      p(w) = 1/(2 K_lam(omega)) * w^(lam-1) * exp(-0.5*(omega/w + omega*w)),  w>0
    """
    if torch.any(w <= 0):
        raise ValueError("w must be > 0")

    # kv is SciPy (numpy), so compute the normalizer as a python float
    lam_f = float(torch.as_tensor(lam).detach().cpu())
    omega_f = float(torch.as_tensor(omega).detach().cpu())

    # Stable log(K_v(omega)) via kve: logK = log(kve(v,omega)) - omega
    v = abs(lam_f)
    kve_val = kve(v, omega_f)
    if kve_val == 0.0 or not np.isfinite(kve_val):
        # extreme fallback: avoid -inf; keeps code running
        kve_val = np.finfo(float).tiny

    logK = np.log(kve_val) - omega_f
    logK = torch.tensor(logK, dtype=w.dtype, device=w.device)

    return (-torch.log(torch.tensor(2.0, dtype=w.dtype, device=w.device)) - logK
            + (lam - 1.0) * torch.log(w)
            - 0.5 * (omega * w + omega / w))

def gig_quadrature_gauss_laguerre(
    J: int,
    lam,
    omega,
    scale: float | None = None,
    renormalize: bool = True,
    dtype: torch.dtype = torch.float64,
    device: torch.device | str = "cpu",
):
    lam = torch.as_tensor(lam, dtype=dtype, device=device)
    omega_param = torch.as_tensor(omega, dtype=dtype, device=device)

    # In your simplified symmetric setting
    chi = omega_param
    psi = omega_param

    if float(chi.detach().cpu()) <= 0 or float(psi.detach().cpu()) <= 0:
        raise ValueError("chi and psi must be > 0 for GIG.")

    # Laguerre nodes/weights (numpy), then move to torch
    x_np, lag_w_np = np.polynomial.laguerre.laggauss(J)
    x = torch.tensor(x_np, dtype=dtype, device=device)            # (J,)
    lag_w = torch.tensor(lag_w_np, dtype=dtype, device=device)    # (J,)

    # Choose scaling c
    if scale is None:
        c = torch.sqrt(chi / psi)  # = 1 for chi=psi
    else:
        c = torch.as_tensor(scale, dtype=dtype, device=device)

    w_nodes = c * x  # (J,)

    # α_i = c * ω_i * exp(x_i) * p(w_i)
    logp = gig_logpdf(w_nodes, lam, omega_param)
    w_weights = c * lag_w * torch.exp(x + logp)

    if renormalize:
        w_weights = w_weights / w_weights.sum()

    return w_nodes, w_weights

# def gaussianEI(f_best, m, sigma, eps=1e-12):
#     sigma = torch.clamp(sigma, min=eps)
#     u = (f_best - m) / sigma
#     # print("u min/max:", u.min().item(), u.max().item())
#     Phi = std_normal.cdf(u)
#     phi = torch.exp(std_normal.log_prob(u))

#     # Phi = 0.5 * (1.0 + torch.erf(u / math.sqrt(2.0)))
#     # phi = torch.exp(-0.5 * u**2) / math.sqrt(2.0 * math.pi)
#     return (f_best - m) * Phi + sigma * phi

def gaussianEI(f_best, m, sigma, xi=0.1, eps=1e-12):
    sigma = torch.clamp(sigma, min=eps)
    thresh = f_best - xi
    u = (thresh - m) / sigma
    # Phi = 0.5 * (1.0 + torch.erf(u / 1.41421356237))
    # phi = torch.exp(-0.5 * u**2) / 2.50662827463
    Phi = 0.5 * (1.0 + torch.erf(u /  math.sqrt(2)))
    phi = torch.exp(-0.5*u**2)/ math.sqrt(2*math.pi)
    return (thresh - m) * Phi + sigma * phi



def mu_beta_sigma_fn(train_xs, train_ys, x, l=-1, omega=1, w=1, v_0=10, a_b=1, b_b=0, k=0.05, cov = 'mat'):
    train_xs = torch.as_tensor(train_xs)
    train_ys = torch.as_tensor(train_ys)
    x = torch.as_tensor(x)

    n = train_ys.numel()
    I = torch.eye(n, dtype=train_xs.dtype, device=train_xs.device)
    if cov == 'mat':
        K11 = matern52_ard_cov_torch(train_xs, train_xs, v_0, w) + k * I
        K12 = matern52_ard_cov_torch(train_xs, x.reshape(-1), v_0, w)   # (n,1) if x is scalar
        K22 = matern52_ard_cov_torch(x.reshape(-1), x.reshape(-1), v_0, w)  # (1,1)
    else: # normal exp
        K11 = SE_cov_torch(train_xs, train_xs, v_0, w) + k * torch.eye(n, dtype=train_xs.dtype, device=train_xs.device)
        K12 = SE_cov_torch(train_xs, x.reshape(-1), v_0, w)   # (n,1) if x is scalar
        K22 = SE_cov_torch(x.reshape(-1), x.reshape(-1), v_0, w)  # (1,1)

    # K11 = matern52_ard_cov_torch(train_xs, train_xs, theta0=1.0, thetad=0.5) + k * torch.eye(n, dtype=train_xs.dtype, device=train_xs.device)
    # K12 = matern52_ard_cov_torch(train_xs, x.reshape(-1), theta0=1.0, thetad=0.5)
    # K22 = matern52_ard_cov_torch(x.reshape(-1), x.reshape(-1), theta0=1.0, thetad=0.5)

    # print("K12 max abs:", K12.abs().max().item(), "K12 min abs:", K12.abs().min().item())

    beta1 = torch.ones(n, dtype=train_xs.dtype, device=train_xs.device) * a_b + train_xs * b_b
    beta2 = torch.ones(1, dtype=train_xs.dtype, device=train_xs.device) * a_b + x.reshape(-1) * b_b

    

    l21 = l - n / 2

    # invK11 = torch.linalg.inv(K11)
    # invK11 = torch.linalg.solve(K11, I) # TODO: solve this gives negative K221: tensor(4373.6427, dtype=torch.float64) tensor(0.2771, dtype=torch.float64) 0.1
    # gamma21 = omega + train_ys @ (invK11 @ train_ys)
    # delta21 = omega + beta1 @ (invK11 @ beta1)
    # mu21 = K12.T @ invK11 @ train_ys
    # K221 = K22 - K12.T @ invK11 @ K12
    # beta21 = beta2 - K12.T @ invK11 @ beta1


    # K11 should be symmetric
    K11 = 0.5 * (K11 + K11.T)

    # robust jitter: scale it to the matrix size
    jitter = 1e-4 * K11.diag().mean()
    K11 = K11 + jitter * torch.eye(K11.shape[0], device=K11.device, dtype=K11.dtype)

    L = torch.linalg.cholesky(K11)  # (n,n), lower-triangular

    # v = L^{-1} K12   (triangular solve)
    v = torch.linalg.solve_triangular(L, K12, upper=False)

    # Schur complement: K22 - ||v||^2
    K221 = K22 - (v.T @ v)   # (m,m); for scalar x this is (1,1)
    # optional tiny clamp only if it's like -1e-12:
    K221 = 0.5 * (K221 + K221.T)

    if K221 < 0:
        print(K221)
        K221 = 1e-4


    # --- ensure column vectors for clean algebra ---
    y = train_ys.reshape(-1, 1)     # (n,1)
    b1 = beta1.reshape(-1, 1)       # (n,1)
    # K11^{-1} y and K11^{-1} beta1 (stable)
    alpha_y = torch.cholesky_solve(y, L)    # (n,1)
    alpha_b = torch.cholesky_solve(b1, L)   # (n,1)

    # gamma21 = omega + y^T K11^{-1} y
    gamma21 = omega + (y.T @ alpha_y).squeeze()

    # delta21 = omega + beta1^T K11^{-1} beta1
    delta21 = omega + (b1.T @ alpha_b).squeeze()

    # mu21 = K12^T K11^{-1} y
    mu21 = (K12.T @ alpha_y)  # (m,1) ; if x is scalar, (1,1)

    # beta21 = beta2 - K12^T K11^{-1} beta1
    beta2_col = beta2.reshape(-1, 1)  # (m,1) if needed
    beta21 = beta2_col - (K12.T @ alpha_b)  # (m,1)

    omega21 = gamma21 * delta21

    

    return mu21.squeeze(), beta21.squeeze(), K221.squeeze(), l21, omega21

def g_regression(train_xs, train_ys, x, w = 1, v_0 = 10, k =0.05, phi=0.0, cov = 'mat'):
    train_xs = torch.as_tensor(train_xs)
    train_ys = torch.as_tensor(train_ys)
    x = torch.as_tensor(x)

    n = train_ys.numel()
    if cov == 'mat':
        K11 = matern52_ard_cov_torch(train_xs, train_xs, v_0, w) + k * torch.eye(n, dtype=train_xs.dtype, device=train_xs.device)
        K12 = matern52_ard_cov_torch(train_xs, x.reshape(-1), v_0, w)   # (n,1) if x is scalar
        K22 = matern52_ard_cov_torch(x.reshape(-1), x.reshape(-1), v_0, w)  # (1,1)
    else: # normal exp
        K11 = SE_cov_torch(train_xs, train_xs, v_0, w) + k * torch.eye(n, dtype=train_xs.dtype, device=train_xs.device)
        K12 = SE_cov_torch(train_xs, x.reshape(-1), v_0, w)   # (n,1) if x is scalar
        K22 = SE_cov_torch(x.reshape(-1), x.reshape(-1), v_0, w)  # (1,1)

    invK11 = torch.linalg.inv(K11)
    n = train_ys.numel()
    # v21 = v + n1
    y = (train_ys - phi).reshape(n, 1)
    # import pdb; pdb.set_trace()
    K123 = K12.flatten()
    mu21 = phi+ K123.T @ (invK11 @ y)
    K221 = K22 - K123.T @ (invK11 @ K123)

    return mu21, K221

def g_slice_sampling(train_xs, train_ys, k = 0.05, iterations =20, s = 1, scale=1.0,
                     init_logw_std=0.1, init_logv0_std=0.1, cov = 'mat', max_steps_out = 20,
                     logv_prior_std = 2.0, logw_prior_std = 2.0):
    
    def step_out(th_min, th_max, logy, log_target, scale, g_cand, f_cand, fixed_para, update_para, max_steps=max_steps_out):
        """
        Expand bracket [th_min, th_max] so that both ends fall outside slice.
        update_para = ['v0', 'w']
        """
        # step out left
        j = 0
        while j < max_steps:
            if log_target(log_update_para = th_min, log_fixed_para = fixed_para, f_cand = f_cand, g_cand = g_cand, update_para = update_para) <= logy:
                break
            th_min = th_min - scale
            j += 1

        # step out right
        k = 0
        while k < max_steps:
            if log_target(log_update_para = th_max, log_fixed_para = fixed_para, f_cand = f_cand, g_cand = g_cand, update_para = update_para) <= logy:
                break
            th_max = th_max + scale
            k += 1

        return th_min, th_max
    
    def log_target(log_update_para, log_fixed_para, f_cand, g_cand, update_para = 'w'):
        if update_para == 'w':
            w_cand, v0_cand = torch.exp(log_update_para), torch.exp(log_fixed_para)
            std_prior = logw_prior_std
        else:
            w_cand, v0_cand = torch.exp(log_fixed_para), torch.exp(log_update_para)
            std_prior = logv_prior_std

        if cov=='mat':
            Sig = matern52_ard_cov_torch(train_xs, train_xs, v0_cand, w_cand)
        else: # se
            Sig = SE_cov_torch(train_xs, train_xs, v0_cand, w_cand)

        Sig_S_inv = torch.linalg.solve(S_th + Sig, I)
        sign, logabsdet = torch.linalg.slogdet(Sig + S_th)

        return (
            - (1.0/2.0/k)*torch.dot(train_ys - f_cand, train_ys - f_cand)
            - 0.5*logabsdet
            - 0.5*(g_cand @ Sig_S_inv @ g_cand)
            - 0.5* log_update_para**2/ std_prior**2
        )

    
    def matrix_inverse(Sigth, Sth, eps = 1e-6):

        ns = Sigth.shape[0]
        I_A = torch.eye(ns, device=Sigth.device, dtype=Sigth.dtype)
        Sigth = 0.5*(Sigth + Sigth.T) + eps * I_A # symmetrize + jitter

        A = Sigth + Sth
        A = 0.5 * (A + A.T) + eps * I_A

        # import pdb;pdb.set_trace()

        L_A = torch.linalg.cholesky(A)
        Y = torch.cholesky_solve(Sth, L_A)
        Rth = Sth - Sth @ Y
        Rth = 0.5*(Rth + Rth.T) + eps*I_A

        return Rth
    
    # sample w, v0
    f = train_ys.clone() + torch.randn_like(train_ys)
    n = len(train_xs)

    device, dtype = f.device, f.dtype

    scale = torch.tensor(scale, device=device, dtype=dtype)
    log_w = (torch.randn((), device=device, dtype=dtype)*init_logw_std)
    log_v0 = (torch.randn((), device=device, dtype=dtype) * init_logv0_std)
    # log_v0 = torch.tensor(0.0, device=device, dtype=dtype)

    log_wis, log_v0is = np.zeros(iterations), np.zeros(iterations)

    S_th = s * torch.eye(n, device=device, dtype=dtype)
    
    for i in tqdm(range(iterations)):
        g = f + torch.randn_like(f)* math.sqrt(s)
        
        # sample w first
        v0, w = torch.exp(log_v0), torch.exp(log_w)
        if cov == 'mat':
            Sig_th = matern52_ard_cov_torch(train_xs, train_xs, v0, w)
        else:
            Sig_th = SE_cov_torch(train_xs, train_xs, v0, w)

        I = torch.eye(n, device=S_th.device, dtype=S_th.dtype)


        R_th = matrix_inverse(Sigth=Sig_th, Sth=S_th)
        # R_th = torch.linalg.inv( torch.linalg.inv(S_th) +  torch.linalg.inv(Sig_th))
        L_R = torch.linalg.cholesky(R_th)
        
        
        # Sig_S_inv = torch.linalg.inv( S_th + Sig_th )
        Sig_S_inv = torch.linalg.solve( S_th + Sig_th, I )
        m_th_g = Sig_th @ Sig_S_inv @ g

        eta = torch.linalg.solve( L_R , (f - m_th_g))

        v = torch.rand(size =(), device=device, dtype=dtype)*scale
        th_min = log_w -v
        th_max = th_min + scale

        u = torch.rand(size=(), device=device, dtype=dtype)

        sign, logabsdet = torch.linalg.slogdet(Sig_th + S_th)
        logt = torch.log(u) - (1.0/2.0/k)*torch.dot(train_ys - f, train_ys - f) \
            - 0.5*logabsdet - 0.5 * (g @ Sig_S_inv @ g) \
            - 0.5* log_w**2/ logw_prior_std**2 # - math.log(2*math.pi) - 0.5*n*math.log(k) 
        
        # print(u)

        th_min, th_max = step_out(th_min, th_max, logt, log_target, scale, g_cand=g, f_cand=f, fixed_para=log_v0, update_para='w', max_steps=max_steps_out)
        # print(th_min, th_max)
        # import pdb;pdb.set_trace()
        
        while True:
            log_w_next = torch.rand(size = (), device=device, dtype=dtype)*(th_max - th_min) + th_min
            w_next = torch.exp(log_w_next)
            
            if cov == 'mat':
                Sig_th_next = matern52_ard_cov_torch(train_xs, train_xs, v0, w_next)
            else:
                Sig_th_next = SE_cov_torch(train_xs, train_xs, v0, w_next)

            
            R_th_next = matrix_inverse(Sig_th_next, S_th)
            # R_th_next = torch.linalg.inv( torch.linalg.inv(S_th) +  torch.linalg.inv(Sig_th_next))
            L_R_next = torch.linalg.cholesky(R_th_next)

            Sig_S_inv_next = torch.linalg.solve( S_th + Sig_th_next, I )
            # Sig_S_inv_next = torch.linalg.inv( S_th + Sig_th_next )
            m_th_g_next = Sig_th_next @ Sig_S_inv_next @ g
            f_next = L_R_next @ eta + m_th_g_next

            sign2, logabsdet2 = torch.linalg.slogdet(Sig_th_next + S_th)
            logt_next = - (1/2/k)*torch.dot(train_ys - f_next, train_ys - f_next) \
                - 0.5*logabsdet2 - 0.5 * g @ Sig_S_inv_next @ g - 0.5* log_w_next**2/ logw_prior_std**2

            if logt_next > logt:
                log_w = log_w_next
                f = f_next
                break
            elif log_w_next < log_w:
                th_min = log_w_next
            else:
                th_max = log_w_next

        
        g = f + torch.randn_like(f)* math.sqrt(s)
        # sample v0
        w, v0 = torch.exp(log_w), torch.exp(log_v0)
        if cov == 'mat':
            Sig_th = matern52_ard_cov_torch(train_xs, train_xs, v0, w)
        else:
            Sig_th = SE_cov_torch(train_xs, train_xs, v0, w)
        
        # R_th = torch.linalg.inv( torch.linalg.inv(S_th) +  torch.linalg.inv(Sig_th))
        R_th = matrix_inverse(Sig_th, S_th)

        L_R = torch.linalg.cholesky(R_th)
        
        I = torch.eye(n, device=S_th.device, dtype=S_th.dtype)

        Sig_S_inv = torch.linalg.solve(S_th + Sig_th, I)
        # Sig_S_inv = torch.linalg.inv( S_th + Sig_th )
        m_th_g = Sig_th @ Sig_S_inv @ g

        # eta = torch.linalg.inv( L_R ) @ (f - m_th_g)
        eta = torch.linalg.solve(L_R, (f - m_th_g))

        v = torch.rand(size =(), device=device, dtype=dtype)*scale
        th_min = log_v0 -v
        th_max = th_min + scale

        u = torch.rand(size=(), device=device, dtype=dtype)

        logt = torch.log(u) - (1/2/k)*torch.dot(train_ys - f, train_ys - f) \
            - 0.5*torch.logdet(Sig_th + S_th) - 0.5 * g @ Sig_S_inv @ g \
            - 0.5* log_v0**2/ logv_prior_std**2# - math.log(2*math.pi) - 0.5*n*math.log(k) 
        
        th_min, th_max = step_out(th_min, th_max, logt, log_target, scale, g_cand=g, f_cand=f, fixed_para=log_w, update_para='v0', max_steps=max_steps_out)
        # print(th_min, th_max)
        
        while True:
            log_v0_next = torch.rand(size = (), device=device, dtype=dtype)*(th_max - th_min) + th_min
            v0_next = torch.exp(log_v0_next)
            
            if cov == 'mat':
                Sig_th_next = matern52_ard_cov_torch(train_xs, train_xs, v0_next, w)
            else:
                Sig_th_next = SE_cov_torch(train_xs, train_xs, v0_next, w)

            # R_th_next = torch.linalg.inv( torch.linalg.inv(S_th) +  torch.linalg.inv(Sig_th_next))
            R_th_next = matrix_inverse(Sig_th_next, S_th)
            try:
                L_R_next = torch.linalg.cholesky(R_th_next)
            except:
                import pdb;pdb.set_trace()

            Sig_S_inv_next = torch.linalg.solve( S_th + Sig_th_next, I )
            # Sig_S_inv_next = torch.linalg.inv( S_th + Sig_th_next )
            m_th_g_next = Sig_th_next @ Sig_S_inv_next @ g
            f_next = L_R_next @ eta + m_th_g_next

            sign2, logabsdet2 = torch.linalg.slogdet(Sig_th_next + S_th)
            logt_next = - (1/2/k)*torch.dot(train_ys - f_next, train_ys - f_next) \
                - 0.5*logabsdet2 - 0.5 * g @ Sig_S_inv_next @ g \
                - 0.5* log_v0_next**2/ logv_prior_std**2

            if logt_next > logt:
                log_v0 = log_v0_next
                f = f_next
                break
            elif log_v0_next < log_v0:
                th_min = log_v0_next
            else:
                th_max = log_v0_next

        log_wis[i] = log_w.item()
        log_v0is[i] = log_v0.item()

    return log_wis, log_v0is




def ghEI(x, f_best, train_xs, train_ys, w =1, v_0 = 1, k =0.05, cov = 'mat', a_b = 0.0):
    mu, beta, K_var, l, omega = mu_beta_sigma_fn(train_xs, train_ys, x, w=w, v_0=v_0, k=k, cov=cov, a_b=a_b)

    # print(mu, beta, K_var, l, omega )
    if torch.isnan(omega):
        import pdb;pdb.set_trace()

    # import pdb;pdb.set_trace()
    w_nodes, w_weights = gig_quadrature_gauss_laguerre(J=20, lam=l, omega=omega)

    m = mu + w_nodes * beta
    sigma = torch.sqrt(w_nodes * torch.clamp(K_var, min=1e-12))

    # print("m min/max:", m.min().item(), m.max().item())
    # print("sigma min/max:", sigma.min().item(), sigma.max().item())
    # u = (f_best - m)/sigma
    # print("u min/max:", u.min().item(), u.max().item())

    EI_nodes = gaussianEI(f_best=f_best, m=m, sigma=sigma)
    return torch.sum(w_weights * EI_nodes)

def gpEI(x, f_best, train_xs, train_ys, w=1, v0=1, k=0.05, eps_var: float = 1e-12, cov = 'mat'):
    m, v = g_regression(train_xs, train_ys, x, w=w, v_0=v0,k=k, cov=cov)  # <-- adapt args if your function differs

    # Make sure shapes are scalars
    m = m.squeeze()
    v = v.squeeze()

    sigma = torch.sqrt(torch.clamp(v, min=eps_var))
    ei = gaussianEI(f_best=f_best, m=m, sigma=sigma)
    return ei


def init_z_from_loose_grid(
        L: float,
        U: float,
        n_bins: int = 20,
        jitter_frac: float = 0.3,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ):
    """
    Sample an initial z such that x = L + (U-L)*sigmoid(z) is near a random loose-grid center,
    with optional Gaussian jitter inside the bin.

    Returns:
      z0: (1,) tensor, requires_grad=False (you can call .requires_grad_() outside)
    """
    L_t = torch.tensor(L, device=device, dtype=dtype)
    U_t = torch.tensor(U, device=device, dtype=dtype)
    width = (U_t - L_t) / n_bins

    # Choose a random bin and its center in x-space
    i = torch.randint(0, n_bins, (1,), device=device).item()
    x_center = L_t + (i + 0.5) * width

    # Jitter around the center (clipped to [L,U])
    x0 = x_center + jitter_frac * width * torch.randn(1, device=device, dtype=dtype)
    x0 = torch.clamp(x0, L_t, U_t)

    # Convert x0 in [L,U] -> p in (0,1) -> z = logit(p)
    eps = torch.tensor(1e-6, device=device, dtype=dtype)
    p = (x0 - L_t) / (U_t - L_t)
    p = torch.clamp(p, eps, 1 - eps)  # avoid inf logit
    z0 = torch.log(p) - torch.log(1 - p)

    return z0

def bayesian_optimization(runs=50, iterations=200):  
    m0 = 7.5

    prev_x = torch.tensor([m0])
    x = (m0 + 0.1 * torch.randn(1)).requires_grad_()
    ys = tgt_fn(prev_x)
    f_best = torch.min(ys)
    xs = torch.tensor([m0])

    opt_ys, opt_xs = torch.zeros(runs + 1), torch.zeros(runs + 1)
    opt_ys[0] = f_best
    opt_xs[0] = m0

    L, U = -5.0, 10.0

    for i in tqdm(range(runs)):
        optimizer = torch.optim.Adam([x], lr=0.05)

        for _ in range(iterations):
            optimizer.zero_grad()
            ei = ghEI(x, f_best, xs, ys, mu_beta_sigma_fn)
            loss = -ei
            loss.backward()
            # print("x:", x.item(), "grad:", x.grad, "is_leaf:", x.is_leaf)
            optimizer.step()

            with torch.no_grad():
                x.clamp_(L, U)

        new_y = tgt_fn(x.detach())

        xs = torch.cat((xs, x.detach()))
        ys = torch.cat((ys, new_y))

        prev_best = f_best
        f_best = torch.minimum(new_y.squeeze(), f_best)

        if new_y.squeeze() < prev_best:
            prev_x = x.detach().clone()
        else:
            with torch.no_grad():
                x.copy_(prev_x)
                x.add_(1e-3 * torch.randn_like(x))  # tiny nudge to avoid zero-derivative at data point
                x.clamp_(L, U)

        opt_ys[i + 1] = f_best
        opt_xs[i + 1] = x.detach().item()

    return opt_ys, opt_xs

def bayesian_optimization_multistart(runs=50, m0 = 6.0, iterations=200, n_starts = 20, model='g',ei_samples = 200, k =0.05,
                                     cov = 'mat', num_leapfrog = 500, step_size = 0.05, l = 1.0, omega = 1.0,
                                     plot = False, mc_ei = True, init_v0 = 10, init_w = 1, burnin = 50,
                                     prior_logv_std=2, prior_logw_std = 2, sampling = 'ss'): 
    
    """
    sampling = ['ss', 'hmc']
    """

    # prev_x = torch.tensor([m0])
    # x = (m0 + 0.1 * torch.randn(1))
    # xs = torch.tensor([6.0000, 9.8492, 5.9548, 6.1307, 6.4322, 6.2060])
    xs = torch.tensor([m0])
    ys = tgt_fn(xs)
    f_best = torch.min(ys)
    print(f_best)
    # xs = torch.tensor([x])

    opt_ys, opt_xs = torch.zeros(runs + 1), torch.zeros(runs + 1)
    opt_ys[0] = f_best
    opt_xs[0] = m0

    L, U = 5.0, 10.0

    best_x, best_ei = torch.tensor([m0]), None

    for i in tqdm(range(runs)):

        print(f'xs:{xs}, ys:{ys}')
        # TODO: sample theta from posterior using MCMC
        
        if model == 'gh':
            if len(xs) >= 5 and len(xs)<=6:
                step_size *= 0.5

            samples, acc_rate = hmc_sample_v0_w_gh(xs, ys, init_v0=init_v0, init_w=init_w, burn_in=burnin, num_leapfrog=num_leapfrog,step_size=step_size, num_samples=ei_samples,
                                        prior_std_logv0=prior_logv_std, prior_std_logw=prior_logw_std, cov = cov,k=k)
            log_v0s, log_ws, betas = samples[:, 0], samples[:, 1], samples[:, 2]
        else:
            if sampling == 'hmc':
                if len(xs) >= 5 and len(xs)<=7:
                    step_size *= 0.5

                    # num_leapfrog = int(1.5*num_leapfrog)

                samples, acc_rate = hmc_sample_v0_w(xs, ys, init_v0=init_v0, init_w=init_w, burn_in=burnin, num_leapfrog=num_leapfrog,step_size=step_size, num_samples=ei_samples,
                                            prior_std_logv0=prior_logv_std, prior_std_logw=prior_logw_std, cov = cov, k =k )
                log_v0s, log_ws = samples[:,0], samples[:,1]
            else: # 'ss'
                while True:
                    try:
                        log_ws, log_v0s = g_slice_sampling(xs, ys, iterations=ei_samples+burnin, scale=10,k=k,s=0.5, max_steps_out=20, cov=cov)
                        break
                    except:
                        continue

                # import pdb;pdb.set_trace()
                log_ws, log_v0s = torch.tensor(log_ws[burnin:], requires_grad=False), torch.tensor(log_v0s[burnin:], requires_grad=False)
                acc_rate = 0.0
        # print(f'acc: {acc_rate}')


        if plot:
            plt.figure()
            # import pdb;pdb.set_trace()
            plt.plot(log_v0s)
            plt.plot(log_ws)
            if model == 'gh':
                plt.plot(betas)
                plt.legend([r'v_0', 'w', r'$beta$'])
            else:
                plt.legend([r'v_0', 'w'])
            # plt.ylim([-5,5])
            plt.savefig(f'figure/bayesian/hmc_samples/{model}_{sampling}/samples_{cov}_{i}.png')

        # try grid search
        ngrid = 200
        xis = torch.linspace(L,U, steps=ngrid )
        eis, ms, vs = torch.zeros([ngrid]), torch.zeros([ngrid]), torch.zeros([ngrid])

        for j in range(ngrid):
            
            if model == 'gh':
                if mc_ei:
                    ei_est = 0.0
                    mis, vis = 0.0, 0.0
                    valid_samples = ei_samples
                    for _, (log_v0, log_w, beta) in enumerate(zip(log_v0s, log_ws, betas)):

                        delta_ei =  ghEI(xis[j], f_best, xs, ys, w=torch.exp(log_w), v_0=torch.exp(log_v0), k=k, cov=cov,a_b=beta).item()
                        if math.isnan(delta_ei):
                            valid_samples-=1
                            continue
                        
                        ei_est += delta_ei
                        
                        mi,betai, vi,li,omegai = mu_beta_sigma_fn(train_xs=xs, train_ys=ys, x=xis[j], cov=cov, l=l, omega=omega, k=k,
                                                            w=torch.exp(log_w), v_0=torch.exp(log_v0), a_b=beta)
                        EW = kve(li + 1, omegai)/ kve(li, omegai)
                        VarW = kve(li+2, omegai)/ kve(li, omegai) -  (kve(li+1, omegai)/ kve(li, omegai))**2

                        mis += mi + EW * betai
                        vis += EW * vi + VarW * betai**2
                        if EW * vi + VarW * betai**2 < 0:
                            import pdb;pdb.set_trace()

                    eis[j] = ei_est/valid_samples

                    ms[j] = mis / valid_samples
                    vs[j] =  vis / valid_samples
                else:
                    # log_v0, log_w, beta = torch.mean(log_v0s), torch.mean(log_ws), torch.mean(betas)
                    # print(log_v0, log_w, beta)
                    # eis[j] = ghEI(xis[j], f_best, xs, ys, w=torch.exp(log_w), v_0=torch.exp(log_v0), k=k, cov=cov,a_b=beta).item()
                    # mi,betai, vi,li,omegai = mu_beta_sigma_fn(xs, ys, xis[j], cov=cov, l=l, omega=omega, k=k,
                    #                                     w=torch.exp(log_w), v_0=torch.exp(log_v0), a_b=beta)

                    eis[j] = ghEI(xis[j], f_best, xs, ys, w=init_w, v_0=init_v0, k=k, cov=cov,a_b=1.0).item()

                    mi,betai, vi,li,omegai = mu_beta_sigma_fn(xs, ys, xis[j], cov=cov, l=l, omega=omega, k=k,
                                                        w=init_w, v_0=init_v0, a_b=1)
                    EW = kve(li + 1, omegai)/ kve(li, omegai)
                    VarW = kve(li+2, omegai)/ kve(li, omegai) -  (kve(li+1, omegai)/ kve(li, omegai))**2

                    ms[j] = mi + EW * betai
                    vs[j] = EW * vi + VarW * betai**2


            else:
                ei_est = 0.0
                mis, vis = 0.0, 0.0
                for log_v0, log_w in zip(log_v0s, log_ws):
                    ei_est += gpEI(xis[j], f_best, xs, ys, w=torch.exp(log_w), v0=torch.exp(log_v0), k = k, cov=cov).item()
                    mi, vi = g_regression(xs, ys, xis[j], cov=cov, w=torch.exp(log_w), v_0=torch.exp(log_v0), k = k)
                    mis += mi
                    vis += vi
                eis[j] = ei_est/ei_samples
                ms[j], vs[j] = mis/ei_samples, vis/ei_samples
            
        best_x = torch.tensor([xis[torch.argmax(eis)]])

        # import pdb;pdb.set_trace()


        if plot:
            fig, ax1 = plt.subplots()
            ax1.plot(xis, ms, color = 'tab:red')
            ax1.scatter(xs, ys, color = 'pink')
            ax1.fill_between(xis, ms + 3*torch.sqrt(vs), ms - 3*torch.sqrt(vs), alpha = 0.3, color = 'gray')
            ax1.set_ylabel('GP mean')
            ax2 = ax1.twinx()
            ax2.plot(xis, eis, color = 'tab:blue')
            ax2.set_ylabel('EI')

            plt.savefig(f'figure/bayesian/{model}_{sampling}/eis{i}_{cov}.png')
            plt.close()
            print('saved eis')


        # for _ in range(n_starts):
        #     z = init_z_from_loose_grid(
        #         L, U, n_bins=20,
        #         device=xs.device, 
        #         dtype=xs.dtype
        #     ).requires_grad_()

        #     # z = (torch.randn(1)).requires_grad_()
        #     optimizer = torch.optim.Adam([z], lr=0.05)
        #     print(f'x0:{L + (U-L)*torch.sigmoid(z)}')

        #     for ite in range(iterations):
        #         optimizer.zero_grad()
        #         x = L + (U-L)*torch.sigmoid(z)
        #         # ei = ghEI(x, f_best, xs, ys, mu_beta_sigma_fn)
        #         ei = gpEI(x, f_best, xs, ys, g_regression)
        #         loss = -ei
        #         loss.backward()
        #         if ite == 100:
        #             print("z:", z.item(), "grad:", z.grad, "is_leaf:", z.is_leaf)
        #         optimizer.step()

        #     with torch.no_grad():
        #         x_final = L + (U-L) * torch.sigmoid(z)
        #         ei_final = ghEI(x_final, f_best, xs, ys, mu_beta_sigma_fn)
        #         if best_ei is None or ei_final > best_ei:
        #             best_ei = ei_final
        #             best_x = x_final.detach().clone()

        new_y = tgt_fn(best_x.detach())
        

        # import pdb;pdb.set_trace()
        if best_x not in xs:
            xs = torch.cat((xs, best_x.detach()))
            # ys = torch.cat((ys, ( new_y- y_mean)/y_std))
            ys = torch.cat((ys, ( new_y)))

        # prev_best = f_best
        f_best = torch.minimum(new_y.squeeze(), f_best)

        # if new_y.squeeze() < prev_best:
        #     prev_x = best_x.detach().clone()
        # else:
        #     with torch.no_grad():
        #         x.copy_(prev_x)
        #         x.add_(1e-3 * torch.randn_like(x))  # tiny nudge to avoid zero-derivative at data point

        opt_ys[i + 1] = f_best
        opt_xs[i + 1] = best_x.detach().item()

        print(f'best x:{best_x}, best f:{f_best}, new y:{new_y}')

    return opt_ys, opt_xs

def SE_cov(xs, ys, v_0, w):
    N = len(xs)
    M = len(ys)
    K = np.zeros((N,M))
    
    for i in range(N):
        for j in range(M):
            try:
                K[i,j] = v_0 * np.exp(-w*(xs[i]-ys[j])**2/2)
            except:
                import pdb;pdb.set_trace()

    if N==M:
        return K  + 1e-7*np.identity(N)
    else:
        return K

from tools.tools import * 

def Multivariate_GH_1d(xs, l = 1, omega = 1, w = 1, v_0 = 1, a_b = 1, b_b = 0, k = 1,mu=-1):
    """
    xs: np array, input location
    l: index parameter for GIG
    omega: a, b in GIG
    SE cov: v_0 exp(-w*(x-y)^2/2)
    beta(x) = a_b + b_b *x
    k: noise variance
    mu: mean of gaussian kernel
    """
    
    W = stats.geninvgauss.rvs(p=l, b=omega, size=1)[0]

    N = len(xs)
    # exponential cov function
    K = SE_cov(xs,xs, v_0, w)

    #skew
    beta = np.ones(N)*a_b + xs*b_b

    # assume zero mean
    fs = np.random.multivariate_normal(beta*W + mu, (K)*W)
    ys = fs + np.random.multivariate_normal(beta*0, (k*np.identity(N))*W)

    return fs, ys

def likelihood_grid(lxs = 100, cov = 'se', hmc_sample = False, k=0.1,logv_prior_std = 2.0, logw_prior_std = 2.0):
    

    xs = torch.tensor([7.0, 10.0])
    # ys = tgt_fn(xs)
    torch.manual_seed(2)
    l, omega = 1.0,1.0
    a_b = 1.0
    
    
    W = stats.geninvgauss.rvs(p=l, b=omega, size=1)[0]
    xs = torch.linspace(0, 100, lxs)
    # _, ys = Multivariate_GH_1d(xs, omega=1,a_b=1.0, k=k, mu=0.0)

    # file = np.load(f'xy_k01_N{lxs}.npz')
    # xs = np.array(file['x'])
    # ys = np.array(file['y'])
    # xs, ys = torch.tensor(xs), torch.tensor(ys)

    if cov == 'mat':
        K = matern52_ard_cov_torch(xs,xs, theta0 = 1, w = 1) + k* torch.eye(len(xs))
    else:
        K = SE_cov_torch(xs,xs, v0 = 1, w = 1) + k* torch.eye(len(xs))
    L = torch.linalg.cholesky(K)

    #skew
    beta = torch.ones(lxs)*a_b
    # assume zero mean
    z = torch.randn(K.shape[0], device = K.device, dtype=K.dtype)
    ys = L @ z * math.sqrt(W) + beta*W

    xs = torch.linspace(5.0, 10.0, steps=lxs)
    # xs = torch.tensor([7.0, 10.0])
    ys = tgt_fn(xs)

    
    # z = torch.randn(K.shape[0], device = K.device, dtype=K.dtype)
    # ys = L @ z

    if hmc_sample:
        samples, acc_rate = hmc_sample_v0_w_gh(xs, ys, init_v0=3, init_w=3, burn_in=0,num_leapfrog=200,step_size=0.05, num_samples=200,
                                            prior_std_logv0=20, prior_std_logw=20,prior_std_beta=20, cov=cov)
        print(acc_rate)

        plt.figure()
        plt.plot(samples)
        plt.legend(['v0', 'w', 'beta'])
        # plt.ylim([-5,5])
        plt.savefig(f'figure/bayesian/hmc_samples/samples_gh_{cov}.png')

    # return

    n = 20
    log_v0s = torch.linspace(5,10,n)
    log_ws = torch.linspace(0,4,n)
    # v0s = torch.linspace(0.5,2,n)
    # ws = torch.linspace(0.5,2,n)

    betas = torch.linspace(-1,3,n)
    likelihoods = torch.zeros((n,n))

    X, Y =torch.meshgrid(log_v0s, log_ws)

    theta = [a_b, 1,1,k,0,1.0,1.0]

    for i, (row_x , row_y) in tqdm(enumerate(zip(X, Y))):
        for j, (x , y) in enumerate(zip(row_x, row_y)):
            
            likelihoods[i,j] = g_likelihood(xs, ys, v0 = torch.exp(x), w=torch.exp(y), cov = cov, k=k) \
                - 0.5* x**2/ logv_prior_std**2 - 0.5* y**2/ logw_prior_std**2
            # likelihoods[i,j] = gh_logprob(xs, ys, l=1.0, omega=1.0, k=k,
            #                               v_0= torch.exp(x), w= torch.exp(y), 
            #                               a_b=1.0, cov=cov) # v_0 = 10, w = 3, 

            # theta[1] = np.exp(x)
            # theta[2] = np.exp(y)
            # # import pdb;pdb.set_trace()
            # likelihoods[i,j] = logprob(xs, ys, theta, l=theta[-2], p=len(xs), omega=theta[-1])

    idx = torch.argmax(likelihoods)
    idx = torch.unravel_index(idx, likelihoods.shape)
    print(torch.exp(X[idx]), torch.exp(Y[idx]))
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, np.exp(likelihoods-likelihoods[idx]), cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
    ax.set_xlabel('log v0')
    ax.set_ylabel('log w')
    fig.savefig(f'figure/bayesian/ssamples/tgt_gp/likelihood_g_v0_w_{cov}_{lxs}data_k{k}.png')
    plt.show()

def actual_min(n =200):
    xs = torch.linspace(5,10, n)
    ys = torch.zeros_like(xs)
    for i, x in enumerate(xs):
        ys[i] = tgt_fn(x)

    minidx = torch.argmin(ys)
    print(min(ys), xs[minidx])

def bayOpwrap(model = 'gh', fn = 'tgt', cov = 'se', m0=7.0, runs = 10, sampling='hmc', k =0.1, plot = True):
    opt_ys, opt_xs = bayesian_optimization_multistart(m0=m0, runs=runs, n_starts=1, model = model, k =k, l=1.0, omega=1.0, 
                                                      plot=plot, cov=cov, mc_ei=True, ei_samples=200,
                                                      init_w=1, init_v0=10, sampling=sampling)

    plt.figure()
    # plt.plot(opt_ys.detach().cpu().numpy())
    plt.plot(opt_xs.detach().cpu().numpy(), tgt_fn(opt_xs), label = 'BayOp', alpha = 0.5, linestyle = '--')
    plt.scatter(opt_xs.detach().cpu().numpy(), tgt_fn(opt_xs), label = 'BayOp')
    plt.plot(xs_true, ys_true, label = 'Tgt')
    plt.savefig(f'figure/bayesian/{fn}/{model}/{sampling}/m0{m0}_fns_{cov}.png')

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(opt_ys.detach().cpu().numpy())
    plt.ylabel('Opt ys')

    plt.subplot(2,1,2)
    plt.plot(opt_xs.detach().cpu().numpy())
    plt.ylabel('Explore xs')
    plt.savefig(f'figure/bayesian/{fn}/{model}/{sampling}/m0{m0}_xys_{cov}.png')

if __name__ == '__main__':
    xs_true = torch.linspace(5.0, 10.0, steps=500)
    ys_true = tgt_fn(xs_true)
    # plt.plot(xs, ys)
    # plt.savefig('figure/bayesian/tgt.png')
    
    
    # bayOpwrap(model='g', runs=20, cov='mat', sampling='hmc', k=0.1, plot=False, m0=6)
    bayOpwrap(model='gh', runs=20, cov='mat', sampling='hmc', k=0.1, plot=False, m0=float(6))

    # actual_min()
    
    # likelihood_grid(lxs=10, cov='mat',k=10)

    # TODO：implement slice-sampling in bayop, for GH process, add prior; debug slice-sampling, could use Elliptical SS if the performance is not ideal
    # TODO: find a dataset for regression
    cov = 'mat'
    k = 0.1
    # xs = torch.linspace(5.0, 10.0, steps=10)
    # if cov == 'mat':
    #     K = matern52_ard_cov_torch(xs,xs, theta0 = 1, w = 1) + k* torch.eye(len(xs))
    # else:
    #     K = SE_cov_torch(xs,xs, v0 = 1, w = 1) + k* torch.eye(len(xs))
    # # L = torch.linalg.cholesky(K)
    
    xs = torch.tensor([ 7.0000, 10.0000,  9.5226,  5.0000,  5.9548,  5.8040,  6.1307]) # ,  9.9246,  5.0000,  5.4523,  8.3417,  8.3668
    
    # z = torch.randn(K.shape[0], device = K.device, dtype=K.dtype)
    ys = tgt_fn(xs) # + z*math.sqrt(k)

    # # ys = L @ z

    # log_ws, log_v0s = g_slice_sampling(xs, ys, iterations=500, scale=0.5, k=0.5,s=5, cov=cov, max_steps_out=10)  # TODO: check this test case, in bayOp function it worked fine, why doesn't work here?
    # print("max Δ log_w:", np.max(np.abs(np.diff(log_ws))))
    # print("max Δ log_v0:", np.max(np.abs(np.diff(log_v0s))))
    # print("unique log_w (6dp):", np.unique(np.round(log_ws, 6)).size)
    # print("unique log_v0 (6dp):", np.unique(np.round(log_v0s, 6)).size)
    # plt.figure()
    # plt.plot(log_ws, label = 'log w')
    # plt.plot(log_v0s, label = 'log v0')
    # plt.legend()
    # plt.savefig('figure/bayesian/ssamples/tgt_gp/samples.png')


    # xis = torch.linspace(5,10, steps=200 )
    # for i, xi in enumerate(xis):  # tensor(4373.6427, dtype=torch.float64) tensor(0.2771, dtype=torch.float64) 0.1
    # mi,betai, vi,li,omegai = mu_beta_sigma_fn(xs, ys, x=6.0050, cov=cov, l=1, omega=1, k=k,
    #                                                         w=torch.tensor(0.2771), v_0=torch.tensor(4373.6427), a_b=0.1)
