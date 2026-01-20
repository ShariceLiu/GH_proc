from scipy import special, stats
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import jax.numpy as jnp
import jax

from tools.tools import *
from functools import partial
from jax.scipy.linalg import solve_triangular
from matplotlib import cm

import pandas as pd

def read_wine(N=400, random =False, seedn=1):
    winepath = "C:/Users/95414/Desktop/CUED/phd/year1/mycode/data/winequality/winequality-white.csv"
    winedata = pd.read_csv(winepath, sep=';')

    # split features/target
    y = winedata["quality"]
    X = winedata.drop(columns=["quality"])

    # compute sample mean and (population) SD for each column
    mu = X.mean(axis=0)
    sd = X.std(axis=0, ddof=0)           # ddof=0 matches scikit-learn's StandardScaler

    # avoid division by zero if a column is constant
    sd = sd.replace(0, np.nan)

    # standardized features
    X_std = (X - mu) / sd

    # (optional) keep the scaler to invert later or apply to new data
    scaler_params = {"mean": mu, "sd": sd}

    # If you want a single DataFrame with standardized X and original y:
    xs = X_std.to_numpy()
    ys = y.to_numpy() # .astype(dtype=np.float32)
    # import pdb;pdb.set_trace()
    ys -= int(ys.mean())
    np.random.seed(seedn)

    if random:
        random_idx = np.random.choice(N, size=N, replace=False)
        xs = xs[random_idx]
        ys = ys[random_idx]
    else:
        xs = xs[:N]
        ys = ys[:N]

    return xs.T, ys



def Multivariate_GH(xs, l = -1, omega = 1, ws = np.ones(11),  v_0 = 1, a_b = 1.0,  k = 1):
    """
    xs: np array, input location, 11 D, should be like 11 cols, N rows
    l: index parameter for GIG
    omega: a, b in GIG
    SE cov: v_0 exp( - sum_i w_i*(x_i-y_i)^2/2  )
    beta(x) = a_b
    k: noise variance
    """
    W = stats.geninvgauss.rvs(p=l, b=omega, size=1)[0]

    N = xs.shape[1]
    # exponential cov function
    K = SE_cov_2d(xs,xs, v_0,ws)

    #skew
    beta = np.ones(N)*a_b

    # assume zero mean
    fs = np.random.multivariate_normal(beta*W, (K)*W)
    ys = fs + np.random.multivariate_normal(beta*0, (k*np.identity(N))*W)

    return fs, ys

def regression(train_xs, train_ys, test_xs, l = -1, omega = 1, ws = np.ones(11),  \
               v_0 = 1, a_b = 1, b_b = np.zeros(11), k =1,\
                mu = 0.0):
    K11 = SE_cov_multid(train_xs, train_xs, v_0, ws) + k*np.identity(len(train_ys))
    K12 = SE_cov_multid(train_xs, test_xs, v_0, ws)
    K22 = SE_cov_multid(test_xs, test_xs, v_0, ws)

    beta1 = np.ones(len(train_ys))*a_b + b_b @ train_xs
    beta2 = np.ones(test_xs.shape[1])*a_b + b_b @ test_xs

    l21 = l - len(train_ys)/2
    invK11 = np.linalg.inv(K11)
    gamma21 = omega + (train_ys-mu) @ (invK11 @ (train_ys-mu))
    delta21 = omega + beta1 @ (invK11 @ beta1)
    mu21 = mu+ K12.T @ invK11 @ (train_ys-mu)
    K221 = K22 - K12.T @ invK11 @ K12
    beta21 = beta2 - K12.T @ invK11 @ beta1

    omega21 = gamma21 * delta21
    k21 = np.sqrt(delta21/ gamma21)

    if special.kn(l21, omega21)==0:
        # import pdb;pdb.set_trace()
        n = np.abs(l21)
        EW = 1 - (n-0.5)/omega21
        VarW = (1/omega21 + (n**2/4 + 2*n + 15/16) / omega21**2)
    else:
        EW = special.kn(l21+1, omega21)/special.kn(l21, omega21)
    
        VarW = special.kn(l21+2, omega21)/special.kn(l21, omega21)\
            -(special.kn(l21+1,  omega21)/special.kn(l21, omega21))**2
    mean21 = mu21 + EW*beta21
    var21 =  EW* K221 + VarW * np.outer(beta21, beta21)

    return mean21, var21

def st_regression(train_xs, train_ys, test_xs, v=3, ws = np.ones(11), v_0 = 1, k =1,phi=0.0):
    K11 = SE_cov_multid(train_xs, train_xs, v_0, ws) + k*np.identity(len(train_ys))
    K12 = SE_cov_multid(train_xs, test_xs, v_0, ws)
    K22 = SE_cov_multid(test_xs, test_xs, v_0, ws)

    invK11 = np.linalg.inv(K11)
    n1=len(train_ys)
    # v21 = v + n1
    phi21 = K12.T @ invK11 @ (train_ys - phi) + phi
    K221 = K22 - K12.T @ invK11 @ K12

    beta1 = (train_ys-phi) @ (invK11@ (train_ys-phi))
    var21 = (v+ beta1 -2)/(v+ n1-2)*K221

    return phi21, var21

def g_regression(train_xs, train_ys, test_xs, ws = np.ones(11), v_0 = 1, k =1,phi=0.0):
    K11 = SE_cov_multid(train_xs, train_xs, v_0, ws) + k*np.identity(len(train_ys))
    K12 = SE_cov_multid(train_xs, test_xs, v_0, ws)
    K22 = SE_cov_multid(test_xs, test_xs, v_0, ws)

    invK11 = np.linalg.inv(K11)
    n1=len(train_ys)
    # v21 = v + n1
    mu21 = phi+ K12.T @ invK11 @ (train_ys-phi)
    K221 = K22 - K12.T @ invK11 @ K12

    return mu21, K221

def HMC_jx(x, y, l, p, omega, k, num_leapfrog_steps, step_size, num_steps, seed, theta0):

    def leapfrog_step(grad, step_size, i, leapfrog_state):
        theta, m, tlp_grad = leapfrog_state

        step_size = step_size * 0.95** i
        
        m += 0.5 * step_size * tlp_grad
        theta += step_size *m
        tlp_grad = grad(x, y, theta, l, p, omega, k)
        m += 0.5* step_size * tlp_grad
        # jax.debug.print('tlp_grad: {} theta: {}', tlp_grad, theta)
        return theta, m, tlp_grad
    
    def hmc_step(grad, num_lf_steps, step_size, theta, seed):
        m_seed, mh_seed = jax.random.split(seed)
        tlp_grad = grad(x,y, theta, l, p, omega, k) 
        tlp = multid_logprob_jx(x,y, theta, l, p, omega, k)
        m = jax.random.normal(m_seed, theta.shape)
        energy = 0.5* jnp.square(m).sum() - tlp

        jax.debug.print('log prob: {} theta: {}, energy:{}', tlp, theta[jnp.array([0,1,2],int)], energy)

        new_theta, new_m = theta, m
        new_theta, new_m, _ = jax.lax.fori_loop(
            0,
            num_lf_steps,
            partial(leapfrog_step, grad, step_size),
            (theta, m, tlp_grad)
        )

        
        new_energy = 0.5* jnp.square(new_m).sum() - multid_logprob_jx(x,y, new_theta, l, p, omega, k)
        new_tlp = multid_logprob_jx(x,y, new_theta, l, p, omega, k)

        jax.debug.print('new log prob: {} new theta: {}, new energy:{}', new_tlp, new_theta[jnp.array([0,1,2],int)], new_energy)
        log_accept_ratio = energy - new_energy
        is_accepted = jnp.log(jax.random.uniform(mh_seed,[])) < log_accept_ratio
        # select proposed state if accepted
        theta = jnp.where(is_accepted, new_theta, theta)
        hmc_op = {'theta': theta,
                  'is_accepted': is_accepted,
                  'log_acc_ratio':log_accept_ratio}
        return theta, hmc_op
    
    # initialize
    theta = theta0
    # create a seed for each step
    # seed = jax.random.PRNGKey(seed)
    seeds = jax.random.split(seed, num_steps)
    # repeatedly run hmc and accumulate the outputs
    _, hmc_output = jax.lax.scan(partial(hmc_step, multid_grad_jx,num_leapfrog_steps, step_size), theta, seeds)
    return hmc_output

def HMC_jx_adapt_ep(x, y, l, p, omega, k,
           num_leapfrog_steps, step_size, num_steps, seed, theta0,
           chol_M=None, adapt_eps=False, target_accept=0.8, Lmin=30):
    """
    HMC with optional DENSE mass matrix M (given via its Cholesky factor chol_M, lower-triangular).
    If chol_M is None -> identity metric.
    """
    D = theta0.size
    if chol_M is None:
        chol_M = jnp.eye(D)

    # helpers for metric
    def sample_momentum(key):
        z = jax.random.normal(key, (D,))
        return chol_M @ z                           # p ~ N(0, M)

    def Minv_dot(v):
        # x = M^{-1} v via triangular solves (no explicit inverse)
        y = solve_triangular(chol_M, v, lower=True)
        x = solve_triangular(chol_M.T, y, lower=False)
        return x

    def kinetic(p):
        return 0.5 * jnp.dot(p, Minv_dot(p))

    def leapfrog_step(grad, step_size, i, leapfrog_state):
        theta, m, tlp_grad = leapfrog_state
        eps_i = step_size                           # constant within trajectory
        m     = m + 0.5 * eps_i * tlp_grad
        theta = theta + eps_i * Minv_dot(m)         # q' = M^{-1} p
        tlp_grad = grad(x, y, theta, l, p, omega, k)
        # jax.debug.print('tlp_grad: {} theta: {}', tlp_grad, theta)
        m     = m + 0.5 * eps_i * tlp_grad
        return theta, m, tlp_grad

    def hmc_step(grad, step_size, num_lf_steps, theta, seed, adapt_eps):
        m_seed, mh_seed = jax.random.split(seed)

        tlp_grad = grad(x, y, theta, l, p, omega, k)
        tlp      = multid_logprob_jx(x, y, theta, l, p, omega, k)

        m = sample_momentum(m_seed)                 # p ~ N(0, M)

        energy = kinetic(m) - tlp
        # jax.debug.print('log prob: {} theta: {}', tlp, theta)

        new_theta, new_m, _ = jax.lax.fori_loop(
            0, num_lf_steps,
            partial(leapfrog_step, grad, step_size),
            (theta, m, tlp_grad)
        )

        
        new_tlp = multid_logprob_jx(x, y, new_theta, l, p, omega, k)
        new_energy = kinetic(new_m) - new_tlp

        # jax.debug.print('log prob: {} new theta: {}', new_tlp, new_theta[jnp.array([0,1,2],int)])


        log_accept_ratio = energy - new_energy
        accept_prob = jnp.minimum(1.0, jnp.exp(log_accept_ratio))
        is_accepted = jnp.log(jax.random.uniform(mh_seed, ())) < log_accept_ratio

        theta = jnp.where(is_accepted, new_theta, theta)

        step_size_new = step_size
        if adapt_eps:
            step_size_new = step_size * jnp.exp(0.05 * (accept_prob - target_accept))

        hmc_op = {
            "theta": theta,
            "is_accepted": is_accepted,
            "log_acc_ratio": log_accept_ratio,
            "accept_prob": accept_prob,
            "step_size": step_size_new,
        }
        return theta, hmc_op

    # run scan
    seeds = jax.random.split(seed, num_steps)
    Ls    = jax.random.randint(seeds[0], (num_steps,), Lmin, num_leapfrog_steps + 1)

    def scan_body(state, carry):
        theta, eps = state
        key, L = carry
        theta, out = hmc_step(multid_grad_jx, eps, L, theta, key, adapt_eps)
        return (theta, out["step_size"]), out

    (theta_final, eps_final), outs = jax.lax.scan(
        scan_body, (theta0, step_size), (seeds, Ls)
    )
    return outs

def HMC_fn_jx_adapt_ep(xs, ys, k=0.1, save=True, seedn=1400,
              eps0=1e-3, Lmax=100, Lmin = 40, num_steps=1100,
              burn_in=300, load=False, num_warmup=800, init_scale=0.1, w0=-4.0, adapt_ep=True, l =-1):

    xs, ys = read_wine(N=360, seedn = seedn, random=True)

    seed = jax.random.PRNGKey(seedn)

    jax.debug.print('shape: {}', xs.shape)

    if load:
        theta0 = jnp.array(jnp.load(f'data/theta_seed{seedn}_w1.npz')['theta'])
    else:
        key_init, seed_warmup = jax.random.split(seed)
        # dim = xs.shape[0] + 3

        mean0 = jnp.ones(3)*w0 # v0, w
        mean0 = mean0.at[0].set(jnp.array(0.5))
        mean0 = mean0.at[1].set(jnp.array(-0.7))

        # mean0 = jnp.array([4.0, 1.5, 4.0])

        jax.debug.print('mean0: {}', mean0)

        dim = 3 # beta, v0, w1
        theta0 = init_scale * jax.random.normal(key_init, (dim,)) + mean0 # random init

    # ---- Warm-up 1: identity metric, adapt ε
    warm1 = HMC_jx_adapt_ep(xs, ys, l=l, p=xs.shape[1], omega=1, k=k,
                num_leapfrog_steps=Lmax, step_size=eps0,
                num_steps=num_warmup, seed=seed, theta0=theta0,
                chol_M=None, adapt_eps=adapt_ep, target_accept=0.8, Lmin=Lmin)

    thetas_warm1 = warm1["theta"]              # (num_warmup, D)
    theta_last   = thetas_warm1[-1]
    eps_hat      = warm1["step_size"][-1]

    def mean_acc(outs):
        return float(jnp.asarray(outs["accept_prob"]).mean())

    # after warm1
    eps_hat = float(warm1["step_size"][-1])
    acc = mean_acc(warm1)
    jax.debug.print(str(acc))

    # # push acceptance into 0.75–0.85
    # while acc > 0.90:
    #     eps_hat *= 2.0
    #     warm1 = HMC_jx_adapt_ep(xs, ys, l=-1, p=xs.shape[1], omega=1, k=k,
    #         num_leapfrog_steps=Lmax, step_size=eps_hat, num_steps=400,
    #         seed=seed, theta0=theta0, chol_M=None, adapt_eps=True, target_accept=0.8)
    #     acc = mean_acc(warm1)

    # while acc < 0.60:
    #     eps_hat *= 0.5
    #     warm1 = HMC_jx_adapt_ep(xs, ys, l=-1, p=xs.shape[1], omega=1, k=k,
    #         num_leapfrog_steps=Lmax, step_size=eps_hat, num_steps=400,
    #         seed=seed, theta0=theta0, chol_M=None, adapt_eps=True, target_accept=0.8)
    #     acc = mean_acc(warm1)

    eps_hat = float(warm1["step_size"][-1])
    theta_last = warm1["theta"][-1]

    # ---- Learn DENSE mass matrix from warm-up draws
    # S = jnp.cov(thetas_warm1, rowvar=False)
    # # ridge jitter to ensure PD
    # jitter = 1e-4 * (jnp.trace(S) / S.shape[0])
    # S = S + jitter * jnp.eye(S.shape[0])
    # chol_M = jnp.linalg.cholesky(S)

    # jax.debug.print('S:{}, col_M:{}', S, chol_M)
    S = jnp.cov(thetas_warm1, rowvar=False)

    # shrink toward diagonal to reduce noise (optional but helpful)
    tau = 0.1
    S = (1 - tau) * jnp.diag(jnp.diag(S)) + tau * S

    # normalize so median variance is ~1
    med_var = jnp.median(jnp.diag(S))
    S_norm = S / jnp.where(med_var > 0, med_var, 1.0)

    # clip eigenvalues to avoid extremes
    evals, Q = jnp.linalg.eigh(S_norm)
    floor = 1e-3          # relative to normalized scale
    ceil  = 1e+3
    evals = jnp.clip(evals, floor, ceil)
    S_tilde = (Q * evals) @ Q.T

    # jitter and Cholesky
    jitter = 1e-6
    chol_M = jnp.linalg.cholesky(S_tilde + jitter * jnp.eye(S_tilde.shape[0]))

    # return

    # ---- Warm-up 2: freeze metric, adapt ε a bit more
    # warm2 = HMC_jx_adapt_ep(xs, ys, l=-1, p=xs.shape[1], omega=1, k=k,
    #                num_leapfrog_steps=Lmax, step_size=eps_hat,
    #                num_steps=num_warmup//2, seed=seed, theta0=theta_last,
    #                chol_M=chol_M, adapt_eps=True, target_accept=0.8)

    # theta_last = warm2["theta"][-1]
    # eps_hat    = warm2["step_size"][-1]

    jax.debug.print(str(eps_hat))

    # ---- Sampling: metric + step size frozen
    hmc_output = HMC_jx_adapt_ep(xs, ys, l=l, p=xs.shape[1], omega=1, k=k,
                   num_leapfrog_steps=Lmax, step_size=eps_hat,
                   num_steps=num_steps, seed=seed, theta0=theta_last,
                   chol_M=chol_M, adapt_eps=False)
    
    acc = mean_acc(hmc_output)
    jax.debug.print(str(acc))

    tlp = multid_logprob_jx(xs, ys, hmc_output['theta'][-1,:], l=l, p=xs.shape[1], omega=1, k=k)
    jax.debug.print('log likelihood:{}', tlp)
    
    plt.figure()
    # plt.subplot(2,2,1)
    # plt.plot(np.array(hmc_output['theta'])[burn_in:, 0])
    # plt.xlabel(r'$\beta$')
    plt.subplot(2,1,1)
    plt.plot((np.array(hmc_output['theta'])[burn_in:, 1]))
    plt.xlabel(r'$v_0$')
    plt.subplot(2,1,2)
    plt.plot((np.array(hmc_output['theta'])[burn_in:, 2]))
    plt.xlabel(r'$w$') # k
    # plt.subplot(3,2,4)
    # plt.plot((np.array(hmc_output['theta'])[burn_in:, 3]))
    # plt.xlabel(r'$w_1$')
    # plt.subplot(3,2,5)
    # plt.plot((np.array(hmc_output['theta'])[burn_in:, 4]))
    # plt.xlabel(r'$w_2$')
    # plt.subplot(3,2,6)
    # plt.plot((np.array(hmc_output['theta'])[burn_in:, 5]))
    # plt.xlabel(r'$w_3$')
    if save:
        plt.savefig(f'figure\\wine\\para_fitting\\w1\\para_seed{seedn}.png')

    


    plt.figure()
    plt.hist(np.exp(np.minimum(np.array(hmc_output['log_acc_ratio']), 0)))
    plt.xlabel('accept ratio')
    if save:
        plt.savefig(f'figure\\wine\\para_fitting\\w1\\acc_seed{seedn}.png')

    # autocorrelation
    s = pd.Series(np.array(hmc_output['theta'])[:, 1])
    lags = np.arange(int(num_steps/2))
    acf = [s.autocorr(lag=i) for i in lags]
    plt.figure()
    plt.plot(lags, acf)
    if save:
        plt.savefig(f'figure\\wine\\para_fitting\\w1\\acf_seed{seedn}.png')
    plt.show()

    if save:
        jnp.savez(f'data/theta_seed{seedn}_w1.npz', theta = np.array(hmc_output['theta'])[burn_in:, :].mean(axis=0))
    
    return np.array(hmc_output['theta'])[-500:, :].mean(axis=0)

def HMC_jx_adapt_ep_st(x, y,  p, v, k,
           num_leapfrog_steps, step_size, num_steps, seed, theta0,
           chol_M=None, adapt_eps=False, target_accept=0.8, Lmin=30):
    """
    HMC with optional DENSE mass matrix M (given via its Cholesky factor chol_M, lower-triangular).
    If chol_M is None -> identity metric.
    """
    D = theta0.size
    if chol_M is None:
        chol_M = jnp.eye(D)

    # helpers for metric
    def sample_momentum(key):
        z = jax.random.normal(key, (D,))
        return chol_M @ z                           # p ~ N(0, M)

    def Minv_dot(vi):
        # x = M^{-1} v via triangular solves (no explicit inverse)
        y = solve_triangular(chol_M, vi, lower=True)
        x = solve_triangular(chol_M.T, y, lower=False)
        return x

    def kinetic(p):
        return 0.5 * jnp.dot(p, Minv_dot(p))

    def leapfrog_step(grad, step_size, i, leapfrog_state):
        theta, m, tlp_grad = leapfrog_state
        eps_i = step_size                           # constant within trajectory
        m     = m + 0.5 * eps_i * tlp_grad
        theta = theta + eps_i * Minv_dot(m)         # q' = M^{-1} p
        tlp_grad = grad(x, y, theta, p, v, k)
        m     = m + 0.5 * eps_i * tlp_grad
        return theta, m, tlp_grad

    def hmc_step(grad, step_size, num_lf_steps, theta, seed, adapt_eps):
        m_seed, mh_seed = jax.random.split(seed)

        tlp_grad = grad(x, y, theta, p, v, k)
        tlp      = multid_logprob_jx_st(x, y, theta, p, v, k)

        m = sample_momentum(m_seed)                 # p ~ N(0, M)

        energy = kinetic(m) - tlp

        jax.debug.print('log prob: {} theta: {}', tlp, theta)

        new_theta, new_m, _ = jax.lax.fori_loop(
            0, num_lf_steps,
            partial(leapfrog_step, grad, step_size),
            (theta, m, tlp_grad)
        )

        new_tlp = multid_logprob_jx_st(x, y, new_theta, p, v, k)
        new_energy = kinetic(new_m) - new_tlp
        # checkify.check(len(new_tlp)>1, "lop accept scalar")

        log_accept_ratio = energy - new_energy

        accept_prob = jnp.minimum(1.0, jnp.exp(log_accept_ratio))
        is_accepted = jnp.log(jax.random.uniform(mh_seed, ())) < log_accept_ratio

        # jax.debug.print('is accepted:{}, new theta {}, theta: {}', is_accepted, new_theta, theta)
        theta = jnp.where(is_accepted, new_theta, theta)

        step_size_new = step_size
        if adapt_eps:
            step_size_new = step_size * jnp.exp(0.05 * (accept_prob - target_accept))

        hmc_op = {
            "theta": theta,
            "is_accepted": is_accepted,
            "log_acc_ratio": log_accept_ratio,
            "accept_prob": accept_prob,
            "step_size": step_size_new,
        }
        return theta, hmc_op

    # run scan
    seeds = jax.random.split(seed, num_steps)
    Ls    = jax.random.randint(seeds[0], (num_steps,), Lmin, num_leapfrog_steps + 1)

    def scan_body(state, carry):
        theta, eps = state
        key, L = carry
        theta, out = hmc_step(multid_grad_jx_st, eps, L, theta, key, adapt_eps)
        return (theta, out["step_size"]), out

    (theta_final, eps_final), outs = jax.lax.scan(
        scan_body, (theta0, step_size), (seeds, Ls)
    )
    return outs

def HMC_fn_jx_adapt_ep_st(xs, ys, k=0.1, save=True, seedn=1400,
              eps0=1e-3, Lmax=100, Lmin = 40, num_steps=1100,
              burn_in=300, load=False, num_warmup=800, v=3,
              init_scale = 0.1, w0=-5, adapt_ep = False):

    xs, ys = read_wine(N=360, seedn=seedn, random=True)

    seed = jax.random.PRNGKey(seedn)

    if load:
        theta0 = jnp.array(jnp.load(f'data/theta_st_{int(-w0)}_w1.npz')['theta'])

    else:

        key_init, seed_warmup = jax.random.split(seed)
        # mean0 = jnp.ones(xs.shape[0] + 1)*w0
        mean0 = jnp.ones(2)*w0 # v0, w
        mean0 = mean0.at[0].set(jnp.array(1.1))

        theta0 = init_scale * jax.random.normal(key_init, (2,)) + mean0
        

    # ---- Warm-up 1: identity metric, adapt ε
    warm1 = HMC_jx_adapt_ep_st(xs, ys, p=ys.shape[0], v=v, k=k,
                   num_leapfrog_steps=Lmax, step_size=eps0,
                   num_steps=num_warmup, seed=seed, theta0=theta0,
                   chol_M=None, adapt_eps=adapt_ep, target_accept=0.8, Lmin=Lmin)

    thetas_warm1 = warm1["theta"]              # (num_warmup, D)
    theta_last   = thetas_warm1[-1]
    eps_hat      = warm1["step_size"][-1]

    def mean_acc(outs):
        return float(jnp.asarray(outs["accept_prob"]).mean())

    # after warm1
    eps_hat = float(warm1["step_size"][-1])
    acc = mean_acc(warm1)
    jax.debug.print(str(acc))
    jax.debug.print(str(eps_hat))

    # # push acceptance into 0.75–0.85
    # while acc > 0.90:
    #     eps_hat *= 2.0
    #     warm1 = HMC_jx_adapt_ep(xs, ys, l=-1, p=xs.shape[1], omega=1, k=k,
    #         num_leapfrog_steps=Lmax, step_size=eps_hat, num_steps=400,
    #         seed=seed, theta0=theta0, chol_M=None, adapt_eps=True, target_accept=0.8)
    #     acc = mean_acc(warm1)

    # while acc < 0.60:
    #     eps_hat *= 0.5
    #     warm1 = HMC_jx_adapt_ep(xs, ys, l=-1, p=xs.shape[1], omega=1, k=k,
    #         num_leapfrog_steps=Lmax, step_size=eps_hat, num_steps=400,
    #         seed=seed, theta0=theta0, chol_M=None, adapt_eps=True, target_accept=0.8)
    #     acc = mean_acc(warm1)

    eps_hat = float(warm1["step_size"][-1])
    theta_last = warm1["theta"][-1]

    # ---- Learn DENSE mass matrix from warm-up draws
    S = jnp.cov(thetas_warm1, rowvar=False)

    # shrink toward diagonal to reduce noise (optional but helpful)
    tau = 0.1
    S = (1 - tau) * jnp.diag(jnp.diag(S)) + tau * S

    # normalize so median variance is ~1
    med_var = jnp.median(jnp.diag(S))
    S_norm = S / jnp.where(med_var > 0, med_var, 1.0)

    # clip eigenvalues to avoid extremes
    evals, Q = jnp.linalg.eigh(S_norm)
    floor = 1e-3          # relative to normalized scale
    ceil  = 1e+3
    evals = jnp.clip(evals, floor, ceil)
    S_tilde = (Q * evals) @ Q.T

    # jitter and Cholesky
    jitter = 1e-6
    chol_M = jnp.linalg.cholesky(S_tilde + jitter * jnp.eye(S_tilde.shape[0]))

    # ---- Warm-up 2: freeze metric, adapt ε a bit more
    warm2 = HMC_jx_adapt_ep_st(xs, ys, p=ys.shape[0], v=v, k=k,
                   num_leapfrog_steps=Lmax, step_size=eps_hat,
                   num_steps=num_warmup//2, seed=seed, theta0=theta_last,
                   chol_M=chol_M, adapt_eps=adapt_ep, target_accept=0.8)

    theta_last = warm2["theta"][-1]
    eps_hat    = warm2["step_size"][-1]

    # ---- Sampling: metric + step size frozen
    hmc_output = HMC_jx_adapt_ep_st(xs, ys, p=ys.shape[0], v=v, k=k,
                   num_leapfrog_steps=Lmax, step_size=eps_hat,
                   num_steps=num_steps, seed=seed, theta0=theta_last,
                   chol_M=chol_M, adapt_eps=adapt_ep)
    
    acc = mean_acc(hmc_output)
    jax.debug.print(str(acc))

    tlp = multid_logprob_jx_st(xs, ys, hmc_output['theta'][-1,:], n=xs.shape[1],v=3, k=k)
    jax.debug.print('log likelihood:{}', tlp)

    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(np.array(hmc_output['theta'])[burn_in:, 0])
    plt.xlabel(r'$v_0$')
    plt.subplot(2,1,2)
    plt.plot((np.array(hmc_output['theta'])[burn_in:, 1]))
    plt.xlabel(r'$w$')
    # plt.subplot(3,2,3)
    # plt.plot((np.array(hmc_output['theta'])[burn_in:, 2]))
    # plt.xlabel(r'$w_2$') # k
    # plt.subplot(3,2,4)
    # plt.plot((np.array(hmc_output['theta'])[burn_in:, 3]))
    # plt.xlabel(r'$w_3$')
    # plt.subplot(3,2,5)
    # plt.plot((np.array(hmc_output['theta'])[burn_in:, 4]))
    # plt.xlabel(r'$w_4$')
    # plt.subplot(3,2,6)
    # plt.plot((np.array(hmc_output['theta'])[burn_in:, 5]))
    # plt.xlabel(r'$w_5$')
    if save:
        plt.savefig(f'figure\\wine\\para_fitting\\st\\w1\\para_seed{seedn}.png')

    


    plt.figure()
    plt.hist(np.exp(np.minimum(np.array(hmc_output['log_acc_ratio']), 0)))
    plt.xlabel('accept ratio')
    if save:
        plt.savefig(f'figure\\wine\\para_fitting\\st\\w1\\acc_seed{seedn}.png')

    # autocorrelation
    s = pd.Series(np.array(hmc_output['theta'])[:, 0])
    lags = np.arange(int(num_steps/2))
    acf = [s.autocorr(lag=i) for i in lags]
    plt.figure()
    plt.plot(lags, acf)
    if save:
        plt.savefig(f'figure\\wine\\para_fitting\\st\\w1\\acf_seed{seedn}.png')
    plt.show()

    if save:
        mean = np.array(hmc_output['theta'])[burn_in:, :].mean(axis=0)
        mean0 = np.ones(xs.shape[0] + 1)*w0
        mean0[0] = mean[0]
        lgprob=multid_logprob_jx_st(xs, ys, mean0, n=ys.shape[0], v=3,k=0.1)
        print(f'mean: {mean}, logprob: {lgprob}')
        jnp.savez(f'data/theta_st_{int(-w0)}_w1.npz', theta = mean, logprob = lgprob)
    
    return np.array(hmc_output['theta'])[-500:, :].mean(axis=0)

def HMC_jx_adapt_ep_g(x, y,  p, k,
           num_leapfrog_steps, step_size, num_steps, seed, theta0,
           chol_M=None, adapt_eps=False, target_accept=0.8, Lmin=30):
    """
    HMC with optional DENSE mass matrix M (given via its Cholesky factor chol_M, lower-triangular).
    If chol_M is None -> identity metric.
    """
    D = theta0.size
    if chol_M is None:
        chol_M = jnp.eye(D)

    # helpers for metric
    def sample_momentum(key):
        z = jax.random.normal(key, (D,))
        return chol_M @ z                           # p ~ N(0, M)

    def Minv_dot(vi):
        # x = M^{-1} v via triangular solves (no explicit inverse)
        y = solve_triangular(chol_M, vi, lower=True)
        x = solve_triangular(chol_M.T, y, lower=False)
        return x

    def kinetic(p):
        return 0.5 * jnp.dot(p, Minv_dot(p))

    def leapfrog_step(grad, step_size, i, leapfrog_state):
        theta, m, tlp_grad = leapfrog_state
        eps_i = step_size                           # constant within trajectory
        m     = m + 0.5 * eps_i * tlp_grad
        theta = theta + eps_i * Minv_dot(m)         # q' = M^{-1} p
        tlp_grad = grad(x, y, theta, p, k)
        m     = m + 0.5 * eps_i * tlp_grad
        return theta, m, tlp_grad

    def hmc_step(grad, step_size, num_lf_steps, theta, seed, adapt_eps):
        m_seed, mh_seed = jax.random.split(seed)

        tlp_grad = grad(x, y, theta, p,  k)
        tlp      = multid_logprob_jx_g(x, y, theta,  k, p)

        m = sample_momentum(m_seed)                 # p ~ N(0, M)

        energy = kinetic(m) - tlp

        new_theta, new_m, _ = jax.lax.fori_loop(
            0, num_lf_steps,
            partial(leapfrog_step, grad, step_size),
            (theta, m, tlp_grad)
        )

        new_tlp = multid_logprob_jx_g(x, y, new_theta, k, p)
        new_energy = kinetic(new_m) - new_tlp
        # checkify.check(len(new_tlp)>1, "lop accept scalar")

        log_accept_ratio = energy - new_energy

        accept_prob = jnp.minimum(1.0, jnp.exp(log_accept_ratio))
        is_accepted = jnp.log(jax.random.uniform(mh_seed, ())) < log_accept_ratio

        # jax.debug.print('is accepted:{}, new theta {}, theta: {}', is_accepted, new_theta, theta)
        theta = jnp.where(is_accepted, new_theta, theta)

        step_size_new = step_size
        if adapt_eps:
            step_size_new = step_size * jnp.exp(0.05 * (accept_prob - target_accept))

        hmc_op = {
            "theta": theta,
            "is_accepted": is_accepted,
            "log_acc_ratio": log_accept_ratio,
            "accept_prob": accept_prob,
            "step_size": step_size_new,
        }
        return theta, hmc_op

    # run scan
    seeds = jax.random.split(seed, num_steps)
    Ls    = jax.random.randint(seeds[0], (num_steps,), Lmin, num_leapfrog_steps + 1)

    def scan_body(state, carry):
        theta, eps = state
        key, L = carry
        theta, out = hmc_step(multid_grad_jx_g, eps, L, theta, key, adapt_eps)
        return (theta, out["step_size"]), out

    (theta_final, eps_final), outs = jax.lax.scan(
        scan_body, (theta0, step_size), (seeds, Ls)
    )
    return outs

def HMC_fn_jx_adapt_ep_g(xs, ys, k=0.1, save=True, seedn=1400,
              eps0=1e-3, Lmax=100, Lmin = 40, num_steps=1100,
              burn_in=300, load=False, num_warmup=800, v=3,
              init_scale = 0.1, w0=-5, adapt_ep = False):

    xs, ys = read_wine(N=360, seedn=seedn, random=True)

    seed = jax.random.PRNGKey(seedn)

    if load:
        theta0 = jnp.array(jnp.load(f'data/theta_g_{int(-w0)}_w1.npz')['theta'])
        
    else:
        key_init, seed_warmup = jax.random.split(seed)
        # mean0 = jnp.ones(xs.shape[0] + 1)*w0
        mean0 = jnp.ones(2)*w0 # v0, w
        mean0 = mean0.at[0].set(jnp.array(2))

        theta0 = init_scale * jax.random.normal(key_init, (2,)) + mean0
        

    # ---- Warm-up 1: identity metric, adapt ε
    warm1 = HMC_jx_adapt_ep_g(xs, ys, p=ys.shape[0], k=k,
                   num_leapfrog_steps=Lmax, step_size=eps0,
                   num_steps=num_warmup, seed=seed, theta0=theta0,
                   chol_M=None, adapt_eps=adapt_ep, target_accept=0.8, Lmin=Lmin)

    thetas_warm1 = warm1["theta"]              # (num_warmup, D)
    theta_last   = thetas_warm1[-1]
    eps_hat      = warm1["step_size"][-1]

    def mean_acc(outs):
        return float(jnp.asarray(outs["accept_prob"]).mean())

    # after warm1
    eps_hat = float(warm1["step_size"][-1])
    acc = mean_acc(warm1)
    jax.debug.print(str(acc))
    jax.debug.print(str(eps_hat))

    # # push acceptance into 0.75–0.85
    # while acc > 0.90:
    #     eps_hat *= 2.0
    #     warm1 = HMC_jx_adapt_ep(xs, ys, l=-1, p=xs.shape[1], omega=1, k=k,
    #         num_leapfrog_steps=Lmax, step_size=eps_hat, num_steps=400,
    #         seed=seed, theta0=theta0, chol_M=None, adapt_eps=True, target_accept=0.8)
    #     acc = mean_acc(warm1)

    # while acc < 0.60:
    #     eps_hat *= 0.5
    #     warm1 = HMC_jx_adapt_ep(xs, ys, l=-1, p=xs.shape[1], omega=1, k=k,
    #         num_leapfrog_steps=Lmax, step_size=eps_hat, num_steps=400,
    #         seed=seed, theta0=theta0, chol_M=None, adapt_eps=True, target_accept=0.8)
    #     acc = mean_acc(warm1)

    eps_hat = float(warm1["step_size"][-1])
    theta_last = warm1["theta"][-1]

    # ---- Learn DENSE mass matrix from warm-up draws
    S = jnp.cov(thetas_warm1, rowvar=False)

    # shrink toward diagonal to reduce noise (optional but helpful)
    tau = 0.1
    S = (1 - tau) * jnp.diag(jnp.diag(S)) + tau * S

    # normalize so median variance is ~1
    med_var = jnp.median(jnp.diag(S))
    S_norm = S / jnp.where(med_var > 0, med_var, 1.0)

    # clip eigenvalues to avoid extremes
    evals, Q = jnp.linalg.eigh(S_norm)
    floor = 1e-3          # relative to normalized scale
    ceil  = 1e+3
    evals = jnp.clip(evals, floor, ceil)
    S_tilde = (Q * evals) @ Q.T

    # jitter and Cholesky
    jitter = 1e-6
    chol_M = jnp.linalg.cholesky(S_tilde + jitter * jnp.eye(S_tilde.shape[0]))

    # ---- Warm-up 2: freeze metric, adapt ε a bit more
    warm2 = HMC_jx_adapt_ep_g(xs, ys, p=ys.shape[0], k=k,
                   num_leapfrog_steps=Lmax, step_size=eps_hat,
                   num_steps=num_warmup//2, seed=seed, theta0=theta_last,
                   chol_M=chol_M, adapt_eps=adapt_ep, target_accept=0.8)

    theta_last = warm2["theta"][-1]
    eps_hat    = warm2["step_size"][-1]

    # ---- Sampling: metric + step size frozen
    hmc_output = HMC_jx_adapt_ep_g(xs, ys, p=ys.shape[0], k=k,
                   num_leapfrog_steps=Lmax, step_size=eps_hat,
                   num_steps=num_steps, seed=seed, theta0=theta_last,
                   chol_M=chol_M, adapt_eps=adapt_ep)
    
    acc = mean_acc(hmc_output)
    jax.debug.print(str(acc))

    tlp = multid_logprob_jx_g(xs, ys, hmc_output['theta'][-1,:], n=xs.shape[1], k=k)
    jax.debug.print('log likelihood:{}', tlp)
    
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(np.array(hmc_output['theta'])[burn_in:, 0])
    plt.xlabel(r'$v_0$')
    plt.subplot(2,1,2)
    plt.plot((np.array(hmc_output['theta'])[burn_in:, 1]))
    plt.xlabel(r'$w$')
    # plt.subplot(3,2,3)
    # plt.plot((np.array(hmc_output['theta'])[burn_in:, 2]))
    # plt.xlabel(r'$w_2$') # k
    # plt.subplot(3,2,4)
    # plt.plot((np.array(hmc_output['theta'])[burn_in:, 3]))
    # plt.xlabel(r'$w_3$')
    # plt.subplot(3,2,5)
    # plt.plot((np.array(hmc_output['theta'])[burn_in:, 4]))
    # plt.xlabel(r'$w_4$')
    # plt.subplot(3,2,6)
    # plt.plot((np.array(hmc_output['theta'])[burn_in:, 5]))
    # plt.xlabel(r'$w_5$')
    if save:
        plt.savefig(f'figure\\wine\\para_fitting\\g\\w1\\para_seed{seedn}.png')

    


    plt.figure()
    plt.hist(np.exp(np.minimum(np.array(hmc_output['log_acc_ratio']), 0)))
    plt.xlabel('accept ratio')
    if save:
        plt.savefig(f'figure\\wine\\para_fitting\\g\\w1\\acc_seed{seedn}.png')

    # autocorrelation
    s = pd.Series(np.array(hmc_output['theta'])[:, 0])
    lags = np.arange(int(num_steps/2))
    acf = [s.autocorr(lag=i) for i in lags]
    plt.figure()
    plt.plot(lags, acf)
    if save:
        plt.savefig(f'figure\\wine\\para_fitting\\g\\w1\\acf_seed{seedn}.png')
    plt.show()

    if save:
        mean = np.array(hmc_output['theta'])[burn_in:, :].mean(axis=0)
        mean0 = np.ones(xs.shape[0] + 1)*w0
        mean0[0] = mean[0]
        lgprob=multid_logprob_jx_g(xs, ys, mean0, n=ys.shape[0], k=0.1)
        print(f'mean: {mean}, logprob: {lgprob}')
        jnp.savez(f'data/theta_g_{int(-w0)}_w1.npz', theta = mean, logprob = lgprob)
    
    return np.array(hmc_output['theta'])[-500:, :].mean(axis=0)


def HMC_fn_jx(xs, ys, k=0.1, save = True, seed = jax.random.PRNGKey(1400), eps0 = 1e-3,num_leapfrog_steps=100,
    num_steps=1100, burn_in = 100, load = False):
    xs, ys = read_wine(N=360)
    
    if load:
        theta = jnp.array(jnp.load('data/theta.npz')['theta'])
    else:
        # theta = jnp.ones(xs.shape[0]+3)*0.1
        # theta = theta.at[0].set(0.0)
        # theta = theta.at[1].set(1.0)

        theta = jnp.zeros(xs.shape[0]+3)

    hmc_output = HMC_jx(xs, ys, l=-1, p=xs.shape[1], omega=1, k=k, num_leapfrog_steps=num_leapfrog_steps, step_size=eps0, num_steps=num_steps, seed=seed, theta0=theta)


    plt.figure()
    plt.subplot(3,2,1)
    plt.plot(np.array(hmc_output['theta'])[burn_in:, 0])
    plt.xlabel(r'$\beta$')
    plt.subplot(3,2,2)
    plt.plot((np.array(hmc_output['theta'])[burn_in:, 1]))
    plt.xlabel(r'$v_0$')
    plt.subplot(3,2,3)
    plt.plot((np.array(hmc_output['theta'])[burn_in:, 6]))
    plt.xlabel(r'$w_4$') # k
    plt.subplot(3,2,4)
    plt.plot((np.array(hmc_output['theta'])[burn_in:, 3]))
    plt.xlabel(r'$w_1$')
    plt.subplot(3,2,5)
    plt.plot((np.array(hmc_output['theta'])[burn_in:, 4]))
    plt.xlabel(r'$w_2$')
    plt.subplot(3,2,6)
    plt.plot((np.array(hmc_output['theta'])[burn_in:, 5]))
    plt.xlabel(r'$w_3$')
    if save:
        plt.savefig(f'figure\\wine\\para_fitting\\para_seed{seed}.png')

    


    plt.figure()
    plt.hist(np.exp(np.minimum(np.array(hmc_output['log_acc_ratio']), 0)))
    plt.xlabel('accept ratio')
    if save:
        plt.savefig(f'figure\\wine\\para_fitting\\acc_seed{seed}.png')

    # autocorrelation
    s = pd.Series(np.array(hmc_output['theta'])[:, 0])
    lags = np.arange(int(num_steps/2))
    acf = [s.autocorr(lag=i) for i in lags]
    plt.figure()
    plt.plot(lags, acf)
    if save:
        plt.savefig(f'figure\\wine\\para_fitting\\acf_seed{seed}.png')
    plt.show()

    if save:
        jnp.savez('data/theta.npz', theta = np.array(hmc_output['theta'])[burn_in:, :].mean(axis=0))
    
    return np.array(hmc_output['theta'])[-500:, :].mean(axis=0)
    # print(hmc_output['theta'], hmc_output['is_accepted'], np.exp(np.array(hmc_output['log_acc_ratio'])))

def wine_reg(save=False, model = 'gh',n=360,m=40, return_mse = True, random = True,log_w=-5, beta = 0.0, mu = 0.0, theta=np.ones(14),
             v=3, l =1, omega = 1, seedn=1, k=0.1):

    xs, ys = read_wine(N=400, random=random, seedn=seedn)
    
    train_xs, train_ys = xs[:,:n], ys[:n]
    test_xs, test_ys = xs[:,n:n+m], ys[n:n+m]

    # theta = np.array(np.load('data/theta.npz')['theta'])
    print(theta)
    if model == 'gh':
        # import pdb;pdb.set_trace()
        mean, var = regression(train_xs, train_ys, test_xs, a_b=theta[0], 
                               v_0=np.exp(theta[1]), k=k, ws=np.exp(theta[2:]), mu=mu,
                               omega=omega, l=l,)
    elif model == 'st':
        mean, var = st_regression(train_xs, train_ys, test_xs, v_0=np.exp(theta[0]), k=k, ws=np.exp(theta[1:]), phi=mu, v=v)
    else: # model == 'g':
        mean, var = g_regression(train_xs, train_ys, test_xs, v_0=np.exp(theta[0]), k=k, ws=np.exp(theta[1:]), phi=mu)

    idx = np.where((mean>=0) & (mean<=10))

    mse = np.dot(mean-test_ys, mean-test_ys)/m
    if return_mse:
        return mse
    print(np.dot(mean-test_ys, mean-test_ys)/m)

    from scipy.stats import spearmanr
    rho, p_value = spearmanr(test_ys, mean)
    print("Spearman correlation:", rho)
    print("p-value:", p_value)

    try:
        plt.scatter(train_xs[0,:], train_ys, label='train')
    except:
        import pdb; pdb.set_trace()
    plt.scatter(test_xs[0,:], test_ys, label='test')
    plt.scatter(test_xs[0,idx], mean[idx], label='pred')
    plt.legend()
    if save:
        plt.savefig(f'figure/wine/reg/{model}_dim0.png')
    plt.show()

def likelihood_grid(N=500, random = False,
                    seedn=4, model = 'gh', mls=[0,1],
                    n=10, beta = 0.5, lg_v0 = 1.5, log_w = 4):
    # generate data
    k=0.1

    xs, ys = read_wine(N=360, random=random, seedn=seedn)
    # train_xs, train_ys = xs[:,40:], ys[40:]
    # test_xs, test_ys = xs[:,-40:], ys[-40:]
    
    # grid   
    a_bs = np.linspace(0,10,num=n)
    v_0_lgs = np.linspace(-5,10,num=n)
    wl_lgs = np.linspace(-5,10,num=n)

    # v_0_lgs = np.linspace(-3.5,4.5,num=n)
    # wl_lgs = np.linspace(-3.5,4.5,num=n)
    
    ks = np.linspace(0.01, 0.2, num=n)
    mus = np.linspace(-2,0,num=n)
    ls, omega_logs = np.linspace(10,100,num=n), np.linspace(-6,3,num=n), 

    if model == 'gh':
        list_paras = [r'$a_b$',r'$v_0$' ,r'k',r'$w_1$',r'$w_2$',r'$w_3$',r'$w_4$',r'$w_5$', r'$l$', r'$omega$']
        list_ns = [a_bs, v_0_lgs, ks, wl_lgs,wl_lgs,wl_lgs,wl_lgs,wl_lgs, ls, omega_logs]
    else:
        list_paras = [r'$v_0$',r'$w_1$',r'$w_2$',r'$w_3$',r'$w_4$',r'$w_5$']
        list_ns = [v_0_lgs, wl_lgs,wl_lgs,wl_lgs,wl_lgs,wl_lgs]

    for m in mls:
        for l in mls:
            if l<=m:
                continue
            likelihoods = np.zeros((n,n))
            mses = np.zeros((n,n))
            X, Y =np.meshgrid(list_ns[m], list_ns[l])
        
            for i, (row_x , row_y) in tqdm(enumerate(zip(X, Y))):
                for j, (x , y) in enumerate(zip(row_x, row_y)):
                    theta = np.ones(14)*log_w
                    theta[1] = lg_v0
                    theta[0] = beta
                    # theta[m] = x
                    # theta[l] = y
                    lmda, omega = 1, 1
                    if list_paras[m] == r'$l$':
                        lmda = x
                    elif list_paras[m] == r'$omega$':
                        omega = np.exp(x)
                    else:
                        theta[m] = x

                    if list_paras[l] == r'$l$':
                        lmda = y
                    elif list_paras[l] == r'$omega$':
                        omega = np.exp(y)
                    else:
                        theta[3:] = y
                        # print(theta)
                    
                    if model == 'gh':
                        likelihoods[i,j] = (multid_logprob(xs, ys, theta, l=lmda, p=xs.shape[1], omega=omega,k=0.1))
                    else: # 'st'
                        theta = np.ones(12)
                        theta[0] = 1.0
                        theta[m] = x
                        theta[1:] *= y
                        likelihoods[i,j] = (multid_logprob_st(xs, ys, theta, p=xs.shape[1], v=3,k=0.1))
                        
                    # mses[i,j] = wine_reg(random=False, theta=theta)


            # import pdb; pdb.set_trace()
            idx = np.argmax(likelihoods)
            idx = np.unravel_index(idx, likelihoods.shape)
            print(X[idx], Y[idx])
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            surf = ax.plot_surface(X, Y, likelihoods, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
            ax.set_xlabel(list_paras[m])
            ax.set_ylabel(list_paras[l])
            plt.savefig(f'figure/wine/like_plot/{model}_{list_paras[m]}{list_paras[l]}.png')

            # # import pdb; pdb.set_trace()
            # idx = np.argmin(mses)
            # idx = np.unravel_index(idx, mses.shape)
            # print(X[idx], Y[idx])
            # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            # surf = ax.plot_surface(np.log10(X), np.log10(Y), mses, cmap=cm.coolwarm,
            #                 linewidth=0, antialiased=False)
            # ax.set_xlabel(list_paras[m])
            # ax.set_ylabel(list_paras[l])
            # plt.savefig(f'figure/wine/like_plot/{list_paras[m]}{list_paras[l]}_mse.png')
    plt.show()

def mse_grid(seed =1 ):
    n=5

    nmu = 10
    nb = 5
    mse_gh, mse_st = np.zeros((nmu, nb, 2)), np.zeros((nmu,2))
    mus = np.linspace(0,10, num = nmu)
    betas = np.linspace(0,5, num=nb)
    X, Y = np.meshgrid(betas, mus, indexing='xy')

    # gh
    for i in tqdm(range(len(mus))):
        for j in range(len(betas)):
            mse_ghn = np.zeros(n)
            for k in range(n):
                mse_ghn[k] = wine_reg(model='gh', beta=betas[j], mu = mus[i], random=True, seedn = seed)

            mse_gh[i,j,0] = mse_ghn.mean()
            mse_gh[i,j,1] = mse_ghn.std()


    fig = plt.figure(figsize=plt.figaspect(0.5))
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    surf = ax.plot_surface(X, Y, mse_gh[:,:,0], rstride=1, cstride=1, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
    ax.set_xlabel(r'$\mu$')
    ax.set_ylabel(r'$\beta$')
    ax.set_zlabel(r'mse')

    # st
    for i in tqdm(range(len(mus))):
        mse_stn = np.zeros(n)
        for k in range(n):
            mse_stn[k] = wine_reg(model='st', mu = mus[i], random=True, seedn=seed)

        mse_st[i,0] = mse_stn.mean()
        mse_st[i,1] = mse_stn.std()
            
    

    plt.figure()
    plt.plot(mus, mse_st[:,0])

    plt.show()

    # for i in tqdm(range(n)):
    #     mse_gh[i]=wine_reg(model='gh',beta=5.0, mu=0.0, random=False)
    #     mse_st[i]=wine_reg(model='st', mu=5.0, random= False)
    
    print(np.min(mse_gh[:,:,0]))
    print(np.min(mse_st[:,0]))

    print(mse_gh, mse_st)

def mse_opt(seedn=1, random = False):
    # tune w_l
    n=10
    log_ws = np.linspace(-9,1, num=n, endpoint=False)
    mse_gh, mse_st = np.zeros(n), np.zeros(n)
    for i, lg_w in enumerate(log_ws):
        theta = np.ones(14)*lg_w
        theta[0] = 1.1 # 0.8; 1,1
        theta[1] = -0.7 # -1.8; -0.7
        mse_gh[i] = wine_reg(return_mse=True,n=310,m=50, save=True, log_w=lg_w,
                             theta=theta, random=random, seedn=seedn)
        theta = np.ones(12)*lg_w
        theta[0] = 1.8
        mse_st[i] = wine_reg(model='st', n=310,m=50,return_mse=True, 
                             save=True,theta=theta, log_w=lg_w, random=random,
                             seedn=seedn)

    
    plt.plot(log_ws, mse_st, label='st')
    plt.plot(log_ws, mse_gh, label='gh')
    plt.legend()
    plt.savefig('figure/wine/reg/mse.png')
    plt.show()

    idx = np.argmin(mse_gh)
    print(log_ws[idx])
    theta = np.ones(14)*log_ws[idx] # -4
    theta[0] = 1.1
    theta[1] = -0.7 # -0.7
    wine_reg(return_mse=False, n=360, m=40, save=True, theta=theta, random=random, seedn=seedn)


    idx = np.argmin(mse_st)
    print(log_ws[idx]) # -5
    theta = np.ones(12)*log_ws[idx]
    theta[0] = 1.8
    wine_reg(model='st', return_mse=False, n=360, m=40, save=True, theta=theta,random=random, seedn=seedn)
   
def mse_stv(lg_w = -5, k =0.1, seed = 1):
    vs = np.linspace(1,10, num=10)+2
    mse_st = np.zeros(10)
    for i, v in enumerate(vs):
        theta = np.ones(12)*lg_w
        theta[0] = 1.8
        mse_st[i] = wine_reg(model='st', n=310,m=50,return_mse=True, 
                             save=True,theta=theta, log_w=lg_w, random=True,
                             v= v, k=k, seedn=seed)

    plt.plot(vs, mse_st, label='st')
    plt.legend()
    plt.savefig('figure/wine/reg/mse_vs.png')
    plt.show()

    idx = np.argmin(mse_st)
    print(vs[idx])
    theta = np.ones(12)*lg_w # -5
    theta[0] = 1.8
    wine_reg(model='st',return_mse=False, n=360, m=40, save=True, theta=theta, random=True, seedn=seed, k=k)

def mse_stk(lg_wst = -5,lg_wgh = -4, num = 11, seed = 1):
    ks = np.linspace(-10,0, num=num)
    mse_st = np.zeros(num)
    mse_gh = np.zeros(num)

    theta_st = np.ones(12)*lg_wst
    theta_st[0] = 1.8

    theta_gh = np.ones(14)*lg_wgh
    theta_gh[0] = 1.1 # 0.8; 1,1
    theta_gh[1] = -0.7 # -1.8; -0.7
    for i, k in enumerate(ks):
        mse_st[i] = wine_reg(model='st', n=310,m=50,return_mse=True, 
                             save=True,theta=theta_st, log_w=lg_wst, random=False,
                             k=np.exp(k), seedn=seed)
        
        mse_gh[i] = wine_reg(return_mse=True,n=310,m=50, save=True, 
                                log_w=lg_wgh,theta=theta_gh, random=False, 
                                k=np.exp(k), seedn=seed)

    plt.plot(ks, mse_st, label='st')
    plt.plot(ks, mse_gh, label='gh')
    plt.legend()
    plt.savefig('figure/wine/reg/mse_ks.png')
    plt.show()

    idx = np.argmin(mse_st)
    print(np.exp(ks[idx]))
    theta = np.ones(12)*lg_wst # -5
    theta[0] = 1.8
    wine_reg(model='st',return_mse=False, n=360, m=40, save=True, theta=theta, random=True, seedn=seed,
             k = np.exp(ks[idx]))

    idx = np.argmin(mse_gh)
    print(np.exp(ks[idx]))
    theta_gh = np.ones(14)*lg_wgh
    theta_gh[0] = 1.1 # 0.8; 1,1
    theta_gh[1] = -0.7 # -1.8; -0.7
    wine_reg(return_mse=False, n=360, m=40, save=True, theta=theta_gh, random=True, seedn=seed,
             k = np.exp(ks[idx]))


def mse_gh_lw(lg_w = -4, k =0.1, seed = 1):
    """
    Tune GH lambda and omega by MSE
    """
    nl = 5
    nw = 5
    mse_gh = np.zeros((nl, nw))
    ls = np.linspace(-10,10, num = nl, endpoint=True)
    omegas = np.linspace(1,20, num=nw)
    X, Y = np.meshgrid(ls, omegas, indexing='xy')

    # gh
    for i in tqdm(range(len(ls))):
        for j in range(len(omegas)):
            theta = np.ones(14)*lg_w
            theta[0] = 1.1 # 0.8; 1,1
            theta[1] = -0.7 # -1.8; -0.7
            mse_gh[i,j] = wine_reg(return_mse=True,n=310,m=50, save=True, 
                                   log_w=lg_w,theta=theta, random=True,seedn=seed, 
                                   l=ls[i], omega=omegas[j], k=k)

    idx = np.argmin(mse_gh)
    idx = np.unravel_index(idx, mse_gh.shape)
    print(X[idx], Y[idx])

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X, Y, mse_gh[:,:], rstride=1, cstride=1, cmap=cm.coolwarm,
                    linewidth=0, antialiased=False)
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'$\omega$')
    ax.set_zlabel(r'mse')

    plt.savefig(f'figure/wine/reg/gh_mse_lws.png')

    theta = np.ones(14)*lg_w
    theta[0] = 1.1 # 0.8; 1,1
    theta[1] = -0.7 # -1.8; -0.7
    wine_reg(return_mse=False,n=360,m=40, save=True, 
                    theta=theta, random=True, seedn=seed, 
                    l=X[idx], omega=Y[idx], k= k)

    # idx = np.argmin(mse_gh)
    # print(log_ws[idx])
    # theta = np.ones(14)*log_ws[idx] # -5
    # theta[0] = 1.1
    # theta[1] = -0.7 # -0.7
    # wine_reg(return_mse=False, n=360, m=40, save=True, theta=theta, random=False)

def compare_lg_like(seedn = 4, beta = 4, lg_w = 4, lg_v = 1.5):
    xs, ys = read_wine(N=360, random=True, seedn=seedn)

    theta = np.ones(14)*lg_v
    theta[0] = beta
    # theta[m] = x
    # theta[l] = y
    lmda, omega = 1, 1
    theta[3:] = lg_w

    likelihood = multid_logprob(xs, ys, theta, l=lmda, p=xs.shape[1], omega=omega,k=0.1)
    print(f'numpy log likelihood: {likelihood}' )

    theta_jnp = jnp.array([beta, lg_v, lg_w])
    new_tlp = multid_logprob_jx(xs, ys, theta_jnp, lmda, xs.shape[1], omega, 0.1)

    jax.debug.print('jax log likelihood: {}', new_tlp)


if __name__ == '__main__':
    HMC_fn_jx_adapt_ep(0,0,save=True,eps0=2e-3,seedn=4, 
                       w0=-2, burn_in=5, num_steps=1000, 
                       Lmax=100,Lmin=1, load=False, num_warmup=50, 
                       adapt_ep=True, l =2)

    # HMC_fn_jx_adapt_ep_st(0,0,save=False,eps0=2e-3, burn_in=50, num_steps=1000, Lmax=80,Lmin=1, 
    #                       load=False, num_warmup=100, w0=-5, adapt_ep=False,
    #                       seedn=4)

    # HMC_fn_jx_adapt_ep_g(0,0,save=True,eps0=2e-4, burn_in=50, num_steps=1000, 
    #                      Lmax=80,Lmin=1, load=True, seedn=4,
    #                      num_warmup=100, w0=-2, adapt_ep=True)

    # HMC_fn_jx(0,0,save=True, eps0=1e-5, num_leapfrog_steps=100, num_steps=1100, burn_in=100, load=False)
    # likelihood_grid(model = 'st', mls = [0,1], seedn=4)
    # likelihood_grid(model = 'gh', mls = [0,3], seedn=4, n=20, beta=5)
    # mse_grid()

    # compare_lg_like(seedn=4, beta=4)

    # mse_opt(random=True, seedn=4) # CV?
    # mse_stv()
    # mse_gh_lw(lg_w=-4)

    # mse_stk(seed=4)

    seedn = 4
    gh_theta = np.ones(13)
    xs, ys = read_wine(N=360, random=True, seedn=seedn)
    # gh_theta[:3] = [0.27790788, -0.9, -1]
    gh_theta[:3] = np.load(f'data/theta_seed{seedn}_w1.npz')['theta']
    gh_theta[3:]*= gh_theta[2]
    gh_theta[0] = -0.5
    print(multid_logprob_jx(xs, ys, gh_theta[:3], l = 1, p = xs.shape[1], omega=1, k=0.1))
    wine_reg(model='gh', theta = gh_theta, return_mse=False)

    st_theta = np.ones(12)
    st_theta[:2] = np.load(f'data/theta_st_{int(5)}_w1.npz')['theta']
    print(multid_logprob_jx_st(xs, ys, st_theta[:2], xs.shape[1], v=3, k=0.1))
    print(st_theta[:2])
    st_theta[2:]*= st_theta[1]
    wine_reg(model='st', theta = st_theta, return_mse=False)

    g_theta = np.ones(12)
    g_theta[:2] = np.load(f'data/theta_g_{int(2)}_w1.npz')['theta']
    print(multid_logprob_jx_g(xs, ys, g_theta[:2], k=0.1, n=xs.shape[1]))
    print(g_theta[:2])
    g_theta[2:]*= g_theta[1]
    wine_reg(model='g', theta = g_theta, return_mse=False)