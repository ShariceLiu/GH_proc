from scipy import special, stats
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import jax.numpy as jnp
import jax

from tools.tools import *
from functools import partial
from matplotlib import cm
from jax.scipy.linalg import solve_triangular

import pandas as pd

def Multivariate_GH_2d(xs, l = -1, omega = 1, w1 = 1, w2 = 1,  v_0 = 1, a_b = 1.0,mu=0, B_b = np.array((0.0,0.0)), k = 1):
    """
    xs: np array, input location, 2D, should be like 2 cols, N rows
    l: index parameter for GIG
    omega: a, b in GIG
    SE cov: v_0 exp( - w_1*(x_1-y_1)^2/2 - w_2*(x_2-y_2)^2/2 )
    beta(x) = a_b + B_b @ x
    k: noise variance
    """
    W = stats.geninvgauss.rvs(p=l, b=omega, size=1)[0]

    N = xs.shape[1]
    # exponential cov function
    K = SE_cov_2d(xs,xs, v_0, w1, w2)

    #skew
    beta = np.ones(N)*a_b + B_b @ xs

    # assume zero mean
    fs = np.random.multivariate_normal(beta*W + np.ones(N)*mu, (K)*W)
    ys = fs + np.random.multivariate_normal(beta*0, (k*np.identity(N))*W)

    return fs, ys

def regression(train_xs, train_ys, test_xs, l = -1, omega = 1, w1 = 1,w2 =1,  v_0 = 1, a_b = 1, b_b = np.array([0.0,0.0]), k =1, mu=0):
    K11 = SE_cov_2d(train_xs, train_xs, v_0, w1, w2) + k*np.identity(len(train_ys))
    K12 = SE_cov_2d(train_xs, test_xs, v_0, w1, w2)
    K22 = SE_cov_2d(test_xs, test_xs, v_0, w1, w2)

    beta1 = np.ones(len(train_ys))*a_b + b_b @ train_xs
    beta2 = np.ones(test_xs.shape[1])*a_b + b_b @ test_xs

    mu1 = np.ones(len(train_ys))*mu
    mu2 = np.ones(test_xs.shape[1])*mu

    l21 = l - len(train_ys)/2
    invK11 = np.linalg.inv(K11)
    gamma21 = omega + (train_ys - mu1) @ (invK11 @ (train_ys - mu1))
    delta21 = omega + beta1 @ (invK11 @ beta1)
    mu21 = mu2 + K12.T @ invK11 @ (train_ys - mu1)
    K221 = K22 - K12.T @ invK11 @ K12
    beta21 = beta2 - K12.T @ invK11 @ beta1

    omega21 = gamma21 * delta21
    k21 = np.sqrt(delta21/ gamma21)

    # import pdb; pdb.set_trace()
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

def reg_fn(k=1):
    n = 20
    N = n*n
    a_b = 0
    # xs = np.random.uniform(0,10, size=100)
    x1s = np.linspace(-10, 10, n)
    x2s = np.linspace(-10, 10, n)
    X, Y = np.meshgrid(x1s, x2s)
    xs = np.vstack([X.ravel(), Y.ravel()])

    ids = np.linspace(start=0, stop=N-1, endpoint=False, num=N, dtype=int)
    train_ids = np.random.choice(ids, size=40, replace=False)
    
    fs, ys = Multivariate_GH_2d(xs,k=k)
    mean, var = regression(xs[:,train_ids], ys[train_ids], xs, k=k)
    std = np.sqrt(np.diagonal(var))

    # plt.fill_between(xs, y1=mean+3*std, y2=mean -3*std, color='pink', alpha =0.5, label='3 std region')
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Plot the surface.
    # import pdb; pdb.set_trace()
    surf = ax.plot_surface(xs[0].reshape(X.shape), xs[1].reshape(X.shape), mean.reshape(X.shape), cmap=cm.Spectral,
                        linewidth=0, antialiased=False,label ='Pred Mean')
    # plt.plot(xs, fs ,label = 'A sample Fn')
    # plt.plot(xs, mean,label ='Pred Mean')
    ax.scatter(xs[0,train_ids], xs[1,train_ids],ys[train_ids].reshape(xs[0,train_ids].shape), label= 'Train', color = 'red')
    plt.legend()
    plt.savefig('figure/syn_data/2d/reg.png')
    plt.show()

def Multivariate_ST_2d(xs, v = 3,  w1 = 1, w2=1, v_0 = 1, k = 1):
    """
    xs: np array, input location
    v/2, tho/2: index parameter for Gamma, rho doesn't matter
    SE cov: v_0 exp(-w*(x-y)^2/2)
    k: noise variance
    """
    rho = 1

    W = stats.gamma.rvs(a=v/2, scale = 1/(rho/2), size=1)[0]
    W = (v-2)/rho/W # inv gamma

    N = xs.shape[1]
    # import pdb;pdb.set_trace()
    # exponential cov function
    K = SE_cov_2d(xs,xs, v_0, w1, w2)

    # assume zero mean
    fs = np.random.multivariate_normal(np.zeros(N), W*K)
    ys = fs + np.random.multivariate_normal(np.zeros(N), W*k*np.identity(N))

    np.savez('xy_st_2d.npz',x=jnp.array(xs), y = jnp.array(ys))

    return fs, ys

def regression_st(train_xs, train_ys, test_xs, v=3, w1 = 1,w2=1, v_0 = 1, k =1):
    K11 = SE_cov_2d(train_xs, train_xs, v_0, w1, w2) + k*np.identity(len(train_ys))
    K12 = SE_cov_2d(train_xs, test_xs, v_0, w1, w2)
    K22 = SE_cov_2d(test_xs, test_xs, v_0, w1, w2)

    invK11 = np.linalg.inv(K11)
    n1=len(train_ys)
    # v21 = v + n1
    phi21 = K12.T @ invK11 @ train_ys
    K221 = K22 - K12.T @ invK11 @ K12

    beta1 = train_ys @ (invK11@ train_ys)
    var21 = (v+ beta1 -2)/(v+ n1-2)*K221

    return phi21, var21

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
        tlp      = multi2d_logprob_jx_st(x, y, theta, p, v, k)

        m = sample_momentum(m_seed)                 # p ~ N(0, M)

        energy = kinetic(m) - tlp

        new_theta, new_m, _ = jax.lax.fori_loop(
            0, num_lf_steps,
            partial(leapfrog_step, grad, step_size),
            (theta, m, tlp_grad)
        )

        new_tlp = multi2d_logprob_jx_st(x, y, new_theta, p, v, k)
        new_energy = kinetic(new_m) - new_tlp
        # checkify.check(len(new_tlp)>1, "lop accept scalar")

        log_accept_ratio = energy - new_energy

        accept_prob = jnp.minimum(1.0, jnp.exp(log_accept_ratio))
        is_accepted = jnp.log(jax.random.uniform(mh_seed, ())) < log_accept_ratio

        jax.debug.print('is accepted:{}, new theta {}, theta: {}', is_accepted, new_theta, theta)
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
        theta, out = hmc_step(multi2d_grad_jx_st, eps, L, theta, key, adapt_eps)
        return (theta, out["step_size"]), out

    (theta_final, eps_final), outs = jax.lax.scan(
        scan_body, (theta0, step_size), (seeds, Ls)
    )
    return outs

def HMC_fn_jx_adapt_ep_st(xs, ys, k=0.1, save=True, seed=jax.random.PRNGKey(1400),
              eps0=1e-3, Lmax=100, Lmin = 40, num_steps=1100,
              burn_in=300, load=False, num_warmup=800, v=3, init_scale=0.1, init_mean = 0, dataset='_st_simu'):

    if load:
        theta0 = jnp.array(jnp.load(f'data/simu/theta_st{dataset}.npz')['theta'])
    else:
        key_init, seed_warmup = jax.random.split(seed)
        dim = xs.shape[0] + 1
        theta0 = init_scale * jax.random.normal(key_init, (dim,)) + init_mean # random init

    # ---- Warm-up 1: identity metric, adapt ε
    warm1 = HMC_jx_adapt_ep_st(xs, ys, p=ys.shape[0], v=v, k=k,
                   num_leapfrog_steps=Lmax, step_size=eps0,
                   num_steps=num_warmup, seed=seed, theta0=theta0,
                   chol_M=None, adapt_eps=True, target_accept=0.8, Lmin=Lmin)

    thetas_warm1 = warm1["theta"]              # (num_warmup, D)
    theta_last   = thetas_warm1[-1]
    eps_hat      = warm1["step_size"][-1]

    def mean_acc(outs):
        return float(jnp.asarray(outs["accept_prob"]).mean())

    # after warm1
    eps_hat = float(warm1["step_size"][-1])
    acc = mean_acc(warm1)
    jax.debug.print(str(acc))

    eps_hat = float(warm1["step_size"][-1])
    theta_last = warm1["theta"][-1]

    # ---- Learn DENSE mass matrix from warm-up draws
    S = jnp.cov(thetas_warm1, rowvar=False)
    # ridge jitter to ensure PD
    jitter = 1e-4 * (jnp.trace(S) / S.shape[0])
    S = S + jitter * jnp.eye(S.shape[0])
    chol_M = jnp.linalg.cholesky(S)

    # ---- Warm-up 2: freeze metric, adapt ε a bit more
    warm2 = HMC_jx_adapt_ep_st(xs, ys, p=ys.shape[0], v=v, k=k,
                   num_leapfrog_steps=Lmax, step_size=eps_hat,
                   num_steps=num_warmup//2, seed=seed, theta0=theta_last,
                   chol_M=chol_M, adapt_eps=True, target_accept=0.8)

    theta_last = warm2["theta"][-1]
    eps_hat    = warm2["step_size"][-1]

    jax.debug.print('theta: {}, eps: {}', theta_last, eps_hat)

    # ---- Sampling: metric + step size frozen
    hmc_output = HMC_jx_adapt_ep_st(xs, ys, p=ys.shape[0], v=v, k=k,
                   num_leapfrog_steps=Lmax, step_size=eps_hat,
                   num_steps=num_steps, seed=seed, theta0=theta_last,
                   chol_M=chol_M, adapt_eps=False)
    
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(np.array(hmc_output['theta'])[burn_in:, 0])
    plt.xlabel(r'$v_0$')
    plt.subplot(2,2,2)
    plt.plot((np.array(hmc_output['theta'])[burn_in:, 1]))
    plt.xlabel(r'$w_1$')
    plt.subplot(2,2,3)
    plt.plot((np.array(hmc_output['theta'])[burn_in:, 2]))
    plt.xlabel(r'$w_2$') # k

    if save:
        plt.savefig(f'figure\\simu2d\\st\\para_seed{seed}{dataset}.png')

    


    plt.figure()
    plt.hist(np.exp(np.minimum(np.array(hmc_output['log_acc_ratio']), 0)))
    plt.xlabel('accept ratio')
    if save:
        plt.savefig(f'figure\\simu2d\\st\\acc_seed{seed}{dataset}.png')

    # autocorrelation
    s = pd.Series(np.array(hmc_output['theta'])[:, 0])
    lags = np.arange(int(num_steps/2))
    acf = [s.autocorr(lag=i) for i in lags]
    plt.figure()
    plt.plot(lags, acf)
    if save:
        plt.savefig(f'figure\\simu2d\\st\\acf_seed{seed}{dataset}.png')
    plt.show()

    if save:
        jnp.savez(f'data/simu/theta_st{dataset}.npz', theta = np.array(hmc_output['theta'])[burn_in:, :].mean(axis=0))
    
    return np.array(hmc_output['theta'])[-500:, :].mean(axis=0)

def HMC_jx_adapt_ep_gh(x, y, l, p, omega, k,
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
        # jax.debug.print('tlp_grad: {} theta: {}', tlp_grad, theta[jnp.array([0,1,2],int)])
        m     = m + 0.5 * eps_i * tlp_grad
        return theta, m, tlp_grad

    def hmc_step(grad, step_size, num_lf_steps, theta, seed, adapt_eps):
        m_seed, mh_seed = jax.random.split(seed)

        tlp_grad = grad(x, y, theta, l, p, omega, k)
        tlp      = multi2d_logprob_jx(x, y, theta, l, p, omega, k)

        m = sample_momentum(m_seed)                 # p ~ N(0, M)

        energy = kinetic(m) - tlp
        # jax.debug.print('log prob: {} theta: {}', tlp, theta[jnp.array([0,1,2],int)])

        new_theta, new_m, _ = jax.lax.fori_loop(
            0, num_lf_steps,
            partial(leapfrog_step, grad, step_size),
            (theta, m, tlp_grad)
        )

        
        new_tlp = multi2d_logprob_jx(x, y, new_theta, l, p, omega, k)
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
        theta, out = hmc_step(multi2d_grad_jx, eps, L, theta, key, adapt_eps)
        return (theta, out["step_size"]), out

    (theta_final, eps_final), outs = jax.lax.scan(
        scan_body, (theta0, step_size), (seeds, Ls)
    )
    return outs

def HMC_fn_jx_adapt_ep_gh(xs, ys, k=0.1, save=True, seedn=1400,
              eps0=1e-3, Lmax=100, Lmin = 40, num_steps=1100,
              burn_in=300, load=False, num_warmup=800,init_scale=0.1, init_mean = 0, dataset = '_simu_gh', suf=''):

    jax.debug.print('shape: {}', xs.shape)
    seed = jax.random.PRNGKey(seedn)

    if load:
        theta0 = jnp.array(jnp.load(f'data/simu/theta_gh{dataset}.npz')['theta'])
    else:
        key_init, seed_warmup = jax.random.split(seed)
        dim = xs.shape[0] + 4
        theta0 = init_scale * jax.random.normal(key_init, (dim,)) + init_mean # random init

    # ---- Warm-up 1: identity metric, adapt ε
    warm1 = HMC_jx_adapt_ep_gh(xs, ys, l=-1, p=xs.shape[1], omega=1, k=k,
                   num_leapfrog_steps=Lmax, step_size=eps0,
                   num_steps=num_warmup, seed=seed, theta0=theta0,
                   chol_M=None, adapt_eps=True, target_accept=0.8, Lmin=Lmin)

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
    warm2 = HMC_jx_adapt_ep_gh(xs, ys, l=-1, p=xs.shape[1], omega=1, k=k,
                   num_leapfrog_steps=Lmax, step_size=eps_hat,
                   num_steps=num_warmup//2, seed=seed, theta0=theta_last,
                   chol_M=chol_M, adapt_eps=True, target_accept=0.8)

    theta_last = warm2["theta"][-1]
    eps_hat    = warm2["step_size"][-1]

    # ---- Sampling: metric + step size frozen
    hmc_output = HMC_jx_adapt_ep_gh(xs, ys, l=-1, p=xs.shape[1], omega=1, k=k,
                   num_leapfrog_steps=Lmax, step_size=eps_hat,
                   num_steps=num_steps, seed=seed, theta0=theta_last,
                   chol_M=chol_M, adapt_eps=False)
    
    plt.figure()
    plt.subplot(2,2,1)
    plt.plot(np.array(hmc_output['theta'])[burn_in:, 0])
    plt.xlabel(r'$\beta$')
    plt.subplot(2,2,2)
    plt.plot((np.array(hmc_output['theta'])[burn_in:, 1]))
    plt.xlabel(r'$v_0$')
    # plt.subplot(3,2,3)
    # plt.plot((np.array(hmc_output['theta'])[burn_in:, 2]))
    # plt.xlabel(r'$k$')
    plt.subplot(2,2,3)
    plt.plot((np.array(hmc_output['theta'])[burn_in:, 3]))
    plt.xlabel(r'$w_1$')
    plt.subplot(2,2,4)
    plt.plot((np.array(hmc_output['theta'])[burn_in:, 4]))
    plt.xlabel(r'$w_2$')
    # plt.subplot(3,2,6)
    # plt.plot((np.array(hmc_output['theta'])[burn_in:, 5]))
    # plt.xlabel(r'$\mu$')

    if save:
        plt.savefig(f'figure\\simu2d\\gh\\para_seed{seedn}{dataset}.png')

    


    plt.figure()
    plt.hist(np.exp(np.minimum(np.array(hmc_output['log_acc_ratio']), 0)))
    plt.xlabel('accept ratio')
    if save:
        plt.savefig(f'figure\\simu2d\\gh\\acc_seed{seedn}{dataset}.png')

    # autocorrelation
    s = pd.Series(np.array(hmc_output['theta'])[:, 0])
    lags = np.arange(int(num_steps/2))
    acf = [s.autocorr(lag=i) for i in lags]
    plt.figure()
    plt.plot(lags, acf)
    if save:
        plt.savefig(f'figure\\simu2d\\gh\\acf_seed{seedn}{dataset}.png')
    plt.show()

    if save:
        jnp.savez(f'data/simu/theta_gh{dataset}{suf}.npz', theta = np.array(hmc_output['theta'])[burn_in:, :].mean(axis=0))
        # move from device to host NumPy
        x_np = np.asarray(jax.device_get(hmc_output['theta']))  # or: np.array(x)

        # make a DataFrame (optionally name the columns)
        cols = [f"c{i}" for i in range(x_np.shape[1])]
        df = pd.DataFrame(x_np, columns=cols)

        # save if you like:
        df.to_csv(f"data/simu/theta_gh{dataset}{suf}.csv", index=False)      # CSV

    return np.array(hmc_output['theta'])[-500:, :].mean(axis=0)


def sample():
    xs = np.linspace(-5, 5, 20)
    ys = np.linspace(-5, 5, 20)

    X, Y = np.meshgrid(xs, ys)
    data = np.vstack([X.ravel(), Y.ravel()])
    fs, ys = Multivariate_GH_2d(data, a_b=-5)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    # Plot the surface.
    surf = ax.plot_surface(X, Y, fs.reshape(X.shape), cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
    plt.savefig('figure/sample_path/2d/beta.png')



def sample2dst(load = 'xy_st_2d.npz', dataset = '_simu_st', train=True):
    xs = jnp.linspace(-5, 5, 20)
    ys = jnp.linspace(-5, 5, 20)

    X, Y = jnp.meshgrid(xs, ys)
    data = jnp.vstack([X.ravel(), Y.ravel()])
    if load:
        xy = np.load(load)
        data = data
        ys = xy['y']
        # np.savez('xy_st_2d.npz',x=jnp.array(xs), y = jnp.array(ys))
    else:
        fs, ys = Multivariate_ST_2d(data, k=0.1)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # Plot the surface.
        surf = ax.plot_surface(X, Y, fs.reshape(X.shape), cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
        plt.savefig('figure/simu2d/st/surf.png')

    x_train, y_train, x_test, y_test = train_test_split_xy(data, ys, train_frac=0.9, seed = 1)

    if train:
        HMC_fn_jx_adapt_ep_st(x_train, y_train, k=0.1, save=True, seed=jax.random.PRNGKey(1400),
                eps0=1e-3, Lmax=100, Lmin = 1, num_steps=1000,
                burn_in=30, load=False, num_warmup=300, v=3, init_mean=-0.5, init_scale=0.1, dataset=dataset)
    
    theta = np.load(f'data/simu/theta_st{dataset}.npz')['theta']
    theta = np.exp(theta)
    mean, var = regression_st(x_train, y_train, x_test, w1=theta[1], w2=theta[2], v_0=theta[0], k=0.1)

    mse = sum((mean-y_test)**2)/len(y_test)

    print(f'MSE:{mse}') # st data: st: MSE:0.15580776537237645; gh: MSE:0.15610134601593018

    return mse
    

def sample2dGH(load = True, path = 'xy_gh_2d.npz', train=True, dataset = '', suf='', k=0.1, mu = 0.0, beta = 1.0,\
               eps0=1e-3, Lmax=100, Lmin = 1, num_steps=1000,
              burn_in=30, num_warmup=300, seed = 1400):
    xs = jnp.linspace(-5, 5, 20)
    ys = jnp.linspace(-5, 5, 20)

    X, Y = jnp.meshgrid(xs, ys)
    data = jnp.vstack([X.ravel(), Y.ravel()])
    if load:
        xy = np.load(path)
        data = data
        ys = xy['y']
        
        # np.savez('xy_st_2d.npz',x=jnp.array(xs), y = jnp.array(ys))
    else:
        fs, ys = Multivariate_GH_2d(data, k=k, mu=mu, a_b=beta)
        np.savez(path,x=jnp.array(data), y = jnp.array(ys),k=k, mu=mu, a_b=beta)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # Plot the surface.
        surf = ax.plot_surface(X, Y, fs.reshape(X.shape), cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
        plt.savefig(f'figure/simu2d/gh/surf{suf}.png')

    x_train, y_train, x_test, y_test = train_test_split_xy(data, ys, train_frac=0.9, seed = 1)

    if train:
        HMC_fn_jx_adapt_ep_gh(x_train, y_train, k=k, save=True, seedn=seed,
              eps0=eps0, Lmax=Lmax, Lmin = Lmin, num_steps=num_steps,
              burn_in=burn_in, load=False, num_warmup=num_warmup, init_mean=-0.0, init_scale=0.1, dataset=dataset, suf=suf)

    theta = np.load(f'data/simu/theta_gh{dataset}{suf}.npz')['theta']
    theta[1:-1] = np.exp(theta[1:-1])
    mean, var = regression(x_train, y_train, x_test, w1=theta[3], w2=theta[4], v_0=theta[1],a_b=theta[0],mu=theta[-1], k=k)

    mse = sum((mean-y_test)**2)/len(y_test)

    print(f'MSE:{mse}') # gh data: gh: MSE:0.26965320110321045; st: MSE:0.2840946707486451

    return mse
    

def like_plot(load = True, path = 'xy_gh_2d.npz', suf='',dataset='', k=0.1, mu = 0.0, beta = 1.0):
    xs = jnp.linspace(-5, 5, 20)
    ys = jnp.linspace(-5, 5, 20)

    X, Y = jnp.meshgrid(xs, ys)
    data = jnp.vstack([X.ravel(), Y.ravel()])
    if load:
        xy = np.load(path)
        data = data
        ys = xy['y']
    else:
        fs, ys = Multivariate_GH_2d(data, k=k, mu=mu, a_b=beta)
        np.savez(path,x=jnp.array(data), y = jnp.array(ys),k=k, mu=mu, a_b=beta)

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # Plot the surface.
        surf = ax.plot_surface(X, Y, fs.reshape(X.shape), cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
        plt.savefig(f'figure/simu2d/gh/surf{suf}.png')

    x_train, y_train, x_test, y_test = train_test_split_xy(data, ys, train_frac=0.9, seed = 1)

    df_csv = pd.read_csv(f"data/simu/theta_gh{dataset}.csv")          # loads into a DataFrame
    x_csv = np.asarray(df_csv.to_numpy())  # back to JAX array if you want

    # import pdb;pdb.set_trace()
    betas = x_csv[:,0]
    plt.figure(figsize=(12,7))
    plt.subplot(2,2,1)
    plt.ylim([0,175])
    plt.hist(betas, bins=20,range=[0.5,4.0], label='HMC estimate')

    n=20
    likelihoods = np.zeros(n)
    a_bs = jnp.linspace(0.5,4.0,num=n)*beta
    for i in range(n):
        theta = jnp.array([a_bs[i], 0.0, 0.0, 0.0, 0.0])
        likelihoods[i] = (multi2d_logprob_jx(x_train, y_train, theta, l=1, p=len(y_train), omega=1,k=0.1))

    # plt.subplot(2,1,2)
    print(max(np.exp(likelihoods)))
    plt.plot(a_bs, np.exp(likelihoods)/1e174*4, label='Likelihood')
    plt.vlines([1.0],ymin=0.0, ymax= 175, label=r'True $\beta$',linestyles='--', colors='pink')
    plt.legend()

    likelihoods = np.zeros(n)
    a_bs = jnp.linspace(-0.8,0.8,num=n)
    for i in range(n):
        theta = jnp.array([1.0, a_bs[i], 0.0, 0.0, 0.0])
        likelihoods[i] = (multi2d_logprob_jx(x_train, y_train, theta, l=1, p=len(y_train), omega=1,k=0.1))

    # plt.subplot(2,1,2)
    print(max(np.exp(likelihoods)))
    plt.subplot(2,2,2)
    plt.ylim([0,175])
    plt.hist(x_csv[:,1], bins=20,range=[-0.8,0.8], label='HMC estimate')
    plt.plot(a_bs, np.exp(likelihoods)/1e173*3, label='Likelihood')
    plt.vlines([0.0],ymin=0.0, ymax= 175, label=r'True $\exp(v_0)$',linestyles='--', colors='pink')
    plt.legend()

    likelihoods = np.zeros(n)
    a_bs = jnp.linspace(-0.8,0.8,num=n)
    for i in range(n):
        theta = jnp.array([1.0, 0.0, 0.0, a_bs[i], 0.0])
        likelihoods[i] = (multi2d_logprob_jx(x_train, y_train, theta, l=1, p=len(y_train), omega=1,k=0.1))
    # plt.subplot(2,1,2)
    print(max(np.exp(likelihoods)))
    plt.subplot(2,2,3)
    plt.ylim([0,250])
    plt.hist(x_csv[:,3], bins=20,range=[-0.8,0.8], label='HMC estimate')
    plt.plot(a_bs, np.exp(likelihoods)/1e173*4, label=r'Likelihood $w_1$')
    plt.vlines([0.0],ymin=0.0, ymax= 250, label=r'True $\exp(w_1)$', linestyles='--', colors='pink')
    plt.legend()

    likelihoods = np.zeros(n)
    a_bs = jnp.linspace(-0.8,0.8,num=n)
    for i in range(n):
        theta = jnp.array([1.0, 0.0, 0.0, 0.0, a_bs[i]])
        likelihoods[i] = (multi2d_logprob_jx(x_train, y_train, theta, l=1, p=len(y_train), omega=1,k=0.1))
    # plt.subplot(2,1,2)
    print(max(np.exp(likelihoods)))
    plt.subplot(2,2,4)
    plt.ylim([0,250])
    plt.hist(x_csv[:,4], bins=20,range=[-0.8,0.8], label='HMC estimate')
    plt.plot(a_bs, np.exp(likelihoods)/1e173*2.5, label=r'Likelihood $w_2$')
    plt.vlines([0.0],ymin=0.0, ymax= 250, label=r'True $\exp(w_2)$',linestyles='--', colors='pink')
    plt.legend()

    # plt.savefig('figure/simu2d/gh/v0_prob.png')

    plt.show()

if __name__ == '__main__':
    # reg_fn(k=0.1)
    # HMC_fn_jx()
    # eps_star, eps_n, alphas = dual_averaging(eps0=1e-4)
    # print(eps_star)

    # Multivariate_ST_2d()
    # sample2dst(load='xy_st_2d.npz', dataset='', train=False) # _simu_gh
    # sample2dGH(load=True,path='xy_gh_2d.npz', dataset='', suf='',mu=0.0, beta=1,k=0.1, train=True,
    #            eps0=1e-3, Lmax=100, Lmin = 1, num_steps=1000,
    #           burn_in=30, num_warmup=300, seed=1)

    like_plot(path='xy_gh_2d.npz')

    