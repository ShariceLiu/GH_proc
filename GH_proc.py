# simulate some sample paths from GH proc
# exponential covariance function
# zero mean
# constant beta? (what effect would beta have on the data point?)

from scipy import special, stats
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import pandas as pd

from tqdm import tqdm

import jax.numpy as jnp
import jax

from tools.tools import *
from functools import partial

def Multivariate_GH_1d(xs, l = -1, omega = 1, w = 1, v_0 = 1, a_b = 1, b_b = 0, k = 1,mu=-1):
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


def regression(train_xs, train_ys, test_xs, l = -1, omega = 1, w = 1, v_0 = 1, a_b = 1, b_b = 0, k =1):
    K11 = SE_cov(train_xs, train_xs, v_0, w) + k*np.identity(len(train_ys))
    K12 = SE_cov(train_xs, test_xs, v_0, w)
    K22 = SE_cov(test_xs, test_xs, v_0, w)

    beta1 = np.ones(len(train_ys))*a_b + train_xs*b_b
    beta2 = np.ones(len(test_xs))*a_b + test_xs*b_b

    l21 = l - len(train_ys)/2
    invK11 = np.linalg.inv(K11)
    gamma21 = omega + train_ys @ (invK11 @ train_ys)
    delta21 = omega + beta1 @ (invK11 @ beta1)
    mu21 = K12.T @ invK11 @ train_ys
    K221 = K22 - K12.T @ invK11 @ K12
    beta21 = beta2 - K12.T @ invK11 @ beta1

    omega21 = gamma21 * delta21
    k21 = np.sqrt(delta21/ gamma21)

    # EW = special.kn(l21+1, omega21)/special.kn(l21, omega21)
    EW = np.exp(log_kn_large_order(l21+1,  omega21) - log_kn_large_order(l21, omega21))
    # VarW = special.kn(l21+2, omega21)/special.kn(l21, omega21)\
    #         -(special.kn(l21+1,  omega21)/special.kn(l21, omega21))**2
    VarW = np.exp(log_kn_large_order(l21+2,  omega21) - log_kn_large_order(l21, omega21))\
            - EW**2
    mean21 = mu21 + EW*beta21
    var21 =  EW* K221 + VarW * np.outer(beta21, beta21)

    return mean21, var21

def reg_fn(k=1):
    N = 100
    a_b = 0
    # xs = np.random.uniform(0,10, size=100)
    xs = np.linspace(start=0, stop=10, endpoint=False, num=100)
    ids = np.linspace(start=0, stop=N-1, endpoint=False, num=N, dtype=int)
    train_ids = np.random.choice(ids, size=10, replace=False)
    
    fs, ys = Multivariate_GH_1d(xs,k=k)
    mean, var = regression(xs[train_ids], ys[train_ids], xs, k=k)
    std = np.sqrt(np.diagonal(var))

    plt.figure()
    plt.fill_between(xs, y1=mean+3*std, y2=mean -3*std, color='pink', alpha =0.5, label='3 std region')
    plt.plot(xs, fs ,label = 'A sample Fn')
    plt.plot(xs, mean,label ='Pred Mean')
    plt.scatter(xs[train_ids], ys[train_ids], label= 'Train', color = 'red')
    plt.legend()
    plt.savefig('figure/syn_data/reg.png')
    plt.show()

def HMC(x, y, l, p, omega, num_leapfrog_steps, step_size, num_steps, seed):

    def leapfrog_step(grad, step_size, leapfrog_state):
        new_theta, m, tlp_grad = leapfrog_state
        m -= 0.5 * step_size * tlp_grad
        new_theta = new_theta + step_size *m
        tlp_grad = grad(x, y, new_theta, l, p, omega)
        m -= 0.5* step_size * tlp_grad
        return new_theta, m, tlp_grad
    
    def hmc_step(theta, num_lf_steps, step_size, seed):
        
        tlp_grad = grad(x,y, theta, l, p, omega) 
        # import pdb; pdb.set_trace()
        tlp = logprob(x,y, theta, l, p, omega)
        m = np.random.multivariate_normal(np.zeros(len(theta)), np.identity(len(theta))*1)
        energy = 0.5* np.square(m).sum()/1.0 - tlp

        new_theta, new_m = theta, m
        log_probs = np.zeros(num_lf_steps)
        for i in tqdm(range(num_lf_steps)):
            tlp = logprob(x,y, new_theta, l, p, omega)
            log_probs[i] = tlp
            new_theta, new_m, tlp_grad = leapfrog_step(grad, step_size, (new_theta, new_m, tlp_grad))

        plt.figure()
        plt.plot(log_probs)
        plt.show()
        import pdb; pdb.set_trace()
        new_energy = 0.5* np.square(new_m).sum()/1.0 - logprob(x,y, new_theta, l, p, omega)
        log_accept_ratio = energy - new_energy
        is_accepted = np.log(np.random.uniform()) < log_accept_ratio
        # select proposed state if accepted
        if is_accepted:
            theta = new_theta
        return theta, is_accepted, log_accept_ratio
    
    # initialize
    theta = np.array([1.0,2.0,2.0]) # sample from some initial dist?
    # create a seed for each step
    seed = jax.random.PRNGKey(seed)
    seeds = jax.random.split(seed, num_steps)
    # repeatedly run hmc and accumulate the outputs
    thetas, is_accepted, log_accept_ratios = np.zeros((num_steps,3)), np.zeros(num_steps), np.zeros(num_steps)
    
    for i in tqdm(range(num_steps)):
        thetas[i,:], is_accepted[i], log_accept_ratios[i] = hmc_step(theta, num_leapfrog_steps, step_size, seed)
        theta = thetas[i,:]

    return thetas, is_accepted, log_accept_ratios

def HMC_jx(x, y, l, p, omega, k, num_leapfrog_steps, step_size, num_steps, seed):

    def leapfrog_step(grad, step_size, i, leapfrog_state):
        theta, m, tlp_grad = leapfrog_state

        step_size = step_size * 0.95** i
        
        m += 0.5 * step_size * tlp_grad
        theta += step_size *m
        tlp_grad = grad(x, y, theta, l, p, omega, k)
        m += 0.5* step_size * tlp_grad
        # jax.debug.print('log prob: {} theta: {}', logprob_jx(x,y, theta, l, p, omega), theta)
        return theta, m, tlp_grad
    
    def hmc_step(grad, num_lf_steps, step_size, theta, seed):
        m_seed, mh_seed = jax.random.split(seed)
        tlp_grad = grad(x,y, theta, l, p, omega, k) 
        tlp = logprob_jx(x,y, theta, l, p, omega, k)
        m = jax.random.normal(m_seed, theta.shape)
        energy = 0.5* jnp.square(m).sum() - tlp

        jax.debug.print('log prob: {} theta: {}', tlp, theta)

        new_theta, new_m = theta, m
        new_theta, new_m, _ = jax.lax.fori_loop(
            0,
            num_lf_steps,
            partial(leapfrog_step, grad, step_size),
            (theta, m, tlp_grad)
        )
        new_energy = 0.5* jnp.square(new_m).sum() - logprob_jx(x,y, new_theta, l, p, omega, k)
        log_accept_ratio = energy - new_energy
        is_accepted = jnp.log(jax.random.uniform(mh_seed,[])) < log_accept_ratio
        # select proposed state if accepted
        theta = jnp.where(is_accepted, new_theta, theta)
        hmc_op = {'theta': theta,
                  'is_accepted': is_accepted,
                  'log_acc_ratio':log_accept_ratio}
        return theta, hmc_op
    
    # initialize
    # theta = 1.2
    theta = jnp.array([0.0, 1.5, 1.5, 0.5, 0.0]) # sample from some initial dist?
    # create a seed for each step
    # seed = jax.random.PRNGKey(seed)
    seeds = jax.random.split(seed, num_steps)
    # repeatedly run hmc and accumulate the outputs
    _, hmc_output = jax.lax.scan(partial(hmc_step, grad_jx,num_leapfrog_steps, step_size), theta, seeds)
    return hmc_output

def HMC_fn():
    N = 100
    a_b = 0
    # xs = np.random.uniform(0,10, size=100)
    xs = np.linspace(start=0, stop=10, endpoint=False, num=100)
    ids = np.linspace(start=0, stop=N-1, endpoint=False, num=N, dtype=int)
    train_ids = np.random.choice(ids, size=10, replace=False)
    
    fs, ys = Multivariate_GH_1d(xs, omega=1,k=1e-5)

    xs = np.array(xs)
    ys = np.array(ys)
    thetas, is_accepted, log_accept_ratios = HMC(xs, ys, l=1, p=1, omega=1, num_leapfrog_steps=5, step_size=1e-3, num_steps=3, seed=1700)
    print(thetas, is_accepted, log_accept_ratios)

    plt.figure()
    plt.subplot(2,2,1)
    plt.hist(thetas[:,0]) # a_b
    plt.xlabel('$a_b$')
    plt.subplot(2,2,2)
    plt.hist(thetas[:,1]) # v_0
    plt.xlabel('$v_0$')
    # plt.subplot(2,2,3)
    # plt.hist(thetas[:,2]) # w
    # plt.xlabel('w')
    plt.subplot(2,2,3)
    plt.hist(np.exp(log_accept_ratios))
    plt.xlabel('accept ratio')

    plt.show()

def HMC_fn_jx(xs, ys, k=0.1, save = True, simulation = False, seed = jax.random.PRNGKey(1400), eps0 = 1e-3,num_leapfrog_steps=100,
    num_steps=1100, burn_in = 100):
    if simulation:
        N = 200
        a_b = 0
        k=0.1
        # xs = np.random.uniform(0,10, size=100)
        # xs = np.linspace(start=0, stop=100, endpoint=False, num=100) # if number too large: cause overflow when calculating bessel fucntion with a large index; too small, overfit..?
        
        # fs, ys = Multivariate_GH_1d(xs, omega=1,k=k)
        # jnp.savez('xy_k01.npz', x = jnp.array(xs), y = jnp.array(ys))

        file = jnp.load(f'xy_k01_N{N}.npz')
        xs = jnp.array(file['x'])
        ys = jnp.array(file['y'])

    
    hmc_output = HMC_jx(xs, ys, l=-1, p=len(xs), omega=1, k=k, num_leapfrog_steps=num_leapfrog_steps, step_size=eps0, num_steps=num_steps, seed=seed)

    plt.figure()
    plt.subplot(3,2,1)
    plt.plot(np.array(hmc_output['theta'])[burn_in:, 0])
    plt.xlabel(r'$\beta$')
    plt.subplot(3,2,2)
    plt.plot((np.array(hmc_output['theta'])[burn_in:, 1])**2)
    plt.xlabel(r'$v_0$')
    plt.subplot(3,2,3)
    plt.plot((np.array(hmc_output['theta'])[burn_in:, 2])**2)
    plt.xlabel(r'$w$')
    plt.subplot(3,2,4)
    plt.plot((np.array(hmc_output['theta'])[burn_in:, 3])**2)
    plt.xlabel(r'$k$')
    plt.subplot(3,2,5)
    plt.plot((np.array(hmc_output['theta'])[burn_in:, 4]))
    plt.xlabel(r'$\mu$')
    # plt.subplot(3,2,6)
    # plt.plot((np.array(hmc_output['theta'])[burn_in:, 5])**2)
    # plt.xlabel(r'$k$')
    if save:
        plt.savefig(f'figure\\para_fitting\\beta_v0_w_mu_seed{seed}_eps1e_3_lf{num_leapfrog_steps}_st{num_steps}.png')


    plt.figure()
    plt.hist(np.exp(np.minimum(np.array(hmc_output['log_acc_ratio']), 0)))
    plt.xlabel('accept ratio')
    if save:
        plt.savefig(f'figure\\para_fitting\\acc_beta_v0_w_mu_seed{seed}_eps1e_3_lf{num_leapfrog_steps}_st{num_steps}.png')
    plt.show()

    return np.array(hmc_output['theta'])[100:, 0].mean(axis=0)
    # print(hmc_output['theta'], hmc_output['is_accepted'], np.exp(np.array(hmc_output['log_acc_ratio'])))

def likelihood_grid(N=500):
    # generate data
    k=0.1
    a_b = 1

    xs = np.linspace(start=0, stop=100, endpoint=False, num=N)
    fs, ys = Multivariate_GH_1d(xs, omega=1,k=k)

    np.savez(f'xy_k01_N{N}.npz', x = np.array(xs), y = np.array(ys))

    file = np.load(f'xy_k01_N{N}.npz')
    xs = np.array(file['x'])
    ys = np.array(file['y'])

    
    # grid
    n = 20
    
    a_bs = np.linspace(0,2,num=n)
    v_0_sqs = np.linspace(0.5,2,num=n)
    wl_sqs = np.linspace(0.5,2,num=n)
    ks = np.linspace(0.01, 0.2, num=n)
    mus = np.linspace(-2,0,num=n)
    l_sqs, omega_sqs = -np.linspace(0.1,10,num=n), np.linspace(0.1,10,num=n), 

    list_paras = [r'$a_b$',r'$v_0$',r'$w_l$', r'k',r'$mu$', r'$l$',  r'$omega$']
    list_ns = [a_bs, v_0_sqs, wl_sqs, ks, mus, l_sqs, omega_sqs]

    mls = [0,4]

    for m in mls:
        for l in mls:
            if l<=m:
                continue
            likelihoods = np.zeros((n,n))
            X, Y =np.meshgrid(list_ns[m], list_ns[l])
        
            for i, (row_x , row_y) in enumerate(zip(X, Y)):
                for j, (x , y) in enumerate(zip(row_x, row_y)):
                    theta = np.array([1.0, 1.0, 1.0, 0.1, -1.0, -1.0, 1.0])
                    theta[m] = x
                    theta[l] = y
                    likelihoods[i,j] = (logprob(xs, ys, theta, l=-1, p=len(xs), omega=1))

            # import pdb; pdb.set_trace()
            idx = np.argmax(likelihoods)
            idx = np.unravel_index(idx, likelihoods.shape)
            print(X[idx], Y[idx])
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            surf = ax.plot_surface(X, Y, likelihoods, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
            ax.set_xlabel(list_paras[m])
            ax.set_ylabel(list_paras[l])
            plt.savefig(f'figure/para_fitting/like_plot/{list_paras[m]}{list_paras[l]}.png')
    plt.show()

    return
    # bivariate plot
    a_v_likelihoods = np.zeros((n,n))
    A, V = np.meshgrid(a_bs, v_0_sqs)
    for i, (row_a_b , row_v_0_sq) in enumerate(zip(A, V)):
        for j, (a_b , v_0_sq) in enumerate(zip(row_a_b, row_v_0_sq)):
            a_v_likelihoods[i,j] = logprob(xs,ys, [a_b, v_0_sq, 1.0], l=-1, p=len(xs), omega=1, k=k)
    
    # plt.figure()
    # plt.contourf(A, V**2,a_v_likelihoods )
    # plt.xlabel(r'$a_b$')
    # plt.ylabel(r'$v_0$')
    # plt.colorbar()
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(A, V**2, a_v_likelihoods, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    ax.set_xlabel(r'$a_b$')
    ax.set_ylabel(r'$v_0$')
    
    plt.savefig('figure/para_fitting/like_plot/av.png')

    # bivariate plot
    a_w_likelihoods = np.zeros((n,n))
    A, W = np.meshgrid(a_bs, wl_sqs)
    for i, (row_a_b , row_wl_sq) in enumerate(zip(A, W)):
        for j, (a_b , wl_sq) in enumerate(zip(row_a_b, row_wl_sq)):
            a_w_likelihoods[i,j] = logprob(xs,ys, [a_b, 1.0, wl_sq], l=-1, p=len(xs), omega=1, k=k)
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(A, W**2, a_w_likelihoods, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    ax.set_xlabel(r'$a_b$')
    ax.set_ylabel(r'$w$')
    plt.savefig('figure/para_fitting/like_plot/aw.png')

    # bivariate plot
    w_v_likelihoods = np.zeros((n,n))
    W, V = np.meshgrid(wl_sqs, v_0_sqs)
    for i, (row_wl , row_v_0_sq) in enumerate(zip(W, V)):
        for j, (wl , v_0_sq) in enumerate(zip(row_wl, row_v_0_sq)):
            w_v_likelihoods[i,j] = logprob(xs,ys, [1.0, v_0_sq, wl], l=-1, p=len(xs), omega=1, k=k)
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(W**2, V**2, w_v_likelihoods, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
    ax.set_xlabel(r'$w$')
    ax.set_ylabel(r'$v_0$')
    plt.savefig('figure/para_fitting/like_plot/wv.png')

    plt.show()
    return

def inf_finance(k=0.01, seed = 1400):
    csvfile = r"C:\Users\95414\Desktop\CUED\phd\year1\mycode\data\data\NVIDIA CORPORATION (01-23-2025 09.30 _ 01-29-2025 16.00).csv"
    nvda_data_reloaded= pd.read_csv(csvfile, index_col=0, parse_dates=True)
    x_ns = jnp.array(nvda_data_reloaded['Price']['2025-01-29'], dtype=jnp.float32)
    x_ns = (x_ns - x_ns[0])
    ys = jnp.flip(x_ns)[:300] # pick the prev 300 data for const mu
    N = len(ys)
    xs = jnp.linspace(0, N,endpoint=False,num=N)
    
    ids = jnp.linspace(start=0, stop=N, endpoint=False, num=N, dtype=int)
    seed = jax.random.PRNGKey(seed)
    rc_seed, seed = jax.random.split(seed)
    train_ids = jax.random.choice(rc_seed, ids, shape=(1,200), replace=False).flatten()

    # beta, v0, w = HMC_fn_jx(xs = xs[train_ids], ys = ys[train_ids], k=k,save=False, seed= seed, eps0=2.5e-4,num_leapfrog_steps=100, num_steps=1100, burn_in=100)
    # print(beta, v0, w)

    beta,v0, w = 0.25, 0.1, 0.1

    mean, var = regression(xs[train_ids], ys[train_ids], xs, a_b=beta, v_0=v0, w=w,  k=k)
    std = np.sqrt(np.diagonal(var))

    plt.figure()
    
    plt.plot(xs, mean,label ='Pred Mean',zorder=1)
    plt.plot(xs, ys ,label = 'True', zorder=2)
    
    plt.fill_between(xs, y1=mean+3*std, y2=mean -3*std, color='pink', alpha =0.5, label='3 std region')
    plt.scatter(xs[train_ids], ys[train_ids], label= 'Train', color = 'red', s=4, zorder=10)
    
    plt.legend()
    plt.savefig('figure/nvidia/reg_k001.png')
    plt.show()

def MH(xs, ys, k=0.1, save = True, simulation = False, eps0 = 1e-3,
    num_steps=1100, burn_in = 100, sigma = 1.5, suffix = ""):
    if simulation:
        N = 100
        a_b = 0
        k=0.1
        # xs = np.random.uniform(0,10, size=100)
        # xs = np.linspace(start=0, stop=100, endpoint=False, num=100) # if number too large: cause overflow when calculating bessel fucntion with a large index; too small, overfit..?
        # ids = np.linspace(start=0, stop=N-1, endpoint=False, num=N, dtype=int)
        # # train_ids = np.random.choice(ids, size=10, replace=False)
        
        # fs, ys = Multivariate_GH_1d(xs, omega=1,k=k)
        # jnp.savez('xy_k01.npz', x = jnp.array(xs), y = jnp.array(ys))

        file = np.load('xy_k01.npz')
        xs = np.array(file['x'])
        ys = np.array(file['y'])

    samples = np.zeros((num_steps, 4))
    log_acc = np.zeros(num_steps)
    # ini = np.array([0.5, 1.5, 1.5, 0.5])
    ini = np.array([0.0, 2.0, 2.0, 1.0])
    var = sigma**2 * np.identity(4)
    var[-1]/= 3
    for i in tqdm(range(num_steps)):
        # propose
        x =np.random.multivariate_normal(np.array([0.0,1.5,1.5, 0.9]), var)
        x[1:] = x[1:]**2

        # draw a uniform rv
        p = np.random.uniform()

        # acceptance rate
        log_a = logprob(xs, ys, x, l=-1, p=len(xs), omega=1) - logprob(xs, ys, ini, l=-1, p=len(xs), omega=1)
        if log_a == np.inf:
            continue
        log_acc[i] = min(0,log_a)
        if np.log(p) < log_a:
            ini = x # accept

        samples[i] = ini

    plt.figure()
    plt.subplot(2,2,1)
    plt.hist(samples[burn_in:, 0])
    plt.subplot(2,2,2)
    plt.hist(samples[burn_in:, 1])
    plt.subplot(2,2,3)
    plt.hist(samples[burn_in:, 2])
    plt.subplot(2,2,4)
    plt.hist(samples[burn_in:, 3])
    plt.savefig(f'figure/para_fitting/MH/para{suffix}.png')

    plt.figure()
    plt.hist(np.exp(log_acc))
    plt.savefig(f'figure/para_fitting/MH/acc{suffix}.png')
    plt.show()
    



    

if __name__ == '__main__':
    # reg_fn(k=1)
    HMC_fn_jx(0,0,simulation=True, save=False, num_leapfrog_steps=50, eps0=1e-3, num_steps=100, burn_in=10)
    # MH(0,0, simulation=True, num_steps=3000, burn_in=200)
    # inf_finance()
    # eps_star, eps_n, alphas = dual_averaging(eps0=1e-4)
    # print(eps_star)

    # likelihood_grid(N=200)

    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.plot(eps_n)
    # plt.subplot(2,1,2)
    # plt.plot(alphas)
    # plt.show()


