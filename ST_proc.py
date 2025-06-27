
from scipy import special, stats
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import jax.numpy as jnp
import jax

from tools.tools import *
from functools import partial

import pandas as pd

def Multivariate_ST_1d(xs, v = 3,  w = 1, v_0 = 1, k = 1):
    """
    xs: np array, input location
    v/2, tho/2: index parameter for Gamma, rho doesn't matter
    SE cov: v_0 exp(-w*(x-y)^2/2)
    k: noise variance
    """
    rho = 1

    W = stats.gamma.rvs(a=v/2, scale = 1/(rho/2), size=1)[0]
    W = (v-2)/rho/W # inv gamma

    N = len(xs)
    # exponential cov function
    K = SE_cov(xs,xs, v_0, w)

    # assume zero mean
    fs = np.random.multivariate_normal(np.zeros(N), W*K)
    ys = fs + np.random.multivariate_normal(np.zeros(N), W*k*np.identity(N))

    return fs, ys

def regression(train_xs, train_ys, test_xs, v=3, w = 1, v_0 = 1, k =1):
    K11 = SE_cov(train_xs, train_xs, v_0, w) + k*np.identity(len(train_ys))
    K12 = SE_cov(train_xs, test_xs, v_0, w)
    K22 = SE_cov(test_xs, test_xs, v_0, w)

    invK11 = np.linalg.inv(K11)
    n1=len(train_ys)
    # v21 = v + n1
    phi21 = K12.T @ invK11 @ train_ys
    K221 = K22 - K12.T @ invK11 @ K12

    beta1 = train_ys @ (invK11@ train_ys)
    var21 = (v+ beta1 -2)/(v+ n1-2)*K221

    return phi21, var21

def reg_fn(k=1):
    N = 100
    # xs = np.random.uniform(0,10, size=100)
    xs = np.linspace(start=0, stop=10, endpoint=False, num=100)
    ids = np.linspace(start=0, stop=N-1, endpoint=False, num=N, dtype=int)
    train_ids = np.random.choice(ids, size=10, replace=False)
    
    fs, ys = Multivariate_ST_1d(xs,k=k)
    mean, var = regression(xs[train_ids], ys[train_ids], xs, k=k)
    std = np.sqrt(np.diagonal(var))

    plt.figure()
    plt.fill_between(xs, y1=mean+3*std, y2=mean -3*std, color='pink', alpha =0.5, label='3 std region')
    plt.plot(xs, fs ,label = 'A sample Fn')
    plt.plot(xs, mean,label ='Pred Mean')
    plt.scatter(xs[train_ids], ys[train_ids], label= 'Train', color = 'red')
    plt.legend()
    plt.savefig('figure/syn_data/st_reg_k01.png')
    plt.show()

def HMC_jx(x, y, p,k, num_leapfrog_steps, step_size, num_steps, seed):

    def leapfrog_step(grad, step_size, i, leapfrog_state):
        theta, m, tlp_grad = leapfrog_state

        step_size = step_size * 0.99** i
        
        m += 0.5 * step_size * tlp_grad
        theta += step_size *m
        tlp_grad = grad(x,y, theta, p, k)
        m += 0.5* step_size * tlp_grad
        # jax.debug.print('log prob: {} theta: {}', logprob_jx(x,y, theta, l, p, omega), theta)
        return theta, m, tlp_grad
    
    def hmc_step(grad, num_lf_steps, step_size, theta, seed):
        m_seed, mh_seed = jax.random.split(seed)
        tlp_grad = grad(x,y, theta, p,k) 
        tlp = logprob_st_jx(x,y, theta, p,k)
        m = jax.random.normal(m_seed, theta.shape)
        energy = 0.5* jnp.square(m).sum() - tlp

        # jax.debug.print('log prob: {} theta: {}', tlp, theta)

        new_theta, new_m = theta, m
        new_theta, new_m, _ = jax.lax.fori_loop(
            0,
            num_lf_steps,
            partial(leapfrog_step, grad, step_size),
            (theta, m, tlp_grad)
        )
        logp = logprob_st_jx(x,y, new_theta, p,k)
        new_energy = 0.5* jnp.square(new_m).sum() - logp
        # jax.debug.print('logp:{} theta:{}',logp, new_theta)
        log_accept_ratio = energy - new_energy
        is_accepted = jnp.log(jax.random.uniform(mh_seed,[])) < log_accept_ratio
        # select proposed state if accepted
        theta = jnp.where(is_accepted, new_theta, theta)
        hmc_op = {'theta': theta,
                  'is_accepted': is_accepted,
                  'log_acc_ratio':log_accept_ratio,
                  'log_like':logprob_st_jx(x, y, theta, p,k)}
        return theta, hmc_op
    
    # initialize
    # theta = 1.2
    theta = jnp.array([0.5,1.2,1.2]) # sample from some initial dist?
    # theta = jnp.array([0.4801838 , 0.9779928 , 0.809532 ,  0.26198778])
    # theta = jnp.array([1.0, 1.0, 1.0, jnp.sqrt(0.1)]) # logp: -139
    # create a seed for each step
    # seed = jax.random.PRNGKey(seed)
    seeds = jax.random.split(seed, num_steps)
    # repeatedly run hmc and accumulate the outputs
    _, hmc_output = jax.lax.scan(partial(hmc_step, grad_st_jx,num_leapfrog_steps, step_size), theta, seeds)
    return hmc_output

def HMC_fn_jx(xs, ys, k=0.1, save = True, simulation = False, seed = jax.random.PRNGKey(1400), eps0 = 2.5e-4,
              num_leapfrog_steps= 50, num_steps= 300, burn_in = 50):

    if simulation:
        N = 100
        
        xs = np.random.uniform(0,10, size=100)
        xs = np.linspace(start=0, stop=100, endpoint=False, num=100) # if number too large: cause overflow when calculating bessel fucntion with a large index; too small, overfit..?
        ids = np.linspace(start=0, stop=N-1, endpoint=False, num=N, dtype=int)
        # train_ids = np.random.choice(ids, size=10, replace=False)
        
        fs, ys = Multivariate_ST_1d(xs, k=k)
        jnp.savez('xy_k01_st.npz', x = jnp.array(xs), y = jnp.array(ys))

        file = jnp.load('xy_k01_st.npz')
        xs = jnp.array(file['x'])
        ys = jnp.array(file['y'])

    
    hmc_output = HMC_jx(xs, ys, p=len(xs),k=k, num_leapfrog_steps=num_leapfrog_steps, step_size=eps0, num_steps=num_steps, seed=seed)

    plt.figure()
    plt.subplot(2,2,1)
    vs = (np.array(hmc_output['theta'])[burn_in:, 0])**2 + 2
    v0s = (np.array(hmc_output['theta'])[burn_in:, 1])**2
    ws = (np.array(hmc_output['theta'])[burn_in:, 2])**2
    plt.hist(vs)
    plt.xlabel(r'$v$')
    plt.subplot(2,2,2)
    plt.hist(v0s)
    plt.xlabel(r'$v_0$')
    plt.subplot(2,2,3)
    plt.hist(ws)
    plt.xlabel(r'$w$')
    print( vs.mean(), v0s.mean(), ws.mean() )
    # plt.subplot(2,2,4)
    # plt.hist((np.array(hmc_output['theta'])[burn_in:, 3])**2, bins = 20)
    # plt.xlabel(r'$k$')
    
    if save:
        plt.savefig(f'figure\\parafitting_st\\acc_beta_v0_w__seed{seed}_eps1e_3_lf{num_leapfrog_steps}_st{num_steps}.png')

    plt.figure()
    plt.hist(np.exp(np.minimum(np.array(hmc_output['log_acc_ratio']), 0)))
    plt.xlabel('accept ratio')
    if save:
        plt.savefig(f'figure\\parafitting_st\\beta_v0_w__seed{seed}_eps1e_3_lf{num_leapfrog_steps}_st{num_steps}.png')

    plt.figure()
    plt.plot(np.array(hmc_output['log_like']))
    if save:
        plt.savefig(f'figure\\parafitting_st\\loglike_seed{seed}_eps1e_3_lf{num_leapfrog_steps}_st{num_steps}.png')
    plt.show()

    return vs.mean(), v0s.mean(), ws.mean()

def grid_search(k=0.1):
    N = 100
    
    xs = np.random.uniform(0,10, size=100)
    xs = np.linspace(start=0, stop=100, endpoint=False, num=100) # if number too large: cause overflow when calculating bessel fucntion with a large index; too small, overfit..?
    ids = np.linspace(start=0, stop=N-1, endpoint=False, num=N, dtype=int)
    # train_ids = np.random.choice(ids, size=10, replace=False)
    
    fs, ys = Multivariate_ST_1d(xs, k=k)
    jnp.savez('xy_k01_st.npz', x = jnp.array(xs), y = jnp.array(ys))

    file = jnp.load('xy_k01_st.npz')
    xs = jnp.array(file['x'])
    ys = jnp.array(file['y'])

    vs = np.linspace(start=0, stop = 2, endpoint= False, num = 10)
    for v in vs:
        logprob_st_jx(xs, ys, []) # theta = [v_sq, v_0_sq, wl_Sq, k_sq]

def inf_finance(k=0.01, seed = 1400):
    csvfile = r"C:\Users\95414\Desktop\CUED\phd\year1\mycode\data\data\NVIDIA CORPORATION (01-23-2025 09.30 _ 01-29-2025 16.00).csv"
    nvda_data_reloaded= pd.read_csv(csvfile, index_col=0, parse_dates=True)
    x_ns = jnp.array(nvda_data_reloaded['Price']['2025-01-29'], dtype=jnp.float32)
    x_ns = (x_ns - x_ns[0])
    ys = jnp.flip(x_ns)[:300] # pick the prev 300 data for const mu
    N=len(ys)
    delta_ts = 1
    xs = jnp.linspace(0, N,endpoint=False,num=N)
    
    ids = jnp.linspace(start=0, stop=N, endpoint=False, num=N, dtype=int)
    seed = jax.random.PRNGKey(seed)
    rc_seed, seed = jax.random.split(seed)
    train_ids = jax.random.choice(rc_seed, ids, shape=(1,200), replace=False).flatten()

    # v, v0, w = HMC_fn_jx(xs = xs[train_ids], ys = ys[train_ids], k=k,save=False, seed= seed)

    v,v0, w = 50, 0.1, 0.1

    mean, var = regression(xs[train_ids], ys[train_ids], xs, v=v, v_0=v0, w=w,  k=k)
    std = np.sqrt(np.diagonal(var))

    plt.figure()
    
    plt.plot(xs, mean,label ='Pred Mean',zorder=1)
    plt.plot(xs, ys ,label = 'True', zorder=2)
    
    plt.fill_between(xs, y1=mean+3*std, y2=mean -3*std, color='pink', alpha =0.5, label='3 std region')
    plt.scatter(xs[train_ids], ys[train_ids], label= 'Train', color = 'red', s=4, zorder=10)
    
    plt.legend()
    plt.savefig('figure/nvidia/st_reg_k001.png')
    plt.show()

    



if __name__ == '__main__':
    # reg_fn(k=1e-1)
    # HMC_fn_jx(save=True)
    inf_finance()
