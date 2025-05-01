# simulate some sample paths from GH proc
# exponential covariance function
# zero mean
# constant beta? (what effect would beta have on the data point?)

from scipy import special, stats
import numpy as np
import matplotlib.pyplot as plt

import jax.numpy as jnp
import jax

from tools.tools import *
from functools import partial

def Multivariate_GH_1d(xs, l = -1, omega = 1, w = 1, v_0 = 1, a_b = 1, b_b = 0):
    """
    xs: np array, input location
    l: index parameter for GIG
    omega: a, b in GIG
    SE cov: v_0 exp(-w*(x-y)^2/2)
    beta(x) = a_b + b_b *x
    """
    W = stats.geninvgauss.rvs(p=l, b=omega, size=1)[0]

    N = len(xs)
    # exponential cov function
    K = SE_cov(xs,xs, v_0, w)

    #skew
    beta = np.ones(N)*a_b + xs*b_b

    # assume zero mean
    fs = np.random.multivariate_normal(beta*W, K*W)

    return fs

def regression(train_xs, train_ys, test_xs, l = -1, omega = 1, w = 1, v_0 = 1, a_b = 1, b_b = 0):
    K11 = SE_cov(train_xs, train_xs, v_0, w)
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

    EW = special.kn(l21+1, omega21)/special.kn(l21, omega21)
    VarW = special.kn(l21+2, omega21)/special.kn(l21, omega21)\
            -(special.kn(l21+1,  omega21)/special.kn(l21, omega21))**2
    mean21 = mu21 + EW*beta21
    var21 =  EW* K221 + VarW * np.outer(beta21, beta21)

    return mean21, var21

def reg_fn():
    N = 100
    a_b = 0
    # xs = np.random.uniform(0,10, size=100)
    xs = np.linspace(start=0, stop=10, endpoint=False, num=100)
    ids = np.linspace(start=0, stop=N-1, endpoint=False, num=N, dtype=int)
    train_ids = np.random.choice(ids, size=10, replace=False)
    
    ys = Multivariate_GH_1d(xs)
    mean, var = regression(xs[train_ids], ys[train_ids], xs)
    std = np.sqrt(np.diagonal(var))

    plt.figure()
    plt.plot(xs, ys ,label = 'Fn')
    plt.scatter(xs[train_ids], ys[train_ids], label= 'Train')
    plt.plot(xs, mean,label ='test')
    plt.fill_between(xs, y1=mean+3*std, y2=mean -3*std, color='pink', alpha =0.5)
    plt.legend()
    plt.savefig('figure/syn_data/reg.png')
    plt.show()

def reg():
    N = 100
    a_b = 0
    # xs = np.random.uniform(0,10, size=100)
    xs = np.linspace(start=0, stop=10, endpoint=False, num=100)
    ids = np.linspace(start=0, stop=N-1, endpoint=False, num=N, dtype=int)
    train_ids = np.random.choice(ids, size=10, replace=False)
    
    ys = Multivariate_GH_1d(xs)
    mean, var = regression(xs[train_ids], ys[train_ids], xs)
    std = np.sqrt(np.diagonal(var))

    plt.figure()
    plt.plot(xs, ys ,label = 'Fn')
    plt.scatter(xs[train_ids], ys[train_ids], label= 'Train')
    plt.plot(xs, mean,label ='test')
    plt.fill_between(xs, y1=mean+3*std, y2=mean -3*std, color='pink', alpha =0.5)
    plt.legend()
    plt.savefig('figure/syn_data/reg.png')
    plt.show()


def HMC(x, y, l, p, omega, num_leapfrog_steps, step_size, num_steps, seed):

    def leapfrog_step(grad, step_size, i, leapfrog_state):
        theta, m, tlp_grad = leapfrog_state
        
        m += 0.5 * step_size * tlp_grad
        theta += step_size *m
        tlp_grad = grad(x, y, theta, l, p, omega)
        m += 0.5* step_size * tlp_grad
        return theta, m, tlp_grad
    
    def hmc_step(grad, num_lf_steps, step_size, theta, seed):
        m_seed, mh_seed = jax.random.split(seed)
        tlp_grad = grad(x,y, theta, l, p, omega) 
        import pdb;pdb.set_trace()
        tlp = logprob(x,y, theta, l, p, omega)
        m = jax.random.normal(m_seed, theta.shape)
        energy = 0.5* jnp.square(m).sum() - tlp

        new_theta, new_m = theta, m
        for i in range(num_lf_steps):
            new_theta, new_m, _ = leapfrog_step(grad, step_size, i, (new_theta, new_m, tlp_grad))
        # new_theta, new_m, _ = jax.lax.fori_loop(
        #     0,
        #     num_lf_steps,
        #     partial(leapfrog_step, grad, step_size),
        #     (theta, m, tlp_grad)
        # )
        new_energy = 0.5* jnp.square(new_m).sum() - logprob(x,y, new_theta, l, p, omega)
        log_accept_ratio = energy - new_energy
        is_accepted = jnp.log(jax.random.uniform(mh_seed,[])) < log_accept_ratio
        # select proposed state if accepted
        theta = jnp.where(is_accepted, new_theta, theta)
        hmc_op = {'theta': theta,
                  'is_accepted': is_accepted,
                  'log_acc_ratio':log_accept_ratio}
        return theta, hmc_op
    
    # initialize
    theta = jnp.array([0.0,1.0,1.0]) # sample from some initial dist?
    # create a seed for each step
    seed = jax.random.PRNGKey(seed)
    seeds = jax.random.split(seed, num_steps)
    # repeatedly run hmc and accumulate the outputs
    for i in range(1):
        hmc_output = hmc_step(grad, num_leapfrog_steps, step_size, theta, seed)
    # _, hmc_output = jax.lax.scan(partial(hmc_step, grad,num_leapfrog_steps, step_size), theta, seeds)
    return hmc_output

if __name__ == '__main__':
    N = 100
    a_b = 0
    # xs = np.random.uniform(0,10, size=100)
    xs = np.linspace(start=0, stop=10, endpoint=False, num=100)
    ids = np.linspace(start=0, stop=N-1, endpoint=False, num=N, dtype=int)
    train_ids = np.random.choice(ids, size=10, replace=False)
    
    ys = Multivariate_GH_1d(xs)

    hmc_output = HMC(xs, ys, l=1, p=1, omega=1, num_leapfrog_steps=100, step_size=0.01, num_steps=50, seed=1700)
