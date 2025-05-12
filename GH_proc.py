# simulate some sample paths from GH proc
# exponential covariance function
# zero mean
# constant beta? (what effect would beta have on the data point?)

from scipy import special, stats
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import jax.numpy as jnp
import jax

from tools.tools import *
from functools import partial

def Multivariate_GH_1d(xs, l = -1, omega = 1, w = 1, v_0 = 1, a_b = 1, b_b = 0, k = 1):
    """
    xs: np array, input location
    l: index parameter for GIG
    omega: a, b in GIG
    SE cov: v_0 exp(-w*(x-y)^2/2)
    beta(x) = a_b + b_b *x
    k: noise variance
    """
    W = stats.geninvgauss.rvs(p=l, b=omega, size=1)[0]

    N = len(xs)
    # exponential cov function
    K = SE_cov(xs,xs, v_0, w)

    #skew
    beta = np.ones(N)*a_b + xs*b_b

    # assume zero mean
    fs = np.random.multivariate_normal(beta*W, (K)*W)
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

    EW = special.kn(l21+1, omega21)/special.kn(l21, omega21)
    VarW = special.kn(l21+2, omega21)/special.kn(l21, omega21)\
            -(special.kn(l21+1,  omega21)/special.kn(l21, omega21))**2
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

        # jax.debug.print('log prob: {} theta: {}', tlp, theta)

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
    theta = jnp.array([0.5,1.5,1.5]) # sample from some initial dist?
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

def dual_averaging(eps0, target=0.65, M_adapt=100,
                   gamma=0.01, t0=200.0, kappa=0.9):
    file = jnp.load('xy.npz')
    xs = jnp.array(file['x'])
    ys = jnp.array(file['y'])

    

    mu = jnp.log(10 * eps0)
    H_bar = 0.0
    log_eps_bar = 0.0
    eps = eps0

    eps_n = [jnp.log(eps)]
    alphas = []

    key = jax.random.PRNGKey(1700)
    for m in tqdm(range(1, M_adapt+1)):
        # 1) run one HMC step with current eps, get acceptance α
        key, subkey = jax.random.split(key)
        # import pdb;pdb.set_trace()
        hmc_output = HMC_jx(xs, ys, l=1, p=1, omega=1, num_leapfrog_steps=50, step_size=eps, num_steps=1, seed=subkey)
        alpha = min(jnp.exp(hmc_output['log_acc_ratio'][0]),1)
        alphas.append(alpha)

        # 2) update H̄
        H_bar = (1 - 1/(m + t0)) * H_bar + (1/(m + t0)) * (target - alpha)

        # 3) propose new log-step
        log_eps = mu - (jnp.sqrt(m)/gamma) * H_bar

        # 4) exponentiate
        eps = jnp.exp(log_eps)

        # 5) update averaged log-step
        log_eps_bar = (m**(-kappa)) * log_eps + (1 - m**(-kappa)) * log_eps_bar

        eps_n.append(jnp.log(eps))

    # At end of warmup:
    eps_star = jnp.exp(log_eps_bar)
    return eps_star, eps_n, alphas

def HMC_fn_jx():
    N = 100
    a_b = 0
    k=0.1
    # xs = np.random.uniform(0,10, size=100)
    # xs = np.linspace(start=0, stop=100, endpoint=False, num=100) # if number too large: cause overflow when calculating bessel fucntion with a large index; too small, overfit..?
    # ids = np.linspace(start=0, stop=N-1, endpoint=False, num=N, dtype=int)
    # # train_ids = np.random.choice(ids, size=10, replace=False)
    
    # fs, ys = Multivariate_GH_1d(xs, omega=1,k=k)
    # jnp.savez('xy_k01.npz', x = jnp.array(xs), y = jnp.array(ys))

    file = jnp.load('xy_k01.npz')
    xs = jnp.array(file['x'])
    ys = jnp.array(file['y'])

    eps0 = 1e-3
    seed = 1400
    num_leapfrog_steps=100
    num_steps=1100
    
    hmc_output = HMC_jx(xs, ys, l=-1, p=len(xs), omega=1, k=k, num_leapfrog_steps=num_leapfrog_steps, step_size=eps0, num_steps=num_steps, seed=jax.random.PRNGKey(seed))

    plt.figure()
    plt.subplot(2,2,1)
    plt.hist(np.array(hmc_output['theta'])[100:, 0])
    plt.xlabel(r'$\beta$')
    plt.subplot(2,2,2)
    plt.hist((np.array(hmc_output['theta'])[100:, 1])**2)
    plt.xlabel(r'$v_0$')
    plt.subplot(2,2,3)
    plt.hist((np.array(hmc_output['theta'])[100:, 2])**2)
    plt.xlabel(r'$w$')
    plt.subplot(2,2,4)
    plt.hist(np.exp(np.minimum(np.array(hmc_output['log_acc_ratio']), 0)))
    plt.xlabel('accept ratio')
    plt.savefig(f'figure\\para_fitting\\beta_v0_w__seed{seed}_eps1e_3_lf{num_leapfrog_steps}_st{num_steps}.png')
    plt.show()


    # print(hmc_output['theta'], hmc_output['is_accepted'], np.exp(np.array(hmc_output['log_acc_ratio'])))


if __name__ == '__main__':
    # reg_fn(k=1)
    HMC_fn_jx()
    # eps_star, eps_n, alphas = dual_averaging(eps0=1e-4)
    # print(eps_star)

    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.plot(eps_n)
    # plt.subplot(2,1,2)
    # plt.plot(alphas)
    # plt.show()


