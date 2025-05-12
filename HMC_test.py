
from scipy import special, stats
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import jax.numpy as jnp
import jax

from functools import partial

def logprob_G(x):
    return -jnp.log(2*jnp.pi)/2 - x**2/2

def grad_G(x):
    return -x

def HMC_jx(num_leapfrog_steps, step_size, num_steps, seed):

    def leapfrog_step(grad, step_size, i, leapfrog_state):
        theta, m, tlp_grad = leapfrog_state

        # step_size = step_size * 0.95** i
        
        m += 0.5 * step_size * tlp_grad
        theta += step_size *m
        tlp_grad = grad(theta)
        m += 0.5* step_size * tlp_grad
        # jax.debug.print('log prob: {} theta: {}', logprob_jx(x,y, theta, l, p, omega), theta)
        return theta, m, tlp_grad
    
    def hmc_step(grad, num_lf_steps, step_size, theta, seed):
        m_seed, mh_seed = jax.random.split(seed)
        tlp_grad = grad(theta) 
        tlp = logprob_G(theta)
        m = jax.random.normal(m_seed)
        energy = 0.5* jnp.square(m).sum() - tlp

        # jax.debug.print('log prob: {} theta: {}', tlp, theta)

        new_theta, new_m = theta, m
        new_theta, new_m, _ = jax.lax.fori_loop(
            0,
            num_lf_steps,
            partial(leapfrog_step, grad, step_size),
            (theta, m, tlp_grad)
        )
        new_energy = 0.5* jnp.square(new_m).sum() - logprob_G(new_theta)
        log_accept_ratio = energy - new_energy
        is_accepted = jnp.log(jax.random.uniform(mh_seed,[])) < log_accept_ratio
        # select proposed state if accepted
        theta = jnp.where(is_accepted, new_theta, theta)
        hmc_op = {'theta': theta,
                  'is_accepted': is_accepted,
                  'log_acc_ratio':log_accept_ratio}
        return theta, hmc_op
    
    # initialize
    theta = 0.0 # sample from some initial dist?
    # create a seed for each step
    seeds = jax.random.split(seed, num_steps)
    # repeatedly run hmc and accumulate the outputs
    _, hmc_output = jax.lax.scan(partial(hmc_step, grad_G,num_leapfrog_steps, step_size), theta, seeds)
    return hmc_output

def HMC_fn_jx():
    N = 1000

    eps0 = 1e-2
    
    hmc_output = HMC_jx(num_leapfrog_steps=30, step_size=eps0, num_steps=N, seed=jax.random.PRNGKey(1600))
    # print(hmc_output['theta'], hmc_output['is_accepted'], np.exp(np.array(hmc_output['log_acc_ratio'])))
    plt.figure()
    plt.subplot(2,1,1)
    plt.hist(hmc_output['theta'])
    plt.xlabel('sample')
    plt.subplot(2,1,2)
    plt.hist(np.exp(np.array(hmc_output['log_acc_ratio'])))
    plt.xlabel('accept ratio')
    plt.show()

if __name__ == '__main__':
    HMC_fn_jx()