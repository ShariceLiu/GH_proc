from scipy import special, stats
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import jax.numpy as jnp
import jax

from tools.tools import *
from functools import partial
from matplotlib import cm

import pandas as pd

def read_wine(N=400, random =False):
    winepath = "C:/Users/95414/Desktop/CUED/phd/year1/mycode/data/winequality/winequality-red.csv"
    winedata = pd.read_csv(winepath, sep=';').to_numpy()

    if random:
        random_idx = np.random.choice(N, size=N, replace=False)
        xs = winedata[random_idx, :-1]
        ys = winedata[random_idx, -1]
    else:
        xs = winedata[:N, :-1]
        ys = winedata[:N, -1]

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
        import pdb;pdb.set_trace()
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

def HMC_jx(x, y, l, p, omega, k, num_leapfrog_steps, step_size, num_steps, seed, theta0):

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
        tlp = multid_logprob_jx(x,y, theta, l, p, omega, k)
        m = jax.random.normal(m_seed, theta.shape)
        energy = 0.5* jnp.square(m).sum() - tlp

        jax.debug.print('log prob: {} theta: {}', tlp, theta[jnp.array([0,1,2],int)])

        new_theta, new_m = theta, m
        new_theta, new_m, _ = jax.lax.fori_loop(
            0,
            num_lf_steps,
            partial(leapfrog_step, grad, step_size),
            (theta, m, tlp_grad)
        )
        new_energy = 0.5* jnp.square(new_m).sum() - multid_logprob_jx(x,y, new_theta, l, p, omega, k)
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


def HMC_fn_jx(xs, ys, k=0.1, save = True, seed = jax.random.PRNGKey(1400), eps0 = 1e-3,num_leapfrog_steps=100,
    num_steps=1100, burn_in = 100, load = False):
    xs, ys = read_wine(N=360)
    
    if load:
        theta = jnp.array(jnp.load('data/theta.npz')['theta'])
    else:
        theta = jnp.ones(xs.shape[0]+3)
        theta.at[0].set(0.0)

    hmc_output = HMC_jx(xs, ys, l=-1, p=xs.shape[1], omega=1, k=k, num_leapfrog_steps=num_leapfrog_steps, step_size=eps0, num_steps=num_steps, seed=seed, theta0=theta)

    plt.figure()
    plt.subplot(3,2,1)
    plt.plot(np.array(hmc_output['theta'])[burn_in:, 0])
    plt.xlabel(r'$\beta$')
    plt.subplot(3,2,2)
    plt.plot((np.array(hmc_output['theta'])[burn_in:, 1])**2)
    plt.xlabel(r'$v_0$')
    plt.subplot(3,2,3)
    plt.plot((np.array(hmc_output['theta'])[burn_in:, 2])**2)
    plt.xlabel(r'$k$')
    plt.subplot(3,2,4)
    plt.plot((np.array(hmc_output['theta'])[burn_in:, 3])**2)
    plt.xlabel(r'$w_1$')
    plt.subplot(3,2,5)
    plt.plot((np.array(hmc_output['theta'])[burn_in:, 4])**2)
    plt.xlabel(r'$w_2$')
    plt.subplot(3,2,6)
    plt.plot((np.array(hmc_output['theta'])[burn_in:, 5])**2)
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
    lags = np.arange(int(num_steps))
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

def wine_reg(save=False, model = 'gh', return_mse = True, random = True, beta = 0.0, mu = 0.0, theta=np.ones(14) ):
    xs, ys = read_wine(N=400, random=random)
    train_xs, train_ys = xs[:,40:], ys[40:]
    test_xs, test_ys = xs[:,-40:], ys[-40:]

    # theta = np.array(np.load('data/theta.npz')['theta'])
    theta[0] = beta
    if model == 'gh':
        mean, var = regression(train_xs, train_ys, test_xs, a_b=theta[0], v_0=theta[1], k=0.1, ws=theta[3:], mu=mu)
    else:
        mean, var = st_regression(train_xs, train_ys, test_xs, v_0=theta[1], k=0.1, ws=theta[3:], phi=mu)
    
    idx = np.where((mean>=0) & (mean<=10))

    mse = np.dot(mean-test_ys, mean-test_ys)/40
    if return_mse:
        return mse
    print(np.dot(mean-test_ys, mean-test_ys)/40)

    plt.scatter(train_xs[0,:], train_ys, label='train')
    plt.scatter(test_xs[0,:], test_ys, label='test')
    plt.scatter(test_xs[0,idx], mean[idx], label='pred')
    plt.legend()
    if save:
        plt.savefig(f'figure/wine/reg/{model}_dim0.png')
    plt.show()

def likelihood_grid(N=500, random = False):
    # generate data
    k=0.1

    xs, ys = read_wine(N=360, random=random)
    # train_xs, train_ys = xs[:,40:], ys[40:]
    # test_xs, test_ys = xs[:,-40:], ys[-40:]
    
    # grid
    n = 5
    
    a_bs = np.linspace(0,2,num=n)
    v_0_sqs = np.linspace(0.5,2,num=n)
    wl_sqs = 10**np.linspace(-5,0,num=n)
    
    ks = np.linspace(0.01, 0.2, num=n)
    mus = np.linspace(-2,0,num=n)
    l_sqs, omega_sqs = -np.linspace(0.1,10,num=n), np.linspace(0.1,10,num=n), 

    list_paras = [r'$a_b$',r'$v_0$' ,r'k',r'$w_1$',r'$w_2$',r'$w_3$',r'$mu$', r'$l$',  r'$omega$']
    list_ns = [a_bs, v_0_sqs, ks, wl_sqs,wl_sqs,wl_sqs,  mus, l_sqs, omega_sqs]

    mls = [3,4]

    for m in mls:
        for l in mls:
            if l<=m:
                continue
            likelihoods = np.zeros((n,n))
            mses = np.zeros((n,n))
            X, Y =np.meshgrid(list_ns[m], list_ns[l])
        
            for i, (row_x , row_y) in tqdm(enumerate(zip(X, Y))):
                for j, (x , y) in enumerate(zip(row_x, row_y)):
                    theta = np.ones(14)
                    theta[3:]*= 1e-4
                    theta[0] = 0.0
                    theta[m] = x
                    theta[l] = y
                    likelihoods[i,j] = (multid_logprob(xs, ys, theta, l=-1, p=xs.shape[1], omega=1,k=0.1))
                    mses[i,j] = wine_reg(random=False, theta=theta)


            # import pdb; pdb.set_trace()
            idx = np.argmax(likelihoods)
            idx = np.unravel_index(idx, likelihoods.shape)
            print(X[idx], Y[idx])
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            surf = ax.plot_surface(np.log10(X), np.log(Y), likelihoods, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
            ax.set_xlabel(list_paras[m])
            ax.set_ylabel(list_paras[l])
            plt.savefig(f'figure/wine/like_plot/{list_paras[m]}{list_paras[l]}.png')

            # import pdb; pdb.set_trace()
            idx = np.argmin(mses)
            idx = np.unravel_index(idx, mses.shape)
            print(X[idx], Y[idx])
            fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
            surf = ax.plot_surface(np.log10(X), np.log10(Y), mses, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
            ax.set_xlabel(list_paras[m])
            ax.set_ylabel(list_paras[l])
            plt.savefig(f'figure/wine/like_plot/{list_paras[m]}{list_paras[l]}_mse.png')
    plt.show()

def mse_grid():
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
                mse_ghn[k] = wine_reg(model='gh', beta=betas[j], mu = mus[i], random=True)

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
            mse_stn[k] = wine_reg(model='st', mu = mus[i], random=True)

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

if __name__ == '__main__':
    # HMC_fn_jx(0,0,save=True,eps0=5e-4, burn_in=500, num_steps=3000, load=True)
    likelihood_grid()