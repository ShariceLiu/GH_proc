from scipy import special, stats
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

import jax.numpy as jnp
import jax

from tools.tools import *
from functools import partial
from matplotlib import cm


def Multivariate_GH_2d(xs, l = -1, omega = 1, w1 = 1, w2 = 1,  v_0 = 1, a_b = 1.0, B_b = np.array((0.0,0.0)), k = 1):
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
    fs = np.random.multivariate_normal(beta*W, (K)*W)
    ys = fs + np.random.multivariate_normal(beta*0, (k*np.identity(N))*W)

    return fs, ys

def regression(train_xs, train_ys, test_xs, l = -1, omega = 1, w1 = 1,w2 =1,  v_0 = 1, a_b = 1, b_b = np.array([0.0,0.0]), k =1):
    K11 = SE_cov_2d(train_xs, train_xs, v_0, w1, w2) + k*np.identity(len(train_ys))
    K12 = SE_cov_2d(train_xs, test_xs, v_0, w1, w2)
    K22 = SE_cov_2d(test_xs, test_xs, v_0, w1, w2)

    beta1 = np.ones(len(train_ys))*a_b + b_b @ train_xs
    beta2 = np.ones(test_xs.shape[1])*a_b + b_b @ test_xs

    l21 = l - len(train_ys)/2
    invK11 = np.linalg.inv(K11)
    gamma21 = omega + train_ys @ (invK11 @ train_ys)
    delta21 = omega + beta1 @ (invK11 @ beta1)
    mu21 = K12.T @ invK11 @ train_ys
    K221 = K22 - K12.T @ invK11 @ K12
    beta21 = beta2 - K12.T @ invK11 @ beta1

    omega21 = gamma21 * delta21
    k21 = np.sqrt(delta21/ gamma21)

    # import pdb; pdb.set_trace()
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

if __name__ == '__main__':
    reg_fn(k=0.1)
    # HMC_fn_jx()
    # eps_star, eps_n, alphas = dual_averaging(eps0=1e-4)
    # print(eps_star)

    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.plot(eps_n)
    # plt.subplot(2,1,2)
    # plt.plot(alphas)
    # plt.show()

    