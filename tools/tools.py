import numpy as np
import jax.numpy as jnp
from scipy import special
import jax

from jax import custom_vjp
import jax.numpy as jnp
import scipy.special

import jax
import jax.numpy as jnp
import scipy.special
from jax.experimental import io_callback

def kn_io(v, x):
    return io_callback(
        lambda v, x: scipy.special.kn(v, x),
        jax.ShapeDtypeStruct(shape=(), dtype=jnp.result_type(v, x)),
        v, x
    )

def log_kn_large_order(v,x):
    """ assume order is fixed, only return terms with x involved """
    w  = x/jnp.abs(v)
    eta = jnp.sqrt(1+w**2)+ jnp.log(w/(1+jnp.sqrt(1+w**2)))
    return - jnp.log(1+w**2)/4 - jnp.abs(v)*eta

def d_kn_large_order(v,x):
    """ derivative """
    w  = x/jnp.abs(v)
    eta = jnp.sqrt(1+w**2)+ jnp.log(w/(1+jnp.sqrt(1+w**2)))
    return - jnp.sqrt(jnp.pi/2/v)*(1+w**2)**(1/4)*jnp.exp(-jnp.abs(v)*eta)/w

def SE_cov(xs, ys, v_0, w):
    N = len(xs)
    M = len(ys)
    K = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            K[i,j] = v_0 * np.exp(-w*(xs[i]-ys[j])**2/2)

    if N==M:
        return K  + 1e-7*np.identity(N)
    else:
        return K

def SE_cov_jnp(xs, ys, v_0, w):
    def cov(i,j):
        return v_0 * jnp.exp(-w*(xs[jnp.array(i,int)]-ys[jnp.array(j,int)])**2/2)

    N = len(xs)
    M = len(ys)
    K = jnp.fromfunction(cov, (N,M))
    
    return K

def dK_dw(xs, ys, v_0, w):
    N = len(xs)
    M = len(ys)
    K = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            K[i,j] = - v_0 *(xs[i]-ys[j])**2/2 *  np.exp(-w*(xs[i]-ys[j])**2/2)

    return K

def dK_dw_jx(xs, ys, v_0, w):
    def cov(i,j):
        return - v_0 *(xs[jnp.array(i,int)]-ys[jnp.array(j,int)])**2/2* jnp.exp(-w*(xs[jnp.array(i,int)]-ys[jnp.array(j,int)])**2/2)

    N = len(xs)
    M = len(ys)
    K = jnp.fromfunction(cov, (N,M))
    return K

def grad(x, f, theta,  l: float, p: int, omega: float):
    """
    beta = a_b (+ b_b x)
    K(x,y) = v_0 exp(- w* (x-y)^2/2)

    so the parameters are
    theta = [a_b, b_b, v_0, w]
    """
    a_b, v_0, w = theta

    K=SE_cov(x, x, v_0, w)
    beta = np.ones(len(x))*a_b
    mu = np.zeros_like(f)
    

    invK = np.linalg.inv(K)
    bKb = beta @ invK @ beta
    deltaf = (f-mu) @ invK @ (f - mu)
    bKb_df = np.sqrt((omega + deltaf)*(omega + bKb))

    ep = 1e-5
    dK = (special.kn((l-p)/2, bKb_df + ep) - special.kn((l-p)/2, bKb_df - ep))/(2*ep)
    # dK = jax.grad(lambda x: special.kn((l-p)/2, x))(bKb_df)
    kbbk = invK @ np.outer(beta, beta) @ invK

    gradbeta = (l - p/2)/(omega + bKb) * (invK @ beta) \
        + dK/special.kn((l-p)/2, bKb_df)\
         * np.sqrt(deltaf/ (omega + bKb)) *( invK @ beta) \
        + invK @ (f - mu)
    grada_b = gradbeta @ np.ones(len(beta))

    gradK = (l - p/2)/2 * ( (invK @ np.outer(f-mu, f-mu) @ invK )/(omega + deltaf) + (kbbk/(omega + beta@invK @ beta)) )\
        + dK/special.kn((l-p)/2, bKb_df) * ( np.sqrt(omega + deltaf)* kbbk /(np.sqrt(omega + bKb)) +\
                np.sqrt((omega + bKb)/(omega + deltaf))* (invK @ np.outer(f-mu, f-mu)@invK) )/2 \
        - invK/2 - invK @ np.outer((f-mu), beta) @ invK
    gradv_0 = np.trace(gradK @ K/v_0)
    gradw = np.trace(gradK @ dK_dw(x, x, v_0, w))
    
    return np.array([grada_b, gradv_0, gradw])

def logprob(x, f, theta,  l: float, p: int, omega: float):
    a_b, v_0, w = theta

    K=SE_cov(x,x, v_0, w)
    beta = np.ones(len(x))*a_b
    mu = np.zeros_like(f)

    # import pdb; pdb.set_trace()
    invK = np.linalg.inv(K)
    deltaf = (f-mu) @ invK @ (f - mu)
    bKb = beta @ invK @ beta
    logp = (l - p/2)/2 * ( np.log(omega + deltaf) - np.log(omega + bKb) )\
        + np.log(special.kn( (l-p)/2,  np.sqrt((omega + deltaf)*(omega + bKb))))\
        - (f-mu) @ invK @beta #- p/2* jnp.log(2*jnp.pi) - jnp.log(special.kn(l, omega)) - np.log(np.linalg.det(K)) (the last det is always zero)
    return logp

def grad_jx(x, f, theta,  l: float, p: int, omega: float, k:float):
    """
    beta = a_b (+ b_b x)
    K(x,y) = v_0^2 exp(- w* (x-y)^2/2)

    so the parameters are
    theta = [a_b, b_b, v_0, w]

    k is the observation noise
    """
    a_b, v_0_sq, wl_Sq = theta
    # a_b = theta
    # jax.debug.print('a_b: {}', a_b)
    # v_0_sq = 1

    wl = wl_Sq**2
    v_0 = v_0_sq**2
    K=SE_cov_jnp(x, x, v_0, w=wl) + jnp.identity(len(x))*k
    beta = jnp.ones(len(x))*a_b
    mu = jnp.zeros_like(f)
    

    invK = jnp.linalg.inv(K)
    bKb = beta @ invK @ beta
    deltaf = (f-mu) @ invK @ (f - mu)
    bKb_df = jnp.sqrt((omega + deltaf)*(omega + bKb))

    # dK = - (kn_io(l-p/2-1, bKb_df)+kn_io(l-p/2+1, bKb_df))/2
    # jax.debug.print('dK:{}',dK)
    kbbk = invK @ jnp.outer(beta, beta) @ invK

    w = bKb_df/ jnp.abs((l-p/2))

    gradbeta = -(l - p/2)/(omega + bKb) * (invK @ beta) \
        + (-jnp.sqrt(1+w**2)/w)\
         * jnp.sqrt((deltaf+omega)/ (omega + bKb)) *( invK @ beta) \
        + invK @ (f - mu) # + dK/kn_io(l-p/2, bKb_df)\ use high order approx
    grada_b = gradbeta @ jnp.ones(len(beta))

    # return grada_b

    gradK = (l - p/2)/2 * ( -(invK @ jnp.outer(f-mu, f-mu) @ invK )/(omega + deltaf) + (kbbk/(omega + beta@invK @ beta)) )\
        + (-jnp.sqrt(1+w**2)/w) * ( -jnp.sqrt(omega + deltaf)* kbbk /(jnp.sqrt(omega + bKb)) -\
                jnp.sqrt((omega + bKb)/(omega + deltaf))* (invK @ jnp.outer(f-mu, f-mu)@invK) )/2 \
        - invK/2 - invK @ jnp.outer( f-mu,beta) @ invK
    gradv_0 = jnp.trace(2* gradK @ ((K-jnp.identity(len(x))*k)/v_0_sq) )  - v_0_sq
    gradw = jnp.trace(gradK @ dK_dw_jx(x, x, v_0, w=wl)*2*wl_Sq) - wl_Sq
    # jax.debug.print('theta:{}, grada {}, gradv: {}',theta, grada_b, gradv_0)
    
    return jnp.array([grada_b, gradv_0, gradw])

def logprob_jx(x, f, theta,  l: float, p: int, omega: float, k:float):
    # a_b, v_0_sq = theta
    a_b, v_0_sq, wl_sq = theta

    v_0 = v_0_sq**2
    wl = wl_sq**2

    K=SE_cov_jnp(x,x, v_0, w=wl)+ jnp.identity(len(x))*k
    beta = jnp.ones(len(x))*a_b
    mu = jnp.zeros_like(f)

    invK = jnp.linalg.inv(K)
    deltaf = (f-mu) @ invK @ (f - mu)
    bKb = beta @ invK @ beta
    logp = (l - p/2)/2 * ( jnp.log(omega + deltaf) - jnp.log(omega + bKb) )\
        + log_kn_large_order(l-p/2,  jnp.sqrt((omega + deltaf)*(omega + bKb))) + (f-mu) @ invK @beta  - v_0/2 \
        - jnp.log(jnp.linalg.det(K/v_0))/2 - p*jnp.log(v_0)/2 - wl/2 #- p/2* jnp.log(2*jnp.pi) - jnp.log(special.kn(l, omega))  - jnp.log(jnp.linalg.det(K + jnp.identity(len(x))*0.4))\
        # - len(x)* (v_0_sq-1)**2/2 # add a prior to prevent overfitting
        # + jnp.log(kn_io( l-p/2,  jnp.sqrt((omega + deltaf)*(omega + bKb))))\ using an approximation cos order too large
    
    jax.debug.print('theta:{}, detK {}, finKb: {}',theta, jnp.log(jnp.linalg.det(K/v_0)), log_kn_large_order( l-p/2,  jnp.sqrt((omega + deltaf)*(omega + bKb))))
    
    return logp