import numpy as np
import jax.numpy as jnp
from scipy import special
import jax

from jax import custom_vjp, lax
import jax.numpy as jnp
import scipy.special

import jax
import jax.numpy as jnp
import scipy.special
from jax.experimental import io_callback

def kn_io(v, x):
    return io_callback(
        lambda v, x: scipy.special.kv(v, x),
        jax.ShapeDtypeStruct(shape=(), dtype=jnp.result_type(v, x)),
        v, x
    )

def kvp_io(v, x):
    return io_callback(
        lambda v, x: scipy.special.kvp(v, x),
        jax.ShapeDtypeStruct(shape=(), dtype=jnp.result_type(v, x)),
        v, x
    )

def log_kn_large_order(v,x):
    """ assume order is fixed, x grows with order, only return terms with x involved """
    w  = x/jnp.abs(v)
    eta = jnp.sqrt(1+w**2)+ jnp.log(w/(1+jnp.sqrt(1+w**2)))
    return - jnp.log(1+w**2)/4 - jnp.abs(v)*eta

def log_kn_large_order_np(v,x):
    """ assume order is fixed, x grows with order, only return terms with x involved """
    w  = x/np.abs(v)
    eta = np.sqrt(1+w**2)+ np.log(w/(1+np.sqrt(1+w**2)))
    return - np.log(1+w**2)/4 - np.abs(v)*eta

def d_kn_large_order(v,x):
    """ derivative """
    w  = x/jnp.abs(v)
    eta = jnp.sqrt(1+w**2)+ jnp.log(w/(1+jnp.sqrt(1+w**2)))
    return - jnp.sqrt(jnp.pi/2/v)*(1+w**2)**(1/4)*jnp.exp(-jnp.abs(v)*eta)/w

def d_log_kn_fix_x(v,x):
    """ derivative wrt order, when x is fixed"""
    ep = 1e-6
    return (log_kn_large_order(v+ep ,x)-log_kn_large_order(v-ep,x))/2/ep

def SE_cov(xs, ys, v_0, w):
    N = len(xs)
    M = len(ys)
    K = np.zeros((N,M))
    
    for i in range(N):
        for j in range(M):
            try:
                K[i,j] = v_0 * np.exp(-w*(xs[i]-ys[j])**2/2)
            except:
                import pdb;pdb.set_trace()

    if N==M:
        return K  + 1e-7*np.identity(N)
    else:
        return K
    
def SE_cov_2d(xs, ys, v_0, w1, w2):
    N = xs.shape[1]
    M = ys.shape[1]
    K = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            K[i,j] = v_0 * np.exp(-w1*(xs[0,i]-ys[0,j])**2/2 -w2*(xs[1,i]-ys[1,j])**2/2 )

    if N==M:
        return K  + 1e-7*np.identity(N)
    else:
        return K
    
def SE_cov_multid(xs, ys, v_0, ws):
    N = xs.shape[1]
    M = ys.shape[1]
    K = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            K[i,j] = v_0 * np.exp( - ws @ ((xs[:,i]-ys[:,j])**2) / 2)

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
    
    if N==M:
        return K  + 1e-7*jnp.identity(N)
    else:
        return K
    
def SE_cov_multid_jnp(xs, ys, v_0, ws):
    def cov(i,j):
        return v_0 * jnp.exp(-jnp.dot(ws, (jnp.take(xs, jnp.array(i, int), axis=1)-jnp.take(ys, jnp.array(j, int), axis=1))**2)/2)

    N = xs.shape[1]
    M = ys.shape[1]
    K = jnp.fromfunction(cov, (N,M))
    
    if N==M:
        return K  + 1e-7*jnp.identity(N)
    else:
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
    # a_b, v_0_sq = theta
    a_b, v_0, wl, k, mu, l, omega = theta

    # v_0 = v_0_sq**2
    # wl = wl_sq**2

    K=SE_cov(x,x, v_0, w=wl)+ np.identity(len(x))*k
    beta = np.ones(len(x))*a_b
    mu = np.zeros_like(f)+mu

    invK = np.linalg.inv(K)
    deltaf = (f-mu) @ invK @ (f - mu)
    bKb = beta @ invK @ beta
    
    sign, logdet = np.linalg.slogdet(K)
    
    logp = (l - p/2)/2 * ( np.log(omega + deltaf) - np.log(omega + bKb) )\
        + (log_kn_large_order(l-p/2,  np.sqrt((omega + deltaf)*(omega + bKb)))) + (f-mu) @ invK @beta  \
        - logdet*sign/2  - np.log(special.kv(l, omega)) # - p/2* np.log(2*np.pi) 
        # - len(x)* (v_0_sq-1)**2/2 - v_0/2 # add a prior to prevent overfitting
        # + jnp.log(kn_io( l-p/2,  jnp.sqrt((omega + deltaf)*(omega + bKb))))\ using an approximation cos order too large
    # logp =   np.log(special.kv(l, omega))
    # import pdb;pdb.set_trace()
    return logp

def grad_jx(x, f, theta,  l: float, p: int, omega: float, k:float):
    """
    beta = a_b (+ b_b x)
    K(x,y) = v_0^2 exp(- w* (x-y)^2/2)

    so the parameters are
    theta = [a_b, b_b, v_0, w]

    k is the observation noise
    """
    a_b, v_0_sq, wl_Sq, k_sq, mu = theta # , l_sq, ome_sq
    # a_b = theta
    # jax.debug.print('a_b: {}', a_b)
    # v_0_sq = 1

    wl = wl_Sq**2
    v_0 = v_0_sq**2
    k = k_sq**2
    # l = - l_sq**2
    # omega = ome_sq**2

    K=SE_cov_jnp(x, x, v_0, w=wl) + jnp.identity(len(x))*k
    beta = jnp.ones(len(x))*a_b
    mu = jnp.zeros_like(f) + mu
    

    invK = jnp.linalg.inv(K)
    bKb = beta @ invK @ beta
    deltaf = (f-mu) @ invK @ (f - mu)
    bKb_df = jnp.sqrt((omega + deltaf)*(omega + bKb))

    # dK = - (kn_io(l-p/2-1, bKb_df)+kn_io(l-p/2+1, bKb_df))/2
    def dK_dv(nu, x, eps=1e-6):
        return (kn_io(nu + eps, x) - kn_io(nu - eps, x)) / (2*eps)
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
    gradv_0 = jnp.trace(2* gradK @ ((K-jnp.identity(len(x))*k)/v_0_sq) ) 
    gradw = jnp.trace(gradK @ dK_dw_jx(x, x, v_0, w=wl)*2*wl_Sq)
    gradk = 2* k_sq*jnp.trace( gradK @ jnp.identity(len(x)))  # gradient wrt k
    

    # gradient wrt mu
    ddelta = -2*(f-mu) @ invK @ jnp.ones_like(f)
    gradmu = (l-p/2)/2 / (omega + deltaf)* ddelta \
        + (-jnp.sqrt(1+w**2)/w) * jnp.sqrt((omega+ bKb)/ (omega + deltaf)) /2 * ddelta \
            - beta @ invK @ jnp.ones_like(f)
    
    # jax.debug.print('ddelta:{}, gradmu {}',ddelta, gradmu)

    return jnp.array([grada_b, gradv_0*2*v_0_sq, gradw*2*wl_Sq, gradk*2*k_sq, gradmu])

    # gradl = ( jnp.log(omega + deltaf) - jnp.log(omega + beta@invK @ beta) )/2\
    #         + d_log_kn_fix_x(v= l - p/2, x=jnp.sqrt((omega + deltaf)*(omega + bKb)))\
    #         - dK_dv(l, omega) / kn_io(l, omega)
    # gradome = (l - p/2)/2*( 1/ (omega + deltaf) - 1/(omega + beta@invK @ beta)) \
    #         + (-jnp.sqrt(1+w**2)/w)* ( jnp.sqrt(omega + deltaf)/(jnp.sqrt(omega + bKb)) + jnp.sqrt((omega + bKb)/(omega + deltaf)))/2 \
    #         - kvp_io(l, omega)/ kn_io(l, omega)
    
    
    # return jnp.array([grada_b, gradv_0, gradw, gradk, gradl, gradome]) # , gradl, gradome
    return jnp.array([grada_b, gradv_0*2*v_0_sq, gradw*2*wl_Sq, gradk*2*k_sq, -gradl*2*l, gradome*2*ome_sq])

def logprob_jx(x, f, theta,  l: float, p: int, omega: float, k:float):
    # a_b, v_0_sq = theta
    a_b, v_0_sq, wl_sq, k_sq, mu = theta #, l_sq, ome_sq

    v_0 = v_0_sq**2
    wl = wl_sq**2
    k = k_sq**2
    # l = - l_sq**2
    # omega = ome_sq**2

    K=SE_cov_jnp(x,x, v_0, w=wl)+ jnp.identity(len(x))*k
    beta = jnp.ones(len(x))*a_b
    mu = jnp.ones(p)*mu

    invK = jnp.linalg.inv(K)
    deltaf = (f-mu) @ invK @ (f - mu)
    bKb = beta @ invK @ beta

    sign, logdet = jnp.linalg.slogdet(K)
    
    logp = (l - p/2)/2 * ( jnp.log(omega + deltaf) - jnp.log(omega + bKb) )\
        + log_kn_large_order(l-p/2,  jnp.sqrt((omega + deltaf)*(omega + bKb))) \
         + (f-mu) @ invK @beta \
        - logdet*sign/2
        # - jnp.log(jnp.linalg.det(K/v_0))/2 - p*jnp.log(v_0)/2  
        # + jnp.log(kn_io(l-p/2,  jnp.sqrt((omega + deltaf)*(omega + bKb))))
        # - jnp.log(kn_io(l, omega))
        # - v_0/2 - wl/2 - k/2 - l /2 - omega / 2 #- p/2* jnp.log(2*jnp.pi)  \
        # - len(x)* (v_0_sq-1)**2/2 # add a prior to prevent overfitting
        # + jnp.log(kn_io( l-p/2,  jnp.sqrt((omega + deltaf)*(omega + bKb))))\ using an approximation cos order too large
    
    # jax.debug.print('theta:{}, detK {}, finKb: {}',theta, logdet, log_kn_large_order( l-p/2,  jnp.sqrt((omega + deltaf)*(omega + bKb))))
    
    return logp

def grad_st_jx(x, f, theta, n, k):
    v_sq, v_0_sq, wl_Sq = theta

    wl = wl_Sq**2
    v_0 = v_0_sq**2
    # k = k_sq**2
    v = 2+ v_sq**2 # v needs to be >2, rest > 0

    K=SE_cov_jnp(x, x, v_0, w=wl) + jnp.identity(len(x))*k

    invK = jnp.linalg.inv(K)
    mu = jnp.zeros_like(f)
    beta = (f-mu) @ invK @ (f - mu)
    alpha = invK @(f-mu)

    gradK = ((v+n)/(v+beta-2)*jnp.outer(alpha, alpha) - invK)/2

    gradv_0 = jnp.trace(2* gradK @ ((K-jnp.identity(len(x))*k)/v_0_sq) ) - v_0_sq # gradient wrt v0
    gradw = jnp.trace(gradK @ dK_dw_jx(x, x, v_0, w=wl))*2*wl_Sq - wl_Sq # gradient wrt wl_sq
    # gradk = 2* k_sq*jnp.trace( gradK @ jnp.identity(len(x)))  - k_sq # gradient wrt k

    gradv = - n/2/(v-2) + jax.scipy.special.digamma((v+n)/2)/2 - jax.scipy.special.digamma(v/2)/2\
            - jnp.log(1 + beta/(v-2))/2 + (v+n)*beta/(2*(v-2)**2 + 2*beta *(v-2))
    gradv = gradv * 2* v_sq   - v_sq
    
    return jnp.array([gradv, gradv_0, gradw])

def logprob_st_jx(x, f, theta, n, k):
    v_sq, v_0_sq, wl_Sq = theta

    wl = wl_Sq**2
    v_0 = v_0_sq**2
    # k = k_sq**2
    v = 2+ v_sq**2 # v needs to be >2, rest > 0

    K=SE_cov_jnp(x,x, v_0, w=wl)+ jnp.identity(len(x))*k

    invK = jnp.linalg.inv(K)
    mu = jnp.zeros_like(f)
    beta = (f-mu) @ invK @ (f - mu)

    c = 1
    logp = -n/2 *jnp.log((v-2)*jnp.pi) - jnp.log(jnp.linalg.det(K*c))/2  \
            + jax.scipy.special.gammaln((v+n)/2) - jax.scipy.special.gammaln(v/2)\
            -(v+n)/2*jnp.log(1 + beta/(v-2)) - v_0/2 - wl/2 - v_sq**2/2 - k/2  # - n*jnp.log(v_0)/2\

    # jax.debug.print('theta:{}, detK {}, k: {}, v0:{}, wl:{}',theta, jnp.log(jnp.linalg.det(K/c)), k, v_0, wl)

    return logp

def logprob_st(x, f, theta, n,k):
    v, v_0, wl = theta # v needs to be >2, rest > 0

    K=SE_cov_jnp(x,x, v_0, w=wl)+ np.identity(len(x))*k

    invK = np.linalg.inv(K)
    mu = np.zeros_like(f)
    beta = (f-mu) @ invK @ (f - mu)

    return -n/2 *np.log((v-2)*np.pi) - - np.log(np.linalg.det(K/v_0))/2 - n*np.log(v_0)/2\
            + scipy.special.gammaln((v+n)/2) - scipy.special.gammaln(v/2)\
            -(v+n)/2*jnp.log(1 + beta/(v-2))


def multid_grad_jx(x, f, theta,  l: float, p: int, omega: float, k:float):
    """
    beta = a_b (+ b_b x)
    K(x,y) = v_0^2 exp(- sum w_i* (x_i-y_i)^2/2)

    so the parameters are
    theta = [a_b, b_b, v_0, w]

    k is the observation noise
    """
    a_b = theta[jnp.array(0,int)]
    v_0_sq = theta[jnp.array(1,int)]
    k_sq = theta[jnp.array(2,int)]
    wl_sqs = theta[jnp.array([3,4,5,6,7,8,9,10,11,12,13], int)]

    # a_b = theta
    # jax.debug.print('a_b: {}', a_b)
    # v_0_sq = 1

    wls = wl_sqs**2
    v_0 = v_0_sq**2
    # k = k_sq**2
    # l = - l_sq**2
    # omega = ome_sq**2

    K=SE_cov_multid_jnp(x, x, v_0, ws=wls) + jnp.identity(p)*k
    beta = jnp.ones(p)*a_b
    mu = jnp.zeros_like(f)
    

    invK = jnp.linalg.inv(K)
    bKb = beta @ invK @ beta
    deltaf = (f-mu) @ invK @ (f - mu)
    bKb_df = jnp.sqrt((omega + deltaf)*(omega + bKb))

    # dK = - (kn_io(l-p/2-1, bKb_df)+kn_io(l-p/2+1, bKb_df))/2
    def dK_dv(nu, x, eps=1e-6):
        return (kn_io(nu + eps, x) - kn_io(nu - eps, x)) / (2*eps)
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
    gradv_0 = jnp.trace(2* gradK @ ((K-jnp.identity(p)*k)/v_0_sq) ) 
    # gradw = jnp.trace(gradK @ dK_dw_jx(x, x, v_0, w=wl)*2*wl_Sq)
    gradk = 2* k_sq*jnp.trace( gradK @ jnp.identity(p))  # gradient wrt k

    def gradw_1d(i, gradw):
        xi = jnp.take(x, jnp.array(i, int), axis=0)
        # jax.debug.print('{}',xi.shape)
        gradw.at[i+3].set(jnp.trace(gradK @ dK_dw_jx(xi, xi, v_0, w=wls[jnp.array(i,int)])) * 2*wl_sqs[jnp.array(i,int)])

        return gradw

    grads = lax.fori_loop(0,11,gradw_1d, jnp.zeros(14))

    grads.at[0].set(jnp.array(grada_b))
    grads.at[1].set(jnp.array(gradv_0))

    gradk = 0.0
    grads.at[2].set(jnp.array(gradk))
    

    # gradient wrt mu
    # ddelta = -2*(f-mu) @ invK @ jnp.ones_like(f)
    # gradmu = (l-p/2)/2 / (omega + deltaf)* ddelta \
    #     + (-jnp.sqrt(1+w**2)/w) * jnp.sqrt((omega+ bKb)/ (omega + deltaf)) /2 * ddelta \
    #         - beta @ invK @ jnp.ones_like(f)
    
    # jax.debug.print('ddelta:{}, gradmu {}',ddelta, gradmu)

    return grads

    # gradl = ( jnp.log(omega + deltaf) - jnp.log(omega + beta@invK @ beta) )/2\
    #         + d_log_kn_fix_x(v= l - p/2, x=jnp.sqrt((omega + deltaf)*(omega + bKb)))\
    #         - dK_dv(l, omega) / kn_io(l, omega)
    # gradome = (l - p/2)/2*( 1/ (omega + deltaf) - 1/(omega + beta@invK @ beta)) \
    #         + (-jnp.sqrt(1+w**2)/w)* ( jnp.sqrt(omega + deltaf)/(jnp.sqrt(omega + bKb)) + jnp.sqrt((omega + bKb)/(omega + deltaf)))/2 \
    #         - kvp_io(l, omega)/ kn_io(l, omega)
    
    
    # return jnp.array([grada_b, gradv_0, gradw, gradk, gradl, gradome]) # , gradl, gradome
    return jnp.array([grada_b, gradv_0*2*v_0_sq, gradw*2*wl_Sq, gradk*2*k_sq, -gradl*2*l, gradome*2*ome_sq])

def multid_logprob_jx(x, f, theta,  l: float, p: int, omega: float, k:float):
    a_b, v_0_sq, k_sq = theta[jnp.array([0,1,2],int)]
    wl_sqs = theta[jnp.array([3,4,5,6,7,8,9,10,11,12,13], int)]

    v_0 = v_0_sq**2
    wls = wl_sqs**2
    # k = k_sq**2
    # l = - l_sq**2
    # omega = ome_sq**2

    K=SE_cov_multid_jnp(x,x, v_0, ws=wls)+ jnp.identity(p)*k
    beta = jnp.ones(p)*a_b
    mu = jnp.zeros(p)

    invK = jnp.linalg.inv(K)
    deltaf = (f-mu) @ invK @ (f - mu)
    bKb = beta @ invK @ beta

    sign, logdet = jnp.linalg.slogdet(K)
    
    logp = (l - p/2)/2 * ( jnp.log(omega + deltaf) - jnp.log(omega + bKb) )\
        + log_kn_large_order(l-p/2,  jnp.sqrt((omega + deltaf)*(omega + bKb))) \
         + (f-mu) @ invK @beta \
        - logdet*sign/2 
        # - v_0/2 - jnp.sum(wls)/2 - k/2 - l /2 - omega / 2 #- p/2* jnp.log(2*jnp.pi)
        # + jnp.log(kn_io(l-p/2,  jnp.sqrt((omega + deltaf)*(omega + bKb))))
        # - jnp.log(kn_io(l, omega))
        
        # - len(x)* (v_0_sq-1)**2/2 # add a prior to prevent overfitting
        # + jnp.log(kn_io( l-p/2,  jnp.sqrt((omega + deltaf)*(omega + bKb))))\ using an approximation cos order too large
    
    # jax.debug.print('theta:{}, detK {}, finKb: {}',theta, logdet, log_kn_large_order( l-p/2,  jnp.sqrt((omega + deltaf)*(omega + bKb))))
    
    return logp

def multid_logprob(x, f, theta,  l: float, p: int, omega: float, k:float):
    a_b, v_0_sq, k_sq = theta[np.array([0,1,2],int)]
    wl_sqs = theta[np.array([3,4,5,6,7,8,9,10,11,12,13], int)]

    v_0 = v_0_sq**2
    wls = wl_sqs**2
    # k = k_sq**2
    # l = - l_sq**2
    # omega = ome_sq**2

    K=SE_cov_multid(x,x, v_0, ws=wls)+ np.identity(p)*k
    beta = np.ones(p)*a_b
    mu = np.zeros(p)

    invK = np.linalg.inv(K)
    deltaf = (f-mu) @ invK @ (f - mu)
    bKb = beta @ invK @ beta

    sign, logdet = np.linalg.slogdet(K)
    
    logp = (l - p/2)/2 * ( np.log(omega + deltaf) - np.log(omega + bKb) )\
        + log_kn_large_order_np(l-p/2,  np.sqrt((omega + deltaf)*(omega + bKb))) \
         + (f-mu) @ invK @beta \
        - logdet*sign/2 
        # - v_0/2 - jnp.sum(wls)/2 - k/2 - l /2 - omega / 2 #- p/2* jnp.log(2*jnp.pi)
        # + jnp.log(kn_io(l-p/2,  jnp.sqrt((omega + deltaf)*(omega + bKb))))
        # - jnp.log(kn_io(l, omega))
        
        # - len(x)* (v_0_sq-1)**2/2 # add a prior to prevent overfitting
        # + jnp.log(kn_io( l-p/2,  jnp.sqrt((omega + deltaf)*(omega + bKb))))\ using an approximation cos order too large
    
    # jax.debug.print('theta:{}, detK {}, finKb: {}',theta, logdet, log_kn_large_order( l-p/2,  jnp.sqrt((omega + deltaf)*(omega + bKb))))
    
    return logp
