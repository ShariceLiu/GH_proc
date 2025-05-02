import numpy as np
import jax.numpy as jnp
from scipy import special
import jax

def SE_cov(xs, ys, v_0, w):
    N = len(xs)
    M = len(ys)
    K = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            K[i,j] = v_0 * np.exp(-w*(xs[i]-ys[j])**2/2)

    return K  + 1e-6*np.identity(N)

def SE_cov_jnp(xs, ys, v_0, w):
    N = len(xs)
    M = len(ys)
    K = jnp.zeros((N,M))
    for i in range(N):
        for j in range(M):
            K= K.at[i,j].set( v_0 * jnp.exp(-w*(xs[i]-ys[j])**2/2) )

    return K

def dK_dw(xs, ys, v_0, w):
    N = len(xs)
    M = len(ys)
    K = np.zeros((N,M))
    for i in range(N):
        for j in range(M):
            K[i,j] = - v_0 *(xs[i]-ys[j])**2/2 *  np.exp(-w*(xs[i]-ys[j])**2/2)

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

def grad_jx(x, f, theta,  l: float, p: int, omega: float):
    """
    beta = a_b (+ b_b x)
    K(x,y) = v_0 exp(- w* (x-y)^2/2)

    so the parameters are
    theta = [a_b, b_b, v_0, w]
    """
    a_b, v_0, w = theta

    K=SE_cov_jnp(x, x, v_0, w)
    beta = jnp.ones(len(x))*a_b
    mu = jnp.zeros_like(f)
    

    invK = jnp.linalg.inv(K)
    bKb = beta @ invK @ beta
    deltaf = (f-mu) @ invK @ (f - mu)
    bKb_df = jnp.sqrt((omega + deltaf)*(omega + bKb))
    dK = (special.kn((l-p)/2, np.array(bKb_df) + 1e-5) - special.kn((l-p)/2, np.array(bKb_df) - 1e-5))/(2e-5)
    # dK = jax.grad(lambda x: special.kn((l-p)/2, x))(bKb_df)
    kbbk = invK @ jnp.outer(beta, beta) @ invK

    gradbeta = (l - p/2)/(omega + bKb) * (invK @ beta) \
        + dK/special.kn((l-p)/2, bKb_df)\
         * jnp.sqrt(deltaf/ (omega + bKb)) *( invK @ beta) \
        + invK @ (f - mu)
    grada_b = gradbeta

    gradK = (l - p/2)/2 * ( (invK @ jnp.outer(f-mu, f-mu) @ invK )/(omega + deltaf) + (kbbk/(omega + beta@invK @ beta)) )\
        + dK/special.kn((l-p)/2, bKb_df) * ( jnp.sqrt(omega + deltaf)* kbbk /(jnp.sqrt(omega + bKb)) +\
                np.sqrt((omega + bKb)/(omega + deltaf))* (invK @ jnp.outer(f-mu, f-mu)@invK) )/2 \
        - invK/2 - invK @ np.outer((f-mu), beta) @ invK
    gradv_0 = jnp.trace(gradK*K/v_0)
    gradw = jnp.trace(gradK * dK_dw(x, x, v_0, w))
    
    return [grada_b, gradv_0, gradw]

def logprob_jx(x, f, theta,  l: float, p: int, omega: float):
    a_b, v_0, w = theta

    K=SE_cov(x,x, v_0, w)
    beta = jnp.ones(len(x))*a_b
    mu = jnp.zeros_like(f)

    invK = jnp.linalg.inv(K)
    deltaf = (f-mu) @ invK @ (f - mu)
    bKb = beta @ invK @ beta
    logp = (l - p/2)/2 * ( jnp.log(omega + deltaf) - jnp.log(omega + bKb) )\
        + jnp.log(special.kn( (l-p)/2,  jnp.sqrt((omega + deltaf)*(omega + bKb))))\
        - jnp.log(jnp.linalg.det(K)) \
        - (f-mu) @ invK @beta #- p/2* jnp.log(2*jnp.pi) - jnp.log(special.kn(l, omega))
    return logp