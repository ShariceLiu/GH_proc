"""Independent dense joint-Gaussian check for two shared functions."""
import unittest
import numpy as np
from scipy import linalg,stats
from threadpoolctl import threadpool_limits
from GH_shared_function_map import gig_mean
from GH_shared_mu_beta_map import shared_mu_beta_calculation,make_target

class TwoSharedTests(unittest.TestCase):
    def test_tiny_noise_against_high_precision(self):
        import mpmath as mp
        x=np.linspace(0,1,8);W=np.array([.7,1.5,2.4])
        Y=np.array([1+.1*x+w*(2+.2*x)+1e-5*np.sin((j+1)*x) for j,w in enumerate(W)])
        q=np.log(np.r_[.3,10,15,1e-10,.4,.02,.03,W])
        result=shared_mu_beta_calculation(Y,x,q)
        with mp.workdps(60):
            def reference(logs):
                v,am,ab,k,w,wm,wb,*ws=[mp.exp(z) for z in logs]
                c=[z/mp.mpf(gig_mean()) for z in ws]
                def entry(i,j):
                    ri,xi=divmod(i,8);rj,xj=divmod(j,8)
                    d=(mp.mpf(x[xi])-mp.mpf(x[xj]))**2
                    value=am*mp.exp(-wm*d/2)+c[ri]*c[rj]*ab*mp.exp(-wb*d/2)
                    if ri==rj:value+=ws[ri]*(v*mp.exp(-w*d/2)+(k if xi==xj else 0))
                    return value
                C=mp.matrix([[entry(i,j) for j in range(24)] for i in range(24)])
                L=mp.cholesky(C);y=mp.matrix(Y.ravel().tolist())
                return -(24*mp.log(2*mp.pi)+2*sum(mp.log(L[i,i]) for i in range(24))+(y.T*mp.cholesky_solve(C,y))[0])/2
            logs=list(map(mp.mpf,q));self.assertAlmostEqual(result['log_likelihood'],float(reference(logs)),places=6)
            fd=[]
            for j in range(len(q)):
                a,b=logs.copy(),logs.copy();a[j]+=mp.mpf('1e-6');b[j]-=mp.mpf('1e-6')
                fd.append(float((reference(a)-reference(b))/mp.mpf('2e-6')))
            np.testing.assert_allclose(result['gradient'],fd,atol=2e-5,rtol=2e-6)

    def test_dense_joint_and_gradients(self):
        rng=np.random.default_rng(31);x=np.array([0.,.4,1.3,2.5]);Y=rng.normal(size=(4,4))
        for W in (np.array([.5,.8,1.5,2.7]),np.ones(4)*1.3):
            q=np.log(np.r_[.3,1.2,.7,.08,.9,.15,1.5,W]);c=W/gig_mean()
            d=(x[:,None]-x)**2;A=.3*np.exp(-.45*d)+.08*np.eye(4)
            Sm=1.2*np.exp(-.075*d);Sb=.7*np.exp(-.75*d)
            C=np.kron(np.diag(W),A)+np.kron(np.ones((4,4)),Sm)+np.kron(np.outer(c,c),Sb)
            cross=np.vstack([np.hstack([Sm]*4),np.hstack([cj*Sb/gig_mean() for cj in c])])
            pri=linalg.block_diag(Sm,Sb/gig_mean()**2)
            expected=cross@linalg.solve(C,Y.ravel());cov=pri-cross@linalg.solve(C,cross.T)
            actual=shared_mu_beta_calculation(Y,x,q,posterior=True)
            self.assertAlmostEqual(actual['log_likelihood'],stats.multivariate_normal.logpdf(Y.ravel(),cov=C),places=10)
            np.testing.assert_allclose(np.r_[actual['mu'],actual['beta']],expected,atol=1e-11)
            np.testing.assert_allclose(actual['mu_cov'],cov[:4,:4],atol=1e-11)
            np.testing.assert_allclose(actual['beta_cov'],cov[4:,4:],atol=1e-11)
            np.testing.assert_allclose(actual['mu_beta_cov'],cov[:4,4:],atol=1e-11)
            target=make_target(Y,x,q[:7]);ds=1e-5*np.eye(len(q))
            fd=np.array([(target(q+d)[0]-target(q-d)[0])/2e-5 for d in ds])
            np.testing.assert_allclose(fd,target(q)[1],atol=3e-7,rtol=3e-6)

if __name__=='__main__':
    with threadpool_limits(1):unittest.main()
