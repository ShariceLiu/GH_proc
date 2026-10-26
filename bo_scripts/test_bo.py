"""Independent numerical checks and deterministic checkpoint continuation."""
import json
import os
os.environ.setdefault('MKL_THREADING_LAYER', 'SEQUENTIAL')
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from scipy import integrate,linalg,special,stats
from gh_bo_core import Posterior,Acquisition,PRIOR_MEAN,PRIOR_SD,kernel,mixing_quadrature


class Checks(unittest.TestCase):
    def test_initial_state_isolation(self):
        from run_pilot import initial_state
        config={'initial_u':[.1,.4],'initial_y':[2.,3.]}
        a=initial_state(config);b=initial_state(config)
        a['u'].append(.8);a['y'].append(-1.)
        self.assertEqual(len(config['initial_u']),2)
        self.assertEqual(len(b['u']),2)

    def setUp(self):
        self.p=Posterior([.1,.4,.8],[-1.,.7,-.3])
        self.q=np.array([.3,-.7,.2,-1.2])

    def test_density_and_gradient(self):
        mu,beta,lv,le=self.q
        c=kernel(self.p.x,self.p.x,np.exp(lv),np.exp(le))+1e-6*np.eye(3)
        def integrand(t):
            w=np.exp(t)
            logprior_w=-np.log(2)-np.log(special.kve(1,1))+1-.5*(w+1/w)
            return np.exp(stats.multivariate_normal.logpdf(self.p.y,mean=np.full(3,mu+beta*w),cov=w*c)+logprior_w+t)
        density=integrate.quad(integrand,-12,12,epsabs=1e-12,epsrel=1e-10)[0]
        prior=-.5*np.sum(((self.q-PRIOR_MEAN)/PRIOR_SD)**2)
        self.assertAlmostEqual(self.p.log_prob(self.q),np.log(density)+prior,places=8)
        value,g=self.p.value_grad(self.q)
        self.assertAlmostEqual(value,-self.p.log_prob(self.q),places=9)
        fd=[]
        for j in range(4):
            d=np.zeros(4);d[j]=1e-5
            fd.append(-(self.p.log_prob(self.q+d)-self.p.log_prob(self.q-d))/2e-5)
        np.testing.assert_allclose(g,fd,rtol=2e-5,atol=2e-6)

    def test_mixing_and_ei(self):
        for p,chi,psi in [(-4.,37.,2.),(-.5,2.,9.),(1.,1.,1.)]:
            w,a=mixing_quadrature(p,chi,psi)
            expected=np.sqrt(chi/psi)*special.kve(p+1,np.sqrt(chi*psi))/special.kve(p,np.sqrt(chi*psi))
            self.assertAlmostEqual(float(w@a),expected,places=8)
        x=np.array([.2,.6,.95])
        mean,b,var,p,chi,psi=self.p.components(self.q,x)
        answer=[]
        for i in range(len(x)):
            def integrand(t):
                w=np.exp(t);sd=np.sqrt(w*var[i]);delta=self.p.y.min()-mean[i]-w*b[i]
                z=delta/sd
                ei=delta*special.ndtr(z)+sd*np.exp(-z*z/2)/np.sqrt(2*np.pi)
                logpdf=(p-1)*t-.5*(chi/w+psi*w)+p/2*np.log(psi/chi)-np.log(2)-np.log(special.kve(p,np.sqrt(chi*psi)))+np.sqrt(chi*psi)
                return ei*np.exp(logpdf+t)
            answer.append(integrate.quad(integrand,-12,12,epsabs=1e-11,epsrel=1e-9)[0])
        np.testing.assert_allclose(Acquisition(self.p,self.q[None,:])(x),answer,rtol=1e-7,atol=1e-10)

    def test_resume(self):
        with tempfile.TemporaryDirectory(prefix='bo_resume_') as directory:
            root=Path(directory)
            script=Path(__file__).with_name('run_pilot.py')
            def run(name,steps):
                subprocess.run([sys.executable,'-B',str(script),'--methods','map','--steps',str(steps),'--output',str(root/name)],check=True,stdout=subprocess.DEVNULL)
                return json.loads((root/name/'seed_000/map/checkpoint.json').read_text())
            one=run('split',1)
            resumed=run('split',2)
            continuous=run('continuous',2)
            np.testing.assert_array_equal(resumed['u'],continuous['u'])
            self.assertEqual(resumed['u'][:6],one['u'])
            self.assertEqual(resumed['chain_endpoints'],continuous['chain_endpoints'])
            unchanged=run('split',2)
            self.assertEqual(resumed,unchanged)


if __name__=='__main__': unittest.main()
