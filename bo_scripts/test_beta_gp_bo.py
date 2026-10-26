"""Numerical equivalence to regression models and all-method resume checks."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import json
import subprocess
import sys
import unittest
from pathlib import Path
import numpy as np
from scipy import integrate,special
from beta_gp_bo import BetaAcquisition,target_for,effective_coordinates
from GH_proc_joint_hmc import conditional_prediction
from GH_fixed_function import fixed_function_prediction


class ModelChecks(unittest.TestCase):
    def setUp(self):
        self.x=np.array([.1,.35,.7]);self.y=np.array([-.8,.4,-.1]);self.xp=np.array([.2,.5,.9])
        self.q=np.log([.7,.4,.002,18.,5.,1.3])
        self.z=effective_coordinates(self.q,'beta_gp')

    def test_gradient(self):
        target=target_for(self.x,self.y);_,gradient=target(self.z)
        fd=[]
        for d in np.eye(6)*1e-5:
            fd.append((target(self.z+d)[0]-target(self.z-d)[0])/2e-5)
        np.testing.assert_allclose(gradient,fd,atol=1e-6,rtol=1e-5)

    def test_full_bayes_components(self):
        reference=conditional_prediction(self.y,self.x,self.xp,self.q,model='beta_gp')
        m,v,weights=BetaAcquisition(self.x,self.y,self.z[None,:],'hmc').components(self.xp)
        np.testing.assert_allclose(m[0],reference['f_mean'],atol=1e-12)
        np.testing.assert_allclose(v[0],np.diag(reference['f_cov']),atol=1e-12)
        # Integrating beta uncertainty must leave a positive conditional variance.
        self.assertTrue(np.all(v>0));self.assertEqual(weights[0],1)

    def test_plugin_matches_reference_and_adaptive_ei(self):
        whole=np.r_[self.x,self.xp]
        p=conditional_prediction(self.y,self.x,whole,self.q,model='beta_gp')
        beta=p['effect_mean']/p['W'];v0,_,k,w,_,_=np.exp(self.q)
        ref=fixed_function_prediction(self.y,self.x,self.xp,v0=v0,k=k,w=w,
                                     beta_train=beta[:3],beta_pred=beta[3:])
        acq=BetaAcquisition(self.x,self.y,self.z[None,:],'map')
        m,v,weights=acq.components(self.xp)
        np.testing.assert_allclose(weights@m,ref['f_mean'],atol=1e-9)
        np.testing.assert_allclose(weights@(v+m*m)-(weights@m)**2,ref['f_var'],atol=1e-9)
        p,chi,psi=ref['W_lambda'],ref['W_chi'],ref['W_psi']
        answer=[]
        for i in range(len(self.xp)):
            def integrand(t):
                W=np.exp(t);mean=ref['location'][i]+W*ref['conditional_beta'][i]
                sd=np.sqrt(W*ref['f_scale_cov'][i,i]);delta=self.y.min()-mean;z=delta/sd
                ei=delta*special.ndtr(z)+sd*np.exp(-z*z/2)/np.sqrt(2*np.pi)
                logmass=p*t-.5*(chi/W+psi*W)+p/2*np.log(psi/chi)-np.log(2)-np.log(special.kve(p,np.sqrt(chi*psi)))+np.sqrt(chi*psi)
                return ei*np.exp(logmass)
            answer.append(integrate.quad(integrand,-12,12,epsabs=1e-11)[0])
        np.testing.assert_allclose(acq(self.xp),answer,rtol=1e-7,atol=1e-10)

    def test_all_methods_resume(self):
        root=Path(__file__).resolve().parents[1]/'bo_figure'/'gh_beta_gp_validation_resume'
        script=Path(__file__).with_name('run_beta_gp.py')
        def run(name,steps):
            subprocess.run([sys.executable,'-B',str(script),'--steps',str(steps),'--warmup','50','--samples','16',
                            '--map-starts','4','--acquisition-draws','8','--output',str(root/name)],
                           check=True,stdout=subprocess.DEVNULL)
            return {m:json.loads((root/name/'seed_000'/m/'checkpoint.json').read_text()) for m in ['map','hmc','slice']}
        first=run('split',1);split=run('split',2);full=run('continuous',2);noop=run('split',2)
        for m in split:
            np.testing.assert_array_equal(split[m]['u'],full[m]['u'])
            np.testing.assert_array_equal(split[m]['chain_endpoints'],full[m]['chain_endpoints'])
            np.testing.assert_array_equal(split[m]['u'][:6],first[m]['u'][:6])
            self.assertEqual(split[m],noop[m])
            self.assertEqual(split[m]['completed_steps'],2)
        for m in ['hmc','slice']:
            self.assertEqual(split[m]['u'][:5],split['map']['u'][:5])


if __name__=='__main__':unittest.main()
