"""Reference-model, gradient, EI and all-24-combination continuation tests."""
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import json
import subprocess
import sys
import unittest
from pathlib import Path
import numpy as np
from scipy import stats,special
from multi_model_bo import Model,ModelAcquisition,MODELS,JOINT
from GH_proc_joint_hmc import conditional_prediction
from GH_fixed_function import fixed_function_prediction,marginal_distribution


class Checks(unittest.TestCase):
    def setUp(self):
        self.x=np.array([.1,.35,.7]);self.y=np.array([-.8,.4,-.1]);self.xp=np.array([.2,.5,.9])

    def test_all_gradients(self):
        for name in MODELS:
            with self.subTest(model=name):
                model=Model(self.x,self.y,name);q=model.base+np.linspace(-.12,.15,len(model.base))
                _,g=model.target(q)
                fd=[(model.target(q+d)[0]-model.target(q-d)[0])/2e-5 for d in np.eye(len(q))*1e-5]
                np.testing.assert_allclose(g,fd,atol=1e-6,rtol=1e-5)

    def test_constant_joint_density(self):
        model=Model(self.x,self.y,'ghp_const_beta');q=model.base.copy();q[3:5]=[.3,-.7]
        v,k,w,W=np.exp(q[[0,1,2,5]]);mu,beta=q[3:5]
        C=W*(v*np.exp(-.5*w*(self.x[:,None]-self.x[None,:])**2)+k*np.eye(3))
        reference=(stats.multivariate_normal.logpdf(self.y,np.full(3,mu+W*beta),C)
          +stats.norm.logpdf(q[:3],model.base[:3],2).sum()+stats.norm.logpdf(mu,0,2)
          +stats.norm.logpdf(beta)+stats.geninvgauss.logpdf(W,1,1)+np.log(W))
        self.assertAlmostEqual(model.target(q)[0],reference,places=10)

    def test_zero_joint_densities_and_integrated_prediction(self):
        for name in ('gp_zero','stp_zero','ghp_zero'):
            with self.subTest(model=name):
                model=Model(self.x,self.y,name);q=model.base+np.linspace(.1,.25,len(model.base))
                v,k,w=np.exp(q[:3]);W=1. if model.gp else np.exp(q[3])
                C=W*(v*np.exp(-.5*w*(self.x[:,None]-self.x[None,:])**2)+k*np.eye(3))
                expected=stats.multivariate_normal.logpdf(self.y,np.zeros(3),C)+stats.norm.logpdf(q[:3],model.base[:3],2).sum()
                if not model.gp:
                    expected+=(stats.invgamma.logpdf(W,2.5,scale=1.5) if name=='stp_zero' else stats.geninvgauss.logpdf(W,1,1))+np.log(W)
                self.assertAlmostEqual(model.target(q)[0],expected,places=10)
                ref=fixed_function_prediction(self.y,self.x,self.xp,v0=v,k=k,w=w,
                    mixing='gp' if model.gp else 'ig' if name=='stp_zero' else 'gig')
                for method in ('map','hmc','slice'):
                    acq=ModelAcquisition(model,q[None,:],method);mean,var,weights=acq.components(self.xp)
                    np.testing.assert_allclose(weights@mean,ref['f_mean'],atol=1e-9)
                    np.testing.assert_allclose(weights@(var+mean**2)-(weights@mean)**2,ref['f_var'],atol=1e-8)
                if model.gp:
                    sd=np.sqrt(ref['f_var']);delta=self.y.min()-ref['f_mean'];z=delta/sd
                    np.testing.assert_allclose(acq(self.xp),delta*stats.norm.cdf(z)+sd*stats.norm.pdf(z),atol=1e-12)

    def test_gaussian_components_match_regression(self):
        for name in MODELS[1:5]:
            with self.subTest(model=name):
                model=Model(self.x,self.y,name);q=model.physical(model.base)
                kind='mu_gp' if model.gp else JOINT[name]
                reference=conditional_prediction(self.y,self.x,self.xp,np.r_[q,0.] if model.gp else q,model=kind)
                mean,var,_=ModelAcquisition(model,model.base[None,:],'hmc').components(self.xp)
                np.testing.assert_allclose(mean[0],reference['f_mean'],atol=1e-12)
                np.testing.assert_allclose(var[0],np.diag(reference['f_cov']),atol=1e-12)

    def test_plugin_moments_and_student_ei(self):
        for name in MODELS:
            with self.subTest(model=name):
                model=Model(self.x,self.y,name);q=model.physical(model.base)
                if model.zero:
                    v,k,w=np.exp(q[:3]);kwargs={}
                elif model.constant:
                    v,k,w,W=np.exp(q[[0,1,2,5]])
                    kwargs=dict(mu_train=np.full(3,q[3]),mu_pred=np.full(3,q[3]),
                                beta_train=np.full(3,q[4]),beta_pred=np.full(3,q[4]))
                else:
                    v,_,k,w,_=np.exp(q[:5]);kind='mu_gp' if model.gp else JOINT[name]
                    prediction=conditional_prediction(self.y,self.x,np.r_[self.x,self.xp],np.r_[q,0.] if model.gp else q,model=kind)
                    estimate=prediction['effect_mean']/(prediction['W'] if name=='ghp_beta_gp' else 1)
                    prefix='beta' if name=='ghp_beta_gp' else 'mu'
                    kwargs={prefix+'_train':estimate[:3],prefix+'_pred':estimate[3:]}
                mixing='gp' if model.gp else 'ig' if name in ('stp_mu_gp','stp_zero') else 'gig'
                ref=fixed_function_prediction(self.y,self.x,self.xp,v0=v,k=k,w=w,mixing=mixing,**kwargs)
                acq=ModelAcquisition(model,model.base[None,:],'map');mean,var,weights=acq.components(self.xp)
                np.testing.assert_allclose(weights@mean,ref['f_mean'],atol=1e-9)
                np.testing.assert_allclose(weights@(var+mean**2)-(weights@mean)**2,ref['f_var'],atol=1e-8)
                if name in ('stp_mu_gp','stp_zero'):
                    a,b=ref['W_shape'],ref['W_scale'];df=2*a;s=np.sqrt(b/a*np.diag(ref['f_scale_cov']))
                    delta=self.y.min()-ref['location'];z=delta/s
                    ei=delta*stats.t.cdf(z,df)+s*(df+z*z)/(df-1)*stats.t.pdf(z,df)
                    np.testing.assert_allclose(acq(self.xp),ei,atol=1e-9,rtol=1e-7)

    def test_resume_all_models_methods(self):
        root=Path(__file__).resolve().parents[1]/'bo_figure'/'eight_model_validation_resume'
        script=Path(__file__).with_name('run_model_comparison.py')
        def run(name,steps):
            command=[sys.executable,'-B',str(script),'--steps',str(steps),'--warmup','50','--samples','8',
                     '--map-starts','4','--acquisition-draws','4','--output',str(root/name)]
            result=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(result.returncode,0,result.stdout+result.stderr)
            return {(model,method):json.loads((root/name/'seed_000'/model/method/'checkpoint.json').read_text())
                    for model in MODELS for method in ['map','hmc','slice']}
        one=run('split',1);split=run('split',2);full=run('continuous',2);noop=run('split',2)
        for key in split:
            with self.subTest(combination=key):
                np.testing.assert_array_equal(split[key]['u'],full[key]['u'])
                np.testing.assert_array_equal(split[key]['chain_endpoints'],full[key]['chain_endpoints'])
                self.assertEqual(split[key]['u'][:6],one[key]['u'][:6])
                self.assertEqual(split[key],noop[key]);self.assertEqual(len(split[key]['u']),7)
                self.assertEqual(split[key]['u'][:5],split[(MODELS[0],'map')]['u'][:5])


if __name__=='__main__':unittest.main()
