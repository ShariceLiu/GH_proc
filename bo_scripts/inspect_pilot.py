"""Audit saved pilot outputs, plot EI histories and summarize diagnostics."""
import argparse
import csv
import json
from pathlib import Path
import os
os.environ.setdefault('MKL_THREADING_LAYER','SEQUENTIAL')
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gh_bo_core import Posterior,Acquisition,objective


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('folder',type=Path)
    args=parser.parse_args();folder=args.folder
    config=json.loads((folder/'experiment.json').read_text())
    rows=[];checks=[]
    for method in ['map','nuts','slice']:
        dest=folder/method
        state=json.loads((dest/'checkpoint.json').read_text())
        assert len(state['u'])==5+state['completed_steps']
        np.testing.assert_array_equal(state['u'][:5],config['initial_u'])
        np.testing.assert_allclose(objective(state['u']),state['y'],rtol=0,atol=1e-12)
        assert len(np.unique(state['u']))==len(state['u'])
        fig,axes=plt.subplots(state['completed_steps'],1,figsize=(8,2.1*state['completed_steps']),squeeze=False)
        for step in range(1,state['completed_steps']+1):
            path=dest/f'step_{step:03d}'
            fitted=np.load(path/'inference.npz');acq=np.load(path/'acquisition.npz')
            diag=json.loads((path/'diagnostics.json').read_text())
            np.testing.assert_array_equal(fitted['train_u'],state['u'][:4+step])
            assert np.isfinite(fitted['chains']).all() and np.isfinite(acq['ei']).all()
            posterior=Posterior(fitted['train_u'],(fitted['train_y']-config['output_center'])/config['output_scale'])
            points=np.array([float(acq['selected_u']),0.,.25,.5,.75,1.])
            low=Acquisition(posterior,fitted['acquisition_draws'],96)(points)
            high=Acquisition(posterior,fitted['acquisition_draws'],192)(points)
            error=float(np.max(np.abs(low-high)))
            np.testing.assert_allclose(low,high,atol=1e-9,rtol=1e-5)
            checks.append({'method':method,'step':step,'max_ei_quadrature_abs_error':error})
            row={'method':method,'step':step,'max_split_rhat':'','min_ess':'','divergences':'','map_gradient_norm':''}
            if method=='map': row['map_gradient_norm']=diag['selected_gradient_norm']
            else:
                row['max_split_rhat']=max(diag['split_rhat']);row['min_ess']=min(diag['ess'])
                if method=='nuts':row['divergences']=sum(c['divergences'] for c in diag['chain_details'])
            rows.append(row)
            ax=axes[step-1,0]
            ax.plot(5+5*acq['u'],acq['ei'])
            ax.axvline(5+5*float(acq['selected_u']),color='tab:red',label='Next query')
            ax.scatter(5+5*fitted['train_u'],np.zeros(len(fitted['train_u'])),marker='|',color='black',label='Observed locations')
            ax.set(ylabel=f'Step {step} EI',xlim=(5,10))
        axes[0,0].legend();axes[-1,0].set_xlabel('x')
        fig.suptitle(f'{method.upper()}: acquisition before each evaluation')
        fig.tight_layout();fig.savefig(dest/'acquisition_history.png',dpi=140);plt.close(fig)
    with (folder/'diagnostics_summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    result={'passed':True,'checks':['Matched initial observations across methods','Objective values reproduced','Exactly five initial points plus recorded steps','No repeated query locations','Saved inference datasets match observation prefixes','Finite samples and acquisition values','96 versus 192 quadrature nodes'], 'quadrature_checks':checks,'diagnostics':rows}
    (folder/'validation.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
