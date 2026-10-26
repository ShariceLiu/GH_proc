"""Fit ten aa spectra independently; no cross-curve training information.

Run python phoneme_ten_individual_curves_experiment.py.
Uses existing source data and observation masks; no downloads required.
"""
import os
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ[key]='1'
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from threadpoolctl import threadpool_limits
from electricity_single_curve_plugin_experiment import run_case, COUNTS, MODELS, LABELS
from electricity_ten_individual_curves_experiment import aggregate, METRICS
from shared_beta_plugin_experiment import write_json

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'figure/phoneme_aa_plugin_five_splits'
OUTPUT=ROOT/'figure/phoneme_aa_ten_individual_curves_plugin'


def prepare():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with np.load(SOURCE/'dataset.npz') as source, np.load(SOURCE/'split_74027/split.npz') as split:
        rows=split['test_rows'][:10]
        data=dict(x=source['x'],frequency_bin=source['frequency_bin'],y=source['y'][rows],
                  speaker_ids=source['speaker_ids'][rows],source_rows=source['source_rows'][rows],
                  source_recordings=source['source_recordings'][rows],
                  context_orders=split['context_orders'][:10])
    assert data['y'].shape==(10,128) and len(np.unique(data['speaker_ids']))==10
    for order in data['context_orders']:
        np.testing.assert_array_equal(np.sort(order),np.arange(128))
    path=OUTPUT/'dataset.npz'
    if path.exists():
        with np.load(path) as saved:
            for key in data:np.testing.assert_array_equal(saved[key],data[key])
    else:np.savez_compressed(path,**data)
    write_json(OUTPUT/'experiment.json',dict(phoneme='aa',n_curves=10,n_bins=128,
        selection='First ten test curves in saved split 74027, in saved order; chosen before fitting.',
        source=str(SOURCE),source_sha256=hashlib.sha256((SOURCE/'dataset.npz').read_bytes()).hexdigest(),
        source_split_sha256=hashlib.sha256((SOURCE/'split_74027/split.npz').read_bytes()).hexdigest(),
        speakers=data['speaker_ids'],source_rows=data['source_rows'],context_seed=84026,
        context_counts=COUNTS,models=MODELS,GHP_lambda=1.,GHP_omega=1.,STP_nu=5.,starts=24,
        preprocessing='Uncentered y divided by current observed context SD only; fixed x in [0,10].',
        prior='Same context-only log-kernel prior as electricity single-curve experiment; SD2.',
        fitting='Independently fit theta and W for each curve/context/model; latent GPs marginalized during MAP.',
        prediction='Fix theta and conditional GP mean mu/beta, then recompute and integrate W.',
        uncertainty='Function and hyperparameter estimation uncertainty omitted.',
        scoring='Held-out observed log-periodogram y, not latent f. 113/98 held-out bins; common test uses98.',
        aggregation='Mean and sample SD (ddof=1) across ten per-curve metrics; RMSE not pooled.'))
    return data


def audit(curve,count,model,folder,score,common):
    case=folder/f'context_{count}'/model
    config=json.loads((case/'configuration.json').read_text())
    ids=np.sort(curve['context_order'][:count]);held=np.setdiff1d(np.arange(128),ids)
    np.testing.assert_array_equal(config['context_ids'],ids)
    scale=curve['y'][ids].std()
    np.testing.assert_allclose(config['response_scale'],scale)
    center=np.log([.5,max(np.mean((curve['y'][ids]/scale)**2),.5),.01,1.,.04])
    np.testing.assert_allclose(config['population_prior_center'],center)
    fit=json.loads((case/'fit.json').read_text())
    assert fit['max_abs_gradient']<1e-3 and fit['max_hessian_eigenvalue']<0
    with np.load(case/'predictions.npz') as p:
        np.testing.assert_array_equal(p['test_ids'],held)
        for record,selection in [(score,held),(common,np.setdiff1d(np.arange(128),curve['context_order'][:30]))]:
            pos=np.searchsorted(held,selection);truth=curve['y'][selection];ci=p['y_ci'][selection]
            metrics=[np.sqrt(np.mean((truth-p['y_mean'][selection])**2)),p['nlpd'][pos].mean(),
                     p['crps'][pos].mean(),np.mean((truth>=ci[:,0])&(truth<=ci[:,1])),np.mean(ci[:,1]-ci[:,0])]
            np.testing.assert_allclose([record[m] for m in METRICS],metrics,atol=1e-12)


def latex(summary,path,common=False):
    caption=(r'Phoneme aa: mean $\pm$ sample SD across ten curves from distinct speakers. '
        r'Each curve independently fits MAP parameters using only 15 or 30 observed bins; '
        r'no parameters or functions learned from other curves. '
        r'GHP fixes $\lambda=\omega=1$; STP fixes $\nu=5$. '
        r'Plug-in $\mu/\beta$; prediction integrates $W$. Metrics score observed log-periodograms. ')
    caption+=('Both counts use the same 98 held-out bins.' if common else 'Scores use 113 or 98 held-out bins, respectively.')
    lines=[r'\begin{table}[htbp]',r'\centering\small',r'\caption{'+caption+'}',
        r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{rlrrrrr}',r'\toprule',
        r'Observed & Model & RMSE & NLPD & CRPS & 95\% coverage (\%) & Width \\',r'\midrule']
    for row in summary.to_dict('records'):
        cells=[str(row['n_train']),LABELS[row['model']]]
        for metric in METRICS:
            factor=100 if metric=='y_coverage_95' else 1
            cells.append(f"${factor*row[metric+'_mean']:.5f} \\pm {factor*row[metric+'_std']:.5f}$")
        lines.append(' & '.join(cells)+r' \\')
        if row['n_train']==15 and row['model']==MODELS[-1]:lines.append(r'\midrule')
    lines += [r'\bottomrule',r'\end{tabular}}',r'\end{table}']
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def main():
    data=prepare();records=[];common_records=[];parameters=[]
    with PdfPages(OUTPUT/'regression.pdf') as pdf:
        for j,speaker in enumerate(data['speaker_ids']):
            print(f'Curve {j+1}/10: {speaker}',flush=True)
            curve=dict(x=data['x'],y=data['y'][j],context_order=data['context_orders'][j])
            folder=OUTPUT/f'curve_{speaker}';folder.mkdir(exist_ok=True)
            for count in COUNTS:
                for model in MODELS:
                    score,common,parameter=run_case(curve,count,model,output=folder)
                    audit(curve,count,model,folder,score,common)
                    identity=dict(speaker=str(speaker),source_row=int(data['source_rows'][j]))
                    records.append(dict(**identity,**score));common_records.append(dict(**identity,**common))
                    parameters.append(dict(**identity,**parameter))
            for items,name in [(records,'per_curve_scores'),(common_records,'common_test_per_curve_scores'),(parameters,'parameters')]:
                pd.DataFrame(items).to_csv(OUTPUT/(name+'.csv'),index=False)
            fig,axes=plt.subplots(4,2,figsize=(12,11),sharex=True,constrained_layout=True)
            for i,model in enumerate(MODELS):
                for col,count in enumerate(COUNTS):
                    with np.load(folder/f'context_{count}'/model/'predictions.npz') as p:
                        ax=axes[i,col];x=data['frequency_bin'];ids=p['context_ids']
                        ax.plot(x,curve['y'],'k-',lw=1,label='Observed spectrum')
                        ax.plot(x,p['y_mean'],label='Predictive mean')
                        ax.fill_between(x,*p['y_ci'].T,alpha=.2,label='95% observation interval')
                        ax.scatter(x[ids],curve['y'][ids],s=15,c='red',label='Revealed bins')
                        ax.set(title=f'{LABELS[model]}, {count} observations',ylabel='Log-periodogram')
                        if i==3:ax.set_xlabel('Original frequency bin (zero-based)')
                        if i==0 and col==0:ax.legend(fontsize=6)
            fig.suptitle(f'Phoneme aa, speaker {speaker}: independently fitted curve')
            pdf.savefig(fig);plt.close(fig)
    for items,name,common in [(records,'scores',False),(common_records,'common_test_scores',True)]:
        summary=aggregate(pd.DataFrame(items),identity='speaker')
        summary.to_csv(OUTPUT/(name+'.csv'),index=False)
        latex(summary,OUTPUT/('comparison_common_test.tex' if common else 'comparison.tex'),common)
        print(name+'\n'+summary.to_string(index=False),flush=True)
    write_json(OUTPUT/'validation.json',dict(passed=True,n_curves=10,n_fits=80,distinct_speakers=True,
        context_only_scaling_and_prior=True,independent_prediction_checks=True,
        metrics_recomputed_from_saved_predictions=True,sample_sd_ddof=1))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
