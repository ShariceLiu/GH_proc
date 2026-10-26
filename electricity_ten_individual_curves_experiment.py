"""Repeat single-curve MAP independently on ten date-selected electricity days.

Run: python electricity_ten_individual_curves_experiment.py
No hyperparameters or scales are shared. Means and sample SDs summarize
per-day metrics, rather than pooling squared errors before taking the root.
"""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from electricity_single_curve_plugin_experiment import SOURCE, OUTPUT as SINGLE, COUNTS, LABELS, MODELS, run_case
from shared_beta_plugin_experiment import write_json

OUTPUT=Path('figure/electricity_mt001_ten_individual_curves_plugin')
METRICS=('y_rmse','y_nlpd','y_crps','y_coverage_95','y_interval_width')


def prepare():
    OUTPUT.mkdir(parents=True,exist_ok=True)
    with np.load(SOURCE) as source:
        rows=source['test_rows'][:10]
        data=dict(x=source['x'].copy(),time_hours=source['time_hours'].copy(),
            y=source['y'][rows].copy(),dates=source['dates'][rows].copy(),
            context_orders=source['context_orders'][:10].copy())
    assert len(data['dates'])==10 and len(np.unique(data['dates']))==10
    np.testing.assert_array_equal(data['dates'],np.sort(data['dates']))
    for order in data['context_orders']:
        np.testing.assert_array_equal(np.sort(order),np.arange(96))
    saved=OUTPUT/'dataset.npz'
    if saved.exists():
        with np.load(saved) as previous:
            for key in data:np.testing.assert_array_equal(previous[key],data[key])
    else:np.savez_compressed(saved,**data)
    write_json(OUTPUT/'experiment.json',dict(customer='MT_001',dates=data['dates'],
        selection='First ten original test dates, chosen before fitting; includes original 6 March experiment',
        source=str(SOURCE),source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        n_curves=10,context_counts=COUNTS,n_models=4,n_fits=80,context_seed=44026,
        fitting='Same single-curve plug-in MAP; each date/context has independently learned kernel/noise/W parameters',
        scaling_and_prior='Only current curve observed readings; exact same rules as single-curve experiment',
        GHP_lambda=1.,GHP_omega=1.,STP_nu=5.,optimization_starts=24,
        aggregation='Equal-weight arithmetic mean and sample SD (ddof=1) of ten per-curve metrics',
        rmse_aggregation='Mean of per-curve RMSE, not pooled RMSE',
        uncertainty_note='SD describes between-day variation, not standard error or confidence interval; dates can be dependent'))
    (OUTPUT/'method.txt').write_text(
        'TEN CURVES, FIT EACH CURVE SEPARATELY\n'
        'Select the first ten saved MT_001 test dates in chronological order.\n'
        'No outcome-dependent selection. Includes 6 March, the original case.\n'
        'Use each date\'s saved nested random 15/30 observations (seed 44026).\n'
        'Repeat exactly electricity_single_curve_plugin_experiment.run_case.\n'
        'For every date and observation count, fit all kernel/noise parameters\n'
        'and W using only those observed readings, with 24 MAP starts.\n'
        'No cross-day parameter sharing, prior-center estimation or scaling.\n'
        'Both SE kernels are learned. Lambda=omega=1 and nu=5 stay fixed.\n'
        'Training likelihood integrates latent GPs. At joint MAP, use the\n'
        'conditional posterior mean function as fixed mu/beta. Recompute and\n'
        'integrate W for prediction. Function/hyperparameter uncertainty omitted.\n'
        'Response is uncentered, scaled by current context SD; priors match\n'
        'the original single-curve code, using only current observed values.\n'
        'Reuse original 6 March fits after verifying its data and masks match.\n'
        'Predictions and independent numerical checks are regenerated.\n'
        'Primary metrics use 81/66 unrevealed readings for 15/30 observations.\n'
        'Common-test metrics use the same 66 targets for both counts each day.\n'
        'Report arithmetic mean and SAMPLE SD (ddof=1) over ten per-day scores.\n'
        'RMSE is averaged per day; do not take square root of pooled MSE.\n'
        'Coverage means/SDs are expressed as percentages/percentage points.\n'
        'Metrics score observed demand y in kW, not unknown latent f.\n'
        'The reported SD describes day-to-day variability, not a confidence\n'
        'interval or standard error. These nearby dates may be dependent.\n',encoding='utf-8')
    return data


def aggregate(frame,identity='date'):
    records=[]
    assert len(frame)==80 and not frame.duplicated([identity,'n_train','model']).any()
    for count in COUNTS:
        for model in MODELS:
            part=frame[(frame.n_train==count)&(frame.model==model)]
            assert len(part)==10
            row=dict(n_train=count,model=model,n_curves=10,n_test_per_curve=int(part.n_test.iloc[0]))
            for metric in METRICS:
                values=part[metric].to_numpy()
                assert np.isfinite(values).all()
                row[metric+'_mean']=float(values.mean())
                row[metric+'_std']=float(np.std(values,ddof=1))
            records.append(row)
    return pd.DataFrame(records)


def latex(summary,path,common=False):
    caption=(r'Electricity MT\_001: mean $\pm$ sample SD across ten dates (6--19 March 2014). '
        r'Each curve is fitted separately using only its 15 or 30 observed points. '
        r'No cross-curve parameter sharing. GHP fixes $\lambda=\omega=1$ and STP fixes $\nu=5$. '
        r'Plug-in functions and MAP hyperparameters; prediction integrates $W$. '
        r'Metrics evaluate observed $y$ in kW; RMSE is averaged per curve. ')
    caption+=(r'Both observation counts are scored on the same 66 held-out readings per day.' if common else
              r'Scores use 81 or 66 held-out readings per day, respectively.')
    lines=[r'\begin{table}[htbp]',r'\centering',r'\small',r'\caption{'+caption+'}',
        r'\label{tab:electricity_ten_individual'+('_common' if common else '')+'}',
        r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{rlrrrrr}',r'\toprule',
        r'Observed & Model & RMSE & NLPD & CRPS & 95\% coverage (\%) & Interval width \\',r'\midrule']
    for row in summary.to_dict('records'):
        cells=[str(row['n_train']),LABELS[row['model']]]
        for metric in METRICS:
            scale=100 if metric=='y_coverage_95' else 1
            digits=2 if metric=='y_coverage_95' else 5
            cells.append(f"${scale*row[metric+'_mean']:.{digits}f} \\pm {scale*row[metric+'_std']:.{digits}f}$")
        lines.append(' & '.join(cells)+r' \\')
        if row['n_train']==15 and row['model']==MODELS[-1]:lines.append(r'\midrule')
    lines += [r'\bottomrule',r'\end{tabular}}',r'\end{table}']
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def audit_case(curve,count,model,folder,score,common):
    case=folder/f'context_{count}'/model
    config=json.loads((case/'configuration.json').read_text())
    ids=np.sort(curve['context_order'][:count]);held=np.setdiff1d(np.arange(96),ids)
    np.testing.assert_array_equal(config['context_ids'],ids)
    scale=float(curve['y'][ids].std())
    np.testing.assert_allclose(config['response_scale'],scale,atol=1e-12)
    center=np.log([.5,max(float(np.mean((curve['y'][ids]/scale)**2)),.5),.01,1.,.04])
    np.testing.assert_allclose(config['population_prior_center'],center,atol=1e-12)
    fit=json.loads((case/'fit.json').read_text())
    assert fit['max_abs_gradient']<1e-3 and fit['max_hessian_eigenvalue']<0
    with np.load(case/'predictions.npz') as p:
        np.testing.assert_array_equal(p['test_ids'],held)
        for expected,selection in [(score,held),(common,np.setdiff1d(np.arange(96),curve['context_order'][:30]))]:
            pos=np.searchsorted(held,selection);truth=curve['y'][selection]
            ci=p['y_ci'][selection]
            actual=dict(y_rmse=np.sqrt(np.mean((truth-p['y_mean'][selection])**2)),
                y_nlpd=p['nlpd'][pos].mean(),y_crps=p['crps'][pos].mean(),
                y_coverage_95=np.mean((truth>=ci[:,0])&(truth<=ci[:,1])),
                y_interval_width=np.mean(ci[:,1]-ci[:,0]))
            for metric in METRICS:np.testing.assert_allclose(expected[metric],actual[metric],atol=1e-12)


def main():
    data=prepare();records=[];common_records=[];params=[]
    with PdfPages(OUTPUT/'regression.pdf') as pdf:
        for j,date in enumerate(data['dates']):
            print(f'Curve {j+1}/10: {date}',flush=True)
            curve=dict(x=data['x'],time_hours=data['time_hours'],y=data['y'][j],
                       date=date,context_order=data['context_orders'][j])
            folder=OUTPUT/f'day_{date}';folder.mkdir(exist_ok=True)
            if j==0:
                with np.load(SINGLE/'dataset.npz') as old:
                    for key in curve:np.testing.assert_array_equal(old[key],curve[key])
                for count in COUNTS:
                    for model in MODELS:
                        dest=folder/f'context_{count}'/model/'fit.json'
                        dest.parent.mkdir(parents=True,exist_ok=True)
                        if not dest.exists():dest.write_bytes((SINGLE/f'context_{count}'/model/'fit.json').read_bytes())
            for count in COUNTS:
                for model in MODELS:
                    score,common,parameter=run_case(curve,count,model,output=folder)
                    audit_case(curve,count,model,folder,score,common)
                    records.append(dict(date=str(date),**score));common_records.append(dict(date=str(date),**common))
                    params.append(dict(date=str(date),**parameter))
            pd.DataFrame(records).to_csv(OUTPUT/'per_curve_scores.csv',index=False)
            pd.DataFrame(common_records).to_csv(OUTPUT/'common_test_per_curve_scores.csv',index=False)
            pd.DataFrame(params).to_csv(OUTPUT/'parameters.csv',index=False)
            fig,axes=plt.subplots(4,2,figsize=(12,11),sharex=True,constrained_layout=True)
            for i,model in enumerate(MODELS):
                for col,count in enumerate(COUNTS):
                    with np.load(folder/f'context_{count}'/model/'predictions.npz') as p:
                        ax=axes[i,col];t=curve['time_hours'];ids=p['context_ids']
                        ax.plot(t,curve['y'],'k-',lw=1,label='Observed demand')
                        ax.plot(t,p['y_mean'],color='tab:blue',label='Predictive mean')
                        ax.fill_between(t,p['y_ci'][:,0],p['y_ci'][:,1],color='tab:blue',alpha=.2,label='95% predictive interval')
                        ax.scatter(t[ids],curve['y'][ids],s=15,color='tab:red',label='Training readings',zorder=5)
                        ax.set(title=f'{LABELS[model]}, {count} observations',ylabel='Demand (kW)',xlim=(0,24))
                        if i==3:ax.set_xlabel('Local time (hours)')
                        if i==0 and col==0:ax.legend(fontsize=6)
            fig.suptitle(f'MT_001: {date}; all parameters fitted from this curve only')
            pdf.savefig(fig);plt.close(fig)
    for frame,name,is_common in [(pd.DataFrame(records),'scores',False),(pd.DataFrame(common_records),'common_test_scores',True)]:
        summary=aggregate(frame)
        summary.to_csv(OUTPUT/(name+'.csv'),index=False)
        latex(summary,OUTPUT/('comparison_common_test.tex' if is_common else 'comparison.tex'),is_common)
        # Independent centered-sum check of sample variance and means.
        for row in summary.to_dict('records'):
            group=frame[(frame.n_train==row['n_train'])&(frame.model==row['model'])]
            for metric in METRICS:
                values=list(group[metric]);mean=sum(values)/10
                sd=(sum((x-mean)**2 for x in values)/9)**.5
                np.testing.assert_allclose([row[metric+'_mean'],row[metric+'_std']],[mean,sd],atol=1e-12)
        print(name+'\n'+summary.to_string(index=False),flush=True)
    write_json(OUTPUT/'validation.json',dict(all_checks_passed=True,n_curves=10,n_fits=80,
        same_source_dates_and_masks=True,context_only_scaling_and_prior=True,
        independent_prediction_checks_each_fit=True,recomputed_metrics_from_saved_predictions=True,
        arithmetic_means_and_sample_sd_checked=True,sd_ddof=1))


if __name__=='__main__':
    with threadpool_limits(limits=1):main()
