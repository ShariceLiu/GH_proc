"""Reuse the 40-day electricity fits, revealing 45 readings per test day.

No training or hyperparameter optimization is performed. Both shared and
individual-function fits are evaluated; original results are left untouched.
"""
import os
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from threadpoolctl import threadpool_limits
import electricity_shared_plugin_experiment as shared
import electricity_independent_plugin_experiment as individual

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'figure/electricity_mt001_45_context_plugin'
MODELS = shared.MODELS
METRICS = shared.METRICS


def table(frame, path, caption):
    names = dict(gp=r'GP $\mu$', stp=r'STP $\mu$',
                 ghp_mu=r'GHP $\mu$', ghp_beta=r'GHP $\beta$')
    lines = [r'\begin{table}[htbp]', r'\centering\small',
             r'\caption{' + caption + '}', r'\begin{tabular}{rllrrrr}',
             r'\toprule',
             r'Observed & Function & Model & RMSE & NLPD & CRPS & Coverage (\%) \\',
             r'\midrule']
    for _, row in frame.iterrows():
        cells = []
        for metric in METRICS[:4]:
            factor = 100 if metric == 'y_coverage_95' else 1
            cells.append(f'${factor*row[metric+"_day_mean"]:.5f} \\pm '
                         f'{factor*row[metric+"_day_sd"]:.5f}$')
        lines.append(f'{int(row.context_count)} & {row.function} & '
                     + names[row.model] + ' & ' + ' & '.join(cells) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT/'figure/electricity_mt001_shared_plugin/dataset.npz'
    with np.load(source) as z:
        data = {key:z[key] for key in z.files}
    summaries, common_summaries, all_scores, all_common = [], [], [], []
    manifest = dict(dataset_sha256=shared.digest(source), context_count=45,
                    held_out_count=51, n_training_days=40, n_test_days=18,
                    fitting='Reuse saved fits without retraining.',
                    contexts='First 45 entries of saved per-day permutations; nested in 15/30.',
                    uncertainty='Plug-in functions and parameters; integrate test-day W.',
                    GHP_lambda=1, GHP_omega=1, STP_nu=5, fits={})
    for kind, module in [('shared', shared), ('individual', individual)]:
        source_kind = 'independent' if kind == 'individual' else 'shared'
        baseline = ROOT/f'figure/electricity_mt001_{source_kind}_plugin'
        destination = OUT/kind
        destination.mkdir(exist_ok=True)
        module.OUTPUT = destination
        module.COUNTS = (45,)
        frames, common = [], []
        for model in MODELS:
            folder = destination/model
            folder.mkdir(exist_ok=True)
            fit_path = baseline/model/'fit.json'
            fit_hash = shared.digest(fit_path)
            shutil.copyfile(fit_path, folder/'fit.json')
            fit = json.loads(fit_path.read_text())
            assert fit['dataset_sha256'] == manifest['dataset_sha256']
            module.predict_model(data, model, fit)
            assert shared.digest(fit_path) == fit_hash
            manifest['fits'][kind+'/'+model] = dict(source=str(fit_path), sha256=fit_hash)
            frames.append(pd.read_csv(folder/'per_day_scores.csv'))
            # Compare all context sizes on exactly the same 51 held-out readings.
            for count in (15,30,45):
                parent = destination if count == 45 else baseline
                for j, row in enumerate(data['test_rows']):
                    path = parent/model/f'context_{count}'/f'day_{data["dates"][row]}.npz'
                    with np.load(path) as p:
                        held = np.setdiff1d(np.arange(96), data['context_orders'][j,:45])
                        ids = np.sort(data['context_orders'][j,:count])
                        np.testing.assert_array_equal(p['context_ids'], ids)
                        assert len(held)==51 and not np.intersect1d(held,ids).size
                        pos = np.searchsorted(p['test_ids'], held)
                        np.testing.assert_array_equal(p['test_ids'][pos], held)
                        truth = data['y'][row,held]
                        ci = p['y_ci'][held]
                        assert np.isfinite(ci).all() and np.all(ci[:,1]>=ci[:,0])
                        common.append(dict(model=model, context_count=count,
                            date=str(data['dates'][row]), n_test=51,
                            y_rmse=float(np.sqrt(np.mean((truth-p['y_mean'][held])**2))),
                            y_nlpd=float(p['y_nlpd'][pos].mean()),
                            y_crps=float(p['y_crps'][pos].mean()),
                            y_coverage_95=float(np.mean((truth>=ci[:,0])&(truth<=ci[:,1]))),
                            y_interval_width=float(np.mean(ci[:,1]-ci[:,0]))))
        scores = pd.concat(frames, ignore_index=True)
        common = pd.DataFrame(common)
        check = scores.merge(common[common.context_count==45], on=['model','context_count','date'], suffixes=('_saved','_checked'))
        assert len(check)==72
        for metric in METRICS:
            np.testing.assert_allclose(check[metric+'_saved'], check[metric+'_checked'], atol=1e-10)
        for frame, collection, aggregate_collection in [(scores,all_scores,summaries),(common,all_common,common_summaries)]:
            frame['function']=kind
            collection.append(frame)
            summary=shared.aggregate(frame)
            summary['function']=kind
            aggregate_collection.append(summary)
        with PdfPages(destination/'regression.pdf') as pdf:
            for row in data['test_rows']:
                fig, axes = plt.subplots(2,2,figsize=(12,7),sharex=True,sharey=True)
                for ax,model in zip(axes.flat,MODELS):
                    with np.load(destination/model/'context_45'/f'day_{data["dates"][row]}.npz') as p:
                        t=data['time_hours']; ids=p['context_ids']
                        ax.fill_between(t,*p['y_ci'].T,alpha=.25,label='95% observation interval')
                        ax.plot(t,p['y_mean'],label='Prediction')
                        ax.plot(t,data['y'][row],'k--',lw=1,label='Recorded demand')
                        ax.scatter(t[ids],data['y'][row,ids],s=12,c='black',label='45 revealed readings')
                        ax.set(title=model,xlabel='Time (hours)',ylabel='Demand (kW)')
                axes[0,0].legend(fontsize=7)
                fig.suptitle(f'{kind} function: {data["dates"][row]}')
                fig.tight_layout();pdf.savefig(fig);plt.close(fig)
    summary=pd.concat(summaries,ignore_index=True)
    common_summary=pd.concat(common_summaries,ignore_index=True)
    summary.to_csv(OUT/'scores.csv',index=False)
    common_summary.to_csv(OUT/'common_test_scores.csv',index=False)
    pd.concat(all_scores).to_csv(OUT/'per_day_scores.csv',index=False)
    pd.concat(all_common).to_csv(OUT/'common_test_per_day_scores.csv',index=False)
    caption = ('Electricity MT\\_001: unchanged parameters learned from 40 complete training days; '
               '18 test days with 45 revealed readings and 51 held-out readings each. '
               'Mean $\\pm$ sample SD of per-day scores; plug-in functions. '
               'GIG $\\lambda=\\omega=1$ and STP $\\nu=5$ remain fixed.')
    table(summary,OUT/'comparison.tex',caption)
    table(common_summary,OUT/'comparison_common_test.tex',
          'Electricity: 15, 30 and 45 revealed readings, scored on the same 51 held-out readings per day. '
          'Unchanged 40-day training fits. Mean $\\pm$ sample SD across 18 days.')
    shared.write_json(OUT/'experiment.json',manifest)
    shared.write_json(OUT/'validation.json',dict(passed=True,predictions=144,
        checks=['Saved fit and dataset hashes','Nested contexts and held-out separation',
                'Independent aggregation from point scores','Recomputed RMSE and coverage',
                'Predictive density checks on first test day per model and function type']))
    print(summary[['function','model',*[m+'_day_mean' for m in METRICS]]].to_string(index=False))


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
