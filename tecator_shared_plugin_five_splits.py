"""Five independent train/test repartitions for shared-function plug-in models.

Run: python tecator_shared_plugin_five_splits.py --workers 2
Resume one split: python tecator_shared_plugin_five_splits.py --seed 24026
No HMC. Every split refits all four models using only its training curves.
"""
import os
for variable in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import json
from pathlib import Path
import time
import traceback

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from GH_shared_function_map import MODELS
from shared_beta_plugin_experiment import write_json, LABELS
from tecator_shared_plugin_experiment import SOURCE, load_data, fit_all, check_alternative_lengths, predict_all
from validate_tecator_shared_plugin import validate

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/SOURCE
OUTPUT = ROOT/'figure'/'tecator_shared_plugin_five_splits'
SEEDS = (24026,24027,24028,24029,24030)
CONTEXT_SEED = 34026
METRICS = ('y_rmse','y_nlpd','y_crps','y_coverage_95','y_interval_width','y_mean_abs_skewness')
TEX_NAMES = dict(gp=r'GP (shared $\hat\mu$)',stp=r'STP (shared $\hat\mu$)',
                 ghp_mu=r'GHP (shared $\hat\mu$)',ghp_beta=r'GHP (shared $\hat\beta$)')


def build_split(seed):
    original = load_data(SOURCE)
    pool = np.sort(np.r_[original['train_rows'],original['test_rows']])
    assert len(pool) == len(np.unique(pool)) == 178
    roles = pd.read_csv(SOURCE/'sample_roles.csv')
    selected = roles.iloc[pool]
    assert not selected[['moisture_pct','fat_pct','protein_pct']].duplicated().any()
    assert not selected.spectrum_group_id.duplicated().any()
    assert not selected.benchmark_split.eq('extrapolation').any()
    permuted = np.random.default_rng(seed).permutation(pool)
    train, test = np.sort(permuted[:147]), np.sort(permuted[147:])
    # The same spectrum gets the same observed wavelengths in every split.
    contexts = np.stack([np.sort(np.random.default_rng(np.random.SeedSequence([CONTEXT_SEED,int(row)]))
                                 .choice(100,15,replace=False)) for row in test])
    held = np.stack([np.setdiff1d(np.arange(100),ids) for ids in contexts])
    y = original['y']; mean = y[train].mean(axis=0)
    scale = float(np.std(y[train]-mean))
    data = dict(wavelength_nm=original['wavelength_nm'],x=original['x'],y=y,
                common_mean=mean,common_scale=np.asarray(scale),residuals=(y-mean)/scale,
                train_rows=train,test_rows=test,context_ids=contexts,evaluation_ids=held)
    Y = y[train]/scale
    center = np.log([.5,max(float(np.mean(Y.mean(axis=0)**2)),.5),.01,1.,.04])
    settings = dict(split_seed=seed,context_seed=CONTEXT_SEED,eligible_spectra=178,
                    n_train=147,n_test=31,training_points=100,context_points=15,test_points=85,
                    source_sha256=hashlib.sha256((SOURCE/'dataset.npz').read_bytes()).hexdigest(),
                    roles_sha256=hashlib.sha256((SOURCE/'sample_roles.csv').read_bytes()).hexdigest(),
                    split='Random repartition of the 178 representatives in the earlier train/test pool; E1/E2 excluded.',
                    grouping='Same first representative per composition triple as the earlier experiment; not verified specimen IDs.',
                    preprocessing='No centering; common scale and prior center calculated from training spectra only.',
                    prior_center=center,prior_sd=2.,GHP_lambda=1.,GHP_omega=1.,STP_nu=5.,
                    optimization_seed=24026,initial_starts=8,additional_shared_w_starts=[1,4,16,64])
    roles['experiment_role'] = 'excluded'
    roles.loc[train,'experiment_role']='train'; roles.loc[test,'experiment_role']='test'
    return data,center,settings,roles


def prepare(seed):
    folder = OUTPUT/f'split_{seed}'; folder.mkdir(parents=True,exist_ok=True)
    data,center,settings,roles = build_split(seed)
    if (folder/'dataset.npz').exists():
        saved = load_data(folder)
        for key in data: np.testing.assert_array_equal(saved[key],data[key])
        old = json.loads((folder/'experiment.json').read_text())
        normalized = json.loads(json.dumps(settings,default=lambda a:a.tolist()))
        if old != normalized: raise ValueError('Saved experiment settings changed')
    else:
        np.savez_compressed(folder/'dataset.npz',**data)
        write_json(folder/'experiment.json',settings)
        roles.to_csv(folder/'sample_roles.csv',index=False)
    return folder,data,center


def summarize_split(folder,data,seed):
    curves=pd.read_csv(folder/'per_curve_scores.csv')
    rows=[]
    for model in MODELS:
        frame=curves[curves.model==model]
        assert len(frame)==31
        row=dict(split_seed=seed,model=model,y_rmse=float(np.sqrt(np.mean(frame.y_rmse**2))))
        row.update({key:float(frame[key].mean()) for key in METRICS[1:]})
        rows.append(row)
    scores=pd.DataFrame(rows);scores.to_csv(folder/'scores.csv',index=False)
    fits={m:json.loads((folder/m/'fit.json').read_text()) for m in MODELS}
    wavelength=data['wavelength_nm']
    fig,ax=plt.subplots(figsize=(9,4))
    ax.plot(wavelength,data['common_mean'],'k--',label='Empirical training mean')
    for m in MODELS: ax.plot(wavelength,fits[m]['population_mean_original'],label=LABELS[m])
    ax.set(xlabel='Wavelength (nm)',ylabel='Absorbance',title=f'Estimated population means: split {seed}')
    ax.legend();fig.tight_layout();fig.savefig(folder/'shared_functions.png',dpi=150);plt.close(fig)
    with PdfPages(folder/'regression.pdf') as pdf:
        for start in range(0,len(data['test_rows']),5):
            subset=data['test_rows'][start:start+5]
            fig,axes=plt.subplots(len(subset),4,figsize=(16,2.5*len(subset)),squeeze=False,sharex=True,sharey='row')
            for i,row in enumerate(subset):
                for j,m in enumerate(MODELS):
                    with np.load(folder/m/f'curve_{row+1:03d}.npz') as p:
                        ax=axes[i,j];ids=p['context_ids']
                        ax.fill_between(wavelength,*p['y_ci'].T,color=f'C{j}',alpha=.25)
                        ax.plot(wavelength,p['y_mean'],color=f'C{j}',lw=1.2)
                        ax.plot(wavelength,data['y'][row],'k--',lw=1)
                        ax.scatter(wavelength[ids],data['y'][row,ids],c='black',s=10)
                        ax.set_title(f'{LABELS[m]}, sample {row+1}',fontsize=9)
                        if j==0: ax.set_ylabel('Absorbance')
                        if i==len(subset)-1: ax.set_xlabel('Wavelength (nm)')
            fig.suptitle(f'Split {seed}: dashed measured spectrum, dots revealed, 95% observation intervals')
            fig.tight_layout();pdf.savefig(fig)
            if start==0:fig.savefig(folder/'regression.png',dpi=120)
            plt.close(fig)
    return scores


def run_split(seed):
    folder,data,center=prepare(seed)
    with (folder/'run.log').open('a',encoding='utf-8',buffering=1) as log,redirect_stdout(log),redirect_stderr(log),threadpool_limits(1):
        try:
            started=time.perf_counter()
            print(f'Starting/resuming split {seed}',flush=True)
            fit_all(folder,data,center,8)
            check_alternative_lengths(folder,data,center)
            fit_seconds=time.perf_counter()-started
            predicted=time.perf_counter();predict_all(folder,data)
            prediction_seconds=time.perf_counter()-predicted
            summarize_split(folder,data,seed)
            validate(folder,expected_data=build_split(seed)[0])
            timings=dict(split_seed=seed,fitting_seconds=fit_seconds,prediction_seconds=prediction_seconds,
                         total_seconds=time.perf_counter()-started,
                         note='Wall time in this invocation; resumed stages may reuse saved results.')
            write_json(folder/'timing_latest.json',timings)
            print('COMPLETE',seed,flush=True)
        except Exception:
            traceback.print_exc();raise
    return seed


def aggregate():
    frames=[]
    for seed in SEEDS:
        folder=OUTPUT/f'split_{seed}'
        validation=json.loads((folder/'validation.json').read_text())
        assert validation['all_checks_passed']
        frames.append(pd.read_csv(folder/'scores.csv'))
    scores=pd.concat(frames,ignore_index=True)
    assert len(scores)==20 and not scores[['split_seed','model']].duplicated().any()
    scores.to_csv(OUTPUT/'split_scores.csv',index=False)
    records=[]
    for model in MODELS:
        frame=scores[scores.model==model]
        row=dict(model=model,n_splits=5)
        for metric in METRICS:
            row[f'{metric}_mean']=float(frame[metric].mean())
            row[f'{metric}_std']=float(frame[metric].std(ddof=1))
        records.append(row)
    summary=pd.DataFrame(records);summary.to_csv(OUTPUT/'summary.csv',index=False)
    lines=[r'\begin{table}[htbp]',r'\centering',
           r'\caption{Tecator shared-function plug-in regression: mean $\pm$ sample standard deviation over five random train/test splits (147/31 spectra). Each test spectrum reveals 15 wavelengths and scores the remaining 85. Scores refer to observed absorbance $y$ in original units.}',
           r'\label{tab:tecator_shared_five_splits}',r'\begin{tabular}{lrrrr}',r'\toprule',
           r'Model & RMSE & NLPD & CRPS & 95\% coverage (\%) \\',r'\midrule']
    display=[]
    for r in records:
        cells=[];d={'Model':LABELS[r['model']]}
        for key,label,scale in [('y_rmse','RMSE',1),('y_nlpd','NLPD',1),('y_crps','CRPS',1),('y_coverage_95','Coverage (%)',100)]:
            precision=2 if scale==100 else 5
            mean,std=r[key+'_mean']*scale,r[key+'_std']*scale
            cells.append(f'${mean:.{precision}f} \\pm {std:.{precision}f}$')
            d[label]=f'{mean:.{precision}f} +/- {std:.{precision}f}'
        lines.append(TEX_NAMES[r['model']]+' & '+' & '.join(cells)+r' \\');display.append(d)
    lines.extend([r'\bottomrule',r'\end{tabular}',r'\end{table}'])
    (OUTPUT/'comparison.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    text=pd.DataFrame(display).to_string(index=False)
    (OUTPUT/'results.txt').write_text(text+'\n\nStandard deviation uses ddof=1 across five split-level scores, not across wavelengths or posterior samples.\nThese are repeated holdout splits, not disjoint cross-validation folds.\n',encoding='utf-8')
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,key in zip(axes,('y_rmse','y_nlpd','y_crps')):
        for j,m in enumerate(MODELS):
            values=scores[scores.model==m][key].to_numpy()
            ax.scatter(np.full(5,j),values,s=25,alpha=.8)
            ax.errorbar(j+.12,values.mean(),yerr=values.std(ddof=1),fmt='k_',capsize=4)
        ax.set_xticks(range(4),['GP mu','STP mu','GHP mu','GHP beta'],rotation=25)
        ax.set_title(key.removeprefix('y_').upper());ax.grid(axis='y',alpha=.2)
    fig.suptitle('Five train/test splits: points are split scores; bars show mean +/- SD')
    fig.tight_layout();fig.savefig(OUTPUT/'comparison.png',dpi=160);fig.savefig(OUTPUT/'comparison.pdf');plt.close(fig)
    write_json(OUTPUT/'validation.json',dict(all_checks_passed=True,splits=5,models=4,
               test_curve_predictions=5*4*31,held_out_points_per_model=5*31*85,std_ddof=1))
    print(text,flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers',type=int,default=2)
    parser.add_argument('--seed',type=int,choices=SEEDS)
    parser.add_argument('--stage',choices=('all','prepare','summarize'),default='all')
    args=parser.parse_args();OUTPUT.mkdir(parents=True,exist_ok=True)
    if args.workers<1:raise ValueError('workers must be positive')
    seeds=(args.seed,) if args.seed is not None else SEEDS
    for seed in seeds:prepare(seed)
    if args.stage=='prepare':return
    if args.stage=='all':
        if args.seed is not None or args.workers==1:
            for seed in seeds:run_split(seed)
        else:
            errors=[]
            with ProcessPoolExecutor(max_workers=args.workers) as pool:
                futures={pool.submit(run_split,seed):seed for seed in seeds}
                for future in as_completed(futures):
                    seed=futures[future]
                    try:future.result();print(f'Split {seed} complete',flush=True)
                    except Exception as error:
                        errors.append(seed);print(f'Split {seed} failed: {error}',flush=True)
            if errors:raise RuntimeError(f'Incomplete splits: {errors}; see run.log and resume with --seed')
    if all((OUTPUT/f'split_{seed}'/'validation.json').exists() for seed in SEEDS):aggregate()


if __name__=='__main__':
    main()
