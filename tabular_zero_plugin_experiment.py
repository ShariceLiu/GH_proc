"""Add zero-function MAP GP/STP/GHP to the exact saved tabular split.

Run: python tabular_zero_plugin_experiment.py --dataset wine
There are three distinct baselines: the two GHP variants coincide at mu=beta=0.
"""
import os
for variable in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[variable] = '1'
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from GH_tabular_hmc import MODELS, ZERO_MODELS, default_center
from shared_beta_plugin_experiment import write_json
from tabular_plugin_experiment import OUTPUT, run_model, TEX_NAMES
from validate_tabular_plugin_experiment import validate

NAMES = dict(gp_zero='GP', stp_zero='STP', ghp_zero='GHP')
ZERO_TEX = {model: name+r' ($\mu=\beta=0$)' for model,name in NAMES.items()}
METRICS = ('y_rmse','y_nlpd','y_crps','y_coverage_95','y_interval_width')


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_provenance(root):
    saved = json.loads((root/'zero_mean_baselines'/'source_provenance.json').read_text())
    for name,expected in saved.items():
        if digest(root/name) != expected:
            raise ValueError(f'Original data or result changed: {name}')


def prepare(dataset, *, output=OUTPUT):
    root = Path(output)/dataset
    original = json.loads((root/'experiment.json').read_text())
    with np.load(root/'dataset.npz') as saved:
        data = {k:saved[k].copy() for k in saved.files}
    assert digest(root/'dataset.npz') == original['dataset_sha256']
    folder = root/'zero_mean_baselines'
    folder.mkdir(exist_ok=True)
    files = ['dataset.npz','experiment.json','scores.csv','comparison.tex','validation.json']
    files += [f'{model}/{name}' for model in MODELS for name in ('fit.json','report.json','predictions.npz')]
    hashes = {name:digest(root/name) for name in files}
    if not (folder/'source_provenance.json').exists():
        write_json(folder/'source_provenance.json',hashes)
    check_provenance(root)
    config = dict(original)
    config.update(models=list(ZERO_MODELS),
        equivalent_original_models={'gp':'gp_zero','tp_mu_gp':'stp_zero','mu_gp':'ghp_zero','beta_gp':'ghp_zero'},
        source_experiment_sha256=hashes['experiment.json'], dataset_file='../dataset.npz',
        kernel='One residual ARD SE; no mu/beta GP or second kernel.',
        fixed_functions='mu=beta=0 on standardized response; retain the original training-y-mean intercept.',
        prior_center=default_center(data['y_scaled'][data['train_ids']],data['x'].shape[1],'gp_zero').tolist(),
        prior_convention='Keep original residual-kernel and noise priors; omit second-kernel priors.',
        fitting=f'Joint log-coordinate MAP in theta and W (GP fixes W=1); {original["starts"]} starts, no component swaps.',
        prediction='Freeze theta, keep mu=beta=0, integrate p(W|training y,theta_hat) from the original mixing prior.')
    path = folder/'experiment.json'
    if path.exists():
        assert json.loads(path.read_text()) == config
    else:
        write_json(path,config)
    method = METHOD.replace('Use 24 starts', f'Use {original["starts"]} starts')
    (folder/'method.txt').write_text(method+f'\nOriginal-unit fixed intercept: {float(data["y_mean"]):.12g}\n',encoding='utf-8')
    print(f'{dataset}: reusing {len(data["train_ids"])} train / {len(data["test_ids"])} test; '
          f'zero functions on centered scale, original-unit intercept {float(data["y_mean"]):.6f}',flush=True)
    return root,folder,data,config


def table(path, rows, caption, label):
    lines = [r'\begin{table}[htbp]',r'\centering',r'\caption{'+caption+'}',
             r'\label{tab:'+label+'}',r'\begin{tabular}{lrrrr}',r'\toprule',
             r'Model & RMSE & NLPD & CRPS & 95\% coverage (\%) \\',r'\midrule']
    for index,row in enumerate(rows):
        if index and row['model'] in ZERO_MODELS and rows[index-1]['model'] not in ZERO_MODELS:
            lines.append(r'\midrule')
        name = ZERO_TEX[row['model']] if row['model'] in ZERO_MODELS else TEX_NAMES[row['model']]
        lines.append(' & '.join([name,f'{row["y_rmse"]:.5f}',f'{row["y_nlpd"]:.5f}',
                                f'{row["y_crps"]:.5f}',f'{100*row["y_coverage_95"]:.2f}'])+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def summarize(root,folder,data,config):
    check_provenance(root)
    old = [json.loads((root/m/'report.json').read_text()) for m in MODELS]
    new = [json.loads((folder/m/'report.json').read_text()) for m in ZERO_MODELS]
    for report in old+new:
        assert report['dataset_sha256'] == config['dataset_sha256']
    keys = ['dataset','model','n_train','n_test',*METRICS]
    pd.DataFrame([{k:r[k] for k in keys} for r in new]).to_csv(folder/'scores.csv',index=False)
    pd.DataFrame([{k:r[k] for k in keys} for r in old+new]).to_csv(root/'scores_with_zero_mean.csv',index=False)
    caption = (f'{config["dataset"].capitalize()} regression on the same {config["n_train"]}/{config["n_test"]} '
               f'train/test split (seed {config["split_seed"]}). '
               r'Scores evaluate observed $y$ in original units. Zero functions apply after training-mean centering; '
               r'the original-unit intercept is retained. Zero-function models use one SE kernel, while the learned '
               r'$\mu/\beta$ models use two. Hyperparameters are MAP estimates and mixing uncertainty is integrated. '
               r'GHP fixes $\lambda=\omega=1$; STP fixes $\nu=5$. The two GHP variants coincide when $\mu=\beta=0$.')
    table(folder/'comparison.tex',new,caption,config['dataset']+'_zero_plugin')
    table(root/'comparison_with_zero_mean.tex',old+new,caption,config['dataset']+'_plugin_with_zero')
    truth = data['y'][data['test_ids']]
    fig,axes = plt.subplots(2,3,figsize=(12,7))
    for j,model in enumerate(ZERO_MODELS):
        with np.load(folder/model/'predictions.npz') as p:
            axes[0,j].scatter(truth,p['y_mean'],s=4,alpha=.4)
            bounds = [min(truth.min(),p['y_mean'].min()),max(truth.max(),p['y_mean'].max())]
            axes[0,j].plot(bounds,bounds,'k--',lw=1)
            axes[0,j].set(title=NAMES[model]+' (zero functions)',xlabel='Observed target',ylabel='Predicted mean')
            axes[1,j].hist(p['pit'],bins=np.linspace(0,1,11),density=True,alpha=.75)
            axes[1,j].axhline(1,color='k',ls='--',lw=1)
            axes[1,j].set(xlabel='Predictive CDF at observation',ylabel='Density',xlim=(0,1))
    fig.suptitle(config['dataset'].capitalize()+': zero functions on centered response, one SE kernel')
    fig.tight_layout(); fig.savefig(folder/'regression.png',dpi=150); fig.savefig(folder/'regression.pdf'); plt.close(fig)
    print(pd.DataFrame([{k:r[k] for k in keys} for r in new]).to_string(index=False),flush=True)


def audit(root,folder,config):
    check_provenance(root)
    validate((config['dataset'],),destination=folder,models=ZERO_MODELS,subfolder='zero_mean_baselines',output=root.parent)
    combined = pd.read_csv(root/'scores_with_zero_mean.csv').set_index('model')
    assert set(combined.index) == set(MODELS+ZERO_MODELS)
    for model in MODELS+ZERO_MODELS:
        path = folder/model if model in ZERO_MODELS else root/model
        report = json.loads((path/'report.json').read_text())
        for metric in METRICS:
            np.testing.assert_allclose(combined.loc[model,metric],report[metric],atol=1e-12)
    check_provenance(root)
    print('Original four-model results unchanged; combined seven-model table checked.',flush=True)


METHOD = '''ZERO-FUNCTION TABULAR GP/STP/GHP: SAME SAVED SPLIT

Use the parent dataset.npz, without resplitting, rescaling or copying the data.
Source hashes preserve the original four fits, predictions, scores and table.
The source experiment's removal of exact duplicates, equal-input grouping and
training-only standardization remain in effect. Source settings and counts
are in experiment.json. This is one split, not a repeated-split estimate.

Set mu(x)=beta(x)=0 for the standardized response. The original-unit intercept
is still the training y mean. This experiment does not force the original-unit
prior mean of wine quality to zero. GP regression can predict nonzero centered
values by conditioning the remaining residual process on the observations.

Keep one ARD SE residual kernel K and noise k; remove the latent mu/beta GP
and all its hyperparameters. K(x,x')=v exp(-0.5 sum_d w_d (x_d-x'_d)^2).
GP: y|theta ~ N(0,K+kI), W=1.
STP: y|W,theta ~ N(0,W(K+kI)), W~IG(2.5,1.5), fixed nu=5.
GHP: same conditional Gaussian, W~GIG(lambda=1,chi=psi=1).
The earlier GHP-mu and GHP-beta families are exactly the SAME model under
these restrictions, so fit a single GHP-zero baseline. There are three unique
zero-function fits, not four independent models. No function is estimated.

Refit residual amplitude, all ARD inverse squared length scales, noise, and
training W (STP/GHP), using joint MAP in log coordinates. Preserve the previous
residual/noise log-Normal priors: Gaussian priors on log parameters, SD=2,
center log([0.5*s,0.1*s,1/D,...,1/D]), s=var(standardized training y).
Use 24 starts, seed 505; no kernel-component swaps for a single kernel.
Include the log-W Jacobian; check stationary gradients and negative local
curvature. This is a multistart local search, not proof of global optimality.

For prediction, fix theta_hat but INTEGRATE W. No point estimate of W is
substituted into predictive variances. With A=K_train+kI and q=y^T A^-1 y:
GHP: W|y,theta_hat ~ GIG(1-n/2,chi=1+q,psi=1).
STP: W|y,theta_hat ~ IG((5+n)/2,(3+q)/2).
At a test point, m=K_*train A^-1 y and S=K_**-K_*train A^-1 K_train*.
The latent posterior mean is m and variance E[W] S. Observation variance
adds E[W] k. There is no Var(W)*beta_effect^2 contribution because beta=0.
Hyperparameter uncertainty is omitted; latent residual-process uncertainty
and mixing uncertainty are retained. The joint MAP W is saved for reference.

Scores: observed-y RMSE, marginal NLPD, CRPS, equal-tail 95% observation
coverage and width in original units. No true latent f is available. Wine
quality is treated as continuous; NLPD scores a density, not integer mass.
Predict in 64-point batches, saving compact pointwise summaries, no dense
test covariance or particles. Verify conditioning, mixing moments, densities,
quantiles and scores independently, and check unchanged original file hashes.

Run: python tabular_zero_plugin_experiment.py --dataset wine
Resume one model: add --stage fit --model gp_zero (or stp_zero, ghp_zero)
Summarize: add --stage summarize
Validate: add --stage validate
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',choices=('wine','concrete','power'),default='wine')
    parser.add_argument('--model',choices=ZERO_MODELS)
    parser.add_argument('--stage',choices=('all','prepare','fit','summarize','validate'),default='all')
    args = parser.parse_args()
    root,folder,data,config = prepare(args.dataset)
    if args.stage in ('all','fit'):
        for model in (ZERO_MODELS if args.model is None else (args.model,)):
            run_model(folder,data,config,model)
    if args.stage in ('all','summarize'):
        summarize(root,folder,data,config)
    if args.stage in ('all','validate'):
        audit(root,folder,config)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
