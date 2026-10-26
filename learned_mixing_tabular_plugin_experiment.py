"""Concrete/Power plug-in MAP, now learning GIG lambda/omega or STP nu.

Uses the saved data and exact original plug-in protocol. The existing GP is
copied as a reference; only STP-mu, GHP-mu and GHP-beta are refitted.
Run: python learned_mixing_tabular_plugin_experiment.py
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from GH_tabular_learned_plugin_map import MODELS, LearnedPluginTarget
from tabular_plugin_experiment import run_model, summarize, TEX_NAMES, OUTPUT as SOURCE
from shared_beta_plugin_experiment import write_json
from validate_tabular_plugin_experiment import validate

OUTPUT = Path('figure/tabular_single_function_learned_mixing_plugin')
DATASETS = ('concrete', 'power')


def exact_copy(source, destination):
    blob = source.read_bytes()
    if destination.exists() and destination.read_bytes() != blob:
        raise RuntimeError(f'Existing file differs from source: {destination}')
    if not destination.exists():
        destination.write_bytes(blob)


def prepare(dataset, output, starts):
    source, folder = SOURCE/dataset, output/dataset
    folder.mkdir(parents=True, exist_ok=True)
    exact_copy(source/'dataset.npz', folder/'dataset.npz')
    with np.load(folder/'dataset.npz') as archive:
        data = {key:archive[key].copy() for key in archive.files}
    old = json.loads((source/'experiment.json').read_text())
    config = dict(old)
    config.update(learn_mixing=True, starts=starts, optimization_seed=505,
        source_experiment=str(source),
        fixed_priors=dict(log_parameter_prior_sd=2.),
        mixing_hyperpriors=dict(lambda_normal_mean=1.,lambda_normal_sd=2.,
            log_omega_normal_mean=0.,log_omega_normal_sd=1.,
            log_nu_minus_two_normal_mean=float(np.log(3.)),log_nu_minus_two_normal_sd=1.),
        fitting='Joint MAP in log(theta),log(W),lambda,log(omega), or log(theta),log(W),log(nu-2); Gaussian functions integrated out',
        prediction='Fix fitted theta,mixing hyperparameters and conditional mean function; recompute and integrate W',
        gp_reference='Copied original GP fit/predictions, unchanged; GP has no mixing distribution',
        initialization='First start uses the corresponding saved fixed-mixing MAP; other starts include random perturbations and kernel exchanges')
    assert config['dataset_sha256'] == hashlib.sha256((folder/'dataset.npz').read_bytes()).hexdigest()
    assert len(data['train_ids']) == 300 and len(data['test_ids']) == old['n_test']
    assert not np.intersect1d(data['input_group_ids'][data['train_ids']], data['input_group_ids'][data['test_ids']]).size
    if (folder/'experiment.json').exists():
        if json.loads((folder/'experiment.json').read_text()) != config:
            raise RuntimeError('Existing experiment configuration differs')
    write_json(folder/'experiment.json', config)
    gp = folder/'gp'
    gp.mkdir(exist_ok=True)
    for name in ('fit.json','report.json','predictions.npz'):
        exact_copy(source/'gp'/name, gp/name)
    return folder, data, config


METHOD = '''CONCRETE/POWER: LEARNED MIXING IN THE ORIGINAL FIXED-FUNCTION PLUG-IN PROTOCOL

Data: exact saved seed-17026 splits and training-only standardization.
Concrete: 300 train, 705 test, eight features. Power: 300 train, 9227 test,
four features. Earlier duplicate removal and grouped identical-X splitting
are retained. Each dataset is one function with one global W.

Three new fits per dataset: STP-mu, GHP-mu and GHP-beta. Reuse the saved
GP-mu fit without changes as the reference; GP has no mixing parameters.
Fit the same two ARD SE kernel amplitudes, per-feature inverse squared
length scales, and noise coefficient, now also learning lambda/omega for
both GHP variants or nu>2 for STP. Same kernel priors as the originals.
Additional proper priors: lambda~Normal(1,2^2), log omega~Normal(0,1),
log(nu-2)~Normal(log 3,1). GIG uses chi=psi=omega, scale=1;
STP uses IG(nu/2,(nu-2)/2), with prior mean W=1.

During fitting, analytically integrate the Gaussian processes, then jointly
maximize the posterior in log theta, log W and the stated mixing coordinates.
This is JOINT MAP, matching the original Concrete/Power plug-in experiment.
It differs from the recent Airfoil marginal-MAP procedure that integrates W
during fitting and retains function uncertainty during prediction.
Include the log-W Jacobian, GIG/IG normalizers and mixing hyperpriors.
Use 24 starts (seed505); first start is the saved fixed-mixing fit, others
include default/swapped components and randomized states. Choose by training
posterior only; check stationarity, finite-difference gradients, negative
local Hessian and absence of active numerical bounds. No global guarantee.

At MAP, estimate one mu/beta function using its conditional Gaussian mean.
Hold theta, mixing hyperparameters and this function fixed. Recompute
p(W|training y,theta_hat,mu_hat/beta_hat,lambda_hat/omega_hat/nu_hat), then
integrate W in the fixed-function prediction formulas. Prediction W therefore
differs from the W point estimate used to obtain the function. Neither
hyperparameter nor mu/beta function uncertainty is propagated. This is an
empirical plug-in method, not full Bayes.

Observed-y metrics only: RMSE, pointwise NLPD, CRPS, equal-tail 95% coverage
and width, in original response units. No test targets enter fitting.
Predict in 64-row batches; save summaries, never dense test covariances.
Check densities, quantiles, mixing moments and CRPS independently. The final
validator checks raw cached data, original split, training scaling, file
hashes, fitted mixing parameters, independent conditional calculations and
metric aggregation. Comparison_with_fixed_mixing includes unchanged earlier
fixed-lambda=omega=1 / nu=5 baselines; primary comparison includes the learned
models and the unchanged GP reference. A single W cannot supply repeated
realizations to identify the mixing population; estimates depend on priors.
'''


def write_table(table, path, *, include_fixed=False):
    caption = (r'Concrete and Power single-function plug-in regression, split seed 17026: '
        r'300 training rows per dataset; 705 Concrete and 9,227 Power test rows. '
        r'All scores concern observed $y$ in original units. Learned rows jointly MAP-fit '
        r'kernel/noise parameters, $W$, and GHP $\lambda,\omega$ or STP $\nu$. '
        r'The conditional mean $\hat\mu$ or $\hat\beta$ is fixed for prediction; '
        r'$W$ is reconditioned and integrated. GP is the unchanged earlier reference.')
    if include_fixed:
        caption += r' Fixed rows use $\lambda=\omega=1$ or $\nu=5$.'
    lines = [r'\begin{table}[htbp]',r'\centering',r'\small',r'\caption{'+caption+'}',
        r'\label{tab:tabular_learned_mixing'+('_with_fixed' if include_fixed else '')+'}',
        r'\resizebox{\textwidth}{!}{%',r'\begin{tabular}{lllrrrrr}',r'\toprule',
        r'Dataset & Model & Mixing & RMSE & NLPD & CRPS & 95\% coverage (\%) & Width \\',r'\midrule']
    last_dataset = None
    for row in table.itertuples():
        if last_dataset is not None and row.dataset != last_dataset:
            lines.append(r'\midrule')
        last_dataset = row.dataset
        cells = [row.dataset.capitalize(),TEX_NAMES[row.model],row.mixing,
                 f'{row.y_rmse:.5f}',f'{row.y_nlpd:.5f}',f'{row.y_crps:.5f}',
                 f'{100*row.y_coverage_95:.2f}',f'{row.y_interval_width:.4f}']
        lines.append(' & '.join(cells)+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}}',r'\end{table}']
    path.write_text('\n'.join(lines)+'\n',encoding='utf-8')


def summarize_learned(output, datasets):
    # Reuse the existing saved-prediction plots and standard score aggregation.
    summarize(datasets, destination=output, output=output)
    primary = pd.read_csv(output/'scores.csv')
    primary['mixing'] = np.where(primary.model=='gp','N/A','Learned')
    write_table(primary,output/'comparison.tex')
    old = pd.read_csv(SOURCE/'scores.csv')
    old = old[old.dataset.isin(datasets) & old.model.isin(MODELS)].copy()
    old['mixing'] = 'Fixed'
    combined = pd.concat([primary, old],ignore_index=True)
    order = {'gp':0,'tp_mu_gp':1,'mu_gp':2,'beta_gp':3}
    combined['model_order'] = combined.model.map(order)
    combined = combined.sort_values(['dataset','model_order','mixing']).drop(columns='model_order')
    combined.to_csv(output/'scores_with_fixed_mixing.csv',index=False)
    write_table(combined,output/'comparison_with_fixed_mixing.tex',include_fixed=True)
    params, mixrows = [], []
    for dataset in datasets:
        folder = output/dataset
        with np.load(folder/'dataset.npz') as data:
            y_scale, x_scale = float(data['y_scale']),data['x_scale'].copy()
        for model in MODELS:
            fit=json.loads((folder/model/'fit.json').read_text())
            report=json.loads((folder/model/'report.json').read_text())
            mixing=fit['mixing_hyperparameters']
            values=np.exp(fit['log_parameters'])
            factors=np.r_[np.full(3,y_scale**2),1/x_scale**2,1/x_scale**2,1.]
            for name,value,factor in zip(fit['parameter_names'],values,factors):
                params.append(dict(dataset=dataset,model=model,parameter=name,
                    standardized_value=value,original_units_value=value*factor))
            for name,value in mixing.items():
                params.append(dict(dataset=dataset,model=model,parameter='lambda' if name=='lam' else name,
                    standardized_value=value,original_units_value=value))
            posterior=report['prediction_W_posterior']
            mixrows.append(dict(dataset=dataset,model=model,lambda_=mixing.get('lam'),
                omega=mixing.get('omega'),nu=mixing.get('nu'),W_joint_map=report['W_joint_map'],
                W_prediction_mean=posterior['W_mean'],W_prediction_std=np.sqrt(posterior['W_var']),
                W_prediction_variance=posterior['W_var']))
    pd.DataFrame(params).to_csv(output/'parameters.csv',index=False)
    pd.DataFrame(mixrows).to_csv(output/'mixing_parameters.csv',index=False)
    for dataset in datasets:
        write_table(primary[primary.dataset==dataset],output/dataset/'comparison.tex')
        write_table(combined[combined.dataset==dataset],output/dataset/'comparison_with_fixed_mixing.tex',include_fixed=True)
    print(combined.to_string(index=False),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    parser.add_argument('--dataset',choices=DATASETS)
    parser.add_argument('--starts',type=int,default=24)
    args=parser.parse_args()
    datasets=(args.dataset,) if args.dataset else DATASETS
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'method.txt').write_text(METHOD,encoding='utf-8')
    for dataset in datasets:
        folder,data,config=prepare(dataset,args.output,args.starts)
        train=data['train_ids']
        for model in MODELS:
            target=LearnedPluginTarget(data['y_scaled'][train],data['x_scaled'][train],model)
            old=json.loads((SOURCE/dataset/model/'fit.json').read_text())
            warm_start=np.r_[old['effective_log_parameters'],target.h0]
            del target
            run_model(folder,data,config,model,initial_points=[warm_start])
    summarize_learned(args.output,datasets)
    validate(datasets,destination=args.output,output=args.output)


if __name__=='__main__':
    with threadpool_limits(limits=1):
        main()
