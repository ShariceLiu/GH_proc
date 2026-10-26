"""Wine: 60% training, seven distinct MAP plug-in/zero-function models.

Run: python wine_train60_plugin_experiment.py
Resume: rerun the same command; completed starts and models are reused.
"""
import argparse
import hashlib
import json
from pathlib import Path

from tabular_plugin_experiment import (OUTPUT as OLD_OUTPUT, prepare, run_model,
                                       summarize, TEX_NAMES)
import numpy as np
import pandas as pd
from scipy import optimize, linalg, stats
from threadpoolctl import threadpool_limits
from GH_tabular_hmc import ALL_MODELS, MODELS, ZERO_MODELS, coordinates, has_mixing, default_center
from shared_beta_plugin_experiment import write_json
from validate_tabular_plugin_experiment import validate
from GH_tabular_plugin_map import se_kernel
from predictive_crps import gaussian_mixture_crps_closed_form
import tabular_zero_plugin_experiment as zero

OUTPUT = OLD_OUTPUT.parent/'tabular_single_function_plugin_train60'
STARTS = 6


def source_case(root, model):
    return root/'zero_mean_baselines'/model if model in ZERO_MODELS else root/model


def warm_starts(old, new, model, fit):
    """Preserve original-unit covariance when changing training standardization."""
    q = np.asarray(fit['log_parameters']).copy()
    d = new['x'].shape[1]
    amplitude_shift = 2*np.log(old['y_scale']/new['y_scale'])
    length_shift = 2*np.log(new['x_scale']/old['x_scale'])
    if model in ZERO_MODELS:
        q[:2] += amplitude_shift
        q[2:2+d] += length_shift
    else:
        q[:3] += amplitude_shift
        q[3:3+d] += length_shift
        q[3+d:3+2*d] += length_shift
    z = coordinates(q, model)
    points = [z]
    if model not in ZERO_MODELS:
        swapped = z.copy()
        swapped[[0,1]] = z[[1,0]]
        swapped[3:3+d], swapped[3+d:3+2*d] = z[3+d:3+2*d], z[3:3+d]
        points.append(swapped)
    return points


def prepare_run():
    old_root = OLD_OUTPUT/'wine'
    with np.load(old_root/'dataset.npz') as archive:
        old = {key:archive[key] for key in archive.files}
    n_train = int(.6*len(old['y']))
    root,data,config = prepare('wine', n_train=n_train, output=OUTPUT,
                               starts=STARTS, numerical_backend='condensed')
    for key in ('x','y','source_row_ids'):
        np.testing.assert_array_equal(old[key],data[key])
    assert np.isin(old['train_ids'],data['train_ids']).all()
    assert np.isin(data['test_ids'],old['test_ids']).all()
    paths = ['dataset.npz','experiment.json','scores_with_zero_mean.csv','comparison_with_zero_mean.tex']
    for model in ALL_MODELS:
        paths += [str((source_case(old_root,model)/name).relative_to(old_root))
                  for name in ('fit.json','predictions.npz','report.json')]
    hashes = {name:hashlib.sha256((old_root/name).read_bytes()).hexdigest() for name in paths}
    provenance = root/'previous_results_sha256.json'
    if provenance.exists():
        assert json.loads(provenance.read_text()) == hashes
    else:
        write_json(provenance,hashes)
    points = {}
    for model in ALL_MODELS:
        fit = json.loads((source_case(old_root,model)/'fit.json').read_text())
        points[model] = warm_starts(old,data,model,fit)
    path = root/'initial_points.json'
    if path.exists():
        previous = json.loads(path.read_text())
        for model in ALL_MODELS:
            np.testing.assert_array_equal(previous[model],points[model])
    else:
        write_json(path,points)
    note = f'''
60% TRAINING EXTENSION
This run has {len(data['train_ids'])} training and {len(data['test_ids'])} test rows.
Seed 17026 extends the old 300-row training set; new test rows are a subset
of the old test set. All standardization and priors use the NEW training set.
Six full-data optimization starts per model replace the previous 24 starts.
For GP and GP-zero, start 1 uses the old corresponding fit, rescaled to the
new training units; that old fit used only a subset of the new training set.
GP start 2 exchanges effective kernel components. Remaining GP starts use
seeded random initializations. The old-fit candidates are in initial_points.json.
STP/GHP reuse the six optimized effective covariance parameter vectors from
the corresponding GP (two-kernel or zero-function) starts. Initialize log W
at its conditional prior mode for those effective amplitudes, then reoptimize
ALL parameters under the STP/GHP posterior. This is computational reuse of
training-only starts, not fixed GP parameters. Each mixing model records its
actual starts and GP checkpoint hash in initialization.json. The separate
model objectives and convergence checks are unchanged.
This is exact dense GP likelihood/gradient computation with only symmetric
pair distances stored; there is no inducing-point/subset approximation.
Models run sequentially with four BLAS threads to limit memory use.
Mixing moments/densities use stable Bessel evaluation with a recurrence when
the large posterior order overflows SciPy's scaled Bessel function.
The seven-model table is comparison_with_zero_mean.tex. The separate
comparison_train_sizes.tex scores BOTH training sizes on the SAME {len(data['test_ids'])}
test rows using saved old predictions, not the old larger test aggregate.
No test targets enter fitting or selection of the optimization start.
This remains a single-split, two-stage plug-in experiment, not full Bayes.
gp_function_uncertainty_diagnostic.json separately checks what happens if
mu uncertainty is integrated at the SAME fitted GP hyperparameters (no refit).
It does not replace the requested plug-in GP row in any comparison table.
Actual output: {root.relative_to(OUTPUT.parent.parent)}
Actual run/resume command: python wine_train60_plugin_experiment.py
'''
    path = root/'method.txt'
    method = path.read_text().replace(
        f'Use {STARTS} L-BFGS-B starts, seed 505, alternating the two kernel assignments.',
        f'Use {STARTS} L-BFGS-B starts; the initialization strategy is specified below.')
    method = method[:method.index('\nResults:')]
    path.write_text(method+note,encoding='utf-8')
    return root,data,config,old,points


def model_starts(root,folder,data,model,old_points):
    if not has_mixing(model):
        return old_points[model]
    base_model = 'gp_zero' if model in ZERO_MODELS else 'gp'
    source = source_case(root,base_model)/'optimization_starts.json'
    attempts = json.loads(source.read_text())
    assert len(attempts)==STARTS
    center = default_center(data['y_scaled'][data['train_ids']],data['x'].shape[1],model)
    loading = np.zeros(len(center))
    loading[0] = 1
    loading[1 if model in ZERO_MODELS else 2] = 1
    if model=='beta_gp':
        loading[1] = 2
    points = []
    for attempt in attempts:
        z = np.array(attempt['effective_log_parameters'])
        def derivative(t):
            mixing = -2.5+1.5*np.exp(-t) if model in ('tp_mu_gp','stp_zero') else 1-np.sinh(t)
            return mixing+loading@(z-center-loading*t)/4
        log_w = optimize.brentq(derivative,-25,25,xtol=1e-13)
        assert abs(derivative(log_w)) < 1e-10
        points.append(np.r_[z,log_w])
    case = folder/model
    case.mkdir(exist_ok=True)
    metadata = dict(source_model=base_model,source_checkpoint=str(source.relative_to(root)),
                    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                    method='Transport each optimized GP effective covariance; optimize conditional log-W prior; refit all coordinates.',
                    effective_log_initial_points=[z.tolist() for z in points])
    path = case/'initialization.json'
    if path.exists():
        assert json.loads(path.read_text())==metadata
    else:
        write_json(path,metadata)
    return points


def compare_training_sizes(root,data,old):
    rows = []
    old_root = OLD_OUTPUT/'wine'
    ids = data['test_ids']
    truth = data['y'][ids]
    for model in ALL_MODELS:
        for size,folder in [(len(old['train_ids']),old_root),(len(data['train_ids']),root)]:
            with np.load(source_case(folder,model)/'predictions.npz') as p:
                positions = np.searchsorted(p['test_ids'],ids)
                np.testing.assert_array_equal(p['test_ids'][positions],ids)
                mean = p['y_mean'][positions]
                ci = p['y_ci'][positions]
                rows.append(dict(model=model,n_train=size,n_test=len(ids),
                    y_rmse=float(np.sqrt(np.mean((truth-mean)**2))),
                    y_nlpd=float(p['nlpd'][positions].mean()),
                    y_crps=float(p['crps'][positions].mean()),
                    y_coverage_95=float(((truth>=ci[:,0])&(truth<=ci[:,1])).mean()),
                    y_interval_width=float((ci[:,1]-ci[:,0]).mean())))
    table = pd.DataFrame(rows)
    table.to_csv(root/'scores_train_sizes.csv',index=False)
    lines = [r'\begin{table}[htbp]',r'\centering',r'\small',
        r'\caption{Wine regression: both training sizes evaluated on the same '+str(len(ids))+
        r' held-out rows. The 300-row training set is a subset of the 60\% training set. '
        r'Scores use observed quality in original units. Hyperparameters and functions are plug-in estimates.}',
        r'\label{tab:wine_training_sizes}',r'\begin{tabular}{lrrrrr}',r'\toprule',
        r'Model & Training rows & RMSE & NLPD & CRPS & 95\% coverage (\%) \\',r'\midrule']
    for index,row in enumerate(rows):
        if index and index%2==0:
            lines.append(r'\midrule')
        label = zero.ZERO_TEX[row['model']] if row['model'] in ZERO_MODELS else TEX_NAMES[row['model']]
        lines.append(' & '.join([label,str(row['n_train']),f'{row["y_rmse"]:.5f}',
            f'{row["y_nlpd"]:.5f}',f'{row["y_crps"]:.5f}',f'{100*row["y_coverage_95"]:.2f}'])+r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (root/'comparison_train_sizes.tex').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    hashes = json.loads((root/'previous_results_sha256.json').read_text())
    for name,expected in hashes.items():
        assert hashlib.sha256((old_root/name).read_bytes()).hexdigest() == expected
    write_json(root/'training_size_validation.json',dict(all_checks_passed=True,
        old_results_unchanged=True,old_train_subset_new_train=True,new_test_subset_old_test=True,
        common_test_rows=len(ids),comparison_rows=len(rows),
        original_unit_intercept=float(data['y_mean']),
        result_size_MiB=sum(p.stat().st_size for p in root.rglob('*') if p.is_file())/2**20))
    print(table.to_string(index=False),flush=True)


def diagnose_gp_uncertainty(root,data):
    """Check whether omitted mu uncertainty contributes to GP undercoverage."""
    fit = json.loads((root/'gp'/'fit.json').read_text())
    pars = np.exp(fit['log_parameters'])
    v,a,k = pars[:3]
    d = data['x'].shape[1]
    w1,w2 = pars[3:3+d],pars[3+d:3+2*d]
    train,test = data['train_ids'],data['test_ids']
    x,y = data['x_scaled'][train],data['y_scaled'][train]
    C = se_kernel(x,x,v,w1)+se_kernel(x,x,a,w2)+k*np.eye(len(x))
    L = linalg.cholesky(C,lower=True)
    alpha = linalg.cho_solve((L,True),y)
    means,variances = [],[]
    for start in range(0,len(test),64):
        xp = data['x_scaled'][test[start:start+64]]
        cross = se_kernel(xp,x,v,w1)+se_kernel(xp,x,a,w2)
        solved = linalg.solve_triangular(L,cross.T,lower=True)
        means.append(data['y_mean']+data['y_scale']*(cross@alpha))
        variances.append(data['y_scale']**2*(v+a+k-np.sum(solved**2,axis=0)))
    mean,variance = np.concatenate(means),np.concatenate(variances)
    with np.load(root/'gp'/'predictions.npz') as plugin:
        np.testing.assert_allclose(mean,plugin['y_mean'],atol=1e-8,rtol=1e-9)
        assert np.all(variance>=plugin['y_var']-1e-9)
        ratio = float(np.mean(variance/plugin['y_var']))
    truth = data['y'][test]
    sd = np.sqrt(variance)
    crps = gaussian_mixture_crps_closed_form(truth,mean[None,:],variance[None,:])
    report = dict(method='Integrate mu GP uncertainty at the existing GP MAP hyperparameters; no refit or hyperparameter averaging.',
        same_predictive_mean_verified=True,mean_variance_ratio_to_plugin=ratio,
        y_rmse=float(np.sqrt(np.mean((truth-mean)**2))),
        y_nlpd=float(-stats.norm.logpdf(truth,mean,sd).mean()),y_crps=float(crps.mean()),
        y_coverage_95=float(np.mean(abs(truth-mean)<=stats.norm.ppf(.975)*sd)),
        y_interval_width=float(2*stats.norm.ppf(.975)*sd.mean()))
    write_json(root/'gp_function_uncertainty_diagnostic.json',report)
    print('GP uncertainty diagnostic (not a replacement fit):',report,flush=True)


def write_zero_method(folder,data):
    # Start from the common template so this is safe to repeat independently.
    method = zero.METHOD.replace('Use 24 starts',f'Use {STARTS} starts')
    method = method[:method.index('\nRun:')]
    method += ('\nRun/resume: python wine_train60_plugin_experiment.py\n'
               'This run uses six starts. GP-zero has one rescaled old-fit start and\n'
               'five seeded random starts. STP-zero and GHP-zero transport all six\n'
               'optimized GP-zero solutions, initialize W from its conditional prior\n'
               'mode, and reoptimize every parameter. See each initialization.json.\n'
               f'Original-unit fixed intercept: {float(data["y_mean"]):.12g}\n')
    (folder/'method.txt').write_text(method,encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=('all','prepare','fit','summarize'),default='all')
    parser.add_argument('--model',choices=ALL_MODELS)
    args = parser.parse_args()
    root,data,config,old,points = prepare_run()
    if args.stage=='prepare':
        return
    if args.stage in ('all','fit'):
        for model in MODELS:
            if args.model is None or model==args.model:
                run_model(root,data,config,model,initial_points=model_starts(root,root,data,model,points))
    if args.model is not None and args.model in MODELS:
        return
    if not (root/'zero_mean_baselines'/'source_provenance.json').exists():
        summarize(('wine',),destination=root,output=OUTPUT)
        validate(('wine',),destination=root,output=OUTPUT)
    _,folder,_,zero_config = zero.prepare('wine',output=OUTPUT)
    write_zero_method(folder,data)
    if args.stage in ('all','fit'):
        for model in ZERO_MODELS:
            if args.model is None or model==args.model:
                run_model(folder,data,zero_config,model,initial_points=model_starts(root,folder,data,model,points))
    if args.model is not None:
        return
    zero.summarize(root,folder,data,zero_config)
    zero.audit(root,folder,zero_config)
    diagnose_gp_uncertainty(root,data)
    compare_training_sizes(root,data,old)


if __name__=='__main__':
    with threadpool_limits(limits=4):
        main()
