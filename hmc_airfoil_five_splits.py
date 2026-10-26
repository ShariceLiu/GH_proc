"""Fit three additional Airfoil splits and summarize five splits with mean +/- SD.

Reuses completed seeds 17026/17027; refits every model on seeds 17028--17030.
SD is the sample standard deviation (ddof=1) of five per-split scores, not
a standard error or variation across HMC draws. Run with --stage all (default).
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from threadpoolctl import threadpool_limits

from GH_tabular_hmc import ALL_MODELS, MODELS
from hmc_airfoil_experiment import prepare, summarize as summarize_four, write_json
from hmc_airfoil_resplit_experiment import fit_worker, crps_worker
from hmc_airfoil_zero_experiment import prepare as prepare_zero, summarize, case_folder
from validate_airfoil_experiment import validate as validate_four
from validate_airfoil_zero_experiment import validate as validate_seven

SEEDS = (17026, 17027, 17028, 17029, 17030)
NEW_SEEDS = SEEDS[2:]
BASE = Path('figure/airfoil_single_function_hmc')
OUTPUT = Path('figure/airfoil_five_splits_hmc')
METRICS = ('y_rmse', 'y_mean_marginal_nlpd', 'y_crps', 'y_coverage_95')
LABELS = ['GP (two SE)', 'STP + GP mean', r'GHP ($\mu$ GP)', r'GHP ($\beta$ GP)',
          r'GP ($\mu=0$, one SE)', r'STP ($\mu=0$, one SE)', r'GHP ($\mu=\beta=0$, one SE)']


def root_for(seed):
    return BASE if seed == SEEDS[0] else Path(str(BASE)+f'_split{seed}')


def freeze_previous():
    """Record and subsequently verify every file in the two completed runs."""
    hashes = {}
    for seed in SEEDS[:2]:
        root = root_for(seed)
        for path in sorted(root.rglob('*')):
            if path.is_file():
                hashes[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    file = OUTPUT/'previous_results_provenance.json'
    if file.exists():
        if hashes != json.loads(file.read_text()):
            raise ValueError('Previously completed experiment artifacts changed')
    else:
        write_json(file, hashes)


def prepare_all():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    freeze_previous()
    splits = {}
    for seed in SEEDS:
        root = root_for(seed)
        if seed in NEW_SEEDS:
            prepare(root, n_train=300, split_seed=seed)
            (root/'logs').mkdir(exist_ok=True)
        with np.load(root/'dataset.npz') as d:
            assert int(d['split_seed']) == seed
            order = np.random.default_rng(seed).permutation(len(d['y']))
            assert_array_equal(d['train_ids'], np.sort(order[:300]))
            assert_array_equal(d['test_ids'], np.sort(order[300:]))
            splits[seed] = d['train_ids'].copy()
    overlap = pd.DataFrame([[len(np.intersect1d(splits[a], splits[b])) for b in SEEDS]
                           for a in SEEDS], index=SEEDS, columns=SEEDS)
    overlap.to_csv(OUTPUT/'training_overlap.csv', index_label='split_seed')
    write_json(OUTPUT/'experiment.json', dict(split_seeds=SEEDS, new_split_seeds=NEW_SEEDS,
               n_train=300, n_test=1203, models=ALL_MODELS, summary='mean and sample SD (ddof=1) of per-split scores',
               standardization='training rows only, separately for each split',
               fixed_gig=dict(l=1., omega=1.), fixed_stp_nu=5.,
               sampling=dict(chains=4, warmup=1000, draws=1500, target_accept=.95,
                             original_models_seed=18026, zero_models_seed=24026,
                             retries='double warmup/draws and raise target acceptance if diagnostics fail'),
               interpretation='Random row splits of one dataset overlap; SD measures split sensitivity, not standard error.'))
    print('Prepared splits '+', '.join(map(str, SEEDS)), flush=True)


def pool_jobs(worker, workers):
    failures = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs = {pool.submit(worker, str(root_for(seed)), model): (seed, model)
                for seed in NEW_SEEDS for model in ALL_MODELS}
        for job in as_completed(jobs):
            seed, model = jobs[job]
            try:
                result = job.result()
                print(f'{worker.__name__} split={seed} model={model}: {result}', flush=True)
            except Exception as exc:
                failures.append(dict(seed=seed, model=model, error=repr(exc)))
                print(f'FAILED split={seed} model={model}: {exc!r}', flush=True)
    if failures:
        write_json(OUTPUT/f'{worker.__name__}_failures.json', failures)
        raise RuntimeError(f'{len(failures)} jobs require attention; completed jobs were saved')


def score_all(workers):
    for seed in NEW_SEEDS:
        root = root_for(seed)
        if not (root/'zero_mean_baselines'/'source_provenance.json').exists():
            summarize_four(root)
        validate_four(root)
        prepare_zero(root)
    pool_jobs(crps_worker, min(workers, 3))
    for seed in NEW_SEEDS:
        root = root_for(seed)
        summarize(root)
        validate_seven(root)
        path = root/'comparison_with_zero_mean_crps.tex'
        text = path.read_text(encoding='utf-8')
        text = text.replace('Airfoil regression: the same', f'Airfoil regression (split seed {seed}): the same')
        text = text.replace(r'\label{tab:airfoil_seven_models}', rf'\label{{tab:airfoil_seven_models_split{seed}}}')
        path.write_text(text, encoding='utf-8')


def aggregate():
    frames = []
    reference = None
    for seed in SEEDS:
        root = root_for(seed)
        validation = json.loads((root/'zero_mean_baselines'/'validation.json').read_text())
        assert validation['validation_passed']
        with np.load(root/'dataset.npz') as d:
            x, y, train = d['x'], d['y'], d['train_ids']
            assert int(d['split_seed']) == seed
            assert len(train) == 300 and len(d['test_ids']) == 1203
            if reference is None:
                reference = (x.copy(), y.copy())
            assert_array_equal(x, reference[0])
            assert_array_equal(y, reference[1])
            assert_allclose(d['x_mean'], x[train].mean(axis=0))
            assert_allclose(d['x_scale'], x[train].std(axis=0))
            assert_allclose(d['y_mean'], y[train].mean())
            assert_allclose(d['y_scale'], y[train].std())
        table = pd.read_csv(root/'scores_with_zero_mean_crps.csv')
        assert list(table.model) == list(ALL_MODELS)
        assert table.diagnostics_ok.all() and np.isfinite(table[list(METRICS)]).all().all()
        assert (table.n_train == 300).all() and (table.n_test == 1203).all()
        for _, row in table.iterrows():
            report = json.loads((case_folder(root, row.model)/'report.json').read_text())
            crps = json.loads((root/'crps'/f'{row.model}.json').read_text())
            for metric in METRICS:
                assert_allclose(row[metric], crps[metric] if metric == 'y_crps' else report[metric], atol=1e-12)
        table.insert(0, 'split_seed', seed)
        frames.append(table)
    raw = pd.concat(frames, ignore_index=True)
    raw.to_csv(OUTPUT/'scores_by_split.csv', index=False)
    rows = []
    for model in ALL_MODELS:
        selected = raw[raw.model == model]
        assert selected.split_seed.nunique() == len(SEEDS) == len(selected)
        result = dict(model=model, n_splits=len(selected))
        for metric in METRICS:
            values = selected[metric].to_numpy()
            result[metric+'_mean'] = float(values.mean())
            result[metric+'_sd'] = float(values.std(ddof=1))
            # Independent direct calculation verifies SD uses n-1, not n or sqrt(n).
            direct = np.sqrt(sum((float(v)-result[metric+'_mean'])**2 for v in values)/(len(values)-1))
            assert_allclose(result[metric+'_sd'], direct, atol=1e-14)
        rows.append(result)
    summary = pd.DataFrame(rows)
    summary.to_csv(OUTPUT/'scores_mean_std.csv', index=False)
    lines = [r'% Requires \usepackage{booktabs}', r'\begin{table}[htbp]', r'\centering', r'\small',
             r'\caption{Airfoil regression: mean $\pm$ sample standard deviation across five random '
             r'train/test splits (seeds 17026--17030), each with 300 training and 1,203 test rows. '
             r'All models are refitted per split using HMC. RMSE and CRPS are in dB; coverage is in percent '
             r'(its standard deviation is in percentage points). GHP fixes $\lambda=\omega=1$; STP fixes '
             r'$\nu=5$. Zero means refer to the training-standardized response. All scores concern $y$.}',
             r'\label{tab:airfoil_five_splits}', r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Model & RMSE ($y$) & NLPD ($y$) & CRPS ($y$) & 95\% coverage (\%) \\', r'\midrule']
    for i, row in enumerate(rows):
        if i == len(MODELS):
            lines.append(r'\midrule')
        cells = []
        for metric in METRICS:
            scale = 100 if metric == 'y_coverage_95' else 1
            digits = 2 if scale == 100 else 5
            cells.append(f'${scale*row[metric+"_mean"]:.{digits}f} \\pm {scale*row[metric+"_sd"]:.{digits}f}$')
        lines.append(LABELS[i]+' & '+' & '.join(cells)+r' \\')
    lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    (OUTPUT/'comparison_mean_std.tex').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write_json(OUTPUT/'summary.json', dict(validation_passed=True, split_seeds=SEEDS, total_fits=len(raw),
               sample_sd_ddof=1, all_diagnostics_ok=bool(raw.diagnostics_ok.all()),
               max_rhat=float(raw.max_rhat.max()), min_bulk_ess=float(raw.min_bulk_ess.min()),
               min_tail_ess=float(raw.min_tail_ess.min()), retained_divergences=int(raw.divergences.sum()),
               scores=rows))
    freeze_previous()
    print(summary.to_string(index=False), flush=True)
    print('Complete: 35 fits across five splits, mean +/- sample SD saved; previous results unchanged.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--stage', choices=['prepare', 'fit', 'score', 'aggregate', 'all'], default='all')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    with threadpool_limits(limits=1):
        prepare_all()
        if args.stage in ['fit', 'all']:
            pool_jobs(fit_worker, args.workers)
        if args.stage in ['score', 'all']:
            score_all(args.workers)
        if args.stage in ['aggregate', 'all']:
            aggregate()


if __name__ == '__main__':
    main()
