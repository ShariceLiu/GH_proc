"""Repeat the seven-model Airfoil comparison on a new random row split.

Defaults change only split seed 17026 -> 17027, retaining all model priors,
HMC seeds/settings and the 300/1203 sizes. Completed fits can be resumed.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import redirect_stdout, redirect_stderr
import hashlib
import json
import multiprocessing
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from GH_tabular_hmc import ALL_MODELS, MODELS
from hmc_airfoil_experiment import prepare, fit_model, summarize as summarize_four, write_json
from hmc_airfoil_zero_experiment import case_folder, prepare as prepare_zero, score_crps, summarize
from validate_airfoil_experiment import validate as validate_four
from validate_airfoil_zero_experiment import validate as validate_seven


def fit_worker(root, model):
    root = Path(root)
    with np.load(root/'dataset.npz') as saved:
        data = {key: saved[key].copy() for key in saved.files}
    with (root/'logs'/f'{model}.log').open('a', encoding='utf-8') as log:
        with redirect_stdout(log), redirect_stderr(log), threadpool_limits(limits=1):
            report = fit_model(case_folder(root, model).parent, data, model,
                               chains=4, warmup=1000, draws=1500,
                               seed=18026 if model in MODELS else 24026,
                               target_accept=.95, max_retries=1)
    if not report['diagnostics_ok']:
        raise RuntimeError(f'{model}: HMC diagnostics failed; inspect logs')
    return {key: report[key] for key in ['model', 'y_rmse', 'y_mean_marginal_nlpd',
            'max_rhat', 'min_bulk_ess', 'min_tail_ess', 'divergences']}


def crps_worker(root, model):
    root = Path(root)
    with (root/'logs'/f'crps_{model}.log').open('a', encoding='utf-8') as log:
        with redirect_stdout(log), redirect_stderr(log), threadpool_limits(limits=1):
            score_crps(root, [model])
    return json.loads((root/'crps'/f'{model}.json').read_text())['y_crps']


def run_pool(worker, root, workers):
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn')) as pool:
        jobs = {pool.submit(worker, str(root), model): model for model in ALL_MODELS}
        for job in as_completed(jobs):
            print(f'{worker.__name__} {jobs[job]}: {job.result()}', flush=True)


def compare(root, original, split_seed):
    frames = []
    for folder in [original, root]:
        table = pd.read_csv(folder/'scores_with_zero_mean_crps.csv')
        with np.load(folder/'dataset.npz') as saved:
            table.insert(0, 'split_seed', int(saved['split_seed']))
        frames.append(table)
    pd.concat(frames, ignore_index=True).to_csv(root/'comparison_by_split.csv', index=False)
    latex = root/'comparison_with_zero_mean_crps.tex'
    text = latex.read_text(encoding='utf-8')
    text = text.replace('Airfoil regression: the same', f'Airfoil regression (random split seed {split_seed}): the same')
    text = text.replace(r'\label{tab:airfoil_seven_models}', rf'\label{{tab:airfoil_seven_models_split{split_seed}}}')
    latex.write_text(text, encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('figure/airfoil_single_function_hmc_split17027'))
    parser.add_argument('--original', type=Path, default=Path('figure/airfoil_single_function_hmc'))
    parser.add_argument('--split-seed', type=int, default=17027)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--stage', choices=['prepare', 'fit', 'score', 'all'], default='all')
    args = parser.parse_args()
    if args.root.resolve() == args.original.resolve():
        parser.error('Use a separate output directory for the new split')
    if args.workers < 1:
        parser.error('workers must be positive')
    with threadpool_limits(limits=1):
        data = prepare(args.root, 300, args.split_seed)
        (args.root/'logs').mkdir(exist_ok=True)
        with np.load(args.original/'dataset.npz') as old:
            np.testing.assert_array_equal(data['x'], old['x'])
            np.testing.assert_array_equal(data['y'], old['y'])
            assert not np.array_equal(data['train_ids'], old['train_ids'])
            shared = len(np.intersect1d(data['train_ids'], old['train_ids']))
            old_seed = int(old['split_seed'])
        sources = [p for p in args.original.rglob('*') if p.is_file()]
        hashes = {str(p.relative_to(args.original)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
        provenance = args.root/'previous_split_provenance.json'
        if provenance.exists():
            assert hashes == json.loads(provenance.read_text())
        else:
            write_json(provenance, hashes)
        write_json(args.root/'split_comparison.json', dict(old_split_seed=old_seed,
                   new_split_seed=args.split_seed, shared_training_rows=shared,
                   replaced_training_rows=300-shared, n_train=300, n_test=1203,
                   identical_raw_data=True, preprocessing='new training rows only',
                   sampler='Same priors, HMC seeds and settings as earlier fits'))
        print(f'Split {old_seed} -> {args.split_seed}: {shared}/300 training rows overlap', flush=True)
        if args.stage in ['fit', 'all']:
            run_pool(fit_worker, args.root, args.workers)
        if args.stage in ['score', 'all']:
            # Freeze four-model reports before the extension records their hashes.
            if not (args.root/'zero_mean_baselines'/'source_provenance.json').exists():
                summarize_four(args.root)
            validate_four(args.root)
            prepare_zero(args.root)
            run_pool(crps_worker, args.root, min(3, args.workers))
            summarize(args.root)
            validate_seven(args.root)
            compare(args.root, args.original, args.split_seed)
            for name, expected in hashes.items():
                assert hashlib.sha256((args.original/name).read_bytes()).hexdigest() == expected
            print('Complete: all seven models validated; previous split unchanged.', flush=True)


if __name__ == '__main__':
    main()
