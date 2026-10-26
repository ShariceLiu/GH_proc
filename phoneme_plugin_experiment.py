"""Speaker-disjoint Phoneme-aa spectrum reconstruction with plug-in GP/STP/GHP.

Run: python phoneme_plugin_experiment.py --stage all
Uses a fixed random pool of 60 speakers, one segment each, and five 40/20 splits.
Reuses the Gasoline fitting/prediction routines with a separate output directory.
"""
import os
for variable in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[variable] = '1'
import argparse
import gzip
import hashlib
import io
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

import gasoline_plugin_experiment as pipeline
from gasoline_plugin_experiment import MODELS, SHARING, COUNTS, METRICS, arrays, digest
from shared_beta_plugin_experiment import write_json

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT/'data'/'real_datasets'/'phoneme'
OUTPUT = ROOT/'figure'/'phoneme_aa_plugin_five_splits'
SOURCE_URL = 'https://hastie.su.domains/ElemStatLearn/datasets/phoneme.data'
PHONEME = 'aa'
POOL_SEED = 74026
SEEDS = (74027, 74028, 74029, 74030, 74031)
CONTEXT_SEED = 84026


def source_data():
    """Keep the original source bytes losslessly compressed, including speaker IDs."""
    SOURCE.mkdir(parents=True, exist_ok=True)
    raw_path, compressed = SOURCE/'phoneme.data', SOURCE/'phoneme.data.gz'
    if compressed.exists():
        content = gzip.decompress(compressed.read_bytes())
    else:
        if raw_path.exists():
            content = raw_path.read_bytes()
        else:
            import requests
            response = requests.get(SOURCE_URL, timeout=60)
            response.raise_for_status(); content = response.content
        compressed.write_bytes(gzip.compress(content, compresslevel=9, mtime=0))
        assert gzip.decompress(compressed.read_bytes()) == content
        # Only this experiment's freshly downloaded, byte-identical raw copy.
        if raw_path.exists() and raw_path.read_bytes() == content:
            raw_path.unlink()
    frame = pd.read_csv(io.BytesIO(content))
    columns = [f'x.{i}' for i in range(1, 257)]
    assert frame.shape == (4509, 259) and not frame.isna().any().any()
    assert np.isfinite(frame[columns].to_numpy()).all()
    assert not frame[columns].duplicated().any()
    # Labels look like train.dr1.mcpm0.sa1: use person ID, not utterance path.
    parts = frame.speaker.str.split('.')
    assert parts.map(len).eq(4).all()
    frame['speaker_id'] = parts.str[2]
    assert frame.speaker_id.str.fullmatch('[mf][a-z0-9]+').all()
    counts = frame.groupby('g').agg(curves=('speaker_id', 'size'), speakers=('speaker_id', 'nunique'))
    metadata = dict(url=SOURCE_URL, raw_sha256=hashlib.sha256(content).hexdigest(),
        compressed_sha256=digest(compressed), raw_bytes=len(content), compressed_bytes=compressed.stat().st_size,
        rows=len(frame), bins=256, speakers=int(frame.speaker_id.nunique()),
        class_counts=counts.to_dict(orient='index'),
        reference='Hastie, Tibshirani and Friedman, Elements of Statistical Learning: Phoneme data from TIMIT.',
        documentation='https://raw.githubusercontent.com/cran/ElemStatLearn/master/man/phoneme.Rd',
        units='Source log-periodogram values; frequency is indexed by bin, without assuming a Hz conversion.')
    path = SOURCE/'source.json'
    if path.exists():
        assert json.loads(path.read_text()) == metadata
    else:
        write_json(path, metadata)
    return frame


def select_pool(frame):
    """Preselect by labels and seeded randomness only, never spectrum values/scores."""
    eligible = frame[frame.g == PHONEME]
    speakers = np.sort(eligible.speaker_id.unique())
    rng = np.random.default_rng(POOL_SEED)
    selected = np.sort(rng.choice(speakers, 60, replace=False))
    rows = np.sort([int(rng.choice(eligible.index[eligible.speaker_id == speaker])) for speaker in selected])
    assert len(rows) == 60 and frame.loc[rows, 'speaker_id'].nunique() == 60
    return rows


def build_data(frame):
    rows = select_pool(frame); selected = np.arange(0, 256, 2)
    y = frame.loc[rows, [f'x.{i+1}' for i in selected]].to_numpy(dtype=float)
    return dict(y=y, frequency_bin=selected, source_channel_ids=selected,
        x=selected.astype(float)/selected[-1]*10., source_rows=rows,
        source_row_ids=frame.loc[rows, 'row.names'].to_numpy(),
        speaker_ids=frame.loc[rows, 'speaker_id'].to_numpy(dtype=str),
        source_recordings=frame.loc[rows, 'speaker'].to_numpy(dtype=str))


def prepare():
    frame = source_data(); data = build_data(frame)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    path = OUTPUT/'dataset.npz'
    if path.exists():
        old = arrays(path)
        for key in data:
            np.testing.assert_array_equal(old[key], data[key])
    else:
        np.savez_compressed(path, **data)
    metadata = json.loads((SOURCE/'source.json').read_text())
    settings = dict(dataset_sha256=digest(path), source_sha256=metadata['raw_sha256'],
        phoneme=PHONEME, pool_seed=POOL_SEED, pool_speakers=60, curves_per_speaker=1,
        eligible_curves=int((frame.g == PHONEME).sum()),
        eligible_speakers=int(frame.loc[frame.g == PHONEME, 'speaker_id'].nunique()),
        split_seeds=list(SEEDS), context_seed=CONTEXT_SEED, n_train=40, n_test=20,
        bins=128, bin_rule='Retain original zero-based bins 0,2,...,254; no smoothing or aggregation.',
        context_counts=list(COUNTS), models=list(MODELS), function_sharing=list(SHARING),
        GHP_lambda=1., GHP_omega=1., STP_nu=5., prior_log_sd=2.,
        selection='One randomly selected aa segment per speaker, from 60 fixed randomly selected speakers.',
        split='Five random 40/20 partitions of the same 60-speaker pool; no speaker crosses a split.',
        preprocessing='Training-only common scale; no centering or per-curve normalization.',
        prediction='Plug-in MAP theta and mu/beta; retain posterior uncertainty in test-curve W.',
        score_units='Original source log-periodogram units; observed y only.',
        aggregation='Pooled RMSE within each split; mean other metrics. Mean and sample SD of five split scores.',
        baseline='Same priors, 12/24 optimization starts and prediction algorithms as Gasoline; different source/splits.')
    settings_path = OUTPUT/'experiment.json'
    if settings_path.exists():
        assert json.loads(settings_path.read_text()) == settings
    else:
        write_json(settings_path, settings)
    pool = pd.DataFrame(dict(curve=np.arange(1, 61), source_row_index=data['source_rows'],
        source_row_id=data['source_row_ids'], speaker=data['speaker_ids'], recording=data['source_recordings'], phoneme=PHONEME))
    pool.to_csv(OUTPUT/'speaker_pool.csv', index=False)
    for seed in SEEDS:
        folder = OUTPUT/f'split_{seed}'; folder.mkdir(exist_ok=True)
        perm = np.random.default_rng(seed).permutation(60)
        train, test = np.sort(perm[:40]), np.sort(perm[40:])
        assert not np.intersect1d(data['speaker_ids'][train], data['speaker_ids'][test]).size
        orders = np.stack([np.random.default_rng(np.random.SeedSequence([CONTEXT_SEED, int(row)]))
                           .permutation(128) for row in test])
        scale = float(np.std(data['y'][train]-data['y'][train].mean(0)))
        Y = data['y'][train]/scale
        center = np.log([.5, max(float(np.mean(Y.mean(0)**2)), .5), .01, 1., .04])
        split = dict(train_rows=train, test_rows=test, context_orders=orders,
                     common_scale=np.asarray(scale), prior_center=center)
        path = folder/'split.npz'
        if path.exists():
            saved = arrays(path)
            for key in split:
                np.testing.assert_array_equal(saved[key], split[key])
        else:
            np.savez_compressed(path, **split)
        roles = pool.copy(); roles['role'] = 'test'; roles.loc[train, 'role'] = 'train'
        roles.to_csv(folder/'speaker_roles.csv', index=False)
    if not (OUTPUT/'spectra_overview.pdf').exists():
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(data['frequency_bin'], data['y'].T, lw=.6, alpha=.35)
        ax.set(xlabel='Frequency-bin index', ylabel='Log-periodogram',
               title='Phoneme aa: one segment from each of 60 speakers, 128 retained bins')
        fig.tight_layout(); fig.savefig(OUTPUT/'spectra_overview.pdf'); plt.close(fig)
    (OUTPUT/'method.txt').write_text(METHOD, encoding='utf-8')
    return data


def validate_design(data, seeds=SEEDS):
    """Audit source selection and speaker separation independently of fit results."""
    frame = source_data(); expected = build_data(frame)
    for key in expected:
        np.testing.assert_array_equal(data[key], expected[key])
    assert len(np.unique(data['speaker_ids'])) == 60
    assert frame.loc[data['source_rows'], 'g'].eq(PHONEME).all()
    for seed in seeds:
        split = arrays(OUTPUT/f'split_{seed}'/'split.npz')
        train, test = split['train_rows'], split['test_rows']
        assert len(np.unique(data['speaker_ids'][train])) == 40
        assert len(np.unique(data['speaker_ids'][test])) == 20
        assert not np.intersect1d(data['speaker_ids'][train], data['speaker_ids'][test]).size
        roles = pd.read_csv(OUTPUT/f'split_{seed}'/'speaker_roles.csv')
        np.testing.assert_array_equal(roles.loc[roles.role == 'train', 'speaker'], data['speaker_ids'][train])
        np.testing.assert_array_equal(roles.loc[roles.role == 'test', 'speaker'], data['speaker_ids'][test])
    return dict(speaker_disjoint=True, one_curve_per_speaker=True, phoneme=PHONEME, selected_speakers=60)


def summarize(data):
    caption = (r'Phoneme aa spectrum reconstruction: mean $\pm$ sample SD across five speaker-disjoint '
        r'40/20 splits of 60 speakers, one segment per speaker. Each curve has 128 frequency bins; '
        r'15 or 30 bins are revealed. Scores evaluate held-out observed log-periodograms. '
        r'Hyperparameters and functions are plug-in estimates.')
    pipeline.summarize(data, output=OUTPUT, seeds=SEEDS, caption=caption, table_label='phoneme_aa_plugin',
        coordinate_key='frequency_bin', xlabel='Frequency-bin index', ylabel='Log-periodogram')
    pool = pd.read_csv(OUTPUT/'speaker_pool.csv')
    for name in ('per_curve_scores.csv', 'common_per_curve_scores.csv'):
        scores = pd.read_csv(OUTPUT/name)
        scores.merge(pool, on='curve', validate='many_to_one').to_csv(OUTPUT/name, index=False)


METHOD = '''PHONEME AA: SHARED AND INDIVIDUAL PLUG-IN REGRESSION

Task: reconstruct a partially observed log-periodogram, not classify its phoneme.
Source: Elements of Statistical Learning Phoneme data (4509 curves, 256 bins).
The original text is cached losslessly in gzip with a SHA256 checksum. The stored
speaker field is an utterance path; extract its person-ID component for grouping.
The downloaded file has 437 distinct speaker IDs; older narrative documentation
mentions 50. Use the actual IDs rather than that inconsistent historical count.

Choose the aa vowel before evaluating any model. Pool seed 74026 selects 60
speakers uniformly without replacement from those with aa segments, then one
aa segment uniformly from each selected speaker. Selection never uses spectrum
values or model scores. speaker_pool.csv maps each curve to its source row and
speaker. This is a small initial subset, not a claim about the full speech corpus.

Retain every second original bin (0,2,...,254): 128 original observations, without
smoothing or averaging. Model x=bin/254*10; plots use the original bin index.
Seeds 74027--74031 partition the same 60 speakers into 40 training/20 test speakers.
No speaker is in both roles in a split; each speaker contributes just one curve.
All 128 points on training curves are used. Context seed 84026 and pool curve ID
give nested random sets of 15/30 revealed bins, identical for every model/split.
Score the remaining 113/98 bins; common-target scores use the same 98 bins for both.

Exactly the Gasoline fitting and prediction pipeline: common training-only
scale = SD(Y_train - wavelength/bin-wise training mean), no centering or per-curve
normalization. Five log parameters (v,a,k,w,w_h) have Normal priors with SD=2,
center log([.5,max(mean(training_mean_scaled**2),.5),.01,1,.04]). SE covariances
are v*exp(-w*(x-x')**2/2). Match function priors on h=mu or h=E_prior[W]*beta.

GP fixes W=1. STP plus GP-prior mu fixes W~IG(2.5,1.5), equivalent to nu=5 for
the residual Student-t process. GHP-mu and GHP-beta fix GIG(lambda=1,chi=psi=1).
GHP-beta uses mu=0; the other models use beta=0. The STP training model includes
an unscaled GP-prior mean and therefore is not a pure Student-t process.

Shared: integrate one common GP-prior function from the training likelihood,
jointly MAP-fit log(theta) and training log(W), then freeze its conditional GP
posterior mean. Individual: integrate independent per-curve functions to fit
common theta and training W. On each test curve, freeze theta, optimize log W
from only that curve's context with its function marginalized, and plug in the
conditional GP mean as that curve's mu/beta. Both use log-coordinate MAP with
log-W Jacobians. Twelve shared and 24 individual starts; select the best checked
stationary solution. This multistart local search is not a global-mode guarantee.

Given the fixed function and theta, recompute the test W posterior from the
original mixing prior and the fixed-function context likelihood. Integrate W
for Gaussian/Student-t/GH prediction. Do not reuse a fitted context W posterior
as a new prior. Pointwise equal-tail 95% intervals include observation noise and
remaining W uncertainty, but omit function and hyperparameter uncertainty.

Scores: observed-y RMSE, marginal NLPD, CRPS, 95% coverage and interval width,
in source log-periodogram units. No latent f truth is available. Pool squared
errors within each split before taking the square root; average other metrics.
Report mean +/- sample SD over five SPLIT-LEVEL scores, not per-curve SD or SE.
The splits overlap, so their SD is descriptive, not independent-replicate inference.

Fits are sequential with one BLAS thread. Only compact prediction summaries are
saved; no dense posterior covariance matrices, particles or HMC draws. Resume
from per-start checkpoints. Plot the first two test curve IDs per split without
choosing examples by performance. Validation checks source and speaker separation,
training-only preprocessing, posterior gradients/Hessians, conditioning, predictive
densities, intervals, CRPS reference calculations, aggregation and file provenance.

Run: python phoneme_plugin_experiment.py --stage all
Resume one case: --stage run --seed 74027 --sharing shared --model gp
Summarize: --stage summarize
Validate: python validate_phoneme_plugin.py
Dependencies are already available: NumPy, SciPy, pandas, matplotlib, requests,
threadpoolctl. No audio files or new packages are needed.
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=('all', 'prepare', 'run', 'summarize'), default='all')
    parser.add_argument('--seed', type=int, choices=SEEDS)
    parser.add_argument('--sharing', choices=SHARING)
    parser.add_argument('--model', choices=MODELS)
    args = parser.parse_args(); data = prepare()
    if args.stage in ('all', 'run'):
        for seed in SEEDS if args.seed is None else (args.seed,):
            for sharing in SHARING if args.sharing is None else (args.sharing,):
                for model in MODELS if args.model is None else (args.model,):
                    folder, split, fit = pipeline.fit_case(data, seed, sharing, model, output=OUTPUT)
                    pipeline.predict_case(data, seed, sharing, model, folder, split, fit)
    if args.stage in ('all', 'summarize'):
        summarize(data)


if __name__ == '__main__':
    with threadpool_limits(limits=1):
        main()
