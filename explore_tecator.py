"""Cache and visualize Tecator spectra without fitting a regression model.

Run: python explore_tecator.py
Each row is a curve: x = wavelength, y = measured absorbance. Composition
measurements are metadata, not the regression target in our curve experiment.
The original source, including its permission note, is kept in the cache.
"""

import argparse
import hashlib
import io
import json
from pathlib import Path
from urllib.request import urlopen

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
DEFAULT_CACHE = ROOT / 'data' / 'real_datasets' / 'tecator'
DEFAULT_OUTPUT = ROOT / 'figure' / 'tecator_exploration'
SOURCE_URL = 'https://lib.stat.cmu.edu/datasets/tecator'
WAVELENGTH_SOURCE = 'https://search.r-project.org/CRAN/refmans/fsemipar/html/Tecator.html'
GROUPS = [('C', 129), ('M', 43), ('T', 43), ('E1', 8), ('E2', 17)]


def parse_tecator(raw):
    """Parse the 100 spectra + 22 PCA + 3 composition fields per sample."""
    source_text = raw.decode('ascii')
    # This marker also occurs in a header example; data follow its LAST occurrence.
    header, marker, numeric_text = source_text.rpartition('extrapolation_examples=25')
    if not marker:
        raise ValueError('Tecator data marker not found; refusing to parse the response.')
    numeric = np.loadtxt(io.StringIO(numeric_text))
    if numeric.shape != (6000, 5) or not np.isfinite(numeric).all():
        raise ValueError(f'Expected 6000 finite rows of 5 numbers, got {numeric.shape}.')
    records = numeric.reshape(240, 125)
    spectra = records[:, :100].copy()
    pca = records[:, 100:122].copy()
    composition = records[:, 122:125].copy()
    if np.any((composition < 0) | (composition > 100)):
        raise ValueError('Invalid composition percentages.')
    if not np.allclose(composition[0], [60.5, 22.5, 16.7], atol=1e-10, rtol=0):
        raise ValueError('First-sample check failed: moisture/fat/protein order changed.')
    sample_id = np.arange(1, 241)
    source_subset = np.repeat([group for group, _ in GROUPS], [n for _, n in GROUPS])
    split = np.where(sample_id <= 172, 'train',
                     np.where(sample_id <= 215, 'test', 'extrapolation'))
    _, first, inverse = np.unique(spectra, axis=0, return_index=True, return_inverse=True)
    samples = pd.DataFrame({
        'sample_id': sample_id,
        # Equal spectra share the ID of their first source-file occurrence.
        'spectrum_group_id': first[inverse] + 1,
        'source_subset': source_subset,
        'benchmark_split': split,
        'moisture_pct': composition[:, 0],
        'fat_pct': composition[:, 1],
        'protein_pct': composition[:, 2],
    })
    # The raw file supplies the wavelength range, but not a coordinate column.
    # Use the equally spaced 850--1050 nm grid documented by fsemipar.
    return {
        'wavelength_nm': np.linspace(850.0, 1050.0, 100),
        'spectra': spectra,
        'samples': samples,
        'source_pca': pca,
        'source_notes': header + marker + '\n',
    }


def load_tecator(cache_dir=DEFAULT_CACHE, *, refresh=False):
    """Return original curves/metadata; use the cached source on subsequent calls."""
    cache_dir = Path(cache_dir)
    raw_path = cache_dir / 'tecator_raw.txt'
    if raw_path.exists() and not refresh:
        raw = raw_path.read_bytes()
    else:
        print(f'Downloading {SOURCE_URL}', flush=True)
        with urlopen(SOURCE_URL, timeout=60) as response:
            raw = response.read()
    data = parse_tecator(raw)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if refresh or not raw_path.exists():
        raw_path.write_bytes(raw)
    data['source_sha256'] = hashlib.sha256(raw).hexdigest()
    return data


def save_data(data, cache_dir):
    """Save numeric NPZ, readable CSVs, and original permission/provenance notes."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    samples = data['samples']
    wavelength = data['wavelength_nm']
    pd.DataFrame(data['spectra'], index=pd.Index(samples.sample_id, name='sample_id'),
                 columns=[f'channel_{i:03d}' for i in range(1, 101)]).to_csv(
                     cache_dir / 'spectra.csv')
    pd.DataFrame({'channel': np.arange(1, 101), 'wavelength_nm': wavelength}).to_csv(
        cache_dir / 'wavelengths.csv', index=False)
    samples.to_csv(cache_dir / 'samples.csv', index=False)
    duplicates = samples.loc[samples.spectrum_group_id.duplicated(keep=False)].copy()
    duplicates.sort_values(['spectrum_group_id', 'sample_id']).to_csv(
        cache_dir / 'duplicate_spectra.csv', index=False)
    train_groups = samples.loc[samples.benchmark_split.eq('train'), 'spectrum_group_id']
    test_matches_train = samples.benchmark_split.eq('test') & samples.spectrum_group_id.isin(train_groups)
    np.savez_compressed(
        cache_dir / 'tecator.npz', wavelength_nm=wavelength, spectra=data['spectra'],
        sample_id=samples.sample_id.to_numpy(),
        spectrum_group_id=samples.spectrum_group_id.to_numpy(),
        source_subset=samples.source_subset.to_numpy(dtype='U2'),
        benchmark_split=samples.benchmark_split.to_numpy(dtype='U13'),
        moisture_pct=samples.moisture_pct.to_numpy(), fat_pct=samples.fat_pct.to_numpy(),
        protein_pct=samples.protein_pct.to_numpy(), source_pca=data['source_pca'])
    (cache_dir / 'source_notes.txt').write_text(data['source_notes'], encoding='utf-8')
    metadata = {
        'source_url': SOURCE_URL,
        'source_sha256': data['source_sha256'],
        'wavelength_reference': WAVELENGTH_SOURCE,
        'wavelength_construction': '100 equally spaced points including 850 and 1050 nm',
        'n_spectra': 240, 'n_wavelengths': 100,
        'source_subset_counts': samples.source_subset.value_counts(sort=False).to_dict(),
        'benchmark_split_counts': samples.benchmark_split.value_counts(sort=False).to_dict(),
        'absorbance_definition': '-log10(transmittance)',
        'absorbance_range': [float(data['spectra'].min()), float(data['spectra'].max())],
        'missing_or_nonfinite_values': int((~np.isfinite(data['spectra'])).sum()),
        'duplicate_spectra': int(pd.DataFrame(data['spectra']).duplicated().sum()),
        'n_unique_spectra': int(samples.spectrum_group_id.nunique()),
        'test_spectra_matching_training': int(test_matches_train.sum()),
        'test_sample_ids_matching_training': samples.loc[test_matches_train, 'sample_id'].tolist(),
        'duplicate_handling': 'Preserved and labeled by spectrum_group_id; no rows removed.',
        'repeated_composition_rows': int(samples[
            ['moisture_pct', 'fat_pct', 'protein_pct']].duplicated().sum()),
        'composition_ranges_pct': {
            key: [float(samples[key].min()), float(samples[key].max())]
            for key in ['moisture_pct', 'fat_pct', 'protein_pct']},
        'preprocessing': 'None. Raw spectra preserved; source PCA retained separately.',
        'experiment_interpretation': 'One spectrum per curve; wavelength -> absorbance.',
        'permission_note': 'See source_notes.txt and tecator_raw.txt; retain with redistribution.',
    }
    (cache_dir / 'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    return metadata


def style_axes(ax, ylabel='Absorbance'):
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel(ylabel)
    ax.set_xlim(850, 1050)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.18)


def save_figure(fig, output_dir, name):
    for extension in ['png', 'pdf']:
        fig.savefig(output_dir / f'{name}.{extension}', dpi=180, facecolor='white')
    plt.close(fig)


def plot_tecator(data, output_dir=DEFAULT_OUTPUT):
    """Plot raw curves, a training-only mean, and source extrapolation subsets."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    x, y, samples = data['wavelength_nm'], data['spectra'], data['samples']
    train = samples.benchmark_split.eq('train').to_numpy()
    standard = samples.benchmark_split.ne('extrapolation').to_numpy()
    mean = y[train].mean(axis=0)
    low, high = np.quantile(y[train], [0.025, 0.975], axis=0)
    # Select actual training curves at six evenly spaced ranks of fat content.
    train_rows = np.flatnonzero(train)
    ordered = train_rows[np.argsort(samples.fat_pct.to_numpy()[train], kind='stable')]
    selected = ordered[np.rint(np.linspace(0, len(ordered) - 1, 6)).astype(int)]
    samples.iloc[selected].to_csv(output_dir / 'selected_samples.csv', index=False)
    pd.DataFrame({'wavelength_nm': x, 'training_mean': mean,
                  'training_empirical_q025': low, 'training_empirical_q975': high}).to_csv(
                      output_dir / 'training_curve_summary.csv', index=False)

    plt.rcParams.update({'font.size': 10, 'axes.titlesize': 12, 'axes.titleweight': 'bold',
                         'legend.fontsize': 9, 'savefig.bbox': 'tight'})
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.3), layout='constrained')
    fig.suptitle('Tecator: observed near-infrared spectra', fontsize=18, fontweight='bold')
    ax = axes[0, 0]
    segments = np.stack([np.broadcast_to(x, y[standard].shape), y[standard]], axis=-1)
    norm = Normalize(samples.loc[standard, 'fat_pct'].min(),
                     samples.loc[standard, 'fat_pct'].max())
    collection = LineCollection(segments, cmap='viridis', norm=norm, linewidths=0.8, alpha=0.65)
    collection.set_array(samples.loc[standard, 'fat_pct'].to_numpy())
    ax.add_collection(collection)
    ax.autoscale_view()
    ax.set_title('A  All 215 standard spectra')
    fig.colorbar(collection, ax=ax, label='Fat content (%)', pad=0.02)
    style_axes(ax)

    ax = axes[0, 1]
    ax.fill_between(x, low, high, color='#9dc9e2', alpha=0.65,
                    label='Pointwise 2.5--97.5% across curves')
    ax.plot(x, mean, color='#163e60', linewidth=2.5, label='Mean of 172 training curves')
    ax.set_title('B  Common shape and spread in training curves')
    ax.legend(loc='upper left', frameon=False)
    style_axes(ax)

    colors = plt.get_cmap('viridis')(np.linspace(0.05, 0.9, len(selected)))
    for row, color in zip(selected, colors):
        label = f"#{samples.sample_id.iloc[row]} | fat {samples.fat_pct.iloc[row]:.1f}%"
        axes[1, 0].plot(x, y[row], color=color, linewidth=2, label=label)
        axes[1, 1].plot(x, y[row] - mean, color=color, linewidth=2, label=label)
    axes[1, 0].set_title('C  Six training curves spanning fat content')
    axes[1, 0].legend(frameon=False, ncol=2, loc='upper left')
    style_axes(axes[1, 0])
    axes[1, 1].axhline(0, color='0.4', linestyle='--', linewidth=1)
    axes[1, 1].set_title('D  Same curves minus the common training mean')
    style_axes(axes[1, 1], 'Absorbance deviation')
    fig.supxlabel('Raw absorbance is preserved. The shaded band is empirical spread, not a posterior interval.',
                  fontsize=10)
    save_figure(fig, output_dir, 'spectra_overview')

    fig, axes = plt.subplots(1, 2, figsize=(13.2, 4.7), layout='constrained')
    fig.suptitle('Tecator: original standard and extrapolation subsets',
                 fontsize=16, fontweight='bold')
    ax = axes[0]
    ax.plot(x, y[standard].T, color='0.72', alpha=0.22, linewidth=0.65)
    ax.plot([], [], color='0.65', label='Standard C + M + T (215)')
    for subset, color, label in [('E1', '#d55e00', 'E1: high-fat subset (8)'),
                                  ('E2', '#0072b2', 'E2: high-protein subset (17)')]:
        values = y[samples.source_subset.eq(subset)]
        ax.plot(x, values.T, color=color, alpha=0.65, linewidth=1.1)
        ax.plot([], [], color=color, label=label)
    ax.legend(frameon=False, loc='upper left')
    ax.set_title('Observed spectra')
    style_axes(ax)
    ax = axes[1]
    for mask, color, label, marker in [
        (train, '#777777', 'Training C + M (172)', 'o'),
        (samples.source_subset.eq('T'), '#009e73', 'Test T (43)', 's'),
        (samples.source_subset.eq('E1'), '#d55e00', 'High-fat E1 (8)', '^'),
        (samples.source_subset.eq('E2'), '#0072b2', 'High-protein E2 (17)', 'D'),
    ]:
        ax.scatter(samples.loc[mask, 'fat_pct'], samples.loc[mask, 'protein_pct'],
                   s=30, c=color, label=label, marker=marker, alpha=0.75)
    ax.set_xlabel('Fat content (%)')
    ax.set_ylabel('Protein content (%)')
    ax.set_title('Measured composition (metadata)')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.18)
    ax.legend(frameon=False)
    save_figure(fig, output_dir, 'source_subsets')
    return selected


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    data = load_tecator(args.cache_dir, refresh=args.refresh)
    metadata = save_data(data, args.cache_dir)
    selected = plot_tecator(data, args.output_dir)
    summary = {**metadata, 'selected_sample_ids': data['samples'].sample_id.iloc[selected].tolist(),
               'selection_rule': 'Six evenly spaced ranks of training fat content, not a random sample.',
               'mean_and_empirical_band_fit_on': 'Training C + M only (172 curves)',
               'regression_models_fitted': False}
    (args.output_dir / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
    print(f'Data: {args.cache_dir.resolve()}\nFigures: {args.output_dir.resolve()}', flush=True)


if __name__ == '__main__':
    main()
