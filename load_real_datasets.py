"""Load the six regression datasets as pandas (X, y) pairs, with local CSV caches.

Run: python load_real_datasets.py
Missing dependency in the original Anaconda environment: ucimlrepo==0.0.7.
The first download requires internet; subsequent runs use the workspace cache.
No encoding, scaling, splitting or model fitting is performed here.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml, fetch_california_housing


UCI_IDS = {'concrete': 165, 'airfoil': 291, 'power': 294}
NAMES = (*UCI_IDS, 'insurance', 'kin8nm', 'california')
DEFAULT_CACHE = Path(__file__).resolve().parent/'data'/'real_datasets'
INSURANCE_URL = 'https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv'


def load_datasets(names=NAMES, *, cache_dir=DEFAULT_CACHE, refresh=False):
    """Return {name: (DataFrame, numeric Series)} in original target units."""
    cache_dir = Path(cache_dir)
    datasets = {}
    for name in names:
        if name not in NAMES:
            raise ValueError(f'Unknown dataset: {name}')
        folder = cache_dir/name
        x_path, y_path = folder/'features.csv', folder/'target.csv'
        if not refresh and x_path.exists() and y_path.exists():
            X, targets = pd.read_csv(x_path), pd.read_csv(y_path)
            if targets.shape[1] != 1:
                raise ValueError(f'{name}: expected exactly one cached target')
            y = targets.iloc[:, 0]
        else:
            if name in UCI_IDS:
                try:
                    from ucimlrepo import fetch_ucirepo
                except ImportError as exc:
                    raise ImportError('UCI downloads require ucimlrepo. Install with: '
                                      'python -m pip install ucimlrepo==0.0.7') from exc
                data = fetch_ucirepo(id=UCI_IDS[name])
                X, targets = data.data.features, data.data.targets
                if targets is None or targets.shape[1] != 1:
                    raise ValueError(f'{name}: expected exactly one UCI target')
                y = targets.iloc[:, 0]
            elif name == 'insurance':
                insurance = pd.read_csv(INSURANCE_URL)
                X, y = insurance.drop(columns='charges'), insurance['charges']
            elif name == 'kin8nm':
                X, y = fetch_openml(data_id=44980, as_frame=True, return_X_y=True,
                                    parser='pandas', data_home=str(cache_dir/'sklearn_cache'))
            else:
                X, y = fetch_california_housing(as_frame=True, return_X_y=True,
                                                data_home=str(cache_dir/'sklearn_cache'))
            if not isinstance(X, pd.DataFrame) or not isinstance(y, pd.Series):
                raise TypeError(f'{name}: expected pandas features and a single target Series')
            if not X.index.equals(y.index):
                raise ValueError(f'{name}: feature and target rows are misaligned')
            y = pd.to_numeric(y, errors='raise')
            folder.mkdir(parents=True, exist_ok=True)
            X.to_csv(x_path, index=False)
            y.to_csv(y_path, index=False)
        if len(X) != len(y) or not len(X):
            raise ValueError(f'{name}: empty or mismatched feature/target rows')
        datasets[name] = X, pd.to_numeric(y, errors='raise')
    return datasets


def describe_datasets(datasets):
    rows = []
    for name, (X, y) in datasets.items():
        numeric = X.select_dtypes(include='number')
        rows.append(dict(dataset=name, n_rows=len(X), n_features=X.shape[1],
                         target=str(y.name), target_dtype=str(y.dtype),
                         categorical_columns=list(X.select_dtypes(exclude='number').columns),
                         missing_X=int(X.isna().sum().sum()), missing_y=int(y.isna().sum()),
                         infinite_numeric_X=int(np.isinf(numeric.to_numpy(dtype=float)).sum()),
                         infinite_y=int(np.isinf(y.to_numpy(dtype=float)).sum()),
                         duplicate_feature_rows=int(X.duplicated().sum()),
                         duplicate_full_rows=int(pd.concat([X, y.rename('__target__')], axis=1).duplicated().sum())))
    return rows


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=NAMES, default=list(NAMES))
    parser.add_argument('--cache-dir', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--refresh', action='store_true')
    args = parser.parse_args()
    result = {}
    for name in args.datasets:
        print(f'Loading {name}...', flush=True)
        result.update(load_datasets([name], cache_dir=args.cache_dir, refresh=args.refresh))
        X, y = result[name]
        print(f'{name}: X={X.shape}, y={y.shape}', flush=True)
    summary = describe_datasets(result)
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    (args.cache_dir/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    print(json.dumps(summary, indent=2), flush=True)
