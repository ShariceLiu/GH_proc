"""Validate the added ARD baselines, seven-model scores and CRPS artifacts."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numpy.testing import assert_allclose, assert_array_equal
from threadpoolctl import threadpool_limits

from GH_tabular_hmc import MODELS, ZERO_MODELS, ALL_MODELS, default_center, has_mixing
from hmc_airfoil_experiment import write_json
from hmc_airfoil_zero_experiment import case_folder
from validate_airfoil_experiment import validate as validate_fits


def validate(root):
    root = Path(root)
    folder = root/'zero_mean_baselines'
    # Reuse the independently implemented block-Gaussian and density checks on
    # the new fits, placing their validation report in the new model directory.
    checked = validate_fits(root, models=ZERO_MODELS, output_path=folder/'prediction_validation.json',
                            model_root=folder)
    assert checked['all_hmc_diagnostics_ok']
    hashes = json.loads((folder/'source_provenance.json').read_text())
    for name, expected in hashes.items():
        assert hashlib.sha256((root/name).read_bytes()).hexdigest() == expected
    with np.load(root/'dataset.npz') as saved:
        data = {key: saved[key].copy() for key in saved.files}
    test = data['test_ids']
    original = pd.read_csv(root/'scores.csv')
    combined = pd.read_csv(root/'scores_with_zero_mean_crps.csv')
    pd.testing.assert_frame_equal(original, combined.iloc[:len(MODELS)][original.columns].reset_index(drop=True))
    rows = []
    for model in ALL_MODELS:
        source = case_folder(root, model)
        report = json.loads((source/'report.json').read_text())
        assert report['diagnostics_ok']
        if model in ZERO_MODELS:
            assert_allclose(report['prior_sd'], 2.)
            center = default_center(data['y_scaled'][data['train_ids']], data['x'].shape[1])
            assert_allclose(report['prior_center'], center[np.r_[0, 2, np.arange(3, 8)]])
            factors = np.r_[np.full(2, float(data['y_scale'])**2), 1./data['x_scale']**2]
            if has_mixing(model):
                factors = np.r_[factors, 1.]
            for j, (std, orig) in enumerate(zip(report['parameters'], report['parameters_original_units'])):
                for key in ['mean', 'std', 'lower', 'median', 'upper']:
                    assert_allclose(orig[key], std[key]*factors[j])
        crps = json.loads((root/'crps'/f'{model}.json').read_text())
        assert crps['validation_passed'] and crps['all_components_used']
        assert hashlib.sha256((source/'predictions.npz').read_bytes()).hexdigest() == crps['provenance']['prediction_sha256']
        assert hashlib.sha256((root/'dataset.npz').read_bytes()).hexdigest() == crps['provenance']['dataset_sha256']
        with np.load(root/'crps'/f'{model}.npz') as saved:
            assert_array_equal(saved['test_ids'], test)
            values = saved['y_crps']
        assert values.shape == (1203,) and np.all(np.isfinite(values)) and np.all(values >= 0)
        assert_allclose(values.mean(), crps['y_crps'], atol=1e-12)
        assert_allclose(combined.loc[combined['model'] == model, 'y_crps'].iloc[0], crps['y_crps'], atol=1e-12)
        for key in ['y_rmse', 'y_mean_marginal_nlpd', 'y_coverage_95', 'y_mean_interval_width']:
            assert_allclose(combined.loc[combined['model'] == model, key].iloc[0], report[key], atol=1e-12)
        assert crps['numerical']['n_components'] == report['settings']['chains']*report['settings']['draws']
        errors = np.array(crps['numerical']['integration_error_estimate'])
        tails = np.array(crps['numerical']['tail_error_bound'])
        assert np.all(errors+tails <= 1e-8+1e-8*values)
        rows.append(dict(model=model, y_crps=crps['y_crps'],
                         max_numerical_error_estimate=float(np.max(errors+tails)),
                         n_independent_checks=len(crps['independent_checks']),
                         max_independent_difference=max(c['absolute_difference'] for c in crps['independent_checks'])))
    result = dict(validation_passed=True, original_four_models_unchanged=True,
                  identical_split_and_preprocessing=True, scores_in_original_units=True,
                  latent_f_not_scored=True, predictions=checked, crps=rows)
    write_json(folder/'validation.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ['predictions', 'crps']}, indent=2), flush=True)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('figure/airfoil_single_function_hmc'))
    args = parser.parse_args()
    with threadpool_limits(limits=1):
        validate(args.root)
