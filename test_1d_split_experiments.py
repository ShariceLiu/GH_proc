"""Check experimental boundaries and exclusion of held-out observations."""

import unittest

import numpy as np
from numpy.testing import assert_array_equal, assert_allclose

from hmc_1d_split_experiments import make_split, prior_center_from_training, SCENARIOS


class TestSplits(unittest.TestCase):
    def setUp(self):
        x = np.linspace(0., 10., 181)
        self.data = dict(x=x, y=np.sin(x), f=np.cos(x),
                         train_ids=np.arange(0, 181, 2), test_ids=np.arange(1, 181, 2))

    def test_counts_regions_and_disjointness(self):
        counts = {'sparse': (15, 166), 'gap': (72, 37), 'extrapolation': (55, 72)}
        for scenario in SCENARIOS:
            with self.subTest(scenario=scenario):
                d = make_split(self.data, scenario)
                self.assertEqual((len(d['train_ids']), len(d['test_ids'])), counts[scenario])
                self.assertTrue(np.isin(d['train_ids'], self.data['train_ids']).all())
                self.assertEqual(len(np.intersect1d(d['train_ids'], d['test_ids'])), 0)
                assert_array_equal(np.sort(np.r_[d['train_ids'], d['test_ids'], d['unused_ids']]), np.arange(181))
                assert_array_equal(d['x'], self.data['x'])
                assert_array_equal(d['y'], self.data['y'])
                xt, xs = d['x'][d['train_ids']], d['x'][d['test_ids']]
                if scenario == 'gap':
                    self.assertTrue(np.all((xt < 4) | (xt > 6)))
                    self.assertTrue(np.all((xs >= 4) & (xs <= 6)))
                elif scenario == 'extrapolation':
                    self.assertTrue(np.all(xt <= 6))
                    self.assertTrue(np.all(xs > 6))
                else:
                    assert_allclose(xt[[0, -1]], [0, 10])

    def test_held_out_values_and_truth_do_not_change_split_or_prior(self):
        for scenario in SCENARIOS:
            d = make_split(self.data, scenario)
            changed = dict(self.data, y=self.data['y'].copy(), f=self.data['f']*1e6)
            changed['y'][d['all_unobserved_ids']] = 1e9
            other = make_split(changed, scenario)
            assert_array_equal(d['train_ids'], other['train_ids'])
            assert_array_equal(d['test_ids'], other['test_ids'])
            assert_array_equal(d['y'][d['train_ids']], other['y'][other['train_ids']])
            assert_allclose(prior_center_from_training(d['y'][d['train_ids']]),
                            prior_center_from_training(other['y'][other['train_ids']]))

    def test_invalid_split(self):
        for count in [1, 91, 15.5]:
            with self.assertRaises(ValueError):
                make_split(self.data, 'sparse', count)
        with self.assertRaises(ValueError):
            make_split(self.data, 'other')


if __name__ == '__main__':
    unittest.main()
