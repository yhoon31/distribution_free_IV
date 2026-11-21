
import unittest

import numpy as np

from iv_methods import conformal_radius
from run_feedback_application import completion_bounds, disjoint_splits, width_summary


class CompletionBoundsTests(unittest.TestCase):
    def test_brackets_every_missing_score_completion(self):
        observed = np.array([True, False, True, False, True, True])
        known_scores = np.array([.2, .9, 1.7, 3.0])
        for alpha in [.1, .2, .4, .7]:
            best, worst = completion_bounds(known_scores, observed, alpha)
            for left in [0., .05, 1., 20., 1e12]:
                for right in [0., .4, 2., 200.]:
                    completed = np.zeros(len(observed))
                    completed[observed] = known_scores
                    completed[~observed] = [left, right]
                    radius = conformal_radius(completed, alpha)
                    self.assertLessEqual(best, radius)
                    self.assertLessEqual(radius, worst)

    def test_no_missing_matches_fully_observed_conformal(self):
        scores = np.array([2., 1., 5., 4.])
        for alpha in [.1, .2, .5]:
            expected = conformal_radius(scores, alpha)
            self.assertEqual(completion_bounds(scores, np.ones(4, bool), alpha),
                             (expected, expected))

    def test_finite_sample_rank_can_exceed_observed_count(self):
        best, worst = completion_bounds(np.arange(1., 10.),
                                       np.array([True]*9+[False]), .1)
        self.assertEqual(best, 9.)
        self.assertTrue(np.isinf(worst))

    def test_missing_outcome_scores_rejected(self):
        with self.assertRaises(ValueError):
            completion_bounds(np.array([1., np.nan]), np.array([True, False, True]), .1)


class SplittingAndSummaryTests(unittest.TestCase):
    def test_capped_splits_keep_all_groups_disjoint(self):
        groups = np.repeat(np.arange(60), 4)
        train, calibration, target = disjoint_splits(groups, 91, 35, 20)
        self.assertEqual(len(train), 35)
        self.assertEqual(len(calibration), 20)
        for a, b in [(train, calibration), (train, target), (calibration, target)]:
            self.assertFalse(set(groups[a]) & set(groups[b]))
        again = disjoint_splits(groups, 91, 35, 20)
        for actual, expected in zip(again, (train, calibration, target)):
            np.testing.assert_array_equal(actual, expected)

    def test_unconditional_width_summaries_preserve_infinity(self):
        summary = width_summary([2., np.inf, np.inf])
        self.assertTrue(np.isinf(summary['mean_length']))
        self.assertTrue(np.isinf(summary['median_length']))
        self.assertEqual(summary['finite_mean_length'], 2.)
        self.assertAlmostEqual(summary['infinite_fraction'], 2/3)


if __name__ == '__main__':
    unittest.main()
