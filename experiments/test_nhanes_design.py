
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

import run_experiments as runner


def survey_frame():
    """Unequal PSU sizes make accidental splitting of a group detectable."""
    groups = np.repeat(np.arange(24), np.arange(9, 33))
    return pd.DataFrame({
        "cluster_id": groups,
        "household_id": np.arange(len(groups)),
        "row_id": np.arange(len(groups)),
    })


class SurveySplitTests(unittest.TestCase):
    def test_group_prefix_selects_whole_groups_and_first_crossing(self):
        frame = survey_frame()
        selected, groups = runner.take_group_prefix(
            frame, "cluster_id", 100, np.random.default_rng(74)
        )
        self.assertGreaterEqual(selected.sum(), 100)
        self.assertLess(frame.cluster_id.isin(groups[:-1]).sum(), 100)
        for _, values in pd.Series(selected).groupby(frame.cluster_id):
            self.assertEqual(values.nunique(), 1)

    def test_training_and_reservoir_are_psu_disjoint_with_fixed_caps(self):
        frame = survey_frame()
        for seed in range(20):
            with self.subTest(seed=seed):
                train, pool = runner.cluster_split_pools(frame, seed, 120, 200)
                self.assertEqual(len(train), 120)
                self.assertEqual(len(pool), 200)
                self.assertEqual(len(np.unique(train)), len(train))
                self.assertEqual(len(np.unique(pool)), len(pool))
                self.assertFalse(set(frame.cluster_id.iloc[train]) &
                                 set(frame.cluster_id.iloc[pool]))

    def test_pilot_training_and_reservoir_psus_are_pairwise_disjoint(self):
        frame = survey_frame()
        pilot, _ = runner.take_group_prefix(
            frame, "cluster_id", 60, np.random.default_rng(81)
        )
        remaining = frame.loc[~pilot].reset_index(drop=True)
        train, pool = runner.cluster_split_pools(remaining, 93, 120, 200)
        partitions = [set(frame.cluster_id[pilot]),
                      set(remaining.cluster_id.iloc[train]),
                      set(remaining.cluster_id.iloc[pool])]
        for i in range(len(partitions)):
            for j in range(i):
                self.assertFalse(partitions[i] & partitions[j])

    def test_calibration_and_test_psus_are_disjoint_on_repeated_trials(self):
        frame = survey_frame()
        _, pool = runner.cluster_split_pools(frame, 54, 120, 200)
        reservoir = frame.iloc[pool]
        rng = np.random.default_rng(123)
        for trial in range(30):
            with self.subTest(trial=trial):
                calibration, test = runner.trial_indices(reservoir, rng, 60, 70, "psu")
                self.assertEqual(len(calibration), 60)
                self.assertEqual(len(test), 70)
                self.assertFalse(set(calibration) & set(test))
                self.assertFalse(set(reservoir.cluster_id.iloc[calibration]) &
                                 set(reservoir.cluster_id.iloc[test]))

    def test_impossible_psu_trial_raises_instead_of_reusing_cluster(self):
        frame = pd.DataFrame({"cluster_id": np.zeros(12, dtype=int)})
        with self.assertRaisesRegex(ValueError, "disjoint pseudo-PSU"):
            runner.trial_indices(frame, np.random.default_rng(4), 5, 5, "psu")

    def test_missing_psu_identifier_raises(self):
        with self.assertRaisesRegex(ValueError, "cluster_id"):
            runner.cluster_split_pools(pd.DataFrame({"household_id": [1, 2, 3]}), 3)

    def test_family_trials_preserve_fes_draws_and_rng_continuation(self):
        # The pre-NHANES FES runner used one permutation per trial. Subsequent
        # draws represent the masking RNG that must also remain unchanged.
        frame = pd.DataFrame({"household_id": np.arange(1500)})
        original_rng = np.random.default_rng(20261818)
        current_rng = np.random.default_rng(20261818)
        for trial in range(10):
            with self.subTest(trial=trial):
                old_order = original_rng.permutation(len(frame))
                calibration, test = runner.trial_indices(frame, current_rng, 500, 500, "family")
                np.testing.assert_array_equal(calibration, old_order[:500])
                np.testing.assert_array_equal(test, old_order[500:1000])
                np.testing.assert_array_equal(current_rng.integers(0, 5, 500),
                                              original_rng.integers(0, 5, 500))
                np.testing.assert_array_equal(current_rng.random(500), original_rng.random(500))


class MissingOutcomeMetricsTests(unittest.TestCase):
    @staticmethod
    def metrics(y, observed, radii, known=False):
        return runner.metric_rows(
            "synthetic", "natural", 0, 0.1, "Synthetic method", np.asarray(radii),
            np.zeros(len(y)), np.asarray(y), np.asarray(observed), 2.0, known, np.nan,
        )

    def test_bounds_are_sharp_for_mixture_of_finite_and_infinite_sets(self):
        values = [0.0, 4.0, np.nan, np.nan]
        observed = [True, True, False, False]
        radii = [1.0, 1.0, 1.0, np.inf]
        result = self.metrics(values, observed, radii)
        self.assertTrue(np.isnan(result["coverage"]))
        self.assertEqual(result["observed_coverage"], 0.5)
        self.assertEqual(result["coverage_lower"], 0.5)
        self.assertEqual(result["coverage_upper"], 0.75)
        # Unknown outcomes can attain both endpoints. The infinite set must
        # cover its missing outcome even at the pessimistic endpoint.
        lower = self.metrics([0.0, 4.0, 10.0, 10.0], [True] * 4, radii, known=True)
        upper = self.metrics([0.0, 4.0, 0.0, 10.0], [True] * 4, radii, known=True)
        self.assertEqual(lower["coverage"], result["coverage_lower"])
        self.assertEqual(upper["coverage"], result["coverage_upper"])
        self.assertTrue(np.isinf(result["mean_length"]))
        self.assertEqual(result["median_length"], 4.0)
        self.assertEqual(result["finite_median_length"], 4.0)
        self.assertEqual(result["infinite_fraction"], 0.25)

    def test_hidden_values_cannot_change_reported_metrics(self):
        clean = self.metrics([0.0, 4.0, np.nan, np.nan], [True, True, False, False],
                             [1.0, 1.0, 1.0, np.inf])
        poisoned = self.metrics([0.0, 4.0, 1e200, -1e200], [True, True, False, False],
                                [1.0, 1.0, 1.0, np.inf])
        for name in ["observed_coverage", "coverage_lower", "coverage_upper",
                     "mean_length", "median_length", "finite_median_length", "infinite_fraction"]:
            self.assertEqual(clean[name], poisoned[name])

    def test_no_observed_outcomes_gives_no_empirical_coverage(self):
        result = self.metrics([np.nan, np.nan], [False, False], [1.0, 1.0])
        self.assertTrue(np.isnan(result["coverage"]))
        self.assertTrue(np.isnan(result["observed_coverage"]))
        self.assertEqual(result["coverage_lower"], 0.0)
        self.assertEqual(result["coverage_upper"], 1.0)

    def test_all_infinite_sets_have_known_full_coverage_despite_missing_outcomes(self):
        result = self.metrics([0.0, np.nan], [True, False], [np.inf, np.inf])
        self.assertEqual(result["coverage_lower"], 1.0)
        self.assertEqual(result["coverage_upper"], 1.0)
        self.assertTrue(np.isinf(result["median_length"]))
        self.assertTrue(np.isnan(result["finite_median_length"]))

    def test_complete_outcomes_collapse_bounds_to_measured_coverage(self):
        result = self.metrics([0.0, 2.0, 5.0], [True] * 3, [1.0] * 3, known=True)
        self.assertAlmostEqual(result["coverage"], 2.0 / 3.0)
        self.assertEqual(result["coverage_lower"], result["coverage"])
        self.assertEqual(result["coverage_upper"], result["coverage"])


class ScenarioCohortTests(unittest.TestCase):
    def test_natural_cohort_uses_its_own_rows_and_feature_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            data = base / "data" / "nhanes"
            data.mkdir(parents=True)
            pd.DataFrame({"row_id": [1, 2], "age": [40, 50], "outcome": [13.0, np.nan],
                          "observed": [1, 0], "household_id": [1, 2],
                          "instrument": [0, 1]}).to_csv(data / "prepared.csv", index=False)
            (data / "prepared_metadata.json").write_text(json.dumps({
                "numeric_features": ["age"], "categorical_features": [], "cohort": "broad",
            }))
            pd.DataFrame({"row_id": [2], "age": [50], "region": ["north"],
                          "outcome": [np.nan], "observed": [0],
                          "household_id": [2], "instrument": [1]}).to_csv(
                              data / "prepared_natural.csv", index=False)
            (data / "prepared_natural_metadata.json").write_text(json.dumps({
                "numeric_features": ["age"], "categorical_features": ["region"], "cohort": "natural",
            }))
            with patch.object(runner, "BASE", base):
                broad, broad_x, broad_meta, broad_features = runner.load_data("nhanes")
                natural, natural_x, natural_meta, natural_features = runner.load_data("nhanes", "natural")
            self.assertEqual(broad.row_id.tolist(), [1, 2])
            self.assertEqual(natural.row_id.tolist(), [2])
            self.assertEqual(broad_meta["cohort"], "broad")
            self.assertEqual(natural_meta["cohort"], "natural")
            self.assertEqual(broad_features, ["age"])
            self.assertEqual(natural_features, ["age", "region_north"])
            self.assertEqual(broad_x.shape, (2, 1))
            self.assertEqual(natural_x.shape, (1, 2))
            self.assertTrue(natural.outcome.isna().all())

    def test_natural_loader_falls_back_to_prepared_cohort(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            data = base / "data" / "heckman"
            data.mkdir(parents=True)
            pd.DataFrame({"age": [40], "row_id": [1], "household_id": [1],
                          "instrument": [0], "outcome": [13.0], "observed": [1]}).to_csv(
                              data / "prepared.csv", index=False)
            (data / "prepared_metadata.json").write_text(json.dumps({
                "numeric_features": ["age"], "categorical_features": [],
            }))
            with patch.object(runner, "BASE", base):
                frame, _, _, _ = runner.load_data("heckman", "natural")
            self.assertEqual(frame.age.tolist(), [40])


if __name__ == "__main__":
    unittest.main()
