

import unittest

import numpy as np

from iv_methods import (
    IVPredictor,
    Prepared,
    calibration_curve,
    conformal_radius,
    last_crossing,
    weighted_radii,
)
from scipy.stats import norm


class ConstantPrediction:
    """Minimal deterministic stand-in for an independently fitted predictor."""

    def __init__(self, value):
        self.value = value

    def predict(self, x):
        return np.full(len(x), self.value)


class EstimatingEquationTests(unittest.TestCase):
    def setUp(self):
        # An exact distribution with a single X stratum and P(Z)=(1/3, 2/3).
        self.z = np.array([0, 1, 1])
        self.e_true = np.array([1.0 / 3.0, 2.0 / 3.0])
        self.m_true = np.array([[0.15, 0.30, 0.45], [0.05, 0.10, 0.20]])
        self.h = self.m_true[self.z]
        self.truth = 1.0 - self.m_true.sum(axis=0)

    def test_correct_instrument_propensity_cancels_arbitrary_regression(self):
        m_wrong = np.array([[0.02, 0.12, 0.90], [0.18, 0.35, 0.05]])
        actual = calibration_curve(
            np.broadcast_to(m_wrong, (3, 2, 3)),
            self.h,
            np.broadcast_to(self.e_true, (3, 2)),
            self.z,
        )
        np.testing.assert_allclose(actual, self.truth, atol=2e-15)

    def test_correct_regression_cancels_arbitrary_instrument_propensity(self):
        actual = calibration_curve(
            np.broadcast_to(self.m_true, (3, 2, 3)),
            self.h,
            np.broadcast_to([0.8, 0.2], (3, 2)),
            self.z,
        )
        np.testing.assert_allclose(actual, self.truth, atol=2e-15)

    def test_both_wrong_have_exact_product_remainder(self):
        m_wrong = np.array([[0.02, 0.12, 0.90], [0.18, 0.35, 0.05]])
        e_wrong = np.array([0.6, 0.4])
        actual = calibration_curve(
            np.broadcast_to(m_wrong, (3, 2, 3)),
            self.h,
            np.broadcast_to(e_wrong, (3, 2)),
            self.z,
        )
        remainder = -np.sum(
            ((e_wrong - self.e_true) / e_wrong)[:, None]
            * (m_wrong - self.m_true),
            axis=0,
        )
        np.testing.assert_allclose(actual - self.truth, remainder, atol=2e-15)

    def test_multi_arm_correction_uses_actual_instrument_only(self):
        m = np.array(
            [
                [[0.1, 0.3], [0.2, 0.4], [0.0, 0.1]],
                [[0.2, 0.2], [0.1, 0.2], [0.3, 0.4]],
                [[0.0, 0.1], [0.1, 0.3], [0.1, 0.2]],
                [[0.1, 0.2], [0.2, 0.3], [0.2, 0.2]],
            ]
        )
        z = np.array([2, 0, 1, 2])
        e = np.array([[0.2, 0.3, 0.5], [0.3, 0.3, 0.4],
                      [0.5, 0.4, 0.1], [0.1, 0.2, 0.7]])
        # Includes an unobserved row: h=0, while its augmentation is nonzero.
        h = np.array([[0.0, 0.0], [0.0, 0.8], [0.2, 0.4], [0.0, 0.3]])
        expected = []
        for t in range(2):
            row_values = []
            for i in range(4):
                identified_mass = 0.0
                for arm in range(3):
                    correction = 0.0
                    if z[i] == arm:
                        correction = (h[i, t] - m[i, arm, t]) / e[i, arm]
                    identified_mass += m[i, arm, t] + correction
                row_values.append(1.0 - identified_mass)
            expected.append(np.mean(row_values))
        np.testing.assert_allclose(calibration_curve(m, h, e, z), expected)


class ThresholdTests(unittest.TestCase):
    def test_nonmonotone_curve_uses_last_crossing(self):
        grid = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
        curve = np.array([1.0, 0.4, 0.7, 0.3, 0.1])
        self.assertAlmostEqual(last_crossing(grid, curve, 0.1, 0.5), 2.5)

    def test_infinite_tail_despite_earlier_crossing(self):
        self.assertTrue(np.isinf(last_crossing(
            np.array([0.0, 1.0, 2.0]), np.array([1.0, 0.1, 0.7]), 0.7, 0.5
        )))

    def test_equality_in_constant_tail_is_infinite(self):
        self.assertTrue(np.isinf(last_crossing(
            np.array([0.0, 1.0, 2.0]), np.array([1.0, 0.1, 0.5]), 0.5, 0.5
        )))

    def test_tied_crossing_keeps_rightmost_equality(self):
        actual = last_crossing(
            np.array([0.0, 1.0, 2.0, 3.0]),
            np.array([1.0, 0.5, 0.5, 0.1]), 0.1, 0.5,
        )
        self.assertAlmostEqual(actual, 2.0)


class ConformalTests(unittest.TestCase):
    def test_ordinary_conformal_finite_sample_order_statistic(self):
        scores = np.array([3.0, 1.0, 2.0])
        self.assertEqual(conformal_radius(scores, 0.25), 3.0)
        self.assertEqual(conformal_radius(scores, 0.50), 2.0)
        self.assertTrue(np.isinf(conformal_radius(scores, 0.10)))

    def test_equal_weights_reproduce_ordinary_conformal(self):
        scores = np.array([4.0, 1.0, 2.0, 2.0, 7.0, 5.0])
        for alpha in [0.05, 0.10, 0.20, 0.30, 0.50]:
            expected = conformal_radius(scores, alpha)
            actual = weighted_radii(scores, np.ones(6), np.ones(4), alpha)
            np.testing.assert_array_equal(actual, np.repeat(expected, 4))

    def test_test_point_infinite_atom_can_make_radius_infinite(self):
        actual = weighted_radii(
            np.array([1.0, 2.0, 3.0]), np.ones(3), np.array([1.0, 4.0]), 0.25
        )
        np.testing.assert_array_equal(actual, [3.0, np.inf])

    def test_weighted_ties_and_common_rescaling(self):
        scores = np.array([2.0, 5.0, 2.0])
        weights = np.array([1.0, 1.0, 3.0])
        test_weights = np.array([1.0, 2.0])
        base = weighted_radii(scores, weights, test_weights, 0.5)
        scaled = weighted_radii(scores, 7.0 * weights, 7.0 * test_weights, 0.5)
        np.testing.assert_array_equal(base, [2.0, 2.0])
        np.testing.assert_array_equal(base, scaled)


class NuisanceAndIntegrationTests(unittest.TestCase):
    def test_gaussian_integration_matches_analytic_identified_mass(self):
        model = IVPredictor(known_e=[0.5, 0.5], grid_size=1201)
        model.k = 2
        model.score = ConstantPrediction(0.0)
        model.response_model = ConstantPrediction(0.6)
        model.policy = ConstantPrediction(0)
        model.parameters = lambda x: (
            np.full((len(x), 2), 0.6),
            np.zeros((len(x), 2)),
            np.ones((len(x), 2)),
        )
        prepared = model.prepare(np.zeros((2, 1)))
        mass = 2 * norm.cdf(prepared.grid) - 1
        # Equal densities give soft weights 1/2; the hard policy always uses 0.
        np.testing.assert_allclose(prepared.m_soft[0, 0], 0.3 * mass, atol=1e-6)
        np.testing.assert_allclose(prepared.m_soft[0, 1], 0.3 * mass, atol=1e-6)
        np.testing.assert_allclose(prepared.m_policy[0, 0], 0.6 * mass, atol=2e-6)
        np.testing.assert_array_equal(prepared.m_policy[:, 1], 0.0)
        np.testing.assert_allclose(prepared.gamma_plugin, 0.6, atol=1e-12)

    def test_full_calibration_resolves_both_sides_of_score_jumps(self):
        model = IVPredictor()
        model.location, model.scale = 0.0, 1.0
        model.weights_at_observed = lambda x, y, r, z: (r.astype(float), r.astype(float))
        prepared = Prepared(
            grid=np.array([0.0, 10.0]),
            m_soft=np.zeros((4, 1, 2)),
            m_policy=np.zeros((4, 1, 2)),
            e=np.ones((4, 1)),
            response=np.ones(4),
            center=np.zeros(4),
            gamma_plugin=np.ones(4),
        )
        result, _, _ = model.calibrate(
            prepared, np.arange(4), np.zeros(4, dtype=int), np.ones(4, dtype=bool),
            np.array([1.0, 2.0, 3.0, 4.0]), np.zeros((4, 1)), [0.5],
        )
        # U(t)=0.5 throughout [2,3), so its superlevel set has supremum 3.
        self.assertAlmostEqual(result["Softmax IV"][0], 3.0)
        self.assertAlmostEqual(result["Policy IV"][0], 3.0)

    def test_poisoning_missing_outcomes_cannot_change_fitting_or_weights(self):
        rng = np.random.default_rng(631)
        x = rng.normal(size=(100, 2))
        z = np.arange(100) % 2
        r = np.arange(100) % 5 != 0
        y = x[:, 0] + rng.normal(size=100)
        missing_nan, missing_poison = y.copy(), y.copy()
        missing_nan[~r] = np.nan
        missing_poison[~r] = 1e200
        options = dict(seed=19, known_e=[0.5, 0.5], n_trees=8, min_leaf=5)
        a = IVPredictor(**options).fit(x, z, r, missing_nan)
        b = IVPredictor(**options).fit(x, z, r, missing_poison)
        self.assertEqual(a.location, b.location)
        self.assertEqual(a.scale, b.scale)
        for actual, expected in zip(a.parameters(x[:7]), b.parameters(x[:7])):
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(a.score.predict(x[:7]), b.score.predict(x[:7]))
        for actual, expected in zip(
            a.weights_at_observed(x, missing_nan, r, z),
            b.weights_at_observed(x, missing_poison, r, z),
        ):
            np.testing.assert_array_equal(actual, expected)
            np.testing.assert_array_equal(actual[~r], 0.0)


if __name__ == "__main__":
    unittest.main()
