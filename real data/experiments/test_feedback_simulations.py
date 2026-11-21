
import unittest

import numpy as np
import pandas as pd
from scipy.stats import norm

from feedback_simulations import (
    SCENARIOS,
    encode_instrument,
    perturb_outcome,
    response_table,
)


class FeedbackMechanismTests(unittest.TestCase):
    def setUp(self):
        self.v = np.array([-3.1, -1.2, -.35, .2, .9, 2.4])
        self.u = np.array([-2., -.8, .3, 1.1, -.5, 2.])
        self.mass = np.array([.07, .11, .23, .31, .19, .09])

    def test_probabilities_are_finite_and_valid_for_all_designs(self):
        for k in (3, 5, 6):
            for scenario in SCENARIOS:
                with self.subTest(k=k, scenario=scenario):
                    p = response_table(scenario, self.v, self.u, k)
                    self.assertEqual(p.shape, (len(self.v), k))
                    self.assertTrue(np.isfinite(p).all())
                    self.assertTrue(((p >= 0) & (p <= 1)).all())

    def test_heterogeneous_design_has_exact_identified_mass_one(self):
        # For a valid randomized IV, q_z at an outcome atom is its full
        # outcome mass times the arm's response probability. No parametric
        # outcome distribution is used in this identity.
        for k in (3, 5, 6):
            p = response_table('heterogeneous_mnar', self.v, self.u, k)
            q = self.mass[:, None] * p
            self.assertAlmostEqual(q.max(axis=1).sum(), 1., places=14)
            np.testing.assert_allclose(p.max(axis=1), 1., atol=2e-15)

    def test_every_arm_can_be_best_and_nonwinning_arms_are_not_identical(self):
        for k in (3, 5, 6):
            # At the stated arm centers, each arm must uniquely maximize
            # observation probability for its corresponding outcome.
            v = norm.ppf((np.arange(k) + .5) / k)
            p = response_table('heterogeneous_mnar', v, np.zeros(k), k)
            np.testing.assert_array_equal(p.argmax(axis=1), np.arange(k))
            # An asymmetric residual avoids incidental symmetric ties.
            row = response_table('heterogeneous_mnar', [.173], [0.], k)[0]
            self.assertEqual(len(np.unique(row)), k)

    def test_weakening_preserves_outcome_selection_but_reduces_identification(self):
        strong = response_table('heterogeneous_mnar', self.v, self.u)
        mean_response = float(self.mass @ strong.mean(axis=1))
        previous_index = 1.
        for scenario, strength in [('weak_mnar', .25), ('irrelevant_mnar', 0.)]:
            p = response_table(scenario, self.v, self.u)
            # Uniform Z means row averages specify the entire marginal
            # selection law conditional on X and Y, not just its mean.
            np.testing.assert_allclose(p.mean(axis=1), strong.mean(axis=1), atol=2e-15)
            np.testing.assert_allclose(np.ptp(p, axis=1),
                                       strength * np.ptp(strong, axis=1), atol=2e-15)
            identified_mass = float((self.mass[:, None] * p).max(axis=1).sum())
            expected = strength + (1-strength) * mean_response
            self.assertAlmostEqual(identified_mass, expected, places=14)
            self.assertLess(identified_mass, previous_index)
            previous_index = identified_mass

    def test_irrelevant_instrument_remains_mnar(self):
        p = response_table('irrelevant_mnar', self.v, np.zeros_like(self.u))
        np.testing.assert_allclose(p, np.repeat(p[:, :1], 5, axis=1))
        self.assertGreater(np.ptp(p[:, 0]), .01)

    def test_mar_probabilities_ignore_outcomes_and_shift_covariates(self):
        for scenario in ('mar_x', 'mar_xz', 'observed_iv_mar'):
            p = response_table(scenario, self.v, self.u)
            changed = response_table(scenario, self.v * 1e6 + 1e6, self.u)
            np.testing.assert_array_equal(p, changed)
            response = p.mean(axis=1)
            observed_u_mean = np.average(self.u, weights=self.mass * response)
            self.assertGreater(observed_u_mean, self.mass @ self.u)
        p_x = response_table('mar_x', self.v, self.u)
        np.testing.assert_allclose(p_x, np.repeat(p_x[:, :1], 5, axis=1))
        p_xz = response_table('mar_xz', self.v, self.u)
        self.assertTrue((np.ptp(p_xz, axis=1) > 0).all())

    def test_known_mar_weights_restore_full_joint_and_marginal_laws(self):
        # Arbitrary finite X,Y mass with an independent uniform randomized Z.
        # Observed submass divided by actual p recovers the full X,Y,Z law;
        # marginalization over Z followed by 1/mean_z p recovers X,Y.
        for scenario in ('mar_x', 'mar_xz'):
            p = response_table(scenario, self.v, self.u)
            full_joint = np.repeat(self.mass[:, None] / 5, 5, axis=1)
            observed_joint = full_joint * p
            np.testing.assert_allclose(observed_joint / p, full_joint)
            np.testing.assert_allclose(observed_joint.sum(axis=1) / p.mean(axis=1),
                                       self.mass)

    def test_native_mar_requires_xz_weights_without_exclusion(self):
        # At fixed X, make Y=Z, invalidating Y independent of Z given X.
        # MAR(X,Z) still holds, but inverse P(R|X) cannot restore the Y law.
        joint_mass = np.array([.3, .7])
        z = np.array([0, 1])
        p = response_table('observed_iv_mar', [-1., 1.], [0., 0.], 2)
        actual = p[np.arange(2), z]
        observed = joint_mass * actual
        np.testing.assert_allclose(observed / actual, joint_mass)
        marginal_p = joint_mass @ actual
        self.assertFalse(np.allclose(observed / marginal_p, joint_mass))

    def test_direct_effect_changes_outcomes_using_each_rows_assigned_instrument(self):
        base = np.array([-1.2, .1, .9, 2.5, 4.])
        z = np.array([4, 0, 3, 1, 2])
        np.testing.assert_array_equal(perturb_outcome(base, z, 5, 1.7, 0.), base)
        shifted = perturb_outcome(base, z, 5, 1.7, .5)
        np.testing.assert_allclose(shifted-base, np.array([1., -1., .5, -.5, 0.])*.85)
        # Repeat an identical pre-shift outcome over each instrument arm:
        # the conditional outcome means now differ, proving exclusion fails.
        equal_base = np.full(5, 3.)
        conditional_means = perturb_outcome(equal_base, np.arange(5), 5, 1.7, .5)
        self.assertAlmostEqual(np.ptp(conditional_means), 1.7)


class ObservedInstrumentEncodingTests(unittest.TestCase):
    def test_continuous_cutpoints_use_training_records_only(self):
        frame = pd.DataFrame({'instrument': [1., 2., 3., 4., 100., 200.]})
        train = np.arange(4)
        z_a, meta_a = encode_instrument(frame, train, {'instrument_type': 'continuous'})
        changed = frame.copy()
        changed.loc[4:, 'instrument'] = [-1e6, 1e6]
        z_b, meta_b = encode_instrument(changed, train, {'instrument_type': 'continuous'})
        self.assertEqual(meta_a, meta_b)
        np.testing.assert_array_equal(z_a[train], z_b[train])
        self.assertEqual(z_b[4], 0)
        self.assertEqual(z_b[5], 2)


if __name__ == '__main__':
    unittest.main()
