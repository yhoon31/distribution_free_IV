
import unittest

import numpy as np
from scipy.stats import norm

from feedback_simulations import response_table as historical_response_table


def observed_envelope(outcome_given_z, instrument_probability, response):
    """Reconstruct pi_z f_obs,z from finite joint masses, then integrate max."""
    joint = outcome_given_z * instrument_probability[None, :]
    observed = joint * response
    response_given_z = observed.sum(axis=0) / instrument_probability
    outcome_given_observed_z = observed / observed.sum(axis=0, keepdims=True)
    subdensity = response_given_z[None, :] * outcome_given_observed_z
    return float(subdensity.max(axis=1).sum())


class ObservedGammaMathematics(unittest.TestCase):
    def setUp(self):
        self.residual = np.array([-3.1, -1.2, -.35, .2, .9, 2.4])
        self.covariate_index = np.array([-2., -.8, .3, 1.1, -.5, 2.])
        self.mass = np.array([.07, .11, .23, .31, .19, .09])

    def probabilities(self, family, gamma, residual=None, u=None, k=5, anchor=0):
        # The runner's public interface is imported lazily so the mathematical
        # reconstruction above is also usable from the saved-output auditor.
        from run_observed_gamma import response_probabilities
        if residual is None:
            residual = self.residual
        if u is None:
            u = np.zeros(len(residual))
        return response_probabilities(family, residual, u, k, gamma, anchor)

    def test_exact_index_with_nonuniform_instrument_probabilities(self):
        # Positive but highly unequal frequencies do not enter the envelope.
        for k in (2, 3, 5):
            conditional_y = np.repeat(self.mass[:, None], k, axis=1)
            frequencies = [np.arange(1, k+1, dtype=float),
                           np.geomspace(.001, 1., k)]
            for weights in frequencies:
                e = weights / weights.sum()
                for family in ('anchored', 'crossing'):
                    for gamma in (.90, .95, .99, 1.):
                        with self.subTest(k=k, family=family, gamma=gamma, e=e):
                            p = self.probabilities(family, gamma, k=k, anchor=k-1)
                            index = observed_envelope(conditional_y, e, p)
                            self.assertAlmostEqual(index, gamma, places=13)

    def test_anchor_controls_the_maximum_with_distinct_selective_other_arms(self):
        for k in (2, 3, 5):
            for anchor in range(k):
                p = self.probabilities('anchored', .99, k=k, anchor=anchor)
                np.testing.assert_allclose(p[:, anchor], .99, rtol=0, atol=0)
                np.testing.assert_allclose(p.max(axis=1), .99, rtol=0, atol=0)
                other = np.arange(k) != anchor
                self.assertTrue((p[:, other] < .99).all())
                self.assertTrue((p[:, other] > 0).all())
                self.assertTrue((np.ptp(p[:, other], axis=0) > .01).all())
                row = self.probabilities('anchored', .99, residual=[.173],
                                         k=k, anchor=anchor)[0]
                self.assertEqual(len(np.unique(row)), k)

    def test_crossing_allows_every_arm_to_win(self):
        for k in (2, 3, 5):
            centers = norm.ppf((np.arange(k)+.5)/k)
            p = self.probabilities('crossing', .95, residual=centers, k=k)
            np.testing.assert_array_equal(p.argmax(axis=1), np.arange(k))
            np.testing.assert_allclose(p.max(axis=1), .95, atol=1e-15)
            baseline = historical_response_table('heterogeneous_mnar',
                centers, np.zeros(k), k)
            np.testing.assert_allclose(p, .95*baseline, atol=1e-15)

    def test_increasing_gamma_preserves_normalized_selection_law(self):
        for family in ('anchored', 'crossing'):
            base = self.probabilities(family, 1.)
            for gamma in (.90, .95, .99):
                p = self.probabilities(family, gamma)
                np.testing.assert_allclose(p/gamma, base, atol=2e-15)
                self.assertTrue((p <= base).all())

    def test_invalid_instrument_anchor_is_lower_bound_not_exact_index(self):
        # Anchor arm 0 has Y on the first atom; another arm has Y on the
        # second. Exclusion fails and the second arm adds disjoint mass.
        conditional_y = np.array([[1., 0., .5], [0., 1., .5]])
        e = np.array([.2, .3, .5])
        for gamma in (.9, 1.):
            p = self.probabilities('anchored', gamma, residual=[-3., 0.], k=3)
            index = observed_envelope(conditional_y, e, p)
            self.assertGreater(index, gamma)
            if gamma == 1.:
                self.assertGreater(index, 1.)

    def test_crossing_envelope_has_no_gamma_lower_bound_without_exclusion(self):
        # Each arm's Y occurs only where the other arm has the largest
        # response probability. max_z p_z=g alone is insufficient without IV.
        residual = norm.ppf([.25, .75])
        conditional_y = np.array([[0., 1.], [1., 0.]])
        e = np.array([.3, .7])
        p = self.probabilities('crossing', .99, residual=residual, k=2)
        np.testing.assert_allclose(p.max(axis=1), .99)
        self.assertLess(observed_envelope(conditional_y, e, p), .99)

    def test_probabilities_are_valid_in_outcome_tails(self):
        residual = np.array([-1e6, -8., 0., 8., 1e6])
        for family in ('anchored', 'crossing'):
            for gamma in (.90, .95, .99, 1.):
                p = self.probabilities(family, gamma, residual=residual)
                self.assertTrue(np.isfinite(p).all())
                self.assertTrue(((0 <= p) & (p <= 1)).all())

    def test_original_logistic_control_preserves_the_old_response_law(self):
        for k in (3, 5):
            actual = self.probabilities('original_logistic', None,
                u=self.covariate_index, k=k)
            expected = historical_response_table('observed_iv_mnar',
                self.residual, self.covariate_index, k)
            np.testing.assert_array_equal(actual, expected)


if __name__ == '__main__':
    unittest.main()
