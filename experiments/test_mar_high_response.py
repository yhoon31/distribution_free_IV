
import unittest

import numpy as np
from scipy.special import expit

from feedback_simulations import response_table


class HighResponseMARTests(unittest.TestCase):
    def test_x_only_mar_is_exactly_invariant_to_y_and_z(self):
        rng = np.random.default_rng(173)
        u = np.r_[-3., 3., rng.uniform(-3., 3., 998)]
        v = rng.normal(size=len(u))
        for intercept in (.3, 4.):
            for k in (3, 5, 6):
                p = response_table('mar_x', v, u, k, intercept)
                changed_y = response_table('mar_x', 1e6*v+123., u, k, intercept)
                np.testing.assert_array_equal(p, changed_y)
                for arm in range(k):
                    np.testing.assert_array_equal(p[:, arm], p[:, 0])
                np.testing.assert_array_equal(np.ptp(p, axis=1), np.zeros(len(u)))
                self.assertTrue(np.all((p > 0.) & (p < 1.)))
                self.assertGreater(np.ptp(p[:, 0]), 0.)
                order = np.argsort(u)
                self.assertTrue(np.all(np.diff(p[order, 0]) >= 0.))
                np.testing.assert_allclose(p[:, 0].min(), expit(intercept-3.6), rtol=1e-15)
                np.testing.assert_allclose(p[:, 0].max(), expit(intercept+3.6), rtol=1e-15)

    def test_x_only_gamma_equals_response_and_every_policy_mass(self):
        rng = np.random.default_rng(714)
        u = rng.uniform(-3., 3., 1000)
        p = response_table('mar_x', rng.normal(size=len(u)), u, response_intercept=4.)
        gamma = p.max(axis=1).mean()
        # Equal response across arms makes the envelope, uniform mixture, and
        # arbitrary conditional mixtures identical (up to arithmetic roundoff).
        random_policy = rng.dirichlet(np.ones(p.shape[1]), size=len(u))
        np.testing.assert_allclose(p.mean(), gamma, rtol=1e-15)
        np.testing.assert_allclose(np.sum(p*random_policy, axis=1), p[:, 0], rtol=1e-15)

    def test_default_matches_historical_formula_exactly(self):
        u = np.array([-3., -1.1, 0., .8, 3.])
        v = np.array([-.7, 1., 5., -.2, 2.])
        for k in (3, 5, 6):
            for scenario in ('mar_xz', 'observed_iv_mar'):
                expected = expit(.3+1.2*u[:, None]+np.linspace(-1., 1., k)[None, :])
                np.testing.assert_array_equal(response_table(scenario, v, u, k), expected)
            expected_x = np.repeat(expit(.3+1.2*u)[:, None], k, axis=1)
            np.testing.assert_array_equal(response_table('mar_x', v, u, k), expected_x)

    def test_high_response_is_mar_and_pointwise_more_observed(self):
        rng = np.random.default_rng(173)
        u, v = rng.uniform(-3., 3., 1000), rng.normal(size=1000)
        uniforms = rng.random(1000)
        for scenario in ('mar_x', 'mar_xz', 'observed_iv_mar'):
            high = response_table(scenario, v, u, response_intercept=4.)
            changed_y = response_table(scenario, 1e6*v, u, response_intercept=4.)
            low = response_table(scenario, v, u)
            np.testing.assert_array_equal(high, changed_y)
            self.assertTrue(np.all(high > low))
            self.assertTrue(np.all(high < 1.))
            self.assertTrue(np.all((uniforms < low[:, 0]) <= (uniforms < high[:, 0])))
            self.assertGreater(high.max(axis=1).mean(), low.max(axis=1).mean())

    def test_intercept_changes_do_not_modify_mnar_controls(self):
        for scenario in ('heterogeneous_mnar', 'weak_mnar', 'irrelevant_mnar',
                         'observed_iv_mnar', 'invalid_half', 'invalid_one'):
            default = response_table(scenario, [-1., .2, 2.], [-2., 0., 2.])
            changed = response_table(scenario, [-1., .2, 2.], [-2., 0., 2.],
                                     response_intercept=4.)
            np.testing.assert_array_equal(default, changed)


if __name__ == '__main__':
    unittest.main()
