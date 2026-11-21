
from __future__ import annotations

import json

import numpy as np
import pandas as pd


def compare_outputs(primary, fine, natural=False):
    filename = 'intervals.csv' if natural else 'trials.csv'
    keys = ['row_id', 'target', 'method'] if natural else ['trial', 'target', 'method']
    left, right = pd.read_csv(primary / filename), pd.read_csv(fine / filename)
    if not natural:
        left = left[left.trial < right.trial.nunique()]
        dleft, dright = pd.read_csv(primary / 'diagnostics.csv'), pd.read_csv(fine / 'diagnostics.csv')
        dleft = dleft[dleft.trial < dright.trial.nunique()]
        pd.testing.assert_frame_equal(dleft[['trial', 'method', 'randomization_hash']].reset_index(drop=True),
                                      dright[['trial', 'method', 'randomization_hash']].reset_index(drop=True))
        with np.load(primary / 'split.npz') as ps, np.load(fine / 'split.npz') as fs:
            for name in ps.files:
                np.testing.assert_array_equal(ps[name], fs[name])
    else:
        for name in ['split.csv', 'response_predictions.csv']:
            pd.testing.assert_frame_equal(pd.read_csv(primary / name), pd.read_csv(fine / name), check_exact=True)
    left, right = (frame.sort_values(keys).reset_index(drop=True) for frame in [left, right])
    pd.testing.assert_frame_equal(left[keys], right[keys], check_exact=True)
    width = 'length' if natural else 'mean_length'
    finite = np.isfinite(left[width]) & np.isfinite(right[width])
    flips = int((np.isfinite(left[width]) != np.isfinite(right[width])).sum())
    delta = float(np.abs(left.loc[finite, width] - right.loc[finite, width]).max()) if finite.any() else 0.
    coverage = None if natural else float(np.abs(left.coverage - right.coverage).max())
    original_design = json.loads((primary / 'design.json').read_text(encoding='utf-8'))
    fine_design = json.loads((fine / 'design.json').read_text(encoding='utf-8'))
    if natural:
        mass_difference = max(abs(original_design['calibration_diagnostics'][method]['identified_mass_eif'] -
                                  fine_design['calibration_diagnostics'][method]['identified_mass_eif'])
                              for method in ['Policy IV', 'Softmax IV'])
        scale = original_design['outcome_scale']
    else:
        mass_difference = float(np.abs(dleft.identified_mass_eif.to_numpy() - dright.identified_mass_eif.to_numpy()).max())
        scale = original_design['reference_residual_sd']
    baseline = ~left.method.isin(['Policy IV', 'Softmax IV', 'Policy IV (known e)', 'Softmax IV (known e)'])
    pd.testing.assert_frame_equal(left.loc[baseline].reset_index(drop=True),
                                  right.loc[baseline].reset_index(drop=True), check_exact=True)
    return dict(paired_rows=len(left), finite_infinite_changes=flips,
                maximum_finite_width_change=delta, maximum_coverage_change=coverage,
                maximum_identified_mass_change=mass_difference,
                maximum_halfwidth_change_in_reference_sd=delta / (2 * scale),
                material_difference_flag=bool(flips or (coverage is not None and coverage > .01) or delta / (2 * scale) > .01),
                source_splits_and_randomness_or_response_predictions_identical=True)
