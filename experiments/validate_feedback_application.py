
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from estimate_natural_gamma import load_cohort
from run_feedback_application import BASE, DEFAULT_OUTPUT, METHODS, json_write, width_summary


def validate_dataset(root, dataset):
    out = root / dataset
    info = json.loads((out / 'design.json').read_text())
    # Draft bundle: saved Windows-relative paths also work on Linux/macOS.
    source = BASE / info['source'].replace('\\', '/')
    assert hashlib.sha256(source.read_bytes()).hexdigest() == info['source_sha256']
    df, _, _, response, outcome, _, _ = load_cohort(dataset)
    split = pd.read_csv(out / 'split.csv', dtype={'split_group': str})
    intervals = pd.read_csv(out / 'intervals.csv')
    summary = pd.read_csv(out / 'summary.csv')
    np.testing.assert_array_equal(split.row_id, df.row_id)
    np.testing.assert_array_equal(split.observed.to_numpy(bool), response)
    target_rows = split.loc[split.membership == 'missing_target', 'row_id']
    assert len(target_rows) == info['n_missing_targets']
    target_source = df.row_id.isin(target_rows)
    assert np.isnan(outcome[target_source]).all()
    assert not response[target_source].any()
    assert set(intervals.row_id) == set(target_rows)
    assert (intervals.observed == 0).all()
    assert 'outcome' not in intervals and 'coverage' not in summary
    group_sets = [set(split.loc[split.membership.isin(names), 'split_group'])
                  for names in [['train'], ['calibration'], ['missing_target', 'target_pool']]]
    assert all(not group_sets[i] & group_sets[j] for i in range(3) for j in range(i+1, 3))
    assert not intervals.duplicated(['row_id', 'target', 'method']).any()
    assert not intervals[['center', 'lower', 'upper', 'length']].isna().any().any()
    assert (intervals.lower <= intervals.upper).all()
    np.testing.assert_allclose(intervals.upper - intervals.lower, intervals.length)
    np.testing.assert_array_equal(np.isfinite(intervals.length), intervals.finite)
    assert set(intervals.method) == set(METHODS)
    for (target, method), frame in intervals.groupby(['target', 'method']):
        assert set(frame.row_id) == set(target_rows)
        row = summary[(np.isclose(summary.target, target)) & (summary.method == method)].iloc[0]
        expected = width_summary(frame.length.to_numpy())
        for name, value in expected.items():
            np.testing.assert_allclose(row[name], value, equal_nan=True)
        worst = intervals[(np.isclose(intervals.target, target)) &
                          (intervals.method == 'Worst-case CP (X,Z)')].set_index('row_id').loc[frame.row_id]
        containment = ((frame.lower.to_numpy() >= worst.lower.to_numpy()) &
                       (frame.upper.to_numpy() <= worst.upper.to_numpy()))
        np.testing.assert_array_equal(frame.contained_in_worst_case, containment)
        np.testing.assert_allclose(row.contained_in_worst_case_fraction, containment.mean())
    for target in intervals.target.unique():
        best = intervals[(np.isclose(intervals.target, target)) &
                         (intervals.method == 'Best-case CP (X,Z)')].set_index('row_id')
        worst = intervals[(np.isclose(intervals.target, target)) &
                          (intervals.method == 'Worst-case CP (X,Z)')].set_index('row_id').loc[best.index]
        np.testing.assert_allclose(best.center, worst.center)
        assert (best.lower >= worst.lower).all() and (best.upper <= worst.upper).all()
    return dict(dataset=dataset, missing_targets=len(target_rows), interval_rows=len(intervals),
                input_hash_matches=True, targets_genuinely_missing=True,
                splitting_groups_disjoint=True, summaries_recomputed=True,
                no_imputed_outcomes_or_coverage=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    manifest = json.loads((args.output / 'run_manifest.json').read_text())
    audits = [validate_dataset(args.output, dataset) for dataset in manifest['datasets']]
    combined = pd.read_csv(args.output / 'summary.csv')
    pieces = pd.concat([pd.read_csv(args.output / dataset / 'summary.csv')
                        for dataset in manifest['datasets']], ignore_index=True)
    key = ['dataset', 'target', 'method']
    pd.testing.assert_frame_equal(combined.sort_values(key).reset_index(drop=True),
                                  pieces.sort_values(key).reset_index(drop=True))
    result = dict(passed=True, datasets=audits, combined_summary_matches=True)
    json_write(args.output / 'validation.json', result)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
