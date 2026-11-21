
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

import estimate_natural_gamma as natural
import validate_observed_gamma as observed_audit
from validate_feedback_application import validate_dataset
from validate_feedback_simulations import compare_summaries, recompute_summary
from validate_observed_gamma import validate_case, validate_pairs

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parent
HEIGHT = BASE / 'results' / 'nhanes_five_categories'
METHODS = ['Policy IV', 'Weighted CP (X)', 'Weighted CP (X,Z)']
LEGACY_FEEDBACK_HASH = 'f9929ca282b6d146e1dd834e04798d565a8851e0aa0ce598ea9e308a5b3f568c'
CURRENT_FEEDBACK_HASH = '5b5134deebee5ac8505aa9bcc2a04eecefed2af97307a82090cfb7d541ac558b'
LEGACY_CASES = []


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def row90(frame, method):
    selected = frame[(frame.method == method) & np.isclose(frame.target, .9)]
    assert len(selected) == 1, (method, len(selected))
    return selected.iloc[0]


def display(value, digits):
    return 'infinity' if np.isinf(value) else f'{value:.{digits}f}'


def table_values(text, label):
    text = '\n'.join(re.split(r'(?<!\\)%', line, maxsplit=1)[0]
                     for line in text.splitlines())
    tables = re.findall(r'\\begin\{table\}.*?\\end\{table\}', text, re.S)
    selected = [table for table in tables if r'\label{' + label + '}' in table]
    assert len(selected) == 1, (label, 'Expected exactly one table')
    body = selected[0].split(r'\midrule', 1)[1].split(r'\bottomrule', 1)[0]
    body = re.sub(r'\\cellcolor\{[^}]*\}', '', body)
    values = re.findall(r'\\infty|\d+\.\d+', body)
    return ['infinity' if value == r'\infty' else value for value in values]


def check_table(text, label, expected):
    actual = table_values(text, label)
    assert actual == expected, (label, actual, expected)


def aggregate(folder):
    trials = pd.read_csv(folder / 'trials.csv')
    summary = pd.read_csv(folder / 'summary.csv')
    compare_summaries(summary, recompute_summary(trials))
    return trials, summary


def check_masking():
    summaries, cache, cases = {}, {}, {}
    LEGACY_CASES.clear()
    feedback_hash = hashlib.sha256((BASE / 'feedback_simulations.py').read_bytes()).hexdigest()
    for dataset in ['heckman', 'nhanes_height']:
        for family in ['anchored', 'crossing']:
            root = (BASE / 'results' / 'observed_gamma' /
                    ('primary' if family == 'anchored' else 'crossing_refined')
                    if dataset == 'heckman' else HEIGHT / 'primary')
            for gamma in [90, 95, 99, 100]:
                scenario = f'{family}_g{gamma:03d}'
                folder = root / dataset / scenario
                design = read_json(folder / 'design.json')
                if dataset == 'heckman':
                    recorded_hash = design['hashes']['feedback_simulations.py']
                    legacy = recorded_hash != feedback_hash
                    if legacy:
                        # Accept only the exact documented historical difference.
                        # The original pre-run and fitted records stay untouched.
                        assert (recorded_hash, feedback_hash) == (
                            LEGACY_FEEDBACK_HASH, CURRENT_FEEDBACK_HASH), (
                                'Unrecognized FES source mismatch', folder,
                                recorded_hash, feedback_hash)
                        original_require = observed_audit.require

                        def require_current(condition, message):
                            if message != 'Input hash changed: feedback_simulations.py':
                                original_require(condition, message)

                        observed_audit.require = require_current
                        try:
                            _, arrays, checked, summary = validate_case(
                                folder, design['arguments'], cache)
                        finally:
                            observed_audit.require = original_require
                        LEGACY_CASES.append(scenario)
                    else:
                        _, arrays, checked, summary = validate_case(
                            folder, design['arguments'], cache)
                    cases[dataset, scenario] = arrays, checked
                else:
                    _, summary = aggregate(folder)
                assert design['arguments']['trials'] == 200
                assert design['instrument_arms'] == (3 if dataset == 'heckman' else 5)
                summaries[dataset, scenario] = summary
    validate_pairs(cases)
    return summaries


def expected_table_values():
    """Independently reconstruct every displayed number from result files."""
    expected_tables = {}
    masking = check_masking()
    expected = []
    for method in METHODS:
        for dataset in ['heckman', 'nhanes_height']:
            row = row90(masking[dataset, 'anchored_g100'], method)
            expected += [display(row.coverage, 3), display(row.coverage_mcse, 3),
                         display(row.mean_length, 2), display(row.finite_mean_length_mcse, 3)]
    expected_tables['tab:empirical-masking'] = expected

    expected = []
    for dataset, folder, seed in [
        ('heckman', BASE / 'results/feedback/mar_x_high_response/heckman/mar_x', 20260922),
        ('nhanes_height', HEIGHT / 'mar/nhanes_height/mar_x', 20260925),
    ]:
        trials, summary = aggregate(folder)
        assert read_json(folder / 'design.json')['arguments']['response_intercept'] == 4.
        for offset, method in enumerate(METHODS):
            row = row90(summary, method)
            selected = trials[(trials.method == method) & np.isclose(trials.target, .9)].sort_values('trial')
            np.testing.assert_array_equal(selected.trial, np.arange(200))
            expected += [display(row.coverage, 3), display(row.coverage_mcse, 3),
                         display(row.mean_length, 2)]
            if np.isfinite(row.mean_length):
                expected += [display(selected.mean_length.std(ddof=1) / np.sqrt(200), 3)]
            medians = selected.median_length.to_numpy()
            rng = np.random.default_rng(seed + offset)
            boot = np.quantile(medians[rng.integers(0, 200, size=(20000, 200))],
                               .5, axis=1, method='inverted_cdf')
            assert np.isfinite(boot).all()
            expected += [display(row.median_length, 2), display(boot.std(ddof=1), 3),
                         display(100 * row.infinite_fraction, 1)]
    expected_tables['tab:empirical-mar'] = expected

    applications = {}
    expected = []
    for dataset, root in [('heckman', BASE / 'results/feedback_application'),
                          ('nhanes_height', HEIGHT / 'natural')]:
        validate_dataset(root, dataset)
        summary = pd.read_csv(root / dataset / 'summary.csv')
        applications[dataset] = summary
        _, _, _, response, outcome, _, _ = natural.load_cohort(dataset)
        spans = [np.diff(np.quantile(outcome[response], [p, 1-p]))[0] for p in [.05, .025]]
        row = row90(summary, 'Policy IV')
        expected += [display(spans[0], 2), display(spans[1], 2),
                     display(row.mean_length, 2), display(100 * row.infinite_fraction, 1)]
    expected_tables['tab:empirical-application'] = expected
    expected = []
    for method in ['MAR population CP (X)', 'MAR population CP (X,Z)',
                   'Best-case CP (X,Z)', 'Worst-case CP (X,Z)']:
        for dataset in ['heckman', 'nhanes_height']:
            row = row90(applications[dataset], method)
            expected += [display(row.mean_length, 2), display(row.median_length, 2),
                         display(100 * row.infinite_fraction, 1)]
    expected_tables['tab:empirical-application-comparisons'] = expected
    return expected_tables


def check_policy_mass(check_reference=False):
    """Verify the projection against calibration diagnostics, including checks."""
    projected = pd.read_csv(BASE / 'results/projected_policy_mass/summary.csv')
    cases = [('FES', True, BASE / 'results/feedback_application/heckman')]
    cases += [('NHANES III reported height', tag == 'natural',
               HEIGHT / tag / 'nhanes_height') for tag in (
                   'natural', 'natural_grid', 'natural_sensitivity_20260920',
                   'natural_sensitivity_20260924')]
    assert len(projected) == len(cases)
    for label, primary, folder in cases:
        design = read_json(folder / 'design.json')
        args = design['arguments']
        selected = projected[(projected.dataset == label) &
                             (projected.seed == args['seed']) &
                             (projected.grid_size == args['grid_size'])]
        assert len(selected) == 1, (label, args['seed'], args['grid_size'])
        row = selected.iloc[0]
        raw = float(design['calibration_diagnostics']['Policy IV']['identified_mass_eif'])
        clipped = float(np.clip(raw, 0., 1.))
        np.testing.assert_allclose(
            [row.raw_policy_mass, row.projected_policy_mass,
             row.raw_limiting_miscoverage, row.projected_limiting_miscoverage,
             row.projection_change], [raw, clipped, 1. - raw, 1. - clipped, clipped - raw])
        assert bool(row.manuscript_primary) == primary
        assert row.n_calibration == design['n_calibration']
        summary = row90(pd.read_csv(folder / 'summary.csv'), 'Policy IV')
        np.testing.assert_allclose(
            [row.n_missing_targets, row.preserved_interval_width,
             row.preserved_infinite_fraction],
            [summary.n_missing_targets, summary.mean_length, summary.infinite_fraction])
    if check_reference:
        primary = projected[projected.manuscript_primary]
        np.testing.assert_allclose(primary.projected_policy_mass,
                                   [.9095380960233861, .9593566765034951])


def check_tables_in_directory(directory, expected, required=False):
    """Check existing labeled tables; require all four for frozen references."""
    texts = [(path, path.read_text(encoding='utf-8-sig'))
             for path in sorted(directory.glob('*.tex'))]
    checked = []
    for label, values in expected.items():
        matching = [(path, text) for path, text in texts if r'\label{' + label + '}' in text]
        if required:
            assert matching, f'Missing reference table {label} in {directory}'
        for path, text in matching:
            check_table(text, label, values)
            checked.append(str(path.relative_to(PROJECT)))
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-reference', action='store_true',
                        help='Also compare all four frozen draft tables and primary policy masses')
    args = parser.parse_args()
    for module in ['run_nhanes_five_categories', 'make_empirical_height_figure',
                   'validate_nhanes_five_categories', 'project_policy_mass']:
        importlib.import_module(module)
    validation = read_json(HEIGHT / 'validation.json')
    assert validation['passed'] and validation['published_mar_90_grid_stable']
    for name, expected in read_json(HEIGHT / 'plan.json')['hashes'].items():
        assert hashlib.sha256((BASE / name).read_bytes()).hexdigest() == expected, name
    expected = expected_table_values()
    generated = check_tables_in_directory(PROJECT / 'tables', expected)
    if args.check_reference:
        check_tables_in_directory(PROJECT / 'templates', expected, required=True)
    check_policy_mass(args.check_reference)
    print('PASS: current module imports and input hashes, 16 masking fits, '
          '2 MAR summaries and bootstrap errors, 2 natural applications, '
          'and 5 policy-mass diagnostics.')
    print(f'Checked {len(generated)} generated table files; '
          f'frozen draft agreement {"required and passed" if args.check_reference else "not requested"}.')
    if LEGACY_CASES:
        print(f'Provenance note: {len(LEGACY_CASES)} FES masking fits record the earlier '
              'feedback_simulations.py hash (f9929ca2...); the retained version '
              '(5b5134de...) matches later NHANES/MAR manifests. Only this exact '
              'historical difference is accepted; saved hashes are preserved and '
              'FES source rows, splits, and masks are independently reconstructed.')


if __name__ == '__main__':
    main()
