
from __future__ import annotations

import hashlib
import json
import re
from pathlib import PurePosixPath, PureWindowsPath

import numpy as np
import pandas as pd
from scipy.special import expit

from check_observed_gamma_grid import compare_fit, json_safe
from run_experiments import BASE
from run_feedback_application import METHODS, width_summary
from validate_feedback_simulations import reconstruct_source, require
from validation_helpers import compare_outputs
import validate_feedback_simulations as simulation_audit
import validate_observed_gamma as historical

ROOT = BASE / 'results' / 'nhanes_five_categories'
DATASET = 'nhanes_height'
NATIVE = BASE / 'data' / 'nhanes'
CODES = [7005, 7016, 7020, 7024, 7067]
LEVELS = [str(code) for code in CODES]


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path, value):
    path.write_text(json.dumps(json_safe(value), indent=2, allow_nan=False) + '\n', encoding='utf-8')


def audit_source():
    """Extract literal CDC columns without using the preparation implementation."""
    layout = re.search(r'\bINPUT\s+(.*?);', (NATIVE / 'adult.sas').read_text(), re.S).group(1)
    positions = {name: (int(first) - 1, int(last or first)) for name, first, last in
                 re.findall(r'^\s*(\w+)\s+(\d+)(?:-(\d+))?\s*$', layout, re.M)}
    fields = ['SEQN', 'DMPFSEQ', 'HAINTVR', 'HAM5S']
    rows = []
    with (NATIVE / 'adult.dat').open(encoding='ascii') as handle:
        for line in handle:
            values = [line[slice(*positions[field])].strip() for field in fields]
            rows.append([float(value) if value and value != '.' else np.nan for value in values])
    raw = pd.DataFrame(rows, columns=fields)
    mapping = read_json(BASE / 'data' / DATASET / 'source_mapping.json')
    require(mapping['source_sha256'] == historical.digest(NATIVE / 'adult.dat') and
            mapping['layout_sha256'] == historical.digest(NATIVE / 'adult.sas'),
            'Original CDC source/layout checksum differs from preparation')
    require(len(raw) == 20050 and raw.SEQN.is_unique, 'CDC adult source dimensions differ')
    valid = raw.HAM5S.notna() & ~raw.HAM5S.isin([888, 999])
    require(valid.sum() == 19129 and raw.loc[valid, 'HAM5S'].min() == 41 and
            raw.loc[valid, 'HAM5S'].max() == 92, 'CDC reported-height frequencies differ')
    families = np.sort(raw.DMPFSEQ.unique())
    design_families = np.random.default_rng(20262515).permutation(families)[:int(np.ceil(.2 * len(families)))]
    design_mask = raw.DMPFSEQ.isin(design_families)
    design = raw.loc[design_mask, ['SEQN', 'DMPFSEQ', 'HAINTVR']].copy()
    design.columns = ['row_id', 'family', 'adult_interviewer']
    design.adult_interviewer = design.adult_interviewer.mask(design.adult_interviewer == 8888)
    saved_design = pd.read_csv(NATIVE / 'native_design_holdout.csv')
    pd.testing.assert_frame_equal(saved_design.reset_index(drop=True), design.reset_index(drop=True),
                                  check_dtype=False, check_exact=True)
    counts = design.adult_interviewer.value_counts()
    selected = sorted(int(code) for code in sorted(counts.index, key=lambda code: (-counts[code], code))[:5])
    require(selected == CODES, 'Outcome-blind native five-interviewer selection changed')
    wanted = raw.loc[~design_mask & raw.HAINTVR.isin(CODES)].set_index('SEQN')
    original = pd.read_csv(NATIVE / 'prepared_natural.csv')
    frame = pd.read_csv(BASE / 'data' / DATASET / 'prepared_natural.csv')
    np.testing.assert_array_equal(frame.SEQN, wanted.index)
    np.testing.assert_array_equal(frame.instrument, wanted.HAINTVR)
    np.testing.assert_array_equal(frame.household_id, wanted.DMPFSEQ)
    np.testing.assert_array_equal(frame.HAM5S, wanted.HAM5S)
    response = wanted.HAM5S.notna() & ~wanted.HAM5S.isin([888, 999])
    np.testing.assert_array_equal(frame.observed, response.astype(int))
    np.testing.assert_array_equal(frame.outcome, wanted.HAM5S.where(response))
    unchanged = [key for key in original if key not in ['outcome', 'observed', 'measurement_status']]
    pd.testing.assert_frame_equal(frame[unchanged], original[unchanged], check_exact=True)
    for stem in ['prepared', 'prepared_natural']:
        metadata = read_json(BASE / 'data' / DATASET / (stem + '_metadata.json'))
        require(metadata['outcomes_imputed'] is False and metadata['outcome_transformation'] == 'none',
                'Height was imputed or transformed')
        require(metadata['outcome'] == 'HAM5S' and metadata['observed_rows'] == int(response.sum()),
                'Height source metadata differs')
        require(not set(metadata['numeric_features'] + metadata['categorical_features']) &
                {'HAM5S', 'HAM6S', 'outcome'}, 'An outcome was included in X')
    require((BASE / 'data' / DATASET / 'prepared.csv').read_bytes() ==
            (BASE / 'data' / DATASET / 'prepared_natural.csv').read_bytes(), 'MAR and native sources differ')
    order = np.random.default_rng(20260918 + 32771).permutation(len(frame))
    _, first = np.unique(frame.household_id.to_numpy()[order], return_index=True)
    cohort = frame.iloc[np.sort(order[first])].reset_index(drop=True)
    require(cohort.household_id.is_unique, 'Natural sample contains duplicate households')
    qs = np.quantile(cohort.loc[cohort.observed.astype(bool), 'outcome'], [.025, .05, .5, .95, .975])
    return cohort, dict(passed=True, source_rows=len(raw), source_observed=int(valid.sum()),
        source_sha256=historical.digest(NATIVE / 'adult.dat'),
        prepared_rows=len(frame), prepared_observed=int(response.sum()),
        native_codes=CODES, design_holdout_and_selection_independently_reproduced=True,
        original_outcomes_raw_interviewers_and_covariates_verified=True,
        one_family_rows=len(cohort), one_family_observed=int(cohort.observed.sum()),
        one_family_missing=int((1 - cohort.observed).sum()),
        respondent_quantiles=dict(zip(['p025', 'p05', 'median', 'p95', 'p975'], qs.tolist())),
        respondent_span90=float(qs[3] - qs[1]), respondent_span95=float(qs[4] - qs[0]))


def expected_protocol():
    cases = {}
    for family in ['anchored', 'crossing']:
        for gamma in [90, 95, 99, 100]:
            cases[('masking', 'primary', f'{family}_g{gamma:03d}')] = (20260919, 200, 1601 if family == 'crossing' else 801)
        cases[('masking', 'grid', f'{family}_g100')] = (20260919, 10, 3201 if family == 'crossing' else 1601)
    for seed in [20260920, 20260924]:
        cases[('masking', f'sensitivity_{seed}', 'crossing_g100')] = (seed, 100, 1601)
        cases[('natural', f'natural_sensitivity_{seed}', 'natural')] = (seed, None, 801)
    cases[('mar', 'mar', 'mar_x')] = (20260919, 200, 801)
    cases[('mar', 'mar_grid', 'mar_x')] = (20260919, 10, 1601)
    cases[('natural', 'natural', 'natural')] = (20260919, None, 801)
    cases[('natural', 'natural_grid', 'natural')] = (20260919, None, 1601)
    return cases


def audit_plan():
    plan, run = read_json(ROOT / 'plan.json'), read_json(ROOT / 'run.json')
    require(plan['completed'] is False and run['completed'] is True, 'Frozen plan or completed run missing')
    for key in plan:
        if key not in {'completed', 'case_status'}:
            require(plan[key] == run[key], f'Frozen plan differs: {key}')
    expected = expected_protocol()
    require(len(run['jobs']) == len(expected) == 18, 'Incomplete five-category experiment matrix')
    actual = {(job['kind'], job['tag'], job['scenario']) for job in run['jobs']}
    require(actual == set(expected), 'Fit matrix differs from fixed protocol')
    require(len(run['case_status']) == len(expected) and
            all(status['completed'] is True for status in run['case_status']), 'Incomplete case statuses')
    statuses = {(status['kind'], status['tag'], status['scenario']) for status in run['case_status']}
    require(statuses == actual, 'Completed case statuses differ from jobs')
    for job in run['jobs']:
        require(job['dataset'] == DATASET, 'Unexpected dataset')
        seed, trials, grid = expected[job['kind'], job['tag'], job['scenario']]
        args = job['arguments']
        require(args['seed'] == seed and args['grid_size'] == grid and
                args['trees'] == 120 and args['min_leaf'] == 20, 'Fitting protocol changed')
        if job['kind'] == 'natural':
            require(args['targets'] == [.9] and args['max_train'] == 2000 and
                    args['max_calibration'] == 700 and args['split_unit'] == 'household', 'Natural protocol changed')
        else:
            require(args['trials'] == trials and args['targets'] == [.8, .9, .95] and
                    args['n_train'] == args['pool_size'] == 2000, 'Masking protocol changed')
    return run


def provenance(directory, job, plan):
    design, before = read_json(directory / 'design.json'), read_json(directory / 'inputs_before_run.json')
    require(design['hashes'] == before == plan['hashes'], 'Plan/pre-run/fitted hashes differ')
    for name, digest in before.items():
        require(historical.digest(BASE / name) == digest, f'Changed input: {name}')
    require(design['arguments'] == job['arguments'], 'Saved fit arguments differ from plan')
    require(design['original_outcomes_retained'] is True, 'Original outcomes flag missing')
    return design


def audit_masking(directory, job, plan, cache):
    design = provenance(directory, job, plan)
    require(design['original_instrument_retained'] is True and
            design['instrument_coarsened'] is False and design['instrument_arms'] == 5,
            'Native five-category instrument was not retained')
    require(design['instrument'] == {'levels': LEVELS}, 'Native five-category encoding differs')
    result, arrays, checked, _ = historical.validate_case(directory, job['arguments'], cache)
    require(checked == design, 'Design changed during audit')
    for part in ['pilot', 'train', 'pool']:
        np.testing.assert_array_equal(np.asarray(CODES)[arrays[part + '_z']], arrays[part + '_original_z'])
    regions = np.bincount(arrays['pool_probability_table'].argmax(axis=1), minlength=5)
    if design['response_design'] == 'crossing':
        require((regions > 0).all(), 'A crossing winning region is absent')
    result.update(passed=True, original_five_category_encoding_verified=True,
                  winning_region_counts=regions.tolist())
    return result, arrays, design


def audit_mar(directory, job, plan, cache):
    design = provenance(directory, job, plan)
    args = job['arguments']
    require(design['instrument_arms'] == design['randomized_instrument_arms'] == 5 and
            design['instrument_assignment_probabilities'] == [.2] * 5 and
            design['instrument_randomized_by_experiment'] is True and
            design['mar_response_intercept'] == 4. and design['direct_effect_delta'] == 0., 'Incorrect MAR law')
    # The generic historical audit assumes its original intercept of 0.3.
    # Evaluate the declared high-response MAR law independently for this audit.
    previous = simulation_audit.response_table

    def independent_mar_law(scenario, residual, u, k):
        require(scenario == 'mar_x' and k == 5, 'Unexpected MAR audit response law')
        return np.repeat(expit(4. + 1.2 * np.asarray(u))[:, None], k, axis=1)

    simulation_audit.response_table = independent_mar_law
    try:
        result, arrays, _, _ = simulation_audit.validate_case(directory.parent.parent, DATASET, 'mar_x', args, cache)
    finally:
        simulation_audit.response_table = previous
    source = reconstruct_source(DATASET, False, args, cache)
    train, pool = source['train'], source['pool']
    probability = expit(4. + 1.2 * source['u'])
    for part in ['train', 'pool']:
        np.testing.assert_array_equal(arrays[part + '_z'], source['z'][source[part]])
        np.testing.assert_allclose(arrays[part + '_covariate_index'], source['u'][source[part]], atol=1e-13)
    np.testing.assert_array_equal(arrays['train_uniform'], source['uniforms'])
    np.testing.assert_allclose(arrays['train_probability'], probability[train], rtol=2e-13)
    np.testing.assert_array_equal(arrays['train_r'], source['uniforms'] < probability[train])
    require(set(arrays['train_z']) == set(range(5)), 'MAR training arms missing')
    for key in ['empirical_gamma', 'mean_response_probability', 'mean_assigned_response_probability', 'mean_max_response_probability']:
        np.testing.assert_allclose(design[key], probability[pool].mean())
    trials, diagnostics = pd.read_csv(directory / 'trials.csv'), pd.read_csv(directory / 'diagnostics.csv')
    nc = nt = min(500, len(pool) // 2)
    stream = np.random.default_rng(args['seed'] + 19937)
    for trial in range(args['trials']):
        order = stream.permutation(len(pool))
        ci, ti = order[:nc], order[nc:nc + nt]
        assigned, uniform = stream.integers(0, 5, len(pool)), stream.random(nc)
        observed = uniform < probability[pool][ci]
        # The saved Windows-run hashes serialized permutation indices as int32,
        # assignments as int64, and uniforms as float64, all little-endian.
        # NumPy 2 uses int64 permutation indices on Windows; preserve the
        # recorded byte representation while checking the exact replayed draws.
        hash_arrays = [np.asarray(ci, dtype='<i4'), np.asarray(ti, dtype='<i4'),
                       np.asarray(assigned, dtype='<i8'), np.asarray(uniform, dtype='<f8')]
        digest = hashlib.sha256(b''.join(a.tobytes() for a in hash_arrays)).hexdigest()
        require((diagnostics.loc[diagnostics.trial == trial, 'randomization_hash'] == digest).all(), 'MAR trial randomization differs')
        require((trials.loc[trials.trial == trial, 'calibration_observed'] == observed.sum()).all(), 'MAR calibration mask differs')
    result.update(passed=True, randomized_five_arm_instrument_verified=True,
                  all_training_and_trial_randomness_independently_replayed=True,
                  response_law='expit(4+1.2u), identical across five arms and independent of Y',
                  actual_mean_response=float(probability[pool].mean()))
    return result


def audit_natural(directory, job, plan, cohort):
    design = provenance(directory, job, plan)
    args = job['arguments']
    require(design['original_instrument_retained'] is True and design['instrument_coarsened'] is False
            and design['instrument_arms'] == 5,
            'Natural instrument was coarsened')
    require(design['instrument'] == dict(type='original categorical Z', levels=LEVELS, coarsened_instrument=False),
            'Natural instrument metadata differs')
    require(design['additional_xz_features'] == ['instrument_' + level for level in LEVELS],
            'Natural X,Z comparators did not retain five categories')
    groups = cohort.household_id.astype(str).to_numpy()
    unique = np.unique(groups)
    stream = np.random.default_rng(args['seed'])
    order = stream.permutation(unique)
    a, b = len(unique) // 2, 3 * len(unique) // 4
    parts = [np.flatnonzero(np.isin(groups, chosen)) for chosen in [order[:a], order[a:b], order[b:]]]
    parts[0] = np.sort(stream.permutation(parts[0])[:args['max_train']])
    parts[1] = np.sort(stream.permutation(parts[1])[:args['max_calibration']])
    response = cohort.observed.to_numpy(bool)
    target = parts[2][~response[parts[2]]]
    membership = np.full(len(cohort), 'unused', dtype=object)
    for indices, label in zip(parts, ['train', 'calibration', 'target_pool']):
        membership[indices] = label
    membership[target] = 'missing_target'
    z = np.searchsorted(CODES, cohort.instrument.to_numpy())
    split = pd.read_csv(directory / 'split.csv', dtype={'split_group': str})
    for name, expected in [('row_id', cohort.row_id), ('household_id', cohort.household_id),
                           ('split_group', groups), ('membership', membership),
                           ('observed', response.astype(int)), ('instrument_arm', z)]:
        np.testing.assert_array_equal(split[name], expected)
    for name, indices in zip(['train', 'calibration', 'target_pool'], parts):
        require(design['n_' + name] == len(indices), 'Natural split sample size differs')
    require(design['n_train_observed'] == int(response[parts[0]].sum()) and
            design['n_calibration_observed'] == int(response[parts[1]].sum()) and
            design['n_missing_targets'] == len(target), 'Natural observed/target counts differ')
    require(set(z[parts[0]]) == set(range(5)) and len(design['training_by_arm']) == 5,
            'Natural training instrument arms missing')
    for arm in design['training_by_arm']:
        selected = z[parts[0]] == arm['arm']
        require(arm['rows'] == int(selected.sum()) and
                arm['observed'] == int(response[parts[0]][selected].sum()), 'Natural per-arm counts differ')
    require(cohort.outcome.iloc[target].isna().all(), 'Natural target outcome was imputed')
    intervals, summary = pd.read_csv(directory / 'intervals.csv'), pd.read_csv(directory / 'summary.csv')
    require('coverage' not in summary and 'coverage' not in intervals and 'outcome' not in intervals,
            'Natural output reports unknown outcomes or coverage')
    require(set(intervals.method) == set(METHODS) and len(summary) == len(METHODS) * len(args['targets']),
            'Natural method set differs')
    require(not summary.duplicated(['target', 'method']).any(), 'Duplicated natural summaries')
    np.testing.assert_array_equal(np.sort(summary.target.unique()), np.sort(args['targets']))
    require(len(intervals) == len(target) * len(summary) and
            not intervals.duplicated(['row_id', 'target', 'method']).any(), 'Natural interval rows incomplete')
    require(not intervals[['center', 'lower', 'upper', 'length']].isna().any().any() and
            (intervals.observed == 0).all(), 'Invalid natural interval endpoint or response indicator')
    for _, row in summary.iterrows():
        values = intervals[(intervals.method == row.method) & np.isclose(intervals.target, row.target)]
        np.testing.assert_array_equal(values.row_id, cohort.row_id.iloc[target])
        np.testing.assert_array_equal(values.original_instrument, cohort.instrument.iloc[target])
        np.testing.assert_array_equal(values.instrument_arm, z[target])
        np.testing.assert_allclose(values.length, values.upper - values.lower)
        np.testing.assert_allclose(values.lower, values.center - values.length / 2)
        np.testing.assert_allclose(values.upper, values.center + values.length / 2)
        np.testing.assert_array_equal(values.finite, np.isfinite(values.length))
        require((values.lower <= values.upper).all(), 'Natural interval endpoints reversed')
        for name, expected in width_summary(values.length.to_numpy()).items():
            np.testing.assert_allclose(row[name], expected, equal_nan=True)
        worst = intervals[(intervals.method == 'Worst-case CP (X,Z)') & np.isclose(intervals.target, row.target)]
        containment = (values.lower.to_numpy() >= worst.lower.to_numpy()) & (values.upper.to_numpy() <= worst.upper.to_numpy())
        np.testing.assert_array_equal(values.contained_in_worst_case, containment)
        np.testing.assert_allclose(row.contained_in_worst_case_fraction, containment.mean())
    predictions = pd.read_csv(directory / 'response_predictions.csv')
    np.testing.assert_array_equal(predictions.row_id, cohort.row_id.iloc[target])
    require(predictions.probability_response_x.between(.02, .98).all() and
            predictions.probability_response_xz.between(.02, .98).all(), 'Invalid natural response predictions')
    return dict(passed=True, dataset=DATASET, missing_targets=len(target), interval_rows=len(intervals),
                native_five_category_iv_and_full_xz_indicators_verified=True,
                original_missing_outcomes_retained=True, no_unobserved_coverage_claim=True,
                one_family_draw_and_all_group_splits_replayed=True, all_interval_summaries_recomputed=True)


def save_mar_statistics():
    directory = ROOT / 'mar' / DATASET / 'mar_x'
    trials, summary = pd.read_csv(directory / 'trials.csv'), pd.read_csv(directory / 'summary.csv')
    rows = []
    for offset, method in enumerate(['Policy IV', 'Weighted CP (X)', 'Weighted CP (X,Z)']):
        selected = trials[(trials.method == method) & np.isclose(trials.target, .9)].sort_values('trial')
        np.testing.assert_array_equal(selected.trial, np.arange(200))
        row = summary[(summary.method == method) & np.isclose(summary.target, .9)].iloc[0]
        medians = selected.median_length.to_numpy()
        median = float(np.quantile(medians, .5, method='inverted_cdf'))
        require(median == row.median_length, 'Unconditional MAR median differs')
        seed = 20260925 + offset
        resampled = medians[np.random.default_rng(seed).integers(0, 200, size=(20000, 200))]
        boot = np.quantile(resampled, .5, axis=1, method='inverted_cdf')
        require(not np.isnan(boot).any(), 'Invalid bootstrap median')
        infinite = int(np.isinf(boot).sum())
        rows.append(dict(dataset=DATASET, method=method, target=.9, trials=200,
            coverage=row.coverage, coverage_mcse=row.coverage_mcse, mean_width=row.mean_length,
            mean_width_mcse=float(selected.mean_length.std(ddof=1) / np.sqrt(200))
                if np.isfinite(selected.mean_length).all() else np.nan,
            median_of_trial_median_widths=median,
            median_width_bootstrap_se=float(boot.std(ddof=1)) if infinite == 0 else np.nan,
            median_width_bootstrap_replicates=20000, median_width_bootstrap_seed=seed,
            median_width_bootstrap_infinite_count=infinite, infinite_fraction=row.infinite_fraction))
    pd.DataFrame(rows).to_csv(ROOT / 'mar_table_statistics.csv', index=False)
    return dict(statistic='Median of all trial median widths, retaining infinity',
                bootstrap_replicates=20000, seeds=[20260925, 20260926, 20260927],
                mean_width_se_undefined_when_any_trial_mean_is_infinite=True,
                output='mar_table_statistics.csv', rows=rows)


def audit_numerical_refinement(base_run, cache):
    """Keep the original grid flag while auditing any declared follow-up fit."""
    plan_path = ROOT / 'numerical_refinement_plan.json'
    if not plan_path.exists():
        return []
    plan = read_json(plan_path)
    run_path = ROOT / 'numerical_refinement_run.json'
    require(run_path.is_file(), 'Declared numerical-refinement run is incomplete')
    run = read_json(run_path)
    require(plan['completed'] is False and run['completed'] is True,
            'Numerical-refinement plan/run completion differs')
    for key in plan:
        if key not in {'completed', 'case_status'}:
            require(plan[key] == run[key], f'Numerical-refinement plan changed: {key}')
    require(run['hashes'] == base_run['hashes'], 'Numerical refinement changed executable/data hashes')
    require(len(run['jobs']) == len(run['case_status']) == 1 and
            run['case_status'][0]['completed'] is True, 'Numerical-refinement fit/status incomplete')
    job = run['jobs'][0]
    require((job['kind'], job['tag'], job['scenario'], job['dataset']) ==
            ('mar', 'mar_grid_refined', 'mar_x', DATASET), 'Unexpected numerical-refinement protocol')
    require(all(run['case_status'][0][key] == job[key] for key in ['kind', 'tag', 'scenario', 'dataset']),
            'Numerical-refinement completion status differs from job')
    expected = dict(next(job['arguments'] for job in base_run['jobs'] if job['tag'] == 'mar_grid'))
    # Frozen plans use Windows paths; fresh fits may use POSIX paths. Compare
    # each recorded path in its own syntax without rewriting fit provenance.
    path_type = PureWindowsPath if '\\' in expected['tag'] else PurePosixPath
    grid_tag = path_type(expected['tag'])
    refined_tag = path_type(job['arguments']['tag'])
    require(grid_tag.name == 'mar_grid' and refined_tag.name == 'mar_grid_refined'
            and str(grid_tag.parent) == str(refined_tag.parent),
            'Refinement output tag differs from the recorded grid-plan directory')
    expected.update(grid_size=3201, tag=str(grid_tag.with_name('mar_grid_refined')))
    require(job['arguments'] == expected and expected['trials'] == 10,
            'Refinement changed more than the integration grid and output tag')
    directory = ROOT / job['tag'] / DATASET / job['scenario']
    record = audit_mar(directory, job, run, cache)
    record.update(tag=job['tag'], scenario=job['scenario'],
                  manifest=str(run_path.relative_to(BASE)), manifest_sha256=historical.digest(run_path),
                  frozen_plan_sha256=historical.digest(plan_path))
    print('Validated supplemental MAR refinement at 3201 nodes', flush=True)
    return [record]


def published_mar_grid_check(refinement):
    """Audit the three displayed 90% methods separately from known-e controls."""
    methods = ['Policy IV', 'Weighted CP (X)', 'Weighted CP (X,Z)']
    tags = ['mar', 'mar_grid'] + (['mar_grid_refined'] if refinement else [])
    reference, comparisons = None, []
    keys = ['trial', 'method', 'target']
    for tag in tags:
        directory = ROOT / tag / DATASET / 'mar_x'
        frame = pd.read_csv(directory / 'trials.csv')
        take = frame[(frame.trial < 10) & np.isclose(frame.target, .9) & frame.method.isin(methods)]
        take = take.sort_values(keys).reset_index(drop=True)
        require(len(take) == 30 and not take.duplicated(keys).any() and
                set(take.trial) == set(range(10)), 'Published MAR comparison rows missing')
        if reference is None:
            reference = take
        else:
            pd.testing.assert_frame_equal(take, reference, check_exact=True)
            np.testing.assert_array_equal(np.isfinite(take.mean_length), np.isfinite(reference.mean_length))
            np.testing.assert_array_equal(take.infinite_fraction, reference.infinite_fraction)
            comparisons.append(dict(tag=tag, grid_size=read_json(directory / 'design.json')['arguments']['grid_size'],
                                    paired_rows=len(take), exact_trial_outputs_unchanged=True))
    return dict(target=.9, methods=methods, trials=10, reference_grid=801,
                comparisons=comparisons, coverage_and_all_width_metrics_exactly_unchanged=True,
                finite_infinite_decisions_identical=True,
                qualification='Checks the first ten displayed-method repetitions; the original all-method MAR grid flag remains reported.')


def refinement_changes():
    """Expose every changed saved row, including sensitivity at other targets."""
    first = ROOT / 'mar_grid' / DATASET / 'mar_x'
    second = ROOT / 'mar_grid_refined' / DATASET / 'mar_x'
    keys = ['trial', 'method', 'target']
    left, right = [pd.read_csv(folder / 'trials.csv').sort_values(keys).reset_index(drop=True)
                   for folder in [first, second]]
    pd.testing.assert_frame_equal(left[keys], right[keys], check_exact=True)
    metrics = ['coverage', 'mean_length', 'median_length', 'finite_mean_length',
               'finite_median_length', 'infinite_fraction']
    a, b = left[metrics].to_numpy(float), right[metrics].to_numpy(float)
    changed = ((a != b) & ~(np.isnan(a) & np.isnan(b))).any(axis=1)
    scale = read_json(first / 'design.json')['reference_residual_sd']
    records = []
    for index in np.flatnonzero(changed):
        before, after = left.iloc[index], right.iloc[index]
        width_delta = (float(after.mean_length - before.mean_length)
                       if np.isfinite(before.mean_length) and np.isfinite(after.mean_length) else None)
        coverage_delta = float(after.coverage - before.coverage)
        flip = bool(before.infinite_fraction != after.infinite_fraction)
        records.append(dict(trial=int(before.trial), method=before.method, target=float(before.target),
            grid_1601={key: before[key] for key in metrics}, grid_3201={key: after[key] for key in metrics},
            coverage_change=coverage_delta, finite_mean_width_change=width_delta,
            finite_infinite_change=flip,
            material_difference_flag=bool(flip or abs(coverage_delta) > .01 or
                (width_delta is not None and abs(width_delta) / (2 * scale) > .01))))
    flagged = [record for record in records if record['material_difference_flag']]
    require({(row['trial'], row['method'], row['target']) for row in flagged} == {(2, 'Policy IV', .8)},
            'The documented remaining MAR sensitivity is no longer confined to Policy IV trial 2 at 80%')
    known_e = left.method.eq('Policy IV (known e)') & np.isclose(left.target, .9)
    pd.testing.assert_frame_equal(left.loc[known_e].reset_index(drop=True),
                                  right.loc[known_e].reset_index(drop=True), check_exact=True)
    return dict(changed_rows=records, material_rows=flagged,
        known_propensity_policy_90_exactly_unchanged_between_1601_and_3201=True,
        qualification='The 1601-to-3201-node comparison retains a flagged Policy IV change at the 80% target in trial 2: width changes by about 0.07305 inches and coverage by 0.002. The displayed three 90% methods are exactly unchanged across all three grids. Numerical convergence across all targets is not established; thresholds and historical flags are unchanged.')


def main():
    run = audit_plan()
    cohort, source = audit_source()
    cache, cases, records = {}, {}, []
    for job in run['jobs']:
        directory = ROOT / job['tag'] / DATASET
        if job['kind'] != 'natural':
            directory /= job['scenario']
        if job['kind'] == 'masking':
            record, arrays, design = audit_masking(directory, job, run, cache)
            if job['tag'] == 'primary':
                cases[(DATASET, job['scenario'])] = arrays, design
        elif job['kind'] == 'mar':
            record = audit_mar(directory, job, run, cache)
        else:
            record = audit_natural(directory, job, run, cohort)
        records.append(dict(record, tag=job['tag'], scenario=job['scenario']))
        print(f'Validated five-category NHANES {job["tag"]}/{job["scenario"]}', flush=True)
    paired = historical.validate_pairs(cases)
    grids = {family: compare_fit(ROOT / 'primary', ROOT / 'grid', DATASET, family + '_g100')
             for family in ['anchored', 'crossing']}
    grids['mar'] = compare_outputs(ROOT / 'mar' / DATASET / 'mar_x', ROOT / 'mar_grid' / DATASET / 'mar_x')
    grids['natural'] = compare_outputs(ROOT / 'natural' / DATASET, ROOT / 'natural_grid' / DATASET, True)
    additional = audit_numerical_refinement(run, cache)
    if additional:
        grids['mar_refinement'] = compare_outputs(ROOT / 'mar_grid' / DATASET / 'mar_x',
                                                  ROOT / 'mar_grid_refined' / DATASET / 'mar_x')
    numerical_review = refinement_changes() if additional else None
    published_mar = published_mar_grid_check(bool(additional))
    pieces = []
    for job in run['jobs']:
        directory = ROOT / job['tag'] / DATASET
        if job['kind'] != 'natural':
            directory /= job['scenario']
        pieces.append(pd.read_csv(directory / 'summary.csv').assign(
            fit_kind=job['kind'], fit_tag=job['tag'], fit_seed=job['arguments']['seed'], instrument_arms=5))
    expected, actual = pd.concat(pieces, ignore_index=True), pd.read_csv(ROOT / 'summary.csv')
    keys = ['dataset', 'fit_kind', 'fit_tag', 'scenario', 'target', 'method']
    pd.testing.assert_frame_equal(actual.sort_values(keys).reset_index(drop=True),
                                  expected.sort_values(keys).reset_index(drop=True), check_dtype=False)
    result = dict(passed=True, fits=len(records), additional_numerical_fits=len(additional),
                  total_fits=len(records) + len(additional), additional_numerical_cases=additional,
                  source=source, cases=records,
                  primary_gamma_pairing=paired, grid_checks=grids, combined_summaries_verified=True,
                  published_mar_90_grid_check=published_mar,
                  published_mar_90_grid_stable=True,
                  numerical_convergence_all_targets=not any(
                      comparison['material_difference_flag'] for comparison in grids.values()),
                  numerical_refinement_review=numerical_review,
                  mar_table_statistics=save_mar_statistics(),
                  interpretation='Audits source data, implementation, and empirical summaries; it does not validate interviewer exclusion or estimate coverage on genuinely missing outcomes.')
    write_json(ROOT / 'validation.json', result)
    print(json.dumps(dict(passed=True, fits=len(records), additional_numerical_fits=len(additional),
                          total_fits=result['total_fits'], grid_checks=len(grids))), flush=True)


if __name__ == '__main__':
    main()
