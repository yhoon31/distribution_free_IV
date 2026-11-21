
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm

from run_experiments import BASE
from validate_feedback_simulations import (
    compare_summaries, recompute_summary, reconstruct_source, require,
)


ROOT = BASE/'results'/'observed_gamma'
METHODS = {'Softmax IV', 'Policy IV', 'Naive CP', 'Weighted CP (X)',
           'Naive CP (X,Z)', 'Weighted CP (X,Z)'}
FIT_ARGUMENTS = ['trials', 'targets', 'seed', 'n_train', 'pool_size',
                 'trees', 'min_leaf', 'grid_size']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def independent_probabilities(family, gamma, anchor, residual, u, k):
    """Evaluate the stated mechanism without calling the new runner."""
    residual, u = np.asarray(residual), np.asarray(u)
    if family == 'original_logistic':
        return expit(.3+.7*u[:, None]-.8*residual[:, None]**2+
                     np.linspace(-1., 1., k)[None, :])
    if family == 'anchored':
        coefficients = np.empty(k)
        coefficients[anchor] = 0.
        coefficients[np.arange(k) != anchor] = np.linspace(.4, 1., k-1)
        p = gamma*(.10+.65*np.exp(-.75*residual[:, None]**2)*coefficients[None, :])
        p[:, anchor] = gamma
        return p
    if family == 'crossing':
        centers = norm.ppf((np.arange(k)+.5)/k)
        raw = .1+.9*np.exp(-.5*((residual[:, None]-centers[None, :])/.8)**2)
        return gamma*raw/raw.max(axis=1, keepdims=True)
    raise AssertionError(f'Unknown response family: {family}')


def compare_historical_control(directory, dataset, args, trials, diagnostics):
    """Compare every saved numerical trial to the already published control."""
    historical_tag = {20260919: 'primary', 20260920: 'sensitivity_20260920'}.get(args['seed'])
    if historical_tag is None:
        return dict(checked=False, reason='No historical run for this seed')
    old = BASE/'results'/'feedback'/historical_tag/dataset/'observed_iv_mnar'
    design = json.loads((old/'design.json').read_text())
    keys = [key for key in FIT_ARGUMENTS if key != 'trials']
    if any(design['arguments'][key] != args[key] for key in keys):
        return dict(checked=False, reason='Historical fit settings differ')
    expected = pd.read_csv(old/'trials.csv')
    expected = expected[expected.trial < args['trials']].copy()
    require(expected.trial.nunique() == args['trials'], 'Insufficient historical control repetitions')
    keys = ['trial', 'method', 'target']
    columns = [c for c in expected.columns if c not in ['scenario', 'gamma']]
    actual = trials.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual[columns], expected[columns],
        check_dtype=False, check_exact=True)
    expected_diagnostics = pd.read_csv(old/'diagnostics.csv')
    expected_diagnostics = expected_diagnostics[expected_diagnostics.trial < args['trials']]
    keys = ['trial', 'method']
    pd.testing.assert_frame_equal(diagnostics[expected_diagnostics.columns].sort_values(keys).reset_index(drop=True),
        expected_diagnostics.sort_values(keys).reset_index(drop=True),
        check_dtype=False, check_exact=True)
    return dict(checked=True, trial_rows=len(actual), historical_directory=str(old.relative_to(BASE)))


def validate_metrics(trials, diagnostics, args, dataset, scenario, n_test):
    require(set(trials.method) == METHODS, 'Unexpected comparison methods')
    require(set(trials.dataset) == {dataset} and set(trials.scenario) == {scenario}, 'Incorrect trial labels')
    require(not trials.duplicated(['trial', 'method', 'target']).any(), 'Duplicated trial rows')
    np.testing.assert_allclose(np.sort(trials.target.unique()), np.sort(args['targets']))
    require(len(trials) == args['trials']*len(METHODS)*len(args['targets']), 'Incomplete trial rows')
    for _, group in trials.groupby(['method', 'target']):
        require(set(group.trial) == set(range(args['trials'])), 'Incomplete trial numbering')
    require((trials.seed == args['seed']).all(), 'Wrong seed')
    require((trials.n_test == n_test).all() and (trials.n_test_observed == n_test).all(),
            'Incorrect test sample size')
    for metric in ['coverage', 'observed_coverage', 'coverage_lower', 'coverage_upper', 'infinite_fraction']:
        require(trials[metric].between(0, 1).all(), f'Invalid {metric}')
    for metric in ['observed_coverage', 'coverage_lower', 'coverage_upper']:
        np.testing.assert_allclose(trials.coverage, trials[metric], atol=2e-15)
    for metric in ['mean_length', 'median_length']:
        require(trials[metric].notna().all() and (trials[metric] >= 0).all(), f'Invalid {metric}')
    scalar = trials[trials.method.isin(['Softmax IV', 'Policy IV', 'Naive CP', 'Naive CP (X,Z)'])]
    require(scalar.infinite_fraction.isin([0., 1.]).all(), 'Scalar intervals partially infinite')
    require((scalar.loc[scalar.infinite_fraction == 1, 'coverage'] == 1).all(), 'Infinite interval failed coverage')
    require(len(diagnostics) == 2*args['trials'], 'Incomplete diagnostic rows')
    require(not diagnostics.duplicated(['trial', 'method']).any(), 'Duplicate diagnostics')
    require(set(diagnostics.method) == {'Softmax IV', 'Policy IV'}, 'Wrong diagnostic methods')
    require(np.isfinite(diagnostics.identified_mass_eif).all(), 'Nonfinite identified-mass diagnostics')


def validate_case(directory, args, cache):
    for name in ['design.json', 'inputs_before_run.json', 'split.npz',
                 'trials.csv', 'summary.csv', 'diagnostics.csv']:
        require((directory/name).is_file(), f'Incomplete output: {directory/name}')
    design = json.loads((directory/'design.json').read_text())
    dataset, scenario = design['dataset'], design['scenario']
    family, gamma, anchor = design['response_design'], design['response_envelope'], design['anchor_arm']
    require(directory.name == scenario and directory.parent.name == dataset, 'Design path mismatch')
    require(all(design['arguments'][key] == args[key] for key in FIT_ARGUMENTS), 'Batch/fit arguments differ')
    pre_run = json.loads((directory/'inputs_before_run.json').read_text())
    require(design['hashes'] == pre_run, 'Pre-run provenance differs from fitted provenance')
    for name, expected in pre_run.items():
        require(digest(BASE/name) == expected, f'Input hash changed: {name}')
    source = reconstruct_source(dataset, True, args, cache)
    for name in ['path', 'metadata_path']:
        hash_name = 'sha256' if name == 'path' else 'metadata_sha256'
        source_name = source['source'][name].replace('\\', '/')
        require(pre_run[source_name] == source['source'][hash_name], 'Missing pre-run source hash')
    frame, train, pool, pilot = (source[key] for key in ['frame', 'train', 'pool', 'pilot'])
    with np.load(directory/'split.npz', allow_pickle=False) as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    for name, indices in [('pilot', pilot), ('train', train), ('pool', pool)]:
        np.testing.assert_array_equal(arrays[name+'_rows'], frame.row_id.to_numpy()[indices])
        np.testing.assert_array_equal(arrays[name+'_z'], source['z'][indices])
        np.testing.assert_array_equal(arrays[name+'_original_z'], frame.instrument.to_numpy()[indices])
        np.testing.assert_array_equal(arrays[name+'_y'], source['y'][indices])
        require(design['n_'+name] == len(indices), f'Incorrect {name} sample size')
    for left, right in [(pilot, train), (pilot, pool), (train, pool)]:
        require(not set(frame.row_id.iloc[left]) & set(frame.row_id.iloc[right]), 'Split source rows overlap')
        require(not set(frame.household_id.iloc[left]) & set(frame.household_id.iloc[right]), 'Split households overlap')
    require(design['original_cohort_rows'] == source['original_n'], 'Incorrect original cohort count')
    require(design['original_missing_rows'] == source['original_missing'], 'Incorrect original missing count')
    require(design['eligible_complete_case_households'] == len(frame), 'Incorrect eligible cohort count')
    require(design['original_instrument_retained'] and design['original_outcomes_retained'], 'Retention labels missing')
    require(design['instrument'] == source['instrument'], 'Original instrument encoding differs')
    require(design['instrument_arms'] == source['k'], 'Wrong instrument arm count')
    require(design['exclusion_status'] == 'unverified after complete-case selection', 'Unjustified exclusion claim')
    k = source['k']
    expected_anchor = int(np.bincount(source['z'][train], minlength=k).argmax())
    require(anchor == expected_anchor and design['anchor'] == anchor, 'Anchor is not training-modal Z')
    for key, expected in [('reference_residual_sd', source['scale']),
                          ('covariate_index_mean', source['u_mean']), ('covariate_index_sd', source['u_sd'])]:
        np.testing.assert_allclose(design[key], expected, rtol=1e-12, atol=1e-12)
    for name, indices in [('train', train), ('pool', pool)]:
        residual = (source['y'][indices]-source['reference'][indices])/source['scale']
        p = independent_probabilities(family, gamma, anchor, residual, source['u'][indices], k)
        np.testing.assert_allclose(arrays[name+'_probability_table'], p, rtol=2e-13, atol=2e-15)
        actual = p[np.arange(len(indices)), source['z'][indices]]
        np.testing.assert_allclose(arrays[name+'_probability'], actual, rtol=2e-13, atol=2e-15)
    np.testing.assert_array_equal(arrays['train_uniforms'], source['uniforms'])
    np.testing.assert_array_equal(arrays['train_r'], source['uniforms'] < arrays['train_probability'])
    require(design['n_train_observed'] == int(arrays['train_r'].sum()), 'Training respondent count differs')
    require(design['gamma_under_exclusion'] == gamma, 'Wrong conditional identification claim')
    require(design['gamma_lower_bound_without_exclusion'] == (gamma if family == 'anchored' else None),
            'Wrong unconditional envelope lower bound')
    if gamma is not None:
        np.testing.assert_allclose(arrays['pool_probability_table'].max(axis=1), gamma, atol=2e-15)
    np.testing.assert_allclose(design['empirical_mean_max_response_probability'],
                               arrays['pool_probability_table'].max(axis=1).mean())
    np.testing.assert_allclose(design['mean_response_probability_actual_z'], arrays['pool_probability'].mean())
    np.testing.assert_allclose(design['mean_response_probability'], arrays['pool_probability'].mean())
    np.testing.assert_allclose(design['mean_conditional_arm_range'], np.ptp(arrays['pool_probability_table'], axis=1).mean())
    for name in ['train', 'pool']:
        np.testing.assert_allclose(design['anchor_training_fraction' if name == 'train' else 'anchor_pool_fraction'],
                                   (arrays[name+'_z'] == anchor).mean())
    for arm in design['training_by_arm']:
        take = arrays['train_z'] == arm['arm']
        require(arm['total'] == int(take.sum()) and arm['observed'] == int((take & arrays['train_r']).sum()),
                'Per-arm training count differs')
    e = arrays['pool_e']
    require(e.shape == (len(pool), k) and np.isfinite(e).all() and (e > 0).all(), 'Invalid fitted propensities')
    np.testing.assert_allclose(e.sum(axis=1), 1., atol=1e-12)
    np.testing.assert_allclose(design['gamma_plugin_mean'], arrays['pool_gamma_plugin'].mean())
    np.testing.assert_allclose(design['gamma_plugin_above_one'], (arrays['pool_gamma_plugin'] > 1).mean())
    nc = nt = min(500, len(pool)//2)
    require(design['n_calibration'] == nc and design['n_test'] == nt, 'Incorrect evaluation counts')
    for name in ['calibration_idx', 'test_idx', 'calibration_r', 'calibration_uniforms']:
        require(arrays[name].shape == (args['trials'], nc), f'Wrong shape: {name}')
    trial_rng = np.random.default_rng(args['seed']+19937)
    for trial in range(args['trials']):
        perm = trial_rng.permutation(len(pool))
        ci, ti = perm[:nc], perm[nc:nc+nt]
        uniforms = trial_rng.random(nc)
        np.testing.assert_array_equal(arrays['calibration_idx'][trial], ci)
        np.testing.assert_array_equal(arrays['test_idx'][trial], ti)
        np.testing.assert_array_equal(arrays['calibration_uniforms'][trial], uniforms)
        np.testing.assert_array_equal(arrays['calibration_r'][trial], uniforms < arrays['pool_probability'][ci])
    trials = pd.read_csv(directory/'trials.csv')
    diagnostics = pd.read_csv(directory/'diagnostics.csv')
    validate_metrics(trials, diagnostics, args, dataset, scenario, nt)
    if gamma is None:
        require(trials.gamma.isna().all(), 'Control has a false exact Gamma label')
    else:
        np.testing.assert_allclose(trials.gamma, gamma)
    observed_by_trial = arrays['calibration_r'].sum(axis=1)
    np.testing.assert_array_equal(trials.calibration_observed, observed_by_trial[trials.trial])
    np.testing.assert_array_equal(diagnostics.observed, observed_by_trial[diagnostics.trial])
    actual_e = e[arrays['calibration_idx'], arrays['pool_z'][arrays['calibration_idx']]]
    min_e = actual_e.min(axis=1)
    np.testing.assert_allclose(diagnostics.min_e, min_e[diagnostics.trial])
    np.testing.assert_allclose(diagnostics.max_inverse_e, 1/min_e[diagnostics.trial])
    for method, group in diagnostics.groupby('method'):
        np.testing.assert_allclose(design['identified_mass_eif_mean'][method], group.identified_mass_eif.mean())
    summary = pd.read_csv(directory/'summary.csv')
    compare_summaries(summary, recompute_summary(trials))
    historical = compare_historical_control(directory, dataset, args, trials, diagnostics) if family == 'original_logistic' else None
    audit = dict(dataset=dataset, scenario=scenario, trial_rows=len(trials),
        pre_run_hashes_verified=pre_run, original_rows_outcomes_and_instruments_verified=True,
        disjoint_households_verified=True, pilot_generator_reproduced=True,
        training_and_all_calibration_masks_reproduced=True,
        all_trial_splits_reproduced=True, summary_and_mcse_recomputed=True,
        historical_control=historical)
    return audit, arrays, design, summary


def validate_pairs(cases):
    audit = []
    paired_keys = ['pilot_rows', 'train_rows', 'pool_rows', 'pilot_z', 'train_z', 'pool_z',
                   'pilot_original_z', 'train_original_z', 'pool_original_z',
                   'pilot_y', 'train_y', 'pool_y', 'train_uniforms',
                   'calibration_idx', 'test_idx', 'calibration_uniforms', 'pool_e']
    for dataset in sorted({key[0] for key in cases}):
        selected = [value for (ds, _), value in cases.items() if ds == dataset]
        reference = selected[0][0]
        for arrays, _ in selected[1:]:
            for name in paired_keys:
                np.testing.assert_array_equal(arrays[name], reference[name])
        for family in ['anchored', 'crossing']:
            family_cases = sorted([value for value in selected if value[1]['response_design'] == family],
                                  key=lambda value: value[1]['response_envelope'])
            for (left, dl), (right, dr) in zip(family_cases[:-1], family_cases[1:]):
                for name in ['train_probability_table', 'pool_probability_table']:
                    np.testing.assert_allclose(left[name]/dl['response_envelope'],
                                               right[name]/dr['response_envelope'], atol=2e-15)
                for name in ['train_r', 'calibration_r']:
                    require((left[name] <= right[name]).all(), 'Common-uniform masks are not nested as Gamma increases')
        audit.append(dict(dataset=dataset, cases=len(selected),
            all_source_rows_outcomes_instruments_and_propensities_paired=True,
            all_uniforms_and_trial_splits_paired=True, response_scaling_and_nested_masks_verified=True))
    return audit


def validate_grid_refinement(tag, cases):
    """Check that refinement changes integration, not samples or simulation."""
    baseline_tag = {'crossing_refined': 'primary',
                    'crossing_refined_s20260920': 'sensitivity_20260920'}.get(tag)
    if baseline_tag is None:
        return None
    audit = []
    for (dataset, scenario), (arrays, design) in cases.items():
        baseline = ROOT/baseline_tag/dataset/scenario
        original = json.loads((baseline/'design.json').read_text())
        require(original['hashes'] == design['hashes'], 'Inputs or runner changed during grid refinement')
        for key in FIT_ARGUMENTS:
            if key != 'grid_size':
                require(original['arguments'][key] == design['arguments'][key], f'Refinement changed {key}')
        require(original['arguments']['grid_size'] == 801 and design['arguments']['grid_size'] == 1601,
                'Unexpected integration-grid comparison')
        for key in ['response_design', 'response_envelope', 'anchor_arm', 'instrument',
                    'gamma_under_exclusion', 'gamma_lower_bound_without_exclusion',
                    'exclusion_status', 'reference_residual_sd', 'covariate_index_mean',
                    'covariate_index_sd', 'mean_response_probability_actual_z']:
            require(original[key] == design[key], f'Refinement changed the experiment: {key}')
        with np.load(baseline/'split.npz', allow_pickle=False) as saved:
            require(set(saved.files) == set(arrays), 'Saved arrays differ after grid refinement')
            for name in saved.files:
                # The fitted envelope integral is itself a quadrature output.
                if name != 'pool_gamma_plugin':
                    np.testing.assert_array_equal(arrays[name], saved[name],
                                                  err_msg=f'Refinement changed {dataset}/{scenario}/{name}')
        trial_keys = ['trial', 'method', 'target']
        before = pd.read_csv(baseline/'trials.csv').sort_values(trial_keys).reset_index(drop=True)
        after = pd.read_csv(ROOT/tag/dataset/scenario/'trials.csv').sort_values(trial_keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(before[trial_keys], after[trial_keys], check_exact=True)
        baseline_methods = ~before.method.isin(['Softmax IV', 'Policy IV'])
        pd.testing.assert_frame_equal(before.loc[baseline_methods].reset_index(drop=True),
            after.loc[baseline_methods].reset_index(drop=True), check_dtype=False, check_exact=True)
        iv = ~baseline_methods
        finite_before = before.infinite_fraction == 0
        finite_after = after.infinite_fraction == 0
        both_finite = iv & finite_before & finite_after
        coverage_change = np.abs(after.coverage-before.coverage)
        length_change = np.abs(after.loc[both_finite, 'mean_length']-before.loc[both_finite, 'mean_length'])
        audit.append(dict(dataset=dataset, scenario=scenario,
            baseline_tag=baseline_tag, original_grid=801, refined_grid=1601,
            source_rows_outcomes_instruments_masks_and_splits_identical=True,
            fitted_instrument_propensities_identical=True,
            non_iv_trial_outputs_exactly_identical=True,
            compared_iv_rows=int(iv.sum()),
            iv_finite_infinite_changes=int((iv & (finite_before != finite_after)).sum()),
            maximum_iv_coverage_change=float(coverage_change[iv].max()),
            maximum_iv_finite_mean_length_change=float(length_change.max()) if len(length_change) else None))
    return dict(baseline_tag=baseline_tag, cases=audit,
                qualification='These checks isolate the integration-grid change; they do not establish a quadrature-error bound or validate the IV assumption.')


def validate_tag(tag, cache):
    root = ROOT/tag
    require((root/'run.json').is_file(), f'Incomplete batch: {root}')
    manifest = json.loads((root/'run.json').read_text())
    args = manifest['arguments']
    require(args['tag'] == tag, 'Batch tag mismatch')
    audits, cases, summaries = [], {}, []
    for dataset in args['datasets']:
        for scenario in manifest['scenarios']:
            audit, arrays, design, summary = validate_case(root/dataset/scenario, args, cache)
            audits.append(audit)
            cases[(dataset, scenario)] = (arrays, design)
            summaries.append(summary)
    compare_summaries(pd.read_csv(root/'summary.csv'), pd.concat(summaries, ignore_index=True))
    paired = validate_pairs(cases)
    refinement = validate_grid_refinement(tag, cases)
    result = dict(passed=True, tag=tag, cases=len(audits),
        total_trial_rows=sum(a['trial_rows'] for a in audits), case_audits=audits,
        paired_design_audits=paired, integration_grid_refinement=refinement,
        combined_summary_matches=True)
    (root/'validation.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(f'{tag}: passed {len(audits)} cases, {result["total_trial_rows"]} trial records', flush=True)
    return dict(tag=tag, passed=True, cases=len(audits), total_trial_rows=result['total_trial_rows'],
                validation_sha256=digest(root/'validation.json'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tags', nargs='+', default=['primary', 'sensitivity_20260920'])
    args = parser.parse_args()
    cache = {}
    results = [validate_tag(tag, cache) for tag in args.tags]
    (ROOT/'validation.json').write_text(json.dumps(dict(passed=True, tags=results), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
