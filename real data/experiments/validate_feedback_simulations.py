
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

from feedback_simulations import SCENARIOS, encode_instrument, perturb_outcome, response_table
from run_experiments import BASE, load_data

ROOT = BASE / 'results' / 'feedback'
FIT_ARGUMENTS = ['trials', 'targets', 'seed', 'n_train', 'pool_size', 'trees', 'min_leaf', 'grid_size']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def expected_methods(scenario):
    methods = ['Softmax IV', 'Policy IV', 'Naive CP', 'Weighted CP (X)',
               'Naive CP (X,Z)', 'Weighted CP (X,Z)']
    native = scenario.startswith('observed_iv')
    if not native:
        methods += ['Softmax IV (known e)', 'Policy IV (known e)']
    if scenario in ['mar_x', 'mar_xz', 'observed_iv_mar']:
        methods += ['MAR CP (known p; X,Z)']
        if not native:
            methods += ['MAR CP (known p; X)']
    return set(methods)


def recompute_summary(trials):
    rows = []
    for key, group in trials.groupby(['dataset', 'scenario', 'method', 'alpha']):
        row = dict(zip(['dataset', 'scenario', 'method', 'alpha'], key))
        row.update(target=1-key[-1], trials=len(group))
        for name in ['coverage', 'observed_coverage', 'coverage_lower', 'coverage_upper',
                     'infinite_fraction', 'finite_mean_length', 'finite_median_length']:
            values = group[name].to_numpy(float)
            values = values[~np.isnan(values)]
            row[name] = float(np.mean(values)) if len(values) else np.nan
            row[name+'_mcse'] = float(np.std(values, ddof=1)/np.sqrt(len(values))) if len(values) > 1 else np.nan
        row['mean_length'] = float(np.mean(group.mean_length))
        row['median_length'] = float(np.quantile(group.median_length, .5, method='inverted_cdf'))
        finite = group.finite_median_length.dropna().to_numpy(float)
        row['finite_length_median'] = float(np.median(finite)) if len(finite) else np.nan
        row['finite_trial_count'] = len(finite)
        rows.append(row)
    return pd.DataFrame(rows)


def compare_summaries(actual, expected):
    keys = ['dataset', 'scenario', 'method', 'alpha']
    actual = actual.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    require(set(actual.columns) == set(expected.columns), 'Summary columns mismatch')
    pd.testing.assert_frame_equal(actual, expected[actual.columns], check_dtype=False,
                                  check_exact=False, atol=2e-12, rtol=2e-10)


def reconstruct_source(dataset, native, args, cache):
    key = (dataset, native, args['seed'], args['n_train'], args['pool_size'])
    if key in cache:
        return cache[key]
    original, original_x, meta, _ = load_data(dataset, 'natural' if native else None)
    source_stem = 'prepared_natural' if native and (BASE/'data'/dataset/'prepared_natural.csv').exists() else 'prepared'
    source_path = BASE/'data'/dataset/(source_stem+'.csv')
    metadata_path = BASE/'data'/dataset/(source_stem+'_metadata.json')
    keep = original.observed.to_numpy(bool)
    frame, x = original.loc[keep].reset_index(drop=True), original_x[keep]
    rng = np.random.default_rng(args['seed'])
    shuffled = rng.permutation(len(frame))
    _, first = np.unique(frame.household_id.to_numpy()[shuffled], return_index=True)
    keep = shuffled[np.sort(first)]
    frame, x = frame.iloc[keep].reset_index(drop=True), x[keep]
    require(frame.household_id.is_unique, 'More than one selected record per household')
    require(frame.observed.astype(bool).all() and np.isfinite(frame.outcome).all(),
            'Simulation source includes originally missing outcomes')
    order = rng.permutation(len(frame))
    n_pilot = min(1000, max(50, len(frame)//10))
    pilot, remaining = order[:n_pilot], order[n_pilot:]
    n_train = min(args['n_train'], int(.45*len(remaining)))
    train, pool = remaining[:n_train], remaining[n_train:n_train+args['pool_size']]
    y = frame.outcome.to_numpy(float)
    reference = RandomForestRegressor(n_estimators=100, min_samples_leaf=30,
        random_state=args['seed'], n_jobs=1, oob_score=True).fit(x[pilot], y[pilot])
    scale = max(float(np.std(y[pilot]-reference.oob_prediction_)), 1e-6)
    reference_mean = reference.predict(x)
    u_mean = float(reference_mean[pilot].mean())
    u_sd = max(float(reference_mean[pilot].std()), 1e-6)
    u = np.clip((reference_mean-u_mean)/u_sd, -3, 3)
    if native:
        z, instrument = encode_instrument(frame, train, meta)
        k = int(z.max())+1
    else:
        k, instrument = 5, {}
        z = rng.integers(0, k, len(frame))
    uniforms = rng.random(len(train))
    value = dict(frame=frame, y=y, z=z, k=k, instrument=instrument, pilot=pilot, train=train,
        pool=pool, uniforms=uniforms, reference=reference_mean, scale=scale, u=u,
        u_mean=u_mean, u_sd=u_sd, original_n=len(original),
        original_missing=int((~original.observed.astype(bool)).sum()),
        source=dict(path=str(source_path.relative_to(BASE)), sha256=digest(source_path),
                    metadata_path=str(metadata_path.relative_to(BASE)), metadata_sha256=digest(metadata_path)))
    cache[key] = value
    return value


def validate_case(root, dataset, scenario, args, cache):
    directory = root/dataset/scenario
    for filename in ['design.json', 'split.npz', 'trials.csv', 'summary.csv', 'diagnostics.csv']:
        require((directory/filename).is_file(), f'Incomplete case: {directory/filename}')
    design = json.loads((directory/'design.json').read_text())
    require(design['dataset'] == dataset and design['scenario'] == scenario, 'Design case mismatch')
    require(all(design['arguments'][key] == args[key] for key in FIT_ARGUMENTS), 'Design arguments differ from batch')
    for name, expected in design['hashes'].items():
        require(digest(BASE/name) == expected, f'Fitted source-code hash mismatch: {name}')
    native = scenario.startswith('observed_iv')
    source = reconstruct_source(dataset, native, args, cache)
    frame, train, pool, pilot = (source[name] for name in ['frame', 'train', 'pool', 'pilot'])
    with np.load(directory/'split.npz') as saved:
        arrays = {name: saved[name].copy() for name in saved.files}
    for name, indices in [('pilot', pilot), ('train', train), ('pool', pool)]:
        np.testing.assert_array_equal(arrays[name+'_rows'], frame.row_id.to_numpy()[indices])
        require(design['n_'+name] == len(indices), f'{name} count mismatch')
    for left, right in [(pilot, train), (pilot, pool), (train, pool)]:
        require(not set(frame.row_id.iloc[left]) & set(frame.row_id.iloc[right]), 'Source rows overlap')
        require(not set(frame.household_id.iloc[left]) & set(frame.household_id.iloc[right]), 'Households overlap')
    require(design['original_cohort_rows'] == source['original_n'], 'Original cohort count mismatch')
    require(design['original_missing_rows'] == source['original_missing'], 'Original missing count mismatch')
    require(design['eligible_complete_case_households'] == len(frame), 'Eligible household count mismatch')
    require(design['original_instrument_retained'] == native, 'Instrument-retention label mismatch')
    require(design['instrument'] == source['instrument'], 'Native instrument mapping differs from original source')
    require(design['instrument_arms'] == source['k'], 'Instrument arm count mismatch')
    np.testing.assert_array_equal(arrays['train_z'], source['z'][train])
    for key, expected in [('reference_residual_sd', source['scale']),
                          ('covariate_index_mean', source['u_mean']), ('covariate_index_sd', source['u_sd'])]:
        np.testing.assert_allclose(design[key], expected, rtol=1e-12, atol=1e-12)
    delta = {'invalid_half': .5, 'invalid_one': 1.}.get(scenario, 0.)
    require(design['direct_effect_delta'] == delta, 'Direct outcome effect mismatch')
    shifted_y = perturb_outcome(source['y'][train], source['z'][train], source['k'], source['scale'], delta)
    table = response_table(scenario, (shifted_y-source['reference'][train])/source['scale'],
                           source['u'][train], source['k'])
    probabilities = table[np.arange(len(train)), source['z'][train]]
    np.testing.assert_allclose(arrays['train_probability'], probabilities, rtol=2e-12, atol=2e-12)
    np.testing.assert_array_equal(arrays['train_r'], source['uniforms'] < probabilities)
    require(design['n_train_observed'] == int(arrays['train_r'].sum()), 'Observed training count mismatch')
    pdiag = response_table(scenario, (source['y'][pool]-source['reference'][pool])/source['scale'],
                           source['u'][pool], source['k'])
    np.testing.assert_allclose(design['mean_conditional_arm_range'], np.ptp(pdiag, axis=1).mean())
    if not native and not delta:
        np.testing.assert_allclose(design['empirical_gamma'], pdiag.max(axis=1).mean())
        np.testing.assert_allclose(design['mean_response_probability'], pdiag.mean())
    trials = pd.read_csv(directory/'trials.csv')
    expected = expected_methods(scenario)
    require(set(trials.method) == expected, f'Method set mismatch in {dataset}/{scenario}')
    require(not trials.duplicated(['trial', 'method', 'target']).any(), 'Duplicated trial/method/target')
    for (method, target), group in trials.groupby(['method', 'target']):
        require(len(group) == args['trials'] and set(group.trial) == set(range(args['trials'])),
                f'Incomplete repetitions for {method}/{target}')
    np.testing.assert_allclose(np.sort(trials.target.unique()), np.sort(args['targets']))
    require(len(trials) == args['trials']*len(expected)*len(args['targets']), 'Total trial count mismatch')
    require(set(trials.dataset) == {dataset} and set(trials.scenario) == {scenario}, 'Trial labels mismatch')
    require((trials.seed == args['seed']).all(), 'Seed mismatch')
    n_test = min(500, len(pool)//2)
    require((trials.n_test == n_test).all() and (trials.n_test_observed == n_test).all(), 'Incorrect evaluation sizes')
    require(trials.calibration_observed.between(0, n_test).all(), 'Calibration observed count out of range')
    for metric in ['coverage', 'observed_coverage', 'coverage_lower', 'coverage_upper', 'infinite_fraction']:
        require(trials[metric].between(0, 1).all(), f'{metric} out of range')
    for metric in ['mean_length', 'median_length']:
        require(trials[metric].notna().all() and (trials[metric] >= 0).all(), f'{metric} invalid')
    np.testing.assert_allclose(trials.coverage, trials.observed_coverage)
    np.testing.assert_allclose(trials.coverage, trials.coverage_lower)
    np.testing.assert_allclose(trials.coverage, trials.coverage_upper)
    for method in ['Softmax IV', 'Policy IV', 'Naive CP', 'Naive CP (X,Z)',
                   'Softmax IV (known e)', 'Policy IV (known e)']:
        subset = trials[trials.method == method]
        if not len(subset):
            continue
        require(subset.infinite_fraction.isin([0., 1.]).all(), 'Scalar-radius method partially infinite')
        require((subset.loc[subset.infinite_fraction == 1, 'coverage'] == 1).all(), 'Infinite interval misses an outcome')
    saved_summary = pd.read_csv(directory/'summary.csv')
    compare_summaries(saved_summary, recompute_summary(trials))
    diagnostics = pd.read_csv(directory/'diagnostics.csv')
    require(len(diagnostics) == 2*args['trials'], 'Diagnostic rows incomplete')
    require(not diagnostics.duplicated(['trial', 'method']).any(), 'Duplicate diagnostics')
    require(set(diagnostics.method) == {'Softmax IV', 'Policy IV'}, 'Diagnostic methods mismatch')
    require(np.isfinite(diagnostics.identified_mass_eif).all(), 'Nonfinite EIF diagnostic')
    audit = dict(dataset=dataset, scenario=scenario, trials_per_method_target=args['trials'],
        trial_rows=len(trials), methods=len(expected), targets=args['targets'],
        code_hashes_verified=design['hashes'], source_at_validation=source['source'],
        complete_case_eligibility_verified=True, disjoint_rows_and_households=True,
        training_probabilities_and_masks_reproduced=True,
        original_instrument_encoding_verified=native,
        summary_and_mcse_recomputed=True, reference_forest_reproduced=True,
        outcome_effect_delta=delta,
        arm_range_interpretation='Baseline unshifted-outcome design diagnostic, not the realized invalid-design first stage.' if delta else 'Conditional response range for the stated masking law.')
    return audit, arrays, design, saved_summary


def validate_pairs(cases):
    rows = []
    for dataset in sorted({key[0] for key in cases}):
        names = ['heterogeneous_mnar', 'weak_mnar', 'irrelevant_mnar']
        if not all((dataset, name) in cases for name in names):
            continue
        strong, weak, irrelevant = [cases[(dataset, name)] for name in names]
        for name in ['pilot_rows', 'train_rows', 'pool_rows', 'train_z']:
            np.testing.assert_array_equal(strong[0][name], weak[0][name])
            np.testing.assert_array_equal(strong[0][name], irrelevant[0][name])
        np.testing.assert_allclose(weak[0]['train_probability'],
            .25*strong[0]['train_probability']+.75*irrelevant[0]['train_probability'], atol=2e-12)
        np.testing.assert_allclose([strong[1]['mean_response_probability'], weak[1]['mean_response_probability']],
                                  irrelevant[1]['mean_response_probability'], atol=2e-12)
        np.testing.assert_allclose(weak[1]['mean_conditional_arm_range'], .25*strong[1]['mean_conditional_arm_range'])
        np.testing.assert_allclose(irrelevant[1]['mean_conditional_arm_range'], 0., atol=2e-12)
        np.testing.assert_allclose(weak[1]['empirical_gamma'], .25+.75*irrelevant[1]['empirical_gamma'])
        rows.append(dict(dataset=dataset, paired_rows_and_instruments_identical=True,
                         marginal_response_preserved=True, weak_strength=.25,
                         irrelevant_strength=0., identification_identity_verified=True))
    return rows


def mechanism_audit():
    residual = np.linspace(-6, 6, 1003)
    u = np.sin(residual)
    strong = response_table('heterogeneous_mnar', residual, u)
    for name, strength in [('weak_mnar', .25), ('irrelevant_mnar', 0.)]:
        current = response_table(name, residual, u)
        np.testing.assert_allclose(current.mean(axis=1), strong.mean(axis=1), atol=2e-15)
        np.testing.assert_allclose(np.ptp(current, axis=1), strength*np.ptp(strong, axis=1), atol=2e-15)
    for name in ['mar_x', 'mar_xz', 'observed_iv_mar']:
        np.testing.assert_array_equal(response_table(name, residual, u), response_table(name, residual+999, u))
    np.testing.assert_allclose(strong.max(axis=1), 1., atol=2e-15)
    return dict(grid_points=len(residual), rowwise_response_mean_preserved=True,
                mar_independent_of_outcome=True, strong_envelope_exactly_one=True)


def validate_tag(tag, cache):
    root = ROOT/tag
    require((root/'run.json').exists(), f'Batch incomplete or missing: {root}/run.json')
    args = json.loads((root/'run.json').read_text())['arguments']
    require(args['tag'] == tag, 'Tag differs from manifest')
    require(set(args['scenarios']).issubset(SCENARIOS), 'Unknown scenario in manifest')
    audits, cases, summaries = [], {}, []
    for dataset in args['datasets']:
        for scenario in args['scenarios']:
            audit, arrays, design, summary = validate_case(root, dataset, scenario, args, cache)
            audits.append(audit)
            cases[(dataset, scenario)] = (arrays, design)
            summaries.append(summary)
    compare_summaries(pd.read_csv(root/'summary.csv'), pd.concat(summaries, ignore_index=True))
    paired = validate_pairs(cases)
    result = dict(passed=True, tag=tag, cases=len(audits), total_trial_rows=sum(a['trial_rows'] for a in audits),
        case_audits=audits, paired_design_audits=paired, mechanism_audit=mechanism_audit(),
        combined_summary_matches=True,
        provenance_qualification='Source CSV and metadata hashes are captured at validation, since run manifests saved only executable hashes. Exact reconstruction of cohort rows, pilot estimates, instrument encoding, training response probabilities and masks checks consistency with these current sources; it is not a retrospective original-input checksum.',
        trial_assignment_qualification='Per-trial Z and split indices were not serialized. Native training Z is verified directly; source-code hashes verify that native evaluation Z is taken from the original cohort. No claim of a stored per-trial assignment audit is made.')
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
