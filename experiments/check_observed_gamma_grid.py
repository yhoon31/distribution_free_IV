"""Audit paired grid refinement for the retained-original-Z Gamma experiment.

Compare saved fits only, preserving every finite/infinite decision. Numerical
review flags are not statistical tests or evidence of population coverage.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from run_experiments import BASE


ROOT = BASE/'results'/'observed_gamma'
KEYS = ['dataset', 'scenario', 'seed', 'trial', 'alpha', 'method']
METRICS = ['coverage', 'mean_length', 'median_length', 'finite_mean_length',
           'finite_median_length', 'infinite_fraction', 'n_test',
           'n_test_observed', 'calibration_observed']
TRIAL_ARRAYS = {'calibration_idx', 'test_idx', 'calibration_r', 'calibration_uniforms'}
FIXED_ARGUMENTS = ['targets', 'seed', 'n_train', 'pool_size', 'trees', 'min_leaf']


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def json_safe(value):
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return 'infinity' if value == np.inf else '-infinity' if value == -np.inf else None
    return value


def differences(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    both_finite = np.isfinite(a) & np.isfinite(b)
    delta = np.abs(a[both_finite]-b[both_finite])
    changed = ~np.isclose(a, b, atol=2e-12, rtol=2e-10, equal_nan=True)
    return dict(differing_rows=int(changed.sum()),
        max_absolute_finite_difference=float(delta.max()) if len(delta) else 0.,
        finite_nonfinite_disagreements=int((np.isfinite(a) != np.isfinite(b)).sum())), changed


def compare_fit(primary, diagnostic, dataset, scenario):
    pdir, ddir = primary/dataset/scenario, diagnostic/dataset/scenario
    pdsg, ddsg = [json.loads((directory/'design.json').read_text()) for directory in (pdir, ddir)]
    pargs, dargs = pdsg['arguments'], ddsg['arguments']
    trials = dargs['trials']
    require(pargs['trials'] >= trials, f'{dataset}/{scenario}: insufficient primary trials')
    require(all(pargs[key] == dargs[key] for key in FIXED_ARGUMENTS),
            f'{dataset}/{scenario}: nuisance arguments differ')
    require(pdsg['hashes'] == ddsg['hashes'], f'{dataset}/{scenario}: input hashes differ')
    require(dargs['grid_size'] == 2*pargs['grid_size']-1,
            f'{dataset}/{scenario}: diagnostic must double the integration resolution')
    require(pdsg['reference_residual_sd'] == ddsg['reference_residual_sd'],
            f'{dataset}/{scenario}: pilot scale differs')

    compared_arrays = []
    with np.load(pdir/'split.npz', allow_pickle=True) as ps, np.load(ddir/'split.npz', allow_pickle=True) as ds:
        require(set(ps.files) == set(ds.files), f'{dataset}/{scenario}: split schemas differ')
        for key in ps.files:
            # The plug-in mass is itself a numerical integration result.
            if key == 'pool_gamma_plugin':
                continue
            pa = ps[key][:trials] if key in TRIAL_ARRAYS else ps[key]
            np.testing.assert_array_equal(pa, ds[key], err_msg=f'{dataset}/{scenario}/{key}')
            compared_arrays.append(key)
        plugin_stats, _ = differences(ps['pool_gamma_plugin'], ds['pool_gamma_plugin'])

    p = pd.read_csv(pdir/'trials.csv')
    p = p[p.trial < trials].sort_values(KEYS).reset_index(drop=True)
    d = pd.read_csv(ddir/'trials.csv').sort_values(KEYS).reset_index(drop=True)
    require(not p.duplicated(KEYS).any() and not d.duplicated(KEYS).any(), 'Duplicate row keys')
    pd.testing.assert_frame_equal(p[KEYS], d[KEYS], check_dtype=False, check_exact=True)
    require(set(p.trial) == set(range(trials)), 'Incomplete comparison trials')
    require(len(p) == trials*len(dargs['targets'])*6, 'Incomplete comparison methods/targets')
    pd.testing.assert_frame_equal(p[['target', 'gamma']], d[['target', 'gamma']],
                                  check_dtype=False, check_exact=True)
    changed = np.zeros(len(p), dtype=bool)
    metrics = {}
    for metric in METRICS:
        metrics[metric], current = differences(p[metric], d[metric])
        changed |= current
    require(all(metrics[key]['differing_rows'] == 0 for key in
                ('n_test', 'n_test_observed', 'calibration_observed')),
            'Evaluation sample sizes changed')
    scale = pdsg['reference_residual_sd']
    max_width = max(metrics[key]['max_absolute_finite_difference'] for key in
                    ('mean_length', 'median_length', 'finite_mean_length', 'finite_median_length'))
    halfwidth_change = max_width/(2*scale)

    pdiag, ddiag = [pd.read_csv(directory/'diagnostics.csv').set_index(['trial', 'method'])
                   for directory in (pdir, ddir)]
    require(not pdiag.index.has_duplicates and not ddiag.index.has_duplicates,
            'Duplicate diagnostic row keys')
    pdiag = pdiag.loc[ddiag.index]
    for key in ('observed', 'min_e', 'max_inverse_e'):
        np.testing.assert_array_equal(pdiag[key], ddiag[key], err_msg=f'Diagnostic {key} differs')
    mass_stats, _ = differences(pdiag.identified_mass_eif, ddiag.identified_mass_eif)
    boundary_flips, changed_trial_rows = [], []
    infinity_changed = p.infinite_fraction.to_numpy() != d.infinite_fraction.to_numpy()
    for row in np.flatnonzero(changed):
        pr, dr = p.iloc[row], d.iloc[row]
        item = {key: pr[key] for key in KEYS}
        item.update(target=pr.target)
        for label, frame, diag in [('primary', pr, pdiag), ('diagnostic', dr, ddiag)]:
            values = {key: frame[key] for key in ['coverage', 'mean_length', 'finite_mean_length', 'infinite_fraction']}
            if (pr.trial, pr.method) in diag.index:
                values.update(diag.loc[(pr.trial, pr.method)].to_dict())
                values['tail_estimate'] = 1-values['identified_mass_eif']
                values['identified_mass_minus_target'] = values['identified_mass_eif']-pr.target
                values['tail_rule_requires_infinity'] = bool(values['tail_estimate'] >= pr.alpha-1e-12)
            item[label] = values
        changed_trial_rows.append(item)
        if infinity_changed[row]:
            boundary_flips.append(item)

    result = dict(dataset=dataset, scenario=scenario, rows=len(p), trials=trials,
        changed_rows=int(changed.sum()), infinity_changed_rows=int(infinity_changed.sum()),
        split_and_training_masks_identical=True, exact_arrays_compared=compared_arrays,
        nuisance_arguments_and_executable_hashes_identical=True, exact_row_keys_identical=True,
        metrics=metrics, max_interval_halfwidth_change_in_pilot_sd=halfwidth_change,
        material_difference_flag=bool(infinity_changed.any() or
            metrics['coverage']['max_absolute_finite_difference'] > .01+1e-12 or halfwidth_change > .01),
        pilot_residual_sd=scale, plugin_identified_mass=plugin_stats,
        diagnostic_identified_mass_eif=mass_stats, boundary_flips=boundary_flips,
        changed_trial_rows=changed_trial_rows,
        primary_trials=str((pdir/'trials.csv').relative_to(BASE)),
        diagnostic_trials=str((ddir/'trials.csv').relative_to(BASE)))
    print(f'{dataset} {scenario}: {len(p)} paired rows, {int(changed.sum())} changed, '
          f'{int(infinity_changed.sum())} infinite-set flips, material={result["material_difference_flag"]}', flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--primary-tag', default='primary')
    parser.add_argument('--diagnostic-tag', default='grid_check')
    args = parser.parse_args()
    primary, diagnostic = ROOT/args.primary_tag, ROOT/args.diagnostic_tag
    config = json.loads((diagnostic/'run.json').read_text())
    primary_config = json.loads((primary/'run.json').read_text())
    fits = [compare_fit(primary, diagnostic, dataset, scenario)
            for dataset in config['arguments']['datasets'] for scenario in config['scenarios']]
    report = dict(passed_pairing_and_provenance_checks=True,
        purpose='Doubled numerical integration grid with training data, original Z/Y, response masks, and trial randomness fixed.',
        primary_grid_points=primary_config['arguments']['grid_size'],
        diagnostic_grid_points=config['arguments']['grid_size'],
        trials_per_fit=config['arguments']['trials'], key_columns=KEYS,
        comparison_metrics=METRICS,
        qualification='A paired numerical diagnostic on the listed frozen fits and first ten trials; no population-coverage claim or uniform quadrature guarantee.',
        material_flag_rule='Any changed infinite-set decision, >0.01 absolute coverage change, or >0.01 pilot residual SD change in interval half-width on a compared row. Numerical-review thresholds, not statistical tests.',
        notes=['Infinity flags retained exactly as computed; thresholds and recorded results were not changed.',
               'Mass diagnostics include tail-rule margins for every changed infinite-set decision; finite-to-nonfinite differences are reported separately from finite width changes.'],
        fits=fits, total_compared_rows=sum(fit['rows'] for fit in fits),
        total_changed_rows=sum(fit['changed_rows'] for fit in fits),
        total_infinity_changed_rows=sum(fit['infinity_changed_rows'] for fit in fits),
        any_material_difference=any(fit['material_difference_flag'] for fit in fits))
    (diagnostic/'validation.json').write_text(json.dumps(json_safe(report), indent=2, allow_nan=False), encoding='utf-8')
    print(f'Saved {diagnostic/"validation.json"}', flush=True)


if __name__ == '__main__':
    main()
