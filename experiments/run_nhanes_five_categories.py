
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import platform

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')

import pandas as pd
import feedback_simulations as feedback
import run_observed_gamma as observed
import run_feedback_application as application
import estimate_natural_gamma as natural

BASE = Path(__file__).resolve().parent
ROOT = BASE / 'results' / 'nhanes_five_categories'
DATASET = 'nhanes_height'
NATIVE_CODES = [7005, 7016, 7020, 7024, 7067]
natural.LABELS[DATASET] = 'NHANES III reported height'


def hashes(dataset=DATASET):
    names = ['run_nhanes_five_categories.py', 'run_observed_gamma.py',
             'feedback_simulations.py', 'iv_methods.py', 'run_experiments.py',
             'run_feedback_application.py', 'estimate_natural_gamma.py']
    names += [f'data/{dataset}/{name}' for name in (
        'prepared.csv', 'prepared_metadata.json', 'prepared_natural.csv',
        'prepared_natural_metadata.json', 'source_mapping.json')]
    names += ['data/nhanes/native_design_holdout.csv',
              'data/nhanes/prepared_natural.csv',
              'data/nhanes/prepared_natural_metadata.json']
    return {name: hashlib.sha256((BASE / name).read_bytes()).hexdigest() for name in names}


def write_json(path, value):
    application.json_write(path, value)


def benchmark(family, gamma=100, seed=20260919, trials=200, grid=None, tag=None):
    tag = tag or ('primary' if seed == 20260919 else f'sensitivity_{seed}')
    grid = grid or (1601 if family == 'crossing' else 801)
    scenario = 'mar_x' if family == 'mar' else f'{family}_g{gamma:03d}'
    kind = 'mar' if family == 'mar' else 'masking'
    if kind == 'mar' and tag == 'primary':
        tag = 'mar'
    args = dict(datasets=[DATASET], trials=trials, targets=[.8, .9, .95],
                seed=seed, n_train=2000, pool_size=2000, trees=120,
                min_leaf=20, grid_size=grid, tag=tag)
    if kind == 'mar':
        # The unchanged five-arm runner joins this tag to its output Path.
        # An absolute tag selects the new result root without changing its code.
        args.update(tag=str(ROOT / tag), response_intercept=4.)
    return dict(kind=kind, family=family, tag=tag, scenario=scenario,
                dataset=DATASET, arguments=args)


def natural_job(tag='natural', seed=20260919, grid=801):
    args = dict(output=str(ROOT / tag), overwrite=False, targets=[.9],
                seed=seed, trees=120, min_leaf=20, grid_size=grid,
                max_train=2000, max_calibration=700, split_unit='household')
    return dict(kind='natural', family='natural', tag=tag, scenario='natural',
                dataset=DATASET, arguments=args)


def schedule():
    jobs = [benchmark(family, gamma) for family in ('anchored', 'crossing')
            for gamma in (90, 95, 99, 100)]
    jobs += [benchmark('mar')]
    jobs += [benchmark('crossing', seed=seed, trials=100)
             for seed in (20260920, 20260924)]
    jobs += [benchmark(family, trials=10, grid=grid, tag=tag)
             for family, grid, tag in [('anchored', 1601, 'grid'),
                                      ('crossing', 3201, 'grid'),
                                      ('mar', 1601, 'mar_grid')]]
    jobs += [natural_job(), natural_job('natural_grid', grid=1601)]
    jobs += [natural_job(f'natural_sensitivity_{seed}', seed=seed)
             for seed in (20260920, 20260924)]
    return jobs


def directory(task):
    out = ROOT / task['tag'] / task['dataset']
    return out if task['kind'] == 'natural' else out / task['scenario']


def complete(task, before):
    out = directory(task)
    if not (out / 'design.json').exists():
        return False
    design = json.loads((out / 'design.json').read_text())
    if design.get('hashes') != before or design['arguments'] != task['arguments']:
        raise ValueError(f'Refusing stale-result reuse: {out}')
    required = ['summary.csv', 'inputs_before_run.json']
    required += (['intervals.csv', 'split.csv', 'response_predictions.csv']
                 if task['kind'] == 'natural' else ['trials.csv', 'diagnostics.csv', 'split.npz'])
    return all((out / filename).is_file() for filename in required)


def execute(task):
    before = hashes()
    out = directory(task)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / 'inputs_before_run.json', before)
    args = argparse.Namespace(**task['arguments'])
    if task['kind'] == 'natural':
        application.run_dataset(DATASET, args)
    elif task['kind'] == 'mar':
        feedback.run_one(DATASET, task['scenario'], args)
    else:
        old_root, old_hashes = observed.ROOT, observed.input_hashes
        observed.ROOT, observed.input_hashes = ROOT, hashes
        try:
            observed.run_one(DATASET, task['scenario'], args)
        finally:
            observed.ROOT, observed.input_hashes = old_root, old_hashes
    if hashes() != before:
        raise RuntimeError('Inputs changed while fitting')
    design = json.loads((out / 'design.json').read_text())
    count = len(design['training_by_arm']) if task['kind'] == 'natural' else design['instrument_arms']
    if count != 5:
        raise AssertionError(f'Expected five instrument categories; got {count}')
    design.update(hashes=before, wrapper_hashes=before, instrument_arms=5,
        grouping='five', instrument_coarsened=False,
        original_instrument_retained=task['kind'] != 'mar',
        original_instrument_values_retained=task['kind'] != 'mar',
        original_response_retained=task['kind'] == 'natural',
        original_outcomes_retained=True, instrument_randomized_by_experiment=task['kind'] == 'mar',
        native_selected_interviewers=NATIVE_CODES,
        analysis_revision='Five-category NHANES height revision, 2026-10-01',
        case=dict(kind=task['kind'], tag=task['tag'], scenario=task['scenario'], grouping='five'))
    if task['kind'] == 'mar':
        design.update(randomized_instrument_arms=5, instrument_assignment_probabilities=[.2]*5)
    write_json(out / 'design.json', design)
    return dict(kind=task['kind'], tag=task['tag'], scenario=task['scenario'],
                dataset=DATASET, seed=args.seed, completed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    jobs = schedule()
    before = hashes()
    plan = dict(protocol='NHANES reported height with five original interviewer categories',
                date='2026-10-01', dataset=DATASET, results_root=str(ROOT),
                jobs=jobs, hashes=before, native_codes=NATIVE_CODES,
                mar_randomized_categories=5, python=platform.python_version(),
                historical_results_retained=True, completed=False, case_status=[])
    if args.dry_run:
        print(json.dumps(plan, indent=2))
        return
    ROOT.mkdir(parents=True, exist_ok=True)
    if (ROOT / 'plan.json').exists():
        if not args.resume:
            raise FileExistsError('Use --resume to continue the recorded plan')
        previous = json.loads((ROOT / 'plan.json').read_text())
        if previous['jobs'] != jobs or previous['hashes'] != before:
            raise ValueError('The recorded plan or input hashes changed')
    else:
        write_json(ROOT / 'plan.json', plan)
    pending = []
    for task in jobs:
        if args.resume and complete(task, before):
            plan['case_status'].append(dict(kind=task['kind'], tag=task['tag'],
                scenario=task['scenario'], dataset=DATASET, completed=True, reused=True))
        else:
            pending.append(task)
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(execute, task) for task in pending]
        for future in as_completed(futures):
            plan['case_status'].append(future.result())
            write_json(ROOT / 'run.json', plan)
            print(f'Five-category revision: {len(plan["case_status"])}/{len(jobs)} complete', flush=True)
    plan['completed'] = True
    write_json(ROOT / 'run.json', plan)
    frames = [pd.read_csv(directory(task) / 'summary.csv').assign(
        fit_kind=task['kind'], fit_tag=task['tag'], fit_seed=task['arguments']['seed'],
        instrument_arms=5) for task in jobs]
    pd.concat(frames, ignore_index=True).to_csv(ROOT / 'summary.csv', index=False)


if __name__ == '__main__':
    main()
