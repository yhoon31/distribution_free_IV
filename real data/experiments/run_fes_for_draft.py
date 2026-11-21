
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path

for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ.setdefault(name, '1')

BASE = Path(__file__).resolve().parent


def schedule():
    common = dict(datasets=['heckman'], trials=200, targets=[.8, .9, .95],
                  seed=20260919, n_train=2000, pool_size=2000, trees=120,
                  min_leaf=20)
    jobs = []
    for family, tag, grid in [('anchored', 'primary', 801),
                              ('crossing', 'crossing_refined', 1601)]:
        for gamma in [90, 95, 99, 100]:
            scenario = f'{family}_g{gamma:03d}'
            jobs.append(dict(kind='masking', scenario=scenario,
                             arguments=dict(common, tag=tag, grid_size=grid)))
    jobs.append(dict(kind='mar', scenario='mar_x', arguments=dict(
        common, tag='mar_x_high_response', grid_size=801, response_intercept=4.)))
    jobs.append(dict(kind='natural', scenario='natural', arguments=dict(
        datasets=['heckman'], targets=[.7, .8, .9, .95], seed=20260919,
        trees=120, min_leaf=20, grid_size=801, max_train=2000,
        max_calibration=700, split_unit='household', overwrite=False,
        output=str(BASE / 'results' / 'feedback_application'))))
    return jobs


def directory(task):
    args = task['arguments']
    if task['kind'] == 'natural':
        return Path(args['output']) / 'heckman'
    family = 'observed_gamma' if task['kind'] == 'masking' else 'feedback'
    return BASE / 'results' / family / args['tag'] / 'heckman' / task['scenario']


def execute(task):
    args = argparse.Namespace(**task['arguments'])
    if task['kind'] == 'masking':
        import run_observed_gamma
        run_observed_gamma.run_one('heckman', task['scenario'], args)
    elif task['kind'] == 'mar':
        import feedback_simulations
        feedback_simulations.run_one('heckman', task['scenario'], args)
    else:
        import run_feedback_application
        run_feedback_application.run_dataset('heckman', args)
    return dict(kind=task['kind'], scenario=task['scenario'], completed=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    if args.workers < 1:
        parser.error('--workers must be positive')
    jobs = schedule()
    if args.dry_run:
        print(json.dumps(jobs, indent=2))
        return
    for task in jobs:
        if directory(task).exists():
            raise FileExistsError(f'Use a fresh rerun workspace: {directory(task)}')
    plan = dict(jobs=jobs, completed=False)
    root = BASE / 'results'
    root.mkdir(parents=True, exist_ok=True)
    (root / 'fes_draft_plan.json').write_text(json.dumps(plan, indent=2) + '\n')
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        completed = list(executor.map(execute, jobs))
    plan.update(completed=True, case_status=completed)
    (root / 'fes_draft_run.json').write_text(json.dumps(plan, indent=2) + '\n')
    print('Completed 8 FES masking fits, 1 MAR fit, and 1 natural application.')


if __name__ == '__main__':
    main()
