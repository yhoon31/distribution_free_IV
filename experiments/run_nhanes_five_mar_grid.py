
from __future__ import annotations

import argparse
import json

import run_nhanes_five_categories as protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    task = protocol.benchmark('mar', trials=10, grid=3201, tag='mar_grid_refined')
    plan = dict(purpose='Resolve known-propensity MAR comparator integration-grid sensitivity; retain original primary protocol',
                jobs=[task], hashes=protocol.hashes(), completed=False,
                comparison_tags=['mar_grid', 'mar_grid_refined'])
    plan_path = protocol.ROOT / 'numerical_refinement_plan.json'
    if plan_path.exists():
        saved = json.loads(plan_path.read_text())
        if saved['jobs'] != plan['jobs'] or saved['hashes'] != plan['hashes']:
            raise ValueError('Recorded supplemental plan or input hashes changed')
    else:
        protocol.ROOT.mkdir(parents=True, exist_ok=True)
        protocol.write_json(plan_path, plan)
    if protocol.complete(task, plan['hashes']):
        result = dict(kind=task['kind'], tag=task['tag'], scenario=task['scenario'],
                      dataset=task['dataset'], seed=task['arguments']['seed'],
                      completed=True, reused=True)
    else:
        result = protocol.execute(task)
    plan.update(completed=True, case_status=[result])
    protocol.write_json(protocol.ROOT / 'numerical_refinement_run.json', plan)
    print('Supplemental MAR grid verified and complete.')


if __name__ == '__main__':
    main()
