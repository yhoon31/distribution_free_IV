"""Build the JRSSB empirical artifacts or refit in a separate workspace.

Examples (run from this directory):
  python -B reproduce.py outputs
  python -B reproduce.py rerun --work-dir ../empirical-rerun --workers 2
  python -B reproduce.py rerun --work-dir ../empirical-rerun --prepare-only
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def run(script, *arguments):
    command = [sys.executable, '-B', str(ROOT / script), *map(str, arguments)]
    print('Running: ' + subprocess.list2cmdline(command), flush=True)
    environment = dict(os.environ)
    for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
        environment.setdefault(name, '1')
    subprocess.run(command, cwd=ROOT, env=environment, check=True)


def outputs(check_reference=True):
    (ROOT / 'figures').mkdir(exist_ok=True)
    run('experiments/validate_nhanes_five_categories.py')
    run('experiments/project_policy_mass.py')
    flags = ['--check-reference'] if check_reference else []
    run('render_tables.py', *flags)
    run('experiments/make_empirical_height_figure.py')
    run('experiments/validate_draft_results.py', *flags)


def prepare(target):
    target = target.expanduser().resolve()
    if target == ROOT or target.is_relative_to(ROOT):
        raise ValueError('Choose a work directory outside the reference bundle.')
    if target.exists():
        raise FileExistsError(f'Choose a new work directory; already exists: {target}')
    target.mkdir(parents=True)
    scripts = ROOT / 'experiments'

    def ignore(directory, names):
        skipped = {'__pycache__', '.pytest_cache'}
        if Path(directory) == scripts:
            skipped |= {'results', 'empirical_height_figure_verification.json'}
        return set(names) & skipped

    shutil.copytree(scripts, target / 'experiments', ignore=ignore)
    for name in ['reproduce.py', 'render_tables.py', 'requirements.txt', 'README.md']:
        shutil.copy2(ROOT / name, target / name)
    shutil.copytree(ROOT / 'templates', target / 'templates')
    fixed = Path('experiments/results/heckman/natural/design.json')
    (target / fixed).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(ROOT / fixed, target / fixed)
    (target / 'figures').mkdir()
    (target / 'rerun_workspace.json').write_text(json.dumps(dict(
        source_bundle=str(ROOT), fitting_started=False,
        note='Fresh output tree; original frozen absolute paths were not rewritten.'),
        indent=2) + '\n', encoding='utf-8')
    return target


def fit(workers):
    marker = ROOT / 'rerun_workspace.json'
    if not marker.is_file():
        raise ValueError('Use rerun --work-dir to create a fresh fit workspace first.')
    metadata = json.loads(marker.read_text())
    if metadata['fitting_started']:
        raise ValueError('This workspace has already started fitting. Use a new work directory.')
    metadata['fitting_started'] = True
    marker.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    run('experiments/run_fes_for_draft.py', '--workers', workers)
    run('experiments/run_nhanes_five_categories.py', '--workers', workers)
    run('experiments/run_nhanes_five_mar_grid.py')
    outputs(check_reference=False)
    metadata['completed'] = True
    marker.write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest='command', required=True)
    output_parser = subparsers.add_parser('outputs', help='Validate and rebuild saved-result artifacts')
    output_parser.add_argument('--allow-changed-results', action='store_true',
        help='Validate internal consistency without requiring the old draft numbers')
    rerun_parser = subparsers.add_parser('rerun', help='Copy inputs and fit into a new directory')
    rerun_parser.add_argument('--work-dir', type=Path, required=True)
    rerun_parser.add_argument('--workers', type=int, default=2)
    rerun_parser.add_argument('--prepare-only', action='store_true',
        help='Create the clean workspace and stop before fitting')
    fit_parser = subparsers.add_parser('fit', help='Fit inside a prepared clean workspace')
    fit_parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    if hasattr(args, 'workers') and args.workers < 1:
        parser.error('--workers must be positive')
    if args.command == 'outputs':
        outputs(check_reference=not args.allow_changed_results)
    elif args.command == 'fit':
        fit(args.workers)
    else:
        target = prepare(args.work_dir)
        command = [sys.executable, '-B', str(target / 'reproduce.py'), 'fit',
                   '--workers', str(args.workers)]
        print('Prepared: ' + str(target), flush=True)
        if args.prepare_only:
            print('Next: ' + subprocess.list2cmdline(command), flush=True)
        else:
            subprocess.run(command, cwd=target, check=True)


if __name__ == '__main__':
    main()
