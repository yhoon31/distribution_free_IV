
from pathlib import Path
import hashlib
import io
import json
import zipfile

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parent
OUT = BASE / 'results' / 'projected_policy_mass'


def digest(content):
    return hashlib.sha256(content).hexdigest()


def read_verified(relative):
    path = PROJECT / relative
    if path.is_file():
        content = path.read_bytes()
        return content, dict(path=relative, sha256=digest(content), storage='loose')
    for manifest_path in sorted((BASE / 'archives').glob('*_manifest.json')):
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
        member = next((m for m in manifest.get('files', []) if m['path'] == relative), None)
        if member is None:
            continue
        archive = PROJECT / manifest['archive']
        assert digest(archive.read_bytes()) == manifest['archive_sha256']
        with zipfile.ZipFile(archive) as zipped:
            content = zipped.read(relative)
        assert len(content) == member['bytes'] and digest(content) == member['sha256']
        return content, dict(path=relative, sha256=digest(content),
            storage=manifest['archive'], archive_sha256=manifest['archive_sha256'])
    raise FileNotFoundError(relative)


def project(value):
    return float(np.clip(value, 0., 1.))


def main():
    root = 'experiments/results/nhanes_five_categories/'
    validation = json.loads((PROJECT / root / 'validation.json').read_text())
    if not validation['passed']:
        raise ValueError('Five-category NHANES fits must pass validation')
    cases = [('FES', True, 'experiments/results/feedback_application/heckman')]
    cases += [('NHANES III reported height', tag == 'natural',
               root + tag + '/nhanes_height') for tag in (
                   'natural', 'natural_grid',
                   'natural_sensitivity_20260920',
                   'natural_sensitivity_20260924')]
    rows, sources = [], []
    for label, manuscript, directory in cases:
        content, source = read_verified(directory + '/design.json')
        design = json.loads(content)
        sources.append(source)
        content, source = read_verified(directory + '/summary.csv')
        summary = pd.read_csv(io.BytesIO(content))
        sources.append(source)
        take = summary[(summary.method == 'Policy IV') & np.isclose(summary.target, .9)]
        assert len(take) == 1
        take = take.iloc[0]
        raw = float(design['calibration_diagnostics']['Policy IV']['identified_mass_eif'])
        assert np.isfinite(raw)
        projected = project(raw)
        tail_raw, tail_projected = 1. - raw, 1. - projected
        assert 0. <= projected <= 1. and 0. <= tail_projected <= 1.
        np.testing.assert_allclose(tail_projected, project(tail_raw), atol=2e-16)
        assert (tail_raw >= .1) == (tail_projected >= .1)
        assert float(take.infinite_fraction) == 0.
        rows.append(dict(dataset=label, manuscript_primary=manuscript,
            seed=design['arguments']['seed'], grid_size=design['arguments']['grid_size'],
            n_calibration=design['n_calibration'],
            raw_policy_mass=raw, projected_policy_mass=projected,
            raw_limiting_miscoverage=tail_raw,
            projected_limiting_miscoverage=tail_projected,
            projection_change=projected - raw,
            target=.9, n_missing_targets=int(take.n_missing_targets),
            preserved_interval_width=float(take.mean_length),
            preserved_infinite_fraction=float(take.infinite_fraction)))

    # Check the threshold-equivalence identity at all observed tails, the
    # projection boundaries, and representative out-of-range values. The
    # accompanying analytical statement applies to every real u and 0<alpha<1.
    values = np.r_[[-2., -.01, 0., .1, .5, .9, 1., 1.01, 2.],
                   [row['raw_limiting_miscoverage'] for row in rows]]
    for alpha in [.05, .1, .2, .5, .95]:
        np.testing.assert_array_equal(values >= alpha, np.clip(values, 0., 1.) >= alpha)
    OUT.mkdir(parents=True, exist_ok=True)
    table = OUT / 'summary.csv'
    pd.DataFrame(rows).to_csv(table, index=False)
    audit = dict(passed=True, cases=len(rows), sources=sources,
        source_script_sha256=digest(Path(__file__).read_bytes()),
        summary_sha256=digest(table.read_bytes()),
        estimator='theta_projected = min(1, max(0, theta_raw)); project after averaging',
        complement_identity='1 - Pi(theta_raw) = Pi(1 - theta_raw)',
        threshold_identity='For every real u and 0 < alpha < 1, Pi(u) >= alpha iff u >= alpha',
        calibration_implementation='Original interval calibration retained; no clipping of knots before linear interpolation',
        refitting_required=False, source_results_modified=False,
        interpretation='Projected learned-policy mass is a diagnostic, not an estimate certifying Gamma or missing-outcome coverage',
        primary_results=[r for r in rows if r['manuscript_primary']])
    (OUT / 'analysis.json').write_text(json.dumps(audit, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(pd.DataFrame(rows)[['dataset', 'seed', 'grid_size', 'raw_policy_mass',
        'projected_policy_mass', 'projected_limiting_miscoverage', 'preserved_interval_width']].to_string(index=False))


if __name__ == '__main__':
    main()
