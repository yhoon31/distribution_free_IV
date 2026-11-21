
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
import platform
import time

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.ensemble import RandomForestRegressor

from feedback_simulations import encode_instrument, response_table
from iv_methods import IVPredictor, conformal_radius, weighted_radii
from run_experiments import BASE, load_data, metric_rows, summarize

ROOT = BASE/'results'/'observed_gamma'
DEPENDENCIES = ['run_observed_gamma.py', 'feedback_simulations.py',
                'iv_methods.py', 'run_experiments.py']
FIT_ARGUMENTS = ['trials', 'targets', 'seed', 'n_train', 'pool_size',
                 'trees', 'min_leaf', 'grid_size']


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_scenario(scenario):
    if scenario == 'original_logistic':
        return 'original_logistic', None
    design, encoded = scenario.rsplit('_g', 1)
    if design not in ('anchored', 'crossing') or not encoded.isdigit():
        raise ValueError(f'Unknown response design: {scenario}')
    gamma = int(encoded)/100
    if not 0 < gamma <= 1:
        raise ValueError('Response envelope must lie in (0,1]')
    return design, gamma


def response_probabilities(design, residual, covariate_index, k, gamma=None, anchor=0):
    """P(R=1|X,Y,Z=z); only this table changes between paired scenarios.

    Anchored and crossing have pointwise maximum gamma. This implies the
    IV Gamma equals gamma under exclusion. Without exclusion, anchored
    retains Gamma >= gamma because its anchor subdensity integrates to
    gamma; crossing has no such unconditional guarantee.
    """
    v = np.asarray(residual, float)
    u = np.asarray(covariate_index, float)
    if v.ndim != 1 or u.shape != v.shape or not np.isfinite(v).all() or not np.isfinite(u).all():
        raise ValueError('Residuals and covariate indices must be finite matching vectors')
    if k < 2 or not 0 <= anchor < k:
        raise ValueError('Need at least two instrument arms and a valid anchor')
    if design == 'original_logistic':
        return response_table('observed_iv_mnar', v, u, k)
    if gamma is None or not np.isfinite(gamma) or not 0 < gamma <= 1:
        raise ValueError('Response envelope must lie in (0,1]')
    if design == 'crossing':
        return gamma*response_table('heterogeneous_mnar', v, u, k)
    if design != 'anchored':
        raise ValueError(f'Unknown response design: {design}')
    other = np.flatnonzero(np.arange(k) != anchor)
    p = np.full((len(v), k), gamma, dtype=float)
    c = np.linspace(.4, 1., k-1)
    p[:, other] = gamma*(.10+.65*c[None, :]*np.exp(-.75*v[:, None]**2))
    return p


def input_hashes(dataset):
    stem = 'prepared_natural' if (BASE/'data'/dataset/'prepared_natural.csv').exists() else 'prepared'
    files = DEPENDENCIES + [f'data/{dataset}/{stem}.csv', f'data/{dataset}/{stem}_metadata.json']
    return {name: sha256(BASE/name) for name in files}


def run_one(dataset, scenario, args):
    started = time.time()
    hashes = input_hashes(dataset)  # Capture provenance before reading/fitting.
    design_name, gamma = parse_scenario(scenario)
    out = ROOT/args.tag/dataset/scenario
    out.mkdir(parents=True, exist_ok=True)
    (out/'inputs_before_run.json').write_text(json.dumps(hashes, indent=2), encoding='utf-8')
    df, x, meta, _ = load_data(dataset, 'natural')
    original_n, original_missing = len(df), int((~df.observed.astype(bool)).sum())
    keep = df.observed.to_numpy(bool)
    df, x = df.loc[keep].reset_index(drop=True), x[keep]
    rng = np.random.default_rng(args.seed)
    shuffled = rng.permutation(len(df))
    _, first = np.unique(df.household_id.to_numpy()[shuffled], return_index=True)
    keep = shuffled[np.sort(first)]
    df, x = df.iloc[keep].reset_index(drop=True), x[keep]
    order = rng.permutation(len(df))
    npilot = min(1000, max(50, len(df)//10))
    pilot, remaining = order[:npilot], order[npilot:]
    ntrain = min(args.n_train, int(.45*len(remaining)))
    train, pool = remaining[:ntrain], remaining[ntrain:ntrain+args.pool_size]
    if len(pool) < 100:
        raise ValueError('Insufficient disjoint evaluation records')
    y = df.outcome.to_numpy(float)
    reference = RandomForestRegressor(n_estimators=100, min_samples_leaf=30,
        random_state=args.seed, n_jobs=1, oob_score=True).fit(x[pilot], y[pilot])
    ref_scale = max(float(np.std(y[pilot]-reference.oob_prediction_)), 1e-6)
    ref = reference.predict(x)
    umean, usd = float(ref[pilot].mean()), max(float(ref[pilot].std()), 1e-6)
    u = np.clip((ref-umean)/usd, -3, 3)
    allz, native_info = encode_instrument(df, train, meta)
    k = int(allz.max())+1
    zt = allz[train]
    anchor = int(np.bincount(zt, minlength=k).argmax())
    pt = response_probabilities(design_name, (y[train]-ref[train])/ref_scale,
                                u[train], k, gamma, anchor)
    uniforms_train = rng.random(len(train))
    pt_actual = pt[np.arange(len(train)), zt]
    rt = uniforms_train < pt_actual
    leaf = min(args.min_leaf, max(5, len(train)//100))
    model = IVPredictor(seed=args.seed, n_trees=args.trees, min_leaf=leaf,
        grid_size=args.grid_size).fit(x[train], zt, rt, np.where(rt, y[train], np.nan))
    xpool, yp, zp = x[pool], y[pool], allz[pool]
    prep = model.prepare(xpool)
    onehot = np.eye(k)[allz]
    xtxz = np.c_[x[train], onehot[train]]
    score_xz = RandomForestRegressor(n_estimators=args.trees, min_samples_leaf=leaf,
        random_state=args.seed+701, n_jobs=1).fit(xtxz[rt], y[train][rt])
    response_xz = RandomForestRegressor(n_estimators=args.trees, min_samples_leaf=leaf,
        random_state=args.seed+702, n_jobs=1).fit(xtxz, rt.astype(float))
    # Match the old runner's arm-wise prediction calls exactly for its control.
    xz_centers = np.column_stack([score_xz.predict(np.c_[xpool, np.tile(np.eye(k)[j], (len(pool), 1))]) for j in range(k)])
    xz_response = np.clip(np.column_stack([response_xz.predict(np.c_[xpool, np.tile(np.eye(k)[j], (len(pool), 1))]) for j in range(k)]), .02, .98)
    centers_z = xz_centers[np.arange(len(pool)), zp]
    response_z = xz_response[np.arange(len(pool)), zp]
    pp = response_probabilities(design_name, (yp-ref[pool])/ref_scale, u[pool], k, gamma, anchor)
    pactual = pp[np.arange(len(pool)), zp]
    rows, diagnostics = [], []
    calibration_idx, test_idx, calibration_r, calibration_uniforms = [], [], [], []
    alphas = 1-np.asarray(args.targets)
    nc = nt = min(500, len(pool)//2)
    trial_rng = np.random.default_rng(args.seed+19937)
    for trial in range(args.trials):
        perm = trial_rng.permutation(len(pool))
        ci, ti = perm[:nc], perm[nc:nc+nt]
        uniforms = trial_rng.random(nc)
        rc = uniforms < pactual[ci]
        calibration_idx.append(ci)
        test_idx.append(ti)
        calibration_r.append(rc)
        calibration_uniforms.append(uniforms)
        yc = np.where(rc, yp[ci], np.nan)
        cal, diag, scores = model.calibrate(prep, ci, zp[ci], rc, yc, xpool[ci], alphas)
        center = model.location + model.scale*prep.center[ti]

        def record(name, radius, center_value, scale_value, a):
            row = metric_rows(dataset, scenario, trial, a, name, radius, center_value,
                yp[ti], np.ones(nt, bool), scale_value, True, np.nan if gamma is None else gamma)
            row.update(seed=args.seed, calibration_observed=int(rc.sum()))
            rows.append(row)

        for method, radii in cal.items():
            for a, radius in zip(alphas, radii):
                record(method, radius, center, model.scale, a)
        scores_z = np.abs(yp[ci][rc]-centers_z[ci][rc])
        for a in alphas:
            record('Weighted CP (X)', weighted_radii(scores[rc], 1/prep.response[ci][rc], 1/prep.response[ti], a), center, model.scale, a)
            record('Naive CP (X,Z)', conformal_radius(scores_z, a), centers_z[ti], 1., a)
            record('Weighted CP (X,Z)', weighted_radii(scores_z, 1/response_z[ci][rc], 1/response_z[ti], a), centers_z[ti], 1., a)
        for method in ('Softmax IV', 'Policy IV'):
            diagnostics.append(dict(trial=trial, method=method, **diag[method], **diag['calibration']))
    if input_hashes(dataset) != hashes:
        raise RuntimeError('An input changed during this experiment; outputs not finalized')
    design = dict(dataset=dataset, scenario=scenario, response_design=design_name,
        response_envelope=gamma, gamma_under_exclusion=gamma,
        gamma_lower_bound_without_exclusion=gamma if design_name=='anchored' else None,
        anchor_arm=anchor, anchor=anchor,
        anchor_training_fraction=float((zt==anchor).mean()),
        anchor_pool_fraction=float((zp==anchor).mean()),
        anchor_selection='most frequent encoded training arm before masking; lowest label breaks ties',
        seed=args.seed, arguments=vars(args), original_cohort_rows=original_n,
        original_missing_rows=original_missing, eligible_complete_case_households=len(df),
        n_pilot=len(pilot), n_train=len(train), n_pool=len(pool), n_calibration=nc, n_test=nt,
        n_train_observed=int(rt.sum()), instrument_arms=k, instrument=native_info,
        original_instrument_retained=True, original_outcomes_retained=True,
        exclusion_status='unverified after complete-case selection',
        reference_residual_sd=ref_scale, covariate_index_mean=umean, covariate_index_sd=usd,
        mean_response_probability=float(pactual.mean()),
        mean_response_probability_actual_z=float(pactual.mean()),
        empirical_mean_max_response_probability=float(pp.max(axis=1).mean()),
        mean_conditional_arm_range=float(np.ptp(pp, axis=1).mean()),
        training_by_arm=[dict(arm=j, total=int((zt==j).sum()), observed=int(((zt==j)&rt).sum())) for j in range(k)],
        nuisance_min_leaf=leaf, gamma_plugin_mean=float(prep.gamma_plugin.mean()),
        gamma_plugin_above_one=float((prep.gamma_plugin>1).mean()),
        identified_mass_eif_mean={method: float(np.mean([d['identified_mass_eif'] for d in diagnostics if d['method']==method])) for method in ('Softmax IV', 'Policy IV')},
        elapsed_seconds=time.time()-started,
        target='Originally observed cohort, sampled one record per household; full population of this artificial masking experiment',
        propensity='Estimated from original Z and X; no randomized/known-e substitution',
        software=dict(python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__,
                      scipy=scipy.__version__, sklearn=sklearn.__version__),
        hashes=hashes)
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'trials.csv', index=False)
    summarize(frame).to_csv(out/'summary.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(out/'diagnostics.csv', index=False)
    np.savez_compressed(out/'split.npz',
        pilot_rows=df.row_id.to_numpy()[pilot], train_rows=df.row_id.to_numpy()[train], pool_rows=df.row_id.to_numpy()[pool],
        pilot_z=allz[pilot], train_z=zt, pool_z=zp,
        pilot_original_z=df.instrument.to_numpy()[pilot], train_original_z=df.instrument.to_numpy()[train], pool_original_z=df.instrument.to_numpy()[pool],
        pilot_y=y[pilot], train_y=y[train], pool_y=yp,
        train_r=rt, train_uniforms=uniforms_train, train_probability=pt_actual, train_probability_table=pt,
        pool_probability=pactual, pool_probability_table=pp, pool_e=prep.e,
        pool_gamma_plugin=prep.gamma_plugin,
        calibration_idx=np.asarray(calibration_idx), test_idx=np.asarray(test_idx),
        calibration_r=np.asarray(calibration_r), calibration_uniforms=np.asarray(calibration_uniforms))
    (out/'design.json').write_text(json.dumps(design, indent=2), encoding='utf-8')
    print(f'{dataset} {scenario}: {len(frame)} rows, {time.time()-started:.1f}s', flush=True)
    return str(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', default=['heckman', 'nhanes'])
    parser.add_argument('--designs', nargs='+', choices=['anchored', 'crossing'], default=['anchored', 'crossing'])
    parser.add_argument('--gammas', nargs='+', type=float, default=[.9, .95, .99, 1.])
    parser.add_argument('--trials', type=int, default=200)
    parser.add_argument('--targets', nargs='+', type=float, default=[.8, .9, .95])
    parser.add_argument('--seed', type=int, default=20260919)
    parser.add_argument('--n-train', type=int, default=2000)
    parser.add_argument('--pool-size', type=int, default=2000)
    parser.add_argument('--trees', type=int, default=120)
    parser.add_argument('--min-leaf', type=int, default=20)
    parser.add_argument('--grid-size', type=int, default=801)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--tag', default='primary')
    parser.add_argument('--reuse', action='store_true')
    args = parser.parse_args()
    if not all(0 < t < 1 for t in args.targets) or args.trials < 1:
        parser.error('Need positive trials and coverage targets strictly between zero and one')
    if not all(np.isfinite(g) and 0 < g <= 1 and np.isclose(g*100, round(g*100)) for g in args.gammas):
        parser.error('Gammas must be percentage values in (0,1]')
    scenarios = ['original_logistic'] + [f'{d}_g{round(g*100):03d}' for d in args.designs for g in args.gammas]
    jobs = []
    for dataset in args.datasets:
        for scenario in scenarios:
            out = ROOT/args.tag/dataset/scenario
            if args.reuse and (out/'design.json').exists():
                saved = json.loads((out/'design.json').read_text())
                complete = all((out/name).is_file() for name in ['trials.csv', 'summary.csv', 'diagnostics.csv', 'split.npz', 'inputs_before_run.json'])
                if complete and all(saved['arguments'][key] == vars(args)[key] for key in FIT_ARGUMENTS) and saved['hashes'] == input_hashes(dataset):
                    print(f'Reusing {dataset} {scenario}', flush=True)
                    continue
                raise ValueError(f'Refusing stale/incomplete reuse: {out}')
            jobs.append((dataset, scenario))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [executor.submit(run_one, dataset, scenario, args) for dataset, scenario in jobs]
        for future in as_completed(futures):
            future.result()
    root = ROOT/args.tag
    frames = [pd.read_csv(root/dataset/scenario/'summary.csv') for dataset in args.datasets for scenario in scenarios]
    pd.concat(frames, ignore_index=True).to_csv(root/'summary.csv', index=False)
    (root/'run.json').write_text(json.dumps(dict(arguments=vars(args), scenarios=scenarios,
        python=platform.python_version()), indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
