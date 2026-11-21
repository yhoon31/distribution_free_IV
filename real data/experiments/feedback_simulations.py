
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor

from iv_methods import IVPredictor, conformal_radius, weighted_radii
from run_experiments import BASE, load_data, metric_rows, summarize

SCENARIOS = ['heterogeneous_mnar', 'weak_mnar', 'irrelevant_mnar',
             'mar_x', 'mar_xz', 'invalid_half', 'invalid_one',
             'observed_iv_mnar', 'observed_iv_mar']


def response_table(scenario, residual, covariate_index, k=5, response_intercept=.3):
    """P(R=1|X,Y,Z=z), with row-wise Y residual and X-only index.

    The first three designs share exactly the same marginal selection law
    under uniform assignment. Only arm contrasts change. The response intercept
    changes only the MAR designs; its default reproduces the historical runs.
    """
    v = np.asarray(residual, float)
    u = np.asarray(covariate_index, float)
    arm = np.linspace(-1., 1., k)
    if scenario == 'mar_x':
        return np.repeat(expit(response_intercept + 1.2*u)[:, None], k, axis=1)
    if scenario in ('mar_xz', 'observed_iv_mar'):
        return expit(response_intercept + 1.2*u[:, None] + arm[None, :])
    if scenario == 'observed_iv_mnar':
        return expit(.3 + .7*u[:, None] - .8*v[:, None]**2 + arm[None, :])
    if scenario not in SCENARIOS:
        raise ValueError(scenario)
    centers = norm.ppf((np.arange(k) + .5)/k)
    raw = .1 + .9*np.exp(-.5*((v[:, None]-centers[None, :])/.8)**2)
    strong = raw/raw.max(axis=1, keepdims=True)
    strength = {'weak_mnar': .25, 'irrelevant_mnar': 0.}.get(scenario, 1.)
    average = strong.mean(axis=1, keepdims=True)
    return average + strength*(strong-average)


def perturb_outcome(y, z, k, scale, delta):
    """Prespecified direct Z effect; zero returns unmodified real outcomes."""
    return np.asarray(y, float) + delta*scale*np.linspace(-1., 1., k)[z]


def encode_instrument(df, train, meta):
    if meta['instrument_type'] == 'continuous':
        cuts = np.quantile(df.instrument.to_numpy(float)[train], [1/3, 2/3])
        return np.searchsorted(cuts, df.instrument.to_numpy(float), side='right'), {'cutpoints': cuts.tolist()}
    levels = sorted(df.instrument.astype(str).unique())
    return pd.Categorical(df.instrument.astype(str), categories=levels).codes, {'levels': levels}


def run_one(dataset, scenario, args):
    started = time.time()
    response_intercept = getattr(args, 'response_intercept', .3)
    native = scenario.startswith('observed_iv')
    df, x, meta, _ = load_data(dataset, 'natural' if native else None)
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
    pilot = order[:npilot]
    remaining = order[npilot:]
    ntrain = min(args.n_train, int(.45*len(remaining)))
    train = remaining[:ntrain]
    pool = remaining[ntrain:ntrain+args.pool_size]
    if len(pool) < 100:
        raise ValueError('Insufficient disjoint evaluation records')
    y = df.outcome.to_numpy(float)
    reference = RandomForestRegressor(n_estimators=100, min_samples_leaf=30,
        random_state=args.seed, n_jobs=1, oob_score=True).fit(x[pilot], y[pilot])
    ref_scale = max(float(np.std(y[pilot]-reference.oob_prediction_)), 1e-6)
    ref = reference.predict(x)
    umean, usd = float(ref[pilot].mean()), max(float(ref[pilot].std()), 1e-6)
    u = np.clip((ref-umean)/usd, -3, 3)
    native_info = {}
    if native:
        allz, native_info = encode_instrument(df, train, meta)
        k = int(allz.max())+1
    else:
        k = 5
        allz = rng.integers(0, k, len(df))
    delta = {'invalid_half': .5, 'invalid_one': 1.}.get(scenario, 0.)
    yt = perturb_outcome(y[train], allz[train], k, ref_scale, delta)
    vtrain = (yt-ref[train])/ref_scale
    pt = response_table(scenario, vtrain, u[train], k, response_intercept)
    uniforms_train = rng.random(len(train))
    rt = uniforms_train < pt[np.arange(len(train)), allz[train]]
    zt = allz[train]
    leaf = min(args.min_leaf, max(5, len(train)//100))
    model = IVPredictor(seed=args.seed, n_trees=args.trees, min_leaf=leaf,
        grid_size=args.grid_size).fit(x[train], zt, rt, np.where(rt, yt, np.nan))
    prep = model.prepare(x[pool])
    # All baseline comparisons share a score within each predictor specification.
    onehot = np.eye(k)[allz]
    xtxz = np.c_[x[train], onehot[train]]
    score_xz = RandomForestRegressor(n_estimators=args.trees, min_samples_leaf=leaf,
        random_state=args.seed+701, n_jobs=1).fit(xtxz[rt], yt[rt])
    response_xz = RandomForestRegressor(n_estimators=args.trees, min_samples_leaf=leaf,
        random_state=args.seed+702, n_jobs=1).fit(xtxz, rt.astype(float))
    # Cache predictions for every possible randomized test arm, avoiding per-trial RF calls.
    xpool = x[pool]
    xz_centers = np.column_stack([score_xz.predict(np.c_[xpool, np.tile(np.eye(k)[j], (len(pool), 1))]) for j in range(k)])
    xz_response = np.clip(np.column_stack([response_xz.predict(np.c_[xpool, np.tile(np.eye(k)[j], (len(pool), 1))]) for j in range(k)]), .02, .98)
    rows, diagnostics = [], []
    alphas = 1-np.asarray(args.targets)
    nc = nt = min(500, len(pool)//2)
    trial_rng = np.random.default_rng(args.seed+19937)
    for trial in range(args.trials):
        perm = trial_rng.permutation(len(pool))
        ci, ti = perm[:nc], perm[nc:nc+nt]
        zpool = allz[pool] if native else trial_rng.integers(0, k, len(pool))
        yp = perturb_outcome(y[pool], zpool, k, ref_scale, delta)
        p = response_table(scenario, (yp-ref[pool])/ref_scale, u[pool], k, response_intercept)
        pactual = p[np.arange(len(pool)), zpool]
        uniforms_calibration = trial_rng.random(nc)
        rc = uniforms_calibration < pactual[ci]
        trial_stream_hash = hashlib.sha256(b''.join(
            a.tobytes() for a in (ci, ti, zpool, uniforms_calibration))).hexdigest()
        yc = np.where(rc, yp[ci], np.nan)
        cal, diag, scores = model.calibrate(prep, ci, zpool[ci], rc, yc, xpool[ci], alphas)
        center = model.location + model.scale*prep.center[ti]
        def record(name, radius, center_value, scale_value, a):
            d = metric_rows(dataset, scenario, trial, a, name, radius, center_value,
                yp[ti], np.ones(nt, bool), scale_value, True, np.nan)
            d.update(seed=args.seed, calibration_observed=int(rc.sum()))
            rows.append(d)
        for method, radii in cal.items():
            for a, radius in zip(alphas, radii):
                record(method, radius, center, model.scale, a)
        if not native:
            fitted_e = prep.e
            prep.e = np.full_like(fitted_e, 1/k)
            known, _, _ = model.calibrate(prep, ci, zpool[ci], rc, yc, xpool[ci], alphas)
            prep.e = fitted_e
            for method in ('Softmax IV', 'Policy IV'):
                for a, radius in zip(alphas, known[method]):
                    record(method+' (known e)', radius, center, model.scale, a)
        centers_z = xz_centers[np.arange(len(pool)), zpool]
        response_z = xz_response[np.arange(len(pool)), zpool]
        scores_z = np.abs(yp[ci][rc]-centers_z[ci][rc])
        for a in alphas:
            record('Weighted CP (X)', weighted_radii(scores[rc], 1/prep.response[ci][rc], 1/prep.response[ti], a), center, model.scale, a)
            record('Naive CP (X,Z)', conformal_radius(scores_z, a), centers_z[ti], 1., a)
            record('Weighted CP (X,Z)', weighted_radii(scores_z, 1/response_z[ci][rc], 1/response_z[ti], a), centers_z[ti], 1., a)
            if scenario in ('mar_x', 'mar_xz', 'observed_iv_mar'):
                record('MAR CP (known p; X,Z)', weighted_radii(scores_z, 1/pactual[ci][rc], 1/pactual[ti], a), centers_z[ti], 1., a)
                if not native:
                    marginal_p = p.mean(axis=1)
                    record('MAR CP (known p; X)', weighted_radii(scores[rc], 1/marginal_p[ci][rc], 1/marginal_p[ti], a), center, model.scale, a)
        for method in ('Softmax IV', 'Policy IV'):
            diagnostics.append(dict(trial=trial, method=method,
                randomization_hash=trial_stream_hash, **diag[method]))
    # This index is an empirical-cohort functional, not a population confidence bound.
    pdiag = response_table(scenario, (y[pool]-ref[pool])/ref_scale, u[pool], k, response_intercept)
    design = dict(dataset=dataset, scenario=scenario, seed=args.seed,
        arguments=vars(args), original_cohort_rows=original_n, original_missing_rows=original_missing,
        eligible_complete_case_households=len(df), n_pilot=len(pilot), n_train=len(train),
        n_pool=len(pool), n_calibration=nc, n_test=nt, n_train_observed=int(rt.sum()),
        instrument_arms=k, instrument=native_info, original_instrument_retained=native,
        exclusion_status='unverified after complete-case selection' if native else ('violated by direct outcome shift' if delta else 'valid by randomized assignment'),
        direct_effect_delta=delta, reference_residual_sd=ref_scale,
        covariate_index_mean=umean, covariate_index_sd=usd,
        exact_gamma=1. if scenario=='heterogeneous_mnar' else None,
        empirical_gamma=float(pdiag.max(axis=1).mean()) if not native and not delta else None,
        mean_response_probability=float(pdiag.mean()) if not native and not delta else None,
        mar_response_intercept=response_intercept if scenario in ('mar_x', 'mar_xz', 'observed_iv_mar') else None,
        mean_assigned_response_probability=float(pdiag[np.arange(len(pool)), allz[pool]].mean()) if native else float(pdiag.mean()),
        mean_max_response_probability=float(pdiag.max(axis=1).mean()),
        max_response_identification_note='Equals the empirical-cohort identification index under conditional IV validity; validity is unverified for recorded instruments.',
        mean_conditional_arm_range=float(np.ptp(pdiag, axis=1).mean()),
        covariate_shift_index_population_mean=float(u[pool].mean()),
        covariate_shift_index_respondent_mean=float(np.average(u[pool], weights=pdiag.mean(axis=1))) if not native and not delta else None,
        training_by_arm=[dict(arm=j, total=int((zt==j).sum()), observed=int(((zt==j)&rt).sum())) for j in range(k)],
        nuisance_min_leaf=leaf, gamma_plugin_mean=float(prep.gamma_plugin.mean()),
        gamma_plugin_above_one=float((prep.gamma_plugin>1).mean()),
        elapsed_seconds=time.time()-started,
        target='Originally observed cohort, sampled one record per household; full-population of this artificial masking experiment',
        hashes={name:hashlib.sha256((BASE/name).read_bytes()).hexdigest() for name in ['feedback_simulations.py', 'iv_methods.py']})
    out = BASE/'results'/'feedback'/args.tag/dataset/scenario
    out.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(out/'trials.csv', index=False)
    summarize(frame).to_csv(out/'summary.csv', index=False)
    pd.DataFrame(diagnostics).to_csv(out/'diagnostics.csv', index=False)
    (out/'design.json').write_text(json.dumps(design, indent=2), encoding='utf-8')
    np.savez_compressed(out/'split.npz', pilot_rows=df.row_id.to_numpy()[pilot],
        train_rows=df.row_id.to_numpy()[train], pool_rows=df.row_id.to_numpy()[pool],
        train_z=zt, train_r=rt, train_probability=pt[np.arange(len(train)), zt],
        train_uniform=uniforms_train, train_covariate_index=u[train],
        pool_z=allz[pool], pool_covariate_index=u[pool])
    print(f'{dataset} {scenario}: {len(frame)} rows, {time.time()-started:.1f}s', flush=True)
    return str(out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', default=['heckman', 'nhanes'])
    parser.add_argument('--scenarios', nargs='+', choices=SCENARIOS, default=SCENARIOS)
    parser.add_argument('--trials', type=int, default=200)
    parser.add_argument('--targets', nargs='+', type=float, default=[.8, .9, .95])
    parser.add_argument('--seed', type=int, default=20260919)
    parser.add_argument('--n-train', type=int, default=2000)
    parser.add_argument('--pool-size', type=int, default=2000)
    parser.add_argument('--trees', type=int, default=120)
    parser.add_argument('--min-leaf', type=int, default=20)
    parser.add_argument('--grid-size', type=int, default=801)
    parser.add_argument('--response-intercept', type=float, default=.3,
        help='Logit intercept in MAR response laws (historical default: 0.3).')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--tag', default='primary')
    parser.add_argument('--reuse', action='store_true')
    args = parser.parse_args()
    if not all(0 < t < 1 for t in args.targets):
        parser.error('Targets must lie strictly between zero and one')
    if not np.isfinite(args.response_intercept):
        parser.error('Response intercept must be finite')
    jobs = []
    for ds in args.datasets:
        for scenario in args.scenarios:
            out = BASE/'results'/'feedback'/args.tag/ds/scenario
            if args.reuse and (out/'design.json').exists() and (out/'trials.csv').exists():
                design = json.loads((out/'design.json').read_text())
                keys = ['trials','targets','seed','n_train','pool_size','trees','min_leaf','grid_size','response_intercept']
                if all(design['arguments'].get(key, .3 if key=='response_intercept' else None) == vars(args)[key] for key in keys) and all(hashlib.sha256((BASE/name).read_bytes()).hexdigest()==digest for name,digest in design['hashes'].items()):
                    print(f'Reusing {ds} {scenario}', flush=True)
                    continue
                raise ValueError(f'Refusing stale reuse: {out}')
            jobs.append((ds, scenario))
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(run_one, ds, scenario, args): (ds, scenario) for ds, scenario in jobs}
        for f in as_completed(futures):
            f.result()
    root = BASE/'results'/'feedback'/args.tag
    summaries = [pd.read_csv(root/ds/scenario/'summary.csv') for ds in args.datasets for scenario in args.scenarios]
    pd.concat(summaries, ignore_index=True).to_csv(root/'summary.csv', index=False)
    (root/'run.json').write_text(json.dumps(dict(arguments=vars(args), python=platform.python_version()), indent=2))


if __name__ == '__main__':
    main()
