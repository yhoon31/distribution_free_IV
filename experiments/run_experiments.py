
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
import platform
import re
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from iv_methods import IVPredictor, weighted_radii

BASE = Path(__file__).resolve().parent
ALPHAS = np.arange(.05, .301, .05)


def log(s):
    print(time.strftime('%H:%M:%S'), s, flush=True)


def load_data(dataset, scenario=None):
    stem = 'prepared'
    if scenario == 'natural' and (BASE/'data'/dataset/'prepared_natural.csv').exists():
        stem = 'prepared_natural'
    path = BASE/'data'/dataset/(stem+'.csv')
    if not path.exists():
        raise FileNotFoundError(f"Missing authorized data: {path}")
    df = pd.read_csv(path)
    required = {'row_id', 'household_id', 'outcome', 'observed', 'instrument'}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f'{dataset}: missing required prepared columns {sorted(missing)}')
    if not df.row_id.is_unique or df.row_id.isna().any() or df.household_id.isna().any():
        raise ValueError('Prepared row IDs must be unique and grouping IDs nonmissing')
    if not df.observed.isin([0, 1]).all() or not np.array_equal(df.outcome.notna(), df.observed.astype(bool)):
        raise ValueError('Observed flags must exactly match known prepared outcomes')
    meta = json.loads((path.parent/(stem+'_metadata.json')).read_text())
    numeric, categorical = meta['numeric_features'], meta['categorical_features']
    if df[numeric+categorical].isna().any().any():
        raise ValueError('Missing covariates must be explicitly handled before encoding')
    xdf = pd.get_dummies(df[numeric+categorical], columns=categorical, dtype=float)
    if xdf.isna().any().any():
        raise ValueError('Prepare or explicitly encode missing covariates first')
    meta = dict(meta, encoded_features=list(xdf.columns))
    return df, xdf.to_numpy(float), meta, list(xdf.columns)


def split_pools(df, seed, n_train=2000, max_pool=2000):
    """Disjoint households for training and evaluation reservoir."""
    rng = np.random.default_rng(seed)
    groups = df.household_id.astype(str).to_numpy()
    unique = np.unique(groups)
    order = rng.permutation(unique)
    # Target rows approximately; group integrity is never sacrificed.
    lengths = pd.Series(groups).value_counts()
    n_group_train = np.searchsorted(np.cumsum(lengths[order].to_numpy()),
                                   min(n_train, int(.45*len(df)))) + 1
    train = np.flatnonzero(np.isin(groups, order[:n_group_train]))
    pool = np.flatnonzero(~np.isin(groups, order[:n_group_train]))
    # At most one record per household in evaluation avoids cluster reuse.
    pool = rng.permutation(pool)
    _, first = np.unique(groups[pool], return_index=True)
    pool = pool[np.sort(first)][:max_pool]
    return train, pool


def take_group_prefix(df, column, target, rng):
    """Choose whole randomly ordered groups until reaching a target row count."""
    groups = rng.permutation(df[column].unique())
    counts = df[column].value_counts()
    n = int(np.searchsorted(np.cumsum(counts[groups].to_numpy()), target)) + 1
    selected = groups[:n]
    return df[column].isin(selected).to_numpy(), selected


def cluster_split_pools(df, seed, n_train=2000, max_pool=2000):
    """Sensitivity: disjoint public survey pseudo-PSUs for training/reservoir."""
    if 'cluster_id' not in df:
        raise ValueError('PSU splitting requires a documented cluster_id column')
    rng = np.random.default_rng(seed)
    selected, _ = take_group_prefix(df, 'cluster_id', min(n_train, int(.45*len(df))), rng)
    train = rng.permutation(np.flatnonzero(selected))[:n_train]
    pool = rng.permutation(np.flatnonzero(~selected))[:max_pool]
    if len(np.unique(df.cluster_id.to_numpy()[pool])) < 4:
        raise ValueError('Insufficient evaluation pseudo-PSUs for clustered sensitivity')
    return train, pool


def trial_indices(pool_frame, rng, nc, nt, split_unit):
    if split_unit == 'family':
        order = rng.permutation(len(pool_frame))
        return order[:nc], order[nc:nc+nt]
    # Trial calibration/test records never share a pseudo-PSU. Extra records
    # in selected clusters are not reassigned to the other part of the trial.
    for _ in range(100):
        selected, _ = take_group_prefix(pool_frame, 'cluster_id', nc, rng)
        if (~selected).sum() >= nt:
            return (rng.permutation(np.flatnonzero(selected))[:nc],
                    rng.permutation(np.flatnonzero(~selected))[:nt])
    raise ValueError('Could not form disjoint pseudo-PSU calibration/test samples')


def scenario_gamma(scenario):
    """Decode exact percentage-index scenarios without changing old names."""
    if scenario == 'natural':
        return np.nan
    if scenario == 'mcar':
        return .9
    match = re.fullmatch(r'mnar_g([0-9]{3})', scenario)
    if match and 1 <= int(match.group(1)) <= 100:
        return int(match.group(1)) / 100.
    raise ValueError(f'Unknown scenario {scenario!r}; use mnar_g001 through mnar_g100, mcar, or natural')


def alphas_for_targets(targets=None):
    if targets is None:
        return ALPHAS.copy()
    values = np.asarray(targets, float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all() or np.any((values <= 0) | (values >= 1)):
        raise ValueError('Coverage targets must be finite values strictly between zero and one')
    return np.sort(np.unique(1-values))


def masking_probabilities(y, x, gamma, reference, ref_scale, control=False, k=5,
                          design='dominant'):
    """True response probabilities for every randomized instrument arm.

    Both MNAR designs have max_z p_z(x,y)=gamma and the same mean_z p_z(x,y).
    The complementary design moves the high-response arm across outcome regions;
    its fixed normal-quantile cutpoints do not assume normally distributed Y.
    """
    if not np.isfinite(gamma) or not 0 < gamma <= 1 or k < 2:
        raise ValueError('Masking requires 0 < gamma <= 1 and at least two arms')
    if design not in ('dominant', 'complementary'):
        raise ValueError('Unknown masking design')
    if not np.isfinite(ref_scale) or ref_scale <= 0:
        raise ValueError('Reference scale must be finite and positive')
    if control:
        if design != 'dominant':
            raise ValueError('The MCAR control uses the original dominant-arm design')
        selective = np.full(len(y), .65)
        best = np.zeros(len(y), dtype=int)
    else:
        v = (np.asarray(y) - reference.predict(x))/ref_scale
        selective = .75*np.exp(-.75*v*v)
        best = (np.zeros(len(y), dtype=int) if design == 'dominant' else
                np.searchsorted(norm.ppf(np.arange(1, k)/k), v, side='right'))
    return gamma*np.where(np.arange(k)[None, :] == best[:, None],
                          1., selective[:, None])


def mask(y, x, rng, gamma, reference, ref_scale, control=False, k=5,
         design='dominant'):
    """Generate a valid randomized IV and masking with exact index gamma."""
    z = rng.integers(0, k, len(y))
    probabilities = masking_probabilities(y, x, gamma, reference, ref_scale,
                                          control, k, design)
    p = probabilities[np.arange(len(y)), z]
    r = rng.random(len(y)) < p
    return z, r, p


def metric_rows(dataset, scenario, trial, alpha, method, radius, center,
                y, r, scale, known_outcomes, gamma, diagnostic=None):
    radius = np.broadcast_to(np.asarray(radius), (len(y),))
    widths = 2*radius*scale
    finite = np.isfinite(widths)
    observed = np.asarray(r, bool)
    hit = np.zeros(len(y), bool)
    hit[observed] = np.abs(y[observed]-center[observed]) <= radius[observed]*scale + 1e-12
    row = dict(dataset=dataset, scenario=scenario, trial=trial, alpha=float(alpha),
        target=1-float(alpha), method=method, gamma=gamma,
        coverage=float(hit.mean()) if known_outcomes else np.nan,
        observed_coverage=float(hit[observed].mean()) if observed.any() else np.nan,
        coverage_lower=float((hit | ~finite).mean()), coverage_upper=float((hit | ~observed).mean()),
        mean_length=float(widths.mean()), median_length=float(np.quantile(widths, .5, method='inverted_cdf')),
        finite_mean_length=float(widths[finite].mean()) if finite.any() else np.nan,
        finite_median_length=float(np.median(widths[finite])) if finite.any() else np.nan,
        infinite_fraction=float((~finite).mean()), n_test=len(y), n_test_observed=int(observed.sum()))
    if diagnostic:
        row.update(diagnostic)
    return row


def run_scenario(dataset, df, x, meta, scenario, gamma, args):
    is_natural = scenario == 'natural'
    alphas = np.asarray(getattr(args, 'alphas', ALPHAS), float)
    mask_design = getattr(args, 'mask_design', 'dominant')
    if not is_natural:
        eligible = df.observed.astype(bool).to_numpy()
        df, x = df.loc[eligible].reset_index(drop=True), x[eligible]
    # A prespecified random member per household defines the analysis target
    # and removes within-household duplication in every part of the experiment.
    pre_household_n = len(df)
    household_rng = np.random.default_rng(args.seed + 32771)
    shuffled = household_rng.permutation(len(df))
    _, first = np.unique(df.household_id.to_numpy()[shuffled], return_index=True)
    keep = np.sort(shuffled[first])
    df, x = df.iloc[keep].reset_index(drop=True), x[keep]
    if not is_natural:
        # The masking rule is learned on a separate pilot sample. Conditional
        # on that sample, the same fixed mechanism applies to all train/test rows.
        pilot_rng = np.random.default_rng(args.seed + 65537)
        if args.split_unit == 'psu':
            pilot, pilot_groups = take_group_prefix(df, 'cluster_id', 1000, pilot_rng)
        else:
            pilot_groups = pilot_rng.permutation(df.household_id.unique())[
                :min(1000, max(50, len(df.household_id.unique())//10))]
            pilot = df.household_id.isin(pilot_groups).to_numpy()
        pilot_x, pilot_y = x[pilot], df.outcome.to_numpy(float)[pilot]
        if args.split_unit == 'psu' and len(pilot_y) > 1000:
            pilot_take = pilot_rng.permutation(len(pilot_y))[:1000]
            pilot_x, pilot_y = pilot_x[pilot_take], pilot_y[pilot_take]
        df, x = df.loc[~pilot].reset_index(drop=True), x[~pilot]
    splitter = cluster_split_pools if args.split_unit == 'psu' else split_pools
    train, pool = splitter(df, args.seed, args.n_train, args.pool_size)
    y = df.outcome.to_numpy(float)
    rng = np.random.default_rng(args.seed + int(gamma*1000 if np.isfinite(gamma) else 0) + (10000 if scenario=='mcar' else 0))
    xt, xp, yt, yp = x[train], x[pool], y[train], y[pool]
    if is_natural:
        if meta['instrument_type'] == 'continuous':
            cut = np.quantile(df.instrument.to_numpy(float)[train], [1/3, 2/3])
            allz = np.searchsorted(cut, df.instrument.to_numpy(float), side='right')
            instrument_info = {'cutpoints': cut.tolist(), 'K': 3}
        else:
            cats = sorted(df.instrument.astype(str).unique())
            allz = pd.Categorical(df.instrument.astype(str), categories=cats).codes
            instrument_info = {'levels': cats, 'K': len(cats)}
        zt, rt = allz[train], df.observed.to_numpy(bool)[train]
        zp, rp = allz[pool], df.observed.to_numpy(bool)[pool]
    else:
        # Reference only defines the design; it sees no method-training or test Y.
        reference = RandomForestRegressor(n_estimators=100, min_samples_leaf=30,
            random_state=args.seed, n_jobs=1, oob_score=True).fit(pilot_x, pilot_y)
        ref_scale = max(float(np.std(pilot_y-reference.oob_prediction_)), 1e-6)
        zt, rt, _ = mask(yt, xt, rng, gamma, reference, ref_scale, scenario=='mcar', args.mask_arms, mask_design)
        instrument_info = {'K':args.mask_arms, 'reference_residual_sd':ref_scale,
                           'pilot_rows':len(pilot_y), 'pilot_groups':len(pilot_groups),
                           'pilot_split_unit':args.split_unit, 'mask_design':mask_design,
                           'complementary_cutpoints':norm.ppf(np.arange(1,args.mask_arms)/args.mask_arms).tolist()
                               if mask_design=='complementary' else None}
    # Deliberate NaNs make outcome leakage fail visibly.
    fit_y = np.where(rt, yt, np.nan)
    log(f'{dataset} {scenario}: fit n={len(train)}, observed={rt.sum()}, pool={len(pool)}')
    model = IVPredictor(seed=args.seed, n_trees=args.trees, min_leaf=args.min_leaf,
        grid_size=args.grid_size).fit(xt, zt, rt, fit_y)
    prep = model.prepare(xp, progress=log)
    root = BASE/'results'/dataset
    if args.output_tag:
        root = root/args.output_tag
    outdir = root/scenario
    outdir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(outdir/'split.npz', train_rows=df.iloc[train].row_id.to_numpy(),
        pool_rows=df.iloc[pool].row_id.to_numpy(), train_z=zt, train_r=rt)
    info = dict(dataset=dataset, scenario=scenario, n_eligible=len(df),
        n_before_household_sampling=pre_household_n, n_train=len(train),
        n_train_observed=int(rt.sum()), n_pool=len(pool),
        n_calibration=min(500,len(pool)//2), n_test=min(500,len(pool)//2),
        instrument=instrument_info, outcome_scale=model.scale, outcome_location=model.location,
        gamma_plugin_mean=float(prep.gamma_plugin.mean()),
        gamma_plugin_above_one=float(np.mean(prep.gamma_plugin>1)),
        min_e=float(prep.e.min()), grid_step=float(prep.grid[1]), grid_max=float(prep.grid[-1]),
        reference_controls=meta, seed=args.seed, arguments=vars(args))
    info['gamma_true'] = None if is_natural else float(gamma)
    info['targets'] = (1-alphas).tolist()
    info['training_by_arm'] = [dict(arm=int(j), rows=int(np.sum(zt==j)),
        observed=int(np.sum((zt==j)&rt))) for j in range(model.k)]
    if 'cluster_id' in df:
        tg, pg = set(df.cluster_id.iloc[train]), set(df.cluster_id.iloc[pool])
        info['survey_clusters'] = dict(training=len(tg), reservoir=len(pg),
                                      shared_training_reservoir=len(tg & pg))
    (outdir/'design.json').write_text(json.dumps(info,indent=2))
    rows, diags, subgroup_rows = [], [], []
    def record(method, radius, alpha, trial, ti, center, eval_r, diagnostic):
        rows.append(metric_rows(dataset, scenario, trial, alpha, method, radius,
            center, yp[ti], eval_r, model.scale, not is_natural, gamma, diagnostic))
        if dataset == 'nhanes' and np.isclose(alpha, .1):
            test_frame = df.iloc[pool[ti]]
            sex = pd.to_numeric(test_frame.sex).to_numpy()
            age = test_frame.age.to_numpy()
            groups = {'Male':sex==1, 'Female':sex==2,
                      'Age 17-39':age<40, 'Age 40-64':(age>=40)&(age<65),
                      'Age 65+':age>=65}
            radii = np.broadcast_to(np.asarray(radius), (len(ti),))
            for label, selected in groups.items():
                if selected.any():
                    value = metric_rows(dataset, scenario, trial, alpha, method,
                        radii[selected], center[selected], yp[ti][selected], eval_r[selected],
                        model.scale, not is_natural, gamma, diagnostic)
                    value['subgroup'] = label
                    subgroup_rows.append(value)
    for trial in range(args.trials):
        nc, nt = info['n_calibration'], info['n_test']
        ci, ti = trial_indices(df.iloc[pool], rng, nc, nt, args.split_unit)
        if is_natural:
            zc, rc = zp[ci], rp[ci]
        else:
            zc, rc, _ = mask(yp[ci], xp[ci], rng, gamma, reference, ref_scale, scenario=='mcar', args.mask_arms, mask_design)
        yc = np.where(rc, yp[ci], np.nan)
        calibrated, diagnostics, scores = model.calibrate(prep, ci, zc, rc, yc, xp[ci], alphas)
        diagnostic = dict(calibration_observed=int(rc.sum()))
        for name in ['Softmax IV','Policy IV']:
            diags.append(dict(trial=trial, method=name, **diagnostics[name], **diagnostic))
        variants = [(calibrated, '')]
        if not is_natural:
            # Same trained functions/calibration data, replace only assignment e
            # with randomized design truth: clean robustness sensitivity.
            estimated_e = prep.e
            prep.e = np.full_like(estimated_e, 1 / args.mask_arms)
            known, known_diag, _ = model.calibrate(prep, ci, zc, rc, yc, xp[ci], alphas)
            prep.e = estimated_e
            variants.append(({k:v for k,v in known.items() if k.endswith(' IV')}, ' (known e)'))
        center = model.location + model.scale*prep.center[ti]
        eval_r = rp[ti] if is_natural else np.ones(nt,bool)
        for cal, suffix in variants:
            for method, radii in cal.items():
                for j, alpha in enumerate(alphas):
                    record(method+suffix,radii[j],alpha,trial,ti,center,eval_r,diagnostic)
        for alpha in alphas:
            wr = weighted_radii(scores[rc], 1/prep.response[ci][rc],
                                1/prep.response[ti], alpha)
            record('Weighted CP',wr,alpha,trial,ti,center,eval_r,diagnostic)
        if (trial+1) % 25 == 0:
            log(f'{dataset} {scenario}: trial {trial+1}/{args.trials}')
    pd.DataFrame(rows).to_csv(outdir/'trials.csv', index=False)
    pd.DataFrame(diags).to_csv(outdir/'diagnostics.csv', index=False)
    if subgroup_rows:
        pd.DataFrame(subgroup_rows).to_csv(outdir/'subgroup_trials.csv', index=False)
    return pd.DataFrame(rows)


def summarize(df):
    rows=[]
    for keys,g in df.groupby(['dataset','scenario','method','alpha']):
        d=dict(zip(['dataset','scenario','method','alpha'],keys))
        d['target']=1-d['alpha']; d['trials']=len(g)
        for col in ['coverage','observed_coverage','coverage_lower','coverage_upper',
                    'infinite_fraction','finite_mean_length','finite_median_length']:
            values=g[col].dropna()
            d[col]=values.mean() if len(values) else np.nan
            d[col+'_mcse']=values.std(ddof=1)/np.sqrt(len(values)) if len(values)>1 else np.nan
        # Infinity is preserved in unconditional summaries.
        d['mean_length']=g.mean_length.mean()
        d['median_length']=np.quantile(g.median_length,.5,method='inverted_cdf')
        finite_values=g.finite_median_length.dropna()
        d['finite_length_median']=finite_values.median() if len(finite_values) else np.nan
        d['finite_trial_count']=int(g.finite_median_length.notna().sum())
        rows.append(d)
    return pd.DataFrame(rows)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--dataset',default='heckman')
    p.add_argument('--scenarios',nargs='+',default=['mnar_g080','mnar_g090','mnar_g100','mcar','natural'])
    p.add_argument('--trials',type=int,default=200)
    p.add_argument('--seed',type=int,default=20260918)
    p.add_argument('--n-train',type=int,default=2000)
    p.add_argument('--pool-size',type=int,default=2000)
    p.add_argument('--trees',type=int,default=120)
    p.add_argument('--min-leaf',type=int,default=20)
    p.add_argument('--grid-size',type=int,default=801)
    p.add_argument('--mask-arms',type=int,default=5)
    p.add_argument('--mask-design',choices=['dominant','complementary'],default='dominant')
    p.add_argument('--targets',nargs='+',type=float,default=None,
                   help='Coverage targets strictly between zero and one; default .70 through .95')
    p.add_argument('--output-tag',default='')
    p.add_argument('--split-unit',choices=['family','psu'],default='family')
    args=p.parse_args()
    if args.mask_arms < 2:
        p.error('--mask-arms must be at least two')
    try:
        args.alphas = alphas_for_targets(args.targets).tolist()
        for scenario in args.scenarios:
            scenario_gamma(scenario)
        if args.mask_design != 'dominant' and 'mcar' in args.scenarios:
            raise ValueError('The MCAR control uses --mask-design dominant')
    except ValueError as error:
        p.error(str(error))
    df,x,meta,features=load_data(args.dataset)
    results=[]
    for scenario in args.scenarios:
        if scenario == 'natural' and meta.get('natural_supported') is False:
            raise ValueError(f'{args.dataset}: no documented natural selection instrument; use masking scenarios only')
        sdf,sx,smeta,_ = load_data(args.dataset, scenario) if scenario=='natural' else (df,x,meta,features)
        gamma=scenario_gamma(scenario)
        results.append(run_scenario(args.dataset,sdf,sx,smeta,scenario,gamma,args))
    out=BASE/'results'/args.dataset
    if args.output_tag:
        out=out/args.output_tag
    combined=pd.concat(results,ignore_index=True)
    combined.to_csv(out/'all_trials.csv',index=False)
    summarize(combined).to_csv(out/'summary.csv',index=False)
    (out/'environment.json').write_text(json.dumps(dict(python=platform.python_version(),
        numpy=np.__version__,pandas=pd.__version__,sklearn=sklearn.__version__,features=features),indent=2))
    log(f'Finished; {len(combined)} trial/method/level rows written to {out}')


if __name__=='__main__':
    main()
