
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np
import pandas as pd
from scipy.stats import t as student_t
import sklearn
from sklearn.ensemble import (ExtraTreesRegressor, RandomForestClassifier,
                              RandomForestRegressor)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from run_experiments import load_data

BASE = Path(__file__).resolve().parent
DATASETS = ['heckman', 'nhanes', 'mroz', 'cps', 'nlsy97',
            'nhanes_hba1c', 'nhanes_cholesterol']
SEED = 20260918
FAMILY_SIZE = 7
CACHE_VERSION = 2
LABELS = {'heckman': 'FES wages', 'nhanes': 'NHANES hemoglobin',
          'mroz': 'Mroz wages', 'cps': 'CPS wages', 'nlsy97': 'NLSY97 earnings',
          'nhanes_hba1c': 'NHANES HbA1c',
          'nhanes_cholesterol': 'NHANES cholesterol'}


def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def json_write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def training_edges(y, observed, bins):
    """Training-only equal-frequency bins; ties intact; tails cover all R."""
    values = np.asarray(y, float)[np.asarray(observed, bool)]
    if not np.isfinite(values).all() or len(values) == 0:
        raise ValueError('Training respondents must have finite outcomes')
    # Quantile boundaries cannot split equal values. Duplicate boundaries are
    # removed. Using side=right gives one deterministic bin to any genuine atom.
    cuts = np.unique(np.quantile(values, np.arange(1, bins) / bins))
    return cuts


def outcome_bins(y, observed, cuts):
    answer = np.full(len(y), -1, dtype=int)
    observed = np.asarray(observed, bool)
    answer[observed] = np.searchsorted(cuts, np.asarray(y)[observed], side='right')
    return answer


def subprobability_targets(y, observed, cuts):
    b = outcome_bins(y, observed, cuts)
    target = np.zeros((len(y), len(cuts) + 1), dtype=float)
    ids = np.flatnonzero(observed)
    target[ids, b[ids]] = 1.
    return target


def clip_propensity(e, floor):
    """Explicit floor then row renormalization; raw probabilities are retained."""
    p = np.maximum(np.asarray(e, float), floor)
    return p / p.sum(axis=1, keepdims=True)


def selector_components(q, z, observed, b):
    """q has shape (rows, arms, bins); never indexes Y for a missing row."""
    q = np.asarray(q, float)
    n, k, nb = q.shape
    action = q.argmax(axis=1)
    selector = action[:, None, :] == np.arange(k)[None, :, None]
    m = (q * selector).sum(axis=2)
    plugin = m.sum(axis=1)
    h = np.zeros(n)
    ids = np.flatnonzero(observed)
    h[ids] = (action[ids, b[ids]] == z[ids]).astype(float)
    return plugin, m, h, action


def aipw_score(plugin, m, h, z, e):
    ids = np.arange(len(z))
    return plugin + (h - m[ids, z]) / e[ids, z]


def cluster_inference(values, groups, alpha=.05):
    """Cluster sandwich for an individual-weighted mean, with t critical values."""
    values = np.asarray(values, float)
    _, inverse = np.unique(np.asarray(groups).astype(str), return_inverse=True)
    g = int(inverse.max()) + 1
    if g < 2:
        raise ValueError('At least two independent evaluation groups required')
    mean = float(values.mean())
    sums = np.bincount(inverse, weights=values - mean)
    se = float(np.sqrt(g / (g - 1) * np.sum(sums**2)) / len(values))
    if se == 0:
        p = float(mean <= .9)
    else:
        p = float(student_t.sf((mean - .9) / se, g - 1))
    return dict(estimate=mean, se=se, clusters=g,
                ci95_low=mean - float(student_t.ppf(.975, g - 1)) * se,
                ci95_high=mean + float(student_t.ppf(.975, g - 1)) * se,
                lower95=mean - float(student_t.ppf(1-alpha, g - 1)) * se,
                p_above_09=p)


def foldwise_lower(values, groups, fold, family_size=1, alpha=.05):
    """Average simultaneous fold bounds; avoids treating fitted rules as fixed.

    This remains asymptotic: it requires valid conditional DR remainder control
    and independent clusters. Union bounds address fold/model-selection issues,
    not misspecification or an invalid instrument.
    """
    levels = np.unique(fold)
    result = 0.
    for level in levels:
        take = fold == level
        fit = cluster_inference(np.asarray(values)[take], np.asarray(groups)[take],
                                alpha=alpha / (len(levels) * family_size))
        result += take.mean() * fit['lower95']
    return float(result)


def independent_group_folds(groups, folds, seed):
    """Assign whole groups using IDs and randomization only, never X/Z/R/Y."""
    groups = np.asarray(groups).astype(str)
    unique = np.unique(groups)
    shuffled = np.random.default_rng(seed).permutation(unique)
    for heldout in np.array_split(shuffled, folds):
        take = np.isin(groups, heldout)
        yield np.flatnonzero(~take), np.flatnonzero(take)


def load_cohort(dataset):
    df, x, meta, features = load_data(dataset, 'natural')
    if meta.get('natural_supported') is False:
        raise ValueError(f'{dataset} has no documented original instrument')
    # Same prespecified one-member-per-family draw as the preceding natural
    # experiments, independent of observed outcomes, missingness, and instrument.
    rng = np.random.default_rng(SEED + 32771)
    order = rng.permutation(len(df))
    _, first = np.unique(df.household_id.to_numpy()[order], return_index=True)
    keep = np.sort(order[first])
    n_source = len(df)
    df, x = df.iloc[keep].reset_index(drop=True), x[keep]
    if meta['instrument_type'] == 'continuous':
        prior = BASE/'results'/dataset/'natural'/'design.json'
        cuts = json.loads(prior.read_text())['instrument']['cutpoints']
        z = np.searchsorted(cuts, df.instrument.to_numpy(float), side='right')
        instrument = dict(type='fixed prior tertiles of original continuous Z',
                          cutpoints=cuts, coarsened_instrument=True)
    else:
        cats = sorted(df.instrument.astype(str).unique())
        z = pd.Categorical(df.instrument.astype(str), categories=cats).codes
        instrument = dict(type='original categorical Z', levels=cats,
                          coarsened_instrument=False)
    group_column = 'cluster_id' if dataset.startswith('nhanes') else 'household_id'
    groups = df[group_column].astype(str).to_numpy()
    y, r = df.outcome.to_numpy(float), df.observed.to_numpy(bool)
    if not np.array_equal(np.isfinite(y), r):
        raise ValueError('Missing outcomes must remain missing')
    source_stem = 'prepared_natural' if (BASE/'data'/dataset/'prepared_natural.csv').exists() else 'prepared'
    source = BASE/'data'/dataset/(source_stem + '.csv')
    info = dict(dataset=dataset, label=LABELS[dataset], n_source=n_source,
                n=len(df), n_observed=int(r.sum()), response_rate=float(r.mean()),
                independent_groups=len(np.unique(groups)), group_column=group_column,
                features=features, instrument=instrument, metadata=meta,
                one_member_seed=SEED+32771, source=str(source.relative_to(BASE)),
                source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                target='Unweighted one-member-per-household prepared original-data cohort',
                conditional_on_nhanes_design=dataset.startswith('nhanes'))
    info['metadata_sha256'] = hashlib.sha256(json.dumps(meta, sort_keys=True).encode()).hexdigest()
    return df, x, z, r, y, groups, info


def propensity_fits(x, z, train, valid, seed, trees):
    k = len(np.unique(z))
    logistic = make_pipeline(StandardScaler(),
                            LogisticRegression(C=1., max_iter=3000))
    forest = RandomForestClassifier(n_estimators=trees, min_samples_leaf=20,
                max_features=.8, random_state=seed, n_jobs=1)
    outputs = {}
    for name, model in [('logistic', logistic), ('forest', forest)]:
        model.fit(x[train], z[train])
        if not np.array_equal(model.classes_, np.arange(k)):
            raise ValueError('An instrument arm is absent from a training fold')
        outputs[name] = model.predict_proba(x[valid])
    outputs['ensemble'] = (outputs['logistic'] + outputs['forest']) / 2
    return outputs


def prespecified_specs(primary_only=False):
    if primary_only:
        return [('rf', 16, SEED)]
    return [(model, bins, SEED) for model in ['rf', 'extra'] for bins in [8, 16, 32]] + [
        ('rf', 16, SEED+1), ('rf', 16, SEED+2)]


def run_dataset(dataset, root, trees=160, folds=5, primary_only=False, reuse=False):
    out = root/dataset
    out.mkdir(parents=True, exist_ok=True)
    df, x, z, r, y, groups, info = load_cohort(dataset)
    n, k = len(df), len(np.unique(z))
    json_write(out/'cohort.json', info)
    pd.DataFrame(dict(row_id=df.row_id, household_id=df.household_id,
                      group=groups, z=z, observed=r.astype(int))).to_csv(out/'cohort_rows.csv', index=False)
    arms = pd.DataFrame(dict(arm=z, observed=r)).groupby('arm').agg(
        n=('observed', 'size'), observed=('observed', 'sum'), response_rate=('observed', 'mean'))
    arms.to_csv(out/'arms.csv')
    all_rows, fold_rows, audits = [], [], []
    propensities = {}
    split_cache = {}
    for model_name, bins, seed in prespecified_specs(primary_only):
        key = f'{model_name}_b{bins}_s{seed}'
        path = out/(key+'.npz')
        audit_path = out/(key+'_audit.json')
        if seed not in split_cache:
            split_cache[seed] = list(independent_group_folds(groups, folds, seed))
        splits = split_cache[seed]
        if reuse and path.exists() and audit_path.exists():
            saved = np.load(path)
            if not np.array_equal(saved['row_id'], df.row_id.to_numpy()):
                raise ValueError('Cached row IDs do not match cohort')
            plugin, m, h, fold = [saved[v] for v in ['plugin', 'm', 'h', 'fold']]
            e_all = {name: saved['e_'+name] for name in ['logistic', 'forest', 'ensemble']}
            low_support = saved['low_support']
            model_audits = json.loads(audit_path.read_text())
            if any(a['trees'] != trees or a['folds'] != folds for a in model_audits):
                raise ValueError('Cached forest/fold settings mismatch')
            if any(a.get('cache_version') != CACHE_VERSION or
                   a.get('source_sha256') != info['source_sha256'] or
                   a.get('metadata_sha256') != info['metadata_sha256']
                   for a in model_audits):
                raise ValueError('Cached implementation/source/metadata mismatch; rerun without --reuse')
        else:
            plugin, m, h = np.empty(n), np.empty((n, k)), np.empty(n)
            fold = np.full(n, -1, dtype=int)
            e_all = {name: np.empty((n, k)) for name in ['logistic', 'forest', 'ensemble']}
            low_support = np.empty(n)
            model_audits = []
            for j, (train, valid) in enumerate(splits):
                assert not set(groups[train]) & set(groups[valid])
                if (seed, j) not in propensities:
                    propensities[seed, j] = propensity_fits(x, z, train, valid, seed+j, trees)
                for name in e_all:
                    e_all[name][valid] = propensities[seed, j][name]
                cuts = training_edges(y[train], r[train], bins)
                target = subprobability_targets(y[train], r[train], cuts)
                q = np.empty((len(valid), k, target.shape[1]))
                leaves = []
                for arm in range(k):
                    inside = z[train] == arm
                    leaf = max(5, min(20, int(inside.sum()) // 30))
                    leaves.append(leaf)
                    model_class = RandomForestRegressor if model_name == 'rf' else ExtraTreesRegressor
                    model = model_class(n_estimators=trees, min_samples_leaf=leaf,
                           max_features=.8, random_state=seed+100*j+arm, n_jobs=1)
                    model.fit(x[train][inside], target[inside])
                    q[:, arm, :] = model.predict(x[valid])
                b = outcome_bins(y[valid], r[valid], cuts)
                gp, mm, hh, actions = selector_components(q, z[valid], r[valid], b)
                plugin[valid], m[valid], h[valid], fold[valid] = gp, mm, hh, j
                chosen_e = np.take_along_axis(e_all['ensemble'][valid], actions, axis=1)
                maxima = q.max(axis=1)
                low_support[valid] = np.sum(maxima * (chosen_e < .01), axis=1)
                ids = np.arange(len(valid))
                # Proper multicategory Brier score, including missingness as a
                # category, on truly held-out original observations.
                actual_q = q[ids, z[valid]]
                pred = np.c_[1-actual_q.sum(axis=1), actual_q]
                truth = np.where(r[valid], b+1, 0)
                brier = np.sum(pred**2, axis=1) - 2*pred[ids, truth] + 1
                record = dict(dataset=dataset, spec=key, fold=j, trees=trees, folds=folds,
                    cache_version=CACHE_VERSION, source_sha256=info['source_sha256'],
                    metadata_sha256=info['metadata_sha256'],
                    n_train=len(train), n_eval=len(valid), groups_eval=len(np.unique(groups[valid])),
                    shared_groups=0, requested_bins=bins, actual_bins=len(cuts)+1,
                    cutpoints=cuts.tolist(), min_leaf_by_arm=leaves,
                    heldout_joint_brier=float(brier.mean()),
                    plugin=float(gp.mean()), conditional_above_one=float(np.mean(gp > 1+1e-10)),
                    subprobability_sum_max=float(q.sum(axis=2).max()))
                model_audits.append(record)
                log(f'{dataset} {key} fold {j+1}/{folds}: n={len(valid)} bins={len(cuts)+1}')
            assert np.all(fold >= 0) and np.isfinite(plugin).all()
            np.savez_compressed(path, row_id=df.row_id.to_numpy(), plugin=plugin, m=m,
                                h=h, fold=fold, low_support=low_support,
                                **{'e_'+name: values for name, values in e_all.items()})
            json_write(audit_path, model_audits)
        audits.extend(model_audits)
        variants = [('ensemble', .01)]
        if (model_name, bins, seed) == ('rf', 16, SEED):
            variants += [('ensemble', .005), ('ensemble', .02), ('logistic', .01), ('forest', .01)]
        for e_name, floor in variants:
            e_raw = e_all[e_name]
            e = clip_propensity(e_raw, floor)
            score = aipw_score(plugin, m, h, z, e)
            stats = cluster_inference(score, groups)
            fold_lower = foldwise_lower(score, groups, fold)
            family_lower = foldwise_lower(score, groups, fold, FAMILY_SIZE)
            selected = plugin > 1+1e-10
            comp_values = selected * (score - 1)
            comp = cluster_inference(comp_values, groups)
            comp_fold_lower = foldwise_lower(comp_values, groups, fold)
            actual_raw = e_raw[np.arange(n), z]
            weights = h / e[np.arange(n), z]
            ess = float(weights.sum()**2 / np.sum(weights**2)) if np.any(weights) else 0.
            primary = (model_name, bins, seed, e_name, floor) == ('rf', 16, SEED, 'ensemble', .01)
            row = dict(dataset=dataset, label=LABELS[dataset], spec=key, primary=primary,
                model=model_name, bins_requested=bins, seed=seed, propensity=e_name, floor=floor,
                n=n, n_observed=int(r.sum()), clusters=len(np.unique(groups)),
                response_rate=float(r.mean()), plugin=float(plugin.mean()),
                learned_mass=stats['estimate'], se=stats['se'],
                ci95_low=stats['ci95_low'], ci95_high=stats['ci95_high'],
                lower95=stats['lower95'], p_above_09=stats['p_above_09'],
                foldwise_lower95=fold_lower, familywise_fold_lower95=family_lower,
                numerical_support_above_09=bool(family_lower > .9),
                conditional_above_one=float(selected.mean()),
                mean_excess_above_one=float(np.maximum(plugin-1, 0).mean()),
                compatibility_moment=comp['estimate'], compatibility_se=comp['se'],
                compatibility_lower95=comp['lower95'], compatibility_fold_lower95=comp_fold_lower,
                fraction_any_arm_below_001=float(np.mean(e_raw.min(axis=1) < .01)),
                fraction_assigned_below_001=float(np.mean(actual_raw < .01)),
                fraction_any_arm_clipped=float(np.mean((e_raw < floor).any(axis=1))),
                raw_assigned_min=float(actual_raw.min()), raw_min_arm_median=float(np.median(e_raw.min(axis=1))),
                raw_min_arm_p05=float(np.quantile(e_raw.min(axis=1), .05)),
                max_inverse_weight=float(1/e[np.arange(n), z].min()),
                selector_ipw_ess=ess, policy_mass_low_support=float(np.sum(m * (e_raw < .01), axis=1).mean()),
                propensity_log_loss=float(log_loss(z, e_raw, labels=np.arange(k))),
                folds=folds, trees=trees)
            all_rows.append(row)
            for level in np.unique(fold):
                take = fold == level
                fs = cluster_inference(score[take], groups[take])
                fold_rows.append(dict(dataset=dataset, spec=key, propensity=e_name,
                                     floor=floor, fold=int(level), n=int(take.sum()), **fs))
        log(f'{dataset} {key}: plugin={plugin.mean():.4f}, main DR={all_rows[-len(variants)]["learned_mass"]:.4f}')
        # Preserve completed work even if a subsequent sensitivity fails.
        pd.DataFrame(all_rows).to_csv(out/'summary.csv', index=False)
        pd.DataFrame(fold_rows).to_csv(out/'fold_summary.csv', index=False)
    json_write(out/'all_model_audits.json', audits)
    return pd.DataFrame(all_rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', default=DATASETS, choices=DATASETS)
    parser.add_argument('--output', type=Path, default=BASE/'results'/'natural_gamma')
    parser.add_argument('--trees', type=int, default=160)
    parser.add_argument('--folds', type=int, default=5)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--primary-only', action='store_true')
    parser.add_argument('--reuse', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    json_write(args.output/'run_specification.json', dict(
        datasets=args.datasets, trees=args.trees, folds=args.folds,
        specs=prespecified_specs(args.primary_only), primary=['rf', 16, SEED, 'ensemble', .01],
        family_size=FAMILY_SIZE, threshold=.9, alpha=.05,
        python=platform.python_version(), numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__,
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        uncertainty='Asymptotic cluster DR learned-selector intervals; not two-sided intervals for full Gamma',
        original_missingness=True, generated_outcomes=False, generated_instruments=False,
        familywise_rule='Average fold limits with alpha/(5*7); assumptions and diagnostics still required'))
    frames = []
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        tasks = {executor.submit(run_dataset, dataset, args.output, args.trees,
                                 args.folds, args.primary_only, args.reuse): dataset
                 for dataset in args.datasets}
        for future in as_completed(tasks):
            frames.append(future.result())
    summary = pd.concat(frames, ignore_index=True)
    summary.to_csv(args.output/'summary.csv', index=False)
    summary[summary.primary].to_csv(args.output/'primary.csv', index=False)
    log('Completed all requested original-data Gamma fits')


if __name__ == '__main__':
    main()
