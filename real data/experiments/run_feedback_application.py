
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
import sklearn
from sklearn.ensemble import RandomForestRegressor

from estimate_natural_gamma import DATASETS, LABELS, load_cohort
from iv_methods import IVPredictor, conformal_radius, weighted_radii

BASE = Path(__file__).resolve().parent
DEFAULT_OUTPUT = BASE / 'results' / 'feedback_application'
METHODS = ['Softmax IV', 'Policy IV', 'Naive CP (X)', 'MAR population CP (X)',
           'Naive CP (X,Z)', 'MAR population CP (X,Z)',
           'Best-case CP (X,Z)', 'Worst-case CP (X,Z)']


def log(message):
    print(time.strftime('%H:%M:%S'), message, flush=True)


def clean_json(value):
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return '+Infinity' if value > 0 else '-Infinity' if value < 0 else None
    if isinstance(value, np.generic):
        return value.item()
    return value


def json_write(path, value):
    path.write_text(json.dumps(clean_json(value), indent=2, allow_nan=False),
                    encoding='utf-8')


def disjoint_splits(groups, seed, max_train=2000, max_calibration=700):
    """50/25/25 assignment by group IDs alone, followed by random row caps."""
    groups = np.asarray(groups).astype(str)
    unique = np.unique(groups)
    if len(unique) < 12:
        raise ValueError('At least twelve splitting groups required')
    rng = np.random.default_rng(seed)
    order = rng.permutation(unique)
    a, b = len(unique) // 2, 3 * len(unique) // 4
    indices = [np.flatnonzero(np.isin(groups, chosen))
               for chosen in [order[:a], order[a:b], order[b:]]]
    indices[0] = np.sort(rng.permutation(indices[0])[:max_train])
    indices[1] = np.sort(rng.permutation(indices[1])[:max_calibration])
    return tuple(indices)


def covariates_with_instrument(df, x, meta):
    """Use the original continuous Z, or full original categorical indicators."""
    if meta['instrument_type'] == 'continuous':
        extra = df.instrument.to_numpy(float)[:, None]
        features = ['original_continuous_instrument']
    else:
        frame = pd.get_dummies(df.instrument.astype(str), prefix='instrument', dtype=float)
        extra, features = frame.to_numpy(float), list(frame.columns)
    return np.c_[x, extra], features


def completion_bounds(observed_scores, response, alpha):
    """Sharp extreme ranks with each unknown nonnegative score in [0, infinity]."""
    response = np.asarray(response, bool)
    scores = np.asarray(observed_scores, float)
    if len(scores) != response.sum() or not np.isfinite(scores).all() or np.any(scores < 0):
        raise ValueError('Supply finite nonnegative scores for observed records only')
    low, high = np.zeros(len(response)), np.full(len(response), np.inf)
    low[response], high[response] = scores, scores
    return conformal_radius(low, alpha), conformal_radius(high, alpha)


def width_summary(widths):
    widths = np.asarray(widths, float)
    if len(widths) == 0 or np.isnan(widths).any() or np.any(widths < 0):
        raise ValueError('Widths must be nonempty, nonnegative, and nonmissing')
    finite = np.isfinite(widths)
    # Inverted-CDF median preserves infinity and avoids inf-inf interpolation.
    return dict(n_missing_targets=len(widths), mean_length=float(widths.mean()),
                median_length=float(np.quantile(widths, .5, method='inverted_cdf')),
                finite_mean_length=float(widths[finite].mean()) if finite.any() else np.nan,
                finite_median_length=float(np.median(widths[finite])) if finite.any() else np.nan,
                infinite_fraction=float((~finite).mean()))


def forest(seed, args):
    return RandomForestRegressor(n_estimators=args.trees,
        min_samples_leaf=args.min_leaf, max_features=1., random_state=seed, n_jobs=1)


def run_dataset(dataset, args):
    started = time.time()
    out = Path(args.output) / dataset
    if (out / 'summary.csv').exists() and not args.overwrite:
        raise FileExistsError(f'{out} already exists; use a new output path or --overwrite')
    out.mkdir(parents=True, exist_ok=True)
    df, x, z, r, y, _, info = load_cohort(dataset)
    group_column = 'cluster_id' if args.split_unit == 'psu' and dataset.startswith('nhanes') else 'household_id'
    groups = df[group_column].astype(str).to_numpy()
    train, calibration, pool = disjoint_splits(groups, args.seed, args.max_train, args.max_calibration)
    target = pool[~r[pool]]
    if not len(target):
        raise ValueError(f'{dataset} has no genuinely missing target outcomes')
    if not np.isnan(y[target]).all():
        raise ValueError('All application target outcomes must be genuinely missing')
    xz, instrument_features = covariates_with_instrument(df, x, info['metadata'])
    log(f'{dataset}: train={len(train)}, observed={r[train].sum()}, calibration={len(calibration)}, missing targets={len(target)}')
    model = IVPredictor(seed=args.seed, n_trees=args.trees, min_leaf=args.min_leaf,
                        grid_size=args.grid_size).fit(x[train], z[train], r[train], y[train])
    # Only calibration rows need numerical subdensity integration.
    prepared = model.prepare(x[calibration], progress=lambda message: log(f'{dataset}: {message}'))
    alphas = 1 - np.asarray(args.targets, float)
    radii_iv, diagnostics, scores_x = model.calibrate(prepared, np.arange(len(calibration)),
        z[calibration], r[calibration], y[calibration], x[calibration], alphas)
    center_x = model.location + model.scale * model.score.predict(x[target])
    score_xz = forest(args.seed + 201, args).fit(xz[train][r[train]],
        (y[train][r[train]] - model.location) / model.scale)
    response_xz = forest(args.seed + 202, args).fit(xz[train], r[train].astype(float))
    center_xz = model.location + model.scale * score_xz.predict(xz[target])
    observed_calibration = calibration[r[calibration]]
    scores_xz = np.abs((y[observed_calibration] - model.location) / model.scale
                       - score_xz.predict(xz[observed_calibration]))
    pcal_x = np.clip(model.response_model.predict(x[observed_calibration]), .02, .98)
    ptarget_x = np.clip(model.response_model.predict(x[target]), .02, .98)
    pcal_xz = np.clip(response_xz.predict(xz[observed_calibration]), .02, .98)
    ptarget_xz = np.clip(response_xz.predict(xz[target]), .02, .98)
    rows, summary = [], []
    for a, alpha in enumerate(alphas):
        best, worst = completion_bounds(scores_xz, r[calibration], alpha)
        radii = {'Softmax IV': radii_iv['Softmax IV'][a],
                 'Policy IV': radii_iv['Policy IV'][a],
                 'Naive CP (X)': radii_iv['Naive CP'][a],
                 'MAR population CP (X)': weighted_radii(scores_x[r[calibration]],
                     1 / pcal_x, 1 / ptarget_x, alpha),
                 'Naive CP (X,Z)': conformal_radius(scores_xz, alpha),
                 'MAR population CP (X,Z)': weighted_radii(scores_xz,
                     1 / pcal_xz, 1 / ptarget_xz, alpha),
                 'Best-case CP (X,Z)': best, 'Worst-case CP (X,Z)': worst}
        worst_lower, worst_upper = center_xz - worst * model.scale, center_xz + worst * model.scale
        for name in METHODS:
            radius = np.broadcast_to(radii[name], (len(target),)) * model.scale
            center = center_xz if '(X,Z)' in name else center_x
            lower, upper, length = center - radius, center + radius, 2 * radius
            frame = pd.DataFrame(dict(dataset=dataset, row_id=df.row_id.iloc[target].to_numpy(),
                household_id=df.household_id.iloc[target].to_numpy(),
                split_group=groups[target], original_instrument=df.instrument.iloc[target].to_numpy(),
                instrument_arm=z[target], observed=0, target=1 - float(alpha), method=name,
                center=center, lower=lower, upper=upper, length=length,
                finite=np.isfinite(length), contained_in_worst_case=(lower >= worst_lower) & (upper <= worst_upper)))
            rows.append(frame)
            summary.append(dict(dataset=dataset, label=LABELS[dataset], target=1 - float(alpha),
                method=name, **width_summary(length),
                contained_in_worst_case_fraction=float(frame.contained_in_worst_case.mean())))
    intervals, summary_frame = pd.concat(rows, ignore_index=True), pd.DataFrame(summary)
    intervals.to_csv(out / 'intervals.csv', index=False)
    summary_frame.to_csv(out / 'summary.csv', index=False)
    membership = np.full(len(df), 'unused', dtype=object)
    membership[train], membership[calibration], membership[pool] = 'train', 'calibration', 'target_pool'
    membership[target] = 'missing_target'
    pd.DataFrame(dict(row_id=df.row_id, household_id=df.household_id, split_group=groups,
        membership=membership, observed=r.astype(int), instrument_arm=z)).to_csv(out / 'split.csv', index=False)
    pd.DataFrame(dict(row_id=df.row_id.iloc[target].to_numpy(),
        probability_response_x=ptarget_x, probability_response_xz=ptarget_xz)).to_csv(out / 'response_predictions.csv', index=False)
    split_groups = [set(groups[ids]) for ids in [train, calibration, pool]]
    overlaps = [len(split_groups[i] & split_groups[j]) for i in range(3) for j in range(i+1, 3)]
    if any(overlaps):
        raise AssertionError('Splitting groups overlap')
    info.update(dict(application_split_group=group_column,
        split_seed=args.seed, arguments=vars(args), n_train=len(train),
        n_train_observed=int(r[train].sum()), n_calibration=len(calibration),
        n_calibration_observed=int(r[calibration].sum()), n_target_pool=len(pool),
        n_missing_targets=len(target), split_group_overlap_counts=overlaps,
        training_by_arm=[dict(arm=int(j), rows=int(np.sum(z[train] == j)),
            observed=int(np.sum((z[train] == j) & r[train]))) for j in range(model.k)],
        additional_xz_features=instrument_features, outcome_location=model.location,
        outcome_scale=model.scale, mean_calibration_gamma_plugin=float(prepared.gamma_plugin.mean()),
        calibration_diagnostics=diagnostics,
        evaluation='Descriptive intervals for actual missing participants; no coverage computed.',
        method_target='Marginal population; applying the intervals to R=0 does not imply conditional-R=0 coverage.',
        mar_weights='1 / estimated P(R=1|features), clipped to [0.02,0.98]; not missing-population odds weights.',
        naive_bounds='Fill missing nonnegative calibration residuals by 0 (best) or +infinity (worst), with the finite-sample rank ceil((n_calibration+1)*target). Only worst is a conservative population benchmark under exchangeability; best is an optimistic bound, not a valid method.',
        instrument_validity='Original instruments are candidate instruments; exclusion and outcome independence are unverified.',
        splitting_qualification='Household splits do not establish survey PSU independence.' if group_column == 'household_id' and dataset.startswith('nhanes') else 'Groups are disjoint; survey-weighted or national-population claims are not made.',
        python=platform.python_version(), numpy=np.__version__, sklearn=sklearn.__version__,
        source_script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        iv_methods_sha256=hashlib.sha256((BASE / 'iv_methods.py').read_bytes()).hexdigest(),
        elapsed_seconds=time.time()-started))
    json_write(out / 'design.json', info)
    log(f'{dataset}: saved {len(intervals)} interval records ({time.time()-started:.1f}s)')
    return summary_frame


def interval_figure(root, datasets=('heckman', 'nhanes'), target=.9, count=10, seed=20260919):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    visible_methods = ['Softmax IV', 'Policy IV', 'Naive CP (X,Z)',
                       'MAR population CP (X,Z)', 'Best-case CP (X,Z)', 'Worst-case CP (X,Z)']
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#888888', '#222222']
    fig, axes = plt.subplots(1, len(datasets), figsize=(7 * len(datasets), 7), squeeze=False)
    selected_rows = []
    for ax, dataset in zip(axes[0], datasets):
        frame = pd.read_csv(root / dataset / 'intervals.csv')
        frame = frame[np.isclose(frame.target, target)]
        all_ids = np.sort(frame.row_id.unique())
        chosen = np.random.default_rng(seed).choice(all_ids, min(count, len(all_ids)), replace=False)
        centers = frame[(frame.method == 'Naive CP (X,Z)') & frame.row_id.isin(chosen)]
        chosen = centers.sort_values('center').row_id.to_numpy()
        take = frame[frame.row_id.isin(chosen)]
        selected_rows.append(take)
        finite_ends = take[['lower', 'upper']].to_numpy().ravel()
        finite_ends = finite_ends[np.isfinite(finite_ends)]
        lo, hi = float(finite_ends.min()), float(finite_ends.max())
        margin = max(.10 * (hi - lo), .01)
        lo, hi = lo-margin, hi+margin
        for i, row_id in enumerate(chosen):
            for j, method in enumerate(visible_methods):
                row = take[(take.row_id == row_id) & (take.method == method)].iloc[0]
                yy = i + (j - 2.5) * .12
                if np.isfinite(row.length):
                    ax.plot([row.lower, row.upper], [yy, yy], color=colors[j], lw=1.5)
                    ax.plot([row.center], [yy], '.', color=colors[j], ms=3)
                else:
                    ax.annotate('', (hi, yy), (lo, yy), arrowprops=dict(arrowstyle='<->', color=colors[j], lw=1.2))
        info = json.loads((root / dataset / 'design.json').read_text())
        ax.set(title=LABELS[dataset], xlim=(lo-margin*.2, hi+margin*.2),
               xlabel=info['metadata'].get('outcome_units', 'Outcome'),
               ylabel='Missing-outcome participant (display order)', yticks=np.arange(len(chosen)),
               yticklabels=np.arange(1, len(chosen)+1))
        ax.invert_yaxis()
        ax.grid(axis='x', color='#eeeeee')
    legend = [Line2D([0], [0], color=color, lw=2, label=name)
              for color, name in zip(colors, visible_methods)]
    fig.legend(handles=legend, loc='lower center', ncol=3, bbox_to_anchor=(.5, .015), fontsize=9)
    fig.suptitle(f'Actual missing outcomes: intervals at the {target:.0%} nominal population target\n'
                 'Double arrows denote the entire real line; coverage is unobserved.', fontsize=12)
    fig.tight_layout(rect=(0, .12, 1, .92))
    fig.savefig(root / 'missing_participant_intervals.pdf', bbox_inches='tight')
    fig.savefig(root / 'missing_participant_intervals.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    pd.concat(selected_rows, ignore_index=True).to_csv(root / 'figure_participants.csv', index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=DATASETS, default=DATASETS)
    parser.add_argument('--targets', nargs='+', type=float, default=[.7, .8, .9, .95])
    parser.add_argument('--seed', type=int, default=20260919)
    parser.add_argument('--trees', type=int, default=120)
    parser.add_argument('--min-leaf', type=int, default=20)
    parser.add_argument('--grid-size', type=int, default=801)
    parser.add_argument('--max-train', type=int, default=2000)
    parser.add_argument('--max-calibration', type=int, default=700)
    parser.add_argument('--split-unit', choices=['household', 'psu'], default='household')
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--output', type=str, default=str(DEFAULT_OUTPUT))
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if any(not 0 < value < 1 for value in args.targets):
        parser.error('Targets must lie strictly between zero and one')
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    summaries = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(run_dataset, dataset, args): dataset for dataset in args.datasets}
        for future in as_completed(jobs):
            summaries.append(future.result())
    summary = pd.concat(summaries, ignore_index=True).sort_values(['dataset', 'target', 'method'])
    summary.to_csv(root / 'summary.csv', index=False)
    selected = [dataset for dataset in ['heckman', 'nhanes'] if dataset in args.datasets]
    if selected and any(np.isclose(args.targets, .9)):
        interval_figure(root, selected)
    json_write(root / 'run_manifest.json', dict(arguments=vars(args), datasets=args.datasets,
        summary_rows=len(summary), outcomes_imputed=False, coverage_estimated=False,
        target_qualification='Population-calibrated intervals displayed for genuine R=0 targets, without claiming conditional-R=0 validity.'))


if __name__ == '__main__':
    main()
