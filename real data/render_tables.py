"""Generate the four JRSSB empirical tables from saved experiment results.

The .tex files retain the draft's table layouts; CSV files retain unrounded
estimates. No models are fitted. Use --check-reference to require every printed
number to match the bundled draft table templates before writing any outputs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TARGET = 0.9
DATASETS = ("heckman", "nhanes_height")
LABELS = {"heckman": "FES", "nhanes_height": "NHANES III"}
METHODS = ("Policy IV", "Weighted CP (X)", "Weighted CP (X,Z)")
TOKEN = re.compile(r"\\infty|\d+\.\d+")
CELL_VALUE = re.compile(
    r"(?P<math>\$)?(?P<point>\\infty|\d+\.\d+)(?(math)\$)"
    r"(?P<uncertainty>(?P<gap>\s*)\((?P<mcse>--|\d+\.\d+)\))?")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Inputs:
    def __init__(self, base):
        self.base = base
        self.sources = {}

    def path(self, relative):
        # Original design records were written on Windows; allow relocation.
        relative = Path(str(relative).replace("\\", "/"))
        path = self.base / relative
        self.sources[Path(relative).as_posix()] = sha256(path)
        return path

    def csv(self, relative):
        return pd.read_csv(self.path(relative))

    def json(self, relative):
        return json.loads(self.path(relative).read_text(encoding="utf-8-sig"))


def row90(frame, method):
    selected = frame[(frame.method == method) & np.isclose(frame.target, TARGET)]
    if len(selected) != 1:
        raise ValueError(f"Expected one target-0.9 row for {method}; got {len(selected)}")
    return selected.iloc[0]


def display(value, digits):
    if np.isnan(value) or np.isneginf(value):
        raise ValueError(f"Invalid table value: {value}")
    return r"\infty" if np.isposinf(value) else f"{value:.{digits}f}"


def cell(value, digits, mcse=None):
    """Format a point estimate and optional MCSE (undefined MCSE becomes --)."""
    point = display(value, digits)
    if mcse is None:
        return (point,)
    error = (display(mcse, 3) if np.isfinite(value) and np.isfinite(mcse)
             else "--")
    return point, error


def replace_cell(original, values):
    """Replace one complete numeric cell, preserving the template's wrappers."""
    match = CELL_VALUE.search(original)
    if match is None or bool(match["uncertainty"]) != (len(values) == 2):
        raise ValueError(f"Table cell does not match its estimate/MCSE layout: {original!r}")
    existing = ((match["point"], match["mcse"]) if match["uncertainty"]
                else (match["point"],))
    if existing == values:
        return original
    point = r"$\infty$" if values[0] == r"\infty" else values[0]
    if len(values) == 2:
        point += (match["gap"] or " ") + f"({values[1]})"
    return original[:match.start()] + point + original[match.end():]


def natural_folder(dataset):
    if dataset == "heckman":
        return Path("results/feedback_application/heckman")
    return Path("results/nhanes_five_categories/natural/nhanes_height")


def render(name, cells, rows, coverages=()):
    template = ROOT / "templates" / f"{name}.tex"
    text = template.read_text(encoding="utf-8")
    before, tail = text.split(r"\midrule", 1)
    body, after = tail.split(r"\bottomrule", 1)
    # Exclude the numeric color intensity (!20) by matching decimal numbers only.
    reference = TOKEN.findall(body)
    # Replace cells rather than individual numeric tokens: infinite widths have
    # no finite MCSE, so transitions can change the number of printed numbers.
    parts = re.split(r"(&|\\\\)", body)
    slots = [index for index, part in enumerate(parts) if TOKEN.search(part)]
    if len(slots) != len(cells):
        raise ValueError(f"{name}: {len(cells)} cells for {len(slots)} template slots")
    for index, values in zip(slots, cells):
        parts[index] = replace_cell(parts[index], values)
    body = "".join(parts)
    values = TOKEN.findall(body)
    if any(not np.isfinite(value) or not 0 <= value <= 1 for value in coverages):
        raise ValueError(f"{name}: coverage must be a finite probability")
    colors = iter("green" if value >= TARGET else "red" for value in coverages)
    if len(re.findall(r"\\cellcolor\{[^}]*\}", body)) != len(coverages):
        raise ValueError(f"{name}: coverage color count differs from table layout")
    body = re.sub(r"\\cellcolor\{[^}]*\}",
                  lambda match: r"\cellcolor{" + next(colors) + "!20}", body)
    output = before + r"\midrule" + body + r"\bottomrule" + after
    return dict(name=name, text=output, frame=pd.DataFrame(rows),
                matches_reference=reference == values,
                reference_values=reference, generated_values=values,
                template_sha256=sha256(template))


def masking_table(inputs):
    summaries = {}
    for dataset in DATASETS:
        root = (Path("results/observed_gamma/primary") if dataset == "heckman"
                else Path("results/nhanes_five_categories/primary"))
        summaries[dataset] = inputs.csv(root / dataset / "anchored_g100/summary.csv")
    cells, rows, coverages = [], [], []
    for method in METHODS:
        for dataset in DATASETS:
            row = row90(summaries[dataset], method)
            mean_mcse = row.finite_mean_length_mcse if np.isfinite(row.mean_length) else np.nan
            cells += [cell(row.coverage, 3, row.coverage_mcse),
                      cell(row.mean_length, 2, mean_mcse)]
            coverages.append(float(row.coverage))
            rows.append(dict(dataset=LABELS[dataset], dataset_id=dataset,
                             method=method, target=TARGET, setting=1, g=1.0,
                             trials=int(row.trials), coverage=row.coverage,
                             coverage_mcse=row.coverage_mcse, mean_width=row.mean_length,
                             mean_width_mcse=mean_mcse,
                             infinite_fraction=row.infinite_fraction))
    return render("empirical_masking", cells, rows, coverages)


def mar_table(inputs):
    cells, rows, coverages = [], [], []
    for dataset, folder, seed in [
        ("heckman", "results/feedback/mar_x_high_response/heckman/mar_x", 20260922),
        ("nhanes_height", "results/nhanes_five_categories/mar/nhanes_height/mar_x", 20260925),
    ]:
        summary = inputs.csv(Path(folder) / "summary.csv")
        trials = inputs.csv(Path(folder) / "trials.csv")
        design = inputs.json(Path(folder) / "design.json")
        for offset, method in enumerate(METHODS):
            row = row90(summary, method)
            selected = trials[(trials.method == method) & np.isclose(trials.target, TARGET)].sort_values("trial")
            np.testing.assert_array_equal(selected.trial, np.arange(200))
            medians = selected.median_length.to_numpy()
            rng = np.random.default_rng(seed + offset)
            # Same 20,000-replicate inverted-CDF bootstrap as the paper audit.
            bootstrap = np.quantile(medians[rng.integers(0, 200, size=(20000, 200))],
                                    .5, axis=1, method="inverted_cdf")
            if np.isnan(bootstrap).any() or np.isneginf(bootstrap).any():
                raise ValueError(f"Invalid bootstrap median for {dataset}, {method}")
            median_mcse = (float(bootstrap.std(ddof=1))
                           if np.isfinite(bootstrap).all() else np.nan)
            mean_mcse = (float(selected.mean_length.std(ddof=1) / np.sqrt(200))
                         if np.isfinite(row.mean_length) else np.nan)
            cells += [cell(row.coverage, 3, row.coverage_mcse),
                      cell(row.mean_length, 2, mean_mcse),
                      cell(row.median_length, 2, median_mcse),
                      cell(100 * row.infinite_fraction, 1)]
            coverages.append(float(row.coverage))
            rows.append(dict(dataset=LABELS[dataset], dataset_id=dataset, method=method,
                             target=TARGET, trials=200, coverage=row.coverage,
                             coverage_mcse=row.coverage_mcse, mean_width=row.mean_length,
                             mean_width_mcse=mean_mcse, median_width=row.median_length,
                             median_width_mcse=median_mcse,
                             infinite_percent=100 * row.infinite_fraction,
                             instrument_arms=design["instrument_arms"],
                             bootstrap_seed=seed + offset, bootstrap_replicates=20000))
    return render("empirical_mar", cells, rows, coverages)


def natural_tables(inputs):
    summaries, cells, rows, policy = {}, [], [], []
    for dataset in DATASETS:
        folder = natural_folder(dataset)
        summary = inputs.csv(folder / "summary.csv")
        summaries[dataset] = summary
        design = inputs.json(folder / "design.json")
        # split.csv lists every member of the one-person-per-family cohort.
        # Join these IDs to their original outcomes; missing values stay missing.
        split = inputs.csv(folder / "split.csv")
        source = inputs.csv(design["source"])
        cohort = split[["row_id", "observed"]].merge(
            source[["row_id", "outcome"]], on="row_id", how="left", validate="one_to_one")
        if len(cohort) != design["n"]:
            raise ValueError(f"Cohort size mismatch for {dataset}")
        np.testing.assert_array_equal(cohort.observed.astype(bool), cohort.outcome.notna())
        observed = cohort.loc[cohort.observed.astype(bool), "outcome"].to_numpy()
        if len(observed) != design["n_observed"]:
            raise ValueError(f"Observed cohort size mismatch for {dataset}")
        spans = [float(np.diff(np.quantile(observed, [p, 1-p]))[0]) for p in (.05, .025)]
        row = row90(summary, "Policy IV")
        cells += [cell(spans[0], 2), cell(spans[1], 2),
                  cell(row.mean_length, 2), cell(100 * row.infinite_fraction, 1)]
        rows.append(dict(dataset=LABELS[dataset], dataset_id=dataset, method="Policy IV",
                         target=TARGET, central_90_span=spans[0], central_95_span=spans[1],
                         interval_width=row.mean_length,
                         infinite_percent=100 * row.infinite_fraction,
                         n_cohort=len(cohort), n_observed=len(observed),
                         n_missing_targets=int(row.n_missing_targets)))
        raw = float(design["calibration_diagnostics"]["Policy IV"]["identified_mass_eif"])
        projected = float(np.clip(raw, 0., 1.))
        policy.append(dict(dataset=LABELS[dataset], dataset_id=dataset,
                           target=TARGET, raw_policy_mass=raw,
                           projected_policy_mass=projected,
                           limiting_miscoverage=1. - projected,
                           displayed_policy_mass=f"{projected:.3f}"))
    application = render("empirical_application", cells, rows)
    cells, rows = [], []
    for method in ("MAR population CP (X)", "MAR population CP (X,Z)",
                   "Best-case CP (X,Z)", "Worst-case CP (X,Z)"):
        for dataset in DATASETS:
            row = row90(summaries[dataset], method)
            cells += [cell(row.mean_length, 2), cell(row.median_length, 2),
                      cell(100 * row.infinite_fraction, 1)]
            rows.append(dict(dataset=LABELS[dataset], dataset_id=dataset, method=method,
                             target=TARGET, mean_width=row.mean_length,
                             median_width=row.median_length,
                             infinite_percent=100 * row.infinite_fraction))
    comparisons = render("empirical_application_comparisons", cells, rows)
    return [application, comparisons], pd.DataFrame(policy)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments-dir", type=Path, default=ROOT / "experiments",
                        help="Directory containing data/ and results/ (default: bundled experiments/)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "tables",
                        help="Destination for .tex, .csv and verification JSON (default: tables/)")
    parser.add_argument("--check-reference", action="store_true",
                        help="Require exact agreement with every rounded draft table number")
    args = parser.parse_args()
    inputs = Inputs(args.experiments_dir.resolve())
    tables = [masking_table(inputs), mar_table(inputs)]
    natural, policy = natural_tables(inputs)
    tables.extend(natural)
    mismatches = [table["name"] for table in tables if not table["matches_reference"]]
    if args.check_reference and mismatches:
        raise ValueError("Results differ from the draft tables: " + ", ".join(mismatches))
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    outputs = {}
    for table in tables:
        tex = out / (table["name"] + ".tex")
        csv = out / (table["name"] + ".csv")
        tex.write_text(table["text"], encoding="utf-8", newline="\n")
        table["frame"].to_csv(csv, index=False, lineterminator="\n")
        outputs[tex.name], outputs[csv.name] = sha256(tex), sha256(csv)
    policy_path = out / "policy_mass.csv"
    policy.to_csv(policy_path, index=False, lineterminator="\n")
    outputs[policy_path.name] = sha256(policy_path)
    audit = dict(generated=True, reference_check_requested=args.check_reference,
                 all_tables_match_draft=not mismatches,
                 generator_sha256=sha256(Path(__file__)),
                 input_sha256=inputs.sources, output_sha256=outputs,
                 tables=[{key: value for key, value in table.items()
                          if key not in ("text", "frame")} for table in tables],
                 notes=["Widths use original outcome units; infinite widths remain infinite.",
                        "Undefined MCSEs (infinite widths or nonfinite bootstrap replicates) are shown as -- and left empty in CSVs.",
                        "MAR median MCSE uses the draft's fixed 20,000-replicate bootstrap.",
                        "MAR instrument category counts are read from saved designs into empirical_mar.csv.",
                        "Policy mass is a learned-policy diagnostic, not a coverage certificate."])
    (out / "table_verification.json").write_text(
        json.dumps(audit, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    print(f"Wrote four LaTeX tables, four data CSVs and policy_mass.csv to {out}")
    print("All printed numbers match the draft." if not mismatches
          else "Tables differ from the draft: " + ", ".join(mismatches))


if __name__ == "__main__":
    main()
