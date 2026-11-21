# Empirical data experiments for the JRSSB draft

This folder collects the FES log-wage and NHANES III reported-height experiments
used by `JRSSB/jrssb.tex` and the empirical sections of
`JRSSB/supplements.tex`, reviewed on 2026-11-20. It includes the fitting code,
prepared inputs, source preparation/downloaders, saved fits, validation code,
and generators for every empirical table and figure. The original project
files and frozen fitting records are preserved.

Keep the **internal directory structure** when copying this folder. The inner
`experiments/` is intentional: the original code resolves its data/results there
and its figure in the sibling `figures/`. No files outside this bundle are needed
to rebuild the empirical outputs from the included data/results.

## Setup and reproduce the draft outputs

Use Python **3.12** (the recorded fits used 3.12.3). From this folder:

```sh
python -m venv .venv
```

Activate with `.venv\Scripts\Activate.ps1` in PowerShell or
`source .venv/bin/activate` on macOS/Linux, then:

```sh
python -m pip install -r requirements.txt
python -B reproduce.py outputs
```

This validates the saved fits, recomputes the table entries and bootstrap
standard errors, checks every displayed table number against the draft,
and writes:

| Draft artifact | Generated file | Input experiments |
| --- | --- | --- |
| `fig:empirical-masking` | `figures/plot_empirical_masking.pdf` | FES/NHANES Settings 1 and 2, g = .90, .95, .99, 1 |
| `tab:empirical-masking` | `tables/empirical_masking.tex` and `.csv` | Setting 1, g = 1, nominal coverage .9 |
| `tab:empirical-application` | `tables/empirical_application.tex` and `.csv` | Natural missingness, policy intervals and observed spans |
| Supplement `tab:empirical-mar` | `tables/empirical_mar.tex` and `.csv` | MAR controls, nominal coverage .9 |
| Supplement `tab:empirical-application-comparisons` | `tables/empirical_application_comparisons.tex` and `.csv` | Natural missingness, comparison methods |
| Learned-policy mass in the text and S6.3 | `tables/policy_mass.csv` | Primary natural fits (.910 for FES, .959 for NHANES) |
| Supporting policy-mass diagnostics | `experiments/results/projected_policy_mass/summary.csv` | Primary and NHANES grid/seed checks |

The `.tex` files are table fragments for the existing manuscripts, with their
labels and captions. They use `booktabs` and `xcolor` (with the `table` option),
already loaded by the drafts. Copy the PDF to `JRSSB/plot_empirical_masking.pdf`
when updating the manuscript figure. The table generator keeps the draft layout
and computes its numerical contents from results; it does not simply copy the
old numbers. CSVs retain unrounded values. Infinity is kept as infinity.

`templates/` contains the four reference table layouts and their source hashes;
`tables/table_verification.json` records the generated values and checksums.
The full manuscript, simulation code/figures, Overleaf archives, and portable
LaTeX binaries are not required for these empirical outputs.

## Refit the experiments

The saved-results command above does not refit models. For a complete rerun:

```sh
python -B reproduce.py rerun --work-dir ../empirical-rerun --workers 2
```

The work directory must be new and outside this reference bundle. The command
copies the code and data, retains the fixed FES cutpoint specification, and
starts a clean results tree. It fits 10 FES cases and 19 NHANES cases, then
validates and generates the tables/figure. The NHANES jobs include the grid and
seed checks supporting the saved-results audit. Fits can take substantial time;
`--workers` controls concurrent primary jobs. To inspect the workspace first:

```sh
python -B reproduce.py rerun --work-dir ../empirical-rerun --prepare-only
python -B ../empirical-rerun/reproduce.py fit --workers 2
```

Inspect the exact primary job specifications without fitting:

```sh
python -B experiments/run_fes_for_draft.py --dry-run
python -B experiments/run_nhanes_five_categories.py --dry-run
```

The original saved NHANES plans contain absolute Windows output paths.
Do not use `--resume` on those relocated reference records: their original
provenance is deliberately unchanged. The wrapper creates fresh plans instead.
The top-level fresh-fit command refuses to restart an already-started workspace;
for an interrupted run, either use a new work directory or inspect the original
runners' resume behavior explicitly. It never deletes previous fits.

Fresh outputs are checked for internal consistency. Their agreement with the
historical draft can be checked separately with
`python -B reproduce.py outputs` from the fresh work directory. Use
`python -B reproduce.py outputs --allow-changed-results` to rebuild internally
consistent results that differ from the reference numbers.

## Files to include in a code repository

Include the root Python entry points, `requirements.txt`, this README,
`.gitattributes`, `.gitignore`, `templates/`, all `experiments/*.py`, and the
preparation scripts and provenance documents under `experiments/data/`.
Also retain `experiments/results/heckman/natural/design.json`: it contains the
fixed natural-FES instrument cutpoints and is an **input**, not a dispensable
old output. `.gitattributes` prevents Git line-ending conversion from invalidating
the recorded source hashes. `source_manifest.json` identifies copied files,
their original locations, and the few adapted/new files.

For immediate reproduction without fitting or downloading, this local bundle
also includes prepared data, cached source files, and the 29 saved empirical
fits. Keep `figures/` and `tables/` if readers should see the published outputs.
Row-level result files such as `split.npz`, `split.csv`, and `intervals.csv` are
needed for the independent saved-fit audits, beyond the aggregate summaries.

**FES data provenance:** `experiments/data/heckman/SOURCES.md` records that no
explicit redistribution license was identified for the original authors'
replication archive. Its presence in this local bundle does not resolve that
issue. The code can be published with the downloader and source attribution;
decide separately whether to publish FES microdata or row-level derived results.
If those are omitted, readers must reconstruct the inputs and refit before
using the saved-results workflow. The NHANES source manifests and downloaders
record the CDC public-use sources. This packaging task did not publish anything.

## Code and data map

| Files | Purpose |
| --- | --- |
| `experiments/iv_methods.py` | Prediction and calibration methods |
| `experiments/run_experiments.py` | Shared data loading, fitting helpers, metrics |
| `experiments/feedback_simulations.py` | MAR masking and shared response mechanisms |
| `experiments/run_observed_gamma.py` | Native-instrument Settings 1/2 masking |
| `experiments/run_fes_for_draft.py` | Exactly the 8 FES masking, 1 MAR, 1 natural cases |
| `experiments/estimate_natural_gamma.py`, `run_feedback_application.py` | Natural cohorts, instruments, intervals |
| `experiments/run_nhanes_five_categories.py` | NHANES primary fits and grid/seed checks |
| `experiments/run_nhanes_five_mar_grid.py` | Additional 3,201-node MAR diagnostic |
| `experiments/make_empirical_height_figure.py` | Draft empirical figure |
| `render_tables.py` | Four draft tables and primary policy masses |
| `experiments/project_policy_mass.py` | Policy-mass diagnostics |
| `experiments/validate_*.py`, `check_observed_gamma_grid.py`, `validation_helpers.py` | Independent source/result checks |
| `experiments/test_*.py` | Existing method and design tests |
| `experiments/data/heckman/` | FES data, downloader, preparation, attribution |
| `experiments/data/nhanes_height/` | Reported-height cohort and preparation |
| `experiments/data/nhanes/` | Original sources and outcome-blind interviewer selection needed by the height cohort |

Some shared helpers retain historical dataset names and generic CLI defaults.
Use the bundle entry points above to select the draft experiments. NHANES
parent files retain hemoglobin columns because the height cohort reuses their
fixed interviewer selection and covariates; the draft outcome is reported
height in inches. The current native instrument has three categories for FES
and five original interviewer identities for NHANES. **The MAR control uses
five randomized categories for both datasets.**

## Reconstruct the input data

Prepared inputs are included, so these commands are optional. Run them in a
fresh work directory if comparing reconstructed data with the frozen inputs:

```sh
python -B experiments/data/heckman/prepare_heckman.py
python -B experiments/data/nhanes/prepare_nhanes.py
python -B experiments/data/nhanes/audit_instrument.py
python -B experiments/data/nhanes_height/prepare_height.py
```

The FES and NHANES preparation scripts download missing original sources.
`requests` is used for downloading; `pyreadstat` is needed only when converting
the original Stata file without the cached `wagedata.csv`. `runtime_support.py`
allows the installed converter to be used without the optional Overleaf vendor
archives. `PyMuPDF` in the original requirements is optional PDF inspection
support, not needed for inference or figure/table generation.

To reconstruct only height in isolation from the cached parent cohort:

```sh
python -B experiments/data/nhanes_height/prepare_height.py --output-dir ../height-check
```

## Verification and alignment notes

```sh
python -B -m unittest discover -s experiments -p 'test_*.py'
python -B experiments/validate_nhanes_five_categories.py
python -B experiments/validate_draft_results.py --check-reference
```

Packaging verification passed 58 existing tests, all 19 NHANES saved-fit checks,
all four table comparisons, and exact reconstruction of the five prepared
height CSV/JSON files. Details are in `verification/`. A complete 29-case model
refit was not performed as part of packaging.

Two manuscript alignment edits are marked with `\textcolor{red}{...}` in
`JRSSB/supplements.tex`, Section S6.1: FES MAR has **five** randomized categories,
and its response covariate index uses the **pilot estimator** `mu_0`. These edits
describe the existing experiment; the table values remain unchanged.

All original fitting modules are byte-for-byte copies. The bundled NHANES and
natural-application validators have small path-portability edits; the new
JRSSB validator, FES scheduler, table generator, and reproduction wrapper avoid
dependencies on `main.tex`, synthetic figures, or old output locations.

The original eight FES masking fits record an earlier
`feedback_simulations.py` hash (`f9929ca2...`) whose source version was not found
in the retained archives. The retained helper (`5b5134de...`) matches the later
NHANES/MAR manifests. Validation acknowledges only that exact known mismatch
and independently reconstructs source rows, splits, masks, and summaries.
This bundle does not claim an exact legacy source match or guarantee bitwise
identical new FES fits. Original hashes and numerical results are preserved.

The NHANES diagnostics retain the documented numerical sensitivity at the
supplemental 80% target; the three published 90% MAR methods are stable across
the recorded grids. Coverage on genuinely missing outcomes is unobservable;
natural-application outputs report interval widths and policy diagnostics.
