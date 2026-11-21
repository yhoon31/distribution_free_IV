
from pathlib import Path
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
PROJECT = BASE.parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    audit_path = BASE/'results/nhanes_five_categories/validation.json'
    if not json.loads(audit_path.read_text())['passed']:
        raise ValueError('Five-category NHANES experiments must pass validation before publication')
    families = [('anchored', 'Setting 1', '#0072B2', 'o', '-'),
                ('crossing', 'Setting 2', '#D55E00', 's', '--')]
    gammas = [.90, .95, .99, 1.]
    provenance = []
    fig, axes = plt.subplots(1, 2, figsize=(9., 3.6), sharey=True)
    for ax, dataset, title in zip(axes, ['heckman', 'nhanes_height'],
                                  ['FES wages', 'NHANES III reported height']):
        for family, label, color, marker, style in families:
            values, errors = [], []
            if dataset == 'heckman':
                root = BASE/'results/observed_gamma'/('primary' if family == 'anchored' else 'crossing_refined')
            else:
                root = BASE/'results/nhanes_five_categories/primary'
            for gamma in gammas:
                scenario = f'{family}_g{round(100*gamma):03d}'
                folder = root/dataset/scenario
                design = json.loads((folder/'design.json').read_text())
                frame = pd.read_csv(folder/'summary.csv')
                row = frame[(frame.method == 'Policy IV') & np.isclose(frame.target, .9)]
                expected = dict(seed=20260919, trials=200, trees=120, min_leaf=20,
                    grid_size=801 if family == 'anchored' else 1601,
                    n_train=2000, pool_size=2000)
                assert len(row) == 1 and row.iloc[0].trials == 200
                assert all(design['arguments'][key] == val for key,val in expected.items())
                assert design['instrument_arms'] == (3 if dataset == 'heckman' else 5)
                row = row.iloc[0]
                values.append(row.coverage)
                errors.append(1.96*row.coverage_mcse)
                provenance.append(dict(dataset=dataset, scenario=scenario,
                    summary=str((folder/'summary.csv').relative_to(PROJECT)),
                    summary_sha256=digest(folder/'summary.csv'),
                    design_sha256=digest(folder/'design.json'),
                    coverage=float(row.coverage), coverage_mcse=float(row.coverage_mcse),
                    infinite_fraction=float(row.infinite_fraction)))
            ax.errorbar(gammas, values, yerr=errors, label=label, color=color,
                marker=marker, linestyle=style, linewidth=1.5, markersize=4.5, capsize=3)
        ax.axhline(.9, color='0.4', linestyle=':', linewidth=1.1)
        ax.set(title=title, xlabel=r'Response-envelope parameter $g$', xticks=gammas,
               ylim=(.84, 1.013), yticks=[.85, .90, .95, 1.])
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.grid(alpha=.16)
    axes[0].set_ylabel('Coverage')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False)
    fig.tight_layout(rect=(0, .09, 1, 1))
    out = PROJECT/'figures/plot_empirical_masking.pdf'
    fig.savefig(out, bbox_inches='tight', metadata={'CreationDate': None})
    plt.close(fig)
    (BASE/'empirical_height_figure_verification.json').write_text(json.dumps(dict(
        passed=True, figure=str(out.relative_to(PROJECT)), figure_sha256=digest(out),
        generator_sha256=digest(Path(__file__)), audit_sha256=digest(audit_path),
        conditional_mcse=True, infinite_intervals_included_in_coverage=True,
        summaries=provenance), indent=2)+'\n')
    print(f'Verified {len(provenance)} summaries; wrote {out}')


if __name__ == '__main__':
    main()
