"""Group-level plotting from the metrics database (data/group/*.csv).

Loads the tidy metrics CSV produced by ``main_analysis.py`` and renders the
group-level figures (box/strip plots per condition and context, P30 amplitude
summary) using ``GroupPlotter``.

Usage
-----
    python main_plotting.py [--metrics data/group/database.csv] [--output-dir results/group]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from tms_eeg.config.environment import setup_plotting_backend
from tms_eeg.config.settings import ProjectConfig
from tms_eeg.visualization.group_plots import GroupPlotter

# Channels shown in the per-component boxplots.
BOXPLOT_CHANNELS = ["C3", "FC1", "FC5", "CP1", "CP5", "C4"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        default=None,
        help="Path to the tidy metrics CSV (default: config.paths.metrics_csv)",
    )
    parser.add_argument("--output-dir", default="results/group")
    args = parser.parse_args()

    metrics_path = args.metrics or str(ProjectConfig().paths.metrics_csv)
    if not Path(metrics_path).exists():
        parser.error(f"Metrics file not found: {metrics_path} — run main_analysis.py first.")

    setup_plotting_backend()
    config = ProjectConfig()
    plotter = GroupPlotter(config, output_dir=args.output_dir)

    df = pd.read_csv(metrics_path)
    print(f"Loaded {len(df)} metric rows from {metrics_path}")

    # Peak-to-peak amplitudes per channel/component/condition.
    for component in ("N15-P30", "N15-P60", "N100-P180"):
        for channel in BOXPLOT_CHANNELS:
            plotter.plot_boxplots(
                df, groupby="condition", component=component, channel=channel)

    # GMFP / LMFP P30 amplitude summary.
    plotter.plot_p30_amplitude(df)


if __name__ == "__main__":
    main()