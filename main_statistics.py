"""Group statistics (under construction).

Reads the tidy metrics database produced by ``main_analysis.py``
(``data/group/database.csv``) and will run group-level statistical tests
comparing conditions (8bit 0/1/2) and contexts for each channel x metric.

Planned analyses
----------------
- Mixed-effects models (condition/context as fixed factors, subject as
  random factor) per channel and component.
- Pairwise contrasts with cluster- or permutation-based p-values.
- Distribution / normality checks before parametrising.

Usage
-----
    python main_statistics.py [--metrics data/group/database.csv]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from tms_eeg.config.settings import ProjectConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--metrics",
        default=None,
        help="Path to the tidy metrics CSV (default: config.paths.metrics_csv)",
    )
    args = parser.parse_args()

    metrics_path = args.metrics or str(ProjectConfig().paths.metrics_csv)
    if not Path(metrics_path).exists():
        parser.error(f"Metrics file not found: {metrics_path} — run main_analysis.py first.")

    raise NotImplementedError(
        "main_statistics.py is a stub. Planned: mixed-effects / permutation tests "
        "across conditions and contexts for each channel x metric."
    )


if __name__ == "__main__":
    main()