"""Group-level analysis utilities for collecting and aggregating metrics."""

from __future__ import annotations

import pandas as pd
from pathlib import Path


class MetricsCollector:
    """Collects metrics in tidy (long) format for statistical analysis."""

    COLUMNS = (
        "subject",
        "analysis_level",
        "analysis_type",
        "condition",
        "channel",
        "component",
        "metric",
        "value",
    )
    ANALYSIS_LEVELS = {"channel", "roi"}

    def __init__(self):
        self.rows: list[dict] = []

    def add_row(
        self,
        subject: str,
        analysis_level: str,
        analysis_type: str,
        condition: str,
        channel: str | None,
        component: str | None,
        metric: str,
        value: float,
    ) -> None:
        """Add one long-format metric observation.

        ``channel`` contains an electrode name for channel-level analyses or
        an ROI name for ROI-level analyses. It is ``None`` for metrics that
        are not associated with an individual channel/ROI.
        """
        if analysis_level not in self.ANALYSIS_LEVELS:
            raise ValueError(
                f"Invalid analysis_level {analysis_level!r}; expected one of "
                f"{sorted(self.ANALYSIS_LEVELS)}."
            )

        self.rows.append({
            "subject": subject,
            "analysis_level": analysis_level,
            "analysis_type": analysis_type,
            "condition": condition,
            "channel": channel,
            "component": component,
            "metric": metric,
            "value": value,
        })

    def to_dataframe(self) -> pd.DataFrame:
        """Convert collected rows to a tidy DataFrame."""
        return pd.DataFrame(self.rows, columns=self.COLUMNS)

    def export_csv(
        self,
        output_path: str = "data/group/all_subjects_metrics.csv",
        export_enabled: bool = True,
    ) -> pd.DataFrame | None:
        """Export collected rows to CSV if enabled.

        Args:
            output_path: Path to save the CSV file.
            export_enabled: Whether to actually export.

        Returns:
            The DataFrame if exported, None otherwise.
        """
        df = self.to_dataframe()

        if df.empty:
            raise ValueError(
                "No metrics have been collected; refusing to export an empty "
                "metrics CSV. Add feature results before calling export_csv()."
            )

        if export_enabled:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            print(f"\nMetrics exported to: {path}")
            print(f"Total rows: {len(df)}")
            print(f"Columns: {list(df.columns)}")

        return df
