"""Group-level analysis utilities for collecting and aggregating metrics."""

from __future__ import annotations

import pandas as pd
from pathlib import Path


class MetricsCollector:
    """Collects metrics in tidy (long) format for statistical analysis."""

    def __init__(self):
        self.rows: list[dict] = []

    def add_row(
        self,
        subject: str,
        analysis_type: str,
        condition: str,
        channel: str,
        component: str,
        metric: str,
        value: float,
    ) -> None:
        """Add a single metric row."""
        self.rows.append({
            "subject": subject,
            "analysis_type": analysis_type,
            "condition": condition,
            "channel": channel,
            "component": component,
            "metric": metric,
            "value": value,
        })

    def to_dataframe(self) -> pd.DataFrame:
        """Convert collected rows to a tidy DataFrame."""
        return pd.DataFrame(self.rows)

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

        if export_enabled:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(path, index=False)
            print(f"\nMetrics exported to: {path}")
            print(f"Total rows: {len(df)}")
            print(f"Columns: {list(df.columns)}")

        return df
