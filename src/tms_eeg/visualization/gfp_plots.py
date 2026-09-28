# gfp_plots.py

"""Plots for GMFP and LMFP curves."""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Optional, List, Tuple

from tms_eeg.io.writer import save_figure


class MFPPlotter:
    """Plots GMFP and LMFP curves with optional time-window shading."""

    def __init__(
        self,
        times: np.ndarray,
        config=None,
        writer=None,
    ):
        """
        Parameters
        ----------
        times : np.ndarray
            Time vector from epochs (in seconds).
        config : ProjectConfig, optional
        writer : Writer, optional
        """
        self.times = times
        self.times_ms = times * 1e3
        self.config = config
        self.writer = writer
        self.xlim = (
            config.plots.tep_xlim if config else (-0.01, 0.2)
        )
        self.xlim_ms = (self.xlim[0] * 1e3, self.xlim[1] * 1e3)

    # ------------------------------------------------------------------ #
    #  Main API
    # ------------------------------------------------------------------ #

    def plot_gmfp_lmfp(
        self,
        gmfp: Dict[str, np.ndarray],
        lmfp: Dict[str, np.ndarray],
        time_windows: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """
        Plot GMFP and LMFP side by side for all conditions.
        One figure per condition with 2 subplots.
        """
        if self.config and not self.config.plots.analysis_plots:
            return
            
        conditions = list(gmfp.keys())

        for cond in conditions:
            fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

            self._plot_single_curve(
                axes[0], gmfp[cond], cond,
                title=f"GMFP — {cond}",
                ylabel="GMFP (µV)",
                color="#1f77b4",
                time_windows=time_windows,
            )
            self._plot_single_curve(
                axes[1], lmfp[cond], cond,
                title=f"LMFP — {cond}",
                ylabel="LMFP (µV)",
                color="#d62728",
                time_windows=time_windows,
            )

            fig.tight_layout()
            self._save_figure(fig, "gmfp_lmfp", cond)
            plt.show()
            plt.close(fig)

    def plot_overlay(
        self,
        mfp_data: Dict[str, np.ndarray],
        label: str = "GMFP",
        time_windows: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        """
        Overlay all conditions on a single plot for GMFP or LMFP.
        """
        if self.config and not self.config.plots.analysis_plots:
            return
            
        fig, ax = plt.subplots(figsize=(10, 5))

        for cond, curve in mfp_data.items():
            ax.plot(self.times_ms, curve * 1e6, label=cond, linewidth=1.5)

        if time_windows:
            self._add_time_window_shading(ax, time_windows)

        ax.set_xlim(self.xlim_ms)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel(f"{label} (µV)")
        ax.set_title(f"{label} — All Conditions")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.legend()
        fig.tight_layout()

        self._save_figure(fig, f"{label.lower()}_overlay", "all_conditions")
        plt.show()
        plt.close(fig)

    # ------------------------------------------------------------------ #
    #  Internal
    # ------------------------------------------------------------------ #

    def _plot_single_curve(
        self,
        ax,
        curve: np.ndarray,
        condition: str,
        title: str,
        ylabel: str,
        color: str,
        time_windows: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        ax.plot(self.times_ms, curve * 1e6, color=color, linewidth=1.5)

        if time_windows:
            self._add_time_window_shading(ax, time_windows)

        ax.set_xlim(self.xlim_ms)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
        ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)

    @staticmethod
    def _add_time_window_shading(
        ax,
        time_windows: Dict[str, Tuple[float, float]],
    ):
        """Add semi-transparent vertical spans for each TEP component window."""
        colors = plt.cm.Set2(np.linspace(0, 1, len(time_windows)))
        for (comp, (t0, t1)), color in zip(time_windows.items(), colors):
            ax.axvspan(
                t0 * 1e3, t1 * 1e3,
                alpha=0.15, color=color, label=comp,
            )
            ax.text(
                (t0 + t1) / 2 * 1e3,
                ax.get_ylim()[1] * 0.95,
                comp,
                ha="center", va="top",
                fontsize=7, fontstyle="italic",
                color=color * 0.7,  # slightly darker
            )

    def _save_figure(self, fig, name: str, condition: str):
        """Save figure if save_figs is enabled in config."""
        save_figure(self.config, fig, f"mfp_{name}_{condition}", self.writer)
