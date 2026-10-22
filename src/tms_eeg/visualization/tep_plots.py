"""TEP (TMS-Evoked Potential) visualization class."""

import mne
from typing import List, Optional, Dict
import matplotlib.pyplot as plt
from tms_eeg.analysis.features import mean_field_power
from tms_eeg.io.writer import save_figure

class TEPPlotter:
    """Plots TEP-related visualizations from epochs or evoked objects."""

    def __init__(
        self,
        config,
        xlim: tuple = None,
        joint_times: Optional[List[float]] = None,
        roi_picks: Optional[List[str]] = None,
    ):
        self.config = config
        self.xlim = xlim or config.plot_tep_xlim
        self.joint_times = joint_times or config.plot_tep_joint_times
        self.roi_picks = roi_picks or config.plot_tep_roi_channels

    def plot_evoked_by_symbol(
        self,
        epochs: mne.Epochs,
        picks: Optional[List[str]] = None,
        xlim: tuple = None,
        ylim: tuple = None,
    ):
        """Plot average evokeds by symbol (8-bit 0, 1, 2) for selected channels.

        Parameters
        ----------
        epochs : mne.Epochs
            Filtered epochs with event IDs 1, 2, and 3.
        picks : list[str], optional
            Channels to plot. Defaults to ``roi_picks`` from the config.
        xlim : tuple, optional
            Time window in seconds. Defaults to ``tep_xlim`` from the config.
        ylim : tuple, optional
            Y-axis limits in µV. Defaults to None (automatic scaling).
        """
        if not self.config.plot_analysis:
            return

        picks = picks or self.roi_picks
        xlim_ms = tuple(v * 1e3 for v in (xlim or self.xlim))

        event_to_symbol = self.config.analysis_event_to_symbol

        # Map event IDs to condition names.
        eid_to_cond = {v: k for k, v in epochs.event_id.items()}

        # Compute evokeds in symbol order (0, 1, 2).
        evokeds = {}
        for eid, symbol in sorted(event_to_symbol.items(), key=lambda x: x[1]):
            cond = eid_to_cond.get(eid)
            if cond is None:
                continue
            evokeds[f"8bit {symbol}"] = epochs[cond].average(picks="eeg")

        if not evokeds:
            print("No conditions found to plot.")
            return

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        labels = list(evokeds.keys())
        times_ms = next(iter(evokeds.values())).times * 1e3

        # Cache channel data to avoid repeated copies and picks in the plotting loop.
        ch_data = {
            label: {
                ch: evk.data[evk.ch_names.index(ch)] * 1e6
                for ch in picks if ch in evk.ch_names
            }
            for label, evk in evokeds.items()
        }

        for ch in picks:
            fig, ax = plt.subplots(figsize=(9, 4))
            for i, label in enumerate(labels):
                data = ch_data[label].get(ch)
                if data is None:
                    continue
                ax.plot(
                    times_ms,
                    data,
                    label=label,
                    color=colors[i % len(colors)],
                    linewidth=1.4,
                )
            ax.set_xlim(xlim_ms)
            if ylim is not None:
                ax.set_ylim(ylim)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Amplitude (µV)")
            ax.set_title(f"Average TEP by Symbol — {ch}")
            ax.legend(fontsize=9)
            ax.axhline(0, color="gray", ls="--", lw=0.5)
            ax.axvline(0, color="gray", ls="--", lw=0.5)
            fig.tight_layout()
            plt.show()
            plt.close(fig)

    def plot_mean_tep(
        self,
        evokeds: Dict[str, mne.Evoked],
        xlim: tuple = None,
    ):
        if not self.config.plot_analysis:
            return
            
        xlim = xlim or self.xlim
        xlim_ms = (xlim[0] * 1e3, xlim[1] * 1e3)

        conditions = list(evokeds.keys())
        channels = list(evokeds[conditions[0]].ch_names)
        times = evokeds[conditions[0]].times

        for ch in channels:
            fig, ax = plt.subplots(figsize=(8, 4))
            for cond in conditions:
                signal = evokeds[cond].copy().pick([ch]).data.squeeze()
                ax.plot(times * 1e3, signal * 1e6, label=cond)

            ax.set_xlim(xlim_ms)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Amplitude (µV)")
            ax.set_title(f"Average TEP — {ch}")
            ax.legend()
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.5)
            ax.axvline(0, color="gray", linestyle="--", linewidth=0.5)
            fig.tight_layout()

            save_figure(fig, f"tep_mean_tep_{ch}_all_conditions", self.config)
            plt.show()
            plt.close(fig)
            
    def plot_context_comparison(
        self,
        context_epochs: Dict[str, mne.Epochs],
        contexts: Optional[List[str]] = None,
        xlim: tuple = None,
        picks: Optional[List[str]] = None,
    ):
        """
        Compare selected context TEPs as overlays on shared axes.

        Parameters
        ----------
        context_epochs : dict
            Mapping of context names to ``mne.Epochs`` from ``ContextMapper``.
        contexts : list, optional
            Contexts to plot. Defaults to ``["ctx_01", "ctx_11", "ctx_21"]``.
        xlim : tuple, optional
            Time window in seconds.
        picks : list, optional
            Channels to plot. Defaults to ``roi_picks`` from the config.
        """
        if not self.config.plot_analysis:
            return
            
        contexts = contexts or ["ctx_01", "ctx_11", "ctx_21"]
        xlim = xlim or self.xlim
        xlim_ms = (xlim[0] * 1e3, xlim[1] * 1e3)
        picks = picks or self.roi_picks
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        # Compute evokeds.
        evokeds = {}
        for ctx in contexts:
            if ctx not in context_epochs:
                print(f"⚠ Context '{ctx}' not found; skipping.")
                continue
            evokeds[ctx] = context_epochs[ctx].average(picks="eeg")

        if len(evokeds) < 2:
            print("Fewer than two contexts available; nothing to compare.")
            return

        # --- 1) Overlay by ROI channel ---
        for ch in picks:
            fig, ax = plt.subplots(figsize=(9, 4))
            for i, (ctx, evk) in enumerate(evokeds.items()):
                data = evk.copy().pick([ch]).data.squeeze()
                n_ep = len(context_epochs[ctx])
                ax.plot(
                    evk.times * 1e3,
                    data * 1e6,
                    label=f"{ctx} (n={n_ep})",
                    color=colors[i % len(colors)],
                    linewidth=1.4,
                )
            ax.set_xlim(xlim_ms)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Amplitude (µV)")
            ax.set_title(f"Context Comparison — {ch}")
            ax.legend(fontsize=9)
            ax.axhline(0, color="gray", ls="--", lw=0.5)
            ax.axvline(0, color="gray", ls="--", lw=0.5)
            fig.tight_layout()
            save_figure(fig, f"tep_ctx_compare_{ch}_ctx_01_11_21", self.config)
            plt.show()
            plt.close(fig)

        # --- 2) GFP comparison ---
        fig, ax = plt.subplots(figsize=(9, 4))
        for i, (ctx, evk) in enumerate(evokeds.items()):
            gfp = mean_field_power(evk)
            n_ep = len(context_epochs[ctx])
            ax.plot(
                evk.times * 1e3,
                gfp * 1e6,
                label=f"{ctx} (n={n_ep})",
                color=colors[i % len(colors)],
                linewidth=1.4,
            )
        ax.set_xlim(xlim_ms)
        ax.set_xlabel("Time (ms)")
        ax.set_ylabel("GFP (µV)")
        ax.set_title("Global Field Power — Context Comparison")
        ax.legend(fontsize=9)
        ax.axhline(0, color="gray", ls="--", lw=0.5)
        ax.axvline(0, color="gray", ls="--", lw=0.5)
        fig.tight_layout()
        save_figure(fig, "tep_ctx_compare_gfp_ctx_01_11_21", self.config)
        plt.show()
        plt.close(fig)

        # --- 3) Joint topomap per context ---
        for ctx, evk in evokeds.items():
            fig = evk.plot_joint(
                times=self.joint_times,
                title=f"TEP — {ctx} (n={len(context_epochs[ctx])})",
                ts_args=dict(xlim=xlim),
            )
            save_figure(fig, f"tep_ctx_joint_{ctx}_ctx_01_11_21", self.config)
