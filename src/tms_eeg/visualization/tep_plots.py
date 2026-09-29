"""TEP (TMS-Evoked Potential) visualization class."""

import mne
from typing import List, Optional, Dict
import numpy as np
import matplotlib.pyplot as plt
from tms_eeg.io.writer import save_figure

class TEPPlotter:
    """Plots TEP-related visualizations from epochs or evoked objects."""

    def __init__(
        self,
        config,
        xlim: tuple = None,
        topo_times: Optional[List[float]] = None,
        joint_times: Optional[List[float]] = None,
        roi_picks: Optional[List[str]] = None,
        writer=None,
    ):
        self.config = config
        self.xlim = xlim or config.plot_tep_xlim
        self.topo_times = topo_times or config.plot_tep_topo_times
        self.joint_times = joint_times or config.plot_tep_joint_times
        self.roi_picks = roi_picks or config.plot_tep_roi_channels
        self.writer = writer

    def _save_figure(self, fig, name: str, condition: str):
        """Save figure if io_save_figs is enabled in config."""
        save_figure(fig, f"tep_{name}_{condition}", self.config, self.writer)

    def plot_evoked_by_symbol(
        self,
        epochs: mne.Epochs,
        picks: Optional[List[str]] = None,
        xlim: tuple = None,
        ylim: tuple = None,
    ):
        """
        Plota overlay dos evokeds médios por símbolo (8bit 0, 1, 2) para canais selecionados.

        Parameters
        ----------
        epochs : mne.Epochs
            Objeto epochs filtrado com event IDs 1, 2, 3.
        picks : list[str], optional
            Canais a plotar. Default: roi_picks do config.
        xlim : tuple, optional
            Janela temporal em segundos. Default: tep_xlim do config.
        ylim : tuple, optional
            Limites do eixo Y em µV. Default: None (autoescala).
        """
        if not self.config.plot_analysis:
            return

        picks = picks or self.roi_picks
        xlim_ms = tuple(v * 1e3 for v in (xlim or self.xlim))

        event_to_symbol = self.config.analysis_event_to_symbol  # {1: 0, 2: 1, 3: 2}

        # Mapa inverso: event_id -> nome da condição (apenas dos eids relevantes)
        eid_to_cond = {v: k for k, v in epochs.event_id.items()}

        # Calcula evokeds por símbolo, já ordenados por símbolo (0, 1, 2)
        evokeds = {}
        for eid, symbol in sorted(event_to_symbol.items(), key=lambda x: x[1]):
            cond = eid_to_cond.get(eid)
            if cond is None:
                continue
            evokeds[f"8bit {symbol}"] = epochs[cond].average(picks="eeg")

        if not evokeds:
            print("Nenhuma condição encontrada para plotar.")
            return

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]
        labels = list(evokeds.keys())
        times_ms = next(iter(evokeds.values())).times * 1e3  # eixo X compartilhado

        # Pré-extrai os dados de todos os canais de uma vez (evita .copy().pick() em loop)
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

            self._save_figure(fig, f"mean_tep_{ch}", "all_conditions")
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
        Compara TEPs de contextos selecionados (overlay no mesmo gráfico).

        Parameters
        ----------
        context_epochs : dict
            {context_name: mne.Epochs} retornado pelo ContextMapper.
        contexts : list, optional
            Quais contextos plotar. Default: ["ctx_01", "ctx_11", "ctx_21"].
        xlim : tuple, optional
            Janela temporal em segundos.
        picks : list, optional
            Canais a plotar. Default: roi_picks do config.
        """
        if not self.config.plot_analysis:
            return
            
        contexts = contexts or ["ctx_01", "ctx_11", "ctx_21"]
        xlim = xlim or self.xlim
        xlim_ms = (xlim[0] * 1e3, xlim[1] * 1e3)
        picks = picks or self.roi_picks
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd"]

        # Gera evokeds
        evokeds = {}
        for ctx in contexts:
            if ctx not in context_epochs:
                print(f"⚠ Contexto '{ctx}' não encontrado, pulando.")
                continue
            evokeds[ctx] = context_epochs[ctx].average(picks="eeg")

        if len(evokeds) < 2:
            print("Menos de 2 contextos disponíveis, nada a comparar.")
            return

        # --- 1) Overlay por canal ROI ---
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
            self._save_figure(fig, f"ctx_compare_{ch}", "ctx_01_11_21")
            plt.show()
            plt.close(fig)

        # --- 2) GFP comparison ---
        fig, ax = plt.subplots(figsize=(9, 4))
        for i, (ctx, evk) in enumerate(evokeds.items()):
            gfp = np.std(evk.data, axis=0)
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
        self._save_figure(fig, "ctx_compare_gfp", "ctx_01_11_21")
        plt.show()
        plt.close(fig)

        # --- 3) Joint topomap por contexto ---
        for ctx, evk in evokeds.items():
            fig = evk.plot_joint(
                times=self.joint_times,
                title=f"TEP — {ctx} (n={len(context_epochs[ctx])})",
                ts_args=dict(xlim=xlim),
            )
            self._save_figure(fig, f"ctx_joint_{ctx}", "ctx_01_11_21")
            
    def plot_context_temporal_comparison(
        self,
        context_epochs: Dict[str, mne.Epochs],
        n_splits: int = 2,
        split_labels: Optional[List[str]] = None,
        contexts: Optional[List[str]] = None,
        xlim: tuple = None,
        picks: Optional[List[str]] = None,
    ):
        """
        Compara TEPs de frações temporais (ex: 1ª vs 2ª metade) por contexto.

        Parameters
        ----------
        context_epochs : dict
            {context_name: mne.Epochs}.
        n_splits : int
            Número de divisões (2 = metades, 3 = terços, etc.).
        split_labels : list[str], optional
            Rótulos das frações. Default: ["1/N", "2/N", ...].
        contexts : list, optional
            Contextos a plotar. Default: todos disponíveis.
        xlim : tuple, optional
            Janela temporal em segundos.
        picks : list, optional
            Canais ROI. Default: roi_picks do config.
        """
        if not self.config.plot_analysis:
            return
            
        contexts = contexts or list(context_epochs.keys())
        xlim = xlim or self.xlim
        xlim_ms = (xlim[0] * 1e3, xlim[1] * 1e3)
        picks = picks or self.roi_picks

        if split_labels is None:
            split_labels = [f"{i+1}/{n_splits}" for i in range(n_splits)]
        if len(split_labels) != n_splits:
            raise ValueError("split_labels deve ter comprimento igual a n_splits.")

        cmap = plt.colormaps["tab10"]

        for ctx in contexts:
            if ctx not in context_epochs:
                print(f"⚠ Contexto '{ctx}' não encontrado, pulando.")
                continue

            epochs = context_epochs[ctx]
            n_total = len(epochs)
            if n_total < n_splits:
                print(f"⚠ '{ctx}' tem apenas {n_total} epoch(s), impossível dividir em {n_splits}.")
                continue

            # --- Dividir epochs em N frações (ordem temporal preservada) ---
            indices = np.array_split(np.arange(n_total), n_splits)
            split_evokeds = []
            split_counts = []
            for idx in indices:
                ep_split = epochs[idx]
                split_evokeds.append(ep_split.average(picks="eeg"))
                split_counts.append(len(idx))

            tag = f"{ctx}_splits{n_splits}"

            # --- 1) Overlay por canal ROI ---
            for ch in picks:
                fig, ax = plt.subplots(figsize=(9, 4))
                for i, (evk, label, n_ep) in enumerate(
                    zip(split_evokeds, split_labels, split_counts)
                ):
                    data = evk.copy().pick([ch]).data.squeeze()
                    ax.plot(
                        evk.times * 1e3,
                        data * 1e6,
                        label=f"{label} (n={n_ep})",
                        color=cmap(i),
                        linewidth=1.4,
                    )
                ax.set_xlim(xlim_ms)
                ax.set_xlabel("Time (ms)")
                ax.set_ylabel("Amplitude (µV)")
                ax.set_title(f"Temporal Split — {ctx} — {ch}")
                ax.legend(fontsize=9)
                ax.axhline(0, color="gray", ls="--", lw=0.5)
                ax.axvline(0, color="gray", ls="--", lw=0.5)
                fig.tight_layout()
                self._save_figure(fig, f"temporal_{ch}", tag)
                plt.show()
                plt.close(fig)

            # --- 2) GMFP comparison ---
            fig, ax = plt.subplots(figsize=(9, 4))
            for i, (evk, label, n_ep) in enumerate(
                zip(split_evokeds, split_labels, split_counts)
            ):
                gfp = np.std(evk.data, axis=0)
                ax.plot(
                    evk.times * 1e3,
                    gfp * 1e6,
                    label=f"{label} (n={n_ep})",
                    color=cmap(i),
                    linewidth=1.4,
                )
            ax.set_xlim(xlim_ms)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("GMFP (µV)")
            ax.set_title(f"GMFP — Temporal Split — {ctx}")
            ax.legend(fontsize=9)
            ax.axhline(0, color="gray", ls="--", lw=0.5)
            ax.axvline(0, color="gray", ls="--", lw=0.5)
            fig.tight_layout()
            self._save_figure(fig, "temporal_gmfp", tag)
            plt.show()
            plt.close(fig)

            # --- 3) LMFP comparison (canais ROI) ---
            fig, ax = plt.subplots(figsize=(9, 4))
            for i, (evk, label, n_ep) in enumerate(
                zip(split_evokeds, split_labels, split_counts)
            ):
                evk_roi = evk.copy().pick(picks)
                lmfp = np.std(evk_roi.data, axis=0)
                ax.plot(
                    evk.times * 1e3,
                    lmfp * 1e6,
                    label=f"{label} (n={n_ep})",
                    color=cmap(i),
                    linewidth=1.4,
                )
            ax.set_xlim(xlim_ms)
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("LMFP (µV)")
            ax.set_title(f"LMFP ({', '.join(picks)}) — Temporal Split — {ctx}")
            ax.legend(fontsize=9)
            ax.axhline(0, color="gray", ls="--", lw=0.5)
            ax.axvline(0, color="gray", ls="--", lw=0.5)
            fig.tight_layout()
            self._save_figure(fig, "temporal_lmfp", tag)
            plt.show()
            plt.close(fig)

            # --- 4) Joint topomap por fração ---
            for i, (evk, label, n_ep) in enumerate(
                zip(split_evokeds, split_labels, split_counts)
            ):
                fig = evk.plot_joint(
                    times=self.joint_times,
                    title=f"{ctx} — {label} (n={n_ep})",
                    ts_args=dict(xlim=xlim),
                )
                self._save_figure(fig, f"temporal_joint_{label.replace('/', 'of')}", tag)