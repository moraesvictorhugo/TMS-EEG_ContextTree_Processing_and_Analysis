"""Group-level visualization for TMS-EEG metrics."""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path


class GroupPlotter:
    """Generate group-level plots from metrics database."""

    def __init__(self, config, output_dir: str = "results/group"):
        self.config = config
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        sns.set_theme(style="ticks", font_scale=1.2)

    def plot_boxplots(
        self,
        df: pd.DataFrame,
        groupby: str = "condition",
        component: str = "N15-P30",
        metric: str = "peak_to_peak_uV",
        channel: str = "C3",
    ):
        """
        Boxplot de uma métrica/componente para um canal específico.

        Parameters
        ----------
        df : pd.DataFrame
            Database com colunas: subject, analysis_type, condition,
            channel, component, metric, value.
        groupby : str
            'condition' ou 'context'.
        component : str
            Componente TEP (ex: 'N15-P30').
        metric : str
            Métrica (ex: 'peak_to_peak_uV').
        channel : str
            Canal EEG (ex: 'C3'). Cada ponto = 1 voluntário.
        """
        # --- Filtro ---
        analysis_type = "condition" if groupby == "condition" else "context"
        mask = (
            (df["analysis_type"] == analysis_type)
            & (df["component"] == component)
            & (df["metric"] == metric)
            & (df["channel"] == channel)
        )
        plot_df = df.loc[mask].copy()

        if plot_df.empty:
            print(f"[WARN] Sem dados para {analysis_type}/{component}/{metric}/{channel}")
            return

        plot_df["value"] = pd.to_numeric(plot_df["value"], errors="coerce")
        order = sorted(plot_df["condition"].unique(), key=str)

        # --- Plot ---
        fig, ax = plt.subplots(figsize=(8, 5))

        sns.boxplot(
            data=plot_df,
            x="condition",
            y="value",
            hue="condition",
            order=order,
            hue_order=order,
            palette="Set2",
            width=0.5,
            legend=False,
            ax=ax,
        )
        sns.stripplot(
            data=plot_df,
            x="condition",
            y="value",
            order=order,
            color="0.3",
            alpha=0.6,
            jitter=True,
            size=5,
            ax=ax,
        )

        metric_label = metric.replace("_", " ").replace("uV", "(µV)").replace("ms", "(ms)")
        ax.set_title(f"{component} — {channel} — {metric_label}", fontweight="bold")
        ax.set_xlabel("Contexto" if groupby == "context" else "Condição")
        ax.set_ylabel(metric_label.capitalize())
        sns.despine()  # remove as bordas superior e direita

        fig.tight_layout()

        # --- Salvar ou exibir ---
        if self.config.io_save_figs:
            fname = (
                f"boxplot_{groupby}_{channel}_{component}_{metric}"
                f".{self.config.plot_format}"
            )
            filepath = self.output_dir / fname
            fig.savefig(filepath, dpi=self.config.plot_dpi, bbox_inches="tight")
            print(f"[OK] Salvo: {filepath}")
        else:
            plt.show()

        plt.close(fig)
        
    def plot_p30_amplitude(self, df: pd.DataFrame):
        """Plot P30 peak amplitude for GMFP and LMFP across conditions and contexts."""
        mask = (
            (df["component"] == "P30") &
            (df["metric"] == "peak_amplitude_uV") &
            (df["channel"].isin(["GMFP", "LMFP"]))
        )
        data = df[mask].copy()

        for analysis in ["condition", "context"]:
            subset = data[data["analysis_type"] == analysis]
            if subset.empty:
                continue

            group_col = "condition"
            fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)

            for ax, channel in zip(axes, ["GMFP", "LMFP"]):
                ch_data = subset[subset["channel"] == channel]
                order = sorted(ch_data[group_col].unique())

                sns.boxplot(
                    data=ch_data, x=group_col, y="value",
                    hue=group_col, order=order, palette="Set2",
                    legend=False, ax=ax
                )
                sns.stripplot(
                    data=ch_data, x=group_col, y="value",
                    order=order, color="0.3",
                    dodge=False, alpha=0.6, jitter=True, size=5,
                    legend=False, ax=ax
                )

                ax.set_title(f"{channel} — P30 Amplitude")
                ax.set_ylabel("Amplitude (µV)" if ax == axes[0] else "")
                ax.set_xlabel(analysis.capitalize())

            sns.despine()
            fig.suptitle(f"P30 Peak Amplitude — by {analysis.capitalize()}", fontsize=14)
            fig.tight_layout()

            # --- Salvar ou exibir (controlado pelo config) ---
            if self.config.io_save_figs:
                fname = f"p30_amplitude_by_{analysis}.{self.config.plot_format}"
                filepath = self.output_dir / fname
                fig.savefig(filepath, dpi=self.config.plot_dpi, bbox_inches="tight")
                print(f"[OK] Salvo: {filepath}")
            else:
                plt.show()

            plt.close(fig)

