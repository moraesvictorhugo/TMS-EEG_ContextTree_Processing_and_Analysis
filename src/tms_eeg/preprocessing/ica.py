from pathlib import Path

import mne

from tms_eeg.config.settings import ProjectConfig


class EEGICA:
    def __init__(self, config: ProjectConfig, decisions: dict):
        self.config = config
        self.decisions = decisions
        self.ica: mne.preprocessing.ICA | None = None

    def calculate_rank(self, epochs: mne.Epochs) -> int:
        """Calculate 32 minus bad EEG channels minus one for average reference."""
        eeg_channels = {
            name
            for name, channel_type in zip(
                epochs.ch_names, epochs.get_channel_types()
            )
            if channel_type == "eeg"
        }

        if len(eeg_channels) != 32:
            raise ValueError(
                f"Expected 32 EEG channels in epochs; found "
                f"{len(eeg_channels)}."
            )

        bad_channels = set(self.decisions.get("bad_channels") or [])
        unknown = bad_channels - eeg_channels
        if unknown:
            raise ValueError(
                f"Channels listed in the YAML file are missing from the EEG "
                f"channels: {sorted(unknown)}"
            )

        other_bad_eeg = (set(epochs.info["bads"]) & eeg_channels) - bad_channels
        if other_bad_eeg:
            raise ValueError(
                f"EEG channels marked as bad but missing from the YAML file: "
                f"{sorted(other_bad_eeg)}"
            )

        rank = 32 - len(bad_channels) - 1
        if rank < 1:
            raise ValueError(f"Invalid rank: {rank}")

        return rank

    def fit_ica(self, epochs: mne.Epochs) -> "EEGICA":
        """Fit ICA to EEG channels, excluding bad channels listed in YAML."""
        if not self.config.ica_run:
            return self

        rank = self.calculate_rank(epochs)
        epochs_fit = epochs.copy()
        epochs_fit.info["bads"] = sorted(
            set(epochs_fit.info["bads"])
            | set(self.decisions.get("bad_channels") or [])
        )

        self.ica = mne.preprocessing.ICA(
            n_components=rank,
            random_state=97,
            method="fastica",
        )
        self.ica.fit(epochs_fit, picks="eeg")
        return self

    def apply_ica(self, epochs: mne.Epochs) -> mne.Epochs:
        """Apply ICA artifact rejection to a copy of the epochs.

        As componentes a remover vêm de ``apply_ica_exclude``
        (``ica_exclude`` no YAML de decisões).
        """
        if self.ica is None:
            return epochs

        return self.ica.apply(epochs.copy())

    def plot_components(
        self,
        epochs: mne.Epochs,
        save_path: Path | None = None,
    ) -> None:
        """Display ICA components and optionally save their topographies."""
        if self.ica is None or not self.config.ica_plot_components:
            return

        self.ica.plot_sources(epochs, show_scrollbars=False)
        figures = self.ica.plot_components(inst=epochs)

        if save_path is not None:
            save_path.mkdir(parents=True, exist_ok=True)
            if not isinstance(figures, list):
                figures = [figures]

            for index, figure in enumerate(figures, start=1):
                suffix = "" if index == 1 else f"_{index}"
                figure.savefig(save_path / f"ica_components{suffix}.png")
