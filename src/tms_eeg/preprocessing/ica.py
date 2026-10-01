from pathlib import Path

import mne

from tms_eeg.config.settings import ProjectConfig
from tms_eeg.config.decisions import load_bad_channels  # ajuste ao nome do seu módulo


class EEGICA:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.ica = None

    def calculate_rank(self, epochs: mne.Epochs) -> int:
        """Calcula 32 - número de canais EEG ruins - 1 (referência média)."""
        eeg_channels = {
            name
            for name, channel_type in zip(
                epochs.ch_names, epochs.get_channel_types()
            )
            if channel_type == "eeg"
        }

        if len(eeg_channels) != 32:
            raise ValueError(
                f"Esperados 32 canais EEG nas épocas; encontrados "
                f"{len(eeg_channels)}."
            )

        bad_channels = set(load_bad_channels(self.config))
        unknown = bad_channels - eeg_channels
        if unknown:
            raise ValueError(
                f"Canais do YAML ausentes entre os canais EEG: {sorted(unknown)}"
            )

        other_bad_eeg = (set(epochs.info["bads"]) & eeg_channels) - bad_channels
        if other_bad_eeg:
            raise ValueError(
                f"Canais EEG marcados como ruins, mas ausentes do YAML: "
                f"{sorted(other_bad_eeg)}"
            )

        rank = 32 - len(bad_channels) - 1
        if rank < 1:
            raise ValueError(f"Rank inválido: {rank}")

        return rank

    def fit_ica(self, epochs: mne.Epochs) -> "EEGICA":
        """Ajusta a ICA aos canais EEG, excluindo os ruins do YAML."""
        if not self.config.ica_run:
            return self

        rank = self.calculate_rank(epochs)
        epochs_fit = epochs.copy()
        epochs_fit.info["bads"] = sorted(
            set(epochs_fit.info["bads"]) | set(load_bad_channels(self.config))
        )

        self.ica = mne.preprocessing.ICA(
            n_components=rank,
            random_state=97,
            method="fastica",
        )
        self.ica.fit(epochs_fit, picks="eeg")
        return self

    def apply_ica(
        self, epochs: mne.Epochs, components_to_remove: list = None
    ) -> mne.Epochs:
        """Aplica a ICA para remover artefatos das épocas."""
        if self.ica is None:
            return epochs

        self.ica.exclude = components_to_remove or []
        return self.ica.apply(epochs.copy())

    def plot_components(
        self, epochs: mne.Epochs, save_path: Path = None
    ):
        """Mostra componentes da ICA para inspeção manual."""
        if self.ica is None or not self.config.ica_plot_components:
            return

        self.ica.plot_sources(epochs, show_scrollbars=False)
        self.ica.plot_components(inst=epochs)

        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)
            self.ica.plot_components(
                inst=epochs,
                savefig=str(save_path / "ica_components.png"),
            )