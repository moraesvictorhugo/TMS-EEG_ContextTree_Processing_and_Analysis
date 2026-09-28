from pathlib import Path

import mne

from tms_eeg.config.settings import ProjectConfig


class EEGICA:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.ica = None

    def fit_ica(self, epochs: mne.Epochs) -> "EEGICA":
        """Fit ICA decomposition to epoched data (params from config.ica)."""
        ica_cfg = self.config.ica
        if not ica_cfg.run_ica:
            return self

        self.ica = mne.preprocessing.ICA(
            n_components=ica_cfg.n_components,
            random_state=ica_cfg.random_state,
            method=ica_cfg.method,
        )
        self.ica.fit(epochs)
        return self

    def apply_ica(self, epochs: mne.Epochs, components_to_remove: list = None) -> mne.Epochs:
        """Apply ICA to remove the given components.

        Parameters
        ----------
        epochs : mne.Epochs
            Epoched EEG data.
        components_to_remove : list, optional
            ICA component indices to exclude (e.g. ocular artifacts).
        """
        if self.ica is None:
            return epochs

        self.ica.exclude = list(components_to_remove or [])
        return self.ica.apply(epochs.copy())

    def plot_components(self, epochs: mne.Epochs, save_path: Path = None):
        """Plot ICA components for manual inspection."""
        if self.ica is None or not self.config.ica.plot_components:
            return

        self.ica.plot_sources(epochs, show_scrollbars=False)
        self.ica.plot_components(inst=epochs)

        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)
            self.ica.plot_components(inst=epochs, savefig=str(save_path / "ica_components.png"))