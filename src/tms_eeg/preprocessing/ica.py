from tms_eeg.config.settings import ProjectConfig
import mne
from pathlib import Path

class EEGICA:
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.ica = None
        
    def fit_ica(self, epochs: mne.Epochs) -> 'EEGICA':
        """Fit ICA decomposition to epoched data."""
        ica_cfg = self.config.ica
        
        if not ica_cfg.run_ica:
            return self
            
        # Fit ICA using FastICA
        self.ica = mne.preprocessing.ICA(
            n_components=20, 
            random_state=97,
            method='fastica'
        )
        self.ica.fit(epochs)
        
        return self
        
    def apply_ica(self, epochs: mne.Epochs, components_to_remove: list = None) -> mne.Epochs:
        """Apply ICA to remove artifacts from epochs."""
        if self.ica is None:
            return epochs
            
        # Apply ICA with configured components to remove
        self.ica.exclude = components_to_remove 
            
        epochs_clean = self.ica.apply(epochs.copy())
        return epochs_clean
        
    def plot_components(self, epochs: mne.Epochs, save_path: Path = None):
        """Plot ICA components for manual inspection."""
        if self.ica is None or not self.config.ica.plot_components:
            return
            
        # Plot component sources
        self.ica.plot_sources(epochs, show_scrollbars=False)
        
        # Plot component topographies
        self.ica.plot_components(inst=epochs)
        
        # Save plots if path provided
        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)
            self.ica.plot_components(inst=epochs, savefig=str(save_path / 'ica_components.png'))
