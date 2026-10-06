from pathlib import Path

import mne
import numpy as np
from matplotlib.figure import Figure
from scipy.io import savemat

from tms_eeg import paths


class Writer:
    """Class for saving EEG and EMG data (raw, epochs, evoked) to the processed directory."""
    
    def __init__(self, config):
        """
        Initialize Writer with project configuration.
        
        Parameters
        ----------
        config : ProjectConfig
            Configuration object containing subject_id and other settings
        """
        self.config = config
        self.subject_id = config.subject_id
        
    def _create_processed_dir(self) -> Path:
        """
        Create the processed data directory if it doesn't exist.
        
        Returns
        -------
        Path
            Path to the processed directory for this subject
        """
        processed_dir = paths.processed_dir(self.subject_id)
        processed_dir.mkdir(parents=True, exist_ok=True)
        return processed_dir
    
    def _get_filename(self, base_name: str, suffix: str, extension: str = ".fif") -> str:
        """
        Generate a standardized filename.
        
        Parameters
        ----------
        base_name : str
            Base name for the file (e.g., 'epochs', 'evoked')
        suffix : str
            Additional identifier (e.g., condition name, 'processed')
        extension : str
            File extension (default: '.fif')
            
        Returns
        -------
        str
            Complete filename
        """
        return f"{self.subject_id}_{base_name}_{suffix}{extension}"
    
    def save_epochs(
        self,
        epochs: mne.Epochs,
        subfolder: str = "processed",
        base_name: str = "epochs",
    ) -> None:
        """
        Save processed epochs to the processed directory.
        
        Parameters
        ----------
        epochs : mne.Epochs
            Processed epochs object
        subfolder : str
            Subfolder name within processed directory (default: 'processed')
        base_name : str
            Base filename part ('epochs' or 'emg_epochs')
        """
        # Check if export is enabled in configuration
        if not self.config.io_export_data:
            print("Export skipped: export_data is set to False in configuration")
            return
            
        processed_dir = self._create_processed_dir() / subfolder
        processed_dir.mkdir(exist_ok=True)
        
        filename = self._get_filename(base_name, "processed")
        full_path = processed_dir / filename
        
        print(f"Saving epochs to: {full_path}")
        epochs.save(full_path, overwrite=True)
        print(f"Epochs saved successfully!")
    
    def save_emg_epochs(self, epochs: mne.Epochs, subfolder: str = "emg_processed") -> None:
        """Save EMG epochs to the processed directory with preserved annotations."""
        self.save_epochs(epochs, subfolder=subfolder, base_name="emg_epochs")

    def save_epochs_to_mat(
        self,
        epochs: mne.Epochs,
        symbol_sequence: np.ndarray,
        subfolder: str = "processed",
        window: tuple = (0.015, 0.450),
    ) -> None:
        """
        Save epochs to a .mat file matching the context-tree pipeline format.

        Output MATLAB structure (variable name: `data`):
            data.X_ter : 1 x N_symbols double  -> context-tree symbol sequence (0/1/2)
            data.Y_ter : 2 x N_channels cell
                row 1 -> channel name (char)
                row 2 -> (n_samples x n_epochs) double matrix in microvolts

        Parameters
        ----------
        epochs : mne.Epochs
            Epochs object (data assumed in Volts, MNE default).
        symbol_sequence : np.ndarray
            1-D array of context-tree symbols (length can differ from n_epochs).
        subfolder : str
            Subfolder within processed dir.
        window : tuple
            (tmin, tmax) in seconds to crop before exporting.
        """
        if not self.config.io_export_data:
            print("Export skipped: export_data is set to False in configuration")
            return

        # --- 1. Pick EEG channels and crop to desired window ---
        ep = epochs.copy().pick("eeg").crop(tmin=window[0], tmax=window[1])

        # --- 2. Get data and convert V -> uV ---
        # MNE shape: (n_epochs, n_channels, n_samples)
        arr = ep.get_data() * 1e6  # to microvolts
        n_epochs, n_channels, n_samples = arr.shape

        # --- 3. Build Y_ter cell (2 x n_channels) ---
        # Row 0: channel name (str)
        # Row 1: matrix (n_samples, n_epochs) double
        Y_ter = np.empty((2, n_channels), dtype=object)
        for ch_idx, ch_name in enumerate(ep.ch_names):
            Y_ter[0, ch_idx] = str(ch_name)
            # arr[:, ch_idx, :] has shape (n_epochs, n_samples) -> transpose
            Y_ter[1, ch_idx] = arr[:, ch_idx, :].T.astype(np.float64)

        # --- 4. Build X_ter as 1 x N row vector of double ---
        X_ter = np.asarray(symbol_sequence, dtype=np.float64).reshape(1, -1)

        # --- 5. Wrap in 'data' struct ---
        mat_dict = {
            "data": {
                "X_ter": X_ter,
                "Y_ter": Y_ter,
            }
        }

        # --- 6. Save ---
        processed_dir = self._create_processed_dir() / subfolder
        processed_dir.mkdir(exist_ok=True)

        filename = self._get_filename("epochs", "contexttree", extension=".mat")
        full_path = processed_dir / filename

        print(f"Saving .mat file to: {full_path}")

        savemat(full_path, mat_dict, do_compression=True, long_field_names=True)
        print(".mat file saved successfully!")


def save_figure(fig: Figure, name: str, config) -> None:
    """Save ``fig`` se ``config.io_save_figs`` estiver habilitado —
    único caminho de salvamento de figuras do projeto."""
    if config is None or not config.io_save_figs or fig is None:
        return

    figures_dir = paths.processed_dir(config.subject_id) / config.plot_subfolder
    figures_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{config.subject_id}_{name}_plot.{config.plot_format}"

    try:
        fig.savefig(figures_dir / filename, dpi=config.plot_dpi, bbox_inches="tight")
        print(f"Figure saved: {figures_dir / filename}")
    except Exception as e:
        print(f"Error saving figure '{name}': {e}")
