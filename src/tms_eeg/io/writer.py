from pathlib import Path
from typing import Optional

import mne
import numpy as np
from matplotlib.figure import Figure
from scipy.io import savemat


def save_figure(config, fig: Figure, name: str, writer: Optional["Writer"] = None) -> None:
    """Save a matplotlib figure when figure export is enabled.

    Helper shared by the visualization classes. ``config`` may be None when a
    plotter is used standalone; ``writer`` may be omitted to let the helper
    build one from the config.
    """
    if config is None or fig is None or not getattr(config.io, "save_figs", False):
        return
    if writer is None:
        writer = Writer(config)
    writer.save_figure(fig, name)


class Writer:
    """Save EEG/EMG data and figures to the processed directory."""

    def __init__(self, config):
        self.config = config
        self.subject_id = config.subject_id

    def _create_processed_dir(self) -> Path:
        """Create (if needed) and return data/processed/<subject_id>."""
        processed_dir = self.config.paths.processed_dir / self.subject_id
        processed_dir.mkdir(parents=True, exist_ok=True)
        return processed_dir

    def _get_filename(self, base_name: str, suffix: str, extension: str = ".fif") -> str:
        """Generate a standardized filename: <subject>_<base>_<suffix><extension>."""
        return f"{self.subject_id}_{base_name}_{suffix}{extension}"

    def _save_epochs(self, epochs: mne.Epochs, subfolder: str, base_name: str) -> None:
        """Internal helper: save an Epochs object to data/processed/<id>/<subfolder>."""
        if not self.config.io.export_data:
            print("Export skipped: export_data is set to False in configuration")
            return

        processed_dir = self._create_processed_dir() / subfolder
        processed_dir.mkdir(exist_ok=True)

        full_path = processed_dir / self._get_filename(base_name, "processed")
        print(f"Saving epochs to: {full_path}")
        epochs.save(full_path, overwrite=True)
        print("Epochs saved successfully!")

    def save_epochs(self, epochs: mne.Epochs, subfolder: str = "processed") -> None:
        """Save processed epochs to the processed directory."""
        self._save_epochs(epochs, subfolder, base_name="epochs")

    def save_emg_epochs(self, epochs: mne.Epochs, subfolder: str = "emg_processed") -> None:
        """Save EMG epochs to the processed directory."""
        self._save_epochs(epochs, subfolder, base_name="emg_epochs")

    def _create_figures_dir(self, subfolder: str = "figures") -> Path:
        """Create (if needed) and return the figures directory for this subject."""
        figures_dir = self._create_processed_dir() / subfolder
        figures_dir.mkdir(exist_ok=True)
        return figures_dir

    def save_figure(self, fig: Figure, name: str, subfolder: str = None,
                    fmt: str = None, dpi: int = None) -> None:
        """
        Save a single matplotlib figure.

        Parameters
        ----------
        fig : matplotlib.figure.Figure
            Figure to save
        name : str
            Descriptive name for the file (e.g., 'butterfly_8bits_1')
        subfolder : str
            Subfolder within processed directory (default: from config)
        fmt : str
            Image format (default: from config)
        dpi : int
            Resolution (default: from config)
        """
        if not self.config.io.export_data:
            print("Export skipped: export_data is set to False in configuration")
            return

        # Use config defaults if not provided
        plots = self.config.plots
        subfolder = subfolder or plots.figure_subfolder
        fmt = fmt or plots.figure_format
        dpi = dpi or plots.figure_dpi

        try:
            figures_dir = self._create_processed_dir() / subfolder
            figures_dir.mkdir(exist_ok=True)
            filename = self._get_filename(name, "plot", extension=f".{fmt}")
            full_path = figures_dir / filename

            fig.savefig(full_path, dpi=dpi, bbox_inches="tight")
            print(f"Figure saved: {full_path}")
        except Exception as e:
            print(f"Error saving figure '{name}': {e}")

    def save_epochs_to_mat(
        self,
        epochs: mne.Epochs,
        symbol_sequence: np.ndarray,
        subfolder: str = "processed",
        window: Optional[tuple] = None,
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
        window : tuple, optional
            (tmin, tmax) in seconds to crop before exporting (default: from config).
        """
        if not self.config.io.export_data:
            print("Export skipped: export_data is set to False in configuration")
            return

        window = window or self.config.epochs.mat_window

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
        mat_dict = {"data": {"X_ter": X_ter, "Y_ter": Y_ter}}

        # --- 6. Save ---
        processed_dir = self._create_processed_dir() / subfolder
        processed_dir.mkdir(exist_ok=True)

        filename = self._get_filename("epochs", "contexttree", extension=".mat")
        full_path = processed_dir / filename

        print(f"Saving .mat file to: {full_path}")

        savemat(full_path, mat_dict, do_compression=True, long_field_names=True)
        print(".mat file saved successfully!")