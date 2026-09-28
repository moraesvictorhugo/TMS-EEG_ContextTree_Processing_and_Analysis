# reader.py
from pathlib import Path

import mne

from tms_eeg.config.settings import ProjectConfig


def load_data(config: ProjectConfig, data_type: str = "raw"):
    """Load EEG data from file.

    Args:
        config (ProjectConfig): Project configuration.
        data_type (str): "raw" for Raw (.bdf) or "epochs" for Epochs (-epo.fif).

    Returns:
        mne.io.Raw | mne.Epochs: EEG data.
    """
    if data_type == "raw":
        subject_dir = config.paths.subject_raw_dir(config.subject_id)
        file_path = next(subject_dir.glob("*.bdf"))
        return mne.io.read_raw_bdf(file_path, preload=True)

    if data_type == "epochs":
        subject_dir = config.paths.subject_processed_dir(config.subject_id, "processed_full")
        file_path = next(subject_dir.glob("*_epochs_processed.fif"))
        return mne.read_epochs(file_path, preload=True)

    raise ValueError(f"data_type inválido: '{data_type}'. Use 'raw' or 'epochs'.")


def get_raw_path(config: ProjectConfig) -> str:
    """Return the path of the original raw .bdf file for the subject."""
    bdf_files = list(config.paths.subject_raw_dir(config.subject_id).glob("*.bdf"))
    if not bdf_files:
        raise FileNotFoundError(
            f"Nenhum .bdf encontrado em {config.paths.subject_raw_dir(config.subject_id)}"
        )
    return str(bdf_files[0])
