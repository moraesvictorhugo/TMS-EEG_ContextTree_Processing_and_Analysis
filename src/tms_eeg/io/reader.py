# reader.py
import mne

from tms_eeg.config.settings import ProjectConfig
from tms_eeg.paths import processed_dir, raw_dir


def load_data(config: ProjectConfig, data_type: str = "raw"):
    """Load EEG data from file.

    Args:
        config (ProjectConfig): Project configuration.
        data_type (str): "raw" for Raw (.bdf) or "epochs" for Epochs (-epo.fif).

    Returns:
        mne.io.Raw | mne.Epochs: EEG data.
    """
    if data_type == "raw":
        file_path = get_raw_path(config)
        data = mne.io.read_raw_bdf(file_path, preload=True)

    elif data_type == "epochs":
        subject_processed_dir = processed_dir(config.subject_id)
        mode_folder = {
            "tep": "processed_tep",
            "context_tree": "processed_context_tree",
        }.get(config.mode)

        if mode_folder is None:
            raise ValueError(
                f"Unsupported processing mode {config.mode!r}; "
                "expected 'tep' or 'context_tree'."
            )

        epochs_dir = subject_processed_dir / mode_folder

        file_pattern = "*_epochs_processed.fif"
        file_path = next(
            iter(sorted(epochs_dir.glob(file_pattern))),
            None,
        )
        if file_path is None:
            raise FileNotFoundError(
                f"No processed epochs found for subject {config.subject_id!r}. "
                f"Searched {file_pattern!r} in: {epochs_dir}"
            )
        data = mne.read_epochs(file_path, preload=True)

    else:
        raise ValueError(f"Invalid data_type: '{data_type}'. Use 'raw' or 'epochs'.")

    return data


def get_raw_path(config: ProjectConfig) -> str:
    """Return the subject's original raw BDF file path."""
    subject_raw_dir = raw_dir(config.subject_id)
    bdf_files = list(subject_raw_dir.glob("*.bdf"))
    if not bdf_files:
        raise FileNotFoundError(
            f"No .bdf file found in {subject_raw_dir}"
        )
    return str(bdf_files[0])
