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
        epochs_dir = processed_dir(config.subject_id) / "processed_full"
        file_path = next(epochs_dir.glob("*_epochs_processed.fif"))
        data = mne.read_epochs(file_path, preload=True)

    else:
        raise ValueError(f"data_type inválido: '{data_type}'. Use 'raw' or 'epochs'.")

    return data


def get_raw_path(config: ProjectConfig) -> str:
    """Retorna o caminho do arquivo .bdf raw original para o sujeito."""
    subject_raw_dir = raw_dir(config.subject_id)
    bdf_files = list(subject_raw_dir.glob("*.bdf"))
    if not bdf_files:
        raise FileNotFoundError(
            f"Nenhum .bdf encontrado em {subject_raw_dir}"
        )
    return str(bdf_files[0])
