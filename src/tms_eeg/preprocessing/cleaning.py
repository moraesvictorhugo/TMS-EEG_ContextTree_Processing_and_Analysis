import mne

from tms_eeg.config.settings import ProjectConfig
from tms_eeg.io.yaml_reader import validate_decisions


def apply_bad_channels(inst, decisions: dict, drop: bool = False):
    """Mark (or drop) bad_channels on Raw/Epochs."""
    validate_decisions(decisions)
    bad_chs = decisions["bad_channels"]
    unknown = set(bad_chs) - set(inst.ch_names)
    if unknown:
        raise ValueError(f"Unknown bad channels: {sorted(unknown)}")
    if not bad_chs:
        return inst
    if drop:
        inst.drop_channels(bad_chs)
    else:
        inst.info["bads"] = sorted(set(inst.info["bads"]) | set(bad_chs))
    return inst


def apply_bad_epochs(
    epochs: mne.Epochs,
    decisions: dict,
    config: ProjectConfig,
) -> mne.Epochs:
    """Drop YAML-marked epochs only in TEP mode."""
    if config.mode != "tep":
        return epochs

    validate_decisions(decisions)
    bad_eps = decisions["bad_epochs"]
    if any(index >= len(epochs) for index in bad_eps):
        raise ValueError(
            f"bad_epochs must be current positions from 0 to {len(epochs) - 1}; "
            "compare them with epochs.selection before applying decisions."
        )
    if bad_eps:
        epochs.drop(bad_eps, reason="YAML")
    return epochs


def apply_ica_exclude(ica: mne.preprocessing.ICA, decisions: dict) -> mne.preprocessing.ICA:
    """Set ica.exclude from decisions."""
    validate_decisions(decisions)
    exclusions = decisions["ica_exclude"]
    if any(index >= ica.n_components_ for index in exclusions):
        raise ValueError(f"ICA exclusions must be below {ica.n_components_}.")
    ica.exclude = list(exclusions)
    return ica
