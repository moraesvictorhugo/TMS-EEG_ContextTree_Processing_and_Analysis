import mne

from tms_eeg.config.settings import ProjectConfig


def apply_bad_channels(inst, decisions: dict, drop: bool = False):
    """Mark (or drop) bad_channels on Raw/Epochs."""
    bad_chs = [ch for ch in (decisions.get("bad_channels") or []) if ch in inst.ch_names]
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

    bad_eps = decisions.get("bad_epochs") or []
    if bad_eps:
        epochs.drop(bad_eps, reason="YAML")
    return epochs


def apply_ica_exclude(ica: mne.preprocessing.ICA, decisions: dict) -> mne.preprocessing.ICA:
    """Set ica.exclude from decisions."""
    ica.exclude = list(decisions.get("ica_exclude") or [])
    return ica
