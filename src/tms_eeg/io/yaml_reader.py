import re

import yaml

from tms_eeg.config.settings import ProjectConfig
from tms_eeg.paths import decisions_file


def load_decisions(config: ProjectConfig) -> dict:
    """Load the decisions for the configured subject."""
    path = decisions_file(config.subject_id)
    if not path.is_file():
        raise FileNotFoundError(f"[decisions] File not found: {path}")

    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_bad_channels(config: ProjectConfig) -> list[str]:
    """Read bad_channels from the subject's decisions file."""
    if not re.fullmatch(r"V\d+", config.subject_id):
        raise ValueError(f"Invalid subject_id: {config.subject_id!r}")

    decisions = load_decisions(config)
    if not isinstance(decisions, dict):
        raise ValueError("The decisions file must contain a YAML mapping.")

    bad_channels = decisions.get("bad_channels", [])
    if not isinstance(bad_channels, list) or not all(
        isinstance(channel, str) for channel in bad_channels
    ):
        raise ValueError("'bad_channels' must be a list of channel names.")

    return bad_channels


def load_bad_epochs(config: ProjectConfig) -> list[int]:
    """Read zero-based bad epoch indices from the subject's decisions file."""
    if not re.fullmatch(r"V\d+", config.subject_id):
        raise ValueError(f"Invalid subject_id: {config.subject_id!r}")

    decisions = load_decisions(config)
    if not isinstance(decisions, dict):
        raise ValueError("The decisions file must contain a YAML mapping.")

    bad_epochs = decisions.get("bad_epochs", [])
    if not isinstance(bad_epochs, list) or not all(
        type(epoch) is int and epoch >= 0 for epoch in bad_epochs
    ):
        raise ValueError("'bad_epochs' must be a list of non-negative integer indices.")

    return bad_epochs
