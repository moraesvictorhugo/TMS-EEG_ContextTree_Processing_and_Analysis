import re

import yaml

from tms_eeg.config.settings import ProjectConfig
from tms_eeg.paths import decisions_file


def load_decisions(config: ProjectConfig) -> dict:
    """Load the decisions for the configured subject (única leitura do YAML)."""
    if not re.fullmatch(r"V\d+", config.subject_id):
        raise ValueError(f"Invalid subject_id: {config.subject_id!r}")

    path = decisions_file(config.subject_id)
    if not path.is_file():
        raise FileNotFoundError(f"[decisions] File not found: {path}")

    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}
