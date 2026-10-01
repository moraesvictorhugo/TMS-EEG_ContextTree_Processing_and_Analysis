import re

import yaml

from tms_eeg.config.settings import ProjectConfig
from tms_eeg.paths import decisions_file


def load_decisions(config: ProjectConfig) -> dict:
    """Carrega as decisões do sujeito configurado."""
    path = decisions_file(config.subject_id)
    if not path.is_file():
        raise FileNotFoundError(f"[decisions] File not found: {path}")

    with path.open(encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def load_bad_channels(config: ProjectConfig) -> list[str]:
    """Lê bad_channels do arquivo de decisões do sujeito."""
    if not re.fullmatch(r"V\d+", config.subject_id):
        raise ValueError(f"subject_id inválido: {config.subject_id!r}")

    decisions = load_decisions(config)
    if not isinstance(decisions, dict):
        raise ValueError("O arquivo de decisões deve conter um mapeamento YAML.")

    bad_channels = decisions.get("bad_channels", [])
    if not isinstance(bad_channels, list) or not all(
        isinstance(channel, str) for channel in bad_channels
    ):
        raise ValueError("'bad_channels' deve ser uma lista de nomes de canais.")

    return bad_channels