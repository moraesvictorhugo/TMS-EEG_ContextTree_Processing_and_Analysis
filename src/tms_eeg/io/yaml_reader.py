import yaml

from tms_eeg.config.settings import ProjectConfig
from tms_eeg.paths import decisions_file


def load_decisions(config: ProjectConfig) -> dict:
    """Carrega decisions/sub-XX.yaml do sujeito configurado."""
    path = decisions_file(config.subject_id)
    if not path.exists():
        raise FileNotFoundError(f"[decisions] File not found: {path}")
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
