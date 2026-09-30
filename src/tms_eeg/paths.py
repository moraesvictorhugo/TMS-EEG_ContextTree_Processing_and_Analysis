"""Caminhos canônicos do projeto — única fonte de verdade das pastas."""

from pathlib import Path

# src/tms_eeg/paths.py -> pais: tms_eeg, src, raiz do repositório
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def raw_dir(subject_id: str) -> Path:
    """Pasta de dados brutos do sujeito (data/raw/<subject_id>_data)."""
    return PROJECT_ROOT / "data" / "raw" / f"{subject_id}_data"


def processed_dir(subject_id: str) -> Path:
    """Pasta de dados processados do sujeito (data/processed/<subject_id>)."""
    return PROJECT_ROOT / "data" / "processed" / subject_id

DECISIONS_DIR = PROJECT_ROOT / "decisions"

def decisions_file(subject_id: str) -> Path:
    """Arquivo de decisões do sujeito (decisions/sub-<id>.yaml)."""
    sid = str(subject_id).removeprefix("sub-")
    return DECISIONS_DIR / f"sub-{sid}.yaml"

