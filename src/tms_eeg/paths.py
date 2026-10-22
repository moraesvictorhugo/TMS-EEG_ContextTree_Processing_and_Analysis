"""Canonical project paths."""

from pathlib import Path

# Resolve the repository root from src/tms_eeg/paths.py.
PROJECT_ROOT = Path(__file__).resolve().parents[2]


def raw_dir(subject_id: str) -> Path:
    """Return the subject's raw data directory."""
    return PROJECT_ROOT / "data" / "raw" / f"{subject_id}_data"


def processed_dir(subject_id: str) -> Path:
    """Return the subject's processed data directory."""
    return PROJECT_ROOT / "data" / "processed" / subject_id

DECISIONS_DIR = PROJECT_ROOT / "decisions"

def decisions_file(subject_id: str) -> Path:
    """Return the subject's decisions YAML path."""
    sid = str(subject_id).removeprefix("sub-")
    return DECISIONS_DIR / f"sub-{sid}.yaml"

