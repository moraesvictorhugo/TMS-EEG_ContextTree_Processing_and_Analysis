import re
from pathlib import Path

import yaml

from tms_eeg.config.settings import ProjectConfig
from tms_eeg.paths import decisions_file


def load_decisions(config: ProjectConfig) -> dict:
    """Load decisions for the configured subject."""
    if not re.fullmatch(r"V\d+", config.subject_id):
        raise ValueError(f"Invalid subject_id: {config.subject_id!r}")

    path = decisions_file(config.subject_id)
    if not path.is_file():
        raise FileNotFoundError(f"[decisions] File not found: {path}")

    with path.open(encoding="utf-8") as file:
        decisions = yaml.safe_load(file)
    validate_decisions(decisions)
    return decisions


def validate_decisions(decisions: dict) -> None:
    """Require explicit lists; [] means reviewed with no exclusions for that field.

    This validates structure, not scientific approval or the review date.
    Dataset-specific bounds are checked when exclusions are applied.
    """
    if not isinstance(decisions, dict):
        raise ValueError("Decisions must be a YAML mapping, not empty or a placeholder.")
    for key in ("bad_channels", "bad_epochs", "ica_exclude"):
        values = decisions.get(key)
        if not isinstance(values, list):
            raise ValueError(f"Decision {key!r} must be an explicit list; use [] for none.")
        if key == "bad_channels":
            valid = all(isinstance(value, str) and value.strip() for value in values)
        else:
            valid = all(type(value) is int and value >= 0 for value in values)
        if not valid:
            raise ValueError(f"Invalid {key}: {values!r}. Replace placeholders with valid entries.")
        if len(values) != len(set(values)):
            raise ValueError(f"Duplicate entries in {key}: {values!r}")


def save_epochs_record(
    config: ProjectConfig,
    indexes,
    annotations: list[str],
) -> Path:
    """Write/replace the ``epochs_record`` block in the subject's YAML.

    Each epoch is stored as one ``- {index: ..., annotation: ...}`` item,
    where ``index`` is the value of ``epochs.selection``. The block is always
    appended at the end of the file and replaced on re-runs, keeping the
    other keys (and any comments) untouched.

    Args:
        config: Project configuration (defines the subject's YAML path).
        indexes: Epoch indexes, e.g. ``epochs.selection``.
        annotations: Annotation/condition label of each epoch.

    Returns:
        Path of the updated YAML file.
    """
    if len(indexes) != len(annotations):
        raise ValueError("Epoch indices and annotation labels must have equal lengths.")
    path = decisions_file(config.subject_id)
    lines = (
        path.read_text(encoding="utf-8").splitlines(keepends=True)
        if path.is_file()
        else []
    )

    # Replace the block on re-runs (idempotent)
    start = next(
        (n for n, line in enumerate(lines) if line.startswith("epochs_record:")),
        None,
    )
    if start is not None:
        end = start + 1
        while end < len(lines) and (
            not lines[end].strip() or lines[end][0] in " \t"
        ):
            end += 1
        del lines[start:end]

    if lines and lines[-1].strip():
        lines.append("\n")
    lines.append("epochs_record:\n")
    for index, annotation in zip(indexes, annotations):
        item = yaml.safe_dump(
            {"index": int(index), "annotation": str(annotation)},
            default_flow_style=True,
            sort_keys=False,
        ).strip()
        lines.append(f"  - {item}\n")

    path.write_text("".join(lines), encoding="utf-8")
    print(f"Epochs record saved: {path} ({len(indexes)} epochs)")
    return path
