import re
from pathlib import Path

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
