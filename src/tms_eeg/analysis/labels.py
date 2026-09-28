"""Event-id normalisation helpers for the analysis pipeline."""

import mne


def normalize_event_id(epochs: mne.Epochs, label_rename: dict) -> None:
    """Rewrite epoch condition labels according to ``config.events.label_rename``.

    Lookup is case-insensitive (e.g. "8Bit 1" and "8bit 1" both map). Labels
    not present in the mapping are kept unchanged.

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs whose ``event_id`` labels are rewritten in place.
    label_rename : dict
        Mapping ``{old label: new label}`` (keys compared lowercased).
    """
    rename = {k.lower(): v for k, v in label_rename.items()}
    new_event_id = {}
    for label, code in epochs.event_id.items():
        new_event_id[rename.get(label.lower(), label)] = code
    epochs.event_id = new_event_id