"""Region-of-interest transformations for EEG epochs."""

import mne
import numpy as np


def create_roi_epochs(epochs: mne.Epochs, rois: dict) -> mne.Epochs:
    """Average channels inside each ROI and return a new Epochs object.

    Args:
        epochs: Cleaned MNE Epochs.
        rois: Mapping from ROI names to channel names.

    Returns:
        An MNE Epochs object with one channel per ROI. ROIs with no
        available channel are skipped.

    Raises:
        ValueError: If none of the configured ROIs contain available channels.
    """
    available_ch = set(epochs.ch_names)
    new_data = []
    new_ch_names = []

    for roi_name, ch_names in rois.items():
        # Some channels (e.g. PO3/PO4) may be absent for a given montage.
        present = [ch for ch in ch_names if ch in available_ch]
        if not present:
            print(f"ROI '{roi_name}' skipped: no channels found in data.")
            continue

        idx = [epochs.ch_names.index(ch) for ch in present]
        # epochs.get_data() shape: (n_epochs, n_channels, n_times)
        roi_signal = epochs.get_data()[:, idx, :].mean(axis=1)
        new_data.append(roi_signal)
        new_ch_names.append(roi_name)

    if not new_data:
        raise ValueError("No ROI could be built: check config.analysis_rois.")

    new_data = np.stack(new_data, axis=1)  # (n_epochs, n_rois, n_times)

    info = mne.create_info(
        ch_names=new_ch_names,
        sfreq=epochs.info["sfreq"],
        ch_types="eeg",
    )
    return mne.EpochsArray(
        new_data,
        info,
        events=epochs.events,
        event_id=epochs.event_id,
        tmin=epochs.tmin,
        metadata=epochs.metadata.copy() if epochs.metadata is not None else None,
        selection=epochs.selection.copy(),
        drop_log=epochs.drop_log,
    )