"""Create ROI (region of interest) epochs from cleaned epochs.

Each ROI is the average signal of a group of electrodes defined in
config.analysis_rois. The ROI analysis does not substitute the channel
analysis; it only offers a complementary, lower-noise view.

Output:
    A new mne.Epochs object (one channel per ROI) saved with Writer.
"""

from tms_eeg.config.settings import ProjectConfig
from tms_eeg.io.reader import load_data
from tms_eeg.io.writer import Writer

import mne
import numpy as np


def create_roi_epochs(epochs: mne.Epochs, rois: dict) -> mne.Epochs:
    """Average channels inside each ROI and return a new Epochs object.

    Args:
        epochs: Cleaned mne.Epochs.
        rois: dict {roi_name: [channel_names]} read from config.

    Returns:
        mne.Epochs with one channel per ROI (average of its electrodes).
        ROIs with no available channel are skipped.
    """
    available_ch = set(epochs.ch_names)
    new_data = []
    new_ch_names = []

    for roi_name, ch_names in rois.items():
        # Keep only channels that really exist in the data (e.g. PO3/PO4
        # may be missing for some montages).
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
        raise ValueError(
            "No ROI could be built: check config.analysis_rois."
        )

    new_data = np.stack(new_data, axis=1)  # (n_epochs, n_rois, n_times)

    info = mne.create_info(
        ch_names=new_ch_names,
        sfreq=epochs.info["sfreq"],
        ch_types="eeg",
    )
    roi_epochs = mne.EpochsArray(
        new_data,
        info,
        events=epochs.events,
        event_id=epochs.event_id,
        tmin=epochs.tmin,
    )
    return roi_epochs


def main():
    config = ProjectConfig(subject_id="V00")
    epochs = load_data(config, data_type="epochs")

    rois = config.analysis_rois
    roi_epochs = create_roi_epochs(epochs, rois)

    print(f"ROI epochs created: {len(roi_epochs.ch_names)} ROIs, "
          f"{len(roi_epochs)} epochs")
    print("ROI channels:", roi_epochs.ch_names)

    writer = Writer(config)
    writer.save_epochs(roi_epochs, subfolder="processed_rois",
                       base_name="epochs_rois")


if __name__ == "__main__":
    main()