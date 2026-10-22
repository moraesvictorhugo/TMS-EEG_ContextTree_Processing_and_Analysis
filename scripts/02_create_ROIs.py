"""Create ROI (region of interest) epochs from cleaned epochs.

Each ROI is the average signal of a group of electrodes defined in
config.analysis_rois. The ROI analysis does not substitute the channel
analysis; it only offers a complementary, lower-noise view.

Output:
    A new mne.Epochs object (one channel per ROI) saved with Writer.
"""

from tms_eeg.analysis.roi import create_roi_epochs
from tms_eeg.config.settings import ProjectConfig
from tms_eeg.io.reader import load_data
from tms_eeg.io.writer import Writer


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