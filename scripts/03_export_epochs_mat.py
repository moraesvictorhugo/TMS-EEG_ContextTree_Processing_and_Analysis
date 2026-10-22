"""Main script for exporting epochs to .mat files with different time windows.

Generates two outputs:
  - processed_pre_and_post: full window (pre + post stimulus)
  - processed_only_post: post-stimulus window only

Time windows are read from ProjectConfig (export_pre_and_post_window,
export_only_post_window).
"""

from tms_eeg.config.settings import ProjectConfig
from tms_eeg.io.reader import load_data
from tms_eeg.io.writer import Writer
from tms_eeg.preprocessing.annotation_exporter import EpochAnnotationExporter

# Configuration
config = ProjectConfig(subject_id="V00")
exporter = EpochAnnotationExporter(config)
writer = Writer(config)

# Load processed epochs from disk
epochs = load_data(config, data_type="epochs")

# Extract annotations and symbols (full window)
eeg_indexes, eeg_annotations = exporter.extract_annotations(epochs)
symbols = exporter.map_annotations_to_symbols(eeg_annotations)

# Export processed_pre_and_post (window from config)
exporter.export_to_mat(
    writer,
    epochs,
    symbols,
    subfolder="processed_pre_and_post",
    window=config.export_pre_and_post_window,
)
writer.save_epochs(
    epochs,
    subfolder="processed_pre_and_post",
    base_name="epochs_pre_and_post",
)

# Export processed_only_post (window from config)
epochs_only_post = epochs.copy().crop(
    tmin=config.export_only_post_window[0],
    tmax=config.export_only_post_window[1],
)
eeg_indexes_post, eeg_annotations_post = exporter.extract_annotations(
    epochs_only_post
)
symbols_post = exporter.map_annotations_to_symbols(eeg_annotations_post)

exporter.export_to_mat(
    writer,
    epochs_only_post,
    symbols_post,
    subfolder="processed_only_post",
    window=config.export_only_post_window,
)
writer.save_epochs(
    epochs_only_post,
    subfolder="processed_only_post",
    base_name="epochs_only_post",
)
