### Adapt code below to export epochs to Mat in different windows using 
# context_tree_epochs

from tms_eeg.config.settings import ProjectConfig
from tms_eeg.io.writer import Writer
from tms_eeg.preprocessing.annotation_exporter import EpochAnnotationExporter

# Export
config = ProjectConfig
exporter = EpochAnnotationExporter(config)
writer = Writer(config)

# Import epochs

# Pre and Pos Stim Window ------------------------------------------------------
eeg_indexes_prepost, eeg_annotations_prepost = exporter.extract_annotations(
    epochs_eeg_filtered_pre_and_post_stim
)   

symbols_prepost = exporter.map_annotations_to_symbols(
    eeg_annotations_prepost
)

exporter.export_to_mat(
    writer,
    epochs_eeg_filtered_pre_and_post_stim,
    symbols_prepost,
    subfolder="processed_pre_and_post"
)

writer.save_epochs(
    epochs_eeg_filtered_pre_and_post_stim,
    'processed_pre_and_post'
)