from pytep import apply_sound, apply_sspsir

from tms_eeg.config.environment import setup_plotting_backend
from tms_eeg.config.settings import ProjectConfig
from tms_eeg.io.reader import load_data
from tms_eeg.io.writer import Writer
from tms_eeg.io.yaml_reader import load_decisions
from tms_eeg.preprocessing.annotation_exporter import (
    EpochAnnotationExporter,
    export_epochs_to_decisions,
)
from tms_eeg.preprocessing.annotation_processor import AnnotationProcessor
from tms_eeg.preprocessing.artifacts import ArtifactRemover
from tms_eeg.preprocessing.cleaning import (
    apply_bad_channels,
    apply_bad_epochs,
    apply_ica_exclude,
)
from tms_eeg.preprocessing.downsampling import downsample, downsample_emg_channels
from tms_eeg.preprocessing.epoching import create_epochs
from tms_eeg.preprocessing.filtering import bandpass, notch_filter
from tms_eeg.preprocessing.ica import EEGICA
from tms_eeg.visualization.tep_plots import TEPPlotter

setup_plotting_backend()

# 0. Loading and setting up channels and montage
config = ProjectConfig(subject_id="V00")
mode = config.mode

raw_data = load_data(config)

raw_data.set_channel_types({
    config.ch_eog_label: "eog",
    config.ch_emg_label: "emg",
})
raw_data.set_montage(config.ch_eeg_montage)

# 1. Cubic interpolation of the TMS artifact
raw_data = ArtifactRemover(config).remove_tms_artifact(raw_data, mode="tesa")

# 2. High-Pass filltering for ICA stability
data_filtered = bandpass(
    raw_data,
    band=config.filter_eeg_highpass,
    ch_type="eeg",
    method=config.filter_method,
    iir_order=config.filter_iir_order,
)

# 3. Process annotations and create Epochs from −1000 to +1000 ms
annotation_processor = AnnotationProcessor(config)
data_filtered = annotation_processor.process_annotations(data_filtered)

epochs_eeg = create_epochs(data_filtered, config)

epochs_emg = create_epochs(data_filtered, config, modality="emg")
epochs_emg.pick([config.ch_emg_label])

# 4. Save epoch index and annotation sequence to YAML
export_epochs_to_decisions(config, epochs_eeg)

# Check
tep_plotter = TEPPlotter(config)
tep_plotter.plot_evoked_by_symbol(
    epochs_eeg,
    picks=["FC1", "C3", "C4"],
    xlim=(-0.02, 0.1),
    ylim=(-20, 20)
)

# 5. Mark bad channels and epochs in YAML file
epochs_eeg.plot()

# 6. Remove bad channels and epochs and load ICA components to remove
decisions = load_decisions(config)
epochs_eeg = apply_bad_channels(epochs_eeg, decisions)
epochs_eeg = apply_bad_epochs(epochs_eeg, decisions, config)

# 7. Average reference (ignoring channels marked as bad)
epochs_eeg.set_eeg_reference(config.ch_eeg_reference)

# 8. Fit ICA and mark component to remove in decisions file
ica_processor = EEGICA(config, decisions)
ica_processor.fit_ica(epochs_eeg)
ica_processor.plot_components(epochs_eeg)

# 9. Remove ICA components (based on YAML file)
if ica_processor.ica is not None:
    apply_ica_exclude(ica_processor.ica, decisions)
epochs_eeg = ica_processor.apply_ica(epochs_eeg)

# 10. SOUND
epochs_eeg = apply_sound(epochs_eeg, iter_num=5, lambda_val=0.1)

# 11. SSP-SIR
epochs_eeg = apply_sspsir(epochs_eeg)

# 12. Interpolation of removed channels
epochs_eeg.interpolate_bads(reset_bads=True)

# 13. Downsampling
epochs_eeg = downsample(epochs_eeg, config.raw_downsample_freq)
epochs_emg = downsample_emg_channels(
    epochs_emg, config.epoch_emg_downsample_freq)

# 14. EEG Low-pass and EMG Bandpass Filters
epochs_eeg_filtered = bandpass(
    epochs_eeg,
    band=config.filter_eeg_lowpass,
    ch_type="eeg",
    method=config.filter_method,
    iir_order=config.filter_iir_order,
)

epochs_emg_filtered = bandpass(
    epochs_emg,
    band=config.filter_emg_bandpass,
    ch_type="emg",
    method=config.filter_method,
    iir_order=config.filter_iir_order,
)

# 15. Notch Filter
epochs_eeg_filtered = notch_filter(
    epochs_eeg_filtered,
    notch_freqs=config.filter_notch,
    ch_type="eeg",
    method=config.filter_method,
    width=config.filter_notch_width,
    iir_order=config.filter_iir_order,
    harmonics=1,
)

epochs_emg_filtered = notch_filter(
    epochs_emg_filtered,
    notch_freqs=config.filter_notch,
    ch_type="emg",
    method=config.filter_method,
    width=config.filter_notch_width,
    iir_order=config.filter_iir_order,
    harmonics=1,
)

# 16. Baseline correction ????
# epochs_eeg_filtered.apply_baseline(config.epoch_eeg_baseline)

# Check TEP quality
tep_plotter = TEPPlotter(config)
tep_plotter.plot_evoked_by_symbol(
    epochs_eeg_filtered,
    picks=["FC1", "C3", "C4"],
    xlim=(-0.05, 0.2),
    ylim=(-10,10)
)

# 17. Cropping to −800 to +800 ms (to remove edge effects)
epochs_eeg_filtered = (
    epochs_eeg_filtered.crop(tmin=-0.8, tmax=0.8))

# 18. Export
exporter = EpochAnnotationExporter(config)
writer = Writer(config)

if mode == "tep":
    writer.save_epochs(epochs_eeg_filtered, "processed_tep")

elif mode == "context_tree":
    writer.save_epochs(epochs_eeg_filtered, "processed_context_tree")

    _, eeg_annotations_prepost = exporter.extract_annotations(
        epochs_eeg_filtered
    )

    symbols_prepost = exporter.map_annotations_to_symbols(
        eeg_annotations_prepost
    )

    exporter.export_to_mat(
        writer,
        epochs_eeg_filtered,
        symbols_prepost,
        subfolder="tree_retrieving",
    )

# EMG ---------------------------------------------------------------------
writer.save_emg_epochs(
    epochs_emg_filtered,
    'emg_processed'
)