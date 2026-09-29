import mne
from pytep import apply_sound, apply_sspsir
from scipy.signal import detrend

from tms_eeg.config.environment import setup_plotting_backend
from tms_eeg.config.settings import ProjectConfig
from tms_eeg.io.reader import load_data
from tms_eeg.io.writer import Writer
from tms_eeg.preprocessing.annotation_exporter import EpochAnnotationExporter
from tms_eeg.preprocessing.annotation_processor import AnnotationProcessor
from tms_eeg.preprocessing.artifacts import ArtifactRemover
from tms_eeg.preprocessing.downsampling import downsample, downsample_emg_channels
from tms_eeg.preprocessing.epoching import create_epochs, drop_from_json
from tms_eeg.preprocessing.filtering import bandpass, notch_filter
from tms_eeg.preprocessing.ica import EEGICA
from tms_eeg.visualization.tep_plots import TEPPlotter

setup_plotting_backend()

"""
Steps

    
"""
##############################################################################
# Settings
config = ProjectConfig(subject_id="V00")

# Load data
raw_data = load_data(config)

# Set EOG and EMG channels and set montage
raw_data.set_channel_types({
    config.ch_eog_label: 'eog', config.ch_emg_label: 'emg'})
raw_data.set_montage(config.ch_eeg_montage)

# Artifact removal
raw_data = ArtifactRemover(config).remove_tms_artifact(raw_data, mode='cubic')

# Notch Filter
data_filtered = notch_filter(raw_data, config.filter_notch, "eeg", band=(58, 62), harmonics=3)

# HighPass Filter
data_filtered.filter(l_freq=1, h_freq=None, fir_design='firwin')

# Process annotations to replace Stimulus A with condition labels
annotation_processor = AnnotationProcessor(config)
data_filtered = annotation_processor.process_annotations(data_filtered)









# Create epochs using standard pipeline          # Checar se faz a correção baseline de (-0,25, -0,1) s
epochs_eeg = create_epochs(data_filtered, config)    # e detrend=1

### Exemplo:

epochs = mne.Epochs(
    raw_data,
    events=events_tms,
    event_id=events_id_tms,
    tmin=-0.8,
    tmax=0.8,
    baseline=(-0.25,-0.1),
    preload=True,
    detrend=1,
)

# Create epochs for EMG data
raw_data_emg = raw_data.copy().pick("emg")
epochs_emg = create_epochs(raw_data_emg, config)

# # Baseline correction
# epochs_eeg.apply_baseline(baseline=(-0.25, -0.01))    # ver se não faz diretamente em cima

# Verify artfact duration
tep_plotter = TEPPlotter(config)
tep_plotter.plot_evoked_by_symbol(
    epochs_eeg,
    picks=["FC1", "FC5", "C3", "C4", "CP1", "CP5"],
    xlim=(-0.01, 0.015),
    ylim=(-30, 30)
)

###########################################################################

# Artifact removal
epochs_eeg = ArtifactRemover(config).remove_tms_artifact(epochs_eeg, mode='cubic')

tep_plotter.plot_evoked_by_symbol(
    epochs_eeg,
    picks=["FC1", "FC5", "C3", "C4", "CP1", "CP5"],
    xlim=(-0.01, 0.2),
    ylim=(-30, 30)
)

# Remove bad channels (TP9, TP10, O1, O2, Iz)
epochs_eeg.drop_channels(["TP9", "TP10", "O1", "O2", "Iz"])

# Verify bad epochs -> skip if already on json
epochs_eeg.plot()

###########################################################################

# Drop bad marked channels
#epochs_eeg.drop_channels(epochs_eeg.info['bads'])

# Interpolate channels marked as bad if needed
epochs_eeg.interpolate_bads(reset_bads=True)

# Remove bad trials using pre-identified epoch indices from JSON
# epochs_eeg = drop_from_json(epochs_eeg,
#     "data/idx_epochs_rem_1st_run.json", config.subject_id)

# Linear Detrend in each epoch and channel
epochs_eeg.apply_function(lambda x: detrend(x, type='linear'), picks='all')

# Baseline correction
epochs_eeg.apply_baseline(baseline=config.epoch_baseline)

# Fast ICA
ica_processor = EEGICA(config)
ica_processor.fit_ica(epochs_eeg)
ica_processor.plot_components(epochs_eeg)

############################################################################

# Check and remove eye component
epochs_eeg = ica_processor.apply_ica(epochs_eeg, components_to_remove=[0])

# Baseline correction
epochs_eeg.apply_baseline(baseline=config.epoch_baseline)

# Apply SOUND
epochs_eeg = apply_sound(epochs_eeg, iter_num=5, lambda_val=0.1)

# Set average reference            # checar se epochs_clean.set_eeg_reference("average", projection=False)
epochs_eeg.set_eeg_reference(config.ch_eeg_reference)

# Apply SSP-SIR
epochs_eeg = apply_sspsir(epochs_eeg)

# Downsampling to 1000 and 3000 Hz
epochs_eeg = downsample(epochs_eeg, config.epoch_downsample_freq)
epochs_emg = downsample_emg_channels(epochs_emg, config.epoch_emg_downsample_freq)

# Interpolate again
epochs_eeg = mne.preprocessing.fix_stim_artifact(epochs_eeg, mode='constant', tmin=-0.005, tmax=0.010, baseline=(-0.050, -0.01))

# Filter EEG data
epochs_eeg_filtered = bandpass(epochs_eeg, config.filter_eeg_bandpass, 'eeg')
epochs_eeg_filtered = notch_filter(
    epochs_eeg_filtered, config.filter_notch, band=(58, 62), harmonics=3)

# Filter EMG data
epochs_emg_filtered = bandpass(epochs_emg, config.filter_emg_bandpass, 'emg')
epochs_emg_filtered = notch_filter(
    epochs_emg_filtered, config.filter_notch, band=(58, 62), harmonics=3)

# Verify bad epochs -> skip if already on json
epochs_eeg_filtered.plot()

#########################################################################################

# Remove bad trials using pre-identified epoch indices from JSON
epochs_eeg_filtered = drop_from_json(epochs_eeg_filtered,
    "data/idx_epochs_rem_2nd_run.json", config.subject_id)

# TEP Plots for picked channels
tep_plotter = TEPPlotter(config)
tep_plotter.plot_evoked_by_symbol(
    epochs_eeg_filtered,
    picks=["FC1", "FC5", "C3", "C4", "CP1", "CP5"],
    xlim=(-0.05, 0.2),
    ylim=(-10,10)
)

#########################################################################################
# CROPPED EEG EPOCHS
#########################################################################################

epochs_eeg_filtered_pre_and_post_stim = (
    epochs_eeg_filtered.copy().crop(tmin=-0.05, tmax=0.4)
)

epochs_eeg_filtered_post_stim = (
    epochs_eeg_filtered.copy().crop(tmin=0.015, tmax=0.2)
)

#########################################################################################
# INITIALIZE EXPORTER / WRITER
#########################################################################################

exporter = EpochAnnotationExporter(config)
writer = Writer(config)

#########################################################################################
# FULL EEG EPOCHS
#########################################################################################

writer.save_epochs(
    epochs_eeg_filtered,
    'processed_full'
)

#########################################################################################
# PRE + POST STIM EEG
#########################################################################################

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

#########################################################################################
# POST STIM ONLY EEG
#########################################################################################

eeg_indexes_post, eeg_annotations_post = exporter.extract_annotations(
    epochs_eeg_filtered_post_stim
)

symbols_post = exporter.map_annotations_to_symbols(
    eeg_annotations_post
)

exporter.export_to_mat(
    writer,
    epochs_eeg_filtered_post_stim,
    symbols_post,
    subfolder="processed_post_only"
)

writer.save_epochs(
    epochs_eeg_filtered_post_stim,
    'processed_post_only'
)

#########################################################################################
# EMG
#########################################################################################

emg_epochs_indexes, emg_epochs_annotations = exporter.extract_annotations(
    epochs_emg_filtered
)

writer.save_emg_epochs(
    epochs_emg_filtered,
    'emg_processed'
)
