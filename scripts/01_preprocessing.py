from pytep import apply_sound, apply_sspsir
from scipy.signal import detrend

from tms_eeg.config.environment import setup_plotting_backend
from tms_eeg.config.settings import ProjectConfig
from tms_eeg.io.reader import load_data
from tms_eeg.io.writer import Writer
from tms_eeg.io.yaml_reader import load_bad_channels, load_decisions
from tms_eeg.preprocessing.annotation_exporter import EpochAnnotationExporter
from tms_eeg.preprocessing.annotation_processor import AnnotationProcessor
from tms_eeg.preprocessing.artifacts import ArtifactRemover
from tms_eeg.preprocessing.cleaning import (
    apply_bad_channels,
    apply_bad_epochs,
    apply_ica_exclude,
)
from tms_eeg.preprocessing.downsampling import downsample, downsample_emg_channels
from tms_eeg.preprocessing.epoching import create_epochs, drop_from_json
from tms_eeg.preprocessing.filtering import bandpass, notch_filter
from tms_eeg.preprocessing.ica import EEGICA
from tms_eeg.visualization.tep_plots import TEPPlotter

setup_plotting_backend()

# 0. Loading and setting up channels and montage
config = ProjectConfig(subject_id="V00")

raw_data = load_data(config)

raw_data.set_channel_types({
    config.ch_eog_label: "eog",
    config.ch_emg_label: "emg",
})
raw_data.set_montage(config.ch_eeg_montage)

# 1. Cubic interpolation of the TMS artifact
raw_data = ArtifactRemover(config).remove_tms_artifact(raw_data, mode="tesa")

# 2. Downsampling to 5000 Hz
raw_data = downsample(raw_data, config.raw_downsample_freq)

# 4. Split into two copies: A with 0.1 Hz high-pass and B with 1 Hz high-pass
data_filtered_A = bandpass(
    raw_data.copy(), band=(0.1, None), ch_type="eeg", method="iir"
)
data_filtered_B = bandpass(
    raw_data.copy(), band=(1, None), ch_type="eeg", method="iir"
)

# 5. Process annotations and create Epochs from −1000 to +1000 ms
annotation_processor = AnnotationProcessor(config)
data_filtered_A = annotation_processor.process_annotations(data_filtered_A)
data_filtered_B = annotation_processor.process_annotations(data_filtered_B)

epochs_eeg = create_epochs(data_filtered_A, config)
epochs_ica = create_epochs(data_filtered_B, config)

epochs_emg = create_epochs(data_filtered_A, config, modality="emg")
epochs_emg.pick([config.ch_emg_label])

# Check
tep_plotter = TEPPlotter(config)
tep_plotter.plot_evoked_by_symbol(
    epochs_eeg,
    picks=["FC1", "C3", "C4"],
    xlim=(-0.02, 0.1),
    ylim=(-20, 20)
)

epochs_eeg.plot() # -> save bad channels and epochs in decisions















# 6. Mark bad channels and epochs
decisions = load_decisions(config)

epochs_eeg = apply_bad_channels(epochs_eeg, decisions)
epochs_eeg = apply_bad_epochs(epochs_eeg, decisions)

epochs_ica = apply_bad_channels(epochs_ica, decisions)
epochs_ica = apply_bad_epochs(epochs_ica, decisions)

# Garantir que as épocas correspondem antes do ajuste



import numpy as np




if not np.array_equal(epochs_eeg.events, epochs_ica.events):
    raise ValueError("As épocas A e B não correspondem.")

# 7. Average reference
epochs_eeg.set_eeg_reference(config.ch_eeg_reference)
epochs_ica.set_eeg_reference(config.ch_eeg_reference)

# Ajustar na versão de 1 Hz
ica_processor = EEGICA(config)
ica_processor.fit_ica(epochs_ica)
ica_processor.plot_components(epochs_ica)

# Aplicar a solução na versão de 0,1 Hz
epochs_eeg = ica_processor.apply_ica(
    epochs_eeg,
    components_to_remove=[0],
)





decisions = load_decisions(config)
print("Decisões carregadas:", decisions)

epochs_eeg = apply_bad_channels(epochs_eeg, decisions)
print("A após canais:", epochs_eeg.info["bads"])

epochs_ica = apply_bad_channels(epochs_ica, decisions)
print("B após canais:", epochs_ica.info["bads"])

epochs_eeg = apply_bad_epochs(epochs_eeg, decisions)
print("A após épocas:", epochs_eeg.info["bads"])

epochs_ica = apply_bad_epochs(epochs_ica, decisions)
print("B após épocas:", epochs_ica.info["bads"])

epochs_eeg.set_eeg_reference(config.ch_eeg_reference)
epochs_ica.set_eeg_reference(config.ch_eeg_reference)
print("Após referência:", epochs_eeg.info["bads"], epochs_ica.info["bads"])




print("A:", len(epochs_eeg), "épocas")
print("B:", len(epochs_ica), "épocas")
print("Índices no YAML:", decisions["bad_epochs"])
print("Canais ruins:", epochs_eeg.info["bads"], epochs_ica.info["bads"])












#12 SOUND
epochs_eeg = apply_sound(epochs_eeg, iter_num=5, lambda_val=0.1)

#13 SSP-SIR
epochs_eeg = apply_sspsir(epochs_eeg)

#14 Interpolation of removed channels
epochs_eeg.interpolate_bads(reset_bads=True)

#15 Average reference
epochs_eeg.set_eeg_reference(config.ch_eeg_reference)

#16 Cubic interpolation of the TMS artifact (−5 to +12 ms) -> um pouco maior do que no anterior?
raw_data = ArtifactRemover(config).remove_tms_artifact(raw_data, mode="tesa")

#17 80 Hz FIR low-pass and EMG Filter
epochs_eeg_filtered = bandpass(epochs_eeg, config.filter_eeg_bandpass, 'eeg')

epochs_emg_filtered = bandpass(epochs_emg, config.filter_emg_bandpass, 'emg')


# Notch aqui??
epochs = bandpass(epochs, cfg.filter_eeg_1st_bandpass, "eeg",
                  method=cfg.filter_method,
                  iir_order=cfg.filter_iir_order)




# 18. Baseline from −300 to −20 ms
epochs_eeg.apply_baseline(baseline=config.epoch_baseline)

# 19. Resampling EEG to 500 Hz and EMG 3000 Hz
epochs_eeg = downsample(epochs_eeg, config.epoch_eeg_downsample_freq)
epochs_emg = downsample_emg_channels(epochs_emg, config.epoch_emg_downsample_freq)

# Check TEP quality
tep_plotter = TEPPlotter(config)
tep_plotter.plot_evoked_by_symbol(
    epochs_eeg_filtered,
    picks=["FC1", "FC5", "C3", "C4", "CP1", "CP5"],
    xlim=(-0.05, 0.2),
    ylim=(-10,10)
)

# 20. Cropping to −800 to +800 ms
epochs_eeg_filtered_pre_and_post_stim = (
    epochs_eeg_filtered.copy().crop(tmin=-0.05, tmax=0.4))

epochs_eeg_filtered_post_stim = (
    epochs_eeg_filtered.copy().crop(tmin=0.015, tmax=0.2))

# 21. Export   -> checar: o drop de epocas 
exporter = EpochAnnotationExporter(config)
writer = Writer(config)

writer.save_epochs(epochs_eeg_filtered, 'processed_full')  # Full Epochs -> precisa de um objeto lá atrás de antes da remoção das epocas

# Pre and Pos Stim ---------------------------------------------------------
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


# Pos Stim only ---------------------------------------------------------

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


# EMG ---------------------------------------------------------------------
emg_epochs_indexes, emg_epochs_annotations = exporter.extract_annotations(
    epochs_emg_filtered
)

writer.save_emg_epochs(
    epochs_emg_filtered,
    'emg_processed'
)