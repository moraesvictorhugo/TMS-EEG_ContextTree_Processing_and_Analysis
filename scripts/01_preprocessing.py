import mne
from pytep import apply_sound, apply_sspsir
from scipy.signal import detrend

from tms_eeg.config.environment import setup_plotting_backend
from tms_eeg.config.settings import ProjectConfig
from tms_eeg.io.reader import load_data
from tms_eeg.io.writer import Writer
from tms_eeg.io.yaml_reader import load_decisions
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

# 1. Cubic interpolation of the TMS artifact (−5 to +15 ms, 10 ms anchor)    -> definir qual será a duração
# raw_data = ArtifactRemover(config).remove_tms_artifact(raw_data, mode='cubic')



# Esta função ficou muito melhor, encapsular e usar ela. Em seguida, testar se assim os filtros não dão ringing.
# A CubicSpline interpola o ruído: ela é obrigada a passar por cada amostra das âncoras. As derivadas nas bordas
# do buraco saem desse ruído, e o único segmento cúbico que cobre o buraco de ~12 ms amplifica essas inclinações.
# O resultado é overshoot.


import numpy as np
import mne



def interp_tms_tesa_like(
    raw: mne.io.BaseRaw,
    tms_annotation: str,
    window: tuple = (-0.002, 0.010),  # segundos, relativo ao pulso
    anchor_ms: tuple = (2.0, 2.0),    # equivale ao [1 1] do tesa_interpdata
    order: int = 3,
    picks="eeg",
) -> mne.io.BaseRaw:
    """Interpolação polinomial (cúbica) ao estilo TESA, ajustada só nas âncoras."""
    raw_out = raw.copy().load_data()
    sfreq = raw_out.info["sfreq"]
    tmin, tmax = window

    events, event_id = mne.events_from_annotations(raw_out)
    if tms_annotation not in event_id:
        raise KeyError(f"'{tms_annotation}' não encontrado. Disponíveis: {list(event_id)}")
    tms_events = events[events[:, 2] == event_id[tms_annotation]]

    picks_idx = mne.io.pick._picks_to_idx(raw_out.info, picks)

    s_start = int(np.ceil(tmin * sfreq))
    s_end = int(np.ceil(tmax * sfreq))
    n_pre = max(1, int(round(anchor_ms[0] * 1e-3 * sfreq)))
    n_post = max(1, int(round(anchor_ms[1] * 1e-3 * sfreq)))
    if n_pre + n_post < order + 1:
        raise ValueError("Âncoras insuficientes para o grau do polinômio.")

    data = raw_out._data
    n_times = raw_out.n_times

    for ev in tms_events[:, 0]:
        c = ev - raw_out.first_samp
        a, b = c + s_start, c + s_end            # região interpolada: [a, b)
        pre = np.arange(a - n_pre, a)
        post = np.arange(b, b + n_post)
        if pre[0] < 0 or post[-1] >= n_times:
            print(f"Evento {ev} ignorado: âncoras fora do registro.")
            continue

        anc = np.concatenate([pre, post])
        tgt = np.arange(a, b)

        t_anc = (anc - c) * 1e3 / sfreq          # ms relativo ao pulso
        t_tgt = (tgt - c) * 1e3 / sfreq

        y = data[np.ix_(picks_idx, anc)].T       # (n_anc, n_ch)
        p = np.polyfit(t_anc, y, order)
        data[np.ix_(picks_idx, tgt)] = (np.vander(t_tgt, order + 1) @ p).T

    return raw_out




raw_tesa = interp_tms_tesa_like(raw_data, "Stimulus A", window=(-0.002, 0.010))




# 2. Downsampling to 5000 Hz  -> Pode ser removido daqui se necessário, mas pode tornar o processamento muito pesado
raw_data = downsample(raw_data, config.raw_downsample_freq)

# 3. 60 Hz notch (MNE default)    -> Checar se os filtros vão gerar ringing. Colocar no fim será ruim aplicar o filtro sobre as épocas


# IIR em épocas
epochs.filter(l_freq=62, h_freq=58, method="iir",
              iir_params=dict(order=4, ftype="butter"),
              phase="zero")


# IIR em raw
raw.notch_filter(60, method="iir",
                 iir_params=dict(order=4, ftype="butter"),
                 notch_widths=4, phase="zero")





# ----------    Ajustar o notch de forma a ser menos provável de induzir ringing
data_filtered = notch_filter(raw_data, config.filter_notch)
data_filtered = raw_data
raw_data_emg = data_filtered.copy().pick("emg")

# 4. Split into two copies: A with 0.1 Hz high-pass and B with 1 Hz high-pass
data_filtered_A = raw_data.filter(l_freq=0.1, h_freq=None, fir_design='firwin')
data_filtered_B = data_filtered.filter(l_freq=1, h_freq=None, fir_design='firwin')

# 5. Process annotations and create Epochs from −1000 to +1000 ms
annotation_processor = AnnotationProcessor(config)
data_filtered_A = annotation_processor.process_annotations(data_filtered_A)

epochs_eeg = create_epochs(data_filtered_A, config)
epochs_emg = create_epochs(raw_data_emg, config, modality="emg")

# Check
tep_plotter = TEPPlotter(config)
tep_plotter.plot_evoked_by_symbol(
    epochs_eeg,
    picks=["FC1", "C3", "C4"],
    xlim=(-0.02, 0.1),
    ylim=(-60, 60)
)

epochs_eeg.plot() # -> save bad channels and epochs in decisions

# 6. Mark bad channels and epochs
decisions = load_decisions(config)
epochs_eeg = apply_bad_channels(epochs_eeg, decisions)
epochs_eeg = apply_bad_epochs(epochs_eeg, decisions)

# 7. Average reference
epochs_eeg.set_eeg_reference(config.ch_eeg_reference)

# 8. Replacement of the artifact with a constant (−5 to +10 ms)
epochs_eeg = mne.preprocessing.fix_stim_artifact(epochs_eeg, mode='constant', tmin=-0.005, tmax=0.015, baseline=(-0.050, -0.01))












# 9. Rank calculation: $$32 - n_{bads} - 1$$

#10 ICA fitting (`n_components = rank`)
ica_processor = EEGICA(config)
ica_processor.fit_ica(epochs_eeg)
ica_processor.plot_components(epochs_eeg)

#11 Application of the ICA solution and component removal
epochs_eeg = ica_processor.apply_ica(epochs_eeg, components_to_remove=[0])

#12 SOUND
epochs_eeg = apply_sound(epochs_eeg, iter_num=5, lambda_val=0.1)

#13 SSP-SIR
epochs_eeg = apply_sspsir(epochs_eeg)

#14 Interpolation of removed channels
epochs_eeg.interpolate_bads(reset_bads=True)

#15 Average reference
epochs_eeg.set_eeg_reference(config.ch_eeg_reference)

#16 Cubic interpolation of the TMS artifact (−5 to +10 ms)

#17 80 Hz FIR low-pass and EMG Filter
epochs_eeg_filtered = bandpass(epochs_eeg, config.filter_eeg_bandpass, 'eeg')

epochs_emg_filtered = bandpass(epochs_emg, config.filter_emg_bandpass, 'emg')

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