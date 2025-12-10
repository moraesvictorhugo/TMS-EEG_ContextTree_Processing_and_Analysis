'''
Pipeline Analysis based on: 

https://pypi.org/project/tmseegpy/0.2.1/
'''
import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg') # TkAgg
from scipy.signal import butter, iirnotch, sosfilt, zpk2sos
from scipy import signal
import os
import pandas as pd
import seaborn as sns
from mne.preprocessing import ICA
from mne_icalabel import label_components
from src.modules import plot_functions as pf
from src.modules import processing_functions as pcf

'''
Order of steps

    Load data                                                                                    
    Find/create events                                                                           
    Drop unused channels (e.g., EMG)                                                             
    Remove TMS artifact using baseline data (window: -2 - 5ms)                                   
    Filter raw EEG data (high-pass 1 Hz, low-pass: 250 Hz and notch filter 60 Hz)
    Create epochs (-0.8 to 0.8)
    Average reference
    Remove bad channels (manual or threshold=3)
    Remove bad epochs (manual or threshold=3)
    First ICA (FastICA)
    (Optional and very experimental) PARAFAC decomposition
    (Optional) Second ICA (Infomax)
    (Optional) SSP
    Filter epoched data (low-pass 45 Hz)
    Downsampling (725 Hz)
    TEP plotting
    PCIst
'''

'''
##### Setting Flags
'''
show_plots = True
export_data = False
tree_sequence = False
path_tree_sequence = ''

'''
##### Load data
'''
# Construct the relative path to the EDF file and read it
# file_path = '/home/victormoraes/MEGA/Archive/PD FFCLRP-USP/data_PD_Neuromat/TEPs_2025.07.08.bdf'                           # Pilot 1
# file_path = '/home/victormoraes/MEGA/Archive/PD FFCLRP-USP/data_PD_Neuromat/Carlo-TEP-100%-2025.07.30.bdf'                 # Pilot 2
# file_path = '/home/victormoraes/MEGA/Archive/PD FFCLRP-USP/data_PD_Neuromat/120%rmt.bdf'                                   # Pilot 3
# file_path = '/home/victormoraes/MEGA/Archive/PD FFCLRP-USP/data_PD_Neuromat/Piloto_13-10-25/100_Limiar_50_pulsos.bdf'      # Pilot 4
# file_path = '/home/victormoraes/MEGA/Archive/PD FFCLRP-USP/data_PD_Neuromat/Piloto_13-10-25/120_Limiar_50_pulsos.bdf'      # Pilot 4
# file_path = '/home/victormoraes/MEGA/Archive/PD FFCLRP-USP/data_PD_Neuromat/Piloto_24-10-25/com_ruido.bdf'                 # Pilot 5
file_path = '/home/victormoraes/MEGA/Archive/PD FFCLRP-USP/data_PD_Neuromat/12_03_2025_Debora/alvo5.bdf'  
# file_path = '/home/victormoraes/MEGA/Archive/PD FFCLRP-USP/data_PD_Neuromat/11_07_2025_VictorMalheiro/120MT_50P.bdf'       # Pilot 6

raw = mne.io.read_raw_bdf(file_path, preload=True)

# Load txt file with event codes into a numpy array
if tree_sequence:
    event_codes = np.loadtxt(path_tree_sequence, dtype=int)

# Get metadata and channel names
print(raw.info)
print(raw.ch_names)

if show_plots:
    raw.plot(block=True)

# Adjust channel types
raw.set_channel_types({'EMG': 'emg', 'EOG': 'eog'})  # Adjust names for Pilot 1
# raw.set_channel_types({'emg': 'emg', 'eog': 'eog'})    # Adjust names for Pilot 2 and 3

# Drop non EEG channels and bad channels
bad_ch =['Iz']
raw.drop_channels(bad_ch)    
raw.drop_channels(['EMG', 'EOG'])  

'''
##### Find and create events
'''
# Get events from annotations
events_from_annot, event_dict = mne.events_from_annotations(raw)

# Replace the event code in the third column of events_from_annot
if tree_sequence:
    events_from_annot[:, 2] = event_codes

# Define the event IDs
if tree_sequence:
    event_id = {
        'stimulus_0': 0,
        'stimulus_1': 1,
        'stimulus_2': 2
    }
else:
    event_id = event_dict['Stimulus A']

'''
##### Remove TMS artifact using baseline data
'''
# Apply fix_stim_artifact using these events
mne.preprocessing.fix_stim_artifact(
    raw,
    events=events_from_annot,
    event_id=event_id,
    tmin=-0.005,
    tmax=0.008,
    mode='linear'
)

# Vizualize the artifact removal
if show_plots:
    raw.plot(block=True)

'''
##### Filter raw EEG data
'''
# Filter raw EEG data (high-pass 1 Hz, low-pass: 250 Hz and notch filter 60 Hz)
eeg_highpass_filt = 1
eeg_lowpass_filt = 250

filt_eeg_data = raw.copy().filter(l_freq=eeg_highpass_filt, h_freq=eeg_lowpass_filt,
                                  picks= raw.ch_names, fir_design='firwin')

# Define the frequencies for the notch filter to remove powerline noise and harmonics
freqs = (60, 120, 180, 240, 300)

# Notch filter in EEG data
filt_eeg_data = filt_eeg_data.notch_filter(freqs=freqs, picks=raw.ch_names)

# Plot PSD 
filt_eeg_data.plot_psd(fmax=500)

# Plot filtered data
filt_eeg_data.plot(block=True)

'''
##### Create epochs
'''
# Create epochs
epochs = mne.Epochs(filt_eeg_data, events_from_annot, event_id, tmin=-0.8, tmax=0.8, preload=True)

# Plot epochs
epochs.plot(block = True)

'''
##### Average reference
'''
epochs.set_eeg_reference('average')

'''
##### Plot TEPs before ICA
'''
if tree_sequence:
    pf.plot_average_epochs_grid(epochs, event_id, tmin=-0.1, tmax=0.35, ymin=-20, ymax=20, n_rows=3, n_cols=3)
else:
    epochs_beforeICA = epochs.average()
    pf.plot_evoked_eeg_by_channel_groups(
        epochs_beforeICA,
        tmin=-0.1, tmax=0.35,
        ymin=-20, ymax=20,
        ncols=4,
        window_highlights=[(0.010, 0.035, 'orange', 0.3)],
        split_groups=4
    )

'''
##### Remove bad or unused channels 
'''
# epochs.drop_channels(['EMG', 'EOG'])

'''
##### Remove bad epochs
'''
# Remove bad epochs
# epochs.drop_bad() # verificar parametros

'''
##### First ICA
'''
# Apply ICA (FastICA)
ica = mne.preprocessing.ICA(n_components=20, random_state=97)
ica.fit(epochs)

##### Developing ---------------------------------------------------------------
# IClabel
# from mne_icalabel import label_components

# # Create the standard 10-20 montage
# montage = mne.channels.make_standard_montage('standard_1020')

# # Set the montage to your raw data and epochs
# raw.set_montage(montage)
# epochs.set_montage(montage)

# # Label components
# label_components(inst=epochs, ica=ica, method='iclabel')

# # Extract labels from ICA object
# labels = ica.labels_

# # Identify non-brain components: exclude all but 'brain' (and optionally 'other')
# exclude_components = [idx for idx, label in enumerate(labels) if label not in ['brain', 'other']]

# # Set components to exclude
# ica.exclude = exclude_components

# # Apply ICA to remove those components from the data
# epochs_clean = ica.apply(epochs.copy())

# # Plot cleaned epochs
# epochs_clean.plot(block = True)

##### Developing ---------------------------------------------------------------

### The artifact component should be excluded manually
# Plot ICA components
ica.plot_sources(epochs, show_scrollbars=False, block=True)

# Get fraction of variance explained by all components
explained_var_ratio = ica.get_explained_variance_ratio(epochs)
for channel_type, ratio in explained_var_ratio.items():
    print(f"Fraction of {channel_type} variance explained by all components: {ratio}")

##### Remove bad components
ica.exclude = [0, 1]
epochs_clean = ica.apply(epochs.copy())

# Plot cleaned epochs
epochs_clean.plot(block = True)

'''
##### (Optional and very experimental) PARAFAC decomposition
'''

'''
##### (Optional) Second ICA (Infomax)
'''
# # Apply ICA (Infomax)
# ica = mne.preprocessing.ICA(method='infomax', n_components=20, random_state=97)
# ica.fit(epochs_clean)

# # # Plot ICA components
# ica.plot_sources(epochs_clean, show_scrollbars=False, block=True)

# # # Remove bad components
# ica.exclude = [0, 1]
# epochs_clean = ica.apply(epochs_clean.copy())

# # Plot cleaned epochs
# epochs_clean.plot(block = True)

'''
##### (Optional) SSP
'''

'''
##### Filter epoched data
'''
# Filter epoched data (low-pass 45 Hz)
epochs_clean = epochs_clean.copy().filter(l_freq=None, h_freq=45)

'''
##### Downsampling
'''
# Downsampling (725 Hz)
epochs_clean = epochs_clean.copy().resample(725)

'''
##### Plot average TEPs after ICA
'''
if tree_sequence:
    pf.plot_average_epochs_grid(epochs_clean, event_id, tmin=-0.1, tmax=0.35, ymin=-20, ymax=20, n_rows=3, n_cols=3)
else:
    ### Compute average and standard deviation evoked response
    evoked = epochs_clean.average()
    std_evoked = epochs_clean.get_data().std(axis=0)

    # # Plot evoked potentials for all EEG channels
    pf.plot_evoked_eeg_by_channel_groups(
        evoked,
        tmin=-0.1, tmax=0.35,
        ymin=-20, ymax=20,
        ncols=4,
        window_highlights=[(0.010, 0.035, 'orange', 0.3)],
        split_groups=4
    )
    ### Plot evoked potential for C3 electrode with standard deviation shading
    pf.plot_evoked_with_std(epochs_clean, std_evoked, 'C3', tmin=-0.1, tmax=0.35,
                        highlight_window=(0.015, 0.040))

#### TO-DO ---------------------------------------------------------------
# Calculate average TEPs grouping clusters of electrodes for example:
# C3, FC1, CP1, FC5, CP5, C1, FC3, CP3 and C5
# C4, FC2, CP2, FC6, CP6, C2, FC4, CP4 and C6
# This approach can potentially increase signal-to-noise ratio

'''
##### Calculate peak to peak amplitudes
'''
# # Calculate peak to peak amplitudes evoked response in C3 electrode and convert to uV
ptp_value = pcf.peak_to_peak_amplitude_evoked(evoked, channel_name='C3', tmin=0.01, tmax=0.040)
ptp_value = ptp_value * 1e6
print(f"Peak-to-peak amplitude after averaging (uV): {ptp_value}")

'''
##### Phase synchrony analysis
'''
from mne_connectivity import spectral_connectivity_epochs
import numpy as np

# Define analysis parameters for phase synchrony
seed_electrode = 'C3'  # primary motor cortex electrode
freq_band = (1, 45)    # 
tmin_sync, tmax_sync = 0.010, 0.045  # N15-P30 latency window

# Crop epochs to N15-P30 window
epochs_sync = epochs_clean.copy().crop(tmin=tmin_sync, tmax=tmax_sync)

# Find index of the seed electrode
seed_idx = epochs_sync.ch_names.index(seed_electrode)

# Extract epoch data as numpy array: (n_epochs, n_channels, n_times)
data = epochs_sync.get_data()

# Create indices tuple for connectivity calculation
# We want connectivity between seed and all other channels
indices = (np.full(len(epochs_sync.ch_names), seed_idx), np.arange(len(epochs_sync.ch_names)))

# Compute phase locking value (PLV) between seed and all other electrodes in alpha band
con = spectral_connectivity_epochs(
    data, sfreq=epochs_sync.info['sfreq'], method='plv', mode='fourier',
    fmin=freq_band[0], fmax=freq_band[1], faverage=True, tmin=tmin_sync, tmax=tmax_sync,
    indices=indices, n_jobs=1)

# 'con' is a SpectralConnectivity object with PLV values of shape (n_connections, n_freqs)

# Print PLV values between C3 and all other electrodes (only one frequency bin since faverage=True)
print("\nPhase Locking Value (PLV) between C3 and other electrodes (alpha band, 15-40ms):")
for ch_idx, ch_name in enumerate(epochs_sync.ch_names):
    print(f'{seed_electrode} <-> {ch_name}: {con.get_data()[ch_idx, 0]:.3f}')

'''
##### Time-frequency analysis  --- needs to be finished!
'''    
# Time-frequency analysis of TMS-evoked potentials using Morlet wavelets
frequencies = np.arange(1, 45, 3)  # frequencies from 7 to 30 Hz, step 3 Hz
n_cycles = frequencies / 2.0  # number of cycles per frequency

# Compute induced power using Morlet wavelets, average across trials
power = epochs_clean.compute_tfr('morlet', freqs=frequencies, n_cycles=n_cycles,
                           use_fft=True, return_itc=False, decim=3, average=True)

# Optionally baseline correct power (e.g., using -0.2 to 0 seconds pre-stimulus)
power.apply_baseline(baseline=(-0.1, 0), mode='logratio')

# Plot time-frequency power for all channels
for ch_name in power.ch_names:
    power.plot(picks=[ch_name], title=f'Time-Frequency Power (Morlet) - Channel: {ch_name}')
    plt.show()  # Ensure each plot is displayed separately