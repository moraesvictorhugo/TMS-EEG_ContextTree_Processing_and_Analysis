'''
Pipeline Analysis based on: 

https://pypi.org/project/tmseegpy/0.2.1/
'''

import mne
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg') # Qt5Agg
from scipy.signal import butter, iirnotch, sosfilt, zpk2sos
from scipy import signal
import os
import pandas as pd
import seaborn as sns
from mne.preprocessing import ICA
from mne_icalabel import label_components

# Construct the relative path to the EDF file
file_path = '/home/victormoraes/MEGA/Archive/PD FFCLRP-USP/data_PD_Neuromat/TEPs_2025.07.08.bdf'

'''
Order of steps

    Load data                                                                                    ok
    Find/create events                                                                           ok
    Drop unused channels (e.g., EMG)                                                             ok
    Remove TMS artifact using baseline data (window: -2 - 5ms)                                   ok
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

# Load data
raw = mne.io.read_raw_bdf(file_path, preload=True)

# Get metadata and channel names
print(raw.info)
print(raw.ch_names)

# raw.plot(block=True)

# Adjust channel types
raw.set_channel_types({'EMG': 'emg', 'EOG': 'eog'})  # Adjust names as per your data

# Drop non EEG channels
raw.drop_channels(['EMG', 'EOG'])

# Get events from annotations
events_from_annot, event_dict = mne.events_from_annotations(raw)

# Select the event of interest
target_event_id = event_dict['Stimulus A']  # replace with actual label from event_dict keys

# Apply fix_stim_artifact using these events
mne.preprocessing.fix_stim_artifact(
    raw,
    events=events_from_annot,
    event_id=target_event_id,
    tmin=-0.002,
    tmax=0.005,
    mode='linear'
)

# raw.plot(block=True)

# Filter raw EEG data (high-pass 1 Hz, low-pass: 250 Hz and notch filter 60 Hz)
eeg_highpass_filt = 1
eeg_lowpass_filt = 250

filt_eeg_data = raw.copy().filter(l_freq=eeg_highpass_filt, h_freq=eeg_lowpass_filt, picks=[
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8',
    'P7', 'P8', 'Pz', 'Iz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9',
    'TP10', 'AFz', 'FCz'
], fir_design='firwin')

# Notch filter in EEG data
filt_eeg_data = raw.notch_filter(freqs=60, picks=raw.ch_names)

# Create epochs
epochs = mne.Epochs(raw, events_from_annot, event_dict, tmin=-0.8, tmax=0.8, preload=True)

# Plot epochs
epochs.plot(block = True)

# Average reference
epochs.set_eeg_reference('average')

# Remove bad channels
# epochs.drop_channels(['EMG', 'EOG'])

# Remove bad epochs
epochs.drop_bad() # verificar parametros

# Apply ICA (FastICA)
ica = mne.preprocessing.ICA(n_components=20, random_state=97)
ica.fit(epochs)

# Plot ICA components
ica.plot_sources(epochs, show_scrollbars=False, block=True)

# (Optional and very experimental) PARAFAC decomposition

# (Optional) Second ICA (Infomax)

# (Optional) SSP

# Filter epoched data (low-pass 45 Hz)
epochs = epochs.copy().filter(l_freq=None, h_freq=45)

# Downsampling (725 Hz)
epochs = epochs.copy().resample(725)

# TEP plotting
# Compute average evoked response
evoked = epochs.average()

# Plot evoked potentials for all EEG channels

### Plot evoked potentials for selected EEG channels
eeg_channels = [ch for ch in evoked.ch_names if evoked.get_channel_types(picks=ch)[0] == 'eeg']
n_channels = len(eeg_channels)

# Define time window in seconds and set y-axis limits
tmin, tmax = -0.1, 0.35
ymin, ymax = -20, 20

# Find indices corresponding to this time window
time_mask = (evoked.times >= tmin) & (evoked.times <= tmax)
times = evoked.times[time_mask]

# Split channels roughly in half for two windows
split_idx = n_channels // 2
channel_groups = [eeg_channels[:split_idx], eeg_channels[split_idx:]]

for win_idx, ch_group in enumerate(channel_groups, start=1):
    n_ch = len(ch_group)
    ncols = 4
    nrows = int(np.ceil(n_ch / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 2))
    axes = axes.flatten()

    for i, ch in enumerate(ch_group):
        ch_idx = evoked.ch_names.index(ch)
        # Extract data in the time window and convert to µV
        data = evoked.data[ch_idx, time_mask] * 1e6
        axes[i].plot(times, data)
        axes[i].set_ylim(ymin, ymax)
        axes[i].set_title(ch)
        axes[i].axvline(0, color='r', linestyle='--')  # stimulus onset
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('Amplitude (µV)')
        axes[i].grid(True)
        # Highlight region between 0.015 and 0.060 seconds
        axes[i].axvspan(0.015, 0.060, color='yellow', alpha=0.3)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f'Average Evoked EEG Signals - Window {win_idx}')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
    plt.show()
















# -------------------------------------------------------------------------------



# Load the standard montage
montage = mne.channels.make_standard_montage('standard_1020')

# Drop unused channels
raw.pick_channels(raw.ch_names[:32])

# Apply the montage
raw.set_montage(montage)

# Plot before referencing
raw.plot(picks=raw.ch_names, scalings='auto', title='Raw EEG Data', show=True)

# Reference the data
raw.set_eeg_reference('average')

# Plot after referencing
raw.plot(picks=raw.ch_names, scalings='auto', title='Referenced Raw EEG Data', show=True)

# Apply the notch filter in EEG data
filt_data = raw.notch_filter(freqs=60, picks=raw.ch_names)

# Apply bandpass filters to EEG, EOG, and EMG data
filt_eeg_data = filt_data.copy().filter(l_freq=eeg_highpass_filt, h_freq=None, picks=[
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8',
    'P7', 'P8', 'Pz', 'Iz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9',
    'TP10', 'AFz', 'FCz'
], fir_design='firwin')

filt_eog_data = filt_data.copy().filter(l_freq=eog_highpass_filt, h_freq=eog_lowpass_filt,
    picks=['EOG'], method="iir", n_jobs=2, iir_params=dict(order=8, ftype="butter"))

filt_emg_data = filt_data.copy().filter(l_freq=emg_highpass_filt, h_freq=emg_lowpass_filt, 
    picks=['EMG'], method="iir", n_jobs=2, iir_params=dict(order=8, ftype="butter"))

# Plot filtered data
if bool_plot:
    filt_eeg_data.plot(picks=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8',
                            'P7', 'P8', 'Pz', 'Iz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9',
                            'TP10', 'AFz', 'FCz'], scalings='auto', title='Filtered EEG Data', show=True)

    filt_emg_data.plot(picks=['EMG'], scalings='auto', title='Filtered EMG Data', show=True)

    filt_eog_data.plot(picks=['EOG'], scalings='auto', title='Filtered EOG Data', show=True)



# Plot epochs
# epochs.plot()

# ICA to remove artifacts
ica = mne.preprocessing.ICA(n_components=20, random_state=97)
ica.fit(epochs)

# Label components using ICLabel
labels = label_components(epochs, ica, method='iclabel')

# Find components correlated with EOG channels
eog_indices, eog_scores = ica.find_bads_eog(epochs, ch_name='EOG')
ica.exclude = eog_indices

# Apply ICA to remove artifacts and plot components
epochs_clean = ica.apply(epochs.copy())
# ica.plot_components(inst=epochs_clean, show=True)

# Baseline correction
epochs_clean = epochs_clean.apply_baseline(baseline=(None, 0))

# Apply low pass filter
# epochs_clean = epochs_clean.copy().filter(l_freq=None, h_freq=80)

# Compute average evoked response
evoked = epochs_clean.average()

# Plot evoked potentials for all EEG channels
# evoked.plot(spatial_colors=True, time_unit='s', titles='Average Evoked Response (TEPs)')

# Optional: plot topographic maps at selected latencies
# evoked.plot_topomap(times=[0.01, 0.03, 0.05, 0.07], ch_type='eeg', time_unit='s')

### Plot evoked potentials for selected EEG channels
eeg_channels = [ch for ch in evoked.ch_names if evoked.get_channel_types(picks=ch)[0] == 'eeg']
n_channels = len(eeg_channels)

# Define time window in seconds and set y-axis limits
tmin, tmax = -0.1, 0.35
ymin, ymax = -40, 40

# Find indices corresponding to this time window
time_mask = (evoked.times >= tmin) & (evoked.times <= tmax)
times = evoked.times[time_mask]

# Split channels roughly in half for two windows
split_idx = n_channels // 2
channel_groups = [eeg_channels[:split_idx], eeg_channels[split_idx:]]

for win_idx, ch_group in enumerate(channel_groups, start=1):
    n_ch = len(ch_group)
    ncols = 4
    nrows = int(np.ceil(n_ch / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * 2))
    axes = axes.flatten()

    for i, ch in enumerate(ch_group):
        ch_idx = evoked.ch_names.index(ch)
        # Extract data in the time window and convert to µV
        data = evoked.data[ch_idx, time_mask] * 1e6
        axes[i].plot(times, data)
        axes[i].set_ylim(ymin, ymax)
        axes[i].set_title(ch)
        axes[i].axvline(0, color='r', linestyle='--')  # stimulus onset
        axes[i].set_xlabel('Time (s)')
        axes[i].set_ylabel('Amplitude (µV)')
        axes[i].grid(True)
        # Highlight region between 0.015 and 0.060 seconds
        axes[i].axvspan(0.015, 0.060, color='yellow', alpha=0.3)

    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.suptitle(f'Average Evoked EEG Signals - Window {win_idx}')
    plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
    plt.show()



