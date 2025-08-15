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
from modules import plot_functions as pf
from modules import processing_functions as pcf

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
##### Load data
'''
# Construct the relative path to the EDF file and read it
file_path = '/home/victormoraes/MEGA/Archive/PD FFCLRP-USP/data_PD_Neuromat/TEPs_2025.07.08.bdf'
# file_path = '/home/victormoraes/Downloads/Carlo-TEP-120%-2025.07.30.bdf'

raw = mne.io.read_raw_bdf(file_path, preload=True)

# Get metadata and channel names
print(raw.info)
print(raw.ch_names)

# raw.plot(block=True)

# Adjust channel types
# raw.set_channel_types({'emg': 'emg', 'eog': 'eog'})  # Adjust names as per your data
raw.set_channel_types({'EMG': 'emg', 'EOG': 'eog'})

# raw.plot(block=True, picks=['emg', 'eog'])


# Drop non EEG channels
# raw.drop_channels(['EMG', 'EOG'])

'''
##### Find and create events
'''
# Get events from annotations
events_from_annot, event_dict = mne.events_from_annotations(raw)

# Select the event of interest
target_event_id = event_dict['Stimulus A']  # replace with actual label from event_dict keys

'''
##### Remove TMS artifact using baseline data
'''
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

'''
##### Filter raw EEG data
'''
# Filter raw EEG data (high-pass 1 Hz, low-pass: 250 Hz and notch filter 60 Hz)
eeg_highpass_filt = 1
eeg_lowpass_filt = 250

filt_eeg_data = raw.copy().filter(l_freq=eeg_highpass_filt, h_freq=eeg_lowpass_filt, picks=[
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8',
    'P7', 'P8', 'Pz', 'Iz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9',
    'TP10', 'AFz', 'FCz'
], fir_design='firwin')

# Define the frequencies for the notch filter to remove powerline noise and harmonics
freqs = (60, 120, 180, 240)

# Notch filter in EEG data
filt_eeg_data = filt_eeg_data.notch_filter(freqs=freqs, picks=raw.ch_names)

# Plot PSD 
# filt_eeg_data.plot_psd(fmax=500)

'''
##### Create epochs
'''
# Create epochs
epochs = mne.Epochs(filt_eeg_data, events_from_annot, event_dict, tmin=-0.8, tmax=0.8, preload=True)

# Plot epochs
# epochs.plot(block = True)

'''
##### Average reference
'''
epochs.set_eeg_reference('average')

'''
##### Plot TEPs before ICA (Temporary)
'''
# # Plot raw TEPs
# epochs_beforeICA = epochs.average()
# pf.plot_evoked_eeg_by_channel_groups(
#     epochs_beforeICA,
#     tmin=-0.1, tmax=0.35,
#     ymin=-20, ymax=20,
#     ncols=4,
#     window_highlights=[(0.010, 0.035, 'orange', 0.3), (0.090, 0.190, 'yellow', 0.3)],
#     split_groups=4
# )

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

# Plot ICA components
# ica.plot_sources(epochs, show_scrollbars=False, block=True)

# Get fraction of variance explained by all components
explained_var_ratio = ica.get_explained_variance_ratio(epochs)
for channel_type, ratio in explained_var_ratio.items():
    print(f"Fraction of {channel_type} variance explained by all components: {ratio}")

##### Remove bad components
ica.exclude = [0, 1, 10, 11]               # Indices of the bad components can change in each run
epochs_clean = ica.apply(epochs.copy())

# Plot cleaned epochs
# epochs_clean.plot(block = True)

'''
##### (Optional and very experimental) PARAFAC decomposition
'''

'''
##### (Optional) Second ICA (Infomax)
'''
# # Apply ICA (Infomax)
# ica = mne.preprocessing.ICA(method='infomax', n_components=20, random_state=97)
# ica.fit(epochs_clean)

# # Plot ICA components
# ica.plot_sources(epochs_clean, show_scrollbars=False, block=True)

# # Remove bad components
# ica.exclude = [3, 6, 7, 10, 19]
# epochs_clean = ica.apply(epochs_clean.copy())

# Plot cleaned epochs
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
# # Compute average and standard deviation evoked response
evoked = epochs_clean.average()
std_evoked = epochs_clean.get_data().std(axis=0)

# # Plot evoked potentials for all EEG channels
# pf.plot_evoked_eeg_by_channel_groups(
#     evoked,
#     tmin=-0.1, tmax=0.35,
#     ymin=-20, ymax=20,
#     ncols=4,
#     window_highlights=[(0.010, 0.035, 'orange', 0.3), (0.090, 0.190, 'yellow', 0.3)],
#     split_groups=4
# )

# Plot evoked potential for C3 electrode with standard deviation shading
pf.plot_evoked_with_std(epochs_clean, std_evoked, 'C3', tmin=-0.1, tmax=0.35,
                     highlight_window=(0.015, 0.040))

'''
##### Calculate peak to peak amplitudes
'''
# # Calculate peak to peak amplitudes evoked response in C3 electrode and convert to uV
ptp_value = pcf.peak_to_peak_amplitude_evoked(evoked, channel_name='C3', tmin=0.01, tmax=0.040)
ptp_value = ptp_value * 1e6
print(f"Peak-to-peak amplitude after averaging (uV): {ptp_value}")

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
power.apply_baseline(baseline=(-0.2, 0), mode='logratio')

# Plot time-frequency power for all channels
for ch_name in power.ch_names:
    power.plot(picks=[ch_name], title=f'Time-Frequency Power (Morlet) - Channel: {ch_name}')
    plt.show()  # Ensure each plot is displayed separately
















'''
##### Just exploring ........
'''

# # Plot 10 processed teps for selected EEG channels

# # Select the first 10 epochs only
# epochs_first10 = epochs_clean[:10]

# # Compute the average evoked response from these first 10 epochs
# evoked_first10 = epochs_first10.average()

# # Plot evoked potentials for all EEG channels
# pf.plot_evoked_eeg_by_channel_groups(
#     evoked_first10,
#     tmin=-0.1, tmax=0.35,
#     ymin=-20, ymax=20,
#     ncols=4,
#     window_highlights=[(0.010, 0.035, 'orange', 0.3), (0.090, 0.190, 'yellow', 0.3)],
#     split_groups=4
# )

# # Calc matrix correlation









