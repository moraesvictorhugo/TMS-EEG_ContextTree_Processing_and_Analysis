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

raw = mne.io.read_raw_bdf(file_path, preload=True)

# Get metadata and channel names
print(raw.info)
print(raw.ch_names)

# raw.plot(block=True)

# Adjust channel types
raw.set_channel_types({'EMG': 'emg', 'EOG': 'eog'})  # Adjust names as per your data

# Drop non EEG channels
raw.drop_channels(['EMG', 'EOG'])

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

# Notch filter in EEG data
filt_eeg_data = raw.notch_filter(freqs=60, picks=raw.ch_names)

'''
##### Create epochs
'''
# Create epochs
epochs = mne.Epochs(raw, events_from_annot, event_dict, tmin=-0.8, tmax=0.8, preload=True)

# Plot epochs
# epochs.plot(block = True)

'''
##### Average reference
'''
epochs.set_eeg_reference('average')

'''
##### Plot TEPs before ICA (Temporary)
'''
# Plot raw TEPs
epochs = epochs.average()
pf.plot_evoked_eeg_by_channel_groups(
    epochs,
    tmin=-0.1, tmax=0.35,
    ymin=-20, ymax=20,
    ncols=4,
    window_highlights=[(0.010, 0.035, 'orange', 0.3), (0.090, 0.190, 'yellow', 0.3)],
    split_groups=2
)

'''
##### Remove bad or unused channels 
'''
# epochs.drop_channels(['EMG', 'EOG'])

'''
##### Remove bad epochs
'''
# Remove bad epochs
epochs.drop_bad() # verificar parametros

'''
##### First ICA
'''
# Apply ICA (FastICA)
ica = mne.preprocessing.ICA(n_components=20, random_state=97)
ica.fit(epochs)

# Plot ICA components
ica.plot_sources(epochs, show_scrollbars=False, block=True)

# Get fraction of variance explained by all components
explained_var_ratio = ica.get_explained_variance_ratio(epochs)
for channel_type, ratio in explained_var_ratio.items():
    print(f"Fraction of {channel_type} variance explained by all components: {ratio}")

##### Remove bad components
ica.exclude = [0, 1, 10, 13]
epochs_clean = ica.apply(epochs.copy())

# Plot cleaned epochs
epochs_clean.plot(block = True)

'''
##### (Optional and very experimental) PARAFAC decomposition
'''

'''
##### (Optional) Second ICA
'''
# Apply ICA (Infomax)
ica = mne.preprocessing.ICA(method='infomax', n_components=20, random_state=97)
ica.fit(epochs_clean)

# Plot ICA components
ica.plot_sources(epochs_clean, show_scrollbars=False, block=True)

# Remove bad components
ica.exclude = [3, 6, 7, 10, 19]
epochs_clean = ica.apply(epochs_clean.copy())

# Plot cleaned epochs
epochs_clean.plot(block = True)

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
##### Plot TEPs after ICA
'''
# Compute average evoked response
evoked = epochs_clean.average()

# Plot evoked potentials for all EEG channels
pf.plot_evoked_eeg_by_channel_groups(
    evoked,
    tmin=-0.1, tmax=0.35,
    ymin=-20, ymax=20,
    ncols=4,
    window_highlights=[(0.010, 0.035, 'orange', 0.3), (0.090, 0.190, 'yellow', 0.3)],
    split_groups=1
)








# Calc matrix correlation


# Plot 10 teps for selected EEG channels










