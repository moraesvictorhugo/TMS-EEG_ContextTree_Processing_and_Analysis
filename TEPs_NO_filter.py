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
raw.drop_channels(['EOG'])

'''
##### Find and create events
'''
# Get events from annotations
events_from_annot, event_dict = mne.events_from_annotations(raw)

# Select the event of interest
target_event_id = event_dict['Stimulus A']  # replace with actual label from event_dict keys

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
average_epochs = epochs.average()

pf.plot_evoked_eeg_by_channel_groups(
    average_epochs,
    tmin=-0.1, tmax=0.35,
    ymin=-20, ymax=20,
    ncols=4,
    window_highlights=[(0.010, 0.035, 'orange', 0.3), (0.090, 0.190, 'yellow', 0.3)],
    split_groups=4
)

# # Plot first 10 epochs
# average_10_epochs = epochs[:10]

# pf.plot_evoked_eeg_by_channel_groups(
#     average_10_epochs,
#     tmin=-0.1, tmax=0.35,
#     ymin=-20, ymax=20,
#     ncols=4,
#     window_highlights=[(0.010, 0.035, 'orange', 0.3), (0.090, 0.190, 'yellow', 0.3)],
#     split_groups=4
# )
