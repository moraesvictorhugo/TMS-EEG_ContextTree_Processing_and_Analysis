'''
Main script for data processing

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

### Setting flags and parameters
emg_highpass_filt = 10
emg_lowpass_filt = 450
eeg_highpass_filt = 1
eeg_lowpass_filt = 250
eog_highpass_filt = 0.1
eog_lowpass_filt = 30
notch_freqs = (60, 120, 180, 240, 300, 360)
artefact_escape_ms = 10 / 1000
valor_ms = 60
valor_x = valor_ms / 1000
fs=5000
bool_plot = False
bool_export = False
bool_print = False

# Construct the relative path to the EDF file
file_path = '/home/victormoraes/MEGA/Archive/PD FFCLRP-USP/data_PD_Neuromat/TEPs_2025.07.08.bdf'

# file_path = os.path.join(os.getcwd(), 'data_PD_Neuromat', 'thais_tree.edf')

# Read the EDF file
raw = mne.io.read_raw_bdf(file_path, preload=True)

# # Access the data as a NumPy array
# data = raw.get_data()

# Get metadata and channel names
print(raw.info)
print(raw.ch_names)

# Plot the first channel only
if bool_plot:
    raw.plot(picks=['EMG'], scalings='auto', title='Raw EMG Data', show=True)
    raw.plot(picks=['C3'], scalings='auto', title='Raw C3 EEG Data', show=True)
    raw.plot(picks=['EOG'], scalings='auto', title='Raw EOG Data', show=True)

# Apply the notch filter in EEG data
filt_data = raw.notch_filter(freqs=60, picks=raw.ch_names)

# Apply bandpass filters to EEG, EOG, and EMG data
filt_eeg_data = filt_data.copy().filter(l_freq=eeg_highpass_filt, h_freq=eeg_lowpass_filt, picks=[
    'Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8',
    'P7', 'P8', 'Pz', 'Iz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9',
    'TP10', 'AFz', 'FCz'
], fir_design='firwin')

filt_eog_data = filt_data.copy().filter(l_freq=eog_highpass_filt, h_freq=eog_lowpass_filt,
    picks=['EOG'], method="iir", n_jobs=2, iir_params=dict(order=8, ftype="butter"))

filt_emg_data = filt_data.copy().filter(l_freq=emg_highpass_filt, h_freq=emg_lowpass_filt, 
    picks=['EMG'], method="iir", n_jobs=2, iir_params=dict(order=8, ftype="butter"))

# Get events from annotations and create epochs
events_from_annot, event_dict = mne.events_from_annotations(raw)

epochs = mne.Epochs(raw, events_from_annot, event_dict, tmin=-0.2, tmax=0.5, preload=True)

# Plot epochs
epochs.plot()

# Average epochs to get a single epoch
evoked = epochs.average()
evoked.plot()




















# Plot filtered data
filt_eeg_data.plot(picks=['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T7', 'T8',
                          'P7', 'P8', 'Pz', 'Iz', 'FC1', 'FC2', 'CP1', 'CP2', 'FC5', 'FC6', 'CP5', 'CP6', 'TP9',
                          'TP10', 'AFz', 'FCz'], scalings='auto', title='Filtered EEG Data', show=True)

filt_emg_data.plot(picks=['EMG'], scalings='auto', title='Filtered EMG Data', show=True)

filt_eog_data.plot(picks=['EOG'], scalings='auto', title='Filtered EOG Data', show=True)

# Apply ICA to EEG data
ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(filt_eeg_data)
filt_eeg_data = ica.apply(filt_eeg_data)






















# Plot filtered data
if bool_plot:
    filt_data.plot()

# Extract data for the first channel
apb_filt_data = filt_data.get_data()

# Get trigger instant times
onsets = raw.annotations.onset
descriptions = raw.annotations.description
# print(raw.annotations.onset)

# Get trigger instant times for each trigger
trigger_a_onsets = [onsets[i] for i in range(len(descriptions)) if "A" in descriptions[i]]
trigger_a_onsets = [float(onset) for onset in trigger_a_onsets]

trigger_b_onsets = [onsets[i] for i in range(len(descriptions)) if "B" in descriptions[i]]
trigger_b_onsets = [float(onset) for onset in trigger_b_onsets]

# apb_filt_data = filt_data.copy().pick('Input 33').get_data()

# Create an empty list to store the max and min MEP values
max_apb_emgVal = []
min_apb_emgVal = []

filtered_signal = apb_filt_data.T

for i in trigger_a_onsets:
    # Calcular os índices com base no tempo
    indice_trigger = int((i + artefact_escape_ms) * fs)
    indice_final_busca = int((i + valor_x) * fs)
    
    max_apb_emgVal.append(np.max(filtered_signal[indice_trigger:indice_final_busca]))
    min_apb_emgVal.append(np.min(filtered_signal[indice_trigger:indice_final_busca]))
    
    # Create a scatter plot for the start of the searching
    plt.scatter(indice_trigger, np.zeros_like(indice_trigger), color='k', marker='o', label='Trigger')

    # Create a scatter plot for the end of the searching interval
    plt.scatter(indice_final_busca, np.zeros_like(indice_final_busca), color='red', marker='o', label='Final Busca')

indices = np.array(raw.annotations.onset) * fs

plt.plot(filtered_signal)

for index in indices:
    plt.axvline(x=index, color='green', linestyle='--', label='Event')
   
plt.show()
   
# Converter para arrays numpy se necessário
max_apb_emgVal = np.array(max_apb_emgVal)
min_apb_emgVal = np.array(min_apb_emgVal)

# Get MEP peak-to-peak value in microvolts (uV)
apb_MEPs = (max_apb_emgVal - min_apb_emgVal) *1e6

# Create data frame to concatenate MEPs and the sequence
file_path = os.path.join(os.getcwd(), 'data_PD_Neuromat', 'tree_sequence_246.txt')
sequence = np.genfromtxt(file_path, dtype=int)

# Concatenate MEPs and sequence in different columns
apb_MEPs_sequence = np.column_stack((apb_MEPs, sequence))

# Create column names
column_names = ['MEP', 'stochastic_chain_info']

# Create a DataFrame
df = pd.DataFrame(apb_MEPs_sequence, columns=column_names)

# if bool_export:
#     np.savetxt('MEP_microvoltsvalues_CR.txt', apb_MEPs_sequence, fmt='%.5f')

'''
# ========================================================================
Analysis of the data -> should be another script
# ========================================================================
'''
def create_context_column(df):
    # Initialize the context column with NaN and set its dtype to object
    df['context'] = np.nan  # Initialize with NaN
    df['context'] = df['context'].astype(object)  # Set dtype to object
    
    # Iterate through the DataFrame
    for i in range(1, len(df)):
        # Get the value of stochastic_chain_info from the previous rows
        prev_value = df['stochastic_chain_info'].iloc[i - 1]
        
        if prev_value == 1:
            # If previous value is 1, check further up
            if i > 1:
                prev_value = df['stochastic_chain_info'].iloc[i - 2]
                if prev_value == 0:
                    df.loc[i, 'context'] = '01'  # Use .loc for assignment
                elif prev_value == 1:
                    df.loc[i, 'context'] = '11'  # Use .loc for assignment
                elif prev_value == 2:
                    df.loc[i, 'context'] = '21'  # Use .loc for assignment
        elif prev_value == 0:
            df.loc[i, 'context'] = '0'      # Use .loc for assignment
        elif prev_value == 2:
            df.loc[i, 'context'] = '2'      # Use .loc for assignment
    
    return df

# Create the context column
df = create_context_column(df)

# Agrupa por 'context' e cria boxplot da coluna 'MEP'
grouped = df.groupby('context')['MEP']

fig, ax = plt.subplots(figsize=(8,6), dpi=200)
ax.boxplot([group.values for name, group in grouped], tick_labels=grouped.groups.keys())
ax.set_title('MEPs por Contexto (n = 300)')
ax.set_xlabel('Contexto')
ax.set_ylabel('MEP amplitude [µV]')
# plt.show()

# -------------------------------------------------------------
## Dividir em primeiro terço e terceiro terço
tamanho = len(df)
terco = tamanho // 3

parte1 = df.iloc[:terco]
parte2 = df.iloc[terco:2*terco]
parte3 = df.iloc[2*terco:]

# # Agrupa por 'context' e cria boxplot da coluna 'MEP'
# grouped = parte1.groupby('context')['MEP']

# fig, ax = plt.subplots(figsize=(8,6))
# ax.boxplot([group.values for name, group in grouped], labels=grouped.groups.keys())
# ax.set_title('Boxplot de MEP por Contexto')
# ax.set_xlabel('Contexto')
# ax.set_ylabel('MEP')
# plt.show()


# grouped = parte2.groupby('context')['MEP']

# fig, ax = plt.subplots(figsize=(8,6))
# ax.boxplot([group.values for name, group in grouped], labels=grouped.groups.keys())
# ax.set_title('Boxplot de MEP por Contexto')
# ax.set_xlabel('Contexto')
# ax.set_ylabel('MEP')
# plt.show()

# grouped = parte3.groupby('context')['MEP']

# fig, ax = plt.subplots(figsize=(8,6))
# ax.boxplot([group.values for name, group in grouped], labels=grouped.groups.keys())
# ax.set_title('Boxplot de MEP por Contexto')
# ax.set_xlabel('Contexto')
# ax.set_ylabel('MEP')
# plt.show()

# Plot the difference betwen parte 3 and parte 1 for each context
parte1_ctx = parte1.groupby('context')['MEP'].mean()
parte3_ctx = parte3.groupby('context')['MEP'].mean()

mep_difference = parte1_ctx - parte3_ctx
mep_difference = pd.DataFrame(mep_difference)

# Create boxplot of seaborn
plt.figure(figsize=(8,6), dpi=200)

# Create the boxplot
sns.boxplot(x='context', y='MEP', data=mep_difference, hue='context', palette="pastel", dodge=False, legend=False, width=0.5)

# # Customize the plot
# plt.xlabel(xlabel)
# plt.ylabel(ylabel)

# Add a dashed line at y=0
plt.axhline(y=0, color='gray', linestyle='--', linewidth=1)

# Remove spines on the right and top
sns.despine()

# Show the plot
# plt.show()


# -------------------------------------------------------------

# Create a context column based on a adjusted criteria -> should be fixed the 1st row
def add_context_adj(df):
    # Initialize the context column with NaN and set its dtype to object
    df['context_adj'] = np.nan  # Initialize with NaN
    df['context_adj'] = df['context_adj'].astype(object)  # Set dtype to object
    
    # Iterate through the DataFrame
    for i in range(1, len(df)):
        # Get the value of stochastic_chain_info from the previous rows
        prev_value = df['stochastic_chain_info'].iloc[i]
        
        if prev_value == 1:
            # If previous value is 1, check further up
            if i > 1:
                prev_value = df['stochastic_chain_info'].iloc[i - 1]
                if prev_value == 0:
                    df.loc[i, 'context_adj'] = '01'  # Use .loc for assignment
                elif prev_value == 1:
                    df.loc[i, 'context_adj'] = '11'  # Use .loc for assignment
                elif prev_value == 2:
                    df.loc[i, 'context_adj'] = '21'  # Use .loc for assignment
        elif prev_value == 0:
            df.loc[i, 'context_adj'] = '0'      # Use .loc for assignment
        elif prev_value == 2:
            df.loc[i, 'context_adj'] = '2'      # Use .loc for assignment
    
    return df

result_df = add_context_adj(df)
print(result_df)

single_pulses = result_df[result_df['context_adj'].isin(['01', '11', '21'])]

# Agrupa por 'context' e cria boxplot da coluna 'MEP'
grouped_adj = single_pulses.groupby('context_adj')['MEP']

fig, ax = plt.subplots(figsize=(8,6), dpi=200)
ax.boxplot([group.values for name, group in grouped_adj], tick_labels=grouped_adj.groups.keys())
ax.set_title('Efeito dos pulsos inibitório, simples e excitatório sobre o simples')
ax.set_xlabel('Sequência de pulsos')
ax.set_ylabel('MEP amplitude [µV]')
plt.show()


# inib_exct = result_df[~result_df['context_adj'].isin(['01', '11', '21'])]

# # Agrupa por 'context' e cria boxplot da coluna 'MEP'
# grouped_adj = inib_exct.groupby('context_adj')['MEP']

# fig, ax = plt.subplots(figsize=(8,6), dpi=200)
# ax.boxplot([group.values for name, group in grouped_adj], tick_labels=grouped_adj.groups.keys())
# ax.set_title('Efeito dos pulsos inibitório, simples e excitatório sobre o simples')
# ax.set_xlabel('Sequência de pulsos')
# ax.set_ylabel('MEP amplitude [µV]')
# plt.show()