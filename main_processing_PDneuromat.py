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
highpass_filt = 20
lowpass_filt = 500
notch_freqs = (60, 120, 180, 240, 300, 360)
artefact_escape_ms = 10 / 1000
valor_ms = 60
valor_x = valor_ms / 1000
fs=5000
bool_plot = False
bool_export = False
bool_print = False

# Construct the relative path to the EDF file
file_path = os.path.join(os.getcwd(), 'data_PD_Neuromat', 'thais_tree.edf')

# Read the EDF file
raw = mne.io.read_raw_edf(file_path, preload=True)

# Access the data as a NumPy array
data = raw.get_data()

# Get metadata and channel names
print(raw.info)
print(raw.ch_names)

# Plot the first channel only
if bool_plot:
    raw.plot()

# Apply the notch filter
filt_data = raw.notch_filter(freqs=notch_freqs, trans_bandwidth=2, notch_widths=2, picks=[0])

# Apply a high-pass filter at 20-500 Hz  using an IIR Butterworth filter                       -> IIR x FIR?
filt_data = filt_data.copy().filter(l_freq=highpass_filt, h_freq=lowpass_filt, picks=['Input 33'], method="iir", n_jobs=2, iir_params=dict(order=8, ftype="butter"))

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