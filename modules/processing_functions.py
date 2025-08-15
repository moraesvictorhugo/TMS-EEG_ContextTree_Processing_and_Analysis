

# # Calculate peak-to-peak amplitude
import numpy as np

def peak_to_peak_amplitudes_epochs(epochs, channel_name='C3', tmin=0.01, tmax=0.045):
    """
    Calculate peak-to-peak amplitude for each epoch in the specified time window.

    Parameters:
    - epochs: MNE Epochs object containing your preprocessed data.
    - channel_name: Name of the channel to extract data from (default 'C3').
    - tmin: Start time of the window in seconds (default 0.01, i.e., 10 ms).
    - tmax: End time of the window in seconds (default 0.045, i.e., 45 ms).

    Returns:
    - np.array of peak-to-peak amplitudes for each epoch (length = number of epochs).
    """
    # Extract data for the specified channel and time window
    # epochs.get_data() returns shape (n_epochs, n_channels, n_times)
    ch_index = epochs.ch_names.index(channel_name)
    
    # Convert times to sample indices
    times = epochs.times
    start_idx = np.searchsorted(times, tmin)
    end_idx = np.searchsorted(times, tmax)
    
    # Extract data segment
    data = epochs.get_data()[:, ch_index, start_idx:end_idx]
    
    # Calculate peak-to-peak amplitude for each epoch
    ptp_amplitudes = np.ptp(data, axis=1)
    
    return ptp_amplitudes

def peak_to_peak_amplitude_evoked(evoked, channel_name='C3', tmin=0.01, tmax=0.045):
    """
    Calculate the peak-to-peak amplitude of an averaged evoked response 
    in a specific time window and channel.

    Parameters:
    - evoked: MNE Evoked object (average of epochs).
    - channel_name: The channel from which to extract the data (default 'C3').
    - tmin: Start time of the window in seconds (default 0.01 = 10 ms).
    - tmax: End time of the window in seconds (default 0.045 = 45 ms).

    Returns:
    - ptp_amplitude: Peak-to-peak amplitude (max - min) within the time window.
    """

    # Find the index of the specified channel in the evoked data
    ch_index = evoked.ch_names.index(channel_name)

    # Extract the array of time points for the evoked data (in seconds)
    times = evoked.times

    # Convert time window boundaries from seconds to corresponding sample indices
    start_idx = np.searchsorted(times, tmin)
    end_idx = np.searchsorted(times, tmax)

    # Extract the data segment for the channel and time window of interest
    data = evoked.data[ch_index, start_idx:end_idx]

    # Calculate the peak-to-peak amplitude as the difference between max and min values
    ptp_amplitude = np.ptp(data)

    return ptp_amplitude