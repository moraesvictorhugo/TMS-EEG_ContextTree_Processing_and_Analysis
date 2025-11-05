import numpy as np
from scipy.signal import find_peaks

def peak_to_peak_amplitude_evoked(evoked, channel_name='C3', tmin=0.01, tmax=0.045):
    """
    Calculate the peak-to-peak amplitude and standard deviation of an averaged evoked response 
    in a specific time window and channel.

    Parameters:
    - evoked: MNE Evoked object (average of epochs).
    - channel_name: The channel from which to extract the data (default 'C3').
    - tmin: Start time of the window in seconds (default 0.01 = 10 ms).
    - tmax: End time of the window in seconds (default 0.045 = 45 ms).

    Returns:
    - ptp_amplitude: Peak-to-peak amplitude (max - min) within the time window.
    - std_dev: Standard deviation of the signal within the time window.
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

    # Calculate the standard deviation of the data segment
    std_dev = np.std(data)

    return ptp_amplitude, std_dev
