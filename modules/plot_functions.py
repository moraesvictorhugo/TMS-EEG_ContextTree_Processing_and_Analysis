import numpy as np
import matplotlib.pyplot as plt

def plot_evoked_eeg_by_channel_groups(
    evoked,
    tmin=-0.1,
    tmax=0.35,
    ymin=-20,
    ymax=20,
    ncols=4,
    figsize_per_row=2,
    window_highlights=None,
    split_groups=2,
    suptitle_base="Average Evoked EEG Signals"
):
    """
    Plot evoked EEG potentials for windows of selected channels, grouped for clear visualization.

    Parameters
    ----------
    evoked : mne.Evoked
        The evoked EEG data (MNE Evoked object).
    tmin, tmax : float
        The time window to visualize (seconds).
    ymin, ymax : float
        Limits for y-axis (amplitude in µV).
    ncols : int
        Number of columns in subplot grids.
    figsize_per_row : float
        Height of each subplot row (inches).
    window_highlights : list of tuples or None
        Regions to highlight (start, end, color, alpha).
        Default: [(0.010, 0.035, 'orange', 0.3), (0.090, 0.190, 'yellow', 0.3)]
    split_groups : int
        Number of channel groups/windows to split the EEG channels.
    suptitle_base : str
        Base text for the figure suptitle.

    Returns
    -------
    None
    """

    # Select EEG channels only
    eeg_channels = [
        ch for ch in evoked.ch_names
        if evoked.get_channel_types(picks=ch)[0] == 'eeg'
    ]
    n_channels = len(eeg_channels)

    # Time axis setup
    time_mask = (evoked.times >= tmin) & (evoked.times <= tmax)
    times = evoked.times[time_mask]

    # Default highlight windows if not provided
    if window_highlights is None:
        window_highlights = [
            (0.010, 0.035, 'orange', 0.3),
            (0.090, 0.190, 'yellow', 0.3)
        ]

    # Split EEG channels into groups/windows
    split_indices = np.array_split(np.arange(n_channels), split_groups)
    channel_groups = [ [eeg_channels[i] for i in idx] for idx in split_indices ]

    for win_idx, ch_group in enumerate(channel_groups, start=1):
        n_ch = len(ch_group)
        nrows = int(np.ceil(n_ch / ncols))
        fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, nrows * figsize_per_row))
        axes = axes.flatten()

        for i, ch in enumerate(ch_group):
            ch_idx = evoked.ch_names.index(ch)
            # Extract data for this channel and window, convert to µV
            data = evoked.data[ch_idx, time_mask] * 1e6
            axes[i].plot(times, data)
            axes[i].set_ylim(ymin, ymax)
            axes[i].set_title(ch)
            axes[i].axvline(0, color='r', linestyle='--', label='Stimulus')
            axes[i].set_xlabel('Time (s)')
            axes[i].set_ylabel('Amplitude (µV)')
            axes[i].grid(True)
            # Highlight specified regions
            for start, end, color, alpha in window_highlights:
                axes[i].axvspan(start, end, color=color, alpha=alpha)

        # Hide unused axes
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.suptitle(f"{suptitle_base} - Window {win_idx}")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
def plot_evoked_with_std(epochs, std_data, channel_name, tmin=None, tmax=None,
                         highlight_window=None):
    """
    Plot the evoked response with standard deviation shading for a given channel.

    Parameters:
    -----------
    epochs : mne.Epochs
        The cleaned epochs object containing the EEG data.
    std_data : np.ndarray
        Standard deviation array with shape (n_channels, n_times).
    channel_name : str
        The channel name to plot (e.g., 'C3').
    tmin : float or None
        Start time (in seconds) of the time window to plot. If None, plot from start of epochs.
    tmax : float or None
        End time (in seconds) of the time window to plot. If None, plot until end of epochs.
    highlight_window : tuple or None
        Tuple of (start_time, end_time) in seconds for the highlighted region.
    """
    times = epochs.times

    if tmin is not None and tmax is not None:
        time_mask = (times >= tmin) & (times <= tmax)
    else:
        time_mask = slice(None)

    idx = epochs.ch_names.index(channel_name)
    mean_data = epochs.average().data[idx]
    std_dev = std_data[idx]

    plt.figure(figsize=(10, 6), dpi=300)
    plt.plot(times[time_mask], mean_data[time_mask] * 1e6, label='Mean (µV)')
    plt.fill_between(times[time_mask],
                     (mean_data[time_mask] - std_dev[time_mask]) * 1e6,
                     (mean_data[time_mask] + std_dev[time_mask]) * 1e6,
                     alpha=0.3, label='±1 Std Dev')

    # Highlight the specified time window if provided
    if highlight_window is not None:
        plt.axvspan(highlight_window[0], highlight_window[1], color='orange', alpha=0.3)

    plt.axvline(x=0, color='k', linestyle='--', linewidth=1)

    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude (µV)')
    plt.title(f'Evoked response at {channel_name} with variability')
    plt.legend()
    plt.show()
