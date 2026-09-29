def downsample(epochs, freq: float):
    """Downsample epoched EEG data to `freq` Hz."""
    return epochs.copy().resample(freq)


def downsample_emg_channels(epochs, freq: float):
    """Extract EMG channels and downsample them to `freq` Hz."""
    emg_epochs = epochs.copy().pick('emg')
    return emg_epochs.resample(freq)
