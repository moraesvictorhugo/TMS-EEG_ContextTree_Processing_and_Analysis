"""EMG visualization class."""

import mne
from typing import Optional
import matplotlib.pyplot as plt
from tms_eeg.io.writer import save_figure


class EMGPlotter:
    """Plots EMG-related visualizations from epochs."""

    def __init__(self, config, xlim: tuple = None, writer=None):
        self.config = config
        self.xlim = xlim or config.plot_emg_xlim
        self.writer = writer

    def _save_figure(self, fig, condition: str):
        """Save figure if io_save_figs is enabled in config."""
        save_figure(fig, f"emg_{condition}", self.config, self.writer)

    def plot_all(self, epochs: mne.Epochs):
        """Plot EMG evoked for each condition."""
        for condition in epochs.event_id.keys():
            evoked_emg = epochs[condition].average(picks='emg')
            self.plot_emg(evoked_emg, condition)

    def plot_emg(self, evoked_emg: mne.Evoked, condition: str):
        """Plot single EMG evoked."""
        fig = evoked_emg.plot(
            xlim=self.xlim,
            titles=f'EMG - {condition}',
        )
        self._save_figure(fig, condition)
