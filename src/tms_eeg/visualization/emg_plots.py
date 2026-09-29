"""EMG visualization class."""

import mne
from typing import Optional
import matplotlib.pyplot as plt


class EMGPlotter:
    """Plots EMG-related visualizations from epochs."""

    def __init__(self, config=None, xlim: tuple = None, writer=None):
        self.config = config
        self.xlim = xlim or (config.plots.emg_xlim if config else (-0.01, 0.08))
        self.writer = writer

    def _save_figure(self, fig, condition: str):
        """Save figure if save_figs is enabled in config."""
        if self.config and self.config.io.save_figs and fig is not None:
            if self.writer:
                writer = self.writer
            else:
                from tms_eeg.io.writer import Writer
                writer = Writer(self.config)
            
            filename = f"emg_{condition}"
            writer.save_figure(fig, filename)

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
