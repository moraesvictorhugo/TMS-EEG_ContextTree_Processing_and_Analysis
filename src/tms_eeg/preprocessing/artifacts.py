import mne
import numpy as np
from scipy.interpolate import CubicSpline

from tms_eeg.config.settings import ProjectConfig


class ArtifactRemover:
    """Remove the TMS pulse artifact from epoched data."""

    def __init__(self, config: ProjectConfig):
        self.config = config

    def remove_tms_artifact(
        self,
        epochs: mne.BaseEpochs,
        mode: str | None = None,
    ) -> mne.BaseEpochs:
        """Cubic-spline interpolation over the artifact window.

        The interpolation is anchored on the samples immediately before and
        after the artifact window (see ``ArtifactConfig.anchor_window_ms``).

        Parameters
        ----------
        epochs : mne.BaseEpochs
            Epoched EEG data.
        mode : str, optional
            Interpolation mode. Only ``"cubic"`` is supported.
        """
        mode = mode or self.config.artifact.mode_removal_artifact
        if mode != "cubic":
            raise ValueError(f"Only mode='cubic' is supported, got {mode!r}")
        return self._interpolate_cubic(epochs, self.config.artifact.window_removal_artifact)

    def fix_stim_artifact(self, inst: mne.BaseEpochs) -> mne.BaseEpochs:
        """Constant-interpolation pass run after SSP-SIR (mne-native).

        Uses the same artifact window as the main removal step, with the
        ``fix_stim_artifact_baseline`` period from the config.
        """
        cfg = self.config.artifact
        return mne.preprocessing.fix_stim_artifact(
            inst.copy(),
            mode="constant",
            tmin=cfg.window_removal_artifact[0],
            tmax=cfg.window_removal_artifact[1],
            baseline=cfg.fix_stim_artifact_baseline,
        )

    def _interpolate_cubic(
        self,
        epochs: mne.BaseEpochs,
        window: tuple,
    ) -> mne.BaseEpochs:
        epochs_clean = epochs.copy().load_data()
        times = epochs_clean.times
        sfreq = epochs_clean.info["sfreq"]
        tmin, tmax = window

        # anchor window in seconds (ms -> s)
        anchor_ms = self.config.artifact.anchor_window_ms
        n_anchor = int(round(anchor_ms / 1000.0 * sfreq))

        idx_start = int(np.searchsorted(times, tmin))
        idx_end = int(np.searchsorted(times, tmax))

        pre_idx = np.arange(idx_start - n_anchor, idx_start)
        post_idx = np.arange(idx_end, idx_end + n_anchor)

        if pre_idx[0] < 0 or post_idx[-1] >= len(times):
            raise ValueError(
                f"Not enough samples around the artifact window for "
                f"anchor_window_ms={anchor_ms} ms ({n_anchor} samples). "
                f"Reduce the anchor window or widen the epoch."
            )

        anchor_idx = np.concatenate([pre_idx, post_idx])
        anchor_times = times[anchor_idx]
        target_times = times[idx_start:idx_end]

        data = epochs_clean.get_data(copy=False)  # (n_epochs, n_ch, n_times)
        for ep in range(data.shape[0]):
            for ch in range(data.shape[1]):
                y = data[ep, ch, anchor_idx]
                data[ep, ch, idx_start:idx_end] = CubicSpline(anchor_times, y)(target_times)

        return epochs_clean