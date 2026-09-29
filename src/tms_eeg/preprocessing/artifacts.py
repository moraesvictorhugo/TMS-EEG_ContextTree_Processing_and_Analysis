import mne
import numpy as np
from typing import Union
from scipy.interpolate import CubicSpline

from tms_eeg.config.settings import ProjectConfig


class ArtifactRemover:
    MNE_MODES = {"linear", "window", "constant"}
    CUSTOM_MODES = {"cubic"}

    def __init__(self, config: ProjectConfig):
        self.config = config

    def remove_tms_artifact(
        self,
        inst: Union[mne.io.BaseRaw, mne.BaseEpochs],
        mode: str | None = None,
    ) -> Union[mne.io.BaseRaw, mne.BaseEpochs]:

        artifact_cfg = self.config.artifact
        window = artifact_cfg.window_removal_artifact
        mode = mode or artifact_cfg.mode_removal_artifact

        if mode in self.MNE_MODES:
            return self._remove_with_mne(inst, window, mode)

        elif mode in self.CUSTOM_MODES:
            if not isinstance(inst, mne.BaseEpochs):
                raise TypeError(
                    f"Mode '{mode}' is only supported for Epochs, "
                    f"got {type(inst).__name__}."
                )
            return self._interpolate_cubic(inst, window)

        else:
            allowed = self.MNE_MODES | self.CUSTOM_MODES
            raise ValueError(
                f"Unsupported mode '{mode}'. Choose from {sorted(allowed)}."
            )

    # ------------------------------------------------------------------ #
    # MNE-native modes (linear / window / constant)
    # ------------------------------------------------------------------ #
    def _remove_with_mne(
        self,
        inst: Union[mne.io.BaseRaw, mne.BaseEpochs],
        window: tuple,
        mode: str,
    ) -> Union[mne.io.BaseRaw, mne.BaseEpochs]:
        inst_clean = inst.copy().load_data()
        kwargs = dict(tmin=window[0], tmax=window[1], mode=mode)

        if isinstance(inst, mne.io.BaseRaw):
            events, event_id = mne.events_from_annotations(inst)
            tms_annotation = list(self.config.events.trigger_id.keys())[0]
            kwargs["events"] = events
            kwargs["event_id"] = event_id[tms_annotation]
        elif not isinstance(inst, mne.BaseEpochs):
            raise TypeError(f"Unsupported type: {type(inst)}")

        mne.preprocessing.fix_stim_artifact(inst_clean, **kwargs)
        return inst_clean

    # ------------------------------------------------------------------ #
    # Custom cubic spline interpolation — only for Epochs
    # ------------------------------------------------------------------ #
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
        anchor_sec = anchor_ms / 1000.0
        n_anchor = int(round(anchor_sec * sfreq))

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