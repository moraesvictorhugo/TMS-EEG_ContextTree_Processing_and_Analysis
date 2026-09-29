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
            if isinstance(inst, mne.io.BaseRaw):
                return self._interpolate_cubic_raw(inst, window)
            if isinstance(inst, mne.BaseEpochs):
                return self._interpolate_cubic(inst, window)
            raise TypeError(f"Unsupported type: {type(inst).__name__}")

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
    # Custom cubic spline interpolation (Epochs and Raw)
    # ------------------------------------------------------------------ #
    def _get_n_anchor(self, sfreq: float) -> int:
        anchor_ms = self.config.artifact.anchor_window_ms
        n_anchor = int(round(anchor_ms / 1000.0 * sfreq))
        if n_anchor < 2:
            raise ValueError(
                f"anchor_window_ms={anchor_ms} gives too few samples ({n_anchor})."
            )
        return n_anchor

    def _interpolate_cubic(
        self,
        epochs: mne.BaseEpochs,
        window: tuple,
    ) -> mne.BaseEpochs:
        epochs_clean = epochs.copy().load_data()
        times = epochs_clean.times
        sfreq = epochs_clean.info["sfreq"]
        tmin, tmax = window
        n_anchor = self._get_n_anchor(sfreq)

        # Mesmo cálculo com round usado no Raw (índice de t=0 + offsets)
        zero_idx = int(round(-times[0] * sfreq))
        idx_start = zero_idx + int(round(tmin * sfreq))
        idx_end = zero_idx + int(round(tmax * sfreq))

        pre_idx = np.arange(idx_start - n_anchor, idx_start)
        post_idx = np.arange(idx_end, idx_end + n_anchor)

        if pre_idx[0] < 0 or post_idx[-1] >= len(times):
            raise ValueError(
                f"Not enough samples around the artifact window for "
                f"{n_anchor} anchor samples. "
                f"Reduce the anchor window or widen the epoch."
            )

        anchor_idx = np.concatenate([pre_idx, post_idx])
        target_idx = np.arange(idx_start, idx_end)

        data = epochs_clean.get_data(copy=False)  # (n_epochs, n_ch, n_times)
        spline = CubicSpline(anchor_idx, data[:, :, anchor_idx], axis=-1)
        data[:, :, idx_start:idx_end] = spline(target_idx)

        return epochs_clean

    def _interpolate_cubic_raw(
        self,
        raw: mne.io.BaseRaw,
        window: tuple,
    ) -> mne.io.BaseRaw:
        raw_clean = raw.copy().load_data()
        sfreq = raw_clean.info["sfreq"]
        tmin, tmax = window
        n_times = raw_clean.n_times
        n_anchor = self._get_n_anchor(sfreq)

        events, event_id = mne.events_from_annotations(raw_clean)
        tms_annotation = list(self.config.events.trigger_id.keys())[0]
        if tms_annotation not in event_id:
            raise KeyError(
                f"Annotation '{tms_annotation}' not found in raw. "
                f"Available: {list(event_id)}"
            )
        tms_events = events[events[:, 2] == event_id[tms_annotation]]

        off_start = int(round(tmin * sfreq))
        off_end = int(round(tmax * sfreq))

        data = raw_clean._data  # (n_ch, n_times), editado in-place

        for ev_sample in tms_events[:, 0]:
            center = ev_sample - raw_clean.first_samp
            idx_start = center + off_start
            idx_end = center + off_end

            pre_idx = np.arange(idx_start - n_anchor, idx_start)
            post_idx = np.arange(idx_end, idx_end + n_anchor)

            if pre_idx[0] < 0 or post_idx[-1] >= n_times:
                print(f"Skipping event at sample {ev_sample}: not enough anchor samples.")
                continue

            anchor_idx = np.concatenate([pre_idx, post_idx])
            target_idx = np.arange(idx_start, idx_end)

            spline = CubicSpline(anchor_idx, data[:, anchor_idx], axis=-1)
            data[:, idx_start:idx_end] = spline(target_idx)

        return raw_clean