from typing import Union

import mne
import numpy as np
from mne._fiff.pick import _picks_to_idx
from scipy.interpolate import CubicSpline

from tms_eeg.config.settings import ProjectConfig

Picks = Union[str, list, None]


class ArtifactRemover:
    MNE_MODES = {"linear", "window", "constant"}
    CUSTOM_MODES = {"cubic", "tesa"}
    _EXCLUDED_TYPES = {"stim"}

    def __init__(self, config: ProjectConfig):
        self.config = config

    # ------------------------------------------------------------------ #
    # Picks
    # ------------------------------------------------------------------ #
    def _resolve_picks(self, info: mne.Info, picks: Picks = None) -> np.ndarray:
        """None -> todos os canais exceto stim; senão, usa o pick especificado."""
        if picks is None:
            picks = getattr(self.config, "artifact_picks", None)

        if picks is None:
            idx = _picks_to_idx(info, "all", exclude=())
            types = np.array(info.get_channel_types())
            idx = idx[~np.isin(types[idx], list(self._EXCLUDED_TYPES))]
        else:
            idx = _picks_to_idx(info, picks, exclude=())

        if idx.size == 0:
            raise ValueError(f"No channels selected with picks={picks!r}.")
        return idx

    def _get_tms_events(self, raw: mne.io.BaseRaw) -> np.ndarray:
        events, event_id = mne.events_from_annotations(raw)
        tms_annotation = list(self.config.event_trigger_id.keys())[0]
        if tms_annotation not in event_id:
            raise KeyError(
                f"Annotation '{tms_annotation}' not found in raw. "
                f"Available: {list(event_id)}"
            )
        return events[events[:, 2] == event_id[tms_annotation]]

    # ------------------------------------------------------------------ #
    # Dispatcher
    # ------------------------------------------------------------------ #
    def remove_tms_artifact(
        self,
        inst: Union[mne.io.BaseRaw, mne.BaseEpochs],
        mode: str | None = None,
        picks: Picks = None,
    ) -> Union[mne.io.BaseRaw, mne.BaseEpochs]:

        window = self.config.artifact_window
        mode = mode or self.config.artifact_mode
        picks_idx = self._resolve_picks(inst.info, picks)

        if mode in self.MNE_MODES:
            return self._remove_with_mne(inst, window, mode, picks_idx)

        if mode == "tesa":
            if isinstance(inst, mne.io.BaseRaw):
                return self.interp_tms_tesa_like(inst, picks_idx)
            raise TypeError(f"Mode 'tesa' supports only Raw, got {type(inst).__name__}")

        if mode == "cubic":
            if isinstance(inst, mne.io.BaseRaw):
                return self._interpolate_cubic_raw(inst, window, picks_idx)
            if isinstance(inst, mne.BaseEpochs):
                return self._interpolate_cubic(inst, window, picks_idx)
            raise TypeError(f"Unsupported type: {type(inst).__name__}")

        allowed = self.MNE_MODES | self.CUSTOM_MODES
        raise ValueError(f"Unsupported mode '{mode}'. Choose from {sorted(allowed)}.")

    # ------------------------------------------------------------------ #
    # MNE-native modes (linear / window / constant)
    # ------------------------------------------------------------------ #
    def _remove_with_mne(self, inst, window: tuple, mode: str, picks_idx: np.ndarray):
        inst_clean = inst.copy().load_data()
        kwargs = dict(tmin=window[0], tmax=window[1], mode=mode, picks=picks_idx)

        if isinstance(inst, mne.io.BaseRaw):
            events, event_id = mne.events_from_annotations(inst)
            tms_annotation = list(self.config.event_trigger_id.keys())[0]
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
        anchor_ms = self.config.artifact_anchor_window_ms
        if not np.isscalar(anchor_ms):
            anchor_ms = min(anchor_ms)
        n_anchor = int(round(anchor_ms / 1000.0 * sfreq))
        if n_anchor < 2:
            raise ValueError(
                f"anchor_window_ms={anchor_ms} gives too few samples ({n_anchor})."
            )
        return n_anchor

    def _interpolate_cubic(self, epochs: mne.BaseEpochs, window: tuple,
                           picks_idx: np.ndarray) -> mne.BaseEpochs:
        epochs_clean = epochs.copy().load_data()
        times = epochs_clean.times
        sfreq = epochs_clean.info["sfreq"]
        tmin, tmax = window
        n_anchor = self._get_n_anchor(sfreq)

        zero_idx = int(round(-times[0] * sfreq))
        idx_start = zero_idx + int(round(tmin * sfreq))
        idx_end = zero_idx + int(round(tmax * sfreq))

        pre_idx = np.arange(idx_start - n_anchor, idx_start)
        post_idx = np.arange(idx_end, idx_end + n_anchor)
        if pre_idx[0] < 0 or post_idx[-1] >= len(times):
            raise ValueError(
                f"Not enough samples around the artifact window for "
                f"{n_anchor} anchor samples. Reduce the anchor window or widen the epoch."
            )

        anchor_idx = np.concatenate([pre_idx, post_idx])
        target_idx = np.arange(idx_start, idx_end)

        data = epochs_clean._data  # (n_epochs, n_ch, n_times), in-place
        sub = data[:, picks_idx][:, :, anchor_idx]
        spline = CubicSpline(anchor_idx, sub, axis=-1)
        data[:, picks_idx, idx_start:idx_end] = spline(target_idx)

        return epochs_clean

    def _interpolate_cubic_raw(self, raw: mne.io.BaseRaw, window: tuple,
                               picks_idx: np.ndarray) -> mne.io.BaseRaw:
        raw_clean = raw.copy().load_data()
        sfreq = raw_clean.info["sfreq"]
        tmin, tmax = window
        n_times = raw_clean.n_times
        n_anchor = self._get_n_anchor(sfreq)
        tms_events = self._get_tms_events(raw_clean)

        off_start = int(round(tmin * sfreq))
        off_end = int(round(tmax * sfreq))
        data = raw_clean._data

        for ev_sample in tms_events[:, 0]:
            center = ev_sample - raw_clean.first_samp
            idx_start, idx_end = center + off_start, center + off_end

            pre_idx = np.arange(idx_start - n_anchor, idx_start)
            post_idx = np.arange(idx_end, idx_end + n_anchor)
            if pre_idx[0] < 0 or post_idx[-1] >= n_times:
                print(f"Skipping event at sample {ev_sample}: not enough anchor samples.")
                continue

            anchor_idx = np.concatenate([pre_idx, post_idx])
            target_idx = np.arange(idx_start, idx_end)

            spline = CubicSpline(anchor_idx, data[np.ix_(picks_idx, anchor_idx)], axis=-1)
            data[np.ix_(picks_idx, target_idx)] = spline(target_idx)

        return raw_clean

    # ------------------------------------------------------------------ #
    # TESA-like polynomial interpolation (Raw)
    # ------------------------------------------------------------------ #
    def _get_tesa_anchor_ms(self) -> tuple[float, float]:
        anchor = getattr(self.config, "artifact_anchor_window_ms", 2.0)
        if np.isscalar(anchor):
            return float(anchor), float(anchor)
        pre, post = anchor
        return float(pre), float(post)

    def interp_tms_tesa_like(self, raw: mne.io.BaseRaw,
                             picks_idx: np.ndarray | None = None) -> mne.io.BaseRaw:
        """Interpolação polinomial ao estilo TESA, ajustada só nas âncoras."""
        tmin, tmax = self.config.artifact_window
        anchor_ms = self._get_tesa_anchor_ms()
        order = getattr(self.config, "artifact_poly_order", 3)

        raw_out = raw.copy().load_data()
        sfreq = raw_out.info["sfreq"]
        if picks_idx is None:
            picks_idx = self._resolve_picks(raw_out.info)
        tms_events = self._get_tms_events(raw_out)

        s_start = int(np.ceil(tmin * sfreq))
        s_end = int(np.ceil(tmax * sfreq))
        n_pre = max(1, int(round(anchor_ms[0] * 1e-3 * sfreq)))
        n_post = max(1, int(round(anchor_ms[1] * 1e-3 * sfreq)))
        if n_pre + n_post < order + 1:
            raise ValueError(
                f"Insufficient anchors ({n_pre + n_post}) for polynomial order {order}."
            )

        data = raw_out._data
        n_times = raw_out.n_times

        for ev in tms_events[:, 0]:
            c = ev - raw_out.first_samp
            a, b = c + s_start, c + s_end            # região interpolada: [a, b)
            pre = np.arange(a - n_pre, a)
            post = np.arange(b, b + n_post)
            if pre[0] < 0 or post[-1] >= n_times:
                print(f"Skipping event at sample {ev}: anchors out of bounds.")
                continue

            anc = np.concatenate([pre, post])
            tgt = np.arange(a, b)
            t_anc = (anc - c) * 1e3 / sfreq
            t_tgt = (tgt - c) * 1e3 / sfreq

            y = data[np.ix_(picks_idx, anc)].T       # (n_anc, n_ch)
            p = np.polyfit(t_anc, y, order)
            data[np.ix_(picks_idx, tgt)] = (np.vander(t_tgt, order + 1) @ p).T

        return raw_out