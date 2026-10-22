import mne
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def mean_field_power(evoked: mne.Evoked) -> np.ndarray:
    """Return the standard deviation across channels at each time point.

    Shared GMFP/LMFP calculation used by :class:`FeatureExtractor` and TEP plotters.
    """
    return evoked.data.std(axis=0)


class FeatureExtractor:
    """Extract amplitude features from EEG epochs."""

    def __init__(
        self,
        epochs: mne.Epochs,
        channels: List[str],
        time_windows: Dict[str, Tuple[float, float]],
        peak_selection: str = "absolute",
        auc_absolute: bool = False,
    ):
        self.epochs = epochs
        self.channels = channels
        self.time_windows = time_windows
        if peak_selection not in {"absolute", "positive", "negative"}:
            raise ValueError(
                "peak_selection must be 'absolute', 'positive', or 'negative'."
            )
        self.peak_selection = peak_selection
        self.auc_absolute = auc_absolute

    def _window_mask(self, component: str) -> np.ndarray:
        """Return sample mask for a configured window, validating its range."""
        if component not in self.time_windows:
            raise KeyError(f"Time window {component!r} is not configured.")
        t_min, t_max = self.time_windows[component]
        if t_min >= t_max:
            raise ValueError(f"Invalid time window for {component}: {(t_min, t_max)}")
        # Half-open windows prevent a shared boundary sample from belonging
        # to both adjacent components (e.g. N15 and P30 at 20 ms).
        mask = (self.epochs.times >= t_min) & (self.epochs.times < t_max)
        if not mask.any():
            raise ValueError(
                f"Configured time window {component}={(t_min, t_max)} does not "
                "overlap the epochs time range."
            )
        return mask

    def _select_peak(self, signal: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
        """Return selected signed peak amplitude (V) and latency (s)."""
        segment = signal[mask]
        times = self.epochs.times[mask]
        if not segment.size:
            raise ValueError("Cannot select a peak from an empty time window.")
        if self.peak_selection == "positive":
            index = int(np.argmax(segment))
        elif self.peak_selection == "negative":
            index = int(np.argmin(segment))
        else:
            index = int(np.argmax(np.abs(segment)))
        return float(segment[index]), float(times[index])

    def extract_condition_features(
        self,
        component: str = "P30",
        component1: str = "N15",
        component2: str = "P30",
        conditions: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Extract mean, signed peak, signed AUC, and peak-to-peak by channel.

        Amplitudes are returned in microvolts and AUC in microvolt-seconds.
        Features are computed from condition evokeds, not individual epochs.
        """
        if conditions is None:
            conditions = list(self.epochs.event_id)
        p30_mask = self._window_mask(component)
        first_mask = self._window_mask(component1)
        second_mask = self._window_mask(component2)
        rows = []

        for condition, evoked in self.get_evokeds(conditions=conditions).items():
            for channel in self.channels:
                signal = evoked.copy().pick([channel]).data.squeeze()
                p30_segment = signal[p30_mask]
                p30_peak, p30_latency = self._select_peak(signal, p30_mask)
                first_peak, _ = self._select_peak(signal, first_mask)
                second_peak, _ = self._select_peak(signal, second_mask)
                auc_signal = np.abs(p30_segment) if self.auc_absolute else p30_segment
                auc = float(np.trapezoid(auc_signal, self.epochs.times[p30_mask]))
                rows.extend((
                    {"condition": condition, "channel": channel, "component": component,
                     "metric": "mean_amplitude", "value": float(p30_segment.mean() * 1e6),
                     "unit": "uV"},
                    {"condition": condition, "channel": channel, "component": component,
                     "metric": "peak_amplitude", "value": p30_peak * 1e6,
                     "unit": "uV"},
                    {"condition": condition, "channel": channel, "component": component,
                     "metric": "peak_latency", "value": p30_latency * 1e3,
                     "unit": "ms"},
                    {"condition": condition, "channel": channel, "component": component,
                     "metric": "auc", "value": auc * 1e6,
                     "unit": "uV*s"},
                    {"condition": condition, "channel": channel,
                     "component": f"{component1}-{component2}",
                     "metric": "peak_to_peak_amplitude",
                     "value": abs(second_peak - first_peak) * 1e6, "unit": "uV"},
                ))
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    #  Evokeds
    # ------------------------------------------------------------------ #

    def get_evokeds(
        self,
        conditions: Optional[List[str]] = None,
        picks: Optional[List[str]] = None,
    ) -> Dict[str, mne.Evoked]:
        """
        Return the average TEP for each condition.

        Parameters
        ----------
        conditions : list of str, optional
            Conditions to extract. If None, use all keys in ``epochs.event_id``.
        picks : list of str, optional
            Channels to select. If None, use ``self.channels`` (ROI).
        """
        if conditions is None:
            conditions = list(self.epochs.event_id.keys())
        if picks is None:
            picks = self.channels

        return {
            cond: self.epochs[cond].average().pick(picks)
            for cond in conditions
        }

    # ------------------------------------------------------------------ #
    #  Peak-to-Peak
    # ------------------------------------------------------------------ #

    def peak_to_peak(
        self,
        component1: str = "N15",
        component2: str = "P30",
        evokeds: Optional[Dict[str, mne.Evoked]] = None,
    ) -> pd.DataFrame:
        """
        Compute peak-to-peak amplitude between two components for each
        condition and channel.

        Select the largest absolute deflection in each window, regardless
        of polarity.
        """
        if evokeds is None:
            evokeds = self.get_evokeds()

        t1_min, t1_max = self.time_windows[component1]
        t2_min, t2_max = self.time_windows[component2]
        label = f"{component1}-{component2}"
        times = self.epochs.times
        mask1 = (times >= t1_min) & (times <= t1_max)
        mask2 = (times >= t2_min) & (times <= t2_max)

        rows = []
        for cond, evoked in evokeds.items():
            for ch in self.channels:
                signal = evoked.copy().pick([ch]).data.squeeze()

                seg1 = signal[mask1]
                seg2 = signal[mask2]
                val1 = seg1[np.argmax(np.abs(seg1))]
                val2 = seg2[np.argmax(np.abs(seg2))]

                rows.append({
                    "condition": cond,
                    "channel": ch,
                    "component": label,
                    "peak_to_peak_uV": abs(val2 - val1) * 1e6,
                })

        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    #  GMFP — Global Mean Field Power (all EEG channels)
    # ------------------------------------------------------------------ #

    def compute_gmfp(
        self,
        conditions: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute global mean field power across all EEG channels per time point.

        Returns dict {condition: 1-D array of shape (n_times,)}.
        """
        evokeds_all = self._get_evokeds_all_eeg(conditions)
        return {
            cond: mean_field_power(evoked)
            for cond, evoked in evokeds_all.items()
        }

    # ------------------------------------------------------------------ #
    #  LMFP — Local Mean Field Power (ROI channels)
    # ------------------------------------------------------------------ #

    def compute_lmfp(
        self,
        conditions: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Compute local mean field power across ROI channels per time point.

        Returns dict {condition: 1-D array of shape (n_times,)}.
        """
        evokeds_roi = self.get_evokeds(conditions=conditions, picks=self.channels)
        return {
            cond: mean_field_power(evoked)
            for cond, evoked in evokeds_roi.items()
        }

    # ------------------------------------------------------------------ #
    #  Peak amplitude & latency from GMFP / LMFP curves
    # ------------------------------------------------------------------ #

    def extract_mfp_peaks(
        self,
        mfp_data: Dict[str, np.ndarray],
        label: str = "GMFP",
    ) -> pd.DataFrame:
        """
        Extract peak amplitude and latency from GMFP or LMFP curves.

        Parameters
        ----------
        mfp_data : dict
            Mapping of conditions to 1-D arrays from ``compute_gmfp`` or ``compute_lmfp``.
        label : str
            "GMFP" or "LMFP", used in the ``measure`` column.

        Returns
        -------
        pd.DataFrame
            Columns: condition, measure, component, peak_amplitude_uV, peak_latency_ms.
        """
        times = self.epochs.times
        rows = []

        for cond, curve in mfp_data.items():
            for comp, (t_min, t_max) in self.time_windows.items():
                mask = (times >= t_min) & (times <= t_max)
                if not mask.any():
                    continue

                segment = curve[mask]
                times_segment = times[mask]

                peak_idx = np.argmax(segment)
                peak_amp = segment[peak_idx]
                peak_lat = times_segment[peak_idx]

                rows.append({
                    "condition": cond,
                    "measure": label,
                    "component": comp,
                    "peak_amplitude_uV": peak_amp * 1e6,
                    "peak_latency_ms": peak_lat * 1e3,
                })

        return pd.DataFrame(rows)

    def extract_mfp_window_features(
        self,
        mfp_data: Dict[str, np.ndarray],
        component: str = "P30",
        label: str = "GMFP",
    ) -> pd.DataFrame:
        """Summarize an MFP curve within a configured component window."""
        mask = self._window_mask(component)
        times = self.epochs.times[mask]
        rows = []
        for condition, curve in mfp_data.items():
            segment = curve[mask]
            peak_index = int(np.argmax(segment))
            rows.extend((
                {"condition": condition, "measure": label, "component": component,
                 "metric": f"{label.lower()}_mean", "value": float(segment.mean() * 1e6),
                 "unit": "uV"},
                {"condition": condition, "measure": label, "component": component,
                 "metric": f"{label.lower()}_peak", "value": float(segment[peak_index] * 1e6),
                 "unit": "uV"},
                {"condition": condition, "measure": label, "component": component,
                 "metric": f"{label.lower()}_auc",
                 "value": float(np.trapezoid(segment, times) * 1e6),
                 "unit": "uV*s"},
            ))
        return pd.DataFrame(rows)

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    def _get_evokeds_all_eeg(
        self,
        conditions: Optional[List[str]] = None,
    ) -> Dict[str, mne.Evoked]:
        """Return evokeds containing all EEG channels for GMFP."""
        if conditions is None:
            conditions = list(self.epochs.event_id.keys())

        return {
            cond: self.epochs[cond].average().pick("eeg")
            for cond in conditions
        }
