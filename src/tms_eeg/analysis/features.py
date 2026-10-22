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
    ):
        self.epochs = epochs
        self.channels = channels
        self.time_windows = time_windows

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
