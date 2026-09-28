from scipy.signal import filtfilt, iirnotch

from tms_eeg.config.settings import ProjectConfig


class Filter:
    def __init__(self, config: ProjectConfig):
        self.config = config

    def bp_filter(self, inst, ch_type: str):
        """Apply FIR bandpass filter to EEG or EMG data (config.filters.<ch_type>_bandpass)."""
        bp = getattr(self.config.filters, f"{ch_type}_bandpass")
        return inst.copy().filter(
            l_freq=bp[0],
            h_freq=bp[1],
            picks=ch_type,
            method="fir",
            phase="zero",
            fir_design="firwin",
            verbose=True,
        )

    def notch_filter(self, data):
        """Apply a notch filter around ``config.filters.notch_band`` (IIR, zero-phase)."""
        cfg = self.config.filters
        data = data.copy().load_data()
        sfreq = data.info["sfreq"]
        arr = data.get_data()

        low, high = cfg.notch_band
        f0 = (low + high) / 2
        q = f0 / (high - low)

        for h in range(1, cfg.notch_harmonics + 1):
            f = f0 * h
            if f >= sfreq / 2:
                break
            b, a = iirnotch(w0=f, Q=q, fs=sfreq)
            arr = filtfilt(b, a, arr, axis=-1)

        data._data[:] = arr
        return data