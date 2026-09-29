import mne
import numpy as np


def bandpass(inst, band, ch_type: str):
    """FIR bandpass (zero-phase) em EEG ou EMG."""
    l_freq, h_freq = band

    return inst.copy().load_data().filter(
        l_freq=l_freq,
        h_freq=h_freq,
        picks=ch_type,
        method="fir",
        phase="zero",
        fir_design="firwin",
        verbose=True,
    )


def _notch_params(sfreq, notch_freqs, band=None, harmonics=1):
    """Calcula as frequências (com harmônicos) e as larguras do notch."""
    nyq = sfreq / 2

    if band is not None:
        low, high = band
        bases, width = [(low + high) / 2], high - low
    else:
        bases = [notch_freqs] if isinstance(notch_freqs, (int, float)) else list(notch_freqs)
        width = None  # padrão do MNE: freq / 200

    freqs = np.array([f * h for f in bases
                      for h in range(1, harmonics + 1) if f * h < nyq])
    return freqs, width


def notch_filter(inst, notch_freqs, ch_type: str = "eeg", band=None, harmonics=1):
    """Notch FIR (zero-phase) via MNE em Raw, Epochs ou Evoked."""
    inst = inst.copy().load_data() if hasattr(inst, "load_data") else inst.copy()
    freqs, width = _notch_params(inst.info["sfreq"], notch_freqs, band, harmonics)

    if freqs.size == 0:
        return inst

    kwargs = {
        "freqs": freqs,
        "notch_widths": width,
        "method": "fir",
        "phase": "zero",
        "fir_design": "firwin",
        "verbose": True,
    }

    if isinstance(inst, mne.io.BaseRaw):
        return inst.notch_filter(picks=ch_type, **kwargs)

    # Epochs / Evoked: não têm .notch_filter()
    picks = mne.pick_types(inst.info, meg=False, ref_meg=False, **{ch_type: True})
    if len(picks) == 0:
        return inst

    if isinstance(inst, mne.BaseEpochs):
        data: np.ndarray = inst.get_data(copy=True)
    else:
        data = np.asarray(inst.data)

    data[..., picks, :] = mne.filter.notch_filter(
        data[..., picks, :], Fs=inst.info["sfreq"], **kwargs
    )

    if isinstance(inst, mne.BaseEpochs):
        inst._data = data
    else:
        inst.data = data
    return inst
