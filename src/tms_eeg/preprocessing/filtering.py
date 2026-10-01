import mne
import numpy as np
from mne._fiff.pick import _picks_to_idx


def _filter_kwargs(method: str = "fir", iir_order: int = 4) -> dict:
    """Parâmetros comuns FIR/IIR (sempre zero-phase)."""
    if method == "fir":
        return {"method": "fir", "phase": "zero", "fir_design": "firwin"}
    if method == "iir":
        # Butterworth + filtfilt (zero-phase), igual ao pop_tesa_filtbutter
        return {
            "method": "iir",
            "phase": "zero",
            "iir_params": dict(order=iir_order, ftype="butter", output="sos"),
        }
    raise ValueError(f"method deve ser 'fir' ou 'iir', recebido '{method}'.")


def _load(inst):
    return inst.copy().load_data() if hasattr(inst, "load_data") else inst.copy()


def bandpass(inst, band, ch_type: str, method: str = "fir", iir_order: int = 4):
    """Bandpass/highpass/lowpass zero-phase (FIR ou IIR Butterworth).

    band=(1, None) -> highpass | (None, 80) -> lowpass | (1, 80) -> bandpass
    """
    l_freq, h_freq = band
    return _load(inst).filter(
        l_freq=l_freq,
        h_freq=h_freq,
        picks=ch_type,
        verbose=True,
        **_filter_kwargs(method, iir_order),
    )


def iir_bandstop(inst, band, ch_type: str | list[str] | None = None,
                 iir_order: int = 4):
    """Bandstop Butterworth zero-phase (equivalente ao 'bandstop' do TESA).

    Ex.: band=(58, 62). No MNE, l_freq > h_freq gera bandstop.
    Funciona em Raw, Epochs e Evoked.
    """
    low, high = band
    return _load(inst).filter(
        l_freq=high,
        h_freq=low,
        picks=ch_type if ch_type is not None else "data",
        verbose=True,
        **_filter_kwargs("iir", iir_order),
    )


def _notch_params(sfreq, notch_freqs, band=None, harmonics=1, width=None):
    """Calcula as frequências (com harmônicos) e as larguras do notch."""
    nyq = sfreq / 2

    if band is not None:
        low, high = band
        bases, width = [(low + high) / 2], high - low
    else:
        bases = [notch_freqs] if isinstance(notch_freqs, (int, float)) else list(notch_freqs)
        # width=None -> padrão do MNE (freq / 200)

    freqs = np.array([f * h for f in bases
                      for h in range(1, harmonics + 1) if f * h < nyq])
    return freqs, width


def notch_filter(inst, notch_freqs, ch_type: str | list[str] | None = None,
                 band=None, harmonics=1, method: str = "fir",
                 width: float | None = None, iir_order: int = 4):
    """Notch zero-phase (FIR ou IIR Butterworth) em Raw, Epochs ou Evoked.

    Para IIR estilo TESA use method='iir', width=4 (ex.: 58-62 Hz).
    ch_type=None filtra todos os canais de dados.
    """
    inst = _load(inst)
    freqs, width = _notch_params(inst.info["sfreq"], notch_freqs, band,
                                 harmonics, width)
    if freqs.size == 0:
        return inst

    if method == "iir" and width is None:
        width = 4.0  # freq/200 é estreito demais para Butterworth

    picks = _picks_to_idx(
        inst.info, ch_type if ch_type is not None else "data",
        exclude=(), allow_empty=True,
    )
    if len(picks) == 0:
        return inst

    kwargs = {"freqs": freqs, "notch_widths": width, "verbose": True,
              **_filter_kwargs(method, iir_order)}

    if isinstance(inst, mne.io.BaseRaw):
        return inst.notch_filter(picks=picks, **kwargs)

    # Epochs / Evoked: não têm .notch_filter()
    is_epochs = isinstance(inst, mne.BaseEpochs)
    data = inst.get_data(copy=True) if is_epochs else np.asarray(inst.data)

    data[..., picks, :] = mne.filter.notch_filter(
        data[..., picks, :], Fs=inst.info["sfreq"], **kwargs
    )

    if is_epochs:
        inst._data = data
    else:
        inst.data = data
    return inst