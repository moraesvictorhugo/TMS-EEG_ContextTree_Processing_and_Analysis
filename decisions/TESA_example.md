## Pipeline TESA (Rogasch et al., 2017)

This is the sequence recommended in the TESA manual and paper, with ICA performed in two rounds:

| # | Step | TESA function |
|---|---|---|
| 1 | Find TMS pulses | `tesa_findpulse` |
| 2 | Remove the pulse window (e.g., −2 to 10 ms) | `tesa_removedata` |
| 3 | Interpolate the removed window (cubic), before downsampling | `tesa_interpdata` |
| 4 | Downsample (e.g., 5 kHz → 1 kHz) | `pop_resample` |
| 5 | Epoch (−1 to 1 s) and apply baseline correction | `pop_epoch`, `pop_rmbase` |
| 6 | Remove bad channels and trials | Manual or `pop_rejkurt` |
| 7 | **Zero the interpolated window** before ICA | `tesa_removedata` |
| 8 | **ICA 1**: remove the large TMS-evoked muscle artifact | `tesa_fastica`, `tesa_compselect` |
| 9 | Interpolate the removed window again (cubic) | `tesa_interpdata` |
| 10 | **Filter**: 1–100 Hz bandpass and 48–52 Hz bandstop | `tesa_filtbutter` |
| 11 | Zero the interpolated window again | `tesa_removedata` |
| 12 | **ICA 2**: remove blinks, eye movements, residual muscle, and decay | `tesa_fastica`, `tesa_compselect` |
| 13 | Interpolate the window and removed channels | `tesa_interpdata`, `pop_interp` |
| 14 | Re-reference to the average and apply baseline correction | `pop_reref` |

## Notes for the refactor

1. **TESA filtering is not FIR.** `tesa_filtbutter` uses a **fourth-order, zero-phase Butterworth IIR** (`filtfilt`) with a 4 Hz bandstop. Thus, `band=(58, 62)` matches the TESA approach for 60 Hz mains power.
2. **Filtering occurs after ICA 1.** Removing the large muscle artifact before filtering reduces ringing. Cubic interpolation alone is insufficient; TESA follows the same principle.
3. **Harmonics are not notched.** The 100 Hz low-pass removes frequencies at and above 120 Hz; only 60 Hz is covered by the notch.
4. **SOUND and SSP-SIR (`tesa_sound`, `tesa_sspsir`) are alternatives to ICA 1 and are also applied before filtering.**

## MNE equivalent

```python
raw.notch_filter(60, method="iir",
                 iir_params=dict(order=4, ftype="butter"),
                 notch_widths=4, phase="zero")
```

For TESA compatibility, support `method="iir"` and apply the notch **after** artifact removal with ICA, SOUND, or SSP-SIR.