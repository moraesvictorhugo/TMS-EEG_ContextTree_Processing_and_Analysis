# TMS-EEG Preprocessing Pipeline

This approach was adapted from recommendations of:
"Ziemann, Ulf, et al. "Clinical utility and prospective of TMS–EEG: Updated review from an international expert group." Clinical Neurophysiology (2026): 2111487."

## Overview

| # | Step | Data |
|---|---|---|
| 1 | Cubic interpolation of the TMS artifact (−5 to +10 ms, 10 ms anchor) | Continuous |
| 2 | Downsampling to 5000 Hz | Continuous |
| 3 | 60 Hz notch (MNE default) | Continuous |
| 4 | Split into two copies: **A** with 0.1 Hz high-pass and **B** with 1 Hz high-pass | Continuous |
| 5 | Epochs from −1000 to +1000 ms (`detrend=1`) | A and B |
| 6 | Removal of bad channels and bad epochs (identified in A, applied to B) | A and B |
| 7 | Average reference | A and B |
| 8 | Replacement of the artifact with a constant (−5 to +10 ms) | A and B |
| 9 | Rank calculation: $$32 - n_{bads} - 1$$ | — |
| 10 | ICA fitting (`n_components = rank`) | B |
| 11 | Application of the ICA solution and component removal | A |
| 12 | SOUND | A |
| 13 | SSP-SIR | A |
| 14 | Interpolation of removed channels | A |
| 15 | Average reference | A |
| 16 | Cubic interpolation of the TMS artifact (−5 to +10 ms) | A |
| 17 | 80 Hz FIR low-pass | A |
| 18 | Baseline from −300 to −20 ms | A |
| 19 | Resampling to 500 Hz | A |
| 20 | Cropping to −800 to +800 ms | A |
| 21 | Export | A |

---

## Rationale

### 1. Cubic interpolation of the pulse first
The TMS pulse is far larger than the EEG. If it stays in the signal, any later filter or resampling spreads it in time and causes *ringing* and *aliasing*. Removing it first protects every step that follows.

### 2. Downsampling to 5000 Hz
This cuts computational cost while keeping enough resolution for the fast early-response artifacts. MNE's anti-aliasing filter is safe to use because the pulse has already been removed.

### 3. Notch on continuous data
With the default 1 Hz transition, the notch filter is about 3.3 s long, which is longer than the 2 s epoch. Applying it to continuous data:
- keeps it from distorting epoch edges;
- keeps it narrow, so frequencies near 60 Hz are preserved;
- gives ICA, SOUND and SSP-SIR data that is already free of line noise.

It comes before the A/B split so both copies get identical treatment.

### 4. Two copies with different high-pass filters
- **A (0.1 Hz):** keeps the slow TEP components, which are the signal of interest.
- **B (1 Hz):** slow drifts degrade ICA decomposition, and a 1 Hz high-pass gives more stable components.

Both are filtered on continuous data because, with a 1 Hz transition, the 1 Hz filter wouldn't fit within the epoch either.

### 5. Long epochs (±1000 ms) with `detrend=1`
The extra margin absorbs edge effects from later steps and is thrown away in the final crop (step 20). Linear detrending removes any leftover trend in each epoch.

### 6. Removing bads without interpolating
- Bad channels and epochs would contaminate ICA, SOUND and SSP-SIR.
- Interpolating at this point would create channels that are linear combinations of other channels. That lowers the rank and destabilizes ICA.
- A and B end up with exactly the same channels and epochs, so the solution fitted on B can be applied to A.

### 7. Average reference before spatial cleaning
SOUND and SSP-SIR assume the data are average-referenced. This reference also lowers the rank by 1, which step 9 accounts for.

### 8. Replacing the artifact with a constant
The segment interpolated in step 1 is artificial. Replacing it with a constant keeps ICA and SOUND from trying to model a signal that isn't physiological.

### 9. Rank calculation
Fitting ICA with more components than the true rank produces unstable or duplicated components. The $$-n_{bads}$$ term subtracts the removed channels, and the $$-1$$ subtracts the average reference.

### 10–11. ICA fitted on B, applied to A
This gets the best of both copies: the decomposition is stable because it was fitted at 1 Hz, and the cleaning is applied to the 0.1 Hz data, which keeps the TEP. ICA comes first in the cleaning stage because it removes large, stereotyped artifacts such as blinks, eye movements and pulse decay.

### 12. SOUND after ICA
SOUND estimates and suppresses noise that is specific to each channel. That estimate is better once the large, spatially correlated artifacts have been removed.

### 13. SSP-SIR last in cleaning
It removes the TMS-evoked muscle artifact that is still left. Because ICA and SOUND have already cleaned the data, the artifact subspace can be estimated more precisely.

### 14–15. Channel interpolation and re-referencing
Interpolation happens only after cleaning, so bad channels don't affect any spatial estimate. The average reference is then recomputed to include all 32 channels.

### 16. Second artifact interpolation
The constant from step 8 creates steps in the signal. Filtering over those steps would cause *ringing*, so the segment is smoothed before the low-pass.

### 17. 80 Hz low-pass on epochs
With a transition of about 20 Hz, the filter is short enough to fit in the epoch. It removes high-frequency noise and acts as anti-aliasing for resampling, since 80 Hz is well below the 250 Hz Nyquist frequency.

### 18. Baseline from −300 to −20 ms
This uses a stable pre-stimulus window. Stopping at −20 ms leaves a margin before the pulse, where interpolation and filtering may have left some residual effect.

### 19. Resampling to 500 Hz
500 Hz is enough for data low-passed at 80 Hz, and it shrinks the data. It comes after the low-pass to avoid *aliasing*.

### 20. Cropping to ±800 ms
This throws away the edges where filter and resampling effects build up, keeping only the reliable segment.