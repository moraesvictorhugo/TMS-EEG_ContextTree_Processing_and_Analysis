# Phase 1 — V00 development baseline

V00 is a shortened development recording, not generated synthetic data and not a
validated participant. Checks establish software behavior, not scientific validity.
No testing framework, notebooks or cell markers have been introduced.

## How to check

Select the project virtual environment in VS Code. Open `scripts/check_v00.py` and
execute sequential lines/blocks with your interactive-window shortcut. Plain comment
headings identify the steps. Variables such as `event_table`, `selection_table`,
`features`, `direct_values` and `results` remain available for inspection.
Set `show_plots = True` if you want the optional diagnostic graph. It is not a
cleaned TEP. Do not import numbered processing scripts just to inspect functions.

Alternatively, from the repository root:

```bash
uv run python scripts/check_v00.py
```

The script does not execute ICA, SOUND, SSP-SIR, artifact interpolation or the full
filtering workflow. It builds diagnostic epochs with CURRENT baseline/detrend
defaults. It compares selected operations directly and loads existing outputs read
only. MATLAB round trips write only to automatically removed temporary directories.
It does not call the epoch-to-YAML exporter. Restart the interactive session and
run sequentially for a reproducible check; arbitrary out-of-order execution is unsafe.

PASS = software comparison passed. FAIL = inspect the explanation, not automatic
repair. NOT AVAILABLE = recording/output does not cover that case.
REQUIRES VISUAL INSPECTION = graphical or scientific review remains necessary.
The final RuntimeError is intentional when any FAIL exists; it prevents failures
being mistaken for a successful terminal run.

## Observed recording (October 23, 2026)

- Input: `data/raw/V00_data/V00_data.bdf`; no condition text file present.
- 121 seconds, 2,420,000 samples, 20,000 Hz.
- 32 EEG channels plus EOG and EMG after channel-type assignment.
- 18 Stimulus A pulses; condition triggers: 8Bit 1=6, 8Bit 2=9, 8Bit 3=3.
- One BAD_ACQ_SKIP annotation near the end; diagnostic epoch creation lost no trials.
- Symbols: `[1, 1, 0, 1, 1, 0, 2, 1, 0, 2, 1, 0, 2, 1, 0, 1, 1, 0]`.
- Development YAML: bad_channels=[F7, F3], bad_epochs=[2, 8, 15], ica_exclude=[0].
  Review date `2026-09-XX` is incomplete, not approval. YAML was left unchanged.
- 18 diagnostic epochs, then 15 after positional exclusions. No 400-pulse assertion.
- Retained contexts: ctx_0=4, ctx_2=3, ctx_01=1, ctx_11=3, ctx_21=3.
  First pulse has insufficient history for a two-symbol context. Block rules are
  unresolved; these checks use current continuous-sequence behavior only.

## Current preprocessing execution (not an agreed final protocol)

`scripts/01_preprocessing.py` currently runs at top level:

1. Select V00/default mode `tep`; set plotting backend; load BDF, types and montage.
2. TESA-like interpolation (-5 to +12 ms), overriding configured cubic mode.
3. EEG high-pass at 1 Hz, configured fourth-order zero-phase IIR.
4. Normalize annotations; construct EEG/EMG epochs (-0.8, +0.8) s. EEG requests
   baseline (-0.3, -0.02) s and linear epoch-wide detrending; EMG requests neither.
   Epoch construction initially includes all channel types; EMG is picked afterward.
5. Write epochs_record to subject YAML **regardless of io_export_data**; plot/inspect.
6. Load YAML once; mark bad channels; drop positional bad_epochs only in tep mode.
7. Average-reference EEG; fit full-epoch FastICA using 32 - bad EEG - 1 components,
   random_state=97. Display components, then apply the already loaded exclusions.
   Editing YAML after viewing ICA does NOT reload that dictionary automatically.
8. SOUND(iter_num=5, lambda_val=0.1), then SSP-SIR; interpolate marked bad channels.
9. EEG resample uses raw_downsample_freq=5000, not epoch_eeg_downsample_freq=1000.
   EMG resamples to 3000. EEG low-pass 80 Hz; EMG band-pass 20–500 Hz; notch 60 Hz,
   width 4 Hz, one harmonic, configured IIR order 4.
10. Inspect; crop EEG to the same (-0.8, +0.8) s interval; export by mode.
    EMG is saved only on the tep branch.

Known mutation behavior: custom filters/downsampling/interpolation return copies;
annotation processing returns a copy on V00; apply_bad_channels/apply_bad_epochs
modify their input; MNE pick/reference/interpolate/crop modify their object. ICA
application uses a copy when fitted. The check script applies exclusions on copies.

## Output contracts and side effects

- Raw/subject processed paths resolve from repository root via paths.py.
- EEG output: `data/processed/V00/processed_tep/V00_epochs_processed.fif`
  (or processed_context_tree). EMG: `emg_processed/V00_emg_epochs_processed.fif`.
- ROI output: `processed_rois/V00_epochs_rois_processed.fif`, shared across modes.
- Writers normally overwrite. Figures also lack sufficient mode/stage provenance.
- Group database config is `data/group/database.csv`, relative to working directory.
- MATLAB: X_ter is 1 x n_epochs symbols; Y_ter is 2 x n_EEG_channels cell array;
  each signal cell is n_samples x n_epochs in microvolts. Phase 1 requires exactly
  one valid symbol per epoch and checks crop bounds. External consumers must confirm
  this alignment contract; a separate full history would need explicit identities.
- Feature extraction returns condition-average features, not trial features:
  mean/signed peak in uV, latency ms, signed AUC uV*s by current defaults. P30 uses
  [0.020, 0.040) s. Other methods still have inclusive boundaries: see to-solve.md.

## Narrow Phase 1 corrections

- MATLAB exporter forwards `window` to its writer; validates symbols/count/bounds.
- Readers reject multiple matching BDF/epoch files instead of choosing the first.
- YAML must explicitly contain three exclusion lists. Empty lists are valid; missing,
  null, numeric placeholders, duplicates and invalid numeric exclusions are rejected.
  Channel names and index bounds are checked at application. These checks are not
  scientific review approval. Bad-epoch meaning remains CURRENT POSITION, not
  original identity; compare with epochs.selection before applying.
- ROI construction preserves selection, drop_log and metadata without reapplying
  baseline. Existing saved ROI files are NOT modified.
- Event finder assigns resolved codes directly, avoiding remap collisions when
  conditions are absent. A subset of V00 verifies missing-condition coding.
- Annotation replacement raises on errors/unresolved Stimulus A rather than silently
  proceeding with unlabeled pulses; condition parser rejects symbols outside 0/1/2.
- Short condition-sequence repetition and first-text-file selection remain unchanged
  pending a recovery-policy decision (S10). V00 does not exercise that fallback.

## Validation findings and limitations

The in-memory V00 checks cover mappings, rejection identity, context history, ROI
arithmetic/identity, direct P30 calculations, and both MATLAB windows. Small malformed
decision dictionaries exercise missing/placeholder handling without generating EEG.
Existing EEG has 15 trials; EMG has 18, with 15 common original event samples.

**Known failing saved artifact:** existing ROI selection differs from electrode
selection. New in-memory ROI construction passes. Do not erase this failure by
weakening the check. Confirm source mode and explicitly regenerate the old ROI file
using `scripts/02_create_ROIs.py` if desired; then rerun checks. Phase 1 preserves it.

Original BDF, YAML and processed files are checked for changes; the script never
automatically repairs them. No graphics, ICA fitting, SOUND/SSP-SIR or full processing
were validated. No scientific choice was approved. V00 cannot establish behavior for
all recordings, block boundaries, absent text inputs, or physiological inclusion.
MNE warns about existing filenames not ending in _epo.fif; renaming is deferred.

Final execution on October 23, 2026: **30 PASS, 1 FAIL**, with the one failure being
the untouched saved ROI selection mismatch described above. The process exits 1
intentionally. All 31 existing/new Python files parsed successfully and the changed
reusable modules imported. `git diff --check` reported no whitespace errors.
Ambiguous-file branches, text fallback, ICA application/bounds and ROI metadata
propagation are not fully exercised by this V00 run; these remain coverage limits,
not claims of complete correctness. The newly constructed ROI selection/drop log
and the existing EEG/EMG sample/selection/code correspondence were exercised.

See to-solve.md for open decisions and to-do.md for deferred refactoring.