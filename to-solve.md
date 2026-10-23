# Open questions — preserve the research reminders

All entries below are OPEN. Comments are research reminders, not approved methods
and not instructions for an AI to implement an alternative. Fixed study meanings
in README.md take precedence. Record any eventual decision, date, rationale and
affected outputs here. Do not use generated egg-info as a specification.

Sources below are relative to this repository root. Original comments are retained
in the scripts; this document supplies context rather than deleting reminders.

## S01 — ICA component selection and fitting interval

Source: scripts/01_preprocessing.py, step 8:
`Use 95% explained variance, fitting on the prestimulus interval to remove residual blink artifacts.`
Current: preprocessing/ica.py requires 32 EEG channels, subtracts bad channels and
one reference dimension, then fits FastICA on full epochs (random_state=97).
Questions: rank versus explained variance; full versus prestimulus-only fitting;
whether the rank assumptions remain valid. Inspect representative results before
choosing. V00 shape/numerical checks do not settle this.

## S02 — Detrending

Source: scripts/01_preprocessing.py:
`Apply custom detrending to the prestimulus interval only; do not use MNE's epoch-wide detrending.`
Current: create_epochs passes epoch_eeg_detrend=1 to MNE, on the full epoch.
Question: intended detrending interval, fit/application procedure, and order relative
to baseline correction. No custom prestimulus detrending was introduced in Phase 1.

## S03 — Baseline correction and second interpolation

Source: scripts/01_preprocessing.py, `Baseline correction` and step 16:
`Baseline correction or another interpolation pass?`
The commented apply_baseline and second ArtifactRemover calls are proposals.
Current: epoch construction already requests baseline (-0.3, -0.02) s; the later
correction/interpolation calls are inactive. second_artifact_window is configured
but not used by those active operations. Decide whether another pass is warranted,
its interval, channel scope and processing order. Do not activate both by default.

## S04 — Cropping and edge effects

Source: scripts/01_preprocessing.py step 17:
`Cropping to −800 to +800 ms (to remove edge effects)`.
Current: initial epoch and final crop both use (-0.8, 0.8) s, so this does not trim
an extra margin. Decide whether wider initial epochs or another final window was
intended. Check filtering/recording boundaries before changing durations.

## S05 — EEG resampling

Sources: scripts/01_preprocessing.py step 13 and config/settings.py.
Current EEG call uses raw_downsample_freq=5000 Hz; epoch_eeg_downsample_freq=1000 Hz
is not used there. EMG uses 3000 Hz. Decide the final EEG frequency and the correct
setting responsibility; neither configured value is an agreed study requirement.

## S06 — Artifact interpolation, SOUND/SSP-SIR and reference examples

Sources: scripts/01_preprocessing.py, config/settings.py, decisions/TESA_example.md.
Current: script selects TESA-like polynomial interpolation despite configured cubic
default; SOUND uses iter_num=5/lambda_val=0.1 directly in script, then SSP-SIR.
Questions: interpolation mode/anchors/polynomial settings/channel scope; effective
parameters; order and combination of ICA, SOUND, SSP-SIR and filters. TESA notes
are reference material, not the active protocol. Do not change order from examples.

## S07 — LMFP and feature definitions

Sources: analysis/features.py and scripts/04_tep_features.py LMFP TODO.
Current LMFP is dispersion across saved ROI-average channels, not separate field
power within each anatomical ROI. Decide the intended spatial quantity and name.
Also resolve inclusive versus half-open windows across feature methods and desired
peak polarity before consolidating implementations. Phase 1 preserves definitions.

## S08 — Participant evaluation/inclusion

Source: scripts/04_tep_features.py TODO:
`Are MEPs different between 0, 1 and 2?`, `Are MEPs > 50 uV in 1?`,
`Was the EEG muscle artifact successfully removed?`
These are proposals, not implemented exclusion rules. Agree measurement windows,
statistics, threshold meanings, review criteria and recorded reasons. V00 decisions
are development values, not participant inclusion evidence.

## S09 — Predictability, context notation and future analyses

Source: scripts/04_tep_features.py TODO comparisons of expected/unexpected responses,
adaptation (first/last thirds), frequency/time-frequency analyses and trial database.
The TODO's `present pulse + past pulse` wording conflicts with README's fixed
past-to-present notation. Preserve the research intention but use the fixed notation.
Clarify what predictability refers to (current response-evoking pulse, preceding
history, or next event). Do not assign it from a context name without agreement.
Agree trial-level schema, context occurrence counting, adaptation splits, and
statistical analyses separately; these are not implemented by Phase 1 checks.

## S10 — Block boundaries and condition-file recovery

Sources: README.md; preprocessing/annotation_processor.py.
Initial pulse and cross-block history rules are unspecified. Never infer them from
the shortened V00 recording. Text fallback still repeats short sequences and selects
the first text file: decide whether to forbid this or explicitly authorize a recovery
policy. Phase 1 reports this limitation rather than silently changing the policy.
Also review trigger matching tolerance, case handling and one-to-one matching on
other recordings; V00 uses existing `8Bit` labels and has no text fallback file.

## S11 — Existing V00 ROI output and review records

Source: Phase 1 execution of scripts/check_v00.py.
Saved EEG has 15 trials; saved EMG has 18. Existing saved ROI has 15 trials but its
selection indices do not match EEG's original indices. New ROI construction preserves
selection/drop_log/metadata. Existing files were not overwritten. After confirming
their source mode, explicitly regenerate ROIs with scripts/02_create_ROIs.py and
rerun the checks. Do not claim the old file was repaired by changing source code.
The V00 YAML review date is `2026-09-XX`; exclusions are development choices. Decide
how review status/dates should be recorded without treating placeholders as approval.