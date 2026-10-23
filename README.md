# TMS-EEG Context Tree Processing and Analysis

This repository supports the PD-NEUROMAT study of how TMS intensity and stimulus history influence TMS-evoked potentials (TEPs). EEG recordings provide cortical responses; EMG recordings provide motor-evoked potentials (MEPs) for evaluating motor responses.

This README provides the study definitions, data conventions, and development principles needed for researchers and AI assistants to continue the project. Experimental definitions are authoritative. Processing defaults describe the current implementation and may change as methods are agreed.

## Study definitions

### Experimental design

Each volunteer receives **two blocks of 200 trials**, totaling **400 TMS pulses**.

| Stimulus symbol | TMS intensity |
|---|---|
| `0` | 80% resting motor threshold (rMT) |
| `1` | 100% rMT |
| `2` | 120% rMT |

Pulse sequences follow a context-tree model. A **context** is the minimum amount of stimulus history needed to predict the next event.

The model uses contexts `0`, `01`, `11`, `21`, and `2`:

| Context ending at the current pulse | Next pulse |
|---|---|
| `0` | `2` with 70% probability; `1` with 30% probability |
| `01` | `1`, deterministically |
| `11` | `0`, deterministically |
| `21` | `0`, deterministically |
| `2` | `1`, deterministically |

An example sequence is:

```text
2 → 1 → 0 → 2 → 1 → 0 → 1 → 1 → 0
```

The probabilistic choice occurs after `0`; the other transitions follow the table.

The initial stimulus and treatment of block boundaries are not specified in this document. Confirm these from the experimental protocol before implementing sequence generation or block-aware context assignment.

### Context notation and response selection

Context symbols are ordered **from past to present**.

For response analysis, the context ends at the pulse whose evoked response is analyzed:

| Analysis context | Response selected |
|---|---|
| `0` | Response following pulse `0` |
| `01` | Response following `1`, immediately preceded by `0` |
| `11` | Response following `1`, immediately preceded by `1` |
| `21` | Response following `1`, immediately preceded by `2` |
| `2` | Response following pulse `2` |

For example, context `21` predicts a subsequent `0` during sequence generation. However, the response analyzed for context `21` is evoked by its final pulse, `1`.

**Context assignment must use the original stimulus sequence.** Rejecting an EEG epoch must never make originally nonadjacent pulses become neighbors. Preserve original event identities through selection and transformation.

### Research questions

The study investigates:

- Whether responses to intensity `1` differ between contexts `01`, `11`, and `21`.
- How response differences relate to the sequence’s predictability rules.
- Whether responses change with repeated context occurrences, including comparisons of early and late occurrences.
- How electrode-level and region-of-interest (ROI) analyses characterize these effects.
- How MEP amplitudes vary across intensities and support evaluation of each volunteer’s responses.

These are research objectives. Their presence here does not imply that every analysis is implemented.

### Participants and inclusion

The workflow is intended to analyze every volunteer who participated in the study. The configured subject list selects which subjects to process; it does not establish final inclusion.

Post hoc exclusion criteria are being defined. An absence of MEP amplitude differences across intensities `0`, `1`, and `2` is a candidate criterion.

Do not infer a statistical test, threshold, or automatic exclusion rule from this proposal. Participant inclusion decisions and their reasons must remain traceable for reporting.

## Development principles

Keep the project understandable to a researcher with limited Python experience who needs to inspect intermediate results, modify processing steps, and debug the workflow.

- Prefer explicit scientific Python and a visible processing sequence.
- Use functions and modules to clarify responsibilities and remove meaningful repetition.
- Add classes or abstractions only when they solve a concrete need.
- Extend the existing package and numbered scripts before introducing frameworks or generalized pipeline infrastructure.
- Preserve stimulus meanings, original event order, epoch identity, units, and subject decision records.
- Ask the researcher before choosing an unresolved scientific method. Existing defaults, comments, and external examples do not establish agreed study methods.
- Distinguish confirmed study definitions, current implementation, and proposed methods.
- Keep shared parameters in configuration and subject-specific decisions in YAML.
- Validate changes with focused checks appropriate to the affected scientific calculation or transformation.
- State which checks actually ran and which require recordings or interactive review.
- Update documentation when workflow responsibilities or data contracts change. Keep detailed parameter definitions close to their implementation.

## Repository map and sources of truth

```text
scripts/
  01_preprocessing.py       Subject-level EEG/EMG preprocessing
  02_create_ROIs.py         ROI averages from processed epochs
  03_export_epochs_mat.py   MATLAB export branch
  04_tep_features.py        Feature database and condition/context plots
  05_statistics.R          Reserved statistics entry point
  06_plotting.R            Reserved plotting entry point

src/tms_eeg/
  paths.py                 Canonical repository, data, and decision paths
  config/
    settings.py            ProjectConfig: shared processing/analysis settings
    environment.py         Plotting backend selection
  io/                      Recording loading, output writing, YAML handling
  preprocessing/           Annotation processing and signal cleaning
  analysis/                Context mapping, ROIs, features, metric collection
  visualization/           TEP and field-power plots

decisions/                 Subject review records and methodological reference
data/                      Local recordings and generated results
logs_terminal/             Historical execution logs
utils/                     Standalone utilities
pyproject.toml             Package, dependencies, and tool settings
uv.lock                    Locked dependency resolution
```

Use these sources when making changes:

- [Project configuration](src/tms_eeg/config/settings.py) defines shared settings, symbol mappings, contexts, ROIs, and component windows.
- [Path helpers](src/tms_eeg/paths.py) define canonical subject paths.
- [Numbered scripts](scripts/) establish the operations actually executed and their order.
- [Subject decisions](decisions/) preserve review choices for reproducibility and reporting.
- [TESA reference notes](decisions/TESA_example.md) provide methodological background; they do not define the active preprocessing sequence.
- Historical terminal logs describe earlier runs and are not specifications for current behavior.

## Data conventions

### Subjects and input files

Subject IDs use forms such as `V00` or `V07`. Decision loading validates `V` followed by digits.

Raw data is expected under:

```text
data/raw/<subject>_data/
```

The reader loads a BDF from that directory. A condition text file can provide the stimulus sequence when condition annotations are unavailable.

The current raw and text readers select the first matching file. Keep inputs unambiguous. Multiple matching files do not provide explicit block handling.

### Annotations, event codes, and symbols

Annotation processing labels `Stimulus A` pulse events using nearby `8Bit` annotations, or a local text-file sequence when those annotations are absent.

| Annotation label; case varies in recordings | Default event code | Study symbol |
|---|---|---|
| `8bit 1` | `1` | `0` |
| `8bit 2` | `2` | `1` |
| `8bit 3` | `3` | `2` |

Annotation suffixes, numeric event codes, and study symbols are different representations. Use the mappings derived from `ProjectConfig.analysis_name_to_symbol`; do not interchange them.

The text fallback expects one study symbol—`0`, `1`, or `2`—per nonempty line and converts it to the corresponding annotation label.

Its current implementation repeats a sequence shorter than the pulse count. Check annotation alignment rather than assuming this fallback validates the experimental sequence.

### Units and epoch identity

- MNE signal arrays use volts.
- Configuration time values use seconds.
- Exported amplitude features use microvolts (`uV`).
- Exported latencies use milliseconds (`ms`).
- Exported AUC uses microvolt-seconds (`uV*s`).
- `epochs.selection` identifies original event positions after rejection. Context mapping uses this to recover original stimulus history.

### ROI signals

Each ROI signal is the arithmetic mean of the available electrodes in its configured group.

An ROI with no available electrodes is skipped. Check electrode availability when comparing subjects, because the contributing electrodes can differ.

### Output layout

Representative paths for `V00`:

```text
data/raw/V00_data/*.bdf
data/raw/V00_data/*.txt
decisions/sub-V00.yaml

data/processed/V00/
  processed_tep/V00_epochs_processed.fif
  processed_context_tree/V00_epochs_processed.fif
  emg_processed/V00_emg_epochs_processed.fif
  processed_rois/V00_epochs_rois_processed.fif
  processed_pre_and_post/
  processed_only_post/
  figures/

data/group/database.csv
```

The data directory and large recording/output formats are Git-ignored. Subject decision YAML files are versioned.

FIF writers overwrite their target files, and database export replaces its CSV. Figure saving is controlled separately by `io_save_figs`.

## Workflow

### 1. Subject preprocessing

[01_preprocessing.py](scripts/01_preprocessing.py) orchestrates the current sequence:

1. Load the BDF and assign channel types and montage.
2. Interpolate the TMS artifact in continuous data using the script’s TESA-like mode.
3. High-pass filter EEG.
4. Process annotations and construct EEG and EMG epochs.
5. Export original epoch indices and labels to YAML; inspect initial responses and epochs.
6. Load subject decisions, mark bad channels, and reject bad EEG epochs in TEP mode.
7. Apply average reference.
8. Fit and apply ICA.
9. Apply SOUND and SSP-SIR.
10. Interpolate bad channels and resample.
11. Apply EEG low-pass, EMG bandpass, and notch filters.
12. Inspect responses, crop EEG epochs, and export mode-specific results.

The script and called functions define execution. Some comments describe possible alternatives; inspect the implementation before changing scientific behavior.

| Processing mode | YAML bad-epoch rejection | Preprocessing outputs |
|---|---|---|
| `tep` | Applied to EEG epochs | EEG and separate EMG epochs |
| `context_tree` | Skipped | EEG epochs |

Skipping YAML rejection does not guarantee that every pulse survives epoch construction. Recording boundaries or annotation rejection can still remove epochs.

EMG currently does not receive EEG’s YAML trial rejection. Do not assume positional correspondence between saved EEG and EMG trial arrays.

### 2. Subject decisions and manual review

Subject YAML files preserve important decisions for future reporting. Shared configuration can change, so it must not be treated as a historical record of how an individual subject was processed.

| YAML key | Current role |
|---|---|
| `bad_channels` | EEG names marked bad before reference/ICA and interpolated later |
| `bad_epochs` | Zero-based positions passed to EEG epoch rejection in TEP mode |
| `ica_exclude` | Zero-based ICA component indices to remove |
| `epochs_record` | Generated original event indices and annotation labels before manual rejection |
| `notes`, `reviewed` | Human-maintained review context |

The generated `epochs_record.index` comes from `epochs.selection`. In contrast, `bad_epochs` is passed to `epochs.drop()` as positions in the current Epochs object. These may differ after earlier automatic rejection. Check the mapping before recording exclusions.

The preprocessing script loads decisions once before ICA. Editing YAML during ICA inspection does not automatically reload it. Ensure the intended decisions are loaded before applying exclusions or saving final outputs.

Epoch-record export writes or replaces that YAML block even when regular data export is disabled.

The supplied `sub-V00.yaml` contains test decisions; `sub-V01.yaml` contains placeholders. Neither establishes valid decisions for another volunteer.

### 3. ROI construction

[02_create_ROIs.py](scripts/02_create_ROIs.py) loads the selected mode’s processed epochs and creates one averaged channel per configured ROI.

Electrode-level data remains available for complementary analysis.

Both processing modes currently share the `processed_rois` output location. Recording the source preprocessing mode and settings alongside derived outputs is a planned provenance improvement after preprocessing is settled. Until then, ensure saved ROIs match the electrode epochs being analyzed.

### 4. MATLAB export branch

[03_export_epochs_mat.py](scripts/03_export_epochs_mat.py) is intended to export full pre/post and post-only windows for downstream context-tree analysis.

The MATLAB variable `data` contains:

- `X_ter`: a `1 × N` row vector of study symbols.
- `Y_ter`: a `2 × N_channels` cell array. Row 1 contains channel names; row 2 contains sample-by-epoch matrices in microvolts.

The script currently passes a `window` argument that `EpochAnnotationExporter.export_to_mat()` does not accept. Resolve that interface before running this branch, then verify actual crop windows and symbol-to-epoch alignment.

### 5. Features and visualization

[04_tep_features.py](scripts/04_tep_features.py) loads electrode and ROI epochs for `analysis_subjects`, collects condition-level features across subjects, and generates condition/context plots.

Context mapping reconstructs stimulus history from the raw recording.

Current numerical definitions:

- Electrode and ROI features come from **condition-averaged evoked responses**, not individual trials.
- Features include P30 mean amplitude, signed peak amplitude, peak latency, AUC, and N15–P30 peak-to-peak amplitude.
- Default peak selection chooses the largest absolute deflection while retaining its sign.
- Default AUC is signed.
- Field power is the standard deviation across channels at each time sample.
- GMFP uses all EEG channels.
- The current LMFP call computes dispersion across all saved ROI-average channels together. It does not compute separate field power within each anatomical ROI.
- Context-specific numerical features are not added to the current database; contexts feed plots.

The long-format database schema is:

```text
subject, analysis_level, analysis_type, condition, channel,
component, metric, unit, value
```

`analysis_level` is `channel` or `roi`. The `channel` field holds an electrode or ROI name and is empty for aggregate field-power metrics. The active script records `analysis_type="condition"`.

A future trial-level database needs an explicit schema for block/trial identity and inclusion decisions. These are absent from the current export.

## Scientific choices requiring agreement

The study definitions above are fixed. These methods remain open and must not be silently resolved during implementation:

- ICA selection: rank-based fitting versus a variance-based approach such as 95% explained variance.
- Baseline correction and detrending strategy, including any prestimulus-only approach.
- Final EEG resampling settings; 5000 Hz is not a study requirement.
- The scientifically appropriate LMFP definition.
- Post hoc participant exclusion criteria and their statistical definitions.

Inspect both configuration and call sites for current behavior:

- ICA currently calculates rank assuming 32 EEG channels and fits full epochs.
- Epoch construction currently requests baseline correction and detrending.
- EEG resampling currently uses `raw_downsample_freq` rather than `epoch_eeg_downsample_freq`.

These describe execution, not final methodological decisions.

## Setup and practical operation

Use Python 3.11 or newer; `.python-version` currently selects 3.11.

Dependencies are declared in [pyproject.toml](pyproject.toml) and resolved in [uv.lock](uv.lock).

From the repository root:

```bash
uv sync
```

MATLAB Engine (`24.2.*`) and PyTEP SOUND/SSP-SIR are declared dependencies. Preprocessing directly imports and calls SOUND and SSP-SIR. Check the local MATLAB installation and installed libraries’ requirements when preparing the environment.

Plotting selects Qt locally and a notebook backend for detected remote sessions. Interactive epoch and ICA review requires a suitable graphical environment. The remote branch references `ipympl`, which is not explicitly declared in project dependencies.

Before execution:

1. Place each subject’s BDF and any required condition text file in the expected raw directory.
2. Select the subject and mode in the script/configuration.
3. Prepare valid subject YAML decisions.
4. Inspect intermediate epochs and ICA results; record and load decisions before final processing.
5. Generate matching electrode and ROI outputs for every subject selected for feature extraction.

The first three scripts currently select `V00` in their source. Feature extraction uses `analysis_subjects` from configuration. There is no command-line subject/mode interface.

Some scripts execute on import, so inspect their source rather than importing them to explore the workflow.

Run the relevant stages after configuring them:

```bash
uv run python scripts/01_preprocessing.py
uv run python scripts/02_create_ROIs.py
uv run python scripts/04_tep_features.py
```

MATLAB export is a separate branch requiring the interface correction described above. Statistics and final R plotting scripts are reserved entry points.

Subject paths resolve from the repository root through `paths.py`. The configured database CSV path is relative to the working directory; run commands from the repository root.

## Validation expectations

No automated test suite or CI configuration is currently included.

Use focused numerical checks and representative recordings to validate changes. Depending on the change, check:

- Event counts and condition mappings.
- Original epoch identities and context assignments.
- Electrode availability and ROI composition.
- Signal units and time windows.
- Feature calculations.
- Alignment between symbols, EEG/EMG trials, and exported arrays.
- Output schemas and source preprocessing mode.

Report interactive and data-dependent checks separately from checks run automatically.

## Study contact

Victor Hugo Moraes — PD-NEUROMAT, University of São Paulo

Email: moraes.vh@usp.br