"""Readable V00 checks: run sequentially or select lines in the interactive window.

No preprocessing pipeline is run and no decisions or existing outputs are saved.
MATLAB round trips use an automatically deleted temporary directory.
PASS means a software check passed, not that a scientific method was validated.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import mne
import numpy as np
import pandas as pd
from scipy.io import loadmat

from tms_eeg.analysis.context import ContextMapper
from tms_eeg.analysis.features import FeatureExtractor
from tms_eeg.analysis.roi import create_roi_epochs
from tms_eeg.config.settings import ProjectConfig
from tms_eeg.io.reader import get_raw_path, load_data
from tms_eeg.io.writer import Writer
from tms_eeg.io.yaml_reader import load_decisions, validate_decisions
from tms_eeg.paths import decisions_file, processed_dir, raw_dir
from tms_eeg.preprocessing.annotation_exporter import EpochAnnotationExporter
from tms_eeg.preprocessing.annotation_processor import AnnotationProcessor
from tms_eeg.preprocessing.cleaning import apply_bad_channels, apply_bad_epochs
from tms_eeg.preprocessing.epoching import create_epochs, find_events


# Settings and reporting. Keep plots optional for terminal/non-graphical use.
config = ProjectConfig(subject_id="V00", io_export_data=False, io_save_figs=False)
show_plots = False
results = []


def check(description, passed, advice):
    """Print a comparison and keep failures visible in the final summary."""
    status = "PASS" if passed else "FAIL"
    results.append((status, description))
    print(f"{status}: {description}")
    if not passed:
        print(f"  Inspect: {advice}")


# Protect existing files. Do not call export_epochs_to_decisions in this script.
decision_path = decisions_file(config.subject_id)
decision_before = decision_path.read_bytes()
existing_outputs = list(processed_dir(config.subject_id).rglob("*"))
output_before = {
    path: (path.stat().st_size, path.stat().st_mtime_ns)
    for path in existing_outputs if path.is_file()
}
raw_path = Path(get_raw_path(config))
raw_before = (raw_path.stat().st_size, raw_path.stat().st_mtime_ns)
print("Raw input:", raw_path)
print("Condition files:", list(raw_dir(config.subject_id).glob("*.txt")))
print("Decisions (read only):", decision_path)
print("Existing outputs (read only):", list(output_before))


# Recording inventory. Reading lazily avoids loading the entire continuous file.
raw = mne.io.read_raw_bdf(raw_path, preload=False)
raw.set_channel_types({config.ch_eog_label: "eog", config.ch_emg_label: "emg"})
raw.set_montage(config.ch_eeg_montage)
print("Sampling frequency (Hz):", raw.info["sfreq"])
print("Duration (s):", raw.n_times / raw.info["sfreq"])
print("Channel types:", pd.Series(raw.get_channel_types()).value_counts().to_dict())
print("Annotation counts:", pd.Series(raw.annotations.description).value_counts().to_dict())
original_annotations = raw.annotations.description.copy()
pulse_count = sum(original_annotations == "Stimulus A")
processed_raw = AnnotationProcessor(config).process_annotations(raw)
check("Annotation processing leaves its input unchanged",
      np.array_equal(raw.annotations.description, original_annotations), "copy/in-place behavior")
events, event_id = find_events(processed_raw, config)
code_to_label = {code: label for label, code in event_id.items()}
full_sequence = np.array([config.analysis_event_to_symbol[code] for code in events[:, 2]])
event_table = pd.DataFrame({
    "original_index": np.arange(len(events)),
    "sample": events[:, 0],
    "label": [code_to_label[code] for code in events[:, 2]],
    "event_code": events[:, 2],
    "symbol": full_sequence,
})
print(event_table.to_string(index=False))
check("Each pulse has one labeled event", len(events) == pulse_count,
      "annotation matching; never interpret leftover triggers as additional pulses")
check("Labels agree with code-to-symbol mapping",
      all(config.analysis_name_to_symbol[label.lower()] == symbol
          for label, symbol in zip(event_table.label, full_sequence)), "event mappings")

# A real-data subset checks coding when symbol 0 is absent; no generated signals.
subset_raw = processed_raw.copy()
subset_raw.set_annotations(processed_raw.annotations[
    np.isin(processed_raw.annotations.description, ["8Bit 2", "8Bit 3"])
])
subset_events, subset_ids = find_events(subset_raw, config)
check("Missing condition does not renumber remaining event codes",
      subset_ids == {"8Bit 2": 2, "8Bit 3": 3}
      and np.array_equal(subset_events[:, 2], events[np.isin(events[:, 2], [2, 3]), 2]),
      "event-code assignment when conditions are absent")


# Construct diagnostic epochs using CURRENT defaults, not cleaned scientific data.
# create_epochs preloads epochs; do not load the entire raw recording first.
epochs = create_epochs(processed_raw, config)
decisions = load_decisions(config)
validate_decisions({"bad_channels": [], "bad_epochs": [], "ica_exclude": []})
for invalid_decisions in (
    {},
    {"bad_channels": [], "bad_epochs": ["X"], "ica_exclude": []},
):
    try:
        validate_decisions(invalid_decisions)
    except ValueError as error:
        print("PASS: Invalid decision rejected:", error)
        check("Missing/placeholder decision rejected", True, "decision validation")
    else:
        check("Missing/placeholder decision rejected", False, "decision validation")
print("Development decisions:", decisions["bad_channels"], decisions["bad_epochs"], decisions["ica_exclude"])
print("Review field (not scientific approval):", decisions.get("reviewed"))
print("Automatic losses (original indices):", sorted(set(range(len(events))) - set(epochs.selection)))
selected = apply_bad_channels(epochs.copy(), decisions)
selected = apply_bad_epochs(selected, decisions, config)
selection_table = event_table.iloc[selected.selection].copy()
selection_table.insert(0, "current_position", np.arange(len(selected)))
print(selection_table.to_string(index=False))
check("Rejection on a copy leaves original epochs unchanged",
      len(selected) == len(epochs) - len(decisions["bad_epochs"])
      and np.array_equal(epochs.events, events[epochs.selection])
      and epochs.info["bads"] == [],
      "which object received drop()")
check("Retained samples match original event identities",
      np.array_equal(selected.events, events[selected.selection]), "selection and event alignment")

# Invalid development decisions should stop before mutating the input object.
invalid_channels = {**decisions, "bad_channels": ["NOT_A_CHANNEL"]}
try:
    apply_bad_channels(epochs.copy(), invalid_channels)
except ValueError as error:
    print("Expected error:", error)
    check("Unknown bad channel rejected", True, "channel validation")
else:
    check("Unknown bad channel rejected", False, "channel validation")
invalid_epochs = {**decisions, "bad_epochs": [len(epochs)]}
try:
    apply_bad_epochs(epochs.copy(), invalid_epochs, config)
except ValueError as error:
    print("Expected error:", error)
    check("Out-of-range epoch position rejected", True, "epoch validation")
else:
    check("Out-of-range epoch position rejected", False, "epoch validation")


# Contexts: compare the mapper with direct slices of the ORIGINAL sequence.
context_map = ContextMapper(config).classify_epochs(full_sequence, selected.selection)
for name, pattern in config.analysis_context_definitions.items():
    expected_positions = []
    for position, original_index in enumerate(selected.selection):
        history = full_sequence[max(0, original_index - len(pattern) + 1):original_index + 1]
        if history.tolist() == pattern:
            expected_positions.append(position)
    check(f"Original-history assignment for {name}", context_map[name] == expected_positions,
          "do not reconstruct history from retained epochs")
    if not expected_positions:
        print(f"NOT AVAILABLE: {name} is not represented after these exclusions.")
selection_table["context"] = [
    ", ".join(name for name, positions in context_map.items() if position in positions)
    for position in range(len(selected))
]
selection_table["original_history"] = [
    full_sequence[max(0, index - 1):index + 1].tolist() for index in selected.selection
]
print(selection_table.to_string(index=False))


# One ROI: direct arithmetic mean, original identities, and no source mutation.
roi_epochs = create_roi_epochs(selected, config.analysis_rois)
roi_name = roi_epochs.ch_names[0]
contributing = [name for name in config.analysis_rois[roi_name] if name in selected.ch_names]
direct_roi = selected.get_data(picks=contributing).mean(axis=1)
check(f"ROI {roi_name} equals mean of {contributing}",
      np.allclose(roi_epochs.get_data(picks=[roi_name])[:, 0], direct_roi, rtol=1e-10, atol=1e-15),
      "ROI channel membership and averaging axis")
check("ROI preserves original selection and events",
      np.array_equal(roi_epochs.selection, selected.selection)
      and np.array_equal(roi_epochs.events, selected.events), "ROI reconstruction")


# One condition/electrode: independently check P30 amplitude, peak, latency, AUC.
condition = next(name for name in selected.event_id if len(selected[name]))
channel = "C3"
extractor = FeatureExtractor(selected, [channel], config.analysis_time_windows,
                             peak_selection=config.analysis_peak_selection,
                             auc_absolute=config.analysis_auc_absolute)
features = extractor.extract_condition_features(conditions=[condition])
evoked = selected[condition].average(picks=[channel])
start, end = config.analysis_time_windows[config.analysis_p30_component]
mask = (evoked.times >= start) & (evoked.times < end)
signal = evoked.data[0, mask]
times = evoked.times[mask]
peak_index = int(np.argmax(np.abs(signal)))
direct_values = {
    "mean_amplitude": signal.mean() * 1e6,
    "peak_amplitude": signal[peak_index] * 1e6,
    "peak_latency": times[peak_index] * 1e3,
    "auc": np.trapezoid(signal, times) * 1e6,
}
print("Diagnostic feature input:", condition, channel, "(not cleaned TEPs)")
for metric, direct_value in direct_values.items():
    row = features[features.metric == metric].iloc[0]
    print(metric, "direct:", direct_value, "helper:", row.value, row.unit)
    check(f"Direct calculation: {metric}", np.isclose(direct_value, row.value, rtol=1e-10, atol=1e-12),
          "window boundaries, signed peak/AUC, V-to-uV or s-to-ms conversion")


# MATLAB: actual writer round trips in temporary storage, never saved V00 outputs.
exporter = EpochAnnotationExporter(config)
_, labels = exporter.extract_annotations(selected)
symbols = exporter.map_annotations_to_symbols(labels)
export_config = ProjectConfig(subject_id="V00", io_export_data=True, io_save_figs=False)
writer = Writer(export_config)
try:
    exporter.export_to_mat(writer, selected, symbols[:-1], subfolder="unused-validation-path")
except ValueError as error:
    print("Expected error:", error)
    check("Symbol-count mismatch rejected before writing", True, "MATLAB alignment")
else:
    check("Symbol-count mismatch rejected before writing", False, "MATLAB alignment")
for window in (config.export_pre_and_post_window, config.export_only_post_window):
    with TemporaryDirectory(prefix="v00-check-") as folder:
        exporter.export_to_mat(writer, selected, symbols, subfolder=folder, window=window)
        mat_path = Path(folder) / "V00_epochs_contexttree.mat"
        saved = loadmat(mat_path, struct_as_record=False, squeeze_me=False)["data"][0, 0]
        expected = selected.copy().pick("eeg").crop(tmin=window[0], tmax=window[1])
        check(f"MATLAB symbols match retained trials for {window}",
              np.array_equal(saved.X_ter, symbols.reshape(1, -1)), "symbol identity/order")
        check(f"MATLAB shape and units for {window}",
              saved.Y_ter.shape == (2, len(expected.ch_names))
              and np.allclose(saved.Y_ter[1, 0], expected.get_data()[:, 0, :].T * 1e6),
              "sample-by-epoch transpose, crop and microvolt conversion")


# Existing outputs are optional and may predate these corrections.
eeg_folder = processed_dir("V00") / "processed_tep"
if list(eeg_folder.glob("*_epochs_processed.fif")):
    saved_eeg = load_data(config, data_type="epochs")
    emg_path = processed_dir("V00") / "emg_processed" / "V00_emg_epochs_processed.fif"
    if emg_path.is_file():
        saved_emg = mne.read_epochs(emg_path, preload=False)
        common_samples = np.intersect1d(saved_eeg.events[:, 0], saved_emg.events[:, 0])
        print("Saved EEG/EMG counts:", len(saved_eeg), len(saved_emg))
        print("Common original event samples:", len(common_samples))
        eeg_positions = np.flatnonzero(np.isin(saved_eeg.events[:, 0], common_samples))
        emg_positions = np.array([
            np.flatnonzero(saved_emg.events[:, 0] == sample)[0]
            for sample in saved_eeg.events[eeg_positions, 0]
        ])
        check("Saved EEG trials map to matching EMG original identities and codes",
              len(common_samples) == len(saved_eeg)
              and np.array_equal(saved_eeg.selection[eeg_positions], saved_emg.selection[emg_positions])
              and np.array_equal(saved_eeg.events[eeg_positions, 2], saved_emg.events[emg_positions, 2]),
              "match original samples/selection before combining modalities")
        print("REQUIRES VISUAL INSPECTION: EEG and EMG are not assumed to have equal rows.")
    else:
        print("NOT AVAILABLE: saved EMG output.")
    saved_roi_path = processed_dir("V00") / "processed_rois" / "V00_epochs_rois_processed.fif"
    if saved_roi_path.is_file():
        saved_roi = mne.read_epochs(saved_roi_path, preload=False)
        check("Existing ROI output matches electrode event identities",
              np.array_equal(saved_roi.events, saved_eeg.events)
              and np.array_equal(saved_roi.selection, saved_eeg.selection),
              "old ROI files may need explicit regeneration; source mode is not recorded")
else:
    print("NOT AVAILABLE: saved TEP output. In-memory checks above still apply.")


# Optional plot: inspect variables first, then enable this block interactively.
if show_plots:
    import matplotlib.pyplot as plt

    plt.plot(evoked.times * 1e3, evoked.data[0] * 1e6)
    plt.xlabel("Time (ms)")
    plt.ylabel("Amplitude (uV)")
    plt.title(f"V00 diagnostic, not cleaned: {condition}, {channel}")
    plt.show()
print("REQUIRES VISUAL INSPECTION: final TEP/EMG quality and ICA/SOUND/SSP-SIR behavior.")


# Final safety check and readable outcome table. Nothing is repaired automatically.
check("Decision YAML unchanged", decision_path.read_bytes() == decision_before, "unexpected YAML write")
check("Raw recording unchanged",
      (raw_path.stat().st_size, raw_path.stat().st_mtime_ns) == raw_before, "unexpected raw write")
check("Existing outputs unchanged",
      {path: (path.stat().st_size, path.stat().st_mtime_ns)
       for path in processed_dir(config.subject_id).rglob("*") if path.is_file()} == output_before,
      "unexpected save operation")
print(pd.DataFrame(results, columns=["status", "check"]).to_string(index=False))
print("No scientific methods were approved by these checks.")
if any(status == "FAIL" for status, _ in results):
    raise RuntimeError("V00 checks found differences. Read FAIL messages above; files were not repaired.")