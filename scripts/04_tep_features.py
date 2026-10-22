"""Main analysis pipeline for TMS-EEG data."""

import mne

from tms_eeg.analysis.context import ContextMapper
from tms_eeg.analysis.features import FeatureExtractor
from tms_eeg.analysis.group import MetricsCollector

# Set backend
from tms_eeg.config.environment import setup_plotting_backend
from tms_eeg.config.settings import ProjectConfig
from tms_eeg.io.reader import get_raw_path, load_data
from tms_eeg.paths import processed_dir
from tms_eeg.visualization.gfp_plots import MFPPlotter
from tms_eeg.visualization.tep_plots import TEPPlotter

setup_plotting_backend()

'''
# TODO:

1 - Define validation criteria for the first subject:
    - Are MEPs different between 0, 1 and 2?
    - Are MEPs > 50 uV in 1?
    - Was the EEG muscle artifact successfully removed?
    
2 - Features
    - Mean amplitude in P30 (use the window in config)
    - Peak amplitude in P30 (use the window in config)
    - AUC in P30 (use the window in config)
    - N15-P30 peak-to-peak amplitude
    - GMFP - Global measure of the response (only for channel level)
    - LMFP - Local response (using rois)

3 - Analisys level
    - Both roi and channel level

4 - Define TEP comparisons in the time, frequency, and time-frequency domains:
    - Is the TEP response the same in 1 expected (context 21) and 1 unexpected (ctx 01)?
    - Is the TEP response the same in 1 after a predictable 2 (context 21) and 1 after unpredictable (ctx 11)?
    - Are the TEP response in contexts 01, 11 and 21 different? The question here is focused in what was the previous pulse
    - Is there adaptation over time in context 21? Compare its first and last thirds.
    - Is there adaptation over time in context 11? Compare its first and last thirds.

X - Export a long format table for statistics with columns as:
    - Participant ID
    - Block number
    - Trial number
    - TMS pulse intensity (0, 1 or 2)
    - Context (Following Hernández, an example of context in my case is: present pulse + past pulse = 11)
    - Number of times this context appeared
    - Predictability of this context
    - Interval to previous pulse
    - Interval since the previous pulse of the same intensity
    - Pre-pulse cortical state
    - Selected post-pulse features
    - Include invalid participants with an identifier and inclusion status/reason

'''

collector = MetricsCollector()
subjects = ProjectConfig().analysis_subjects
export_enabled = ProjectConfig().io_export_data

for subject_id in subjects:
    config = ProjectConfig(subject_id=subject_id)
    epochs = load_data(config, data_type="epochs")
    roi_epochs_path = (
        processed_dir(subject_id)
        / "processed_rois"
        / f"{subject_id}_epochs_rois_processed.fif"
    )
    if not roi_epochs_path.is_file():
        raise FileNotFoundError(
            f"ROI epochs not found for {subject_id}: {roi_epochs_path}. "
            "Run scripts/02_create_ROIs.py first."
        )
    roi_epochs = mne.read_epochs(roi_epochs_path, preload=True)

    # ── Rename conditions ────────────────────────────────────────────
    label_map = {k.lower(): str(v) for k, v in config.analysis_name_to_symbol.items()}
    epochs.event_id = {label_map[k.lower()]: v for k, v in epochs.event_id.items()}
    roi_epochs.event_id = {
        label_map[k.lower()]: v for k, v in roi_epochs.event_id.items()
    }

    # ================================================================ #
    #  PART 1 — CONDITION-LEVEL FEATURES BY CHANNEL
    # ================================================================ #
    channel_extractor = FeatureExtractor(
        epochs,
        config.analysis_channels_of_interest,
        config.analysis_time_windows,
    )

    # ================================================================ #
    #  PART 2 — CONDITION-LEVEL FEATURES BY ROI
    # ================================================================ #
    roi_extractor = FeatureExtractor(
        roi_epochs,
        roi_epochs.ch_names,
        config.analysis_time_windows,
    )

    # Add extracted feature rows to `collector` in both sections above.
    # Use analysis_level="channel" for channel_extractor results and
    # analysis_level="roi" for roi_extractor results. Each row should carry
    # subject, analysis_type, condition, channel/ROI, component, metric, value.

    # ================================================================ #
    #  PART 3 — CONTEXT-LEVEL ANALYSIS (original channel epochs)
    # ================================================================ #
    tep_plotter = TEPPlotter(config=config)
    mfp_plotter = MFPPlotter(times=epochs.times, config=config)

    # Visualize condition-level TEPs using the saved ROI channels.
    tep_plotter.plot_evoked_by_symbol(roi_epochs, picks=roi_epochs.ch_names)

    raw_path = get_raw_path(config)
    context_mapper = ContextMapper(config)
    context_epochs = context_mapper.get_context_epochs(epochs, raw_path)

    # ── Context evokeds ──────────────────────────────────────────────
    ctx_evokeds = {
        ctx_name: ctx_ep.average().pick(config.analysis_channels_of_interest)
        for ctx_name, ctx_ep in context_epochs.items()
    }

    # ── Context TEP plots ────────────────────────────────────────────
    tep_plotter.plot_mean_tep(evokeds=ctx_evokeds)

    # ── Contexts comparison ──────────────────────────────────────────
    tep_plotter.plot_context_comparison(context_epochs)

# ── Export to CSV in long format if enabled ──
database = collector.export_csv(
    output_path="data/group/database.csv",
    export_enabled=export_enabled,
)
