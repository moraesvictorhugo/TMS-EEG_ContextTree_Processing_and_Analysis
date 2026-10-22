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
    - Mean amplitude in P30 (use the window in config) for both roi and channel level
    - Peak amplitude in P30 (use the window in config) for both roi and channel level
    - AUC in P30 (use the window in config) for both roi and channel level
    - N15-P30 peak-to-peak amplitude for both roi and channel level
    - GMFP - Global measure of the response (only for channel level)
    - LMFP - Local response (using rois only)

3 - Analisys level
    - Both roi and channel level

4 - Define TEP comparisons in the time, frequency, and time-frequency domains:
    - Is the TEP response the same in 1 expected (context 21) and 1 unexpected (ctx 01)?
    - Is the TEP response the same in 1 after a predictable 2 (context 21) and 1 after unpredictable (ctx 11)?
    - Are the TEP response in contexts 01, 11 and 21 different? The question here is focused in what was the previous pulse
    - Is there adaptation over time in context 21? Compare its first and last thirds.
    - Is there adaptation over time in context 11? Compare its first and last thirds.

5 - Export a long format table for statistics with columns as:
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
project_config = ProjectConfig()
subjects = project_config.analysis_subjects
export_enabled = project_config.io_export_data

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
        peak_selection=config.analysis_peak_selection,
        auc_absolute=config.analysis_auc_absolute,
    )

    # ================================================================ #
    #  PART 2 — CONDITION-LEVEL FEATURES BY ROI
    # ================================================================ #
    roi_extractor = FeatureExtractor(
        roi_epochs,
        roi_epochs.ch_names,
        config.analysis_time_windows,
        peak_selection=config.analysis_peak_selection,
        auc_absolute=config.analysis_auc_absolute,
    )

    # ── Channel-level condition features ──────────────────────────────
    channel_features = channel_extractor.extract_condition_features(
        component=config.analysis_p30_component,
        component1=config.analysis_n15_component,
        component2=config.analysis_p30_component,
    )
    for feature in channel_features.itertuples(index=False):
        collector.add_row(
            subject=subject_id,
            analysis_level="channel",
            analysis_type="condition",
            condition=feature.condition,
            channel=feature.channel,
            component=feature.component,
            metric=feature.metric,
            value=feature.value,
            unit=feature.unit,
        )

    # ── ROI-level condition features ──────────────────────────────────
    roi_features = roi_extractor.extract_condition_features(
        component=config.analysis_p30_component,
        component1=config.analysis_n15_component,
        component2=config.analysis_p30_component,
    )
    for feature in roi_features.itertuples(index=False):
        collector.add_row(
            subject=subject_id,
            analysis_level="roi",
            analysis_type="condition",
            condition=feature.condition,
            channel=feature.channel,
            component=feature.component,
            metric=feature.metric,
            value=feature.value,
            unit=feature.unit,
        )

    # GMFP is global across EEG channels, so it is recorded once per condition.
    gmfp = channel_extractor.compute_gmfp()
    gmfp_features = channel_extractor.extract_mfp_window_features(
        gmfp, component=config.analysis_p30_component, label="GMFP"
    )
    gmfp_peaks = channel_extractor.extract_mfp_peaks(
        gmfp, label="GMFP"
    ).query("component == @config.analysis_p30_component")
    for feature in gmfp_features.itertuples(index=False):
        collector.add_row(
            subject=subject_id,
            analysis_level="channel",
            analysis_type="condition",
            condition=feature.condition,
            channel=None,
            component=feature.component,
            metric=feature.metric,
            value=feature.value,
            unit=feature.unit,
        )
    for feature in gmfp_peaks.itertuples(index=False):
        collector.add_row(
            subject=subject_id,
            analysis_level="channel",
            analysis_type="condition",
            condition=feature.condition,
            channel=None,
            component=feature.component,
            metric="gmfp_peak_latency",
            value=feature.peak_latency_ms,
            unit="ms",
        )

    # LMFP is calculated over the saved ROI channels and recorded once per condition.
    lmfp = roi_extractor.compute_lmfp()
    lmfp_features = roi_extractor.extract_mfp_window_features(
        lmfp, component=config.analysis_p30_component, label="LMFP"
    )
    lmfp_peaks = roi_extractor.extract_mfp_peaks(
        lmfp, label="LMFP"
    ).query("component == @config.analysis_p30_component")
    for feature in lmfp_features.itertuples(index=False):
        collector.add_row(
            subject=subject_id,
            analysis_level="roi",
            analysis_type="condition",
            condition=feature.condition,
            channel=None,
            component=feature.component,
            metric=feature.metric,
            value=feature.value,
            unit=feature.unit,
        )
    for feature in lmfp_peaks.itertuples(index=False):
        collector.add_row(
            subject=subject_id,
            analysis_level="roi",
            analysis_type="condition",
            condition=feature.condition,
            channel=None,
            component=feature.component,
            metric="lmfp_peak_latency",
            value=feature.peak_latency_ms,
            unit="ms",
        )

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

    # ── Condition GMFP/LMFP plots ────────────────────────────────────
    if set(gmfp) == set(lmfp):
        mfp_plotter.plot_gmfp_lmfp(
            gmfp,
            lmfp,
            time_windows=config.analysis_time_windows,
        )
    else:
        print(
            f"Skipping combined GMFP/LMFP plot for {subject_id}: "
            "channel and ROI condition sets differ."
        )

# ── Export to CSV in long format if enabled ──
database = collector.export_csv(
    output_path=project_config.analysis_csv_path,
    export_enabled=export_enabled,
)
