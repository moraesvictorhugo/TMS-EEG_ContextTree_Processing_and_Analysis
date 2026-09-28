"""Per-subject analysis: feature extraction by condition and by context.

For every subject in ``config.analysis.subjects`` this script:

1. Loads the preprocessed epochs (``data/processed/<subject>/processed_full``).
2. Normalises the condition labels (8bit 1/2/3 -> 0/1/2).
3. Computes condition-level features (N15-P30 / N15-P60 / N100-P180
   peak-to-peak amplitudes, GMFP / LMFP and their component peaks).
4. Maps the surviving epochs to the context tree and repeats the feature
   extraction per context.
5. Writes a tidy (long) metrics CSV per subject in ``data/group/`` and a
   combined ``database.csv`` consumed by ``main_statistics.py`` and
   ``main_plotting.py``.

Usage
-----
    python main_analysis.py                 # all subjects, with plots
    python main_analysis.py --subject V05   # single subject
    python main_analysis.py --no-plots      # headless
"""

from __future__ import annotations

import argparse

import pandas as pd

from tms_eeg.analysis.context import ContextMapper
from tms_eeg.analysis.features import FeatureExtractor
from tms_eeg.analysis.group import MetricsCollector
from tms_eeg.analysis.labels import normalize_event_id
from tms_eeg.config.environment import setup_plotting_backend
from tms_eeg.config.settings import ProjectConfig
from tms_eeg.io.reader import get_raw_path, load_data
from tms_eeg.io.writer import Writer
from tms_eeg.visualization.gfp_plots import MFPPlotter
from tms_eeg.visualization.tep_plots import TEPPlotter

# Component pairs used for peak-to-peak amplitude extraction.
P2P_PAIRS = [("N15", "P30"), ("N15", "P60"), ("N100", "P180")]


def analyze_subject(subject_id: str, run_plots: bool = True) -> pd.DataFrame:
    """Compute all condition- and context-level features for one subject.

    Returns
    -------
    pd.DataFrame
        Tidy metrics rows for this subject (long format).
    """
    config = ProjectConfig(subject_id=subject_id)
    config.plots.analysis_plots = config.plots.analysis_plots and run_plots

    epochs = load_data(config, data_type="epochs")
    normalize_event_id(epochs, config.events.label_rename)

    writer = Writer(config)
    extractor = FeatureExtractor(
        epochs, config.analysis.channels_of_interest, config.analysis.time_windows)
    tep_plotter = TEPPlotter(config=config, writer=writer)
    mfp_plotter = MFPPlotter(times=epochs.times, config=config, writer=writer)
    collector = MetricsCollector()

    # ==================================================================== #
    #  Part 1 - Condition level (8bit 0/1/2)
    # ==================================================================== #
    evokeds = extractor.get_evokeds()

    if config.plots.analysis_plots:
        tep_plotter.plot_mean_tep(evokeds=evokeds)

    for comp1, comp2 in P2P_PAIRS:
        df = extractor.peak_to_peak(comp1, comp2, evokeds=evokeds)
        collector.collect_peak_to_peak_from_df(
            subject_id, "condition", df, f"{comp1}-{comp2}")

    gmfp = extractor.compute_gmfp()
    lmfp = extractor.compute_lmfp()

    if config.plots.analysis_plots:
        mfp_plotter.plot_gmfp_lmfp(
            gmfp, lmfp, time_windows=config.analysis.time_windows)
        mfp_plotter.plot_overlay(
            gmfp, label="GMFP", time_windows=config.analysis.time_windows)
        mfp_plotter.plot_overlay(
            lmfp, label="LMFP", time_windows=config.analysis.time_windows)

    collector.collect_mfp_peaks_from_df(
        subject_id, "condition", extractor.extract_mfp_peaks(gmfp, label="GMFP"), "GMFP")
    collector.collect_mfp_peaks_from_df(
        subject_id, "condition", extractor.extract_mfp_peaks(lmfp, label="LMFP"), "LMFP")

    # ==================================================================== #
    #  Part 2 - Context level (context tree)
    # ==================================================================== #
    raw_path = get_raw_path(config)
    context_epochs = ContextMapper(config).get_context_epochs(epochs, raw_path)

    for ctx_name, ctx_epochs in context_epochs.items():
        ctx_extractor = FeatureExtractor(
            ctx_epochs, config.analysis.channels_of_interest, config.analysis.time_windows)

        ctx_evokeds = ctx_extractor.get_evokeds()

        # Peak-to-peak per context (label rows with the context name).
        for comp1, comp2 in P2P_PAIRS:
            df = ctx_extractor.peak_to_peak(comp1, comp2, evokeds=ctx_evokeds).copy()
            df["condition"] = ctx_name
            collector.collect_peak_to_peak_from_df(
                subject_id, "context", df, f"{comp1}-{comp2}")

        # GMFP / LMFP peaks per context.
        for label, mfp in (("GMFP", ctx_extractor.compute_gmfp()),
                           ("LMFP", ctx_extractor.compute_lmfp())):
            df = ctx_extractor.extract_mfp_peaks(mfp, label=label).copy()
            df["condition"] = ctx_name
            collector.collect_mfp_peaks_from_df(subject_id, "context", df, label)

    if config.plots.analysis_plots and context_epochs:
        ctx_evokeds_all = {
            name: ctx_ep.average().pick(config.analysis.channels_of_interest)
            for name, ctx_ep in context_epochs.items()
        }
        tep_plotter.plot_mean_tep(evokeds=ctx_evokeds_all)
        tep_plotter.plot_context_comparison(context_epochs)

    df = collector.to_dataframe()
    subject_csv = config.paths.group_dir / f"{subject_id}_metrics.csv"
    subject_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(subject_csv, index=False)
    print(f"\n[{subject_id}] {len(df)} metric rows -> {subject_csv}")

    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subject",
        default=None,
        help="Single subject to analyse (default: all subjects in config)",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Disable analysis figures (config.plots.analysis_plots is ignored)",
    )
    args = parser.parse_args()

    setup_plotting_backend()
    config = ProjectConfig()

    subjects = [args.subject] if args.subject else config.analysis.subjects
    frames = []
    for subject_id in subjects:
        frames.append(analyze_subject(subject_id, run_plots=not args.no_plots))

    # ---- Combined group database (input of statistics / plotting) ----
    if frames:
        database = pd.concat(frames, ignore_index=True)
        if config.io.export_data:
            config.paths.metrics_csv.parent.mkdir(parents=True, exist_ok=True)
            database.to_csv(config.paths.metrics_csv, index=False)
            print(f"\nCombined metrics database -> {config.paths.metrics_csv}")
            print(f"Total rows: {len(database)}")
            print(f"Columns: {list(database.columns)}")


if __name__ == "__main__":
    main()