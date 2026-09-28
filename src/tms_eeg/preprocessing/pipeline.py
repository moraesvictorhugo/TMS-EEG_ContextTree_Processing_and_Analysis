"""End-to-end single-subject preprocessing pipeline.

Reproduces the steps previously scattered in ``main_preprocessing.py``:

    load raw -> annotations -> epoch (EEG + EMG) -> baseline
    -> TMS artifact removal (cubic) -> drop fixed channels
    -> interpolate bads -> (optional 1st-run epoch drops) -> detrend
    -> ICA -> baseline -> SOUND -> average reference -> SSP-SIR
    -> downsample -> constant artifact pass -> filters
    -> (optional 2nd-run epoch drops) -> crop -> export
    (.fif full / pre+post / post-only, EMG, and context-tree .mat).
"""

from __future__ import annotations

import json
from pathlib import Path

import mne
from pytep import apply_sspsir, apply_sound
from scipy.signal import detrend

from tms_eeg.config.settings import ProjectConfig
from tms_eeg.io.reader import load_data
from tms_eeg.io.writer import Writer
from tms_eeg.preprocessing.annotation_exporter import EpochAnnotationExporter
from tms_eeg.preprocessing.annotation_processor import AnnotationProcessor
from tms_eeg.preprocessing.artifacts import ArtifactRemover
from tms_eeg.preprocessing.downsampling import Downsampler
from tms_eeg.preprocessing.epoching import EEGEpocher, EpochDropper
from tms_eeg.preprocessing.filtering import Filter
from tms_eeg.preprocessing.ica import EEGICA
from tms_eeg.visualization.tep_plots import TEPPlotter

# Channels used for the TEP QC previews during preprocessing.
QC_PICKS = ["FC1", "FC5", "C3", "C4", "CP1", "CP5"]


def _read_json(path: Path) -> dict:
    """Load a JSON dict (per-subject records). Missing files yield ``{}``."""
    if path is None or not Path(path).exists():
        return {}
    with open(path) as f:
        return json.load(f)


class PreprocessingPipeline:
    """Run the full preprocessing for one subject and export the results."""

    def __init__(self, config: ProjectConfig, qc: bool = False):
        self.config = config
        self.qc = qc  # show interactive QC plots when True

    # ------------------------------------------------------------------ #
    #  Main entry point
    # ------------------------------------------------------------------ #
    def run(self) -> None:
        c = self.config
        plotter = TEPPlotter(c)

        # ---------- Load & annotate ---------- #
        raw = load_data(c)
        raw.set_channel_types({
            c.channels.eog_label: "eog", c.channels.emg_label: "emg"})
        raw.set_montage(c.channels.eeg_montage)
        raw = AnnotationProcessor(c).process_annotations(raw)

        # ---------- Epoch (EEG + EMG) ---------- #
        epocher = EEGEpocher(c)
        epochs_eeg = epocher.create_epochs(raw)
        epochs_emg = epocher.create_epochs(raw.copy().pick("emg"))
        epochs_eeg.apply_baseline(baseline=c.epochs.baseline)

        if self.qc:
            plotter.plot_evoked_by_symbol(
                epochs_eeg, picks=QC_PICKS, xlim=(-0.01, 0.015), ylim=(-30, 30))

        # ---------- TMS artifact removal (cubic) ---------- #
        remover = ArtifactRemover(c)
        epochs_eeg = remover.remove_tms_artifact(epochs_eeg, mode="cubic")

        if self.qc:
            plotter.plot_evoked_by_symbol(
                epochs_eeg, picks=QC_PICKS, xlim=(-0.01, 0.2), ylim=(-30, 30))

        # ---------- Channels: drop + interpolate ---------- #
        epochs_eeg.drop_channels(c.channels.drop_channels)

        if self.qc:
            # Interactive: mark bad channels / bad epochs.
            epochs_eeg.plot()
        else:
            # Headless: apply the channels recorded during the QC pass.
            epochs_eeg.info["bads"] = _read_json(
                c.paths.bad_channels_file).get(c.subject_id, [])

        epochs_eeg.interpolate_bads(reset_bads=True)

        # ---------- Optional 1st-run epoch drops ---------- #
        if c.epochs.drop_first_run:
            epochs_eeg = EpochDropper(c).drop_from_json(
                epochs_eeg, c.paths.epochs_removed_1st_run_file)

        # ---------- Detrend & baseline ---------- #
        epochs_eeg.apply_function(lambda x: detrend(x, type="linear"), picks="all")
        epochs_eeg.apply_baseline(baseline=c.epochs.baseline)

        # ---------- ICA ---------- #
        ica = EEGICA(c)
        ica.fit_ica(epochs_eeg)
        if self.qc:
            ica.plot_components(epochs_eeg)

        ica_components = _read_json(c.paths.ica_components_file).get(
            c.subject_id, c.ica.fallback_components)
        epochs_eeg = ica.apply_ica(epochs_eeg, components_to_remove=ica_components)
        epochs_eeg.apply_baseline(baseline=c.epochs.baseline)

        # ---------- SOUND + SSP-SIR ---------- #
        epochs_eeg = apply_sound(
            epochs_eeg, iter_num=c.sound.iter_num, lambda_val=c.sound.lambda_val)
        epochs_eeg.set_eeg_reference(c.channels.eeg_reference)
        if c.sound.run_sspsir:
            epochs_eeg = apply_sspsir(epochs_eeg)

        # ---------- Downsample ---------- #
        downsampler = Downsampler(c)
        epochs_eeg = downsampler.downsample(epochs_eeg)
        epochs_emg = downsampler.downsample_emg_channels(epochs_emg)

        # ---------- Constant artifact pass (post-SSP-SIR) ---------- #
        epochs_eeg = remover.fix_stim_artifact(epochs_eeg)

        # ---------- Filters ---------- #
        filt = Filter(c)
        epochs_eeg_filtered = filt.notch_filter(filt.bp_filter(epochs_eeg, "eeg"))
        epochs_emg_filtered = filt.notch_filter(filt.bp_filter(epochs_emg, "emg"))

        if self.qc:
            epochs_eeg_filtered.plot()

        # ---------- Optional 2nd-run epoch drops ---------- #
        if c.epochs.drop_second_run:
            epochs_eeg_filtered = EpochDropper(c).drop_from_json(
                epochs_eeg_filtered, c.paths.epochs_removed_2nd_run_file)

        if self.qc:
            plotter.plot_evoked_by_symbol(
                epochs_eeg_filtered, picks=QC_PICKS, xlim=(-0.05, 0.2), ylim=(-10, 10))

        # ---------- Crop variants for export ---------- #
        epochs_pre_post = epochs_eeg_filtered.copy().crop(
            tmin=c.epochs.crop_pre_and_post[0], tmax=c.epochs.crop_pre_and_post[1])
        epochs_post = epochs_eeg_filtered.copy().crop(
            tmin=c.epochs.crop_post_only[0], tmax=c.epochs.crop_post_only[1])

        # ---------- Export ---------- #
        self._export(epochs_eeg_filtered, epochs_pre_post, epochs_post, epochs_emg_filtered)
        print(f"\n=== Preprocessing finished for {c.subject_id} ===")

    # ------------------------------------------------------------------ #
    #  Export helpers
    # ------------------------------------------------------------------ #
    def _export(
        self,
        epochs_full: mne.Epochs,
        epochs_pre_post: mne.Epochs,
        epochs_post: mne.Epochs,
        epochs_emg: mne.Epochs,
    ) -> None:
        c = self.config
        writer = Writer(c)
        exporter = EpochAnnotationExporter(c)

        # Full EEG epochs.
        writer.save_epochs(epochs_full, "processed_full")

        # Cropped variants: .mat (context-tree) + .fif.
        for epochs, subfolder in (
            (epochs_pre_post, "processed_pre_and_post"),
            (epochs_post, "processed_post_only"),
        ):
            annotations = exporter.extract_annotations(epochs)
            symbols = exporter.map_annotations_to_symbols(annotations)
            exporter.export_to_mat(writer, epochs, symbols, subfolder=subfolder)
            writer.save_epochs(epochs, subfolder)

        # EMG epochs.
        writer.save_emg_epochs(epochs_emg, "emg_processed")