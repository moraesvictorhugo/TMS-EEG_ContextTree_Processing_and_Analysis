"""Central configuration for the TMS-EEG pipeline.

Every pipeline parameter lives here. Edit this file to change how the
preprocessing / analysis / plotting scripts behave.
"""

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
#  REPOSITORY LAYOUT
# --------------------------------------------------------------------------- #

BASE_DIR = Path(__file__).resolve().parents[3]  # project root


@dataclass
class PathsConfig:
    """Input/output directories and per-subject QC records."""

    data_dir: Path = field(default_factory=lambda: BASE_DIR / "data")
    raw_dir: Path = field(default_factory=lambda: BASE_DIR / "data" / "raw")
    processed_dir: Path = field(default_factory=lambda: BASE_DIR / "data" / "processed")
    group_dir: Path = field(default_factory=lambda: BASE_DIR / "data" / "group")

    # Per-subject QC / exclusion records (JSON files keyed by subject id).
    bad_channels_file: Path = field(
        default_factory=lambda: BASE_DIR / "data" / "processed" / "channels_interpolated.json"
    )
    ica_components_file: Path = field(
        default_factory=lambda: BASE_DIR / "data" / "processed" / "components_rem_ica.json"
    )
    epochs_removed_1st_run_file: Path = field(
        default_factory=lambda: BASE_DIR / "data" / "processed" / "idx_epochs_rem_1st_run.json"
    )
    epochs_removed_2nd_run_file: Path = field(
        default_factory=lambda: BASE_DIR / "data" / "processed" / "idx_epochs_rem_2nd_run.json"
    )

    # Group metrics database produced by main_analysis.py.
    metrics_csv: Path = field(default_factory=lambda: BASE_DIR / "data" / "group" / "database.csv")

    # ------------------------------------------------------------------ #
    #  Helpers
    # ------------------------------------------------------------------ #
    def subject_raw_dir(self, subject_id: str) -> Path:
        """Directory holding the raw .bdf for a subject (data/raw/<id>_data)."""
        return self.raw_dir / f"{subject_id}_data"

    def subject_processed_dir(self, subject_id: str, *subpaths: str) -> Path:
        """Processed directory for a subject, e.g. data/processed/<id>/processed_full."""
        path = self.processed_dir / subject_id
        for sub in subpaths:
            path = path / sub
        return path


@dataclass
class IOConfig:
    """Export behaviour shared by every script."""

    export_data: bool = True  # write .fif / .mat / CSV outputs
    save_figs: bool = False  # additionally save every figure to disk

@dataclass
class EventConfig:
    """Raw-annotation handling and symbol mappings."""

    # Annotation accepted as the TMS stimulus trigger.
    trigger_id: dict = field(default_factory=lambda: {'Stimulus A': 1})
    # Replacement labels for the trigger annotation (used by the 8-bit path).
    stimulus_to_8bit_mapping: dict = field(default_factory=lambda: {
        'Stimulus A': ['8bit 1', '8bit 2', '8bit 3'],
    })
    # Event code -> context-tree symbol (0/1/2). Used for context grouping.
    event_to_symbol: dict = field(default_factory=lambda: {1: 0, 2: 1, 3: 2})
    # Annotation label (lowercase, no spaces) -> symbol. Used for .mat export.
    name_to_symbol: dict = field(default_factory=lambda: {
        "8bit1": 0, "8bit2": 1, "8bit3": 2,
    })
    # Epoch condition labels rewritten before feature extraction (analysis).
    label_rename: dict = field(default_factory=lambda: {
        "8bit 1": "0", "8bit 2": "1", "8bit 3": "2",
    })

@dataclass
class ArtifactConfig:
    window_removal_artifact: tuple = (-0.002, 0.015)
    mode_removal_artifact: str = 'cubic'
    anchor_window_ms: float = 5.0
    # Second artifact pass (constant interpolation after SSP-SIR).
    fix_stim_artifact_baseline: tuple = (-0.005, -0.002)

@dataclass
class FilterConfig:
    eeg_bandpass: tuple = (None, 80)
    emg_bandpass: tuple = (20, 500)
    notch_band: tuple = (58, 62)  # single notch around the line-noise peak
    notch_harmonics: int = 3

@dataclass
class SoundConfig:
    """SOUND (artifact suppression) and SSP-SIR."""

    lambda_val: float = 0.1
    iter_num: int = 5
    run_sspsir: bool = True

@dataclass
class ChannelConfig:
    eeg_reference: str = 'average'
    eog_label: str = 'EOG'
    emg_label: str = 'EMG'
    eeg_montage: str = 'standard_1020'
    drop_channels: list = field(default_factory=lambda: ["TP9", "TP10", "O1", "O2", "Iz"])

@dataclass
class EpochConfig:
    window: tuple = (-0.8, 0.8)
    baseline: tuple = (-0.2, -0.01)
    downsample_freq: float = 1000.0
    emg_downsample_freq: float = 3000.0
    crop_pre_and_post: tuple = (-0.05, 0.4)  # 'processed_pre_and_post'
    crop_post_only: tuple = (0.015, 0.2)  # 'processed_post_only'
    mat_window: tuple = (0.015, 0.450)  # .mat export window (seconds)
    drop_first_run: bool = False  # apply data/.../idx_epochs_rem_1st_run.json
    drop_second_run: bool = True  # apply data/.../idx_epochs_rem_2nd_run.json

@dataclass
class ICAConfig:
    run_ica: bool = True
    plot_components: bool = True
    n_components: int = 20
    random_state: int = 97
    method: str = "fastica"
    fallback_components: list = field(default_factory=lambda: [0])

@dataclass
class AnalysisConfig:
    """Feature-extraction settings (main_analysis.py)."""

    # TEMPORARY selection of volunteers for the current analysis batch.
    subjects: list = field(default_factory=lambda: [
        "V04", "V05", "V07", "V08", "V09"])
    channels_of_interest: list = field(default_factory=lambda: [
        "FC1", "FC5", "C3", "CP1", "CP5"])
    time_windows: dict = field(default_factory=lambda: {
        "N15":  (0.012, 0.020),
        "P30":  (0.020, 0.040),
        "N45":  (0.040, 0.055),
        "P60":  (0.050, 0.070),
        "N100": (0.070, 0.150),
        "P180": (0.150, 0.200),
    })
    context_definitions: dict = field(default_factory=lambda: {
        "ctx_0":  [0],
        "ctx_2":  [2],
        "ctx_01": [0, 1],
        "ctx_11": [1, 1],
        "ctx_21": [2, 1],
    })

@dataclass
class PlotConfig:
    figure_format: str = "png"
    figure_dpi: int = 600
    figure_subfolder: str = "figures"
    tep_xlim: tuple = (-0.01, 0.2)
    tep_topo_times: list = field(default_factory=lambda: [
        0.005, 0.01, 0.02, 0.03, 0.04, 0.05,
        0.06, 0.07, 0.08, 0.09, 0.1
    ])
    tep_joint_times: list = field(default_factory=lambda: [
        0.015, 0.03, 0.045, 0.06, 0.1, 0.18
    ])
    tep_roi_channels: list = field(default_factory=lambda: [
        'C3', 'FC1', 'CP1', 'C4', 'FC5', 'CP5'
    ])
    emg_xlim: tuple = (-0.01, 0.08)
    analysis_plots: bool = True

@dataclass
class ProjectConfig:
    """Top-level configuration bundle for a single subject.

    Parameters
    ----------
    subject_id : str
        Overrides the subject processed by the current script run.
    """

    paths: PathsConfig = field(default_factory=PathsConfig)
    io: IOConfig = field(default_factory=IOConfig)
    events: EventConfig = field(default_factory=EventConfig)
    artifact: ArtifactConfig = field(default_factory=ArtifactConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    sound: SoundConfig = field(default_factory=SoundConfig)
    channels: ChannelConfig = field(default_factory=ChannelConfig)
    epochs: EpochConfig = field(default_factory=EpochConfig)
    ica: ICAConfig = field(default_factory=ICAConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    plots: PlotConfig = field(default_factory=PlotConfig)
    subject_id: str = "V04"
