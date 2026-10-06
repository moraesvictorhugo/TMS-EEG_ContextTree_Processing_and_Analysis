from dataclasses import dataclass, field


@dataclass
class ProjectConfig:
    """All project's configs.
    """

    # identification
    subject_id: str = ""

    # mode (context_tree: without trial exclusion, tep: apply trial exclusion)
    mode: str = 'tep'     # context_tree  or tep

    # Input/Output
    io_export_data: bool = False
    io_save_figs: bool = False

    # Raw
    raw_downsample_freq: float = 5000.0

    # Events / triggers
    event_trigger_id: dict = field(default_factory=lambda: {'Stimulus A': 1})

    # Artfact Removing
    artifact_window: tuple = (-0.005, 0.010)
    second_artifact_window: tuple = (-0.005, 0.012)
    artifact_mode: str = 'cubic'
    artifact_anchor_window_ms: float = 1.0

    # Filters
    filter_eeg_1st_bandpass: tuple = (1, None)
    filter_eeg_2nd_bandpass: tuple = (None, 80)
    filter_emg_bandpass: tuple = (20, 500)
    filter_notch: float = 60.0
    filter_method: str = 'iir'          # 'fir' ou 'iir'
    filter_iir_order: int = 4           # TESA: ordem 4 + filtfilt
    filter_notch_width: float = 4.0     # Hz (58–62 para 60 Hz)

    # Channels
    ch_eeg_reference: str = 'average'
    ch_eog_label: str = 'EOG'
    ch_emg_label: str = 'EMG'
    ch_eeg_montage: str = 'standard_1020'

    # Epochs
    epoch_window: tuple = (-1.0, 1.0)
    epoch_eeg_baseline: tuple = (-0.3, -0.02)
    epoch_emg_baseline: tuple | None = None
    epoch_eeg_downsample_freq: float = 1000.0
    epoch_emg_downsample_freq: float = 3000.0
    epoch_eeg_detrend: int | None = 1
    epoch_emg_detrend: int | None = None

    # ICA
    ica_run: bool = True
    ica_plot_components: bool = True

    # Analysis
    analysis_subjects: list = field(default_factory=lambda: [
        "V04", "V05", "V04", "V07", "V08", "V09"])
    analysis_channels_of_interest: list = field(default_factory=lambda: [
        "FC1", "FC5", "C3", "CP1", "CP5"])
    analysis_time_windows: dict = field(default_factory=lambda: {
        "N15":  (0.012, 0.020),
        "P30":  (0.020, 0.040),
        "N45":  (0.040, 0.055),
        "P60":  (0.050, 0.070),
        "N100": (0.070, 0.150),
        "P180": (0.150, 0.200),
    })
    analysis_context_definitions: dict = field(default_factory=lambda: {
        "ctx_0":  [0],
        "ctx_2":  [2],
        "ctx_01": [0, 1],
        "ctx_11": [1, 1],
        "ctx_21": [2, 1],
    })
    # Única fonte dos símbolos/labels 8-bit — event_stimulus_to_8bit_mapping
    # e analysis_event_to_symbol são derivados em __post_init__.
    analysis_name_to_symbol: dict = field(default_factory=lambda: {
        "8bit 1": 0,
        "8bit 2": 1,
        "8bit 3": 2,
    })

    # Figures
    plot_format: str = "png"
    plot_dpi: int = 600
    plot_subfolder: str = "figures"
    plot_tep_xlim: tuple = (-0.01, 0.2)
    plot_tep_joint_times: list = field(default_factory=lambda: [
        0.015, 0.03, 0.045, 0.06, 0.1, 0.18
    ])
    plot_tep_roi_channels: list = field(default_factory=lambda: [
        'C3', 'FC1', 'CP1', 'C4', 'FC5', 'CP5'
    ])
    plot_analysis: bool = True

    def __post_init__(self):
        """Deriva os mapas restantes de ``analysis_name_to_symbol``."""
        stimulus = next(iter(self.event_trigger_id))
        labels = sorted(
            self.analysis_name_to_symbol,
            key=self.analysis_name_to_symbol.get,
        )
        self.event_stimulus_to_8bit_mapping = {stimulus: labels}
        self.analysis_event_to_symbol = {
            idx + 1: self.analysis_name_to_symbol[label]
            for idx, label in enumerate(labels)
        }
