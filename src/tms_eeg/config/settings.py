from dataclasses import dataclass, field


@dataclass
class IOConfig:
    export_data: bool = False
    save_figs: bool = False

@dataclass
class EventConfig:
    event_ids: dict = field(default_factory=lambda: {
        'stimulus_0': 0,
        'stimulus_1': 1,
        'stimulus_2': 2,
    })
    trigger_id: dict = field(default_factory=lambda: {'Stimulus A': 1})
    trigger_8bits: tuple = (1, 2, 3)
    stimulus_to_8bit_mapping: dict = field(default_factory=lambda: {
        'Stimulus A': ['8bit 1', '8bit 2', '8bit 3'],
    })

@dataclass
class ArtifactConfig:
    window_removal_artifact: tuple = (-0.005, 0.010)
    mode_removal_artifact: str = 'cubic'
    anchor_window_ms: float = 10.0

@dataclass
class FilterConfig:
    eeg_bandpass: tuple = (1, 80)
    emg_bandpass: tuple = (20, 500)
    notch: tuple = (60, 120, 180, 240, 300)

@dataclass
class ChannelConfig:
    eeg_reference: str = 'average'
    eog_label: str = 'EOG'
    emg_label: str = 'EMG'
    eeg_montage: str = 'standard_1020'

@dataclass
class EpochConfig:
    window: tuple = (-1.5, 1.5)
    baseline: tuple = (-0.3, -0.01)
    downsample_freq: float = 1000.0
    emg_downsample_freq: float = 3000.0

@dataclass
class ICAConfig:
    run_ica: bool = True
    plot_components: bool = True

@dataclass
class AnalysisConfig:
    subjects: list = field(default_factory=lambda: [
        "V04", "V05", "V04", "V07", "V08", "V09"])
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
    event_to_symbol: dict = field(default_factory=lambda: {
        1: 0,
        2: 1,
        3: 2,
    })
    name_to_symbol: dict = field(default_factory=lambda: {
        "8bit1": 0,
        "8bit2": 1,
        "8bit3": 2,
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
    io: IOConfig = field(default_factory=IOConfig)
    events: EventConfig = field(default_factory=EventConfig)
    artifact: ArtifactConfig = field(default_factory=ArtifactConfig)
    filters: FilterConfig = field(default_factory=FilterConfig)
    channels: ChannelConfig = field(default_factory=ChannelConfig)
    epochs: EpochConfig = field(default_factory=EpochConfig)
    ica: ICAConfig = field(default_factory=ICAConfig)
    analysis: AnalysisConfig = field(default_factory=AnalysisConfig)
    plots: PlotConfig = field(default_factory=PlotConfig)
    subject_id: str = ""
    session: str = "_400pulses_TMS-EEG_ContextTree"
