"""Main analysis pipeline for TMS-EEG data."""

from tms_eeg.config.settings import ProjectConfig
from tms_eeg.io.reader import load_data, get_raw_path
from tms_eeg.io.writer import Writer
from tms_eeg.analysis.features import FeatureExtractor
from tms_eeg.analysis.context import ContextMapper
from tms_eeg.analysis.group import MetricsCollector
from tms_eeg.visualization.tep_plots import TEPPlotter
from tms_eeg.visualization.gfp_plots import MFPPlotter

# Set backend
from tms_eeg.config.environment import setup_plotting_backend
setup_plotting_backend()

collector = MetricsCollector()
subjects = ProjectConfig().analysis_subjects
export_enabled = ProjectConfig().io_export_data

for subject_id in subjects:
    config = ProjectConfig(subject_id=subject_id)
    epochs = load_data(config, data_type="epochs")

    # ── Renomear condições ───────────────────────────────────────────
    label_map = {"8bit 1": "0", "8bit 2": "1", "8bit 3": "2"}
    epochs.event_id = {label_map[k.lower()]: v for k, v in epochs.event_id.items()}

    # ── Shared objects ───────────────────────────────────────────────
    writer = Writer(config)
    extractor = FeatureExtractor(
        epochs,
        config.analysis_channels_of_interest,
        config.analysis_time_windows,
    )
    tep_plotter = TEPPlotter(config=config, writer=writer)
    mfp_plotter = MFPPlotter(times=epochs.times, config=config, writer=writer)

    # ================================================================ #
    #  PART 1 — ANÁLISE POR CONDIÇÃO (8Bit 1 / 2 / 3)
    # ================================================================ #

    # (Cálculos de peak-to-peak / GMFP / LMFP por condição — ver histórico
    #  do git para as versões comentadas anteriores.)

    # ================================================================ #
    #  PART 2 — ANÁLISE POR CONTEXTO (árvore de contexto)
    # ================================================================ #

    raw_path = get_raw_path(config)
    context_mapper = ContextMapper(config)
    context_epochs = context_mapper.get_context_epochs(epochs, raw_path)

    # ── Evokeds por contexto (ROI) ──────────────────────────────────
    ctx_evokeds = {
        ctx_name: ctx_ep.average().pick(config.analysis_channels_of_interest)
        for ctx_name, ctx_ep in context_epochs.items()
    }

    # ── TEP plots por contexto ───────────────────────────────────────
    tep_plotter.plot_mean_tep(evokeds=ctx_evokeds)

    # ── Contexts comparison ──────────────────────────────────────────
    tep_plotter.plot_context_comparison(context_epochs)

# ── Export to CSV if enabled ──
database = collector.export_csv(
    output_path="data/group/database.csv",
    export_enabled=export_enabled,
)
