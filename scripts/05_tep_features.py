"""Main analysis pipeline for TMS-EEG data."""

from tms_eeg.analysis.context import ContextMapper
from tms_eeg.analysis.features import FeatureExtractor
from tms_eeg.analysis.group import MetricsCollector

# Set backend
from tms_eeg.config.environment import setup_plotting_backend
from tms_eeg.config.settings import ProjectConfig
from tms_eeg.io.reader import get_raw_path, load_data
from tms_eeg.visualization.gfp_plots import MFPPlotter
from tms_eeg.visualization.tep_plots import TEPPlotter

setup_plotting_backend()

'''
# To-do:

1 - Define first subject's validation based on criterias as:
    - Are MEPs different between 0, 1 and 2?
    - Are MEPs > 50 uV in 1?
    - EEG muscle artifact was succesfully removed?
    
2 - Features
    - Mean amplitude in P30
    - Peak amplitude in P30
    - AUC in P30
    - N15-P30 peak-to-peak amplitude
    - GMFP - Global measure of the response
    - LMFP - Local response
    - 

3 - Electrodes
    - All of them? or clustered?

4 - Define TEP comparisions for time, frequency and time-frequency domains:
    - Is the TEP response the same in 1 expected (context 21) and 1 unexpected (ctx 01)?
    - Is the TEP response the same in 1 after a predictable 2 (context 21) and 1 after unpredictable (ctx 11)?
    - Are the TEP response in contexts 01, 11 and 21 different? The question here is focused in what was the previous pulse
    - Are there any adaptation in 21 context over time? Compare the first 1/3 od 21 with the last 1/3 21 context
    - Are there any adaptation in 11 context over time? Compare the first 1/3 od 11 with the last 1/3 11 context


X - Export a long format table for statistics with columns as:
    - Participant ID
    - Block number
    - Trial number
    - TMS pulse intensity (0, 1 or 2)
    - Context (Following Hernández, an example of context in my case is: present pulse + past pulse = 11)
    - Number of times this context appeared
    - Predictability of this context
    - Interval to previous pulse
    - Interval to previous same intensity pulse (usefull???)
    - pre-pulse cortical state (how???)
    - features selected post pulse (what???)
    - Volunteer classified as invalid must be included in analisys and export but with a identifier column
    - add a column for included/excluded volunteer and another for reason
    - 



'''

collector = MetricsCollector()
subjects = ProjectConfig().analysis_subjects
export_enabled = ProjectConfig().io_export_data

for subject_id in subjects:
    config = ProjectConfig(subject_id=subject_id)
    epochs = load_data(config, data_type="epochs")

    # ── Renomear condições ───────────────────────────────────────────
    label_map = {k.lower(): str(v) for k, v in config.analysis_name_to_symbol.items()}
    epochs.event_id = {label_map[k.lower()]: v for k, v in epochs.event_id.items()}

    # ── Shared objects ───────────────────────────────────────────────
    extractor = FeatureExtractor(
        epochs,
        config.analysis_channels_of_interest,
        config.analysis_time_windows,
    )
    tep_plotter = TEPPlotter(config=config)
    mfp_plotter = MFPPlotter(times=epochs.times, config=config)

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

# ── Export to CSV in long format if enabled ──
database = collector.export_csv(
    output_path="data/group/database.csv",
    export_enabled=export_enabled,
)
