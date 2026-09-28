# TMS-EEG Context Tree Processing and Analysis

**Created for PD-NEUROMAT research group**

A comprehensive Python package for processing and analyzing Transcranial Magnetic Stimulation combined with Electroencephalography (TMS-EEG) data using context tree methodology. This project enables researchers to investigate how brain responses to TMS pulses vary based on different stimulus contexts and sequences.

## 🧠 Research Context

This project analyzes TMS-evoked potentials (TEPs) in the context of varying stimulus sequences to understand how the brain's response to identical stimuli changes based on preceding context. The context tree analysis allows for examining neural adaptation, prediction, and information processing in response to structured stimulus sequences.

## 🚀 Key Features

- **Comprehensive Preprocessing**: TMS artifact removal, filtering, epoching, ICA decomposition
- **Context Tree Analysis**: Analyzes EEG responses based on different stimulus contexts and sequences
- **TEP Analysis**: Peak-to-peak amplitude calculation, Global/Local Mean Field Power (GMFP/LMFP) analysis
- **Advanced Visualization**: Time-evoked potentials, GFP plots, context comparison plots
- **Group Analysis**: Aggregates results across multiple subjects for statistical analysis
- **Flexible Configuration**: Dataclass-based configuration system for easy customization
- **MATLAB Integration**: Optional SOUND artifact removal algorithm support

## 📋 Table of Contents

- [Installation](#installation)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Preprocessing](#preprocessing)
  - [Analysis](#analysis)
  - [Group Analysis](#group-analysis)
- [Data Processing Pipeline](#data-processing-pipeline)
- [Dependencies](#dependencies)
- [Troubleshooting](#troubleshooting)

## 📦 Installation

### Requirements

- Python 3.11+
- Operating System: Linux, macOS, or Windows

### Using uv (Recommended)

```bash
# Clone the repository
git clone https://github.com/moraesvictorhugo/TMS-EEG_ContextTree_Processing_and_Analysis.git
cd TMS-EEG_ContextTree_Processing_and_Analysis

# Install with uv
uv sync
```

### Using pip

```bash
# Clone the repository
git clone https://github.com/moraesvictorhugo/TMS-EEG_ContextTree_Processing_and_Analysis.git
cd TMS-EEG_ContextTree_Processing_and_Analysis

# Install dependencies
pip install -r requirements.txt
```

### Optional Dependencies

For enhanced functionality, you can install optional dependencies:

```bash
# MATLAB Engine support (requires MATLAB installation)
pip install matlabengine
```

## 📁 Project Structure

```
TMS-EEG_ContextTree_Processing_and_Analysis/
├── main_preprocessing.py            # Preprocessing entry point (ends with epochs exports)
├── main_analysis.py                 # Per-subject feature extraction (condition + context)
├── main_statistics.py               # Group statistics (under construction)
├── main_plotting.py                 # Group-level plots
├── pyproject.toml                   # Project configuration / dependencies
├── uv.lock                          # Dependency lock file
│
├── src/tms_eeg/                     # TMS-EEG package
│   ├── config/                      # ALL pipeline parameters
│   │   ├── environment.py           # Plot backend setup
│   │   └── settings.py              # Dataclass configuration (paths, events, filters, ...)
│   ├── io/                          # File loading and saving
│   │   ├── reader.py                # load_data / get_raw_path
│   │   └── writer.py                # Writer + shared save_figure helper
│   ├── preprocessing/
│   │   ├── pipeline.py              # PreprocessingPipeline orchestrator (end-to-end)
│   │   ├── annotation_processor.py  # Event annotation processing (8-bit / text fallback)
│   │   ├── annotation_exporter.py   # Epoch annotations -> .mat (context-tree format)
│   │   ├── artifacts.py             # TMS artifact removal (cubic + constant pass)
│   │   ├── downsampling.py          # Signal downsampling
│   │   ├── epoching.py              # Epoch creation + EpochDropper (JSON)
│   │   ├── filtering.py             # Bandpass + notch filters
│   │   └── ica.py                   # Independent Component Analysis
│   ├── analysis/
│   │   ├── context.py               # Context tree analysis (ContextMapper)
│   │   ├── features.py              # Feature extraction (P2P, GMFP/LMFP)
│   │   ├── group.py                 # MetricsCollector (tidy long format)
│   │   └── labels.py                # Event-id normalisation
│   └── visualization/
│       ├── tep_plots.py             # TMS-evoked potential plots
│       ├── gfp_plots.py             # GMFP / LMFP curves
│       └── group_plots.py           # Group-level plots
│
├── tests/
│   └── smoke_test.py                # Lightweight tests (no pytest required)
├── utils/                           # Standalone one-off utilities
├── data/
│   ├── raw/                         # Original .bdf files + sequence .txt
│   ├── processed/                   # Exported epochs + per-subject QC JSONs
│   └── group/                       # Metrics CSVs (from main_analysis.py)
└── README.md                        # This file
```

### Key Modules Overview

- **`src/tms_eeg/config/settings.py`**: the single configuration file — all paths and parameters live here.
- **`src/tms_eeg/preprocessing/pipeline.py`**: reproduces the full preprocessing in a single, headless-friendly orchestrator.
- **`src/tms_eeg/analysis/`**: core analysis modules for TEP extraction, context tree analysis, and tidy metrics collection.
- **`src/tms_eeg/visualization/`**: plotting functions for TEPs, GMFP/LMFP, and group-level comparisons.


## ⚡ Quick Start

### 1. Preprocessing

```bash
# Single subject with interactive QC plots (requires a display)
python main_preprocessing.py --subject V04 --qc

# Headless (applies the QC decisions recorded in data/processed/*.json)
python main_preprocessing.py --subject V04
```

This:
- Loads the raw `.bdf`, processes the annotations (8-bit or text file) and creates EEG/EMG epochs
- Removes the TMS artifact (cubic spline), drops+interpolates bad channels, ICA, SOUND,  SSP-SIR
- Filters (bandpass + notch), applies the recorded epoch drops and finishes by exporting
  the `.fif` variants (`processed_full`, `processed_pre_and_post`, `processed_post_only`,
  `emg_processed`) and the context-tree `.mat` files under `data/processed/<subject>/`.

### 2. Analysis (per subject)

```bash
# All subjects in config.analysis.subjects
python main_analysis.py

# Single subject / headless
python main_analysis.py --subject V05 --no-plots
```

This extracts the condition- and context-level features (peak-to-peak amplitudes,
GMFP/LMFP peaks), writes `data/group/<subject>_metrics.csv` and a combined
`data/group/database.csv`.

### 3. Group plotting

```bash
python main_plotting.py [--metrics data/group/database.csv] [--output-dir results/group]
```

Loads the metrics database and renders group box/strip plots and the P30 amplitude
summary (requires `main_analysis.py` to have been run first).

### 4. Group statistics (future)

```bash
python main_statistics.py   # stub — will run group-level statistics
```

## ⚙️ Configuration

All parameters live in a single file: `src/tms_eeg/config/settings.py`.
`ProjectConfig` bundles the sections (`paths`, `io`, `events`, `filters`, `sound`,
`channels`, `epochs`, `ica`, `analysis`, `plots`).

```python
from tms_eeg.config.settings import ProjectConfig

# Create configuration for a specific subject
config = ProjectConfig(subject_id="V07")

# Access configuration sections
print(config.analysis.subjects)  # List of subjects (temporary selection)
print(config.analysis.channels_of_interest)  # EEG channels to analyze
print(config.analysis.time_windows)  # Time windows for analysis
```

### Customizing Analysis Parameters

```python
# Modify configuration
config.analysis.subjects = ["V01", "V02", "V03"]  # Change subject list
config.analysis.channels_of_interest = ["C3", "Cz", "C4"]  # Change channels
config.analysis.time_windows = {
    "N15": (0.012, 0.020),
    "P30": (0.020, 0.040),
    # Add custom time windows
}
```

### Context Tree Configuration

The context tree analysis can be customized by modifying the context definitions:

```python
config.analysis.context_definitions = {
    "ctx_0": [0],           # Current stimulus = 0, any past
    "ctx_2": [2],           # Current stimulus = 2, any past
    "ctx_01": [0, 1],       # Previous = 0, current = 1
    "ctx_11": [1, 1],       # Previous = 1, current = 1
    "ctx_21": [2, 1],       # Previous = 2, current = 1
}
```

## 📊 Usage

### Preprocessing

The preprocessing pipeline (`main_preprocessing.py`) handles:

1. **Data Loading**: Load raw EEG data with proper channel configuration
2. **Annotation Processing**: Replace `Stimulus A` with the 8-bit / text conditions
3. **Epoching**: Create epochs around TMS pulses (EEG + EMG)
4. **Artifact Removal**: Cubic-spline interpolation of the TMS artifact
5. **ICA / SOUND / SSP-SIR**: Remove remaining (ocular, decay) artifacts
6. **Filtering**: Bandpass + notch
7. **Epoch drops**: apply the per-subject exclusions recorded during QC
8. **Exports**: `.fif` (full / cropped / EMG) and context-tree `.mat`

```python
# Example: Custom preprocessing
from tms_eeg.config.settings import ProjectConfig
from tms_eeg.preprocessing.epoching import EEGEpocher
from tms_eeg.preprocessing.artifacts import ArtifactRemover

config = ProjectConfig(subject_id="V07")
# ... preprocessing steps as defined in src/tms_eeg/preprocessing/pipeline.py
```

### Analysis

The analysis pipeline (`main_analysis.py`) performs:

1. **TEP Extraction**: Extract time-evoked potentials for different conditions
2. **Feature Calculation**: Compute peak-to-peak amplitudes and MFP measures
3. **Context Analysis**: Analyze responses based on stimulus context
4. **Visualization**: Generate comprehensive plots

```python
# Example: Custom analysis
from tms_eeg.config.settings import ProjectConfig
from tms_eeg.analysis.features import FeatureExtractor
from tms_eeg.analysis.context import ContextMapper

config = ProjectConfig(subject_id="V07")
# ... analysis steps as defined in main_analysis.py
```

### Group Plotting

The group plotting (`main_plotting.py`) loads the tidy metrics database and
renders box/strip plots per condition/context and the P30 amplitude summary.

### Group Statistics (future)

The group statistics (`main_statistics.py`) will read the same metrics database
and run group-level statistical tests across conditions and contexts.

## 📈 Visualization Capabilities

The project provides comprehensive visualization tools for TMS-EEG analysis:

### TMS-Evoked Potentials (TEPs)
- **Time Course Plots**: Visualize TEP waveforms across different conditions
- **Topographic Maps**: Spatial distribution of brain responses over time
- **ROI Analysis**: Region-of-interest specific responses
- **Joint Plots**: Combined time course and topographic visualization

### Global Field Power (GFP) Analysis
- **GMFP/LMFP Plots**: Global and Local Mean Field Power time courses
- **Peak Detection**: Automatic identification of significant peaks
- **Context Comparisons**: Side-by-side comparison of different stimulus contexts
- **Overlay Plots**: Multiple conditions on the same plot for easy comparison

### Context Tree Analysis Visualization
- **Context Comparison Plots**: Compare responses across different stimulus contexts
- **Temporal Evolution**: Time-based analysis of context effects
- **Branch Analysis**: Specific comparisons between context branches
- **Statistical Overlays**: Confidence intervals and significance markers

### EMG Data

EMG epochs are created and exported during preprocessing (`emg_processed`),
together with the EEG epochs, for downstream (separate) analysis.

### Group-Level Visualizations
- **Statistical Comparisons**: Group means with error bars and significance testing
- **Effect Size Plots**: Visualization of effect sizes across conditions
- **Correlation Analysis**: Relationships between different measures
- **Heatmaps**: Matrix visualization of group-level statistics

## 🔄 Data Processing Pipeline

### Preprocessing Workflow

```
Raw EEG (.bdf)
    ↓
Channel configuration & montage
    ↓
Annotation processing (8-bit / text file)
    ↓
Epoch creation (-0.8 s to +0.8 s, EEG + EMG)
    ↓
TMS artifact removal (cubic spline interpolation)
    ↓
Bad channel drop + interpolation (QC JSON)
    ↓
Detrend + baseline correction
    ↓
ICA (components from QC JSON)
    ↓
SOUND + average reference + SSP-SIR
    ↓
Downsampling (EEG 1000 Hz / EMG 3000 Hz)
    ↓
Constant artifact pass + filters (bandpass + notch)
    ↓
Epoch drops (2nd run, QC JSON)
    ↓
Crop + Export (.fif variants + context-tree .mat)
```

### Analysis Workflow

```
Processed Epochs
    ↓
Condition-based Analysis
    ├── TEP Extraction
    ├── Peak-to-Peak Calculation
    ├── GMFP/LMFP Computation
    └── Visualization
    ↓
Context Tree Analysis
    ├── Context Mapping
    ├── Context-based TEPs
    ├── Feature Extraction
    └── Context Comparisons
    ↓
Group Aggregation
    ├── Feature Collection
    ├── Statistical Analysis
    └── Group Visualizations
```

## 📚 Dependencies

### Core Dependencies

- **MNE-Python** (≥1.11.0): EEG data processing and analysis
- **NumPy** (≥2.4.2): Numerical computing
- **Pandas** (≥3.0.1): Data manipulation and analysis
- **Matplotlib** (≥3.10.8): Plotting and visualization
- **SciPy** (≥1.17.1): Scientific computing
- **scikit-learn** (≥1.8.0): Machine learning utilities

### Optional Dependencies

- **MATLAB Engine** (≥9.10.0): For SOUND artifact removal algorithm
- **PyQt5** (≥5.15.11): GUI components
- **Seaborn** (≥0.13.2): Enhanced statistical visualization

### Development Dependencies

- **Jupyter** (≥1.1.1): Interactive notebooks
- **ipykernel** (≥7.2.0): Jupyter kernel

## 🔧 Troubleshooting

### Common Issues

#### 1. MATLAB Engine Not Found

**Problem**: `ModuleNotFoundError: No module named 'matlab'`

**Solution**: 
- Install MATLAB Engine API for Python
- Or disable MATLAB-dependent features in configuration

#### 2. Memory Issues with Large Datasets

**Problem**: Out of memory errors during processing

**Solution**:
- Process subjects individually
- Reduce epoch length or sampling rate
- Use more efficient data types

#### 3. ICA Convergence Issues

**Problem**: ICA decomposition fails to converge

**Solution**:
- Check data quality and preprocessing
- Adjust ICA parameters in configuration
- Manually inspect and remove bad components

#### 4. Missing Dependencies

**Problem**: Import errors for required packages

**Solution**:
```bash
# Reinstall dependencies
pip install -r requirements.txt
# or
uv sync
```

### Getting Help

If you encounter issues not covered here:

1. Check the project's GitHub Issues page
2. Review the code comments and docstrings
3. Contact the maintainers via email
4. Create a detailed issue report with:
   - Python version
   - Operating system
   - Error message
   - Steps to reproduce

### Known Limitations

- Currently optimized for specific EEG montages and channel configurations
- MATLAB integration requires local MATLAB installation
- Large datasets may require significant memory and processing time
- Some analysis features are still under development

## 📊 Project Outputs

The project generates comprehensive outputs for TMS-EEG analysis:

### Data Files
- **Processed Epochs**: Cleaned and filtered EEG epochs ready for analysis
- **Evoked Responses**: Average TEPs for different conditions and contexts
- **Feature Data**: Peak-to-peak amplitudes, GMFP/LMFP measures, and other extracted features
- **Group Statistics**: Aggregated results across subjects with statistical analysis

### Visualization Outputs
- **TEP Plots**: Time-evoked potential waveforms with topographic maps
- **GFP Analysis**: Global and Local Mean Field Power plots
- **Context Comparisons**: Visualizations of context-dependent responses
- **Group Plots**: Statistical comparisons and effect size visualizations
- **Quality Control**: Preprocessing validation plots and artifact detection

### Analysis Reports
- **Individual Subject Reports**: Comprehensive analysis for each subject
- **Group Analysis Reports**: Statistical summaries across the cohort
- **Context Analysis Reports**: Detailed context tree analysis results
- **Feature Extraction Reports**: Summary of all extracted neurophysiological measures

## 🎯 Research Applications

This project is designed for researchers studying:

- **Neural Adaptation**: How brain responses change with repeated stimulation
- **Context Processing**: How preceding stimuli influence current responses
- **Information Processing**: Brain's ability to predict and process structured sequences
- **Neurological Disorders**: Applications in Parkinson's disease and other neurological conditions
- **Brain Connectivity**: Understanding functional connectivity through TMS-EEG responses

## 🔄 Workflow Integration

The project supports integration into larger research workflows:

1. **Data Import**: Compatible with standard EEG file formats
2. **Batch Processing**: Automated processing of multiple subjects
3. **Custom Analysis**: Flexible configuration for different experimental designs
4. **Export Options**: Multiple output formats for downstream analysis
5. **Reproducibility**: Configuration-based analysis for consistent results

## 📞 Contact

For questions, suggestions, or collaboration opportunities:

- **Project Maintainer**: Victor Hugo Moraes
- **Research Group**: PD-NEUROMAT
- **Email**: moraes.vh@usp.br
- **Institution**: University of São Paulo

---

**Note**: This software is intended for research purposes. Users are responsible for ensuring compliance with their institution's data handling and analysis policies.