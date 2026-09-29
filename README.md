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

### VS Code

Select the project interpreter once per machine: run **Python: Select Interpreter** and choose the `.venv` folder in the workspace. The `.venv` environment is auto-detected on both Linux and Windows.

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
MEPs-and-TEPs-with-Context-Tree/
├── data/                           # Data storage
│   ├── raw/                        # Original, unmodified EEG files
│   └── processed/                  # Output from preprocessing/processing steps
│
├── src/                            # Main source code
│   └── tms_eeg/                    # TMS-EEG package
│       ├── __init__.py
│       ├── config/                 # Configuration files
│       │   ├── __init__.py
│       │   ├── environment.py      # Environment setup
│       │   └── settings.py         # Configuration dataclasses
│       ├── io/                     # File loading and saving
│       │   ├── __init__.py
│       │   ├── reader.py           # Data loading utilities
│       │   └── writer.py           # Data saving utilities
│       ├── preprocessing/          # Signal processing pipeline
│       │   ├── __init__.py
│       │   ├── annotation_processor.py  # Event annotation processing
│       │   ├── artifacts.py        # TMS artifact removal
│       │   ├── downsampling.py     # Signal downsampling
│       │   ├── epoching.py         # Epoch creation
│       │   ├── filtering.py        # Signal filtering
│       │   └── ica.py              # Independent Component Analysis
│       ├── analysis/               # Analysis modules
│       │   ├── __init__.py
│       │   ├── context.py          # Context tree analysis
│       │   ├── features.py         # Feature extraction
│       │   └── group.py            # Group-level analysis
│       └── visualization/          # Plotting and visualization
│           ├── __init__.py
│           ├── emg_plots.py        # EMG signal plots
│           ├── gfp_plots.py        # Global Field Power plots
│           ├── group_plots.py      # Group analysis plots
│           └── tep_plots.py        # TMS-evoked potential plots
│
├── notebooks/                      # Jupyter notebooks for exploration
│   └── 01_data_exploration.ipynb
│
├── results/                        # Analysis results
│   └── group/                      # Group-level results
│
├── trigger/                        # External dependencies (PyTEP-SOUND-SSP-SIR)
│
├── main_preprocessing.py           # Preprocessing pipeline entry point
├── main_analysis.py               # Analysis pipeline entry point
├── main_group.py                  # Group analysis entry point
├── pyproject.toml                 # Project configuration
├── uv.lock                        # Dependency lock file
├── estrutura.txt                  # Project structure documentation
└── README.md                      # This file
```

### Key Modules Overview

- **`src/tms_eeg/config/`**: Configuration management using dataclasses for flexible parameter control
- **`src/tms_eeg/io/`**: Data input/output operations with support for various EEG file formats
- **`src/tms_eeg/preprocessing/`**: Complete preprocessing pipeline including artifact removal, filtering, and ICA
- **`src/tms_eeg/analysis/`**: Core analysis modules for TEP extraction, context tree analysis, and group statistics
- **`src/tms_eeg/visualization/`**: Comprehensive plotting functions for TEPs, GFP, and context comparisons
  - **`tep_plots.py`**: TMS-evoked potential visualization with topographic maps and time courses
  - **`gfp_plots.py`**: Global and Local Mean Field Power analysis and plotting
  - **`emg_plots.py`**: EMG signal visualization for muscle activity monitoring
  - **`group_plots.py`**: Group-level statistical comparisons and visualizations

## ⚡ Quick Start

### 1. Basic Preprocessing

```python
# Run preprocessing pipeline
python main_preprocessing.py
```

This will:
- Load raw EEG data
- Remove TMS artifacts
- Apply filtering and epoching
- Perform ICA decomposition
- Save processed data

### 2. Analysis Pipeline

```python
# Run analysis pipeline
python main_analysis.py
```

This will:
- Load preprocessed epochs
- Extract TEP features (peak-to-peak amplitudes)
- Compute GMFP/LMFP measures
- Perform context tree analysis
- Generate visualizations

### 3. Group Analysis

```python
# Run group-level analysis
python main_group.py
```

This will aggregate results across all subjects and generate group-level statistics.

## ⚙️ Configuration

The project uses a flexible configuration system based on Python dataclasses. Configuration is managed through `src/tms_eeg/config/settings.py`.

### Basic Configuration

```python
from tms_eeg.config.settings import ProjectConfig

# Create configuration for a specific subject
config = ProjectConfig(subject_id="V07")

# Access configuration sections
print(config.analysis_subjects)  # List of subjects
print(config.analysis_channels_of_interest)  # EEG channels to analyze
print(config.analysis_time_windows)  # Time windows for analysis
```

### Customizing Analysis Parameters

```python
# Modify configuration
config.analysis_subjects = ["V01", "V02", "V03"]  # Change subject list
config.analysis_channels_of_interest = ["C3", "Cz", "C4"]  # Change channels
config.analysis_time_windows = {
    "N15": (0.012, 0.020),
    "P30": (0.020, 0.040),
    # Add custom time windows
}
```

### Context Tree Configuration

The context tree analysis can be customized by modifying the context definitions:

```python
config.analysis_context_definitions = {
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
2. **Artifact Removal**: Remove TMS-induced artifacts using baseline correction
3. **Filtering**: Apply bandpass and notch filters
4. **Epoching**: Create epochs around TMS pulses
5. **ICA Decomposition**: Remove remaining artifacts using Independent Component Analysis
6. **Downsampling**: Reduce sampling rate for analysis
7. **Visualization**: Generate quality control plots

```python
# Example: Custom preprocessing
from tms_eeg.config.settings import ProjectConfig
from tms_eeg.preprocessing.epoching import EEGEpocher
from tms_eeg.preprocessing.artifacts import ArtifactRemover

config = ProjectConfig(subject_id="V07")
# ... preprocessing steps as defined in main_preprocessing.py
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

### Group Analysis

The group analysis (`main_group.py`) aggregates results across subjects:

1. **Data Aggregation**: Combine results from all subjects
2. **Statistical Analysis**: Compute group-level statistics
3. **Visualization**: Generate group comparison plots

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

### EMG Analysis
- **Muscle Activity Monitoring**: EMG signal visualization for quality control
- **Artifact Detection**: Identification of muscle artifacts in EEG data
- **Time-Frequency Analysis**: EMG power across different frequency bands

### Group-Level Visualizations
- **Statistical Comparisons**: Group means with error bars and significance testing
- **Effect Size Plots**: Visualization of effect sizes across conditions
- **Correlation Analysis**: Relationships between different measures
- **Heatmaps**: Matrix visualization of group-level statistics

## 🔄 Data Processing Pipeline

### Preprocessing Workflow

```
Raw EEG Data
    ↓
Channel Configuration & Montage
    ↓
TMS Artifact Removal
    ↓
Filtering (Bandpass + Notch)
    ↓
Epoch Creation (-0.8s to +0.8s)
    ↓
ICA Decomposition
    ↓
Downsampling (725 Hz)
    ↓
Processed Epochs
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