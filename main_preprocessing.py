"""Preprocessing entry point: run the full TMS-EEG pipeline for one subject.

The pipeline ends with the epochs exports in ``data/processed/<subject>/``
(``.fif`` full / pre+post / post-only variants, EMG epochs and the
context-tree ``.mat`` files).

Usage
-----
    python main_preprocessing.py --subject V04 --qc    # with interactive QC
    python main_preprocessing.py --subject V04         # headless
"""

import argparse

from tms_eeg.config.environment import setup_plotting_backend
from tms_eeg.config.settings import ProjectConfig
from tms_eeg.preprocessing.pipeline import PreprocessingPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subject",
        default=None,
        help="Subject id (default: first subject in config.analysis.subjects)",
    )
    parser.add_argument(
        "--qc",
        action="store_true",
        help="Show interactive QC plots (pauses the pipeline for inspection)",
    )
    args = parser.parse_args()

    config = ProjectConfig(subject_id=args.subject or ProjectConfig().analysis.subjects[0])
    setup_plotting_backend()

    PreprocessingPipeline(config, qc=args.qc).run()


if __name__ == "__main__":
    main()