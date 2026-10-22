import mne
import numpy as np
from typing import Dict, List, Optional
from tms_eeg.config.settings import ProjectConfig

class ContextMapper:
    """
    Map surviving epochs to contexts in the original stimulus sequence.
    """

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.event_to_symbol = config.analysis_event_to_symbol
        self.context_definitions = config.analysis_context_definitions

    def get_full_sequence(self, raw_path: str) -> np.ndarray:
        """
        Load raw data, process annotations, and extract the full symbol sequence.

        Parameters
        ----------
        raw_path : str
            Path to the raw BDF or FIF file.

        Returns
        -------
        symbols : np.ndarray, shape (n_events,)
            Symbols in their original order.
        event_indices : np.ndarray, shape (n_events,)
            Positional event indices.
        """
        from tms_eeg.preprocessing.annotation_processor import AnnotationProcessor
        from tms_eeg.preprocessing.epoching import find_events

        raw = mne.io.read_raw(raw_path, preload=False, verbose=False)

        # Match annotation processing in the preprocessing pipeline.
        processor = AnnotationProcessor(self.config)
        raw = processor.process_annotations(raw)

        # Use the same event finder for consistency.
        events, event_id = find_events(raw, self.config)

        symbols = np.array([
            self.event_to_symbol[code] for code in events[:, 2]
        ])

        return symbols, np.arange(len(symbols))


    def classify_epochs(
        self,
        full_sequence: np.ndarray,
        surviving_indices: np.ndarray,
    ) -> Dict[str, List[int]]:
        """
        Classify each surviving epoch by context.

        Parameters
        ----------
        full_sequence : np.ndarray
            Full symbol sequence from the raw data.
        surviving_indices : np.ndarray
            Original indices of surviving epochs (``epochs.selection``).

        Returns
        -------
        context_map : dict
            Mapping of context names to positions in the surviving Epochs
            object (0, 1, 2, ...), not original event indices.
        """
        context_map = {name: [] for name in self.context_definitions}

        for epoch_pos, orig_idx in enumerate(surviving_indices):
            current_symbol = full_sequence[orig_idx]

            for ctx_name, pattern in self.context_definitions.items():
                depth = len(pattern)

                if depth == 1:
                    # A depth-one context depends only on the current symbol.
                    if current_symbol == pattern[0]:
                        context_map[ctx_name].append(epoch_pos)

                elif depth >= 2:
                    # Check prior symbols in the original, gap-free sequence.
                    if orig_idx < depth - 1:
                        continue  # Not enough history.

                    # Check consecutive symbols in the original sequence.
                    history_indices = list(
                        range(orig_idx - (depth - 1), orig_idx + 1)
                    )

                    # Extract the pattern from the original sequence.
                    actual_pattern = [
                        full_sequence[i] for i in history_indices
                    ]

                    if actual_pattern == pattern:
                        context_map[ctx_name].append(epoch_pos)

        # Report context counts.
        for ctx_name, indices in context_map.items():
            print(
                f"  Context '{ctx_name}': "
                f"{len(indices)} epochs found"
            )

        return context_map

    def get_context_epochs(
        self,
        epochs: mne.Epochs,
        raw_path: str,
    ) -> Dict[str, mne.Epochs]:
        """
        Extract the raw sequence, classify surviving epochs, and return
        an Epochs subset for each non-empty context.

        Parameters
        ----------
        epochs : mne.Epochs
            Preprocessed Epochs with rejection already applied.
        raw_path : str
            Path to the original raw file.

        Returns
        -------
        context_epochs : dict
            Mapping of context names to Epochs subsets.
        """
        full_sequence, _ = self.get_full_sequence(raw_path)
        surviving_indices = epochs.selection

        print(f"\n{'='*55}")
        print(f"  Context analysis — {self.config.subject_id}")
        print(f"  Full sequence: {len(full_sequence)} events")
        print(f"  Surviving epochs: {len(surviving_indices)}")
        print(f"{'='*55}")

        context_map = self.classify_epochs(
            full_sequence, surviving_indices
        )

        context_epochs = {}
        for ctx_name, epoch_indices in context_map.items():
            if len(epoch_indices) == 0:
                print(
                    f"  ⚠ Context '{ctx_name}': no epochs found; "
                    f"skipping."
                )
                continue
            context_epochs[ctx_name] = epochs[epoch_indices]

        return context_epochs