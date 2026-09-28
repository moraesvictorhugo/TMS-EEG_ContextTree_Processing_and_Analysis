""""context.py - maps surviving epochs to context-tree contexts."""

import mne
import numpy as np
from typing import Dict, List

from tms_eeg.config.settings import ProjectConfig


class ContextMapper:
    """
    Maps surviving epochs to contexts based on the original stimulus
    sequence (context tree).
    """

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.event_to_symbol = config.events.event_to_symbol
        self.context_definitions = config.analysis.context_definitions

    def get_full_sequence(self, raw_path: str) -> np.ndarray:
        """
        Extract the full original symbol sequence of a subject.

        The sequence is loaded from the raw file (reproducing the annotation
        processing of the preprocessing pipeline) and cached next to the
        processed data, so repeated analysis runs do not re-read the raw.

        Parameters
        ----------
        raw_path : str
            Path to the raw .bdf/.fif file.

        Returns
        -------
        symbols : np.ndarray, shape (n_events,)
            Symbol sequence in the original order.
        """
        cache_path = (
            self.config.paths.subject_processed_dir(self.config.subject_id)
            / "stimulus_sequence.npy"
        )
        if cache_path.exists():
            return np.load(cache_path)

        from tms_eeg.preprocessing.annotation_processor import AnnotationProcessor
        from tms_eeg.preprocessing.epoching import EEGEpocher

        raw = mne.io.read_raw(raw_path, preload=False, verbose=False)

        # Reproduce the same annotation processing as the preprocessing pipeline.
        raw = AnnotationProcessor(self.config).process_annotations(raw)

        # Use the same event finding to guarantee consistency.
        events, _ = EEGEpocher(self.config).find_events(raw)

        symbols = np.array([self.event_to_symbol[code] for code in events[:, 2]])

        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, symbols)
        return symbols

    def classify_epochs(
        self,
        full_sequence: np.ndarray,
        surviving_indices: np.ndarray,
    ) -> Dict[str, List[int]]:
        """
        Classify every surviving epoch into contexts.

        Parameters
        ----------
        full_sequence : np.ndarray
            Complete symbol sequence of the raw file.
        surviving_indices : np.ndarray
            Original indices of the surviving epochs (``epochs.selection``).

        Returns
        -------
        context_map : dict
            ``{context_name: [positions in the surviving Epochs object]}``
            (positions, not original indices).
        """
        context_map = {name: [] for name in self.context_definitions}

        for epoch_pos, orig_idx in enumerate(surviving_indices):
            symbol_atual = full_sequence[orig_idx]

            for ctx_name, pattern in self.context_definitions.items():
                depth = len(pattern)

                if depth == 1:
                    # No-history context: current symbol alone determines it.
                    if symbol_atual == pattern[0]:
                        context_map[ctx_name].append(epoch_pos)

                elif depth >= 2:
                    # History contexts need the previous symbols on the
                    # ORIGINAL sequence (no gaps from removed epochs).
                    if orig_idx < depth - 1:
                        continue  # not enough history

                    history_indices = list(
                        range(orig_idx - (depth - 1), orig_idx + 1)
                    )
                    actual_pattern = [full_sequence[i] for i in history_indices]

                    if actual_pattern == pattern:
                        context_map[ctx_name].append(epoch_pos)

        # Log
        for ctx_name, indices in context_map.items():
            print(f"  Contexto '{ctx_name}': {len(indices)} épocas encontradas")

        return context_map

    def get_context_epochs(
        self,
        epochs: mne.Epochs,
        raw_path: str,
    ) -> Dict[str, mne.Epochs]:
        """
        Full pipeline: load the raw sequence, classify the surviving epochs
        and return sub-epochs per context.

        Parameters
        ----------
        epochs : mne.Epochs
            Processed epochs (with rejections already applied).
        raw_path : str
            Path to the raw file.

        Returns
        -------
        context_epochs : dict
            ``{context_name: mne.Epochs}`` subsets.
        """
        full_sequence = self.get_full_sequence(raw_path)
        surviving_indices = epochs.selection

        print(f"\n{'='*55}")
        print(f"  Context Analysis — {self.config.subject_id}")
        print(f"  Sequência completa: {len(full_sequence)} eventos")
        print(f"  Épocas sobreviventes: {len(surviving_indices)}")
        print(f"{'='*55}")

        context_map = self.classify_epochs(full_sequence, surviving_indices)

        context_epochs = {}
        for ctx_name, epoch_indices in context_map.items():
            if len(epoch_indices) == 0:
                print(f"  ⚠ Contexto '{ctx_name}': nenhuma época, pulando.")
                continue
            context_epochs[ctx_name] = epochs[epoch_indices]

        return context_epochs