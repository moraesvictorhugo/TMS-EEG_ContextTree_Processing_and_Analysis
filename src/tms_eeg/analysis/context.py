import mne
import numpy as np
from typing import Dict, List, Optional
from tms_eeg.config.settings import ProjectConfig

class ContextMapper:
    """
    Mapeia épocas sobreviventes a contextos baseados na sequência
    original de estímulos (árvore de contexto).
    """

    def __init__(self, config: ProjectConfig):
        self.config = config
        self.event_to_symbol = config.analysis_event_to_symbol
        self.context_definitions = config.analysis_context_definitions

    def get_full_sequence(self, raw_path: str) -> np.ndarray:
        """
        Carrega o raw, aplica o AnnotationProcessor (mesmo pipeline do
        pré-processamento) e extrai a sequência completa de símbolos.

        Parameters
        ----------
        raw_path : str
            Caminho para o arquivo .bdf/.fif raw.

        Returns
        -------
        symbols : np.ndarray, shape (n_events,)
            Sequência de símbolos na ordem original.
        event_indices : np.ndarray, shape (n_events,)
            Índices posicionais de cada evento.
        """
        from tms_eeg.preprocessing.annotation_processor import AnnotationProcessor
        from tms_eeg.preprocessing.epoching import find_events

        raw = mne.io.read_raw(raw_path, preload=False, verbose=False)

        # Reproduz o mesmo processamento de annotations do pipeline
        processor = AnnotationProcessor(self.config)
        raw = processor.process_annotations(raw)

        # Usa o mesmo find_events para garantir consistência
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
        Classifica cada época sobrevivente em contextos.

        Parameters
        ----------
        full_sequence : np.ndarray
            Sequência completa de símbolos (da raw).
        surviving_indices : np.ndarray
            Índices originais das épocas que sobreviveram
            (epochs.selection).

        Returns
        -------
        context_map : dict
            {context_name: [índices dentro do epochs sobrevivente]}
            Os índices são posições no objeto epochs (0, 1, 2, ...),
            NÃO os índices originais.
        """
        context_map = {name: [] for name in self.context_definitions}

        for epoch_pos, orig_idx in enumerate(surviving_indices):
            symbol_atual = full_sequence[orig_idx]

            for ctx_name, pattern in self.context_definitions.items():
                depth = len(pattern)

                if depth == 1:
                    # Contexto sem história — basta o símbolo atual
                    if symbol_atual == pattern[0]:
                        context_map[ctx_name].append(epoch_pos)

                elif depth >= 2:
                    # Precisa verificar os símbolos anteriores na
                    # sequência ORIGINAL (sem gaps)
                    if orig_idx < depth - 1:
                        continue  # não tem história suficiente

                    # Checa se os anteriores são consecutivos na
                    # sequência original (sem remoção entre eles)
                    history_indices = list(
                        range(orig_idx - (depth - 1), orig_idx + 1)
                    )

                    # Extrai o padrão da sequência original
                    actual_pattern = [
                        full_sequence[i] for i in history_indices
                    ]

                    if actual_pattern == pattern:
                        context_map[ctx_name].append(epoch_pos)

        # Log
        for ctx_name, indices in context_map.items():
            print(
                f"  Contexto '{ctx_name}': "
                f"{len(indices)} épocas encontradas"
            )

        return context_map

    def get_context_epochs(
        self,
        epochs: mne.Epochs,
        raw_path: str,
    ) -> Dict[str, mne.Epochs]:
        """
        Pipeline completo: extrai a sequência do raw, classifica
        as épocas sobreviventes e retorna sub-epochs por contexto.

        Parameters
        ----------
        epochs : mne.Epochs
            Objeto epochs (já pré-processado, com rejeições aplicadas).
        raw_path : str
            Caminho para o arquivo raw original.

        Returns
        -------
        context_epochs : dict
            {context_name: mne.Epochs} — subconjuntos de epochs.
        """
        full_sequence, _ = self.get_full_sequence(raw_path)
        surviving_indices = epochs.selection

        print(f"\n{'='*55}")
        print(f"  Context Analysis — {self.config.subject_id}")
        print(f"  Sequência completa: {len(full_sequence)} eventos")
        print(f"  Épocas sobreviventes: {len(surviving_indices)}")
        print(f"{'='*55}")

        context_map = self.classify_epochs(
            full_sequence, surviving_indices
        )

        context_epochs = {}
        for ctx_name, epoch_indices in context_map.items():
            if len(epoch_indices) == 0:
                print(
                    f"  ⚠ Contexto '{ctx_name}': nenhuma época, "
                    f"pulando."
                )
                continue
            context_epochs[ctx_name] = epochs[epoch_indices]

        return context_epochs