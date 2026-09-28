import re

import mne
import numpy as np
from tms_eeg.config.settings import ProjectConfig

class EEGEpocher:
    def __init__(self, config: ProjectConfig):
        self.config = config

    def find_events(self, raw: mne.io.BaseRaw) -> tuple[np.ndarray, dict]:
        """Find events using configured trigger_id or 8bit labels as fallback."""
        trigger_id = self.config.events.trigger_id
        raw_descriptions = {str(d) for d in raw.annotations.description}

        # Try configured trigger_id (e.g. 'Stimulus A')
        try:
            events, event_id = mne.events_from_annotations(
                raw, event_id=trigger_id,
            )
            return events, event_id
        except ValueError:
            pass

        # Fallback: try 8bit labels from stimulus mapping
        eight_bit_id = {
            label: idx + 1
            for labels in self.config.events.stimulus_to_8bit_mapping.values()
            for idx, label in enumerate(labels)
        }

        # Case-insensitive matching: remap config labels to actual raw labels
        raw_desc_lower = {d.lower(): d for d in raw_descriptions}
        resolved_id = {}
        for label, idx in eight_bit_id.items():
            if label in raw_descriptions:
                resolved_id[label] = idx
            elif label.lower() in raw_desc_lower:
                resolved_id[raw_desc_lower[label.lower()]] = idx

        if resolved_id:
            # Build regexp to match only the resolved labels
            pattern = "|".join(re.escape(k) for k in resolved_id)
            try:
                events, event_id = mne.events_from_annotations(
                    raw, regexp=pattern,
                )
                # Remap event_id values to our resolved indices
                final_event_id = {}
                for ann_label, mne_idx in event_id.items():
                    matched = str(ann_label).lower()
                    for res_label, res_idx in resolved_id.items():
                        if res_label.lower() == matched:
                            final_event_id[ann_label] = res_idx
                            break

                # Remap event codes in the events array
                for ann_label, mne_idx in event_id.items():
                    if ann_label in final_event_id:
                        events[events[:, 2] == mne_idx, 2] = final_event_id[ann_label]

                print(f"[EEGEpocher] Using 8bit event_id: {final_event_id}")
                return events, final_event_id
            except ValueError:
                pass

        # No valid annotations found
        raise ValueError(
            f"[EEGEpocher] No events found with trigger_id={trigger_id} "
            f"or 8bit_id={eight_bit_id}. "
            f"Raw annotations found: {raw_descriptions}"
        )


    def create_epochs(self, raw: mne.io.BaseRaw) -> mne.Epochs:
        """Create epochs from raw data.

        Args:
            raw (mne.io.BaseRaw): Preprocessed raw EEG data.

        Returns:
            mne.Epochs: Epoched EEG data.
        """
        epoch_cfg = self.config.epochs

        events, event_id = self.find_events(raw)

        epochs = mne.Epochs(
            raw,
            events=events,
            event_id=event_id,
            tmin=epoch_cfg.window[0],
            tmax=epoch_cfg.window[1],
            baseline=epoch_cfg.baseline,
            preload=True,
            reject=None,
            flat=None
        )

        return epochs

class EpochDropper:
    def __init__(self, config: ProjectConfig):
        self.config = config

    def drop_from_json(self, epochs: mne.Epochs, json_path: str) -> mne.Epochs:
        import json
        with open(json_path) as f:
            idx = json.load(f).get(self.config.subject_id, [])
        epochs.drop(idx)
        return epochs
