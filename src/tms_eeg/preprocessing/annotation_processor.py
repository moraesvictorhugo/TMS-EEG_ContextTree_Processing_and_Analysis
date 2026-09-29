from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import mne
import numpy as np

from tms_eeg.config.settings import ProjectConfig
from tms_eeg.paths import raw_dir


@dataclass
class TextFileParser:
    """Parser for text files containing trial conditions."""
    
    subject_id: str
    
    @property
    def data_dir(self) -> Path:
        """Get the data directory for the subject."""
        return raw_dir(self.subject_id)
    
    def find_text_file(self) -> Optional[Path]:
        """Find the text file with trial conditions in the subject's data folder."""
        try:
            # Look for .txt files in the subject's data folder
            txt_files = list(self.data_dir.glob("*.txt"))
            
            if not txt_files:
                return None
            
            # Return the first .txt file found
            return txt_files[0]
            
        except Exception as e:
            print(f"Error finding text file: {e}")
            return None
    
    def parse_trial_conditions(self, text_file: Path) -> List[str]:
        """Parse trial conditions from text file.
        
        Expected format: one condition per line, corresponding to each trial.
        Example:
        1
        2
        3
        1
        2
        ...
        """
        try:
            conditions = []
            with open(text_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        conditions.append(f"8bit {int(line) +1}")
            
            return conditions
            
        except Exception as e:
            print(f"Error parsing text file {text_file}: {e}")
            return []


class AnnotationProcessor:
    """Processor for handling annotations with 8-bit triggers or text file fallback."""
    
    def __init__(self, config: ProjectConfig):
        self.config = config
        self.text_parser = TextFileParser(config.subject_id)
    
    def process_annotations(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Process annotations to replace Stimulus A with condition labels.
        
        Args:
            raw: Raw EEG data with annotations
            
        Returns:
            Raw data with processed annotations
        """
        # Step 1: Check for 8-bit triggers
        has_8bit_triggers = self._check_8bit_triggers(raw)
        
        if has_8bit_triggers:
            print("Found 8-bit trigger annotations. Using them for condition labeling.")
            raw_processed = self._replace_with_8bit_labels(raw)
        else:
            print("No 8-bit trigger annotations found. Checking for text file fallback.")
            raw_processed = self._replace_with_text_file_labels(raw)
        
        return raw_processed
    
    def _check_8bit_triggers(self, raw: mne.io.Raw) -> bool:
        """Check if 8-bit trigger annotations exist in the data."""
        try:
            # Look for 8-bit trigger annotations
            annotations = raw.annotations
            trigger_annotations = [desc for desc in annotations.description 
                                 if desc.startswith('8Bit')]
            
            return len(trigger_annotations) > 0
            
        except Exception as e:
            print(f"Error checking for 8-bit triggers: {e}")
            return False
    
    def _replace_with_8bit_labels(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Replace Stimulus A annotations with 8-bit trigger labels.
        
        This method finds Stimulus A events and replaces them with the 
        corresponding 8-bit trigger label from the same time window.
        The matched 8-bit trigger annotations are removed after replacement.
        """
        try:
            annotations = raw.annotations
            
            # Find Stimulus A events
            stimulus_a_indices = []
            stimulus_a_times = []
            for i, desc in enumerate(annotations.description):
                if desc == 'Stimulus A':
                    stimulus_a_indices.append(i)
                    stimulus_a_times.append(annotations.onset[i])
            
            if not stimulus_a_indices:
                print("No Stimulus A events found.")
                return raw
            
            # Find 8-bit trigger events
            trigger_indices = []
            trigger_times = []
            trigger_descriptions = []
            for i, desc in enumerate(annotations.description):
                if desc.startswith('8Bit'):
                    trigger_indices.append(i)
                    trigger_times.append(annotations.onset[i])
                    trigger_descriptions.append(desc)
            
            if not trigger_times:
                print("No 8-bit trigger events found.")
                return raw
            
            stimulus_a_times = np.array(stimulus_a_times)
            trigger_times = np.array(trigger_times)
            
            new_descriptions = list(annotations.description)
            
            # Track which 8-bit trigger indices (in annotations) were matched
            matched_trigger_indices = set()
            
            tolerance = 0.2  # 200ms
            
            for i, stim_time in enumerate(stimulus_a_times):
                time_diff = np.abs(trigger_times - stim_time)
                valid_mask = time_diff < tolerance
                
                if np.any(valid_mask):
                    # Find closest ONLY among valid triggers
                    valid_positions = np.where(valid_mask)[0]
                    closest_valid = valid_positions[np.argmin(time_diff[valid_mask])]
                    
                    new_label = trigger_descriptions[closest_valid]
                    stim_idx = stimulus_a_indices[i]
                    new_descriptions[stim_idx] = new_label
                    
                    # Mark the matched 8-bit trigger for removal
                    matched_trigger_indices.add(trigger_indices[closest_valid])
                    
                    print(f"Replaced Stimulus A at {stim_time:.3f}s with {new_label}")
                else:
                    print(f"No 8-bit trigger found near Stimulus A at {stim_time:.3f}s")
            
            # Build new annotations excluding matched 8-bit triggers
            keep_mask = np.array([
                i not in matched_trigger_indices 
                for i in range(len(annotations))
            ], dtype=bool)

            onsets = np.array(annotations.onset)[keep_mask]
            durations = np.array(annotations.duration)[keep_mask]
            descriptions = np.array(new_descriptions)[keep_mask]

            new_annotations = mne.Annotations(
                onset=onsets,
                duration=durations,
                description=descriptions
            )
            
            raw_processed = raw.copy()
            raw_processed.set_annotations(new_annotations)
            
            print(f"\nTotal replacements: {len(matched_trigger_indices)}")
            print(f"8-bit triggers removed: {len(matched_trigger_indices)}")
            
            return raw_processed
            
        except Exception as e:
            print(f"Error replacing with 8-bit labels: {e}")
            return raw
    
    def _replace_with_text_file_labels(self, raw: mne.io.Raw) -> mne.io.Raw:
        """Replace Stimulus A annotations using text file mapping.
        
        Args:
            raw: Raw EEG data
            
        Returns:
            Raw data with Stimulus A replaced by text file conditions
        """
        try:
            # Find text file
            text_file = self.text_parser.find_text_file()
            
            if text_file is None:
                print("No text file found. Keeping original Stimulus A annotations.")
                return raw
            
            print(f"Found text file: {text_file}")
            
            # Parse trial conditions from text file
            conditions = self.text_parser.parse_trial_conditions(text_file)
            
            if not conditions:
                print("No valid conditions found in text file. Keeping original annotations.")
                return raw
            
            # Get Stimulus A events using safer approach
            stimulus_a_indices = []
            stimulus_a_times = []
            for i, desc in enumerate(raw.annotations.description):
                if desc == 'Stimulus A':
                    stimulus_a_indices.append(i)
                    stimulus_a_times.append(raw.annotations.onset[i])
            
            if not stimulus_a_indices:
                print("No Stimulus A events found. Cannot apply text file mapping.")
                return raw
            
            # Ensure we have enough conditions for all Stimulus A events
            if len(conditions) < len(stimulus_a_indices):
                print(f"Warning: Only {len(conditions)} conditions in text file, "
                      f"but {len(stimulus_a_indices)} Stimulus A events. "
                      f"Repeating conditions to fill remaining events.")
                
                # Repeat conditions to match number of Stimulus A events
                conditions = (conditions * 
                             (len(stimulus_a_indices) // len(conditions) + 1))[
                             :len(stimulus_a_indices)]
            
            # Create new descriptions (start with original)
            new_descriptions = list(raw.annotations.description)
            
            # Replace Stimulus A events with text file conditions
            for i, stim_idx in enumerate(stimulus_a_indices):
                if i < len(conditions):
                    new_descriptions[stim_idx] = conditions[i]
                    print(f"Replaced Stimulus A at {raw.annotations.onset[stim_idx]:.3f}s "
                          f"with {conditions[i]}")
            
            # Create new annotations
            new_annotations = mne.Annotations(
                onset=raw.annotations.onset,
                duration=raw.annotations.duration,
                description=new_descriptions
            )
            
            # Create new raw object with updated annotations
            raw_processed = raw.copy()
            raw_processed.set_annotations(new_annotations)
            
            return raw_processed
            
        except Exception as e:
            print(f"Error replacing with text file labels: {e}")
            return raw
