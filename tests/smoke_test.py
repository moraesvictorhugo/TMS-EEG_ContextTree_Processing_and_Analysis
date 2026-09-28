"""Lightweight smoke tests — runnable without pytest.

Validates the refactored internals on synthetic data.

Usage
-----
    python tests/smoke_test.py
"""

from __future__ import annotations

import numpy as np
import mne

from tms_eeg.analysis.context import ContextMapper
from tms_eeg.analysis.features import FeatureExtractor
from tms_eeg.analysis.group import MetricsCollector
from tms_eeg.analysis.labels import normalize_event_id
from tms_eeg.config.settings import ProjectConfig


def test_config() -> None:
    config = ProjectConfig(subject_id="V04")
    assert config.subject_id == "V04"
    assert config.paths.processed_dir.name == "processed"
    # Subjects list must not contain duplicates.
    assert len(config.analysis.subjects) == len(set(config.analysis.subjects))
    assert config.events.event_to_symbol == {1: 0, 2: 1, 3: 2}
    assert "session" not in config.__dataclass_fields__
    print("  [ok] ProjectConfig")


def test_classify_epochs() -> None:
    config = ProjectConfig(subject_id="V00")
    full = np.array([0, 1, 2, 1, 1, 2, 0, 1, 2, 2], dtype=int)
    mapper = ContextMapper(config)

    results = mapper.classify_epochs(full, np.arange(len(full)))

    assert set(results) == set(config.analysis.context_definitions)
    assert results["ctx_0"] == [0, 6]
    assert results["ctx_2"] == [2, 5, 8, 9]
    assert results["ctx_01"] == [1, 7]
    assert results["ctx_11"] == [4]
    assert results["ctx_21"] == [3]
    print("  [ok] ContextMapper.classify_epochs")


def _synthetic_epochs() -> mne.Epochs:
    rng = np.random.default_rng(0)
    info = mne.create_info(["C3", "C4", "FC1", "CP1"], 250, "eeg")
    raw = mne.io.RawArray(rng.standard_normal((4, 1500)) * 1e-6, info, verbose=False)
    events = np.array([[100, 0, 1], [600, 0, 2], [1200, 0, 3]], dtype=int)
    event_id = {"0": 1, "1": 2, "2": 3}
    return mne.Epochs(
        raw, events, event_id, tmin=-0.1, tmax=0.3, preload=True, verbose=False)


def test_normalize_event_id() -> None:
    epochs = _synthetic_epochs()
    epochs.event_id = {"8bit 1": 1, "8bit 2": 2, "8bit 3": 3}
    config = ProjectConfig()
    normalize_event_id(epochs, config.events.label_rename)
    assert list(epochs.event_id) == ["0", "1", "2"]
    print("  [ok] normalize_event_id")


def test_features() -> None:
    from tms_eeg.io.writer import Writer

    epochs = _synthetic_epochs()
    config = ProjectConfig(subject_id="V00")
    windows = {"N15": (0.012, 0.02), "P30": (0.02, 0.04)}
    extractor = FeatureExtractor(epochs, ["C3", "C4"], windows)

    evokeds = extractor.get_evokeds()
    assert set(evokeds) == {"0", "1", "2"}

    p2p = extractor.peak_to_peak("N15", "P30", evokeds=evokeds)
    assert len(p2p) == 3 * 2  # 3 conditions x 2 channels

    gmfp = extractor.compute_gmfp()
    lmfp = extractor.compute_lmfp()
    assert set(gmfp) == {"0", "1", "2"}
    assert list(gmfp.values())[0].shape == epochs.times.shape

    peaks = extractor.extract_mfp_peaks(gmfp, label="GMFP")
    assert {"condition", "measure", "component", "peak_amplitude_uV", "peak_latency_ms"} <= set(peaks.columns)

    # Writer with synthetic object must at least fail fast (no export here).
    writer = Writer(config)
    assert writer.subject_id == "V00"
    print("  [ok] FeatureExtractor + Writer")


def test_collector() -> None:
    collector = MetricsCollector()
    collector.add_peak_to_peak("V00", "condition", "0", "C3", "N15-P30", 12.0)
    collector.add_mfp_peaks("V00", "condition", "0", "GMFP", "P30", 5.0, 29.0)
    df = collector.to_dataframe()
    assert len(df) == 3  # 1 p2p row + amplitude + latency rows
    assert list(df.columns) == [
        "subject", "analysis_type", "condition", "channel", "component", "metric", "value"]
    print("  [ok] MetricsCollector")


def main() -> None:
    print("Running smoke tests...")
    test_config()
    test_classify_epochs()
    test_normalize_event_id()
    test_features()
    test_collector()
    print("\nAll smoke tests passed.")


if __name__ == "__main__":
    main()