"""Unit tests for barge-in filter."""

import time
from unittest.mock import patch

import numpy as np
import pytest

from server.pipeline.bargein import BargeInFilter


class TestBargeInFilter:
    def test_does_not_trigger_before_min_duration(self):
        filt = BargeInFilter(min_duration_ms=300, min_energy=100.0)
        filt.on_speech_start()

        # Feed a loud chunk but immediately (well under 300ms)
        loud_audio = np.full(480, 5000, dtype=np.int16)
        assert filt.on_speech_audio(loud_audio) is False

    def test_does_not_trigger_below_energy_threshold(self):
        filt = BargeInFilter(min_duration_ms=10, min_energy=5000.0)
        filt.on_speech_start()

        # Feed quiet audio — should not trigger even after enough time
        quiet_audio = np.full(480, 50, dtype=np.int16)
        # Simulate time passing
        filt._speech_start_time = time.perf_counter() - 1.0
        assert filt.on_speech_audio(quiet_audio) is False

    def test_triggers_when_both_thresholds_met(self):
        filt = BargeInFilter(min_duration_ms=10, min_energy=100.0)
        filt.on_speech_start()

        # Simulate time passing
        filt._speech_start_time = time.perf_counter() - 0.5

        loud_audio = np.full(480, 5000, dtype=np.int16)
        assert filt.on_speech_audio(loud_audio) is True

    def test_triggers_only_once(self):
        filt = BargeInFilter(min_duration_ms=10, min_energy=100.0)
        filt.on_speech_start()
        filt._speech_start_time = time.perf_counter() - 0.5

        loud_audio = np.full(480, 5000, dtype=np.int16)
        assert filt.on_speech_audio(loud_audio) is True
        assert filt.on_speech_audio(loud_audio) is False

    def test_reset_allows_retriggering(self):
        filt = BargeInFilter(min_duration_ms=10, min_energy=100.0)
        filt.on_speech_start()
        filt._speech_start_time = time.perf_counter() - 0.5

        loud_audio = np.full(480, 5000, dtype=np.int16)
        assert filt.on_speech_audio(loud_audio) is True

        filt.reset()
        filt.on_speech_start()
        filt._speech_start_time = time.perf_counter() - 0.5
        assert filt.on_speech_audio(loud_audio) is True

    def test_no_trigger_without_speech_start(self):
        filt = BargeInFilter(min_duration_ms=10, min_energy=100.0)
        loud_audio = np.full(480, 5000, dtype=np.int16)
        assert filt.on_speech_audio(loud_audio) is False
