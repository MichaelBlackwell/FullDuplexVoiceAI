"""Unit tests for latency instrumentation."""

import time

import pytest

from server.metrics import LatencyTracker, UtteranceTimings


class TestUtteranceTimings:
    def test_voice_to_voice_latency(self):
        t = UtteranceTimings(utterance_id=1)
        t.speech_end = 100.0
        t.first_frame_to_client = 100.8
        assert abs(t.voice_to_voice_latency - 0.8) < 0.001

    def test_voice_to_voice_latency_zero_when_incomplete(self):
        t = UtteranceTimings(utterance_id=1)
        t.speech_end = 100.0
        assert t.voice_to_voice_latency == 0.0

    def test_log_breakdown_does_not_raise(self):
        t = UtteranceTimings(utterance_id=1)
        t.speech_start = 100.0
        t.speech_end = 101.0
        t.stt_start = 101.0
        t.stt_end = 101.3
        t.llm_first_token = 101.5
        t.llm_done = 102.0
        t.tts_first_chunk_done = 101.6
        t.first_frame_to_client = 101.7
        t.log_breakdown()  # Should not raise


class TestLatencyTracker:
    def test_new_utterance_increments_id(self):
        tracker = LatencyTracker()
        t1 = tracker.new_utterance()
        t2 = tracker.new_utterance()
        assert t1.utterance_id == 1
        assert t2.utterance_id == 2

    def test_current_tracks_latest(self):
        tracker = LatencyTracker()
        assert tracker.current is None
        t = tracker.new_utterance()
        assert tracker.current is t

    def test_finalize_clears_current(self):
        tracker = LatencyTracker()
        tracker.new_utterance()
        tracker.finalize()
        assert tracker.current is None

    def test_percentiles_with_data(self):
        tracker = LatencyTracker()
        for i in range(10):
            t = tracker.new_utterance()
            t.speech_end = float(i)
            t.first_frame_to_client = float(i) + 0.5  # 500ms V2V
            tracker.finalize()
        p = tracker.percentiles()
        assert "p50" in p
        assert "p95" in p
        assert "p99" in p
        assert abs(p["p50"] - 500.0) < 1.0

    def test_percentiles_empty(self):
        tracker = LatencyTracker()
        assert tracker.percentiles() == {}
