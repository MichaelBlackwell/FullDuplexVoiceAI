"""Tests for the VoiceActivityDetector state machine.

Uses a mock VAD model to control speech probabilities directly,
since Silero VAD only responds to real human speech.
"""

from unittest.mock import MagicMock

import numpy as np
import pytest

from server.vad.detector import SpeechState, VoiceActivityDetector


def _silence(duration_ms: int, sample_rate: int = 16000) -> np.ndarray:
    """Generate silence (zeros)."""
    samples = int(sample_rate * duration_ms / 1000)
    return np.zeros(samples, dtype=np.int16)


def _make_detector(
    probabilities: list[float],
    threshold: float = 0.5,
    silence_duration_ms: int = 300,
    pre_speech_ms: int = 300,
) -> VoiceActivityDetector:
    """Create a VoiceActivityDetector with a mock VAD that returns
    predetermined probabilities for each 30ms chunk."""
    vad = VoiceActivityDetector(
        threshold=threshold,
        silence_duration_ms=silence_duration_ms,
        pre_speech_ms=pre_speech_ms,
    )
    # Replace the real Silero model with a mock
    mock_vad = MagicMock()
    prob_iter = iter(probabilities)
    mock_vad.side_effect = lambda chunk: next(prob_iter, 0.0)
    mock_vad.reset = MagicMock()
    vad._vad = mock_vad
    return vad


class TestVoiceActivityDetectorStateMachine:

    def test_starts_in_silence(self):
        vad = _make_detector([])
        assert vad.state == SpeechState.SILENCE

    def test_silence_produces_no_events(self):
        # 100ms = ~3 chunks at 30ms, all below threshold
        vad = _make_detector([0.1, 0.1, 0.1])
        events = vad.feed(_silence(100))
        assert events == []
        assert vad.state == SpeechState.SILENCE

    def test_reset_returns_to_silence(self):
        # Start speech then reset
        vad = _make_detector([0.9, 0.9, 0.9, 0.9])
        vad.feed(_silence(120))  # triggers 4 chunks worth of audio
        vad.reset()
        assert vad.state == SpeechState.SILENCE

    def test_handles_empty_audio(self):
        vad = _make_detector([])
        events = vad.feed(np.array([], dtype=np.int16))
        assert events == []

    def test_handles_small_chunks(self):
        """Feed audio smaller than 30ms — should buffer without error."""
        vad = _make_detector([0.1])
        # 10ms = 160 samples, less than 480 (30ms)
        events = vad.feed(_silence(10))
        assert events == []

    def test_speech_start_event(self):
        """First chunk above threshold emits SPEECH_START."""
        vad = _make_detector([0.9])  # one chunk above threshold
        events = vad.feed(_silence(32))  # 32ms = 512 samples = 1 full chunk
        start_events = [e for e in events if e[0] == SpeechState.SPEECH_START]
        assert len(start_events) == 1
        assert vad.state == SpeechState.SPEAKING

    def test_speech_end_after_silence_threshold(self):
        """Speech followed by enough silence emits SPEECH_END with utterance."""
        # silence_duration_ms=90, chunk_ms=30 → need 3 silence chunks
        # 2 speech chunks + 3 silence chunks = 5 total
        probs = [0.9, 0.9, 0.1, 0.1, 0.1]
        vad = _make_detector(probs, silence_duration_ms=90)
        events = vad.feed(_silence(150))  # 5 × 30ms

        start_events = [e for e in events if e[0] == SpeechState.SPEECH_START]
        end_events = [e for e in events if e[0] == SpeechState.SPEECH_END]

        assert len(start_events) == 1
        assert len(end_events) == 1
        _, utterance = end_events[0]
        assert utterance is not None
        assert len(utterance) > 0
        assert vad.state == SpeechState.SILENCE

    def test_brief_silence_does_not_end_speech(self):
        """A short silence gap (less than threshold) stays in SPEAKING."""
        # silence_duration_ms=96 → need 3 silence chunks at 32ms each
        # speech, silence (2 chunks only), speech → should NOT end
        probs = [0.9, 0.1, 0.1, 0.9, 0.9]
        vad = _make_detector(probs, silence_duration_ms=96)
        events = vad.feed(_silence(160))  # 5 × 32ms

        end_events = [e for e in events if e[0] == SpeechState.SPEECH_END]
        assert len(end_events) == 0
        assert vad.state == SpeechState.SPEAKING

    def test_pre_speech_buffer_included_in_utterance(self):
        """Utterance includes pre-speech ring buffer audio."""
        # pre_speech_ms=90 → 3 chunks of pre-speech
        # 3 silence chunks fill ring buffer, then 1 speech, then 3 silence to end
        probs = [0.1, 0.1, 0.1, 0.9, 0.1, 0.1, 0.1]
        vad = _make_detector(probs, pre_speech_ms=90, silence_duration_ms=90)
        events = vad.feed(_silence(210))  # 7 × 30ms

        end_events = [e for e in events if e[0] == SpeechState.SPEECH_END]
        assert len(end_events) == 1
        _, utterance = end_events[0]

        # Utterance = 3 pre-speech + 1 speech + 3 silence = 7 chunks × 480 = 3360 samples
        # At minimum it should be more than just the 1 speech chunk (480 samples)
        assert len(utterance) > 480

    def test_max_utterance_cap(self):
        """Very long speech is force-ended at safety cap."""
        # Infinite speech probabilities
        probs = [0.9] * 100
        vad = _make_detector(probs, silence_duration_ms=90)
        vad._max_speech_samples = 480 * 3  # 3 chunks (90ms)

        events = vad.feed(_silence(3000))  # feed lots of audio

        end_events = [e for e in events if e[0] == SpeechState.SPEECH_END]
        assert len(end_events) >= 1

    def test_multiple_utterances(self):
        """Two separate speech segments produce two SPEECH_END events."""
        # speech, silence (enough to end), speech, silence (enough to end)
        probs = [0.9, 0.9, 0.1, 0.1, 0.1, 0.9, 0.9, 0.1, 0.1, 0.1]
        vad = _make_detector(probs, silence_duration_ms=90, pre_speech_ms=0)
        events = vad.feed(_silence(300))  # 10 × 30ms

        end_events = [e for e in events if e[0] == SpeechState.SPEECH_END]
        assert len(end_events) == 2


class TestChunkAccumulation:
    """Test the 20ms → 30ms re-chunking logic."""

    def test_20ms_chunks_produce_30ms_processing(self):
        """Feed 20ms chunks (320 samples) — should buffer and process."""
        # 10 × 20ms = 200ms → 6 complete 30ms chunks + 20 samples remainder
        probs = [0.1] * 6
        vad = _make_detector(probs)

        for _ in range(10):
            chunk = _silence(20)
            assert len(chunk) == 320
            vad.feed(chunk)

        assert vad.state == SpeechState.SILENCE
        # Verify the mock was called the right number of times
        assert vad._vad.call_count == 6

    def test_arbitrary_chunk_sizes(self):
        """Feed various chunk sizes — accumulator handles them all."""
        probs = [0.1] * 20  # enough for any combination
        vad = _make_detector(probs)

        for size_ms in [5, 10, 15, 20, 25, 30, 40, 50]:
            events = vad.feed(_silence(size_ms))
            assert isinstance(events, list)


class TestSileroModelIntegration:
    """Integration tests with the real Silero ONNX model.

    These verify the ONNX interface works, not speech detection accuracy.
    """

    def test_model_loads_and_runs(self):
        """Real Silero model loads and processes a chunk without error."""
        vad = VoiceActivityDetector()
        events = vad.feed(_silence(30))
        assert isinstance(events, list)
        assert vad.state == SpeechState.SILENCE

    def test_silence_stays_silent(self):
        """Real model does not detect speech in silence."""
        vad = VoiceActivityDetector()
        events = vad.feed(_silence(1000))
        start_events = [e for e in events if e[0] == SpeechState.SPEECH_START]
        assert len(start_events) == 0

    def test_reset_clears_state(self):
        """Reset works with the real model."""
        vad = VoiceActivityDetector()
        vad.feed(_silence(100))
        vad.reset()
        assert vad.state == SpeechState.SILENCE
