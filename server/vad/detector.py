"""Voice Activity Detector with speech state machine.

Handles 20ms→30ms re-chunking, pre-speech ring buffer, and
speech accumulation with safety caps.
"""

import collections
import enum
import logging

import numpy as np

from server.config import settings
from server.vad.model import SileroVAD

logger = logging.getLogger(__name__)


class SpeechState(enum.Enum):
    SILENCE = "silence"
    SPEECH_START = "speech_start"
    SPEAKING = "speaking"
    SPEECH_END = "speech_end"


class VoiceActivityDetector:
    """Stateful VAD with state machine, chunk accumulation, and ring buffer.

    Feed it resampled 16kHz int16 numpy arrays of any size; it buffers
    and processes in 30ms (480-sample) chunks internally.
    """

    def __init__(
        self,
        threshold: float | None = None,
        silence_duration_ms: int | None = None,
        pre_speech_ms: int | None = None,
        sample_rate: int = 16000,
        chunk_ms: int | None = None,
    ) -> None:
        threshold = threshold if threshold is not None else settings.vad_threshold
        silence_duration_ms = silence_duration_ms if silence_duration_ms is not None else settings.vad_silence_duration_ms
        pre_speech_ms = pre_speech_ms if pre_speech_ms is not None else settings.vad_pre_speech_ms
        chunk_ms = chunk_ms if chunk_ms is not None else settings.vad_chunk_ms

        self._vad = SileroVAD(sample_rate=sample_rate)
        self._threshold = threshold
        self._sample_rate = sample_rate
        self._chunk_samples = sample_rate * chunk_ms // 1000  # 480

        # Chunk accumulator for 20ms → 30ms re-chunking
        self._chunk_buffer = np.array([], dtype=np.int16)

        # State machine
        self._state = SpeechState.SILENCE
        self._silence_frames_needed = max(1, silence_duration_ms // chunk_ms)
        self._silence_frame_count = 0

        # Pre-speech ring buffer (stores 30ms chunks)
        pre_speech_chunks = max(1, pre_speech_ms // chunk_ms)
        self._pre_speech_ring: collections.deque[np.ndarray] = collections.deque(
            maxlen=pre_speech_chunks
        )

        # Speech accumulation buffer
        self._speech_buffer: list[np.ndarray] = []

        # Safety cap
        self._max_speech_samples = int(
            settings.stt_max_utterance_seconds * sample_rate
        )

    @property
    def state(self) -> SpeechState:
        return self._state

    def feed(
        self, audio_16k: np.ndarray
    ) -> list[tuple[SpeechState, np.ndarray | None]]:
        """Feed resampled 16kHz int16 audio. Returns list of state-change events.

        Events emitted:
            (SPEECH_START, None) — speech just began
            (SPEECH_END, concatenated_audio) — speech ended, full utterance attached

        Most calls return an empty list (no state transition).
        """
        events: list[tuple[SpeechState, np.ndarray | None]] = []

        # Append to chunk accumulator
        self._chunk_buffer = np.concatenate([self._chunk_buffer, audio_16k])

        # Process all complete 30ms chunks
        while len(self._chunk_buffer) >= self._chunk_samples:
            chunk = self._chunk_buffer[: self._chunk_samples]
            self._chunk_buffer = self._chunk_buffer[self._chunk_samples :]

            # Run VAD inference
            prob = self._vad(chunk)
            is_speech = prob >= self._threshold

            if self._state == SpeechState.SILENCE:
                self._pre_speech_ring.append(chunk.copy())
                if is_speech:
                    self._state = SpeechState.SPEAKING
                    # Capture pre-speech audio
                    self._speech_buffer = list(self._pre_speech_ring)
                    self._speech_buffer.append(chunk.copy())
                    self._silence_frame_count = 0
                    events.append((SpeechState.SPEECH_START, None))
                    logger.debug("VAD: SILENCE -> SPEAKING (prob=%.3f)", prob)

            elif self._state == SpeechState.SPEAKING:
                self._speech_buffer.append(chunk.copy())

                if not is_speech:
                    self._silence_frame_count += 1
                    if self._silence_frame_count >= self._silence_frames_needed:
                        utterance = np.concatenate(self._speech_buffer)
                        self._speech_buffer.clear()
                        self._pre_speech_ring.clear()
                        self._state = SpeechState.SILENCE
                        self._silence_frame_count = 0
                        events.append((SpeechState.SPEECH_END, utterance))
                        logger.info(
                            "VAD: SPEAKING -> SILENCE (utterance=%d samples, %.2fs)",
                            len(utterance),
                            len(utterance) / self._sample_rate,
                        )
                else:
                    self._silence_frame_count = 0

                # Safety: cap extremely long utterances
                total = sum(len(c) for c in self._speech_buffer)
                if total >= self._max_speech_samples:
                    utterance = np.concatenate(self._speech_buffer)
                    self._speech_buffer.clear()
                    self._state = SpeechState.SILENCE
                    self._silence_frame_count = 0
                    events.append((SpeechState.SPEECH_END, utterance))
                    logger.warning("VAD: forced SPEECH_END (max duration reached)")

        return events

    def reset(self) -> None:
        """Reset all state for a new stream."""
        self._state = SpeechState.SILENCE
        self._chunk_buffer = np.array([], dtype=np.int16)
        self._speech_buffer.clear()
        self._pre_speech_ring.clear()
        self._silence_frame_count = 0
        self._vad.reset()
