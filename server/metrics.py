"""Latency instrumentation for the voice AI pipeline.

Tracks per-utterance timestamps at every pipeline stage and computes
rolling P50/P95/P99 voice-to-voice latency percentiles.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class UtteranceTimings:
    """Timestamps for a single utterance flowing through the pipeline."""

    utterance_id: int = 0
    speech_start: float = 0.0
    speech_end: float = 0.0
    stt_start: float = 0.0
    stt_end: float = 0.0
    llm_first_token: float = 0.0
    llm_done: float = 0.0
    tts_first_chunk_done: float = 0.0
    first_frame_to_client: float = 0.0
    was_interrupted: bool = False

    @property
    def voice_to_voice_latency(self) -> float:
        """End-of-speech to first audio frame delivered to client (seconds)."""
        if self.first_frame_to_client > 0 and self.speech_end > 0:
            return self.first_frame_to_client - self.speech_end
        return 0.0

    def _ms(self, start: float, end: float) -> float:
        if start > 0 and end > 0:
            return (end - start) * 1000
        return 0.0

    def log_breakdown(self) -> None:
        v2v = self.voice_to_voice_latency * 1000
        stt = self._ms(self.stt_start, self.stt_end)
        llm_first = self._ms(self.stt_end, self.llm_first_token)
        llm_total = self._ms(self.stt_end, self.llm_done)
        tts_first = self._ms(self.llm_first_token, self.tts_first_chunk_done)
        tag = " [INTERRUPTED]" if self.was_interrupted else ""

        logger.info(
            "Utterance #%d latency breakdown%s: "
            "V2V=%.0fms | STT=%.0fms | LLM_first=%.0fms | "
            "LLM_total=%.0fms | TTS_first=%.0fms",
            self.utterance_id,
            tag,
            v2v,
            stt,
            llm_first,
            llm_total,
            tts_first,
        )


class LatencyTracker:
    """Per-session latency tracker with rolling percentile computation."""

    def __init__(self, window_size: int = 100) -> None:
        self._window: deque[float] = deque(maxlen=window_size)
        self._counter = 0
        self._current: UtteranceTimings | None = None

    def new_utterance(self) -> UtteranceTimings:
        """Start tracking a new utterance. Returns the timings object to fill in."""
        self._counter += 1
        self._current = UtteranceTimings(utterance_id=self._counter)
        return self._current

    @property
    def current(self) -> UtteranceTimings | None:
        return self._current

    def finalize(self) -> None:
        """Finalize the current utterance: log breakdown and record V2V latency."""
        if self._current is None:
            return

        v2v = self._current.voice_to_voice_latency
        if v2v > 0:
            self._window.append(v2v)
        self._current.log_breakdown()

        # Log percentiles periodically
        if len(self._window) >= 5 and len(self._window) % 5 == 0:
            self._log_percentiles()

        self._current = None

    def _log_percentiles(self) -> None:
        data = sorted(self._window)
        n = len(data)
        if n < 2:
            return
        p50 = data[n // 2] * 1000
        p95 = data[min(int(n * 0.95), n - 1)] * 1000
        p99 = data[min(int(n * 0.99), n - 1)] * 1000
        logger.info(
            "V2V latency percentiles (n=%d): P50=%.0fms P95=%.0fms P99=%.0fms",
            n,
            p50,
            p95,
            p99,
        )

    def percentiles(self) -> dict[str, float]:
        """Get current percentile stats."""
        data = sorted(self._window)
        n = len(data)
        if n < 2:
            return {}
        return {
            "p50": data[n // 2] * 1000,
            "p95": data[min(int(n * 0.95), n - 1)] * 1000,
            "p99": data[min(int(n * 0.99), n - 1)] * 1000,
            "count": float(n),
        }
