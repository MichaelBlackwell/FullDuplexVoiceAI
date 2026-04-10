"""Barge-in filter — confirms real interruptions vs. false positives.

Requires both a minimum speech duration AND a minimum RMS energy level
before confirming that user speech during AI output is a real barge-in.
Filters out brief coughs, background noise, and backchannel utterances.
"""

import logging
import time

import numpy as np

from server.config import settings

logger = logging.getLogger(__name__)


class BargeInFilter:
    """Stateful filter that tracks speech duration and energy since SPEECH_START."""

    def __init__(
        self,
        min_duration_ms: int | None = None,
        min_energy: float | None = None,
    ) -> None:
        self._min_duration_ms = (
            min_duration_ms
            if min_duration_ms is not None
            else settings.barge_in_min_duration_ms
        )
        self._min_energy = (
            min_energy if min_energy is not None else settings.barge_in_min_energy
        )
        self._speech_start_time: float | None = None
        self._rms_accumulator: list[float] = []
        self._triggered = False

    def on_speech_start(self) -> None:
        """Called when VAD transitions to SPEECH_START."""
        self._speech_start_time = time.perf_counter()
        self._rms_accumulator.clear()
        self._triggered = False

    def on_speech_audio(self, audio_16k: np.ndarray) -> bool:
        """Process an audio chunk during active speech.

        Returns True (once) when both duration and energy thresholds are met,
        confirming this is a real barge-in.
        """
        if self._triggered or self._speech_start_time is None:
            return False

        rms = float(np.sqrt(np.mean(audio_16k.astype(np.float32) ** 2)))
        self._rms_accumulator.append(rms)

        elapsed_ms = (time.perf_counter() - self._speech_start_time) * 1000
        avg_rms = sum(self._rms_accumulator) / len(self._rms_accumulator)

        if elapsed_ms >= self._min_duration_ms and avg_rms >= self._min_energy:
            self._triggered = True
            logger.info(
                "Barge-in confirmed: %.0fms speech, avg RMS=%.1f",
                elapsed_ms,
                avg_rms,
            )
            return True

        return False

    def reset(self) -> None:
        """Reset state after barge-in is handled or speech ends."""
        self._speech_start_time = None
        self._rms_accumulator.clear()
        self._triggered = False
