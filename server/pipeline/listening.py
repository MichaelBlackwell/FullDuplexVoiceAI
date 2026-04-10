"""ListeningProcessor — combined VAD + STT pipeline stage.

Conforms to the AudioProcessor protocol. Resamples incoming 48kHz audio
to 16kHz, runs Silero VAD to detect speech boundaries, and triggers
Faster-Whisper transcription on end-of-speech.
"""

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable

import numpy as np
from av import AudioFrame

from server.audio.chunking import frame_to_ndarray
from server.audio.resampling import StreamResampler
from server.config import settings
from server.metrics import LatencyTracker
from server.stt.transcriber import WhisperTranscriber
from server.vad.detector import SpeechState, VoiceActivityDetector

logger = logging.getLogger(__name__)

TranscriptCallback = Callable[[str], Awaitable[None]]
SpeechStartCallback = Callable[[], Awaitable[None]]
SpeechAudioCallback = Callable[[np.ndarray], None]  # sync


class ListeningProcessor:
    """AudioProcessor that detects speech via VAD and transcribes via Whisper.

    Always returns None from process() — no audio output in Phase 2.
    Transcripts are delivered via the on_transcript callback.
    """

    def __init__(
        self,
        on_transcript: TranscriptCallback,
        transcriber: WhisperTranscriber | None = None,
        on_speech_start: SpeechStartCallback | None = None,
        on_speech_audio: SpeechAudioCallback | None = None,
        latency_tracker: LatencyTracker | None = None,
    ) -> None:
        self._on_transcript = on_transcript
        self._transcriber = transcriber
        self._on_speech_start = on_speech_start
        self._on_speech_audio = on_speech_audio
        self._latency_tracker = latency_tracker
        self._resampler: StreamResampler | None = None
        self._detector: VoiceActivityDetector | None = None

    async def start(self) -> None:
        self._resampler = StreamResampler(
            from_rate=settings.audio_sample_rate_webrtc,
            to_rate=settings.audio_sample_rate_stt,
        )
        self._detector = VoiceActivityDetector()

        # Ensure Whisper model is loaded
        if self._transcriber._model is None:
            await self._transcriber.start()

        logger.info("ListeningProcessor started")

    async def process(self, frame: AudioFrame) -> AudioFrame | None:
        """Process incoming 48kHz audio frame through VAD pipeline."""
        # Convert to numpy int16
        audio_48k = frame_to_ndarray(frame)

        # Resample 48kHz → 16kHz
        audio_16k = self._resampler.process_chunk(audio_48k)

        # Feed to VAD detector
        events = self._detector.feed(audio_16k)

        # Handle state-change events
        for state, audio in events:
            if state == SpeechState.SPEECH_START:
                # Record speech start for latency tracking
                if self._latency_tracker:
                    timings = self._latency_tracker.new_utterance()
                    timings.speech_start = time.perf_counter()
                # Notify barge-in system
                if self._on_speech_start:
                    asyncio.create_task(self._on_speech_start())
            elif state == SpeechState.SPEECH_END and audio is not None:
                # Record speech end for latency tracking
                if self._latency_tracker and self._latency_tracker.current:
                    self._latency_tracker.current.speech_end = time.perf_counter()
                # Fire-and-forget STT task so we don't block the pipeline
                asyncio.create_task(self._handle_utterance(audio))

        # Forward audio chunks during active speech for barge-in energy tracking
        if (
            self._on_speech_audio
            and self._detector.state == SpeechState.SPEAKING
        ):
            self._on_speech_audio(audio_16k)

        # No audio output in Phase 2
        return None

    async def _handle_utterance(self, audio_16k):
        """Transcribe an utterance and deliver the result."""
        try:
            # Record STT start time
            if self._latency_tracker and self._latency_tracker.current:
                self._latency_tracker.current.stt_start = time.perf_counter()

            transcript = await self._transcriber.transcribe(audio_16k)

            # Record STT end time
            if self._latency_tracker and self._latency_tracker.current:
                self._latency_tracker.current.stt_end = time.perf_counter()

            if transcript:
                await self._on_transcript(transcript)
        except Exception:
            logger.exception("STT transcription failed")

    async def stop(self) -> None:
        logger.info("ListeningProcessor stopped")
