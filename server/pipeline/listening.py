"""ListeningProcessor — combined VAD + STT pipeline stage.

Conforms to the AudioProcessor protocol. Resamples incoming 48kHz audio
to 16kHz, runs Silero VAD to detect speech boundaries, and triggers
Faster-Whisper transcription on end-of-speech.
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from av import AudioFrame

from server.audio.chunking import frame_to_ndarray
from server.audio.resampling import StreamResampler
from server.config import settings
from server.stt.transcriber import WhisperTranscriber
from server.vad.detector import SpeechState, VoiceActivityDetector

logger = logging.getLogger(__name__)

TranscriptCallback = Callable[[str], Awaitable[None]]


class ListeningProcessor:
    """AudioProcessor that detects speech via VAD and transcribes via Whisper.

    Always returns None from process() — no audio output in Phase 2.
    Transcripts are delivered via the on_transcript callback.
    """

    def __init__(
        self,
        on_transcript: TranscriptCallback,
        transcriber: WhisperTranscriber | None = None,
    ) -> None:
        self._on_transcript = on_transcript
        self._transcriber = transcriber
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
            if state == SpeechState.SPEECH_END and audio is not None:
                # Fire-and-forget STT task so we don't block the pipeline
                asyncio.create_task(self._handle_utterance(audio))

        # No audio output in Phase 2
        return None

    async def _handle_utterance(self, audio_16k):
        """Transcribe an utterance and deliver the result."""
        try:
            transcript = await self._transcriber.transcribe(audio_16k)
            if transcript:
                await self._on_transcript(transcript)
        except Exception:
            logger.exception("STT transcription failed")

    async def stop(self) -> None:
        logger.info("ListeningProcessor stopped")
