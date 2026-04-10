"""Faster-Whisper speech-to-text wrapper.

Loads the model once and runs inference in a thread pool executor
to avoid blocking the asyncio event loop.
"""

import asyncio
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from faster_whisper import WhisperModel

from server.config import settings


def _add_cuda_dll_path() -> None:
    """Add nvidia pip packages' bin dirs to the DLL search path on Windows."""
    try:
        import nvidia.cublas
        cublas_dir = Path(nvidia.cublas.__path__[0]) / "bin"
        if cublas_dir.is_dir():
            os.add_dll_directory(str(cublas_dir))
    except (ImportError, OSError):
        pass
    try:
        import nvidia.cudnn
        cudnn_dir = Path(nvidia.cudnn.__path__[0]) / "bin"
        if cudnn_dir.is_dir():
            os.add_dll_directory(str(cudnn_dir))
    except (ImportError, OSError):
        pass


_add_cuda_dll_path()

logger = logging.getLogger(__name__)


class WhisperTranscriber:
    """Async wrapper around faster-whisper for STT inference.

    A single instance is shared across all connections. The thread pool
    executor serializes GPU inference calls (single worker).
    """

    def __init__(
        self,
        model_size: str | None = None,
        device: str | None = None,
        compute_type: str | None = None,
    ) -> None:
        self._model_size = model_size or settings.stt_model_size
        self._device = device or settings.stt_device
        self._compute_type = compute_type or settings.stt_compute_type
        self._language = settings.stt_language
        self._min_samples = int(
            settings.stt_min_utterance_ms * settings.audio_sample_rate_stt / 1000
        )
        self._model: WhisperModel | None = None
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt")

    async def start(self) -> None:
        """Load the Whisper model (potentially slow — do at startup)."""
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(self._executor, self._load_model)

    def _load_model(self) -> None:
        t0 = time.perf_counter()
        self._model = WhisperModel(
            self._model_size,
            device=self._device,
            compute_type=self._compute_type,
        )
        dt = time.perf_counter() - t0
        logger.info(
            "Whisper model '%s' loaded on %s (%s) in %.2fs",
            self._model_size,
            self._device,
            self._compute_type,
            dt,
        )

    async def transcribe(self, audio_16k: np.ndarray) -> str | None:
        """Transcribe 16kHz int16 audio. Returns transcript or None.

        Runs in a thread pool to avoid blocking the event loop.
        Returns None for empty/too-short/noise-only results.
        """
        if len(audio_16k) < self._min_samples:
            logger.debug("STT: utterance too short (%d samples), skipping", len(audio_16k))
            return None

        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            self._executor, self._transcribe_sync, audio_16k
        )

    def _transcribe_sync(self, audio_16k: np.ndarray) -> str | None:
        """Synchronous transcription (called in thread pool)."""
        # faster-whisper expects float32 in [-1, 1]
        audio_f32 = audio_16k.astype(np.float32) / 32768.0

        t0 = time.perf_counter()
        segments, _info = self._model.transcribe(
            audio_f32,
            language=self._language,
            beam_size=5,
            vad_filter=False,  # We already did VAD
        )

        # Collect all segment texts
        texts = [seg.text for seg in segments]
        transcript = " ".join(texts).strip()

        dt = time.perf_counter() - t0
        duration = len(audio_16k) / settings.audio_sample_rate_stt

        if transcript:
            logger.info(
                "STT [%.2fs audio -> %.3fs inference]: %s", duration, dt, transcript
            )
        else:
            logger.debug("STT: empty transcript from %.2fs audio", duration)
            return None

        return transcript

    async def stop(self) -> None:
        """Shutdown the executor."""
        self._executor.shutdown(wait=False)
