"""Kokoro TTS integration — shared engine + per-session speaker.

KokoroTTS is a singleton that loads the Kokoro model once and provides
async synthesis via a thread pool. TTSSpeaker is a per-session worker
that pulls text chunks from a queue, synthesizes audio, resamples to
48kHz, and pushes AudioFrames to the WebRTC output queue.
"""

import asyncio
import logging
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from av import AudioFrame

from server.audio.chunking import (
    chunk_duration_samples,
    ndarray_to_frame,
    split_into_chunks,
)
from server.audio.resampling import resample_audio
from server.config import settings

logger = logging.getLogger(__name__)

_WEBRTC_RATE = settings.audio_sample_rate_webrtc  # 48000
_TTS_RATE = settings.audio_sample_rate_tts  # 24000
_FRAME_MS = settings.audio_ptime_ms  # 20
_FRAME_SAMPLES = chunk_duration_samples(_WEBRTC_RATE, _FRAME_MS)  # 960


class KokoroTTS:
    """Shared Kokoro TTS engine. Thread-safe for concurrent sessions
    because inference is serialized through a single-worker thread pool.
    """

    def __init__(self) -> None:
        self._pipeline = None
        self._executor = ThreadPoolExecutor(max_workers=1)

    async def start(self) -> None:
        """Load the Kokoro pipeline in the thread pool (can be slow)."""
        loop = asyncio.get_running_loop()
        self._pipeline = await loop.run_in_executor(
            self._executor, self._load_pipeline
        )
        logger.info(
            "KokoroTTS started: lang=%s, device=%s",
            settings.tts_language,
            settings.tts_device,
        )

    @staticmethod
    def _load_pipeline():
        from kokoro import KPipeline

        return KPipeline(lang_code=settings.tts_language, device=settings.tts_device)

    def _synthesize_sync(self, text: str, voice: str) -> np.ndarray:
        """Run Kokoro synthesis (CPU-bound, called in thread pool).

        Returns int16 ndarray at 24kHz.
        """
        chunks = []
        for _graphemes, _phonemes, audio in self._pipeline(text, voice=voice):
            if audio is not None:
                chunks.append(audio)

        if not chunks:
            return np.array([], dtype=np.int16)

        # Kokoro outputs float32 in [-1, 1] — concatenate then convert
        audio_f32 = np.concatenate(chunks)
        audio_i16 = (audio_f32 * 32767).clip(-32768, 32767).astype(np.int16)
        return audio_i16

    async def synthesize_async(self, text: str, voice: str) -> np.ndarray:
        """Async wrapper — dispatches synthesis to thread pool.

        Returns int16 ndarray at 24kHz.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            self._executor, self._synthesize_sync, text, voice
        )

    async def stop(self) -> None:
        self._executor.shutdown(wait=False)
        self._pipeline = None
        logger.info("KokoroTTS stopped")


_DONE_SENTINEL = object()  # Signals end of a response cycle (not end of speaker)


class TTSSpeaker:
    """Per-session TTS speaker with two-stage pipeline.

    Stage 1 (synthesizer): pulls text chunks, runs Kokoro synthesis,
    resamples to 48kHz, and pushes AudioFrames to an intermediate queue.
    This runs ahead of playback so the next sentence is ready before
    the current one finishes.

    Stage 2 (pusher): pulls frames from the intermediate queue and
    pushes them to the WebRTC output queue at real-time pace via
    blocking put().
    """

    def __init__(
        self,
        tts: KokoroTTS,
        output_queue: asyncio.Queue[AudioFrame],
        voice: str | None = None,
        on_first_frame: Callable[[], None] | None = None,
        on_playback_done: Callable[[], None] | None = None,
    ) -> None:
        self._tts = tts
        self._output_queue = output_queue
        self._voice = voice or settings.tts_voice
        self._on_first_frame = on_first_frame
        self._on_playback_done = on_playback_done
        self._text_queue: asyncio.Queue[str | None] = asyncio.Queue()
        # Intermediate buffer between synthesis and playback.
        # Large enough to hold a few sentences worth of pre-synthesized frames.
        self._frame_queue: asyncio.Queue[AudioFrame | None] = asyncio.Queue(
            maxsize=500
        )
        self._cancel_event = asyncio.Event()
        self._synth_task: asyncio.Task | None = None
        self._push_task: asyncio.Task | None = None
        self._first_frame_fired = False

    async def start(self) -> None:
        self._synth_task = asyncio.create_task(self._synthesizer())
        self._push_task = asyncio.create_task(self._pusher())

    async def enqueue(self, text: str) -> None:
        """Add a text chunk for synthesis."""
        await self._text_queue.put(text)

    async def enqueue_done(self) -> None:
        """Signal that no more text is coming for this response cycle.

        A done sentinel flows through both pipeline stages. When the pusher
        processes it (after all preceding audio frames), it fires the
        on_playback_done callback — the correct time to mark the session idle.
        """
        await self._text_queue.put(_DONE_SENTINEL)

    def cancel(self) -> None:
        """Interrupt current speech: clear all queues, abort synthesis."""
        self._cancel_event.set()
        # Drain text queue
        _drain(self._text_queue)
        # Drain intermediate frame queue (unblocks synthesizer if blocked on put)
        _drain(self._frame_queue)
        # Drain output audio queue (unblocks pusher if blocked on put)
        _drain(self._output_queue)
        # Reset first-frame tracking for next response cycle
        self._first_frame_fired = False

    def reset_first_frame(self) -> None:
        """Reset first-frame tracking for a new response cycle."""
        self._first_frame_fired = False

    async def _synthesizer(self) -> None:
        """Stage 1: text → synthesis → resampled frames into intermediate queue."""
        while True:
            text = await self._text_queue.get()
            if text is None:
                # Signal pusher to stop entirely
                await self._frame_queue.put(None)
                break
            if text is _DONE_SENTINEL:
                # Forward done sentinel to pusher (don't stop the task)
                await self._frame_queue.put(_DONE_SENTINEL)
                continue

            self._cancel_event.clear()

            try:
                audio_24k = await self._tts.synthesize_async(text, self._voice)
                if self._cancel_event.is_set() or len(audio_24k) == 0:
                    continue

                audio_48k = resample_audio(audio_24k, _TTS_RATE, _WEBRTC_RATE)

                chunks = split_into_chunks(audio_48k, _FRAME_SAMPLES)
                for chunk in chunks:
                    if self._cancel_event.is_set():
                        break
                    if len(chunk) < _FRAME_SAMPLES:
                        chunk = np.pad(chunk, (0, _FRAME_SAMPLES - len(chunk)))
                    frame = ndarray_to_frame(chunk, sample_rate=_WEBRTC_RATE)
                    await self._frame_queue.put(frame)

            except Exception:
                logger.exception("TTS synthesis failed for chunk: %s", text[:80])

    async def _pusher(self) -> None:
        """Stage 2: pull pre-synthesized frames and push to WebRTC output at real-time pace."""
        while True:
            frame = await self._frame_queue.get()
            if frame is None:
                break
            if frame is _DONE_SENTINEL:
                # All audio for this response has been pushed to the output queue
                if self._on_playback_done:
                    try:
                        self._on_playback_done()
                    except Exception:
                        logger.exception("on_playback_done callback failed")
                continue
            if self._cancel_event.is_set():
                continue
            await self._output_queue.put(frame)

            # Fire on_first_frame callback once per response cycle
            if not self._first_frame_fired and self._on_first_frame:
                self._first_frame_fired = True
                try:
                    self._on_first_frame()
                except Exception:
                    logger.exception("on_first_frame callback failed")

    async def stop(self) -> None:
        """Signal both tasks to exit and wait for them."""
        await self._text_queue.put(None)
        for task in (self._synth_task, self._push_task):
            if task:
                try:
                    await asyncio.wait_for(task, timeout=5.0)
                except asyncio.TimeoutError:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
        logger.info("TTSSpeaker stopped")


def _drain(queue: asyncio.Queue) -> None:
    """Drain all items from a queue without blocking."""
    while not queue.empty():
        try:
            queue.get_nowait()
        except asyncio.QueueEmpty:
            break
