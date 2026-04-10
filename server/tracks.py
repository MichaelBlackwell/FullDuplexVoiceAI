import asyncio
import fractions
import time

import numpy as np
from aiortc import MediaStreamTrack
from av import AudioFrame

from server.audio.chunking import ndarray_to_frame
from server.config import settings

# Pre-compute silence frame parameters
_SAMPLE_RATE = settings.audio_sample_rate_webrtc  # 48000
_FRAME_SAMPLES = _SAMPLE_RATE * settings.audio_ptime_ms // 1000  # 960
_FRAME_DURATION = settings.audio_ptime_ms / 1000  # 0.02s
_SILENCE = np.zeros(_FRAME_SAMPLES, dtype=np.int16)


class OutputAudioTrack(MediaStreamTrack):
    """Custom audio track that reads processed frames from an asyncio.Queue.

    Manages timestamps and real-time pacing so frames are delivered
    at the correct rate for Opus encoding and RTP packetization.
    When the queue is empty, emits silence frames to keep the RTP
    stream alive and prevent WebRTC track drops.
    """

    kind = "audio"

    def __init__(self, queue: asyncio.Queue[AudioFrame]):
        super().__init__()
        self._queue = queue
        self._start: float | None = None
        self._timestamp = 0

    async def recv(self) -> AudioFrame:
        try:
            frame = await asyncio.wait_for(
                self._queue.get(), timeout=_FRAME_DURATION
            )
        except (asyncio.TimeoutError, TimeoutError):
            # No audio available — emit a silence frame to keep RTP alive
            frame = ndarray_to_frame(_SILENCE, sample_rate=_SAMPLE_RATE)

        # Set monotonically increasing timestamps
        frame.pts = self._timestamp
        frame.time_base = fractions.Fraction(1, frame.sample_rate)
        self._timestamp += frame.samples

        # Real-time pacing: sleep to match wall-clock time
        # This prevents burst delivery which would cause client-side jitter
        if self._start is None:
            self._start = time.time()

        target_time = self._start + (self._timestamp / frame.sample_rate)
        wait = target_time - time.time()
        if wait > 0:
            await asyncio.sleep(wait)

        return frame
