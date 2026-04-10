import asyncio
import fractions
import time

from aiortc import MediaStreamTrack
from av import AudioFrame


class OutputAudioTrack(MediaStreamTrack):
    """Custom audio track that reads processed frames from an asyncio.Queue.

    Manages timestamps and real-time pacing so frames are delivered
    at the correct rate for Opus encoding and RTP packetization.
    """

    kind = "audio"

    def __init__(self, queue: asyncio.Queue[AudioFrame]):
        super().__init__()
        self._queue = queue
        self._start: float | None = None
        self._timestamp = 0

    async def recv(self) -> AudioFrame:
        frame = await self._queue.get()

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
