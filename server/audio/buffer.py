import asyncio
import logging

from av import AudioFrame

logger = logging.getLogger(__name__)


class AudioBufferManager:
    """Async audio frame buffer with backpressure and flush support.

    When the buffer is full, the oldest frame is dropped to prevent
    unbounded memory growth. This trades a brief audio skip for
    real-time responsiveness.
    """

    def __init__(self, maxsize: int = 50):
        self._queue: asyncio.Queue[AudioFrame] = asyncio.Queue(maxsize=maxsize)

    async def put(self, frame: AudioFrame) -> None:
        """Add a frame, dropping the oldest if full."""
        if self._queue.full():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                pass
            logger.debug("Buffer full — dropped oldest frame")
        self._queue.put_nowait(frame)

    async def get(self) -> AudioFrame:
        """Get the next frame, blocking until one is available."""
        return await self._queue.get()

    def get_nowait(self) -> AudioFrame:
        """Get a frame without waiting. Raises asyncio.QueueEmpty if empty."""
        return self._queue.get_nowait()

    def clear(self) -> int:
        """Flush all buffered frames. Returns number of frames discarded."""
        count = 0
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
                count += 1
            except asyncio.QueueEmpty:
                break
        return count

    @property
    def size(self) -> int:
        return self._queue.qsize()

    @property
    def empty(self) -> bool:
        return self._queue.empty()

    @property
    def full(self) -> bool:
        return self._queue.full()
