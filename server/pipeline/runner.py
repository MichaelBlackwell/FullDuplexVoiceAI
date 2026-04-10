import asyncio
import logging
from contextlib import suppress

from aiortc import MediaStreamTrack
from aiortc.mediastreams import MediaStreamError

from server.audio.processing import AudioProcessor
from server.config import settings
from server.tracks import OutputAudioTrack

logger = logging.getLogger(__name__)


class PipelineRunner:
    """Connects an incoming audio track to an output track through processors.

    The runner reads frames from the input track, passes them through
    each processor in order, and pushes results to the output queue.
    """

    def __init__(
        self,
        input_track: MediaStreamTrack,
        processors: list[AudioProcessor],
        buffer_max_size: int | None = None,
    ):
        self._input_track = input_track
        self._processors = processors
        max_size = buffer_max_size or settings.buffer_max_size
        self._output_queue: asyncio.Queue = asyncio.Queue(maxsize=max_size)
        self._output_track = OutputAudioTrack(self._output_queue)
        self._task: asyncio.Task | None = None

    @property
    def output_track(self) -> OutputAudioTrack:
        return self._output_track

    async def run(self) -> None:
        """Main processing loop."""
        for proc in self._processors:
            await proc.start()
        logger.info("Pipeline started with %d processor(s)", len(self._processors))

        try:
            while True:
                frame = await self._input_track.recv()

                # Pass through each processor
                for proc in self._processors:
                    frame = await proc.process(frame)
                    if frame is None:
                        break

                if frame is not None:
                    # Push to output queue with backpressure
                    if self._output_queue.full():
                        try:
                            self._output_queue.get_nowait()
                        except asyncio.QueueEmpty:
                            pass
                    self._output_queue.put_nowait(frame)

        except MediaStreamError:
            logger.info("Input track ended")
        except asyncio.CancelledError:
            logger.info("Pipeline cancelled")
        finally:
            for proc in self._processors:
                await proc.stop()
            self._output_track.stop()
            logger.info("Pipeline stopped")

    async def start(self) -> asyncio.Task:
        """Start the pipeline as a background task."""
        self._task = asyncio.create_task(self.run())
        return self._task

    async def stop(self) -> None:
        """Cancel the pipeline task."""
        if self._task and not self._task.done():
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
