from typing import Protocol, runtime_checkable

from av import AudioFrame


@runtime_checkable
class AudioProcessor(Protocol):
    """Interface for audio processing stages in the pipeline.

    Implement this to create custom processors (VAD, STT, etc.).
    """

    async def process(self, frame: AudioFrame) -> AudioFrame | None:
        """Process an audio frame. Return None to drop the frame."""
        ...

    async def start(self) -> None:
        """Called when the pipeline starts."""
        ...

    async def stop(self) -> None:
        """Called when the pipeline stops."""
        ...


class EchoProcessor:
    """Pass-through processor that echoes audio unchanged.

    Used in Phase 1 for the echo test. Will be replaced by the
    real pipeline (VAD -> STT -> LLM -> TTS) in later phases.
    """

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        pass

    async def process(self, frame: AudioFrame) -> AudioFrame:
        return frame
