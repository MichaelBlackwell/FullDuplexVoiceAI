import numpy as np
import soxr


def resample_audio(
    audio: np.ndarray,
    from_rate: int,
    to_rate: int,
) -> np.ndarray:
    """One-shot resample a complete audio buffer.

    Args:
        audio: 1-D int16 or float32 numpy array.
        from_rate: Source sample rate (e.g. 48000).
        to_rate: Target sample rate (e.g. 16000).

    Returns:
        Resampled audio array with same dtype as input.
    """
    if from_rate == to_rate:
        return audio
    return soxr.resample(audio, from_rate, to_rate)


class StreamResampler:
    """Streaming resampler for real-time chunked audio.

    Maintains internal state so chunk boundaries don't introduce artifacts.
    """

    def __init__(
        self,
        from_rate: int,
        to_rate: int,
        num_channels: int = 1,
        dtype: type = np.int16,
    ):
        self._from_rate = from_rate
        self._to_rate = to_rate
        # soxr.ResampleStream expects dtype as numpy dtype string
        self._resampler = soxr.ResampleStream(
            from_rate, to_rate, num_channels, dtype=dtype
        )

    def process_chunk(self, chunk: np.ndarray, last: bool = False) -> np.ndarray:
        """Resample a single chunk. Pass last=True for the final chunk to flush."""
        return self._resampler.resample_chunk(chunk, last=last)

    @property
    def from_rate(self) -> int:
        return self._from_rate

    @property
    def to_rate(self) -> int:
        return self._to_rate
