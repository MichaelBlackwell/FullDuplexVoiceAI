import numpy as np
from av import AudioFrame


def frame_to_ndarray(frame: AudioFrame) -> np.ndarray:
    """Convert an av.AudioFrame to a 1-D mono numpy int16 array.

    Handles both planar (s16p, fltp) and interleaved (s16) formats,
    and downmixes stereo to mono by taking the first channel.
    """
    channels = len(frame.layout.channels)
    arr = frame.to_ndarray()  # int16 or float32 depending on format

    if channels > 1:
        if frame.format.is_planar:
            # Shape: (channels, samples) — take first channel
            arr = arr[0]
        else:
            # Interleaved shape: (1, samples*channels) — de-interleave
            arr = arr.flatten()
            arr = arr[::channels]  # take every Nth sample (first channel)

    arr = arr.flatten()

    # Convert float formats (fltp, flt) to int16
    if arr.dtype in (np.float32, np.float64):
        arr = (arr * 32768).clip(-32768, 32767).astype(np.int16)
    else:
        arr = arr.astype(np.int16)

    return arr


def ndarray_to_frame(
    audio: np.ndarray,
    sample_rate: int = 48000,
    layout: str = "mono",
) -> AudioFrame:
    """Convert a numpy int16 array to an av.AudioFrame."""
    audio = audio.astype(np.int16)
    frame = AudioFrame(format="s16", layout=layout, samples=len(audio))
    frame.sample_rate = sample_rate
    frame.planes[0].update(audio.tobytes())
    return frame


def split_into_chunks(audio: np.ndarray, chunk_samples: int) -> list[np.ndarray]:
    """Split a PCM array into fixed-size chunks. Last chunk may be shorter."""
    return [audio[i : i + chunk_samples] for i in range(0, len(audio), chunk_samples)]


def chunk_duration_samples(sample_rate: int, duration_ms: int = 20) -> int:
    """Calculate number of samples for a given duration."""
    return sample_rate * duration_ms // 1000
