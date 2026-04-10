import numpy as np
import pytest
from av import AudioFrame

from server.audio.chunking import (
    chunk_duration_samples,
    frame_to_ndarray,
    ndarray_to_frame,
    split_into_chunks,
)


def test_chunk_duration_samples():
    assert chunk_duration_samples(48000, 20) == 960
    assert chunk_duration_samples(16000, 30) == 480
    assert chunk_duration_samples(24000, 20) == 480


def test_ndarray_to_frame_and_back():
    original = np.arange(960, dtype=np.int16)
    frame = ndarray_to_frame(original, sample_rate=48000)

    assert frame.sample_rate == 48000
    assert frame.samples == 960

    recovered = frame_to_ndarray(frame)
    np.testing.assert_array_equal(recovered, original)


def test_split_into_chunks_exact():
    audio = np.zeros(960, dtype=np.int16)
    chunks = split_into_chunks(audio, 480)
    assert len(chunks) == 2
    assert all(len(c) == 480 for c in chunks)


def test_split_into_chunks_remainder():
    audio = np.zeros(1000, dtype=np.int16)
    chunks = split_into_chunks(audio, 480)
    assert len(chunks) == 3
    assert len(chunks[0]) == 480
    assert len(chunks[1]) == 480
    assert len(chunks[2]) == 40
