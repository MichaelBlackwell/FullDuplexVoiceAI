import numpy as np
import pytest

from server.audio.resampling import StreamResampler, resample_audio


def test_resample_same_rate():
    audio = np.ones(960, dtype=np.int16)
    result = resample_audio(audio, 48000, 48000)
    np.testing.assert_array_equal(result, audio)


def test_resample_48k_to_16k():
    # 960 samples at 48kHz = 20ms -> should produce ~320 samples at 16kHz
    audio = np.random.randint(-1000, 1000, size=960, dtype=np.int16).astype(np.float32)
    result = resample_audio(audio, 48000, 16000)
    assert len(result) == 320


def test_resample_16k_to_48k():
    audio = np.random.randint(-1000, 1000, size=320, dtype=np.int16).astype(np.float32)
    result = resample_audio(audio, 16000, 48000)
    assert len(result) == 960


def test_stream_resampler_consistency():
    """Streaming resampler should produce similar output to one-shot."""
    np.random.seed(42)
    audio = np.random.randint(-1000, 1000, size=4800, dtype=np.int16)

    # One-shot
    one_shot = resample_audio(audio.astype(np.float32), 48000, 16000)

    # Streaming in chunks of 960
    resampler = StreamResampler(48000, 16000, dtype=np.int16)
    chunks = [audio[i : i + 960] for i in range(0, len(audio), 960)]
    streamed_parts = []
    for i, chunk in enumerate(chunks):
        is_last = i == len(chunks) - 1
        streamed_parts.append(resampler.process_chunk(chunk, last=is_last))
    streamed = np.concatenate(streamed_parts)

    # Lengths should match (or be very close)
    assert abs(len(streamed) - len(one_shot)) <= 2
