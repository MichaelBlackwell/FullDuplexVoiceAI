import asyncio

import numpy as np
import pytest
from av import AudioFrame

from server.audio.buffer import AudioBufferManager
from server.audio.chunking import ndarray_to_frame


def _make_frame(value: int = 0) -> AudioFrame:
    audio = np.full(960, value, dtype=np.int16)
    return ndarray_to_frame(audio)


@pytest.mark.asyncio
async def test_put_and_get():
    buf = AudioBufferManager(maxsize=10)
    frame = _make_frame(42)
    await buf.put(frame)
    assert buf.size == 1
    result = await buf.get()
    assert buf.size == 0
    assert result is frame


@pytest.mark.asyncio
async def test_backpressure_drops_oldest():
    buf = AudioBufferManager(maxsize=3)
    frames = [_make_frame(i) for i in range(4)]

    for f in frames[:3]:
        await buf.put(f)
    assert buf.size == 3

    # This should drop the oldest (value=0) and add value=3
    await buf.put(frames[3])
    assert buf.size == 3

    # First frame out should be value=1 (0 was dropped)
    first = await buf.get()
    arr = first.to_ndarray().flatten()
    assert arr[0] == 1


@pytest.mark.asyncio
async def test_clear():
    buf = AudioBufferManager(maxsize=10)
    for i in range(5):
        await buf.put(_make_frame(i))
    assert buf.size == 5

    count = buf.clear()
    assert count == 5
    assert buf.empty
