"""Unit tests for TTS synthesizer — TTSSpeaker with mocked KokoroTTS."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from av import AudioFrame

from server.tts.synthesizer import TTSSpeaker, _FRAME_SAMPLES, _WEBRTC_RATE


def _make_mock_tts(duration_seconds: float = 0.1) -> MagicMock:
    """Create a mock KokoroTTS that returns synthetic audio at 24kHz."""
    tts = MagicMock()
    sample_count = int(24000 * duration_seconds)
    # Sine wave so we can verify it's not silence
    t = np.linspace(0, duration_seconds, sample_count, endpoint=False)
    audio = (np.sin(2 * np.pi * 440 * t) * 16000).astype(np.int16)
    tts.synthesize_async = AsyncMock(return_value=audio)
    return tts


class TestTTSSpeakerOutput:
    @pytest.mark.asyncio
    async def test_enqueue_produces_frames_in_output_queue(self):
        output_queue: asyncio.Queue[AudioFrame] = asyncio.Queue(maxsize=500)
        tts = _make_mock_tts(duration_seconds=0.1)
        speaker = TTSSpeaker(tts=tts, output_queue=output_queue, voice="af_heart")
        await speaker.start()

        await speaker.enqueue("Hello world.")
        # Give worker time to process
        await asyncio.sleep(0.2)
        await speaker.stop()

        assert not output_queue.empty()
        tts.synthesize_async.assert_called_once_with("Hello world.", "af_heart")

    @pytest.mark.asyncio
    async def test_output_frames_are_48khz_20ms(self):
        output_queue: asyncio.Queue[AudioFrame] = asyncio.Queue(maxsize=500)
        tts = _make_mock_tts(duration_seconds=0.1)
        speaker = TTSSpeaker(tts=tts, output_queue=output_queue, voice="af_heart")
        await speaker.start()

        await speaker.enqueue("Test sentence.")
        await asyncio.sleep(0.2)
        await speaker.stop()

        frame = output_queue.get_nowait()
        assert frame.sample_rate == _WEBRTC_RATE
        assert frame.samples == _FRAME_SAMPLES

    @pytest.mark.asyncio
    async def test_multiple_chunks_processed_in_order(self):
        output_queue: asyncio.Queue[AudioFrame] = asyncio.Queue(maxsize=1000)
        tts = _make_mock_tts(duration_seconds=0.05)
        speaker = TTSSpeaker(tts=tts, output_queue=output_queue, voice="af_heart")
        await speaker.start()

        await speaker.enqueue("First.")
        await speaker.enqueue("Second.")
        await asyncio.sleep(0.3)
        await speaker.stop()

        assert tts.synthesize_async.call_count == 2
        calls = [c.args[0] for c in tts.synthesize_async.call_args_list]
        assert calls == ["First.", "Second."]


class TestTTSSpeakerCancel:
    @pytest.mark.asyncio
    async def test_cancel_drains_text_queue(self):
        output_queue: asyncio.Queue[AudioFrame] = asyncio.Queue(maxsize=500)
        # Slow TTS so items pile up in text queue
        tts = MagicMock()

        async def slow_synthesize(text, voice):
            await asyncio.sleep(5.0)
            return np.zeros(2400, dtype=np.int16)

        tts.synthesize_async = slow_synthesize
        speaker = TTSSpeaker(tts=tts, output_queue=output_queue, voice="af_heart")
        await speaker.start()

        # Enqueue several items — first will block on slow synthesis
        await speaker.enqueue("One.")
        await speaker.enqueue("Two.")
        await speaker.enqueue("Three.")
        await asyncio.sleep(0.05)

        speaker.cancel()

        # Text queue should be drained
        assert speaker._text_queue.empty()
        await speaker.stop()

    @pytest.mark.asyncio
    async def test_cancel_drains_output_queue(self):
        output_queue: asyncio.Queue[AudioFrame] = asyncio.Queue(maxsize=500)
        tts = _make_mock_tts(duration_seconds=0.1)
        speaker = TTSSpeaker(tts=tts, output_queue=output_queue, voice="af_heart")
        await speaker.start()

        await speaker.enqueue("Some text.")
        await asyncio.sleep(0.2)
        # Output queue should have frames now
        assert not output_queue.empty()

        speaker.cancel()
        assert output_queue.empty()
        await speaker.stop()

    @pytest.mark.asyncio
    async def test_cancel_allows_new_speech(self):
        """After cancel, new enqueued text should still be processed."""
        output_queue: asyncio.Queue[AudioFrame] = asyncio.Queue(maxsize=500)
        tts = _make_mock_tts(duration_seconds=0.05)
        speaker = TTSSpeaker(tts=tts, output_queue=output_queue, voice="af_heart")
        await speaker.start()

        await speaker.enqueue("Old speech.")
        await asyncio.sleep(0.15)
        speaker.cancel()

        # Enqueue new text after cancel
        await speaker.enqueue("New speech.")
        await asyncio.sleep(0.15)
        await speaker.stop()

        # Should have processed the new text
        assert tts.synthesize_async.call_count == 2


class TestTTSSpeakerLifecycle:
    @pytest.mark.asyncio
    async def test_stop_exits_cleanly(self):
        output_queue: asyncio.Queue[AudioFrame] = asyncio.Queue(maxsize=500)
        tts = _make_mock_tts()
        speaker = TTSSpeaker(tts=tts, output_queue=output_queue)
        await speaker.start()
        await speaker.stop()
        assert speaker._synth_task.done()
        assert speaker._push_task.done()

    @pytest.mark.asyncio
    async def test_empty_synthesis_result_skipped(self):
        output_queue: asyncio.Queue[AudioFrame] = asyncio.Queue(maxsize=500)
        tts = MagicMock()
        tts.synthesize_async = AsyncMock(return_value=np.array([], dtype=np.int16))
        speaker = TTSSpeaker(tts=tts, output_queue=output_queue)
        await speaker.start()

        await speaker.enqueue("Empty result.")
        await asyncio.sleep(0.1)
        await speaker.stop()

        assert output_queue.empty()

    @pytest.mark.asyncio
    async def test_synthesis_exception_does_not_crash_worker(self):
        output_queue: asyncio.Queue[AudioFrame] = asyncio.Queue(maxsize=500)
        tts = MagicMock()
        call_count = 0

        async def failing_then_ok(text, voice):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Kokoro crashed")
            return np.zeros(2400, dtype=np.int16)

        tts.synthesize_async = failing_then_ok
        speaker = TTSSpeaker(tts=tts, output_queue=output_queue)
        await speaker.start()

        await speaker.enqueue("Fail.")
        await speaker.enqueue("Succeed.")
        await asyncio.sleep(0.2)
        await speaker.stop()

        # Worker should have survived the first failure and processed the second
        assert call_count == 2
        assert not output_queue.empty()
