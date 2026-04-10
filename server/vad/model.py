"""Silero VAD ONNX model wrapper.

Downloads and caches the Silero VAD ONNX model (~2MB), then provides
a per-connection inference wrapper that manages its own hidden state
and context window.
"""

import logging
import urllib.request
from pathlib import Path

import numpy as np
import onnxruntime as ort

logger = logging.getLogger(__name__)

_MODEL_URL = (
    "https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx"
)
_MODEL_DIR = Path(__file__).resolve().parent / "data"
_MODEL_PATH = _MODEL_DIR / "silero_vad.onnx"


def _ensure_model() -> Path:
    """Download the Silero VAD ONNX model if not already cached."""
    if _MODEL_PATH.exists():
        return _MODEL_PATH
    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading Silero VAD ONNX model to %s ...", _MODEL_PATH)
    urllib.request.urlretrieve(_MODEL_URL, _MODEL_PATH)
    logger.info("Silero VAD model downloaded (%.1f KB)", _MODEL_PATH.stat().st_size / 1024)
    return _MODEL_PATH


class SileroVAD:
    """Per-connection Silero VAD inference wrapper using ONNX Runtime.

    Each instance has its own ONNX session, hidden state, and context
    window so multiple connections don't interfere with each other.
    """

    # Context size that gets prepended to each chunk (required by the model)
    _CONTEXT_SIZE = 64  # for 16kHz; would be 32 for 8kHz

    def __init__(self, sample_rate: int = 16000) -> None:
        model_path = _ensure_model()
        opts = ort.SessionOptions()
        opts.inter_op_num_threads = 1
        opts.intra_op_num_threads = 1
        self._session = ort.InferenceSession(
            str(model_path), sess_options=opts, providers=["CPUExecutionProvider"]
        )
        self._sample_rate = sample_rate
        self._sr = np.array(sample_rate, dtype=np.int64)
        self.reset()

    def reset(self) -> None:
        """Reset hidden state and context for a new audio stream."""
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, self._CONTEXT_SIZE), dtype=np.float32)

    def __call__(self, audio_chunk: np.ndarray) -> float:
        """Run VAD on a single chunk. Returns speech probability 0.0–1.0.

        Args:
            audio_chunk: 1-D int16 numpy array, 512 samples (32ms at 16kHz).
        """
        # Silero ONNX expects float32 in [-1, 1]
        audio_f32 = audio_chunk.astype(np.float32) / 32768.0
        audio_f32 = audio_f32[np.newaxis, :]  # shape: (1, 512)

        # Prepend context window (last 64 samples from previous chunk)
        x = np.concatenate([self._context, audio_f32], axis=1)  # shape: (1, 576)

        ort_inputs = {
            "input": x,
            "state": self._state,
            "sr": self._sr,
        }
        output, self._state = self._session.run(None, ort_inputs)

        # Save last 64 samples as context for next call
        self._context = x[:, -self._CONTEXT_SIZE:]

        return float(output[0][0])
