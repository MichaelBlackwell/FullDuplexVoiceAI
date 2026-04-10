# Full-Duplex Voice AI

Real-time low-latency voice AI server with speech detection and transcription. Speak into your browser and see live transcriptions powered by Silero VAD and Faster-Whisper.

## Current Status

**Phase 1 (Foundation)** and **Phase 2 (Listening)** are complete. The system captures audio from the browser via WebRTC, detects speech using Silero VAD, and transcribes it with Faster-Whisper on GPU. Transcripts are displayed in the browser in real time via WebSocket.

Phases 3-6 (LLM integration, TTS, interruption handling, production hardening) are planned. See [full-duplex-voice-ai-design-doc.md](full-duplex-voice-ai-design-doc.md) for the complete roadmap.

## Architecture

```
Browser Mic
    |
    v
 WebRTC (Opus 48kHz)
    |
    v
 aiortc decode -> PCM int16, stereo -> mono extraction
    |
    v
 Resample 48kHz -> 16kHz (soxr streaming)
    |
    v
 Silero VAD (ONNX, CPU, 32ms chunks)
    |  detects SPEECH_START / SPEECH_END
    v
 Faster-Whisper (CTranslate2, CUDA, int8)
    |  transcribes buffered utterance
    v
 WebSocket -> Browser UI
```

## Performance

Measured on RTX 4060 with `medium` model, int8 quantization:

| Stage | Latency |
|-------|---------|
| VAD silence threshold | 300ms |
| Faster-Whisper inference (warmup) | ~800ms |
| Faster-Whisper inference (subsequent) | 200-420ms |
| **End-of-speech to transcript** | **~500-700ms** |

Target was <500ms for STT inference alone -- achieved at 200-420ms after GPU warmup.

## Prerequisites

- Python >= 3.11
- NVIDIA GPU with CUDA support (RTX 3060 or better recommended)
- CUDA Toolkit 12.x (for cuBLAS)
- [uv](https://docs.astral.sh/uv/) package manager

## Quick Start

```bash
# Clone and enter the project
cd FullDuplexVoiceAI

# Install dependencies
uv sync

# Start the server
python -m server.main
```

Open `http://127.0.0.1:8080` in your browser, click **Connect**, and speak. Transcripts appear in real time.

The Silero VAD ONNX model (~2MB) is downloaded automatically on first run. The Faster-Whisper model (~1.5GB for medium) is downloaded from HuggingFace on first run.

## Project Structure

```
server/
  main.py              Entry point (uvicorn on port 8080)
  config.py            Settings (sample rates, VAD/STT params)
  signaling.py         FastAPI app (/offer, /ws/transcripts, /health)
  connection.py        WebRTC peer connection + pipeline setup
  tracks.py            OutputAudioTrack (real-time pacing)
  audio/
    processing.py      AudioProcessor protocol, EchoProcessor
    buffer.py          AsyncQueue-based frame buffer
    chunking.py        PCM frame conversion (stereo->mono, float->int16)
    resampling.py      Streaming resampler (soxr)
  pipeline/
    runner.py          PipelineRunner (input track -> processors -> output)
    listening.py       ListeningProcessor (VAD + STT combined)
  vad/
    model.py           Silero VAD ONNX wrapper (per-connection state)
    detector.py        VoiceActivityDetector (state machine + ring buffer)
  stt/
    transcriber.py     WhisperTranscriber (thread pool, GPU inference)
client/
  index.html           Browser UI (dark theme, transcript display)
  client.js            WebRTC + WebSocket client
tests/
  test_buffer.py       AudioBufferManager tests
  test_chunking.py     Frame conversion tests
  test_resampling.py   Resampling tests
  test_vad_detector.py VAD state machine tests (mocked + integration)
```

## Configuration

All settings are in [server/config.py](server/config.py):

| Setting | Default | Description |
|---------|---------|-------------|
| `stt_model_size` | `"medium"` | Whisper model (`"small"` for faster, `"medium"` for accuracy) |
| `stt_device` | `"cuda"` | Inference device (`"cuda"` or `"cpu"`) |
| `stt_compute_type` | `"int8"` | Quantization (int8 for speed on GPU) |
| `vad_threshold` | `0.5` | Speech probability threshold (0.0-1.0) |
| `vad_silence_duration_ms` | `300` | Silence duration before end-of-speech |
| `vad_pre_speech_ms` | `300` | Pre-speech ring buffer to capture utterance start |
| `stt_language` | `"en"` | Language hint for Whisper |

## Running Tests

```bash
uv sync --extra dev
python -m pytest tests/ -v
```

## Key Design Decisions

- **Silero VAD via ONNX Runtime** (not PyTorch) -- avoids a ~2GB torch dependency. Each connection gets its own ONNX session with isolated hidden state.
- **Combined ListeningProcessor** -- VAD and STT are tightly coupled in one AudioProcessor. VAD accumulates audio and triggers STT on end-of-speech via `asyncio.create_task`.
- **Shared WhisperTranscriber singleton** -- one model instance across all connections. Single-worker ThreadPoolExecutor serializes GPU inference.
- **WebSocket for transcripts** -- more reliable than WebRTC DataChannel for text delivery. DataChannel requires SCTP negotiation that can fail with aiortc.
- **Streaming resampler** -- soxr `ResampleStream` maintains state across chunk boundaries for artifact-free 48kHz to 16kHz conversion.
- **32ms VAD chunks** -- Silero VAD at 16kHz requires exactly 512 samples per chunk. A chunk accumulator handles the 20ms WebRTC frame to 32ms VAD chunk mismatch.
- **64-sample context window** -- Silero's ONNX model requires prepending 64 samples of context from the previous chunk for correct inference.
