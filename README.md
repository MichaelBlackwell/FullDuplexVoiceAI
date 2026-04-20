# Full-Duplex Voice AI

Real-time low-latency voice AI server. Speak into your browser and have a voice conversation with an AI — it listens, thinks, and speaks back. Powered by Silero VAD, Faster-Whisper, configurable LLM (DeepSeek/OpenAI), and Kokoro TTS.

## Current Status

**Phases 1–5 complete.** The full voice pipeline is working with interruption support: browser mic → WebRTC → VAD → Faster-Whisper STT → LLM streaming → Kokoro TTS → WebRTC → speaker. You can have a natural full-duplex voice conversation — interrupt the AI mid-sentence and it stops, listens, and responds to your interruption.

Phase 6 (production hardening) is next. See [full-duplex-voice-ai-design-doc.md](full-duplex-voice-ai-design-doc.md) for the complete roadmap.

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
 LLM Chat Completions (streaming, configurable provider)
    |  streams tokens, chunked at sentence boundaries
    v
 Kokoro TTS (CPU, 24kHz -> resample to 48kHz)
    |  synthesizes speech from sentence chunks
    v
 WebRTC (Opus 48kHz) -> Browser Speaker
```

Transcripts and LLM responses are also displayed in the browser via WebSocket.

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
- [espeak-ng](https://github.com/espeak-ng/espeak-ng/releases) (system dependency, required by Kokoro TTS for phonemization)
- LLM API key (DeepSeek, OpenAI, or any OpenAI-compatible provider)

## Quick Start

```bash
# Clone and enter the project
git clone https://github.com/your-org/FullDuplexVoiceAI.git
cd FullDuplexVoiceAI

# Create and activate a virtual environment
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/macOS
source .venv/bin/activate

# Install dependencies
pip install -e .

# Configure your LLM API key and TTS voice
cp .env.example .env
# Edit .env with your LLM_API_KEY and optionally change TTS_VOICE

# Start the server
python -m server.main
```

Open `http://127.0.0.1:8080` in your browser, click **Connect**, and speak. The AI will listen, transcribe, generate a response, and speak back.

The Silero VAD ONNX model (~2MB) is downloaded automatically on first run. The Faster-Whisper model (~1.5GB for medium) and Kokoro TTS model are downloaded from HuggingFace on first run.

## Project Structure

```
server/
  main.py              Entry point (uvicorn on port 8080)
  config.py            Settings (sample rates, VAD/STT/LLM/TTS params)
  signaling.py         FastAPI app (/offer, /ws/transcripts, /health)
  connection.py        WebRTC peer connection + pipeline orchestration
  tracks.py            OutputAudioTrack (real-time pacing, silence fallback)
  audio/
    processing.py      AudioProcessor protocol
    buffer.py          AsyncQueue-based frame buffer
    chunking.py        PCM frame conversion (stereo->mono, float->int16)
    resampling.py      Streaming + one-shot resampler (soxr)
  session.py           Per-session state machine (IDLE/LISTENING/THINKING/SPEAKING)
  metrics.py           Per-utterance latency instrumentation (P50/P95/P99)
  pipeline/
    runner.py          PipelineRunner (input track -> processors -> output)
    listening.py       ListeningProcessor (VAD + STT combined)
    bargein.py         BargeInFilter (duration + energy threshold)
  vad/
    model.py           Silero VAD ONNX wrapper (per-connection state)
    detector.py        VoiceActivityDetector (state machine + ring buffer)
  stt/
    transcriber.py     WhisperTranscriber (thread pool, GPU inference)
  llm/
    client.py          LLMClient (AsyncOpenAI, streaming completions)
    conversation.py    ConversationManager (per-session history, summarization)
    chunker.py         SentenceChunker (first-chunk clause optimization)
    prompts.py         System prompts for voice conversation
  tts/
    synthesizer.py     KokoroTTS (shared singleton) + TTSSpeaker (per-session)
client/
  index.html           Browser UI (dark theme, transcript display)
  client.js            WebRTC + WebSocket client
tests/
  test_buffer.py       AudioBufferManager tests
  test_chunking.py     Frame conversion tests
  test_resampling.py   Resampling tests
  test_vad_detector.py VAD state machine tests (mocked + integration)
  test_chunker.py      SentenceChunker tests
  test_conversation.py ConversationManager tests
  test_tts_synthesizer.py  TTSSpeaker tests (mocked TTS engine)
  test_bargein.py      BargeInFilter threshold tests
  test_metrics.py      LatencyTracker and UtteranceTimings tests
  test_session.py      SessionState tests
```

## Configuration

Settings are in [server/config.py](server/config.py). LLM and TTS voice are configured via environment variables in `.env`:

| Setting | Default | Description |
|---------|---------|-------------|
| `stt_model_size` | `"medium"` | Whisper model (`"small"` for faster, `"medium"` for accuracy) |
| `stt_device` | `"cuda"` | Inference device (`"cuda"` or `"cpu"`) |
| `stt_compute_type` | `"int8"` | Quantization (int8 for speed on GPU) |
| `vad_threshold` | `0.5` | Speech probability threshold (0.0-1.0) |
| `vad_silence_duration_ms` | `300` | Silence duration before end-of-speech |
| `vad_pre_speech_ms` | `300` | Pre-speech ring buffer to capture utterance start |
| `stt_language` | `"en"` | Language hint for Whisper |
| `LLM_BASE_URL` | `https://api.deepseek.com` | LLM provider endpoint (env var) |
| `LLM_API_KEY` | — | API key for LLM provider (env var, required) |
| `LLM_MODEL` | `deepseek-chat` | Model name (env var) |
| `TTS_VOICE` | `af_heart` | Kokoro voice (env var, see `.env.example` for options) |
| `TTS_DEVICE` | `cpu` | Kokoro inference device: `"cpu"` or `"cuda"` (env var) |
| `tts_language` | `"a"` | Kokoro language code (`"a"` = American English) |
| `barge_in_min_duration_ms` | `300` | Min speech duration (ms) before triggering barge-in |
| `barge_in_min_energy` | `200.0` | Min RMS energy (int16 scale) to confirm barge-in |

## Running Tests

```bash
# With the virtual environment activated:
pip install -e ".[dev]"
python -m pytest tests/ -v
```

## Key Design Decisions

- **Silero VAD via ONNX Runtime** (not PyTorch) -- avoids a ~2GB torch dependency. Each connection gets its own ONNX session with isolated hidden state.
- **Combined ListeningProcessor** -- VAD and STT are tightly coupled in one AudioProcessor. VAD accumulates audio and triggers STT on end-of-speech via `asyncio.create_task`.
- **Shared singletons** -- `WhisperTranscriber`, `LLMClient`, and `KokoroTTS` are each loaded once and shared across all connections. Single-worker ThreadPoolExecutors serialize GPU/CPU inference.
- **WebSocket for transcripts** -- more reliable than WebRTC DataChannel for text delivery. DataChannel requires SCTP negotiation that can fail with aiortc.
- **Streaming resampler** -- soxr `ResampleStream` maintains state across chunk boundaries for artifact-free 48kHz→16kHz conversion. TTS uses one-shot resampling (24kHz→48kHz) since each sentence is independent.
- **32ms VAD chunks** -- Silero VAD at 16kHz requires exactly 512 samples per chunk. A chunk accumulator handles the 20ms WebRTC frame to 32ms VAD chunk mismatch.
- **64-sample context window** -- Silero's ONNX model requires prepending 64 samples of context from the previous chunk for correct inference.
- **First-chunk clause optimization** -- `SentenceChunker` emits the first text chunk at a clause boundary (comma) for minimum time-to-first-audio, then batches subsequent chunks at sentence boundaries.
- **Two-stage TTS speaker** -- Synthesis and frame pushing run as separate async tasks. The synthesizer runs ahead of playback, pre-filling an intermediate frame buffer so there are no gaps between sentences.
- **TTS bypasses processor chain** -- TTS output feeds directly into the `PipelineRunner`'s output queue rather than going through the `AudioProcessor` pipeline. Listening (audio→text) and speaking (text→audio) are independent concurrent streams.
- **Silence frame fallback** -- `OutputAudioTrack` emits silence frames when the queue is empty (20ms timeout), keeping the RTP stream alive during gaps before TTS output begins.
- **Explicit session state machine** -- `SessionState` tracks per-session phase (IDLE/LISTENING/THINKING/SPEAKING). Used by barge-in detection to determine if user speech is an interruption vs. a new turn.
- **Dual-threshold barge-in** -- `BargeInFilter` requires both minimum speech duration (300ms) and minimum RMS energy before confirming an interruption. Filters coughs, background noise, and backchannel utterances ("uh-huh"). Barge-in triggers on SPEECH_START (not SPEECH_END) for faster response.
- **Idempotent interruption** -- `_execute_barge_in()` cancels LLM, drains TTS queues, and notifies the client in one atomic operation. Safe to call multiple times for rapid re-interruptions.
- **Per-utterance latency tracking** -- `UtteranceTimings` records timestamps at every pipeline stage. `LatencyTracker` computes rolling P50/P95/P99 voice-to-voice latency and logs per-utterance breakdowns.
