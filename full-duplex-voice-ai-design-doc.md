# Real-Time Low-Latency Full-Duplex Audio Streaming Server

## Design Document & Development Roadmap

**Author:** Solo project
**Status:** Phase 5 complete (Interruptions). Phase 6 next (Production Hardening).
**Last updated:** April 10, 2026

---

## Problem Statement

Voice AI sounds robotic because most implementations are turn-based. You talk, it waits, then it responds. Real humans interrupt each other mid-sentence, backchannel with "uh-huh" and "right," and overlap constantly. Current voice bots cannot handle this. They feel unnatural and frustrating to use.

Full-duplex with low latency is hard. The system must simultaneously stream audio in both directions, detect when the user interrupts, cancel in-progress responses, and resume listening — all within a few hundred milliseconds. Companies building voice agents, call center automation, or conversational AI will pay top dollar for engineers who can build this.

---

## Architecture Overview

The system uses a **cascaded pipeline** architecture where each stage is a discrete, swappable component connected by async streams. STT and TTS run locally to eliminate API costs and network latency for those stages. Only the LLM calls an external API — the provider is configurable (DeepSeek, OpenAI, or any OpenAI-compatible API) via environment variables.

```
User's Mic
    │
    ▼
┌──────────┐
│  WebRTC  │  ← Real-time bidirectional audio transport
└────┬─────┘
     │ raw PCM audio chunks
     ▼
┌──────────┐
│Silero VAD│  ← Is the user speaking? Started? Stopped? Interrupted?
└────┬─────┘
     │ speech segments only (silence filtered out)
     ▼
┌──────────────┐
│Faster-Whisper│  ← Local speech-to-text (VAD-buffered utterance → transcript)
│  (local GPU) │
└────┬─────────┘
     │ text
     ▼
┌──────────────────────┐
│ LLM Chat             │  ← Streaming token-by-token response generation
│ Completions (stream) │     (DeepSeek, OpenAI, or any compatible API)
└────┬─────────────────┘
     │ text tokens (streaming)
     ▼
┌────────────┐
│ Kokoro 82M │  ← Local text-to-speech (sentence chunks → audio)
│ (local CPU)│
└────┬───────┘
     │ audio chunks
     ▼
┌──────────┐
│  WebRTC  │  ← Same connection, return path
└────┬─────┘
     │
     ▼
User's Speaker
```

### Why Cascaded Instead of End-to-End?

End-to-end speech models like Moshi and NVIDIA PersonaPlex handle full-duplex natively in a single model. They're impressive — PersonaPlex achieves 200ms latency — but they require massive GPU resources (A100/H100), offer limited customization, and are a black box you can't debug.

The cascaded approach lets you understand and control every stage. You can swap Faster-Whisper for Deepgram, Kokoro for ElevenLabs, OpenAI for DeepSeek or Anthropic — each piece is independent. When something breaks, you know exactly where. When you need to optimize, you can profile each stage. This is how most production voice AI systems are built today.

### Why Local STT and TTS?

Running Faster-Whisper and Kokoro locally instead of using paid APIs (Deepgram, ElevenLabs) has three advantages:

1. **Cost:** The only API cost is the LLM provider. STT and TTS are free — you pay only for GPU electricity. Using DeepSeek instead of OpenAI can reduce LLM costs by 10–20x.
2. **Latency:** No network round-trip to external STT/TTS services. Audio stays on the server.
3. **Privacy:** User audio never leaves your machine for transcription or synthesis.

The tradeoff is hardware — you need a machine with a GPU (RTX 3060 12GB or better). And local Whisper is not a true streaming STT like Deepgram — it transcribes buffered utterances rather than streaming word-by-word. This changes the architecture slightly (see Stage 3).

---

## Pipeline Deep Dive

### Stage 1: WebRTC Transport

**What it does:** Moves raw audio between the browser and server in real time.

**Why WebRTC over WebSockets:** WebSockets can technically carry audio, but WebRTC was purpose-built for real-time media and includes critical features you'd otherwise have to build yourself:

| Feature | WebRTC | WebSocket |
|---|---|---|
| NAT traversal (firewalls) | Built-in (ICE/STUN/TURN) | Manual |
| Jitter buffering | Built-in | Manual |
| Echo cancellation (AEC) | Built-in | Manual |
| Auto gain control (AGC) | Built-in | Manual |
| Transport latency | 20–50ms | 100–200ms |
| Encryption | SRTP (mandatory) | TLS (optional) |

**Key concepts to learn:** ICE candidate gathering, STUN/TURN servers, SDP offer/answer exchange, SRTP media encryption, DataChannel for signaling alongside media.

**Implementation:** Use `aiortc` (Python WebRTC library) for the server side. The browser uses the native `RTCPeerConnection` API. Audio format over the wire is typically Opus codec at 48kHz, decoded to 16kHz 16-bit PCM on the server for downstream processing.

### Stage 2: Silero VAD (Voice Activity Detection)

**What it does:** Analyzes incoming audio chunks and answers: is the user speaking right now?

**Why it's critical:** VAD is the brain of the interruption system. It drives three core decisions:

1. **Speech start detected** → Begin buffering audio for STT. If the AI is currently speaking, this is a **barge-in** — immediately stop AI audio playback, cancel in-progress LLM generation, and start listening.
2. **Speech continues** → Keep buffering audio.
3. **Silence detected after speech** → This is **end-of-turn**. The user finished talking. Send the buffered audio to Faster-Whisper for transcription, then trigger the LLM.

**Sensitivity tradeoffs:** Too sensitive and the AI cuts the user off mid-sentence during brief pauses. Too lenient and the AI waits awkwardly long after the user stops. Typical silence threshold is 200–400ms, but this needs calibration based on context.

**Model:** Silero VAD is a small, fast, CPU-only model that processes 30ms audio chunks. No GPU required. It's the most widely used VAD in the voice AI ecosystem — Pipecat, LiveKit, and most production systems use it.

### Stage 3: Faster-Whisper STT (Speech-to-Text) — Local

**What it does:** Converts the user's speech audio into text. Runs locally on your GPU using CTranslate2-optimized Whisper models.

**How it differs from a streaming API like Deepgram:** Deepgram streams partial transcripts word-by-word as the user speaks. Faster-Whisper is a batch model — it takes a complete audio segment and returns the full transcript. This means you rely on Silero VAD to detect end-of-speech, buffer the entire utterance, then pass the buffer to Faster-Whisper for transcription. No interim/partial results during speech.

**How the VAD-buffered approach works:**

1. Silero VAD detects speech start → begin accumulating PCM audio into a buffer.
2. User continues speaking → buffer grows.
3. Silero VAD detects silence (end-of-speech, ~300ms threshold) → pass the complete audio buffer to Faster-Whisper.
4. Faster-Whisper transcribes the utterance → send transcript to LLM.

**Performance:** On an RTX 3060, Faster-Whisper with the `medium` model transcribes a 5-second utterance in ~300–500ms. The `small` model is faster (~200ms) with slightly lower accuracy. The `large-v3` model is most accurate but slower (~800ms). For conversational voice AI, `medium` or `small` is the sweet spot.

**Why Faster-Whisper over vanilla Whisper:** Faster-Whisper uses CTranslate2, a C++ inference engine that provides ~4x speedup over the original PyTorch Whisper implementation with equivalent accuracy. It also supports int8 quantization for further speed gains on supported GPUs.

**Tradeoff vs. Deepgram:** You lose real-time partial transcripts (no "words appearing as you speak" experience). The user has to finish their utterance before transcription begins. This adds ~300–500ms to the pipeline compared to Deepgram's streaming approach. But you pay $0 for STT.

### Stage 4: LLM Chat Completions (Streaming)

**What it does:** Takes the transcribed text plus conversation history and generates a text response, streaming tokens one at a time. The LLM provider is configurable — DeepSeek, OpenAI, or any OpenAI-compatible API — via three environment variables:

```
LLM_BASE_URL=https://api.deepseek.com   # or https://api.openai.com/v1
LLM_API_KEY=your-api-key
LLM_MODEL=deepseek-chat                  # or gpt-4o
```

The `openai` Python SDK is used as the client library for all providers — just point `base_url` at the right endpoint.

**Why streaming matters here:** Non-streaming completion waits for the entire response to generate before returning anything — could be 2–5 seconds for a long answer. Streaming returns tokens as they're generated. Time-to-first-token is typically ~200ms. This means you can start sending text to TTS before the LLM is done thinking.

**Interruption handling:** When VAD detects the user speaking during AI output, you must:

1. Cancel the in-progress streaming completion (close the SSE connection or call abort).
2. Clear the TTS audio buffer.
3. Stop audio playback on the client.
4. Append what was generated so far to conversation history (so the AI has context about what it already said).
5. Begin processing the user's new input.

**Chunking strategy for TTS handoff:** You don't send each individual token to TTS — that would produce choppy, unnatural speech. Instead, buffer tokens until you hit a natural break point (sentence boundary, comma, colon) then send that chunk to TTS. First chunk can be smaller (even a clause) to minimize time-to-first-audio.

**Why not the OpenAI Realtime API:** The Realtime API is a speech-to-speech system — audio goes in, audio comes out. It replaces Faster-Whisper, the LLM, and Kokoro in one call. That's faster to ship, but you learn nothing about the pipeline, can't swap components, and have limited control over voice, latency, and behavior. We'll reference it as an alternative path at the end of the project.

### Stage 5: Kokoro TTS (Text-to-Speech) — Local

**What it does:** Converts the LLM's text tokens into natural-sounding speech audio. Runs locally — on CPU or GPU.

**Why Kokoro:** Kokoro is an 82M parameter open-weight TTS model (Apache 2.0 license) that punches far above its weight. It ranked #1 on the HuggingFace TTS Arena for single-speaker quality, just behind ElevenLabs overall. It runs on CPU (no GPU required for TTS), generates audio at ~90x real-time on a 3090 and fast enough even on 12-year-old CPUs. 54 voices across 8 languages. Completely free.

**How Kokoro synthesis works:**

1. Receive a text chunk from the LLM token buffer (at sentence boundaries).
2. Pass the text through the Kokoro pipeline (`KPipeline`) with your chosen voice.
3. Kokoro returns a numpy array of 24kHz PCM audio.
4. Resample to match WebRTC output format, encode to Opus, and send.

**Building a streaming wrapper:** Kokoro doesn't have a built-in WebSocket streaming server like ElevenLabs. You build this yourself:

1. Create an async queue that receives text chunks from the LLM stage.
2. A worker coroutine pulls chunks from the queue, runs Kokoro synthesis, and pushes audio to the outbound WebRTC stream.
3. On interruption, clear the queue and cancel any in-progress synthesis.

This is actually simpler than integrating with an external API — no WebSocket connection management, no authentication, no rate limits, no reconnection logic. The model is just a function call.

**Chunking tradeoffs:** Send too little text and the voice sounds choppy with incorrect prosody (the model doesn't have enough context to intone properly). Send too much and you add latency waiting for text to accumulate. Sweet spot: send at sentence boundaries or after 15–30 words, whichever comes first. First chunk can be shorter to minimize initial silence.

**Interruption handling:** When the user barges in, clear the text-chunk queue, cancel any in-progress Kokoro synthesis (cancel the thread pool future), and discard any buffered audio that hasn't been played yet.

### Stage 6: WebRTC Return Path

**What it does:** Streams the synthesized audio chunks back to the browser over the same WebRTC peer connection.

The audio from Kokoro (24kHz PCM) is resampled to 48kHz, encoded to Opus, packetized, and sent over the existing SRTP channel. The browser's WebRTC stack handles jitter buffering on the playback side, smoothing out any network variance so the audio doesn't stutter.

---

## Latency Budget

Every millisecond matters. Here's the target budget from user-stops-speaking to user-hears-first-syllable:

| Stage | Target | Notes |
|---|---|---|
| End-of-speech detection | ~200ms | Silence threshold before triggering response |
| WebRTC transport (inbound) | ~30ms | Audio from browser to server |
| Faster-Whisper transcription | ~400ms | Transcribe buffered utterance locally (medium model) |
| LLM time-to-first-token | ~200ms | First token from streaming completion (network) |
| Text buffering for TTS | ~50ms | Accumulate first sendable chunk |
| Kokoro synthesis | ~100ms | Generate first audio chunk locally (sentence fragment) |
| WebRTC transport (outbound) | ~30ms | Audio from server to browser |
| **Total** | **~600–900ms** | **Slightly above natural range, still conversational** |

Human conversational turn-taking is typically 200–500ms. Under 500ms feels natural. Under 1 second is acceptable and still feels responsive. Over 2 seconds feels broken.

**Note:** The latency is ~100–200ms higher than a paid-API version (Deepgram + ElevenLabs) because Faster-Whisper transcribes the full utterance after speech ends rather than streaming partial results during speech. The main optimization lever is model size — using `small` instead of `medium` cuts ~200ms from STT.

---

## Concurrency Model

The server must handle multiple independent async streams simultaneously for each connected user:

1. **Inbound audio stream** — continuously receiving audio from WebRTC.
2. **VAD processing** — analyzing each audio chunk for speech activity.
3. **STT processing** — running Faster-Whisper on buffered utterances (GPU, needs careful scheduling).
4. **LLM stream** — sending text to the LLM API and receiving tokens.
5. **TTS processing** — running Kokoro synthesis on text chunks (CPU or GPU).
6. **Outbound audio stream** — sending audio back through WebRTC.

All six streams run concurrently using Python `asyncio`. Since Faster-Whisper and Kokoro are local compute (not async I/O), they need to run in a thread pool executor or separate process to avoid blocking the event loop. The critical challenge is **coordination between streams** — when VAD detects an interruption, it must signal the LLM stream to cancel, the TTS to stop, and the outbound audio stream to flush, all within a few milliseconds. This is done through asyncio Events and shared state protected by locks.

```python
# Conceptual state machine for a single session
class SessionState:
    LISTENING = "listening"       # User is speaking, AI is silent
    PROCESSING = "processing"     # User stopped, AI is thinking
    SPEAKING = "speaking"         # AI is responding
    INTERRUPTED = "interrupted"   # User barged in, canceling AI response

# Key transitions:
# LISTENING → PROCESSING     (VAD detects end-of-speech)
# PROCESSING → SPEAKING      (first TTS audio chunk ready)
# SPEAKING → INTERRUPTED     (VAD detects speech during AI output)
# INTERRUPTED → LISTENING    (cleanup complete, ready for new input)
```

**Race conditions to watch for:**

- User stops speaking and starts again before Faster-Whisper finishes transcribing — need to cancel the in-progress transcription and extend the utterance buffer.
- LLM finishes generating but user already interrupted — don't send stale audio.
- Two rapid interruptions in succession — ensure cleanup from the first completes before processing the second.
- Faster-Whisper and Kokoro competing for GPU — if both use the GPU, schedule them to not overlap (they never run at the same time in normal flow: STT runs first, then TTS runs after LLM responds).
- Network disconnection mid-stream — graceful cleanup of LLM API connection and local model state.

---

## Tech Stack

| Component | Technology | Role | Cost |
|---|---|---|---|
| Server framework | FastAPI | HTTP endpoints for signaling, health checks | Free |
| Real-time transport | aiortc | Python WebRTC implementation | Free |
| Voice activity detection | Silero VAD | Speech/silence detection, barge-in trigger | Free (CPU) |
| Speech-to-text | Faster-Whisper (medium) | Local transcription | Free (GPU) |
| LLM | Configurable (DeepSeek, OpenAI, etc.) | Response generation | ~$0.02–2.00/hr |
| Text-to-speech | Kokoro 82M | Local voice synthesis | Free (CPU) |
| Async runtime | Python asyncio | Concurrent stream management | Free |
| Audio processing | numpy, audioop | PCM manipulation, resampling | Free |
| Client | Vanilla JS + WebRTC API | Browser-based audio capture and playback | Free |

### Cost Summary

| Service | Cost |
|---|---|
| Faster-Whisper STT | $0 (local) |
| LLM API (DeepSeek) | ~$0.02–0.10/hr (varies by verbosity) |
| LLM API (OpenAI GPT-4o) | ~$0.50–2.00/hr (varies by verbosity) |
| Kokoro TTS | $0 (local) |
| **Total (with DeepSeek)** | **~$0.02–0.10/hr** |
| **Total (with OpenAI)** | **~$0.50–2.00/hr** |

### Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| GPU | RTX 3060 12GB | RTX 3070 or better |
| CPU | 4 cores | 8+ cores (Kokoro runs on CPU) |
| RAM | 16GB | 32GB |
| OS | Ubuntu 22.04+ | Ubuntu 24.04 |

Faster-Whisper `medium` model uses ~2GB VRAM. Kokoro runs on CPU by default. This leaves ample GPU headroom.

---

## Development Roadmap

### Phase 1: Foundation (Weeks 1–2) -- COMPLETE

**Goal:** Establish real-time audio transport between browser and server.

**Module 1 — WebRTC fundamentals**
- Learn ICE/STUN/TURN connectivity, SDP offer/answer, SRTP.
- Set up `aiortc` server that accepts a peer connection.
- Build a minimal browser client that captures mic audio via `getUserMedia` and sends it over WebRTC.
- Echo test: receive audio on server, send it back to client, hear yourself with ~50ms delay.
- Handle connection lifecycle: connect, reconnect on failure, graceful disconnect.

**Module 2 — Audio processing basics**
- Understand PCM, sample rates (16kHz for STT, 24kHz for Kokoro, 48kHz for WebRTC), bit depth (16-bit).
- Decode Opus frames from WebRTC into raw PCM on the server.
- Implement chunking: split continuous audio stream into fixed-size chunks (20ms or 30ms frames).
- Implement resampling: 48kHz WebRTC audio → 16kHz for Faster-Whisper/VAD. 24kHz Kokoro output → 48kHz for WebRTC.
- Build an audio buffer manager that handles backpressure (what happens when downstream is slower than upstream).

**Deliverable:** Browser app where you speak into the mic and hear yourself echoed back through the server with minimal delay. Confirm round-trip latency is under 100ms.

**Implementation notes:**
- Server uses FastAPI + aiortc on port 8080. SDP exchange via `POST /offer`.
- `PipelineRunner` reads frames from input track, passes through `AudioProcessor` chain, outputs to `OutputAudioTrack` with real-time pacing.
- Audio utilities: `frame_to_ndarray` (stereo→mono, handles interleaved s16), `StreamResampler` (soxr stateful), `AudioBufferManager` (async queue with backpressure).
- Echo test confirmed working with `EchoProcessor` pass-through.

---

### Phase 2: Listening (Weeks 3–4) -- COMPLETE

**Goal:** The server can hear and transcribe speech in real time.

**Module 3 — Voice Activity Detection**
- Integrate Silero VAD model (runs on CPU, processes 30ms chunks).
- Implement speech state machine: `SILENCE → SPEECH_START → SPEAKING → SPEECH_END → SILENCE`.
- Calibrate silence threshold (start with 300ms, tune from there).
- Add pre-speech buffer: keep the last 300ms of audio in a ring buffer so you don't lose the beginning of an utterance (VAD has a small detection delay).
- Test with various scenarios: quiet room, background noise, music playing, two people talking.

**Module 4 — Local speech-to-text with Faster-Whisper**
- Install faster-whisper (`pip install faster-whisper`).
- Load the `medium` model (or `small` for faster inference). Test with int8 quantization.
- Build the VAD-buffered transcription flow: VAD detects end-of-speech → pass audio buffer to Faster-Whisper → get transcript.
- Run Faster-Whisper inference in a thread pool executor (`asyncio.run_in_executor`) to avoid blocking the event loop.
- Benchmark transcription latency for different utterance lengths and model sizes.
- Log full transcripts with timestamps for debugging.
- Handle edge cases: very short utterances (one word), very long utterances (30+ seconds), background noise.

**Deliverable:** Speak into the browser, see transcription printed on screen. Measure total time from end-of-speech to transcript available — target under 500ms.

**Implementation notes:**
- Silero VAD uses ONNX Runtime (not PyTorch) to avoid the ~2GB torch dependency for faster-whisper's inference engine. Model is ~2MB, auto-downloaded on first run to `server/vad/data/`.
- Key Silero ONNX details discovered during implementation: (1) requires 512-sample chunks at 16kHz (32ms, not 30ms), (2) requires a 64-sample context window prepended to each chunk, (3) hidden state tensor shape is `[2, 1, 128]` as a single `state` input (not separate h/c).
- Chunk accumulator handles 20ms WebRTC frames → 32ms VAD chunks. Each 20ms frame produces ~320 samples at 16kHz; after 3 frames (60ms) the accumulator yields 2 complete 512-sample chunks.
- `ListeningProcessor` combines VAD + STT as a single `AudioProcessor`. Returns `None` for all frames (no audio output in Phase 2). STT is triggered asynchronously via `asyncio.create_task` on `SPEECH_END`.
- Transcripts delivered to browser via WebSocket (`/ws/transcripts`) rather than WebRTC DataChannel. DataChannel proved unreliable with aiortc (SCTP negotiation issues).
- `WhisperTranscriber` is a singleton shared across connections, using a single-worker `ThreadPoolExecutor` to serialize GPU access.
- Stereo audio from WebRTC required mono extraction (de-interleave and take first channel) -- initial `frame_to_ndarray` flattened both channels, garbling the signal.
- Measured on RTX 4060: `medium` model with int8 quantization achieves 200-420ms inference after GPU warmup (first inference ~800ms). Target of <500ms met.

---

### Phase 3: Thinking (Weeks 5–6) -- COMPLETE

**Goal:** The system generates intelligent responses to what the user says.

**Module 5 — LLM streaming integration**
- Configure LLM provider via env vars (`LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`). Default to DeepSeek.
- Connect to the LLM Chat Completions API using the `openai` SDK with `stream=True`.
- Build conversation history manager: system prompt, user messages, assistant messages, context window management.
- Implement token buffering for TTS handoff: accumulate tokens until a sentence boundary (period, question mark, exclamation, or colon followed by space), then emit the chunk.
- First-chunk optimization: emit the first chunk at a clause boundary (comma) even if it's not a full sentence, to minimize time-to-first-audio.
- Implement stream cancellation: when an interruption signal arrives, abort the streaming request immediately and record what was generated so far.

**Module 6 — Conversation state management**
- Track what the AI has said vs. what was actually played to the user (important for interruptions — if the user interrupted, the AI only "said" what was played, not what was generated).
- Implement conversation history trimming: summarize older turns when context window gets large.
- Add system prompt engineering for voice conversations: instruct the model to give concise, spoken-friendly responses (no bullet points, no markdown, no emojis).

**Deliverable:** Type text into a console, get a streaming text response. Then wire it up: speak into browser → see transcript → see streaming LLM response in real time.

**Implementation notes:**
- `LLMClient` wraps `AsyncOpenAI` with configurable base URL, model, and API key. Shared singleton across sessions.
- `SentenceChunker` buffers streaming tokens: first chunk emits at clause boundary (comma) for minimum latency, subsequent chunks batch at sentence boundaries (`.!?:`). This feeds directly into TTS.
- `ConversationManager` tracks per-session history with `played_text` vs `generated_text` for interruption-aware context. Uses LLM-based summarization (not truncation) when history exceeds 40 messages.
- Stream cancellation via `asyncio.Event` per session — new user transcript sets the event, which aborts the in-progress SSE stream and records partial output.

---

### Phase 4: Speaking (Weeks 7–8) -- COMPLETE

**Goal:** The AI responds with natural-sounding voice.

**Module 7 — Local text-to-speech with Kokoro**
- Install Kokoro (`pip install kokoro soundfile`) and espeak-ng.
- Initialize the `KPipeline` with your chosen language and voice (e.g., `af_heart`, `am_adam`).
- Build the async TTS wrapper:
  - Create an `asyncio.Queue` for text chunks from the LLM.
  - A worker coroutine pulls chunks, runs `pipeline(text, voice=voice)` in a thread pool executor (Kokoro is CPU-bound).
  - Push resulting PCM audio (24kHz numpy array) to the outbound audio queue.
- Handle audio format conversion: Kokoro outputs 24kHz PCM → resample to 48kHz → encode to Opus for WebRTC.
- Implement audio queue: buffer a small amount of TTS audio (~200ms) before starting playback to prevent stuttering.
- Test different voices and compare quality and synthesis speed.

**Module 8 — End-to-end integration (no interrupts)**
- Wire the full pipeline: Mic → WebRTC → VAD → Faster-Whisper → LLM → Kokoro → WebRTC → Speaker.
- Measure end-to-end latency. Target: under 900ms voice-to-voice.
- Profile each stage to identify bottlenecks.
- Test with real conversations: ask questions, have it tell stories, give instructions.
- Fix audio artifacts: clicks between chunks, volume normalization, silence gaps between Kokoro-generated segments.

**Deliverable:** Have a voice conversation with the AI. It listens, thinks, and speaks. No interruption support yet — pure turn-taking.

**Implementation notes:**
- `KokoroTTS` is a shared singleton (like `WhisperTranscriber`) loaded eagerly at server startup. Uses a single-worker `ThreadPoolExecutor` to serialize CPU-bound Kokoro inference across sessions.
- `TTSSpeaker` is a per-session two-stage pipeline: (1) a synthesizer task pulls text from a queue, runs Kokoro, resamples 24kHz→48kHz via one-shot `soxr.resample`, and pushes 20ms AudioFrames into an intermediate buffer; (2) a pusher task feeds frames from the buffer to the WebRTC output queue at real-time pace via blocking `await put()`. This decoupled design lets synthesis run ahead of playback — sentence N+1 is synthesized while sentence N is still playing, eliminating inter-sentence gaps.
- TTS output bypasses the `AudioProcessor` pipeline chain. The `PipelineRunner` exposes its `output_queue` property, and `connection.py` feeds TTS frames directly into it from the LLM response path. This is architecturally clean because the listening pipeline (audio in → processors) and speaking path (text in → TTS → audio out) are independent concurrent streams.
- `OutputAudioTrack.recv()` uses `asyncio.wait_for` with a 20ms timeout. On timeout, it emits a silence frame (960 zero samples at 48kHz) to keep the RTP stream alive and prevent WebRTC track drops during inter-sentence gaps or before TTS output begins.
- Barge-in cancellation already wired: when `send_transcript` fires (user spoke), it cancels the LLM task and calls `TTSSpeaker.cancel()` which sets a cancel event and drains the text queue, intermediate frame queue, and output audio queue.
- Voice is configurable via `TTS_VOICE` environment variable. Default `af_heart`. All available Kokoro voices listed in `.env.example`.

---

### Phase 5: Interruptions (Weeks 9–10) -- COMPLETE

**Goal:** Full-duplex conversation with natural barge-in support.

**Module 9 — Interruption handling**
- Detect barge-in: VAD reports speech while session state is `SPEAKING`.
- On barge-in:
  1. Cancel the LLM streaming request.
  2. Clear the Kokoro text-chunk queue.
  3. Cancel any in-progress Kokoro synthesis (cancel the thread pool future).
  4. Clear the outbound audio buffer.
  5. Stop audio playback on the client (send a control message via WebSocket).
  6. Transition to `LISTENING` state.
  7. Record partial AI response in conversation history.
- Handle false barge-ins: brief coughs, background noise, the user saying "uh-huh" as a backchannel. Options: require minimum speech duration (e.g., 300ms) before triggering interrupt, or use energy threshold in addition to VAD.
- Handle rapid re-interruptions: user interrupts, AI starts new response, user interrupts again. Ensure cleanup is idempotent.

**Module 10 — Latency optimization**
- Experiment with Faster-Whisper model sizes: `small` vs `medium` vs `large-v3` — measure accuracy/latency tradeoff.
- Try int8 quantization for Faster-Whisper to reduce VRAM and speed up inference.
- Optimize Kokoro chunking: experiment with chunk sizes and measure impact on latency vs. prosody quality.
- Consider running Kokoro on GPU instead of CPU if GPU headroom allows (faster synthesis).
- Add latency measurement instrumentation: log timestamps at every pipeline stage for every utterance.
- Implement P50/P95/P99 latency tracking.
- Profile async task scheduling: ensure no stage is starving others of CPU time.
- Investigate speculative execution: start Faster-Whisper transcription on partial audio (before end-of-speech) and update if more speech arrives.

**Deliverable:** Have a natural conversation where you can interrupt the AI mid-sentence. It stops, listens, and responds to your interruption. Measure and log voice-to-voice latency for every turn.

**Implementation notes:**
- **Explicit session state machine:** New `SessionState` class (`server/session.py`) tracks per-session phase: IDLE → LISTENING → THINKING → SPEAKING. Replaces the implicit state previously derived from task lifecycle. Used by barge-in detection to know whether user speech is an interruption.
- **True barge-in on SPEECH_START:** `ListeningProcessor` now fires an `on_speech_start` callback when VAD detects speech, plus an `on_speech_audio` callback on every audio chunk during active speech. Previously, interruption only triggered on SPEECH_END (after user finished speaking). Now interruption triggers mid-utterance once the barge-in filter confirms it.
- **Dual-threshold barge-in filter:** `BargeInFilter` (`server/pipeline/bargein.py`) requires both 300ms of continuous speech AND average RMS energy above a configurable threshold before confirming a barge-in. This filters false positives from coughs, background noise, and brief backchannel utterances like "uh-huh". Both thresholds are configurable via `barge_in_min_duration_ms` and `barge_in_min_energy` settings.
- **Idempotent `_execute_barge_in`:** Centralized async function in `connection.py` that cancels LLM streaming, drains all TTS queues, sends a `stop_playback` WebSocket message to the client, and transitions session to LISTENING. Safe to call multiple times — cancelling an already-cancelled task or draining empty queues are no-ops. Both the barge-in filter trigger and `send_transcript` (SPEECH_END) call this same function.
- **Client-side playback stop:** WebSocket `stop_playback` message causes the browser to briefly mute the `<audio>` element (100ms) to flush the WebRTC jitter buffer of stale audio, then unmute for new audio. The response display is marked with "[interrupted]".
- **Per-utterance latency instrumentation:** `UtteranceTimings` (`server/metrics.py`) records `time.perf_counter()` timestamps at every pipeline stage: speech_start, speech_end, stt_start, stt_end, llm_first_token, llm_done, tts_first_chunk_done, first_frame_to_client. Voice-to-voice latency = speech_end → first_frame_to_client. `LatencyTracker` computes rolling P50/P95/P99 percentiles and logs per-utterance breakdowns.
- **TTS first-frame callback:** `TTSSpeaker` now accepts an `on_first_frame` callback that fires once per response cycle when the first audio frame is pushed to the WebRTC output queue. Used to record `first_frame_to_client` timestamp for V2V latency calculation.
- **Config knobs for optimization:** `tts_device` setting (env: `TTS_DEVICE`, default "cpu") allows switching Kokoro to GPU. `stt_model_size` already existed for Whisper model switching. Speculative STT execution deferred to future work.
- **Conversation history tracks interruptions:** On barge-in cancellation, the full LLM-generated text (including unplayed portions) is recorded via `ConversationManager.add_assistant_message()`. The `was_interrupted` flag is set so the LLM has context about what was cut off.

---

### Phase 6: Production Hardening (Weeks 11–12)

**Goal:** Make it reliable, observable, and deployable.

**Module 11 — Error handling and resilience**
- Faster-Whisper model loading failures: preload model at server startup, health check endpoint.
- Kokoro synthesis errors: handle malformed text, empty strings, extremely long inputs.
- LLM API errors or slow responses: timeout handling, retry with exponential backoff.
- WebRTC connection instability: ICE restart, handling network switches (wifi → cellular).
- GPU memory management: monitor VRAM usage, handle out-of-memory gracefully.
- Graceful degradation: if the LLM API is down, communicate to the user rather than hanging silently.

**Module 12 — Multi-session support**
- Handle multiple concurrent users, each with their own independent pipeline.
- Per-session resource management: each user gets their own VAD instance, audio buffers, LLM context, and TTS queue. Faster-Whisper model is shared (loaded once, handles sequential requests).
- Implement session cleanup on disconnect: cancel all in-progress tasks, free buffers.
- GPU scheduling: if multiple sessions need Faster-Whisper simultaneously, queue transcription requests (model can only process one at a time unless you load multiple instances).
- Add basic load metrics: active sessions, per-session latency, GPU utilization, error rates.

**Module 13 — Client polish**
- Build a clean browser UI: connection status indicator, live transcript display, audio visualizer (waveform or volume meter), latency readout.
- Add push-to-talk as an alternative mode (simpler than full-duplex, useful for noisy environments).
- Handle browser audio permissions gracefully.
- Test across Chrome, Firefox, Safari.

**Deliverable:** A deployable server that handles multiple simultaneous voice conversations with interruption support, error recovery, and a polished browser client.

---

## Alternative Architecture: OpenAI Realtime API

For reference, the OpenAI Realtime API collapses the entire STT → LLM → TTS pipeline into a single WebSocket connection. Audio goes in, audio comes out. It handles VAD, transcription, response generation, and speech synthesis internally.

**Pros:** Dramatically simpler code, potentially lower latency (~300ms), built-in interruption handling, one vendor to manage, no GPU required on your server.

**Cons:** Black box (can't debug individual stages), locked to OpenAI's voices, no component swapping, higher per-minute cost (you pay for STT + LLM + TTS bundled), less control over conversation behavior.

This is worth exploring after completing the from-scratch build as a comparison point. The knowledge you gain from building each stage individually will let you evaluate the tradeoffs intelligently.

---

## Upgrade Paths

Once the core system is working, these are natural next steps:

**Swap to paid APIs for lower latency:** If you need to get under 500ms, swap Faster-Whisper for Deepgram (~$0.46/hr, adds streaming partial transcripts) and/or Kokoro for ElevenLabs (~$0.30/hr, adds WebSocket streaming with lower time-to-first-audio). The architecture is designed to make these swaps easy.

**Semantic VAD:** Silero VAD is acoustic — it detects sound, not meaning. A semantic VAD (like Tencent's approach using a small LLM to classify continue-listening vs. start-speaking) could reduce false interrupts and detect when the user is actually done with their thought, not just pausing.

**End-to-end models:** Moshi and PersonaPlex represent the future — single models that handle full-duplex natively. Worth studying after shipping the cascaded version.

**Pipecat framework:** After building from scratch, refactoring into Pipecat would give you production-grade orchestration with 100+ service integrations. Good second iteration.

**Telephony integration:** Connecting to phone networks via SIP/PSTN for call center use cases.

**Evaluation:** How do you measure conversation quality? Look at the FullDuplexBench benchmark series for metrics on barge-in accuracy, end-of-turn detection, and latency.

---

## References

- Faster-Whisper: github.com/SYSTRAN/faster-whisper (CTranslate2-optimized Whisper, ~4x speedup)
- Kokoro TTS: huggingface.co/hexgrad/Kokoro-82M (82M params, Apache 2.0, #1 on HF TTS Arena)
- Silero VAD: github.com/snakers4/silero-vad
- aiortc: github.com/aiortc/aiortc (Python WebRTC)
- Pipecat framework: github.com/pipecat-ai/pipecat (11k+ stars, production-grade voice AI orchestration)
- FireRedChat: Full-duplex voice system with cascaded and semi-cascaded implementations (arxiv 2509.06502)
- NVIDIA PersonaPlex: End-to-end full-duplex 7B model (arxiv 2602.06053)
- FullDuplexBench: Benchmark suite for evaluating full-duplex voice agents
