import asyncio
import logging
import time
from contextlib import suppress

from aiortc import RTCPeerConnection, RTCSessionDescription

from server.llm.client import LLMClient
from server.llm.conversation import ConversationManager
from server.metrics import LatencyTracker
from server.pipeline.bargein import BargeInFilter
from server.pipeline.listening import ListeningProcessor
from server.pipeline.runner import PipelineRunner
from server.session import SessionState
from server.stt.transcriber import WhisperTranscriber
from server.tts.synthesizer import KokoroTTS, TTSSpeaker

logger = logging.getLogger(__name__)

# Track all active connections for cleanup on shutdown
_peer_connections: set[RTCPeerConnection] = set()
_pipelines: dict[int, PipelineRunner] = {}

# Shared singletons (loaded on first connection)
_transcriber: WhisperTranscriber | None = None
_llm_client: LLMClient | None = None
_tts_engine: KokoroTTS | None = None

# Per-session state
_conversations: dict[int, ConversationManager] = {}
_llm_tasks: dict[int, asyncio.Task] = {}
_cancel_events: dict[int, asyncio.Event] = {}
_tts_speakers: dict[int, TTSSpeaker] = {}
_session_states: dict[int, SessionState] = {}
_barge_in_filters: dict[int, BargeInFilter] = {}
_latency_trackers: dict[int, LatencyTracker] = {}
_system_prompts: dict[int, str | None] = {}


async def _get_transcriber() -> WhisperTranscriber:
    """Get or create the shared WhisperTranscriber singleton."""
    global _transcriber
    if _transcriber is None:
        _transcriber = WhisperTranscriber()
        await _transcriber.start()
    return _transcriber


async def _get_llm_client() -> LLMClient:
    """Get or create the shared LLMClient singleton."""
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
        await _llm_client.start()
    return _llm_client


async def _get_tts_engine() -> KokoroTTS:
    """Get or create the shared KokoroTTS singleton."""
    global _tts_engine
    if _tts_engine is None:
        _tts_engine = KokoroTTS()
        await _tts_engine.start()
    return _tts_engine


async def preload_tts() -> None:
    """Eagerly load the TTS engine at server startup."""
    await _get_tts_engine()


async def _execute_barge_in(pc_id: int) -> None:
    """Execute a confirmed barge-in: stop LLM, TTS, flush audio, notify client.

    Idempotent — safe to call multiple times or when nothing is active.
    """
    from server.signaling import transcript_queues

    session_state = _session_states.get(pc_id)
    if session_state:
        session_state.mark_listening()

    # 1. Cancel LLM streaming
    cancel_event = _cancel_events.get(pc_id)
    if cancel_event:
        cancel_event.set()
    existing_task = _llm_tasks.get(pc_id)
    if existing_task and not existing_task.done():
        existing_task.cancel()
        with suppress(asyncio.CancelledError):
            await existing_task

    # 2-4. Cancel TTS (clears text_queue, frame_queue, output_queue)
    speaker = _tts_speakers.get(pc_id)
    if speaker:
        speaker.cancel()

    # 5. Send stop_playback to client via WebSocket
    queue = transcript_queues.get(pc_id)
    if queue:
        await queue.put({"type": "stop_playback"})


async def create_peer_connection(
    offer_sdp: str, offer_type: str, *, system_prompt: str | None = None
) -> tuple[RTCSessionDescription, int]:
    """Create a new WebRTC peer connection with VAD + STT pipeline.

    Returns:
        Tuple of (SDP answer, peer_id).
    """
    pc = RTCPeerConnection()
    pc_id = id(pc)
    _peer_connections.add(pc)
    _system_prompts[pc_id] = system_prompt
    logger.info("Created peer connection %s", pc_id)

    @pc.on("connectionstatechange")
    async def on_connectionstatechange():
        state = pc.connectionState
        logger.info("Connection %s state: %s", pc_id, state)
        if state in ("failed", "closed"):
            await _cleanup_connection(pc)

    @pc.on("track")
    def on_track(track):
        logger.info("Received %s track from peer %s", track.kind, pc_id)

        if track.kind == "audio":
            # Import here to avoid circular import
            from server.signaling import transcript_queues

            # Create per-session state objects
            conversation = ConversationManager(
                system_prompt=_system_prompts.get(pc_id)
            )
            _conversations[pc_id] = conversation

            session_state = SessionState()
            _session_states[pc_id] = session_state

            barge_in_filter = BargeInFilter()
            _barge_in_filters[pc_id] = barge_in_filter

            latency_tracker = LatencyTracker()
            _latency_trackers[pc_id] = latency_tracker

            # --- Callbacks for ListeningProcessor ---

            async def on_speech_start() -> None:
                """Called when VAD detects speech start."""
                barge_in_filter.on_speech_start()
                session_state.mark_listening()

            def on_speech_audio(audio_16k) -> None:
                """Called synchronously with each audio chunk during speech.

                If barge-in filter triggers AND session is AI-active,
                schedule async barge-in execution.
                """
                if session_state.is_ai_active and barge_in_filter.on_speech_audio(
                    audio_16k
                ):
                    logger.info("Barge-in triggered for peer %s", pc_id)
                    asyncio.ensure_future(_execute_barge_in(pc_id))

            async def send_transcript(text: str) -> None:
                logger.info("Transcript [peer %s]: %s", pc_id, text)
                queue = transcript_queues.get(pc_id)
                if queue:
                    await queue.put({"type": "transcript", "text": text})

                # Cancel any in-progress response (reuses barge-in logic)
                await _execute_barge_in(pc_id)
                barge_in_filter.reset()

                # Trigger LLM response
                conversation.add_user_message(text)
                task = asyncio.create_task(
                    _handle_llm_response(pc_id, conversation)
                )
                _llm_tasks[pc_id] = task

            # Create pipeline SYNCHRONOUSLY so the output track is added
            # before createAnswer — this ensures proper SDP negotiation.
            processor = ListeningProcessor(
                on_transcript=send_transcript,
                transcriber=None,  # Will be set during start
                on_speech_start=on_speech_start,
                on_speech_audio=on_speech_audio,
                latency_tracker=latency_tracker,
            )
            pipeline = PipelineRunner(
                input_track=track,
                processors=[processor],
            )
            _pipelines[pc_id] = pipeline

            # Add the output track NOW (before createAnswer)
            pc.addTrack(pipeline.output_track)

            # Start pipeline asynchronously (loads transcriber if needed)
            asyncio.ensure_future(_start_pipeline(processor, pipeline, pc_id))

        @track.on("ended")
        async def on_ended():
            logger.info("Track ended for peer %s", pc_id)

    # Set remote description (the offer from browser)
    offer = RTCSessionDescription(sdp=offer_sdp, type=offer_type)
    await pc.setRemoteDescription(offer)

    # Create and set local description (our answer)
    answer = await pc.createAnswer()
    await pc.setLocalDescription(answer)

    logger.info("Peer %s: SDP exchange complete", pc_id)
    return pc.localDescription, pc_id


async def _start_pipeline(
    processor: ListeningProcessor, pipeline: PipelineRunner, pc_id: int
) -> None:
    """Load the transcriber, pre-warm the LLM client, create TTS speaker, and start the pipeline."""
    transcriber = await _get_transcriber()
    processor._transcriber = transcriber
    await _get_llm_client()  # Pre-warm so first LLM call is fast

    tts_engine = await _get_tts_engine()

    # on_first_frame callback records latency timestamp
    def on_first_frame() -> None:
        tracker = _latency_trackers.get(pc_id)
        if tracker and tracker.current:
            now = time.perf_counter()
            tracker.current.first_frame_to_client = now
            if tracker.current.tts_first_chunk_done == 0.0:
                tracker.current.tts_first_chunk_done = now

    # on_playback_done fires after all audio for a response has been pushed
    def on_playback_done() -> None:
        session_state = _session_states.get(pc_id)
        if session_state and session_state.is_speaking:
            session_state.mark_idle()
            logger.info("TTS playback done for peer %s, session now IDLE", pc_id)

    speaker = TTSSpeaker(
        tts=tts_engine,
        output_queue=pipeline.output_queue,
        on_first_frame=on_first_frame,
        on_playback_done=on_playback_done,
    )
    await speaker.start()
    _tts_speakers[pc_id] = speaker

    await pipeline.start()


async def _handle_llm_response(pc_id: int, conversation: ConversationManager) -> None:
    """Generate and stream an LLM response for a user utterance."""
    from server.signaling import transcript_queues

    queue = transcript_queues.get(pc_id)
    if not queue:
        return

    cancel_event = asyncio.Event()
    _cancel_events[pc_id] = cancel_event
    accumulated = ""  # Track partial response for cancellation

    session_state = _session_states.get(pc_id)
    if session_state:
        session_state.mark_thinking()

    tracker = _latency_trackers.get(pc_id)
    first_chunk_received = False

    # Reset first-frame tracking on the TTS speaker for this response cycle
    speaker = _tts_speakers.get(pc_id)
    if speaker:
        speaker.reset_first_frame()

    try:
        llm_client = await _get_llm_client()
        messages = conversation.get_messages()
        logger.info(
            "LLM request [peer %s]: %d messages in history", pc_id, len(messages)
        )

        async def on_chunk(chunk: str) -> None:
            nonlocal accumulated, first_chunk_received
            accumulated += chunk
            await queue.put({"type": "llm_chunk", "text": chunk})

            # Record first token time for latency tracking
            if not first_chunk_received:
                first_chunk_received = True
                if tracker and tracker.current:
                    tracker.current.llm_first_token = time.perf_counter()
                # Mark session as speaking on first TTS enqueue
                if session_state:
                    session_state.mark_speaking()

            # Feed sentence chunk to TTS for audio synthesis
            if speaker:
                await speaker.enqueue(chunk)

        full_response = await llm_client.stream_completion(
            messages=messages,
            on_chunk=on_chunk,
            cancel_event=cancel_event,
        )

        # Record LLM completion time
        if tracker and tracker.current:
            tracker.current.llm_done = time.perf_counter()

        await queue.put({"type": "llm_done"})
        conversation.add_assistant_message(full_response)
        logger.info(
            "LLM response complete [peer %s]: %d chars, history now %d messages",
            pc_id, len(full_response), conversation.message_count,
        )

        # Signal TTS that no more text is coming. The session transitions
        # to IDLE only after TTS finishes playing all audio (via on_playback_done).
        if speaker:
            await speaker.enqueue_done()

        # Finalize latency tracking
        if tracker:
            tracker.finalize()

        # Check if summarization is needed
        await conversation.maybe_summarize(llm_client)

    except asyncio.CancelledError:
        logger.info("LLM response cancelled for peer %s", pc_id)
        # Record partial response so conversation history stays coherent
        if accumulated:
            conversation.add_assistant_message(accumulated)
        # Mark as interrupted for latency tracking
        if tracker and tracker.current:
            tracker.current.was_interrupted = True
            tracker.current.llm_done = time.perf_counter()
            tracker.finalize()
        await queue.put({"type": "llm_done"})
    except Exception:
        logger.exception("LLM response failed for peer %s", pc_id)
        if session_state:
            session_state.mark_idle()
        if tracker:
            tracker.finalize()
        await queue.put({"type": "llm_done"})


async def _cleanup_connection(pc: RTCPeerConnection) -> None:
    """Clean up a peer connection and its associated pipeline."""
    pc_id = id(pc)

    # Cancel any in-progress LLM response
    cancel_event = _cancel_events.pop(pc_id, None)
    if cancel_event:
        cancel_event.set()
    llm_task = _llm_tasks.pop(pc_id, None)
    if llm_task and not llm_task.done():
        llm_task.cancel()
        with suppress(asyncio.CancelledError):
            await llm_task

    speaker = _tts_speakers.pop(pc_id, None)
    if speaker:
        await speaker.stop()

    pipeline = _pipelines.pop(pc_id, None)
    if pipeline:
        await pipeline.stop()
    _conversations.pop(pc_id, None)
    _session_states.pop(pc_id, None)
    _barge_in_filters.pop(pc_id, None)
    _latency_trackers.pop(pc_id, None)
    _system_prompts.pop(pc_id, None)
    _peer_connections.discard(pc)
    await pc.close()
    # Clean up transcript queue
    from server.signaling import transcript_queues
    transcript_queues.pop(pc_id, None)
    logger.info("Cleaned up peer connection %s", pc_id)


async def shutdown_all() -> None:
    """Close all active peer connections. Called on server shutdown."""
    global _tts_engine
    logger.info("Shutting down %d peer connection(s)", len(_peer_connections))
    coros = [_cleanup_connection(pc) for pc in list(_peer_connections)]
    if coros:
        await asyncio.gather(*coros, return_exceptions=True)

    if _tts_engine:
        await _tts_engine.stop()
        _tts_engine = None
