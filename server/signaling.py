import asyncio
import logging
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from server.connection import (
    apply_session_settings,
    create_peer_connection,
    preload_tts,
    shutdown_all,
)
from server.config import settings
from server.llm.prompts import VOICE_SYSTEM_PROMPT

logger = logging.getLogger(__name__)

CLIENT_DIR = Path(__file__).resolve().parent.parent / "client"


class OfferRequest(BaseModel):
    sdp: str
    type: str


class OfferResponse(BaseModel):
    sdp: str
    type: str
    peer_id: int


# Maps peer_id -> asyncio.Queue of message dicts
# Messages: {"type": "transcript"|"llm_chunk"|"llm_done", "text": "..."}
transcript_queues: dict[int, asyncio.Queue[dict]] = {}


def create_app() -> FastAPI:
    app = FastAPI(title="Full-Duplex Voice AI")

    @app.post("/offer", response_model=OfferResponse)
    async def offer(request: OfferRequest):
        answer, peer_id = await create_peer_connection(request.sdp, request.type)
        # Create a transcript queue for this peer
        transcript_queues[peer_id] = asyncio.Queue()
        return OfferResponse(sdp=answer.sdp, type=answer.type, peer_id=peer_id)

    @app.websocket("/ws/transcripts")
    async def ws_transcripts(websocket: WebSocket):
        await websocket.accept()
        # Find the most recent peer's queue
        # (simple approach — works for single-user)
        queue = None
        try:
            while True:
                if queue is None and transcript_queues:
                    # Get the latest queue
                    queue = list(transcript_queues.values())[-1]
                if queue:
                    try:
                        message = await asyncio.wait_for(queue.get(), timeout=1.0)
                        await websocket.send_json(message)
                    except asyncio.TimeoutError:
                        pass  # No message yet, keep waiting
                else:
                    await asyncio.sleep(0.5)
        except WebSocketDisconnect:
            logger.info("Transcript WebSocket disconnected")

    # --- Settings API ---

    AVAILABLE_VOICES = [
        "ryan", "aiden", "vivian", "serena",
        "uncle_fu", "dylan", "eric", "ono_anna", "sohee",
    ]

    @app.get("/settings/defaults")
    async def get_defaults():
        return {
            "voices": AVAILABLE_VOICES,
            "default_voice": settings.tts_voice,
            "default_instruct": settings.tts_instruct,
            "default_system_prompt": VOICE_SYSTEM_PROMPT,
        }

    class SettingsRequest(BaseModel):
        peer_id: int
        voice: str | None = None
        instruct: str | None = None
        system_prompt: str | None = None

    @app.post("/settings")
    async def update_settings(request: SettingsRequest):
        result = apply_session_settings(
            request.peer_id, request.voice, request.instruct, request.system_prompt
        )
        return {"status": "ok", **result}

    @app.get("/health")
    async def health():
        return {"status": "ok"}

    @app.on_event("startup")
    async def on_startup():
        logger.info("Pre-loading TTS engine...")
        await preload_tts()
        logger.info("TTS engine ready")

    @app.on_event("shutdown")
    async def on_shutdown():
        logger.info("Shutting down, closing all peer connections")
        await shutdown_all()

    # Serve client files — must be last so API routes take priority
    app.mount("/", StaticFiles(directory=str(CLIENT_DIR), html=True), name="client")

    return app
