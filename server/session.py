"""Per-session state tracking for full-duplex voice AI.

Provides an explicit state machine that tracks whether the AI is idle,
listening to the user, thinking (LLM generating), or speaking (TTS playing).
Used by barge-in detection to decide if user speech is an interruption.
"""

import enum
import logging

logger = logging.getLogger(__name__)


class SessionPhase(enum.Enum):
    IDLE = "idle"
    LISTENING = "listening"
    THINKING = "thinking"
    SPEAKING = "speaking"


class SessionState:
    """Lightweight per-session phase tracker."""

    def __init__(self) -> None:
        self.phase = SessionPhase.IDLE

    def mark_idle(self) -> None:
        self.phase = SessionPhase.IDLE

    def mark_listening(self) -> None:
        self.phase = SessionPhase.LISTENING

    def mark_thinking(self) -> None:
        self.phase = SessionPhase.THINKING

    def mark_speaking(self) -> None:
        self.phase = SessionPhase.SPEAKING

    @property
    def is_speaking(self) -> bool:
        return self.phase == SessionPhase.SPEAKING

    @property
    def is_ai_active(self) -> bool:
        return self.phase in (SessionPhase.THINKING, SessionPhase.SPEAKING)
