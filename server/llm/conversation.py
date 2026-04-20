"""ConversationManager — per-session conversation history and state.

Tracks user/assistant messages, manages the system prompt, and handles
context window trimming via LLM summarization.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from server.config import settings
from server.llm.prompts import SUMMARIZATION_PROMPT, VOICE_SYSTEM_PROMPT

if TYPE_CHECKING:
    from server.llm.client import LLMClient

logger = logging.getLogger(__name__)


@dataclass
class MessageRecord:
    """A single message in conversation history."""

    role: str  # "user" | "assistant"
    content: str  # What was actually played/said
    generated_content: str | None = None  # Full LLM output (before interruption)
    was_interrupted: bool = False


class ConversationManager:
    """Per-session conversation history and state.

    One instance per peer connection. Tracks user/assistant messages,
    manages the system prompt, and handles context window trimming
    via LLM summarization.
    """

    def __init__(self, system_prompt: str | None = None) -> None:
        self._system_prompt = system_prompt or VOICE_SYSTEM_PROMPT
        self._messages: list[MessageRecord] = []
        self._summary: str | None = None

    def set_system_prompt(self, prompt: str) -> None:
        """Update the system prompt for this session."""
        self._system_prompt = prompt

    def add_user_message(self, text: str) -> None:
        """Record a user utterance."""
        self._messages.append(MessageRecord(role="user", content=text))

    def add_assistant_message(
        self,
        played_text: str,
        generated_text: str | None = None,
    ) -> None:
        """Record an assistant response.

        Args:
            played_text: What was actually delivered to the user
                (may be truncated if interrupted).
            generated_text: Full text the LLM generated. If None,
                defaults to played_text (no interruption).
        """
        was_interrupted = generated_text is not None and generated_text != played_text
        self._messages.append(
            MessageRecord(
                role="assistant",
                content=played_text,
                generated_content=generated_text,
                was_interrupted=was_interrupted,
            )
        )

    def get_messages(self) -> list[dict[str, str]]:
        """Return conversation in OpenAI API format.

        Uses content (what was played), not generated_content.
        """
        result: list[dict[str, str]] = [
            {"role": "system", "content": self._system_prompt}
        ]
        if self._summary:
            result.append(
                {
                    "role": "system",
                    "content": f"Previous conversation summary: {self._summary}",
                }
            )
        for msg in self._messages:
            result.append({"role": msg.role, "content": msg.content})
        return result

    async def maybe_summarize(self, llm_client: LLMClient) -> None:
        """Summarize older messages if history exceeds threshold.

        Replaces older messages with a condensed summary, keeping
        the system prompt and recent messages intact.
        """
        if len(self._messages) <= settings.llm_context_max_messages:
            return

        keep_count = settings.llm_context_keep_recent
        old_messages = self._messages[:-keep_count]
        recent_messages = self._messages[-keep_count:]

        # Format old messages for summarization
        transcript = "\n".join(
            f"{msg.role.upper()}: {msg.content}" for msg in old_messages
        )

        # Incorporate existing summary if present
        if self._summary:
            context = (
                f"Existing summary: {self._summary}\n\n"
                f"New messages to incorporate:\n{transcript}"
            )
        else:
            context = transcript

        summary_messages = [
            {"role": "system", "content": SUMMARIZATION_PROMPT},
            {"role": "user", "content": context},
        ]

        try:
            self._summary = await llm_client.complete(summary_messages)
            self._messages = list(recent_messages)
            logger.info(
                "Summarized %d messages into %d chars, keeping %d recent",
                len(old_messages),
                len(self._summary),
                len(recent_messages),
            )
        except Exception:
            logger.exception("Summarization failed, keeping full history")

    @property
    def message_count(self) -> int:
        """Number of messages (excluding system prompt)."""
        return len(self._messages)

    @property
    def history(self) -> list[MessageRecord]:
        """Read-only access to message history."""
        return list(self._messages)
