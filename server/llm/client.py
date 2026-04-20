"""LLMClient — async streaming wrapper around the OpenAI SDK.

Shared singleton across sessions. Uses openai.AsyncOpenAI with custom
base_url for provider flexibility (DeepSeek, OpenAI, any compatible API).
"""

import asyncio
import logging
from collections.abc import Awaitable, Callable

from openai import AsyncOpenAI

from server.config import settings

logger = logging.getLogger(__name__)

TokenCallback = Callable[[str], Awaitable[None]]


class LLMClient:
    """Async streaming LLM client.

    Thread-safe for concurrent sessions because cancellation is scoped
    to a per-stream asyncio.Event, not to the client instance.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        self._base_url = base_url or settings.llm_base_url
        self._api_key = api_key or settings.llm_api_key
        self._model = model or settings.llm_model
        self._client: AsyncOpenAI | None = None

    async def start(self) -> None:
        """Create the AsyncOpenAI client."""
        if not self._api_key:
            raise ValueError(
                "LLM_API_KEY environment variable is not set. "
                "Set it to your API key (e.g. export LLM_API_KEY=sk-...)."
            )
        self._client = AsyncOpenAI(
            base_url=self._base_url,
            api_key=self._api_key,
        )
        logger.info(
            "LLMClient started: model=%s base_url=%s",
            self._model,
            self._base_url,
        )

    async def stream_completion(
        self,
        messages: list[dict[str, str]],
        on_token: TokenCallback | None = None,
        cancel_event: asyncio.Event | None = None,
    ) -> str:
        """Stream a chat completion, calling on_token with each token as it arrives.

        Args:
            messages: Conversation history in OpenAI format.
            on_token: Called with each token for real-time display.
            cancel_event: If set, streaming stops and partial output is returned.

        Returns:
            The complete generated text (all tokens concatenated).
        """
        full_response = ""

        stream = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            stream=True,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
        )

        try:
            async for event in stream:
                if cancel_event and cancel_event.is_set():
                    break

                delta = event.choices[0].delta
                if delta.content:
                    full_response += delta.content
                    if on_token:
                        await on_token(delta.content)
        finally:
            await stream.close()

        return full_response

    async def complete(self, messages: list[dict[str, str]]) -> str:
        """Non-streaming completion. Used for summarization."""
        response = await self._client.chat.completions.create(
            model=self._model,
            messages=messages,
            max_tokens=settings.llm_max_tokens,
            temperature=0.3,
        )
        return response.choices[0].message.content

    async def stop(self) -> None:
        """Close the underlying HTTP client."""
        if self._client:
            await self._client.close()
            self._client = None
            logger.info("LLMClient stopped")
