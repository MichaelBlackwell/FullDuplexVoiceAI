"""CLI test for LLM streaming integration.

Usage: python -m server.llm.test_chat

Type messages and see streaming LLM responses with sentence chunking.
Each chunk is emitted at a sentence boundary (or clause for the first chunk).
"""

import asyncio
import sys

from server.llm.client import LLMClient
from server.llm.conversation import ConversationManager


async def main() -> None:
    client = LLMClient()
    await client.start()
    conversation = ConversationManager()

    print("LLM Chat Test (Ctrl+C to exit)")
    print(f"Model: {client._model} @ {client._base_url}")
    print("---")

    loop = asyncio.get_event_loop()
    chunk_count = 0

    try:
        while True:
            user_input = await loop.run_in_executor(
                None, lambda: input("\nYou: ")
            )
            if not user_input.strip():
                continue

            conversation.add_user_message(user_input)
            messages = conversation.get_messages()

            print("AI: ", end="", flush=True)
            chunk_count = 0

            async def print_chunk(chunk: str) -> None:
                nonlocal chunk_count
                chunk_count += 1
                if chunk_count > 1:
                    # Visual separator between chunks (for debugging)
                    print(f" [{chunk_count}] ", end="", flush=True)
                print(chunk, end="", flush=True)

            full_response = await client.stream_completion(
                messages=messages,
                on_chunk=print_chunk,
            )
            print()  # newline after response

            conversation.add_assistant_message(full_response)

            # Check summarization
            await conversation.maybe_summarize(client)

    except (KeyboardInterrupt, EOFError):
        print("\nBye!")
    finally:
        await client.stop()


if __name__ == "__main__":
    asyncio.run(main())
