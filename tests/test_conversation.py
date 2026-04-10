"""Unit tests for ConversationManager."""

from unittest.mock import AsyncMock, patch

import pytest

from server.llm.conversation import ConversationManager


class TestInitialState:
    def test_empty_history(self):
        cm = ConversationManager()
        assert cm.message_count == 0

    def test_system_prompt_in_messages(self):
        cm = ConversationManager()
        messages = cm.get_messages()
        assert len(messages) == 1
        assert messages[0]["role"] == "system"

    def test_custom_system_prompt(self):
        cm = ConversationManager(system_prompt="Custom prompt")
        messages = cm.get_messages()
        assert messages[0]["content"] == "Custom prompt"


class TestMessageTracking:
    def test_add_user_message(self):
        cm = ConversationManager()
        cm.add_user_message("hello")
        assert cm.message_count == 1
        messages = cm.get_messages()
        assert messages[-1] == {"role": "user", "content": "hello"}

    def test_add_assistant_message(self):
        cm = ConversationManager()
        cm.add_assistant_message("hi there")
        assert cm.message_count == 1
        messages = cm.get_messages()
        assert messages[-1] == {"role": "assistant", "content": "hi there"}

    def test_conversation_order(self):
        cm = ConversationManager()
        cm.add_user_message("question")
        cm.add_assistant_message("answer")
        cm.add_user_message("follow up")
        messages = cm.get_messages()
        assert messages[0]["role"] == "system"
        assert messages[1] == {"role": "user", "content": "question"}
        assert messages[2] == {"role": "assistant", "content": "answer"}
        assert messages[3] == {"role": "user", "content": "follow up"}

    def test_history_returns_copy(self):
        cm = ConversationManager()
        cm.add_user_message("test")
        history = cm.history
        history.clear()
        assert cm.message_count == 1  # original unaffected


class TestInterruptionTracking:
    def test_normal_message_not_interrupted(self):
        cm = ConversationManager()
        cm.add_assistant_message("full response")
        record = cm.history[0]
        assert not record.was_interrupted
        assert record.generated_content is None

    def test_interrupted_message_uses_played_text(self):
        cm = ConversationManager()
        cm.add_assistant_message(
            played_text="Why did the",
            generated_text="Why did the scarecrow win an award? Because he was outstanding.",
        )
        messages = cm.get_messages()
        assert messages[-1]["content"] == "Why did the"

    def test_interrupted_flag_set(self):
        cm = ConversationManager()
        cm.add_assistant_message(
            played_text="partial",
            generated_text="partial response with more text",
        )
        record = cm.history[0]
        assert record.was_interrupted is True
        assert record.generated_content == "partial response with more text"

    def test_same_played_and_generated_not_interrupted(self):
        cm = ConversationManager()
        cm.add_assistant_message(
            played_text="complete response",
            generated_text="complete response",
        )
        record = cm.history[0]
        assert not record.was_interrupted


class TestSummarization:
    @pytest.mark.asyncio
    async def test_no_summarize_under_threshold(self):
        cm = ConversationManager()
        cm.add_user_message("hello")
        cm.add_assistant_message("hi")

        mock_client = AsyncMock()
        await cm.maybe_summarize(mock_client)

        mock_client.complete.assert_not_called()
        assert cm._summary is None

    @pytest.mark.asyncio
    async def test_summarize_when_threshold_exceeded(self):
        cm = ConversationManager()

        with patch("server.llm.conversation.settings") as mock_settings:
            mock_settings.llm_context_max_messages = 10
            mock_settings.llm_context_keep_recent = 4

            # Add 12 messages (6 exchanges)
            for i in range(6):
                cm.add_user_message(f"question {i}")
                cm.add_assistant_message(f"answer {i}")

            mock_client = AsyncMock()
            mock_client.complete = AsyncMock(return_value="Summary of conversation")

            await cm.maybe_summarize(mock_client)

            assert cm.message_count == 4  # kept recent
            assert cm._summary == "Summary of conversation"
            mock_client.complete.assert_called_once()

    @pytest.mark.asyncio
    async def test_summary_appears_in_messages(self):
        cm = ConversationManager()
        cm._summary = "Previous context about the user"
        cm.add_user_message("new question")

        messages = cm.get_messages()
        assert messages[0]["role"] == "system"  # system prompt
        assert messages[1]["role"] == "system"  # summary
        assert "Previous context" in messages[1]["content"]
        assert messages[2] == {"role": "user", "content": "new question"}

    @pytest.mark.asyncio
    async def test_incremental_summarization(self):
        cm = ConversationManager()
        cm._summary = "Existing summary"

        with patch("server.llm.conversation.settings") as mock_settings:
            mock_settings.llm_context_max_messages = 4
            mock_settings.llm_context_keep_recent = 2

            for i in range(3):
                cm.add_user_message(f"q{i}")
                cm.add_assistant_message(f"a{i}")

            mock_client = AsyncMock()
            mock_client.complete = AsyncMock(return_value="Updated summary")

            await cm.maybe_summarize(mock_client)

            # Check that existing summary was passed to the LLM
            call_args = mock_client.complete.call_args[0][0]
            user_content = call_args[1]["content"]
            assert "Existing summary" in user_content

            assert cm._summary == "Updated summary"

    @pytest.mark.asyncio
    async def test_summarization_failure_keeps_history(self):
        cm = ConversationManager()

        with patch("server.llm.conversation.settings") as mock_settings:
            mock_settings.llm_context_max_messages = 4
            mock_settings.llm_context_keep_recent = 2

            for i in range(3):
                cm.add_user_message(f"q{i}")
                cm.add_assistant_message(f"a{i}")

            original_count = cm.message_count

            mock_client = AsyncMock()
            mock_client.complete = AsyncMock(side_effect=Exception("API error"))

            await cm.maybe_summarize(mock_client)

            assert cm.message_count == original_count  # unchanged
            assert cm._summary is None
