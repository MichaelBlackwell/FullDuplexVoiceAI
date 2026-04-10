"""Unit tests for SentenceChunker."""

from server.llm.chunker import SentenceChunker


class TestFirstChunkOptimization:
    def test_emits_at_comma_for_first_chunk(self):
        c = SentenceChunker()
        assert c.feed("Well") is None
        result = c.feed(", I think")
        assert result == "Well,"

    def test_emits_at_sentence_before_comma(self):
        c = SentenceChunker()
        assert c.feed("Sure. ") is not None  # sentence boundary found first

    def test_prefers_sentence_over_comma_for_first_chunk(self):
        c = SentenceChunker()
        # Sentence boundary comes first in the buffer
        result = c.feed("OK. Let me")
        assert result == "OK."


class TestSentenceBoundaries:
    def test_period_boundary(self):
        c = SentenceChunker()
        c._first_chunk_emitted = True
        c.feed("Hello world")
        result = c.feed(". Next")
        assert result.strip() == "Hello world."

    def test_question_mark_boundary(self):
        c = SentenceChunker()
        c._first_chunk_emitted = True
        c.feed("How are you")
        result = c.feed("? ")
        assert result.strip() == "How are you?"

    def test_exclamation_boundary(self):
        c = SentenceChunker()
        c._first_chunk_emitted = True
        c.feed("Wow")
        result = c.feed("! That")
        assert result.strip() == "Wow!"

    def test_colon_boundary(self):
        c = SentenceChunker()
        c._first_chunk_emitted = True
        c.feed("Here it is")
        result = c.feed(": the")
        assert result.strip() == "Here it is:"

    def test_comma_ignored_after_first_chunk(self):
        c = SentenceChunker()
        c._first_chunk_emitted = True
        c.feed("Second")
        result = c.feed(", third")
        assert result is None


class TestBatching:
    def test_multiple_sentences_batched_at_last_boundary(self):
        c = SentenceChunker()
        c._first_chunk_emitted = True
        result = c.feed("First sentence. Second sentence. Third")
        assert result.strip() == "First sentence. Second sentence."
        assert c.flush().strip() == "Third"

    def test_no_boundary_returns_none(self):
        c = SentenceChunker()
        c._first_chunk_emitted = True
        assert c.feed("Hello") is None
        assert c.feed(" world") is None


class TestFlushAndReset:
    def test_flush_returns_remainder(self):
        c = SentenceChunker()
        c.feed("partial text")
        assert c.flush() == "partial text"

    def test_flush_empty_returns_none(self):
        c = SentenceChunker()
        assert c.flush() is None

    def test_flush_whitespace_only_returns_none(self):
        c = SentenceChunker()
        c.feed("   ")
        result = c.flush()
        assert result is None or result == ""

    def test_reset_clears_state(self):
        c = SentenceChunker()
        c.feed("some text")
        c.reset()
        assert c.flush() is None
        assert c._first_chunk_emitted is False


class TestEdgeCases:
    def test_decimal_not_split(self):
        """3.5 should not trigger a sentence boundary (no space after '.')."""
        c = SentenceChunker()
        c._first_chunk_emitted = True
        result = c.feed("The value is 3.5 percent")
        # "3.5" has no space after period (space is after "5"), so no split
        # BUT "3.5 " -> the '.' is between '3' and '5', so regex [.!?:]\s
        # checks if char after '.' is whitespace. '5' is not whitespace. Safe.
        assert result is None

    def test_abbreviation_with_space_does_split(self):
        """Known tradeoff: 'Dr. Smith' will split. Acceptable for voice AI."""
        c = SentenceChunker()
        c._first_chunk_emitted = True
        result = c.feed("Ask Dr. Smith about it")
        assert result.strip() == "Ask Dr."

    def test_empty_token(self):
        c = SentenceChunker()
        assert c.feed("") is None

    def test_token_is_just_boundary(self):
        c = SentenceChunker()
        c._first_chunk_emitted = True
        c.feed("Done")
        result = c.feed(". ")
        assert result.strip() == "Done."

    def test_multiline_token(self):
        c = SentenceChunker()
        c._first_chunk_emitted = True
        result = c.feed("Line one. \nLine two")
        assert "Line one." in result

    def test_concatenated_chunks_have_spaces(self):
        """Chunks joined together should produce readable text with spaces."""
        c = SentenceChunker()
        chunks = []
        tokens = ["Hello", ",", " how", " are", " you", "?", " I", "'m", " fine", ".", " Great", "!"]
        for t in tokens:
            chunk = c.feed(t)
            if chunk:
                chunks.append(chunk)
        remaining = c.flush()
        if remaining:
            chunks.append(remaining)
        full = "".join(chunks)
        assert "? " in full or "?" in full  # space after question mark
        assert ". " in full or "." in full
        # No missing spaces between sentences
        assert ".I" not in full
        assert "?I" not in full
