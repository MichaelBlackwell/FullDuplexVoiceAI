"""SentenceChunker — buffers streaming tokens and emits at sentence boundaries.

First-chunk optimization: the first emission uses clause boundaries (commas)
to minimize time-to-first-audio. Subsequent chunks emit at sentence boundaries
(period, question mark, exclamation, or colon followed by whitespace).
"""

import re

_SENTENCE_END = re.compile(r"[.!?:]\s")
_CLAUSE_END = re.compile(r",\s")


class SentenceChunker:
    """Buffers streaming tokens and emits text at natural boundaries."""

    def __init__(self) -> None:
        self._buffer: str = ""
        self._first_chunk_emitted: bool = False

    def feed(self, token: str) -> str | None:
        """Feed a single token. Returns a chunk to emit, or None.

        First chunk emits at the first clause or sentence boundary.
        Subsequent chunks emit at the last sentence boundary.
        """
        self._buffer += token

        if not self._first_chunk_emitted:
            # Find the FIRST clause or sentence boundary for minimum latency
            match = _SENTENCE_END.search(self._buffer)
            if match is None:
                match = _CLAUSE_END.search(self._buffer)
            if match:
                # Split after the punctuation, keep the space in the remainder
                split_pos = match.start() + 1
                chunk = self._buffer[:split_pos].lstrip()
                self._buffer = self._buffer[split_pos:]
                self._first_chunk_emitted = True
                return chunk if chunk else None
        else:
            # Find the LAST sentence boundary to batch text
            match = None
            for m in _SENTENCE_END.finditer(self._buffer):
                match = m
            if match:
                split_pos = match.start() + 1
                # Keep leading space so concatenation preserves word spacing
                chunk = self._buffer[:split_pos]
                self._buffer = self._buffer[split_pos:]
                return chunk if chunk else None

        return None

    def flush(self) -> str | None:
        """Flush any remaining buffered text. Call when stream ends."""
        text = self._buffer if self._first_chunk_emitted else self._buffer.lstrip()
        self._buffer = ""
        return text if text else None

    def reset(self) -> None:
        """Reset state for a new response."""
        self._buffer = ""
        self._first_chunk_emitted = False
