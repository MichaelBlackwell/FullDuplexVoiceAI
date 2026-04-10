"""Unit tests for session state."""

from server.session import SessionPhase, SessionState


class TestSessionState:
    def test_initial_state_is_idle(self):
        s = SessionState()
        assert s.phase == SessionPhase.IDLE

    def test_mark_speaking(self):
        s = SessionState()
        s.mark_speaking()
        assert s.is_speaking is True
        assert s.is_ai_active is True

    def test_mark_thinking(self):
        s = SessionState()
        s.mark_thinking()
        assert s.is_speaking is False
        assert s.is_ai_active is True

    def test_mark_listening(self):
        s = SessionState()
        s.mark_speaking()
        s.mark_listening()
        assert s.is_speaking is False
        assert s.is_ai_active is False
        assert s.phase == SessionPhase.LISTENING

    def test_mark_idle(self):
        s = SessionState()
        s.mark_speaking()
        s.mark_idle()
        assert s.phase == SessionPhase.IDLE
        assert s.is_ai_active is False
