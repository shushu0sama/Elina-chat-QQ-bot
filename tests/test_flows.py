import asyncio
import pytest
import time
from unittest.mock import Mock

from nonebot_plugin_personal_companion.flows import FlowManager


@pytest.fixture
def mock_llm():
    """Create a mock LLMClient that returns simple predictable responses."""
    llm = Mock()
    llm.chat.return_value = "好的，继续下一步。"
    return llm


@pytest.fixture
def flow_manager(mock_llm):
    """Create a FlowManager with a mock LLM."""
    return FlowManager(mock_llm)


def _run(coro):
    """Helper to run async coroutines in sync tests."""
    return asyncio.run(coro)


class TestSessionLifecycle:
    def test_has_session_none(self, flow_manager):
        assert flow_manager.has_session(1) is False

    def test_start_session(self, flow_manager):
        result = flow_manager.start_session(1, "process", "我压力好大")
        assert flow_manager.has_session(1) is True
        assert flow_manager.get_active_tool(1) == "process"
        assert result != ""

    def test_session_timeout(self, flow_manager):
        flow_manager.start_session(1, "process", "test")

        session = flow_manager._sessions[1]
        session.started_at = time.time() - flow_manager.SESSION_TIMEOUT_SECONDS - 1

        assert flow_manager.has_session(1) is False
        assert 1 not in flow_manager._sessions

    def test_check_exit(self, flow_manager):
        flow_manager.start_session(1, "process", "test")
        assert flow_manager.check_exit(1, "算了，不做了") is True
        assert flow_manager.check_exit(1, "继续") is False

    def test_exit_session(self, flow_manager):
        flow_manager.start_session(1, "process", "test")
        exit_msg = flow_manager.exit_session(1)
        assert 1 not in flow_manager._sessions
        assert exit_msg != ""

    def test_exit_session_none(self, flow_manager):
        assert flow_manager.exit_session(999) == ""


class TestToolDetection:
    def test_detect_process(self, flow_manager):
        for word in ["陪我走流程", "一起走流程", "带我走流程"]:
            result = flow_manager.detect_tool_request(word)
            assert result == "process", f"Failed for: {word}"

    def test_detect_mini(self, flow_manager):
        assert flow_manager.detect_tool_request("走迷你流程") == "mini_process"

    def test_detect_appreciation(self, flow_manager):
        assert flow_manager.detect_tool_request("赞赏感谢") == "appreciation"

    def test_detect_manifest_seed(self, flow_manager):
        assert flow_manager.detect_tool_request("我想显化一段关系") == "manifest_seed"
        assert flow_manager.detect_tool_request("帮我显化更好的事业") == "manifest_seed"

    def test_detect_manifest_diary(self, flow_manager):
        assert flow_manager.detect_tool_request("做显化日记") == "manifest_diary"
        assert flow_manager.detect_tool_request("今天有什么显化证据") == "manifest_diary"

    def test_detect_belief_rewrite(self, flow_manager):
        assert flow_manager.detect_tool_request("帮我改写信念") == "belief_rewrite"
        assert flow_manager.detect_tool_request("我不配拥有这个") == "belief_rewrite"

    def test_detect_obsession_downshift(self, flow_manager):
        assert flow_manager.detect_tool_request("我又执念了") == "obsession_downshift"
        assert flow_manager.detect_tool_request("为什么还没发生") == "obsession_downshift"

    def test_detect_future_self(self, flow_manager):
        assert flow_manager.detect_tool_request("让未来的我跟我说话") == "future_self"

    def test_detect_no_match(self, flow_manager):
        assert flow_manager.detect_tool_request("今天天气真好") is None

    def test_should_invite_process_strong(self, flow_manager):
        assert flow_manager.should_invite_process("我快崩溃了") is True
        assert flow_manager.should_invite_process("压力好大") is True

    def test_should_invite_process_no_match(self, flow_manager):
        assert flow_manager.should_invite_process("今天挺好的") is False

    def test_should_invite_appreciation(self, flow_manager):
        assert flow_manager.should_invite_appreciation("太开心了！") is True
        assert flow_manager.should_invite_appreciation("我拿到offer了") is True

    def test_should_invite_appreciation_no_match(self, flow_manager):
        assert flow_manager.should_invite_appreciation("今天很普通") is False


class TestAdvance:
    def test_advance_middle_step(self, flow_manager):
        flow_manager.start_session(1, "process", "test context")
        result = _run(flow_manager.advance(1, "我看到了身体里的紧张感"))
        assert result is not None
        assert result != ""

    def test_advance_last_step_completes(self, flow_manager):
        flow_manager.start_session(1, "mini_process", "test context")
        steps = flow_manager._get_steps("mini_process")

        # Advance through all steps
        for i in range(len(steps) - 1):
            result = _run(flow_manager.advance(1, f"step {i} response"))
            assert result is not None, f"Step {i} returned None"

        # Last step should complete and remove session
        result = _run(flow_manager.advance(1, "final response"))
        assert 1 not in flow_manager._sessions

    def test_advance_manifest_seed_records_answers_and_completes(self, flow_manager):
        flow_manager.start_session(1, "manifest_seed", "我想显化事业")
        steps = flow_manager._get_steps("manifest_seed")

        for i in range(len(steps) - 1):
            result = _run(flow_manager.advance(1, f"answer {i}"))
            assert result is not None

        result = _run(flow_manager.advance(1, "final action"))
        assert result is not None
        assert 1 not in flow_manager._sessions
        assert flow_manager.llm.chat.called

    def test_advance_no_session(self, flow_manager):
        assert _run(flow_manager.advance(999, "some response")) is None

    def test_multiple_sessions_independent(self, flow_manager):
        flow_manager.start_session(1, "process", "ctx1")
        flow_manager.start_session(2, "mini_process", "ctx2")

        assert flow_manager.get_active_tool(1) == "process"
        assert flow_manager.get_active_tool(2) == "mini_process"

        flow_manager.exit_session(1)
        assert flow_manager.has_session(1) is False
        assert flow_manager.has_session(2) is True


class TestGetSteps:
    def test_process_steps(self, flow_manager):
        steps = flow_manager._get_steps("process")
        assert len(steps) >= 1

    def test_mini_process_steps(self, flow_manager):
        steps = flow_manager._get_steps("mini_process")
        assert len(steps) >= 1

    def test_appreciation_steps(self, flow_manager):
        steps = flow_manager._get_steps("appreciation")
        assert len(steps) >= 1
