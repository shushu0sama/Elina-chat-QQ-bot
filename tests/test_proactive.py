from unittest.mock import Mock
from datetime import datetime, timedelta, timezone

from nonebot_plugin_personal_companion.proactive import ProactiveChat


def test_filter_allowed_ignores_non_numeric_tokens():
    config = Mock(proactive_allow_users="1, abc, 2 # comment")
    chat = ProactiveChat(Mock(), Mock(), config)

    assert chat._filter_allowed([1, 2, 3]) == [1]


def test_should_send_respects_proactive_snooze():
    memory = Mock()
    memory.get_proactive_snooze_until.return_value = (datetime.now(timezone.utc) + timedelta(hours=1)).strftime("%Y-%m-%d %H:%M:%S")
    config = Mock(proactive_cooldown_minutes=0, proactive_interval_minutes=0)
    chat = ProactiveChat(memory, Mock(), config)

    assert chat._should_send(1, datetime.now(timezone.utc)) is False


