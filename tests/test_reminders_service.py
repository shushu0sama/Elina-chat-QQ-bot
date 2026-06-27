import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest

from nonebot_plugin_personal_companion.config import Config
from nonebot_plugin_personal_companion.memory import MemoryStore
from nonebot_plugin_personal_companion.reminders import ReminderService


def _run(coro):
    return asyncio.run(coro)


@pytest.fixture
def store():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = tmp.name
    tmp.close()
    s = MemoryStore(db_path=db_path)
    yield s
    del s
    for ext in ("", "-wal", "-shm"):
        try:
            Path(db_path + ext).unlink()
        except OSError:
            pass


@pytest.fixture
def service(store):
    return ReminderService(store, Config(reminder_timezone="Asia/Shanghai", reminder_max_due_per_scan=20))


def test_create_from_parse_persists_reminder(service, store):
    parsed = service.try_parse("明天晚上8点提醒我做作业")

    reminder_id = service.create_from_parse(1, parsed)

    rows = store.list_active_reminders(1)
    assert rows[0]["id"] == reminder_id
    assert rows[0]["text"] == "做作业"


def test_scan_sends_one_shot_once_and_completes(store, service):
    store.create_reminder(1, "喝水", "one_shot", "Asia/Shanghai", "2000-01-01T08:00", None, "2000-01-01T08:00")
    bot = Mock()
    bot.send_private_msg = AsyncMock()

    with patch("nonebot.get_bot", return_value=bot):
        _run(service.scan_and_send_due())
        _run(service.scan_and_send_due())

    bot.send_private_msg.assert_awaited_once_with(user_id=1, message="你让我提醒你：喝水")
    assert store.get_due_reminders("2000-01-01T09:00") == []


def test_scan_sends_daily_and_advances_next_run(store, service):
    store.create_reminder(1, "写日记", "daily", "Asia/Shanghai", None, "21:00", "2000-01-01T21:00")
    bot = Mock()
    bot.send_private_msg = AsyncMock()

    with patch("nonebot.get_bot", return_value=bot):
        _run(service.scan_and_send_due())

    bot.send_private_msg.assert_awaited_once_with(user_id=1, message="你让我提醒你：写日记")
    rows = store.list_active_reminders(1)
    assert rows[0]["status"] == "active"
    assert rows[0]["next_run_at"] > "2000-01-01T21:00"


def test_scan_handles_bot_unavailable(store, service):
    store.create_reminder(1, "喝水", "one_shot", "Asia/Shanghai", "2000-01-01T08:00", None, "2000-01-01T08:00")

    with patch("nonebot.get_bot", side_effect=RuntimeError("no bot")):
        _run(service.scan_and_send_due())

    assert len(store.get_due_reminders("2000-01-01T09:00")) == 1


def test_scan_marks_failed_send_without_crashing(store, service):
    store.create_reminder(1, "喝水", "one_shot", "Asia/Shanghai", "2000-01-01T08:00", None, "2000-01-01T08:00")
    bot = Mock()
    bot.send_private_msg = AsyncMock(side_effect=RuntimeError("send failed"))

    with patch("nonebot.get_bot", return_value=bot):
        _run(service.scan_and_send_due())

    assert len(store.get_due_reminders("2000-01-01T09:00")) == 1


def test_format_active_reminders_lists_ids_and_times(store, service):
    store.create_reminder(1, "喝水", "one_shot", "Asia/Shanghai", "2026-06-24T08:00", None, "2026-06-24T08:00")
    store.create_reminder(1, "写日记", "daily", "Asia/Shanghai", None, "21:00", "2026-06-24T21:00")

    reply = service.handle_management_command(1, "查看提醒")

    assert "你现在有这些提醒" in reply
    assert "#1 [一次] 2026-06-24 08:00：喝水" in reply
    assert "#2 [每天] 2026-06-24 21:00：写日记" in reply


def test_format_active_reminders_empty(service):
    assert service.handle_management_command(1, "我的提醒") == "你现在没有待提醒事项。"


def test_cancel_reminder_command(store, service):
    reminder_id = store.create_reminder(1, "喝水", "one_shot", "Asia/Shanghai", "2026-06-24T08:00", None, "2026-06-24T08:00")

    reply = service.handle_management_command(1, f"取消提醒 {reminder_id}")

    assert reply == f"已取消提醒 #{reminder_id}。"
    assert store.list_active_reminders(1) == []


def test_cancel_reminder_command_rejects_missing_id(service):
    reply = service.handle_management_command(1, "取消提醒 999")

    assert "没找到可取消的提醒 #999" in reply


def test_cancel_all_reminders_requires_specific_ids(service):
    reply = service.handle_management_command(1, "取消所有提醒")

    assert "只支持按编号取消" in reply
