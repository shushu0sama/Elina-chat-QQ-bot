import tempfile
from pathlib import Path

import pytest

from nonebot_plugin_personal_companion.memory import MemoryStore


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


def test_reminder_tables_created(store):
    conn = store._get_conn()
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    names = {row["name"] for row in tables}

    assert "reminders" in names
    assert "reminder_deliveries" in names


def test_create_reminder_persists_across_store_reopen(store):
    reminder_id = store.create_reminder(1, "喝水", "one_shot", "Asia/Shanghai", "2026-06-24T08:00", None, "2026-06-24T08:00")
    reopened = MemoryStore(store.db_path)

    rows = reopened.list_active_reminders(1)

    assert rows[0]["id"] == reminder_id
    assert rows[0]["text"] == "喝水"


def test_get_due_reminders_returns_active_due_only(store):
    due_id = store.create_reminder(1, "due", "one_shot", "Asia/Shanghai", "2026-06-23T09:00", None, "2026-06-23T09:00")
    store.create_reminder(1, "future", "one_shot", "Asia/Shanghai", "2026-06-23T11:00", None, "2026-06-23T11:00")
    completed_id = store.create_reminder(1, "done", "one_shot", "Asia/Shanghai", "2026-06-23T08:00", None, "2026-06-23T08:00")
    store.complete_reminder(completed_id)

    rows = store.get_due_reminders("2026-06-23T10:00")

    assert [row["id"] for row in rows] == [due_id]


def test_claim_reminder_occurrence_is_unique(store):
    reminder_id = store.create_reminder(1, "喝水", "one_shot", "Asia/Shanghai", "2026-06-24T08:00", None, "2026-06-24T08:00")

    assert store.claim_reminder_occurrence(reminder_id, "one_shot:1:2026-06-24T08:00") is True
    assert store.claim_reminder_occurrence(reminder_id, "one_shot:1:2026-06-24T08:00") is False


def test_complete_reminder_removes_from_due_query(store):
    reminder_id = store.create_reminder(1, "喝水", "one_shot", "Asia/Shanghai", "2026-06-23T09:00", None, "2026-06-23T09:00")

    store.complete_reminder(reminder_id)

    assert store.get_due_reminders("2026-06-23T10:00") == []


def test_advance_daily_reminder_updates_next_run_at(store):
    reminder_id = store.create_reminder(1, "写日记", "daily", "Asia/Shanghai", None, "21:00", "2026-06-23T21:00")

    store.advance_daily_reminder(reminder_id, "2026-06-24T21:00")

    rows = store.list_active_reminders(1)
    assert rows[0]["next_run_at"] == "2026-06-24T21:00"


def test_cancel_reminder_removes_from_active_list(store):
    reminder_id = store.create_reminder(1, "喝水", "one_shot", "Asia/Shanghai", "2026-06-24T08:00", None, "2026-06-24T08:00")

    assert store.cancel_reminder(1, reminder_id) is True

    assert store.list_active_reminders(1) == []
    assert store.cancel_reminder(1, reminder_id) is False


def test_cancel_reminder_is_user_scoped(store):
    reminder_id = store.create_reminder(1, "喝水", "one_shot", "Asia/Shanghai", "2026-06-24T08:00", None, "2026-06-24T08:00")

    assert store.cancel_reminder(2, reminder_id) is False

    assert len(store.list_active_reminders(1)) == 1
