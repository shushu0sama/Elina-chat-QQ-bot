from datetime import datetime
from zoneinfo import ZoneInfo

from nonebot_plugin_personal_companion.reminders import parse_reminder

TZ = ZoneInfo("Asia/Shanghai")
NOW = datetime(2026, 6, 23, 10, 0, tzinfo=TZ)


def test_parse_one_shot_tomorrow_morning():
    result = parse_reminder("明天8点提醒我做作业", now=NOW)

    assert result is not None
    assert result.ok
    assert result.kind == "one_shot"
    assert result.content == "做作业"
    assert result.next_run_at == datetime(2026, 6, 24, 8, 0, tzinfo=TZ)


def test_parse_one_shot_tomorrow_evening():
    result = parse_reminder("明天晚上8点提醒我做作业", now=NOW)

    assert result is not None
    assert result.ok
    assert result.next_run_at == datetime(2026, 6, 24, 20, 0, tzinfo=TZ)


def test_parse_one_shot_today_afternoon():
    result = parse_reminder("今天下午3点提醒我开会", now=NOW)

    assert result is not None
    assert result.ok
    assert result.next_run_at == datetime(2026, 6, 23, 15, 0, tzinfo=TZ)
    assert result.content == "开会"


def test_parse_daily_evening():
    result = parse_reminder("每天晚上9点提醒我写日记", now=NOW)

    assert result is not None
    assert result.ok
    assert result.kind == "daily"
    assert result.content == "写日记"
    assert result.time_of_day == "21:00"
    assert result.next_run_at == datetime(2026, 6, 23, 21, 0, tzinfo=TZ)


def test_parse_daily_time_already_passed_moves_to_tomorrow():
    result = parse_reminder("每天8点提醒我喝水", now=NOW)

    assert result is not None
    assert result.ok
    assert result.next_run_at == datetime(2026, 6, 24, 8, 0, tzinfo=TZ)


def test_daily_without_time_asks_clarification():
    result = parse_reminder("以后每天提醒我喝水", now=NOW)

    assert result is not None
    assert not result.ok
    assert "每天几点" in result.clarification


def test_parse_daily_without_prefix_asks_clarification():
    result = parse_reminder("每天提醒我喝水", now=NOW)

    assert result is not None
    assert not result.ok
    assert result.kind == "daily"
    assert result.content == "喝水"
    assert "每天几点" in result.clarification


def test_parse_daily_with_time_after_remind_keyword():
    result = parse_reminder("提醒我每天早上8点喝水", now=NOW)

    assert result is not None
    assert result.ok
    assert result.kind == "daily"
    assert result.content == "喝水"
    assert result.time_of_day == "08:00"


def test_parse_daily_tiantian_evening():
    result = parse_reminder("天天晚上9点提醒我写日记", now=NOW)

    assert result is not None
    assert result.ok
    assert result.kind == "daily"
    assert result.time_of_day == "21:00"
    assert result.content == "写日记"


def test_one_shot_without_specific_time_asks_clarification():
    result = parse_reminder("明天下午提醒我上课", now=NOW)

    assert result is not None
    assert not result.ok
    assert result.kind == "one_shot"
    assert result.content == "上课"
    assert "明天下午几点" in result.clarification


def test_parse_one_shot_date_after_remind_keyword():
    result = parse_reminder("提醒我明天晚上8点上课", now=NOW)

    assert result is not None
    assert result.ok
    assert result.next_run_at == datetime(2026, 6, 24, 20, 0, tzinfo=TZ)
    assert result.content == "上课"


def test_parse_one_shot_tomorrow_afternoon_explicit_time():
    result = parse_reminder("明天下午3点提醒我上课", now=NOW)

    assert result is not None
    assert result.ok
    assert result.next_run_at == datetime(2026, 6, 24, 15, 0, tzinfo=TZ)
    assert result.content == "上课"


def test_future_plan_without_reminder_returns_none():
    assert parse_reminder("我明天下午要上课", now=NOW) is None


def test_non_reminder_returns_none():
    assert parse_reminder("今天聊得很开心", now=NOW) is None
