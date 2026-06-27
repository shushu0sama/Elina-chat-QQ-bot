import pytest
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from nonebot_plugin_personal_companion.feishu_calendar import (
    ParsedCalendarRequest,
    FeishuCalendarClient,
    format_calendar_event_confirmation,
    format_calendar_intent_confirmation,
    looks_like_calendar_request,
    parse_calendar_request,
    should_confirm_calendar_request,
)


FIXED_NOW = datetime(2026, 6, 14, 9, 0, tzinfo=ZoneInfo("Asia/Shanghai"))


def test_looks_like_calendar_request_matches_common_phrases():
    assert looks_like_calendar_request("明天下午三点提醒我开会") is True
    assert looks_like_calendar_request("帮我记个会 周一 9:30 站会") is True
    assert looks_like_calendar_request("今天午饭吃什么") is False


def test_weak_calendar_intent_requires_confirmation():
    request = parse_calendar_request("明天下午三点开会", now=FIXED_NOW)

    assert should_confirm_calendar_request("明天下午三点开会") is True
    assert should_confirm_calendar_request("提醒我明天下午三点开会") is False
    assert "你是想让我帮你记到日历里" in format_calendar_intent_confirmation(request)


def test_parse_calendar_request_with_relative_time():
    request = parse_calendar_request("提醒我明天下午3点开会", now=FIXED_NOW)

    assert request.ok is True
    assert request.summary == "开会"
    assert request.start == datetime(2026, 6, 15, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    assert request.end == datetime(2026, 6, 15, 15, 30, tzinfo=ZoneInfo("Asia/Shanghai"))


def test_parse_calendar_request_with_time_range():
    request = parse_calendar_request("安排周一 9:30到10:30 站会", now=FIXED_NOW)

    assert request.ok is True
    assert request.summary == "站会"
    assert request.start == datetime(2026, 6, 15, 9, 30, tzinfo=ZoneInfo("Asia/Shanghai"))
    assert request.end == datetime(2026, 6, 15, 10, 30, tzinfo=ZoneInfo("Asia/Shanghai"))


def test_parse_calendar_request_with_month_day():
    request = parse_calendar_request("安排6月20日下午3点项目复盘", now=FIXED_NOW)

    assert request.ok is True
    assert request.summary == "项目复盘"
    assert request.start == datetime(2026, 6, 20, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai"))


def test_parse_calendar_request_with_day_only():
    request = parse_calendar_request("20号晚上8点聚餐", now=FIXED_NOW)

    assert request.ok is True
    assert request.summary == "聚餐"
    assert request.start == datetime(2026, 6, 20, 20, 0, tzinfo=ZoneInfo("Asia/Shanghai"))


def test_parse_calendar_request_with_chinese_number_time():
    request = parse_calendar_request("明天下午三点开会", now=FIXED_NOW)

    assert request.ok is True
    assert request.summary == "开会"
    assert request.start == datetime(2026, 6, 15, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai"))


def test_parse_calendar_request_with_numeric_slash_date():
    request = parse_calendar_request("6/20 19:30 电影", now=FIXED_NOW)

    assert request.ok is True
    assert request.summary == "电影"
    assert request.start == datetime(2026, 6, 20, 19, 30, tzinfo=ZoneInfo("Asia/Shanghai"))


def test_parse_calendar_request_invalid_date_returns_clarification():
    request = parse_calendar_request("安排6月31日下午3点复盘", now=FIXED_NOW)

    assert request.ok is False
    assert "时间" in request.clarification


def test_parse_calendar_request_invalid_time_returns_clarification():
    request = parse_calendar_request("提醒我明天25点开会", now=FIXED_NOW)

    assert request.ok is False
    assert "时间" in request.clarification


def test_parse_calendar_request_same_weekday_past_time_uses_next_week():
    now = datetime(2026, 6, 15, 10, 0, tzinfo=ZoneInfo("Asia/Shanghai"))
    request = parse_calendar_request("周一 9:30 站会", now=now)

    assert request.ok is True
    assert request.start == datetime(2026, 6, 22, 9, 30, tzinfo=ZoneInfo("Asia/Shanghai"))


def test_parse_calendar_request_missing_time_returns_clarification():
    request = parse_calendar_request("提醒我开会", now=FIXED_NOW)

    assert request.ok is False
    assert "具体时间" in request.clarification or "时间" in request.clarification


def test_format_calendar_event_confirmation_uses_summary_and_time():
    request = ParsedCalendarRequest(
        ok=True,
        summary="开会",
        start=datetime(2026, 6, 15, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
        end=datetime(2026, 6, 15, 15, 30, tzinfo=ZoneInfo("Asia/Shanghai")),
    )

    text = format_calendar_event_confirmation(request)

    assert "开会" in text
    assert "2026-06-15 15:00" in text
    assert "15:30" in text
