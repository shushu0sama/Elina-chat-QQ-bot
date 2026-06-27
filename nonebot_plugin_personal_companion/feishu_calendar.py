from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import httpx

FEISHU_AUTH_URL = "https://open.feishu.cn/open-apis/auth/v3/tenant_access_token/internal"
FEISHU_CREATE_EVENT_PATH = "/open-apis/calendar/v4/calendars/{calendar_id}/events"

_WEEKDAY_MAP = {
    "周一": 0, "星期一": 0, "礼拜一": 0,
    "周二": 1, "星期二": 1, "礼拜二": 1,
    "周三": 2, "星期三": 2, "礼拜三": 2,
    "周四": 3, "星期四": 3, "礼拜四": 3,
    "周五": 4, "星期五": 4, "礼拜五": 4,
    "周六": 5, "星期六": 5, "礼拜六": 5,
    "周日": 6, "周天": 6, "星期日": 6, "星期天": 6, "礼拜日": 6, "礼拜天": 6,
}

_PREFIX_RE = re.compile(r"^(?:提醒我|安排一下|安排|创建日程|建日程|帮我记个会|记个会|帮我预约|预约一下|约一下)[:：\s]*")
_RELATIVE_DAY_PATTERNS = (
    ("大后天", 3),
    ("后天", 2),
    ("明天", 1),
    ("今天", 0),
)
_CN_NUMBERS = {
    "零": 0, "〇": 0,
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    "十一": 11, "十二": 12, "十三": 13, "十四": 14, "十五": 15,
    "十六": 16, "十七": 17, "十八": 18, "十九": 19, "二十": 20,
    "二十一": 21, "二十二": 22, "二十三": 23, "二十四": 24,
    "三十": 30, "三十一": 31,
}

_TIME_HOUR = r"\d{1,2}|零|〇|一|二|两|三|四|五|六|七|八|九|十|十一|十二|十三|十四|十五|十六|十七|十八|十九|二十|二十一|二十二|二十三|二十四"
_TIME_MINUTE = r"\d{1,2}|零|〇|一|二|两|三|四|五|六|七|八|九|十|十一|十二|十三|十四|十五|十六|十七|十八|十九|二十|二十一|二十二|二十三|二十四|三十|三十一"
_TIME_PATTERNS = [
    re.compile(rf"(?P<period>上午|中午|下午|晚上|凌晨|早上|傍晚)?\s*(?P<hour>{_TIME_HOUR})[:：](?P<minute>\d{{1,2}})"),
    re.compile(rf"(?P<period>上午|中午|下午|晚上|凌晨|早上|傍晚)?\s*(?P<hour>{_TIME_HOUR})点(?P<minute>{_TIME_MINUTE})分?"),
    re.compile(rf"(?P<period>上午|中午|下午|晚上|凌晨|早上|傍晚)?\s*(?P<hour>{_TIME_HOUR})点半"),
    re.compile(rf"(?P<period>上午|中午|下午|晚上|凌晨|早上|傍晚)?\s*(?P<hour>{_TIME_HOUR})点"),
]


@dataclass(slots=True)
class ParsedCalendarRequest:
    ok: bool
    summary: str = ""
    start: datetime | None = None
    end: datetime | None = None
    description: str = ""
    clarification: str = ""


class FeishuCalendarClient:
    def __init__(self, app_id: str, app_secret: str, calendar_id: str, timezone_name: str = "Asia/Shanghai"):
        self.app_id = app_id
        self.app_secret = app_secret
        self.calendar_id = calendar_id
        self.timezone = ZoneInfo(timezone_name)
        self._tenant_access_token: str | None = None
        self._tenant_access_token_expiry = datetime.fromtimestamp(0, tz=timezone.utc)

    async def _get_tenant_access_token(self) -> str:
        now = datetime.now(timezone.utc)
        if self._tenant_access_token and now < self._tenant_access_token_expiry:
            return self._tenant_access_token

        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                FEISHU_AUTH_URL,
                json={"app_id": self.app_id, "app_secret": self.app_secret},
                headers={"Content-Type": "application/json"},
            )
            response.raise_for_status()
            payload = response.json()

        if payload.get("code", 0) != 0:
            raise RuntimeError(payload.get("msg", "获取飞书 tenant_access_token 失败"))

        token = payload.get("tenant_access_token") or payload.get("data", {}).get("tenant_access_token")
        expire = int(payload.get("expire", 3600))
        if not token:
            raise RuntimeError("飞书返回了空的 tenant_access_token")

        self._tenant_access_token = token
        self._tenant_access_token_expiry = now + timedelta(seconds=max(expire - 60, 60))
        return token

    async def create_event(self, summary: str, start: datetime, end: datetime, description: str = "") -> dict:
        token = await self._get_tenant_access_token()
        body = {
            "summary": summary,
            "description": description,
            "need_notification": True,
            "start_time": {
                "timestamp": str(int(start.astimezone(timezone.utc).timestamp())),
                "timezone": self.timezone.key,
            },
            "end_time": {
                "timestamp": str(int(end.astimezone(timezone.utc).timestamp())),
                "timezone": self.timezone.key,
            },
        }

        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                f"https://open.feishu.cn{FEISHU_CREATE_EVENT_PATH.format(calendar_id=self.calendar_id)}",
                json=body,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Content-Type": "application/json",
                },
            )
            response.raise_for_status()
            payload = response.json()

        if payload.get("code", 0) != 0:
            raise RuntimeError(payload.get("msg", "创建飞书日程失败"))

        return payload

    async def create_event_from_request(self, request: ParsedCalendarRequest) -> dict:
        if not request.ok or request.start is None or request.end is None:
            raise ValueError(request.clarification or "日程信息不完整")
        return await self.create_event(request.summary, request.start, request.end, request.description)


def has_strong_calendar_intent(text: str) -> bool:
    raw = text.strip()
    if not raw:
        return False
    strong_prefixes = (
        "提醒我", "安排一下", "安排", "创建日程", "建日程", "帮我记个会", "记个会",
        "帮我预约", "预约一下", "约一下",
    )
    if raw.startswith(strong_prefixes):
        return True
    return "提醒我" in raw or "帮我记" in raw or "创建日程" in raw or "建日程" in raw


def looks_like_calendar_request(text: str) -> bool:
    raw = text.strip()
    if not raw:
        return False
    if has_strong_calendar_intent(raw):
        return True
    date_markers = ("今天", "明天", "后天", "大后天", "下周", "下星期", "下礼拜", "本周", "这周", "星期", "周", "礼拜")
    time_markers = ("点半", "点", ":", "：")
    action_markers = ("会", "会议", "开会", "日程", "提醒", "预约", "面试", "面谈", "复盘", "站会", "讨论")
    return any(marker in raw for marker in date_markers) and any(marker in raw for marker in time_markers) and any(marker in raw for marker in action_markers)


def should_confirm_calendar_request(text: str) -> bool:
    return looks_like_calendar_request(text) and not has_strong_calendar_intent(text)


def format_calendar_intent_confirmation(request: ParsedCalendarRequest) -> str:
    if not request.ok or request.start is None:
        return "我听到你提到了一个具体时间。你是想让我帮你记到日历里，还是只是想聊聊这件事？"
    start = request.start.astimezone(request.start.tzinfo or timezone.utc)
    return (
        f"我听到你提到了「{request.summary}」（{start.strftime('%Y-%m-%d %H:%M')}）。"
        "你是想让我帮你记到日历里，还是只是想聊聊这件事？"
    )


def parse_calendar_request(text: str, now: datetime | None = None, timezone_name: str = "Asia/Shanghai", default_duration_minutes: int = 30) -> ParsedCalendarRequest:
    tz = ZoneInfo(timezone_name)
    now = now or datetime.now(tz)
    if now.tzinfo is None:
        now = now.replace(tzinfo=tz)
    else:
        now = now.astimezone(tz)

    raw = text.strip()
    if not raw:
        return ParsedCalendarRequest(ok=False, clarification="我还没看懂你要安排什么日程。")

    body = _PREFIX_RE.sub("", raw).strip()
    if body.startswith("帮我"):
        body = body[2:].strip()

    date_value = _parse_date(body, now)
    if date_value is None:
        return ParsedCalendarRequest(ok=False, clarification="我看到了日程内容，但还缺具体时间。你可以补成类似「明天下午3点开会」这样。")

    try:
        time_info = _parse_time(body)
    except ValueError:
        return ParsedCalendarRequest(ok=False, clarification="这个时间好像不太对。你可以改成类似「明天下午3点」或「6月20日 19:30」。")
    if time_info is None:
        return ParsedCalendarRequest(ok=False, clarification="我还没看懂时间。你可以试试「明天下午3点开会」或「周一 9:30 站会」。")

    start_hour, start_minute, start_end_index = time_info
    try:
        start = datetime.combine(date_value, datetime.min.time()).replace(hour=start_hour, minute=start_minute, tzinfo=tz)
    except ValueError:
        return ParsedCalendarRequest(ok=False, clarification="这个时间好像不太对。你可以改成类似「明天下午3点」或「6月20日 19:30」。")
    end = start + timedelta(minutes=default_duration_minutes)

    if _has_range_separator(body, start_end_index):
        second = _parse_second_time(body[start_end_index:])
        if second is not None:
            try:
                end = datetime.combine(date_value, datetime.min.time()).replace(hour=second[0], minute=second[1], tzinfo=tz)
            except ValueError:
                return ParsedCalendarRequest(ok=False, clarification="结束时间好像不太对。你可以改成类似「9:30到10:30」。")

    if end <= start:
        end = start + timedelta(minutes=default_duration_minutes)

    if start <= now and not _has_explicit_future_date(body):
        start += timedelta(days=1)
        end += timedelta(days=1)

    summary = _strip_datetime_words(body) or "日程"
    return ParsedCalendarRequest(ok=True, summary=summary, start=start, end=end, description=raw)


def format_calendar_event_confirmation(request: ParsedCalendarRequest) -> str:
    assert request.start is not None and request.end is not None
    start = request.start.astimezone(request.start.tzinfo or timezone.utc)
    end = request.end.astimezone(request.end.tzinfo or timezone.utc)
    return (
        f"我已经帮你记到飞书日历里了：{request.summary}\n"
        f"时间：{start.strftime('%Y-%m-%d %H:%M')} - {end.strftime('%H:%M')}"
    )


def _parse_date(body: str, now: datetime) -> date | None:
    for token, offset in _RELATIVE_DAY_PATTERNS:
        if token in body:
            return now.date() + timedelta(days=offset)

    week_modifier = None
    if "下周" in body or "下星期" in body or "下礼拜" in body:
        week_modifier = 1
    elif "本周" in body or "这周" in body or "这星期" in body or "本星期" in body:
        week_modifier = 0

    for token, weekday in _WEEKDAY_MAP.items():
        if token in body:
            if week_modifier == 1:
                week_start = now.date() + timedelta(days=(7 - now.weekday()))
                return week_start + timedelta(days=weekday)
            if week_modifier == 0:
                week_start = now.date() - timedelta(days=now.weekday())
                candidate = week_start + timedelta(days=weekday)
                return candidate if candidate >= now.date() else candidate + timedelta(days=7)
            days_ahead = (weekday - now.weekday() + 7) % 7
            candidate = now.date() + timedelta(days=days_ahead)
            if candidate == now.date() and _time_is_already_past_today(body, now):
                candidate += timedelta(days=7)
            return candidate

    match = re.search(r"(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})日?", body)
    if match:
        year, month, day = (int(x) for x in match.groups())
        try:
            return date(year, month, day)
        except ValueError:
            return None

    match = re.search(r"(?<!\d)(\d{1,2})[-/.月](\d{1,2})[日号]?", body)
    if match:
        month, day = (int(x) for x in match.groups())
        try:
            candidate = date(now.year, month, day)
            if candidate < now.date():
                candidate = date(now.year + 1, month, day)
            return candidate
        except ValueError:
            return None

    match = re.search(r"(?<!\d)(\d{1,2})[日号](?!\d)", body)
    if match:
        day = int(match.group(1))
        candidate = _date_with_day_in_current_or_next_month(now.date(), day)
        if candidate:
            return candidate
    return None


def _parse_time(body: str) -> tuple[int, int, int] | None:
    for pattern in _TIME_PATTERNS:
        match = pattern.search(body)
        if not match:
            continue
        period = match.groupdict().get("period") or ""
        hour = match.group("hour")
        if pattern.pattern.endswith("点半"):
            minute = "30"
        else:
            minute = match.groupdict().get("minute") or "0"
        hour_value, minute_value = _normalize_clock(period, hour, minute)
        return hour_value, minute_value, match.end()
    return None


def _parse_second_time(body: str) -> tuple[int, int] | None:
    time_info = _parse_time(body)
    if time_info is None:
        return None
    return time_info[0], time_info[1]


def _parse_int(text: str) -> int:
    if text.isdigit():
        return int(text)
    if text in _CN_NUMBERS:
        return _CN_NUMBERS[text]
    if text.startswith("十") and len(text) == 2:
        return 10 + _CN_NUMBERS.get(text[1], 0)
    if text.endswith("十") and len(text) == 2:
        return _CN_NUMBERS.get(text[0], 0) * 10
    if "十" in text:
        left, right = text.split("十", 1)
        return _CN_NUMBERS.get(left, 1) * 10 + _CN_NUMBERS.get(right, 0)
    return int(text)


def _time_is_already_past_today(body: str, now: datetime) -> bool:
    try:
        time_info = _parse_time(body)
    except ValueError:
        return False
    if time_info is None:
        return False
    hour, minute, _ = time_info
    return (hour, minute) <= (now.hour, now.minute)


def _date_with_day_in_current_or_next_month(today: date, day: int) -> date | None:
    for month_offset in (0, 1):
        year = today.year
        month = today.month + month_offset
        if month > 12:
            month -= 12
            year += 1
        try:
            candidate = date(year, month, day)
        except ValueError:
            continue
        if candidate >= today:
            return candidate
    return None


def _normalize_clock(period: str, hour_text: str, minute_text: str) -> tuple[int, int]:
    hour = _parse_int(hour_text)
    minute = _parse_int(minute_text)

    if period in {"下午", "晚上", "傍晚"} and hour < 12:
        hour += 12
    elif period == "中午":
        if hour == 0:
            hour = 12
        elif hour < 11:
            hour += 12
    elif period == "凌晨" and hour == 12:
        hour = 0
    elif period in {"上午", "早上"} and hour == 12:
        hour = 0

    return hour, minute


def _has_range_separator(body: str, index: int) -> bool:
    between = body[max(0, index - 3): index + 3]
    return any(token in between for token in ("到", "至", "-", "—", "~"))


def _has_explicit_future_date(body: str) -> bool:
    return any(token in body for token in ("明天", "后天", "大后天", "下周", "下星期", "下礼拜", "本周", "这周", "这星期", "本星期"))


def _strip_datetime_words(body: str) -> str:
    cleaned = body
    for token in ("今天", "明天", "后天", "大后天", "本周", "这周", "这星期", "本星期", "下周", "下星期", "下礼拜"):
        cleaned = cleaned.replace(token, "")
    for token in _WEEKDAY_MAP:
        cleaned = cleaned.replace(token, "")
    for pattern in _TIME_PATTERNS:
        cleaned = pattern.sub("", cleaned)
    cleaned = re.sub(r"(从|自)?\s*(到|至)\s*", "", cleaned)
    cleaned = re.sub(r"(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})日?", "", cleaned)
    cleaned = re.sub(r"(?<!\d)(\d{1,2})[-/.月](\d{1,2})[日号]?", "", cleaned)
    cleaned = re.sub(r"(?<!\d)(\d{1,2})[日号](?!\d)", "", cleaned)
    cleaned = re.sub(r"\d{1,2}[:：]\d{1,2}", "", cleaned)
    cleaned = re.sub(r"[零〇一二两三四五六七八九十]{1,3}点半", "", cleaned)
    cleaned = re.sub(r"[零〇一二两三四五六七八九十]{1,3}点[零〇一二两三四五六七八九十]{1,3}分?", "", cleaned)
    cleaned = re.sub(r"[零〇一二两三四五六七八九十]{1,3}点", "", cleaned)
    cleaned = re.sub(r"\d{1,2}点半", "", cleaned)
    cleaned = re.sub(r"\d{1,2}点\d{1,2}分?", "", cleaned)
    cleaned = re.sub(r"\d{1,2}点", "", cleaned)
    cleaned = re.sub(r"[，,。！？!?；;、]+", " ", cleaned)
    cleaned = _PREFIX_RE.sub("", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned)
    cleaned = re.sub(r"^(帮我|把|给我|请帮我|安排|创建|建|记个|记一下|预约|提醒我|提醒一下|帮我记个会)\s*", "", cleaned)
    return cleaned.strip()
