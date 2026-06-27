from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from nonebot.log import logger

from .config import Config
from .llm_client import LLMClient
from .memory import MemoryStore

_TIME_HOUR = r"\d{1,2}|零|〇|一|二|两|三|四|五|六|七|八|九|十|十一|十二|十三|十四|十五|十六|十七|十八|十九|二十|二十一|二十二|二十三|二十四"
_TIME_MINUTE = r"\d{1,2}|零|〇|一|二|两|三|四|五|六|七|八|九|十|十一|十二|十三|十四|十五|十六|十七|十八|十九|二十|二十一|二十二|二十三|二十四|三十|三十一|四十|五十"
_TIME_PATTERNS = [
    re.compile(rf"(?P<period>上午|中午|下午|晚上|凌晨|早上|傍晚|今晚)?\s*(?P<hour>{_TIME_HOUR})[:：](?P<minute>\d{{1,2}})"),
    re.compile(rf"(?P<period>上午|中午|下午|晚上|凌晨|早上|傍晚|今晚)?\s*(?P<hour>{_TIME_HOUR})点(?P<minute>{_TIME_MINUTE})分?"),
    re.compile(rf"(?P<period>上午|中午|下午|晚上|凌晨|早上|傍晚|今晚)?\s*(?P<hour>{_TIME_HOUR})点半"),
    re.compile(rf"(?P<period>上午|中午|下午|晚上|凌晨|早上|傍晚|今晚)?\s*(?P<hour>{_TIME_HOUR})点"),
]
_CN_NUMBERS = {
    "零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10, "十一": 11, "十二": 12,
    "十三": 13, "十四": 14, "十五": 15, "十六": 16, "十七": 17, "十八": 18,
    "十九": 19, "二十": 20, "二十一": 21, "二十二": 22, "二十三": 23, "二十四": 24,
    "三十": 30, "三十一": 31, "四十": 40, "五十": 50,
}
_DAY_OFFSETS = {"今天": 0, "今晚": 0, "明天": 1, "后天": 2, "大后天": 3}


@dataclass(slots=True)
class ReminderParseResult:
    kind: str
    content: str = ""
    due_at: datetime | None = None
    time_of_day: str | None = None
    next_run_at: datetime | None = None
    confirmation: str = ""
    clarification: str = ""

    @property
    def ok(self) -> bool:
        return not self.clarification and bool(self.content) and self.next_run_at is not None


_TIME_PERIODS = ("上午", "中午", "下午", "晚上", "凌晨", "早上", "傍晚", "今晚")
_DATE_MARKERS = ("今天", "今晚", "明天", "后天", "大后天")
_DAILY_MARKERS = ("每天", "每日", "天天")


def looks_like_local_reminder(text: str) -> bool:
    raw = text.strip()
    if not raw:
        return False
    if any(marker in raw for marker in ("提醒我", "提醒一下我", "叫我")):
        return True
    if "提醒" in raw and any(marker in raw for marker in _DAILY_MARKERS):
        return True
    if "提醒" in raw and any(marker in raw for marker in _DATE_MARKERS):
        return True
    return False


def parse_reminder(text: str, now: datetime | None = None, timezone_name: str = "Asia/Shanghai") -> ReminderParseResult | None:
    raw = text.strip()
    if not looks_like_local_reminder(raw):
        return None

    tz = ZoneInfo(timezone_name)
    now = now or datetime.now(tz)
    if now.tzinfo is None:
        now = now.replace(tzinfo=tz)
    else:
        now = now.astimezone(tz)

    time_info = _parse_time(raw)
    if _is_daily(raw):
        content = _extract_content(raw)
        if not time_info:
            return ReminderParseResult(kind="daily", content=content, clarification=f"想让我每天几点提醒你{content}？比如「每天早上8点提醒我{content}」。")
        hour, minute, _ = time_info
        next_run = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if next_run <= now:
            next_run += timedelta(days=1)
        time_of_day = f"{hour:02d}:{minute:02d}"
        return ReminderParseResult(
            kind="daily",
            content=content,
            time_of_day=time_of_day,
            next_run_at=next_run,
            confirmation=f"好呀，我会每天 {time_of_day} 提醒你：{content}",
        )

    date_value = _parse_date(raw, now)
    if date_value is None:
        return ReminderParseResult(kind="one_shot", content=_extract_content(raw), clarification="可以呀。你想让我什么时候提醒你？比如「明天晚上8点提醒我做作业」。")
    if not time_info:
        content = _extract_content(raw)
        date_text = _extract_date_text(raw)
        period_text = _extract_period_text(raw)
        when_text = f"{date_text}{period_text}" if date_text or period_text else "这个时间"
        return ReminderParseResult(kind="one_shot", content=content, clarification=f"{when_text}几点提醒你{content}？比如「{date_text or '明天'}下午3点提醒我{content}」。")

    hour, minute, _ = time_info
    due_at = datetime.combine(date_value, datetime.min.time()).replace(hour=hour, minute=minute, tzinfo=tz)
    if due_at <= now:
        return ReminderParseResult(kind="one_shot", clarification="这个时间已经过去了。你换一个未来的时间，我再帮你记。")

    content = _extract_content(raw)
    return ReminderParseResult(
        kind="one_shot",
        content=content,
        due_at=due_at,
        next_run_at=due_at,
        confirmation=f"好呀，我会在{_format_when(due_at, now)}提醒你：{content}",
    )


class ReminderService:
    def __init__(self, memory: MemoryStore, config: Config):
        self.memory = memory
        self.config = config
        self.tz = ZoneInfo(config.reminder_timezone)

    def try_parse(self, user_msg: str, now: datetime | None = None) -> ReminderParseResult | None:
        return parse_reminder(user_msg, now=now, timezone_name=self.config.reminder_timezone)

    def handle_management_command(self, user_id: int, user_msg: str) -> str | None:
        raw = user_msg.strip()
        if raw in {"查看提醒", "我的提醒", "提醒列表"}:
            return self.format_active_reminders(user_id)
        match = re.fullmatch(r"(?:取消|删除)提醒\s*#?(\d+)", raw)
        if match:
            reminder_id = int(match.group(1))
            if self.memory.cancel_reminder(user_id, reminder_id):
                return f"已取消提醒 #{reminder_id}。"
            return f"没找到可取消的提醒 #{reminder_id}。你可以发送「查看提醒」看看当前提醒列表。"
        if raw == "取消所有提醒":
            return "为了避免误删，我现在只支持按编号取消。你可以发送「查看提醒」，再用「取消提醒 3」这样的格式取消。"
        return None

    def format_active_reminders(self, user_id: int) -> str:
        rows = self.memory.list_active_reminders(user_id)
        if not rows:
            return "你现在没有待提醒事项。"
        lines = ["你现在有这些提醒："]
        for row in rows:
            kind = "每天" if row["kind"] == "daily" else "一次"
            next_run_at = _format_next_run(row["next_run_at"], self.tz)
            lines.append(f"#{row['id']} [{kind}] {next_run_at}：{row['text']}")
        lines.append("要取消某条提醒，可以发送「取消提醒 编号」。")
        return "\n".join(lines)

    def create_from_parse(self, user_id: int, parsed: ReminderParseResult) -> int:
        if not parsed.ok or parsed.next_run_at is None:
            raise ValueError(parsed.clarification or "提醒信息不完整")
        due_at = parsed.due_at.isoformat(timespec="minutes") if parsed.due_at else None
        return self.memory.create_reminder(
            user_id=user_id,
            text=parsed.content,
            kind=parsed.kind,
            timezone_name=self.config.reminder_timezone,
            due_at=due_at,
            time_of_day=parsed.time_of_day,
            next_run_at=parsed.next_run_at.isoformat(timespec="minutes"),
        )

    async def scan_and_send_due(self):
        if not self.config.reminder_enabled:
            return
        now = datetime.now(self.tz)
        rows = self.memory.get_due_reminders(now.isoformat(timespec="minutes"), self.config.reminder_max_due_per_scan)
        if not rows:
            return
        try:
            from nonebot import get_bot
            bot = get_bot()
        except Exception as e:
            logger.warning(f"Reminder scan skipped: bot unavailable: {e}")
            return

        for row in rows:
            reminder_id = int(row["id"])
            occurrence_key = self._occurrence_key(row, now)
            if not self.memory.claim_reminder_occurrence(reminder_id, occurrence_key):
                continue
            try:
                message = f"你让我提醒你：{row['text']}"
                for chunk in LLMClient.chunk_text(message):
                    await bot.send_private_msg(user_id=int(row["user_id"]), message=chunk)
                self.memory.mark_reminder_sent(reminder_id, occurrence_key)
                if row["kind"] == "daily":
                    self.memory.advance_daily_reminder(reminder_id, self._next_daily_run(row, now).isoformat(timespec="minutes"))
                else:
                    self.memory.complete_reminder(reminder_id)
            except Exception as e:
                logger.warning(f"Failed to send reminder {reminder_id}: {e}")
                self.memory.mark_reminder_failed(reminder_id, occurrence_key, str(e))

    def _occurrence_key(self, row, now: datetime) -> str:
        if row["kind"] == "daily":
            run_at = datetime.fromisoformat(row["next_run_at"])
            return f"daily:{row['id']}:{run_at.date().isoformat()}"
        return f"one_shot:{row['id']}:{row['next_run_at']}"

    def _next_daily_run(self, row, now: datetime) -> datetime:
        hour, minute = (int(part) for part in row["time_of_day"].split(":", 1))
        base = datetime.fromisoformat(row["next_run_at"])
        if base.tzinfo is None:
            base = base.replace(tzinfo=self.tz)
        next_run = base.astimezone(self.tz) + timedelta(days=1)
        next_run = next_run.replace(hour=hour, minute=minute, second=0, microsecond=0)
        while next_run <= now:
            next_run += timedelta(days=1)
        return next_run


def _is_daily(text: str) -> bool:
    return any(marker in text for marker in _DAILY_MARKERS)


def _parse_date(text: str, now: datetime):
    for marker, offset in sorted(_DAY_OFFSETS.items(), key=lambda item: len(item[0]), reverse=True):
        if marker in text:
            return (now + timedelta(days=offset)).date()
    return None


def _extract_date_text(text: str) -> str:
    for marker in sorted(_DAY_OFFSETS, key=len, reverse=True):
        if marker in text:
            return marker
    return ""


def _extract_period_text(text: str) -> str:
    for marker in _TIME_PERIODS:
        if marker in text and marker not in _DAY_OFFSETS:
            return marker
    return ""


def _parse_time(text: str) -> tuple[int, int, int] | None:
    for pattern in _TIME_PATTERNS:
        match = pattern.search(text)
        if not match:
            continue
        hour = _parse_int(match.group("hour"))
        minute = 30 if text[match.start():match.end()].endswith("点半") else _parse_int(match.groupdict().get("minute") or "0")
        if hour is None or minute is None or not (0 <= minute <= 59):
            raise ValueError("invalid time")
        period = match.groupdict().get("period") or ""
        hour = _apply_period(hour, period)
        if not (0 <= hour <= 23):
            raise ValueError("invalid time")
        return hour, minute, match.end()
    return None


def _parse_int(text: str) -> int | None:
    text = text.strip()
    if text.isdigit():
        return int(text)
    return _CN_NUMBERS.get(text)


def _apply_period(hour: int, period: str) -> int:
    if period in {"下午", "晚上", "傍晚", "今晚"} and 1 <= hour < 12:
        return hour + 12
    if period == "中午" and hour < 11:
        return hour + 12
    if period == "凌晨" and hour == 12:
        return 0
    return hour


def _extract_content(text: str) -> str:
    cleaned = text.strip()
    cleaned = re.sub(r"^(请你|请|麻烦你|帮我|以后|之后)?", "", cleaned).strip()
    cleaned = re.sub(r"^(每天|每日|天天|以后每天|之后每天)", "", cleaned).strip()
    cleaned = re.sub(r"^(今天|今晚|明天|后天|大后天)", "", cleaned).strip()
    for pattern in _TIME_PATTERNS:
        cleaned = pattern.sub("", cleaned, count=1).strip()
    cleaned = cleaned.replace("提醒一下我", "").replace("提醒我", "").replace("叫我", "").replace("提醒", "").strip()
    for period in _TIME_PERIODS:
        cleaned = cleaned.replace(period, "").strip()
    cleaned = re.sub(r"^(每天|每日|天天)", "", cleaned).strip()
    cleaned = re.sub(r"^(今天|今晚|明天|后天|大后天)", "", cleaned).strip()
    cleaned = cleaned.strip(" ，。.!！?？：:")
    return cleaned or "这件事"


def _format_when(due_at: datetime, now: datetime) -> str:
    time_text = due_at.strftime("%H:%M")
    days = (due_at.date() - now.date()).days
    if days == 0:
        return f"今天 {time_text}"
    if days == 1:
        return f"明天 {time_text}"
    if days == 2:
        return f"后天 {time_text}"
    return due_at.strftime("%Y-%m-%d %H:%M")


def _format_next_run(next_run_at: str, tz: ZoneInfo) -> str:
    run_at = datetime.fromisoformat(next_run_at)
    if run_at.tzinfo is None:
        run_at = run_at.replace(tzinfo=tz)
    return run_at.astimezone(tz).strftime("%Y-%m-%d %H:%M")
