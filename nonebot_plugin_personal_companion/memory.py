import re
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

from .turn_context import EMOTION_KEYWORDS, detect_emotions


BEIJING_TZ = ZoneInfo("Asia/Shanghai")


class MemoryStore:
    """SQLite-backed memory with full per-user isolation."""

    SHORT_EVENT_MARKERS = [
        "今天", "今晚", "明天", "后天", "周一", "周二", "周三", "周四", "周五", "周六", "周日",
        "星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日",
        "礼拜一", "礼拜二", "礼拜三", "礼拜四", "礼拜五", "礼拜六", "礼拜日",
        "本周", "这周", "周末", "这几天",
    ]
    MEDIUM_EVENT_MARKERS = ["下周", "下星期", "下礼拜"]
    PLANNING_EVENT_MARKERS = ["准备", "打算", "计划", "要去", "出去玩", "考试", "作业", "项目"]
    TIMELINE_EVENT_MARKERS = [
        "考试", "面试", "出去玩", "项目", "作业", "生日", "见面", "旅行", "跑步", "吃",
        "拿到", "解决", "完成", "去了", "去看", "聊天", "宣布", "拥有", "幸福", "报错",
    ]
    GENERIC_RECALL_WORDS = {
        "用户", "今天", "这个", "那个", "事情", "东西", "时候", "问题", "感觉", "自己", "我们", "你们", "他们",
        "最近", "之前", "后来", "然后", "还是", "已经", "现在", "那次", "这次", "一下", "一下子",
    }
    FRESH_MEMORY_WINDOW_DAYS = 2
    SUMMARY_RECALL_WINDOW_DAYS = 7
    RELATIVE_DATE_OFFSETS = {"前天": -2, "昨天": -1, "今天": 0, "今晚": 0, "明天": 1, "后天": 2}

    def __init__(self, db_path: Path | str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    user_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    summary_text TEXT NOT NULL,
                    user_id INTEGER,
                    start_msg_id INTEGER,
                    end_msg_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS key_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    user_id INTEGER,
                    source_msg_id INTEGER,
                    importance INTEGER DEFAULT 1,
                    access_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    emotion_tags TEXT DEFAULT '',
                    entity_tags TEXT DEFAULT '',
                    memory_type TEXT DEFAULT 'fact',
                    status TEXT DEFAULT 'active'
                );
                CREATE TABLE IF NOT EXISTS extraction_checkpoints (
                    user_id INTEGER PRIMARY KEY,
                    last_msg_id INTEGER NOT NULL DEFAULT 0,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS active_users (
                    user_id INTEGER PRIMARY KEY,
                    last_message_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    first_seen_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS proactive_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    content TEXT DEFAULT '',
                    topic_kind TEXT DEFAULT '',
                    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS manifestation_wishes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    raw_content TEXT NOT NULL,
                    status TEXT DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS manifestation_evidence (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    wish_id INTEGER,
                    evidence_type TEXT DEFAULT 'internal',
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS memory_timeline (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    event_date TEXT NOT NULL,
                    event_time TEXT DEFAULT '',
                    content TEXT NOT NULL,
                    source TEXT DEFAULT 'manual',
                    source_msg_id INTEGER,
                    tags TEXT DEFAULT '',
                    importance INTEGER DEFAULT 1,
                    status TEXT DEFAULT 'planned',
                    direction TEXT DEFAULT 'unknown',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS proactive_snoozes (
                    user_id INTEGER PRIMARY KEY,
                    snooze_until TIMESTAMP NOT NULL,
                    reason TEXT DEFAULT '',
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS reminders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    timezone TEXT NOT NULL DEFAULT 'Asia/Shanghai',
                    due_at TEXT,
                    time_of_day TEXT,
                    next_run_at TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'active',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS reminder_deliveries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    reminder_id INTEGER NOT NULL,
                    occurrence_key TEXT NOT NULL,
                    status TEXT NOT NULL,
                    claimed_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    sent_at TEXT,
                    error TEXT,
                    UNIQUE(reminder_id, occurrence_key)
                );
                CREATE INDEX IF NOT EXISTS idx_reminders_due
                    ON reminders(status, next_run_at);
                CREATE INDEX IF NOT EXISTS idx_reminders_user_status
                    ON reminders(user_id, status);
                CREATE INDEX IF NOT EXISTS idx_memory_timeline_user_date
                    ON memory_timeline(user_id, event_date);
                CREATE INDEX IF NOT EXISTS idx_memory_timeline_user_content
                    ON memory_timeline(user_id, content);
            """)
            int_columns = [
                ("user_id", "messages"), ("user_id", "summaries"), ("user_id", "key_memories"),
                ("importance", "key_memories"), ("access_count", "key_memories"),
                ("user_id", "manifestation_wishes"), ("wish_id", "manifestation_evidence"),
            ]
            text_columns = [
                ("content", "proactive_log"), ("topic_kind", "proactive_log"),
                ("emotion_tags", "key_memories"), ("entity_tags", "key_memories"),
                ("memory_type", "key_memories"), ("status", "key_memories"),
                ("status", "manifestation_wishes"), ("raw_content", "manifestation_wishes"),
                ("evidence_type", "manifestation_evidence"),
                ("reason", "proactive_snoozes"),
                ("status", "memory_timeline"), ("direction", "memory_timeline"),
                ("updated_at", "memory_timeline"),
            ]
            for col, table in int_columns:
                try:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} INTEGER")
                except Exception:
                    pass
            for col, table in text_columns:
                try:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT DEFAULT ''")
                except Exception:
                    pass
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_timeline_user_status ON memory_timeline(user_id, status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_memory_timeline_user_direction ON memory_timeline(user_id, direction)")

    @staticmethod
    def _to_beijing_display(timestamp: str | None) -> str:
        if not timestamp:
            return "时间未知"
        text = timestamp.replace("Z", "+00:00")
        try:
            dt = datetime.fromisoformat(text)
        except ValueError:
            try:
                dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
            except ValueError:
                return timestamp
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(BEIJING_TZ).strftime("%Y-%m-%d %H:%M")

    def save_message(self, role: str, content: str, user_id: int | None = None) -> int:
        with self._get_conn() as conn:
            cur = conn.execute(
                "INSERT INTO messages (role, content, user_id) VALUES (?, ?, ?)",
                (role, content, user_id),
            )
            return cur.lastrowid or 0

    def get_recent_messages(self, limit: int = 30, user_id: int | None = None) -> list[dict]:
        with self._get_conn() as conn:
            if user_id is not None:
                rows = conn.execute(
                    "SELECT role, content, created_at FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                    (user_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT role, content, created_at FROM messages ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [
            {
                "role": r["role"],
                "content": r["content"],
                "time": r["created_at"],
                "time_display": self._to_beijing_display(r["created_at"]),
            }
            for r in reversed(rows)
        ]

    def get_messages_since(self, when: str, user_id: int, limit: int = 500) -> list[dict]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT role, content, created_at FROM messages WHERE user_id = ? AND created_at >= ? ORDER BY id ASC LIMIT ?",
                (user_id, when, limit),
            ).fetchall()
        return [
            {
                "role": r["role"],
                "content": r["content"],
                "time": r["created_at"],
                "time_display": self._to_beijing_display(r["created_at"]),
            }
            for r in rows
        ]

    def get_user_message_stats(self, user_id: int) -> dict:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS total, AVG(LENGTH(content)) AS avg_len FROM messages WHERE user_id = ? AND role = 'user'",
                (user_id,),
            ).fetchone()
            first = conn.execute(
                "SELECT first_seen_at FROM active_users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return {
            "total": row["total"] or 0,
            "avg_len": round(row["avg_len"] or 0, 1),
            "first_seen": first["first_seen_at"] if first else None,
        }

    def get_latest_message_id(self, user_id: int) -> int:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(id), 0) AS max_id FROM messages WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return row["max_id"]

    def get_oldest_message_id_after(self, user_id: int, after_id: int) -> int:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(MIN(id), 0) AS min_id FROM messages WHERE user_id = ? AND id > ?",
                (user_id, after_id),
            ).fetchone()
        return row["min_id"]

    def get_extraction_checkpoint(self, user_id: int) -> int:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT last_msg_id FROM extraction_checkpoints WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return row["last_msg_id"] if row else 0

    def save_extraction_checkpoint(self, user_id: int, last_msg_id: int):
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO extraction_checkpoints (user_id, last_msg_id)
                   VALUES (?, ?)
                   ON CONFLICT(user_id) DO UPDATE SET last_msg_id = excluded.last_msg_id, updated_at = CURRENT_TIMESTAMP""",
                (user_id, last_msg_id),
            )

    # ── summaries ──────────────────────────────────────────────

    def message_count_since_last_summary(self, user_id: int) -> int:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(end_msg_id), 0) AS last_id FROM summaries WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            last_id = row["last_id"]
            count_row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM messages WHERE user_id = ? AND id > ?",
                (user_id, last_id),
            ).fetchone()
            return count_row["cnt"]

    def get_summary_range(self, start_msg_id: int, end_msg_id: int, user_id: int) -> tuple[str | None, str | None]:
        with self._get_conn() as conn:
            row = conn.execute(
                """SELECT MIN(created_at) AS start_time, MAX(created_at) AS end_time
                   FROM messages WHERE user_id = ? AND id BETWEEN ? AND ?""",
                (user_id, start_msg_id, end_msg_id),
            ).fetchone()
        if not row or not row["start_time"]:
            return None, None
        return self._to_beijing_display(row["start_time"]), self._to_beijing_display(row["end_time"])

    @staticmethod
    def _strip_summary_header(summary_text: str) -> str:
        return re.sub(r"^\[历史对话摘要｜[^\]]+\]\s*", "", summary_text).strip()

    @staticmethod
    def _summary_timestamp(summary_text: str) -> str | None:
        match = re.match(r"^\[历史对话摘要｜([^\]]+)\]", summary_text)
        return match.group(1) if match else None

    @classmethod
    def _summary_is_fresh(cls, summary_text: str) -> bool:
        timestamp = cls._summary_timestamp(summary_text)
        if not timestamp:
            return True
        try:
            start_text = timestamp.split(" 至 ", 1)[0].strip().replace("北京时间", "")
            start_dt = datetime.strptime(start_text, "%Y-%m-%d %H:%M")
        except ValueError:
            return True
        cutoff = datetime.now(BEIJING_TZ) - timedelta(days=cls.SUMMARY_RECALL_WINDOW_DAYS)
        return start_dt.replace(tzinfo=BEIJING_TZ) >= cutoff

    @classmethod
    def _memory_is_fresh(cls, content: str, created_at: str | None) -> bool:
        if not created_at:
            return True
        try:
            dt = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
        except ValueError:
            return True
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        age_days = (datetime.now(BEIJING_TZ) - dt.astimezone(BEIJING_TZ)).total_seconds() / 86400
        if age_days <= cls.FRESH_MEMORY_WINDOW_DAYS:
            return True
        if any(word in content for word in cls.GENERIC_RECALL_WORDS):
            return False
        return age_days <= cls.SUMMARY_RECALL_WINDOW_DAYS

    @classmethod
    def _summary_is_relevant(cls, summary_text: str, keyword: str) -> bool:
        if not summary_text or not keyword:
            return False
        if keyword in cls.GENERIC_RECALL_WORDS:
            return False
        return keyword in summary_text

    def save_summary(self, summary_text: str, start_msg_id: int, end_msg_id: int, user_id: int):
        start_time, end_time = self.get_summary_range(start_msg_id, end_msg_id, user_id)
        if start_time and end_time:
            summary_text = f"[历史对话摘要｜{start_time} 至 {end_time}]\n{summary_text}"
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO summaries (summary_text, start_msg_id, end_msg_id, user_id) VALUES (?, ?, ?, ?)",
                (summary_text, start_msg_id, end_msg_id, user_id),
            )

    # ── memory timeline ─────────────────────────────────────────

    @staticmethod
    def _validate_date(date_str: str) -> str:
        datetime.strptime(date_str, "%Y-%m-%d")
        return date_str

    @staticmethod
    def _timeline_row(row: sqlite3.Row) -> dict:
        return {
            "id": row["id"],
            "user_id": row["user_id"],
            "event_date": row["event_date"],
            "event_time": row["event_time"] or "",
            "content": row["content"],
            "source": row["source"] or "manual",
            "source_msg_id": row["source_msg_id"],
            "tags": [t.strip() for t in (row["tags"] or "").split(",") if t.strip()],
            "importance": row["importance"] or 1,
            "status": row["status"] or "planned",
            "direction": row["direction"] or "unknown",
            "created_at": row["created_at"],
            "updated_at": row["updated_at"] if "updated_at" in row.keys() else "",
        }

    def add_timeline_entry(self, user_id: int, event_date: str, content: str,
                           event_time: str = "", source: str = "manual",
                           source_msg_id: int | None = None,
                           tags: list[str] | None = None,
                           importance: int = 1,
                           status: str = "planned",
                           direction: str = "unknown") -> int:
        event_date = self._validate_date(event_date)
        tag_text = ",".join(dict.fromkeys(t.strip() for t in (tags or []) if t.strip()))
        with self._get_conn() as conn:
            existing = conn.execute(
                """SELECT id FROM memory_timeline
                   WHERE user_id = ? AND event_date = ? AND content = ? LIMIT 1""",
                (user_id, event_date, content),
            ).fetchone()
            if existing:
                conn.execute(
                    """UPDATE memory_timeline
                       SET event_time = COALESCE(NULLIF(?, ''), event_time),
                           status = ?, direction = ?, updated_at = CURRENT_TIMESTAMP
                       WHERE user_id = ? AND id = ?""",
                    (event_time, status, direction, user_id, existing["id"]),
                )
                return existing["id"]
            cur = conn.execute(
                """INSERT INTO memory_timeline
                   (user_id, event_date, event_time, content, source, source_msg_id, tags, importance, status, direction)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_id, event_date, event_time, content, source, source_msg_id, tag_text, importance, status, direction),
            )
            return cur.lastrowid or 0

    def get_timeline_entries_between(self, user_id: int, start_date: str, end_date: str,
                                     limit: int = 20) -> list[dict]:
        return self.get_timeline_entries(user_id, start_date=start_date, end_date=end_date, limit=limit)

    def get_timeline_entries(self, user_id: int, status: str | list[str] | None = None,
                             direction: str | list[str] | None = None,
                             start_date: str | None = None,
                             end_date: str | None = None,
                             limit: int = 20,
                             order: str = "asc") -> list[dict]:
        clauses = ["user_id = ?"]
        params: list = [user_id]
        if status:
            statuses = [status] if isinstance(status, str) else status
            clauses.append("status IN (" + ",".join("?" for _ in statuses) + ")")
            params.extend(statuses)
        if direction:
            directions = [direction] if isinstance(direction, str) else direction
            clauses.append("direction IN (" + ",".join("?" for _ in directions) + ")")
            params.extend(directions)
        if start_date:
            clauses.append("event_date >= ?")
            params.append(self._validate_date(start_date))
        if end_date:
            clauses.append("event_date <= ?")
            params.append(self._validate_date(end_date))
        params.append(limit)
        sort_direction = "DESC" if order == "desc" else "ASC"
        with self._get_conn() as conn:
            rows = conn.execute(
                f"""SELECT * FROM memory_timeline
                    WHERE {' AND '.join(clauses)}
                    ORDER BY event_date {sort_direction}, event_time {sort_direction}, id {sort_direction} LIMIT ?""",
                tuple(params),
            ).fetchall()
        return [self._timeline_row(r) for r in rows]

    def get_recent_timeline_entries(self, user_id: int, limit: int = 5) -> list[dict]:
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT * FROM memory_timeline
                   WHERE user_id = ?
                   ORDER BY event_date DESC, id DESC LIMIT ?""",
                (user_id, limit),
            ).fetchall()
        return [self._timeline_row(r) for r in rows]

    def retrieve_timeline_entries(self, keywords: list[str], user_id: int, limit: int = 5) -> list[dict]:
        if not keywords:
            return []
        results: dict[int, tuple[sqlite3.Row, float]] = {}
        with self._get_conn() as conn:
            for kw in keywords:
                rows = conn.execute(
                    """SELECT * FROM memory_timeline
                       WHERE user_id = ? AND status != 'suppressed' AND (content LIKE ? OR tags LIKE ?)""",
                    (user_id, f"%{kw}%", f"%{kw}%"),
                ).fetchall()
                for row in rows:
                    score = 1.0 + (row["importance"] or 1) * 0.5 + min((row["access_count"] or 0) * 0.1, 1.5)
                    if row["status"] == "planned":
                        score += 0.4
                    if kw in (row["content"] or ""):
                        score += 0.5
                    if row["id"] not in results or score > results[row["id"]][1]:
                        results[row["id"]] = (row, score)
            ranked = [row for row, _ in sorted(results.values(), key=lambda x: x[1], reverse=True)[:limit]]
            if ranked:
                ids = [row["id"] for row in ranked]
                placeholders = ",".join("?" for _ in ids)
                conn.execute(
                    f"""UPDATE memory_timeline
                        SET access_count = access_count + 1, last_accessed_at = CURRENT_TIMESTAMP
                        WHERE user_id = ? AND id IN ({placeholders})""",
                    tuple([user_id] + ids),
                )
        return [self._timeline_row(r) for r in ranked]

    def update_timeline_entry_status(self, user_id: int, entry_id: int, status: str) -> bool:
        direction = "past" if status in {"done", "cancelled", "expired", "suppressed"} else "future"
        with self._get_conn() as conn:
            cur = conn.execute(
                """UPDATE memory_timeline
                   SET status = ?, direction = ?, updated_at = CURRENT_TIMESTAMP
                   WHERE user_id = ? AND id = ?""",
                (status, direction, user_id, entry_id),
            )
            return cur.rowcount > 0

    def delete_timeline_entry(self, user_id: int, entry_id: int) -> bool:
        with self._get_conn() as conn:
            cur = conn.execute(
                "DELETE FROM memory_timeline WHERE user_id = ? AND id = ?",
                (user_id, entry_id),
            )
            return cur.rowcount > 0

    def build_timeline_overview(self, user_id: int, mode: str = "all", now: datetime | None = None) -> str:
        now = now or datetime.now(BEIJING_TZ)
        today = now.strftime("%Y-%m-%d")
        parts: list[str] = []
        if mode in {"all", "future"}:
            future = self.get_timeline_entries(user_id, status="planned", start_date=today, limit=10)
            parts.append("接下来：")
            parts.extend(self._format_timeline_lines(future) if future else ["（暂无未来时间线）"])
        if mode in {"all", "history"}:
            history = self.get_timeline_entries(user_id, status=["done", "cancelled", "expired"], limit=10, order="desc")
            parts.append("最近历史：")
            parts.extend(self._format_timeline_lines(history) if history else ["（暂无历史时间线）"])
        return "\n".join(parts)

    @staticmethod
    def _format_timeline_lines(entries: list[dict]) -> list[str]:
        status_label = {
            "planned": "计划中",
            "done": "已完成",
            "cancelled": "已取消",
            "expired": "已过期",
            "suppressed": "已隐藏",
        }
        lines = []
        for item in entries:
            event_time = f" {item['event_time']}" if item.get("event_time") else ""
            label = status_label.get(item.get("status", "planned"), item.get("status", "planned"))
            lines.append(f"- #{item['id']} [{label}] {item['event_date']}{event_time}：{item['content']}")
        return lines

    def maybe_add_timeline_entry_from_message(self, user_id: int, content: str,
                                              source_msg_id: int | None = None,
                                              now: datetime | None = None) -> int | None:
        now = now or datetime.now(BEIJING_TZ)
        event_date = self._extract_event_date(content, now)
        if not event_date or not self._looks_like_timeline_event(content):
            return None
        event_time = self._extract_event_time(content)
        status = self._infer_timeline_status(content)
        direction = self._infer_timeline_direction(event_date, status, now)
        tags = [m for m in self.TIMELINE_EVENT_MARKERS if m in content]
        entry = f"用户在 {event_date} 提到：{content}"
        return self.add_timeline_entry(
            user_id=user_id,
            event_date=event_date,
            event_time=event_time,
            content=entry,
            source="message",
            source_msg_id=source_msg_id,
            tags=tags,
            importance=2,
            status=status,
            direction=direction,
        )

    def _extract_event_date(self, text: str, now: datetime) -> str | None:
        relative_offsets = {**self.RELATIVE_DATE_OFFSETS, "大后天": 3, "明早": 1, "明晚": 1}
        for marker in sorted(relative_offsets, key=len, reverse=True):
            if marker in text:
                return (now.date() + timedelta(days=relative_offsets[marker])).strftime("%Y-%m-%d")

        match = re.search(r"(下下周|下周|这周|本周)?(?:周|星期|礼拜)([一二三四五六日天1234567])", text)
        if match:
            prefix = match.group(1) or ""
            weekday = self._weekday_index(match.group(2))
            if weekday is not None:
                base = now.date()
                days = weekday - base.weekday()
                if prefix == "下下周":
                    days += 14
                elif prefix == "下周":
                    days += 7
                elif prefix in {"这周", "本周"}:
                    pass
                elif days < 0:
                    days += 7
                return (base + timedelta(days=days)).strftime("%Y-%m-%d")

        match = re.search(r"(20\d{2})[-/.年](\d{1,2})[-/.月](\d{1,2})日?", text)
        if match:
            year, month, day = (int(x) for x in match.groups())
            try:
                return datetime(year, month, day).strftime("%Y-%m-%d")
            except ValueError:
                return None

        match = re.search(r"(\d{1,2})月(\d{1,2})[日号]?", text)
        if match:
            month, day = (int(x) for x in match.groups())
            try:
                return datetime(now.year, month, day).strftime("%Y-%m-%d")
            except ValueError:
                return None

        match = re.search(r"下个月(\d{1,2})[日号]?", text)
        if match:
            month = now.month + 1
            year = now.year + (1 if month == 13 else 0)
            month = 1 if month == 13 else month
            try:
                return datetime(year, month, int(match.group(1))).strftime("%Y-%m-%d")
            except ValueError:
                return None

        match = re.search(r"(?<!月)(\d{1,2})[日号]", text)
        if match:
            month = now.month
            day = int(match.group(1))
            year = now.year
            try:
                candidate = datetime(year, month, day).date()
            except ValueError:
                return None
            if candidate < now.date() and "上" not in text and "昨天" not in text:
                month += 1
                if month == 13:
                    year += 1
                    month = 1
                try:
                    candidate = datetime(year, month, day).date()
                except ValueError:
                    return None
            return candidate.strftime("%Y-%m-%d")

        if "月底" in text:
            month = now.month
            year = now.year
            if "下个月" in text:
                month += 1
                if month == 13:
                    year += 1
                    month = 1
            next_month = datetime(year + (1 if month == 12 else 0), 1 if month == 12 else month + 1, 1).date()
            return (next_month - timedelta(days=1)).strftime("%Y-%m-%d")

        return None

    @staticmethod
    def _weekday_index(text: str) -> int | None:
        mapping = {"一": 0, "二": 1, "三": 2, "四": 3, "五": 4, "六": 5, "日": 6, "天": 6,
                   "1": 0, "2": 1, "3": 2, "4": 3, "5": 4, "6": 5, "7": 6}
        return mapping.get(text)

    @classmethod
    def _extract_event_time(cls, text: str) -> str:
        match = re.search(r"(\d{1,2}):(\d{2})", text)
        if match:
            return f"{int(match.group(1)):02d}:{match.group(2)}"
        match = re.search(r"(凌晨|早上|早晨|明早|上午|中午|下午|晚上|今晚|明晚)?([零〇一二两三四五六七八九十\d]{1,3})点(半|多|\d{1,2}分?)?", text)
        if not match:
            return ""
        period = match.group(1) or ""
        hour = cls._parse_chinese_number(match.group(2))
        if hour is None:
            return ""
        minute_part = match.group(3) or ""
        minute = 30 if minute_part == "半" else 0
        if minute_part.endswith("分"):
            parsed_minute = cls._parse_chinese_number(minute_part[:-1])
            minute = parsed_minute if parsed_minute is not None else minute
        if period in {"下午", "晚上", "今晚", "明晚"} and hour < 12:
            hour += 12
        elif period == "中午" and hour < 11:
            hour += 12
        elif period == "凌晨" and hour == 12:
            hour = 0
        return f"{hour:02d}:{minute:02d}"

    @staticmethod
    def _parse_chinese_number(text: str) -> int | None:
        if text.isdigit():
            return int(text)
        digits = {"零": 0, "〇": 0, "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}
        if text == "十":
            return 10
        if text.startswith("十"):
            return 10 + digits.get(text[1:], 0)
        if "十" in text:
            left, right = text.split("十", 1)
            tens = digits.get(left, 1)
            ones = digits.get(right, 0) if right else 0
            return tens * 10 + ones
        if len(text) == 1:
            return digits.get(text)
        return None

    @staticmethod
    def _infer_timeline_status(text: str) -> str:
        if any(marker in text for marker in ["取消了", "不去了", "改期了", "延期了", "推迟了"]):
            return "cancelled"
        if any(marker in text for marker in ["完成了", "结束了", "做完了", "搞定了", "考完了", "见完了", "解决了"]):
            return "done"
        if "已经" in text and "了" in text:
            return "done"
        return "planned"

    @staticmethod
    def _infer_timeline_direction(event_date: str, status: str, now: datetime) -> str:
        if status in {"done", "cancelled", "expired", "suppressed"}:
            return "past"
        date_value = datetime.strptime(event_date, "%Y-%m-%d").date()
        if date_value > now.date():
            return "future"
        if date_value < now.date():
            return "past"
        return "same_day"

    def _looks_like_timeline_event(self, text: str) -> bool:
        return any(marker in text for marker in self.TIMELINE_EVENT_MARKERS + self.PLANNING_EVENT_MARKERS)

    # ── key memories ───────────────────────────────────────────

    @staticmethod
    def classify_memory(content: str) -> tuple[str, str]:
        preference_markers = ["喜欢", "讨厌", "偏好", "不喜欢", "爱吃", "想要", "习惯"]
        boundary_markers = ["不要", "别", "不想", "雷区", "介意", "讨厌被", "不喜欢被"]
        completed_markers = ["已经", "完成", "结束", "考完", "吃完", "过了", "拿到", "解决", "好了"]
        past_event_markers = ["今天", "昨天", "昨晚", "前天", "上周", "前几天", "刚才", "之前", "那次", "当时"]
        ongoing_markers = ["正在", "最近", "准备", "打算", "还在", "这几天", "明天", "下周", "项目", "考试", "作业"]
        emotional_markers = ["总是", "经常", "容易", "会因为", "一到", "压力", "焦虑", "难过", "烦躁", "失眠"]

        if any(w in content for w in boundary_markers):
            return "boundary", "active"
        if any(w in content for w in preference_markers):
            return "preference", "active"
        if any(w in content for w in completed_markers):
            return "event", "completed"
        if any(w in content for w in past_event_markers) and not any(w in content for w in ongoing_markers):
            return "event", "completed"
        if any(w in content for w in ongoing_markers):
            return "event", "ongoing"
        if any(w in content for w in emotional_markers):
            return "emotional_pattern", "active"
        return "fact", "active"

    def add_key_memory(self, content: str, source_msg_id: int | None = None,
                       user_id: int | None = None, importance: int = 1,
                       emotion_tags: str = "", entity_tags: str = "",
                       memory_type: str | None = None, status: str | None = None):
        if memory_type is None or status is None:
            inferred_type, inferred_status = self.classify_memory(content)
            memory_type = memory_type or inferred_type
            status = status or inferred_status
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO key_memories
                   (content, source_msg_id, user_id, importance, emotion_tags, entity_tags, memory_type, status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (content, source_msg_id, user_id, importance, emotion_tags, entity_tags, memory_type, status),
            )

    @staticmethod
    def _is_recallable_memory(memory_type: str | None, status: str | None) -> bool:
        if memory_type == "event" and status in {"completed", "expired", "suppressed"}:
            return False
        if memory_type == "manifestation" and status in {"released", "fulfilled", "expired", "suppressed"}:
            return False
        if status == "suppressed":
            return False
        return True

    def retrieve_memories(self, keywords: list[str], user_id: int, limit: int = 5) -> list[str]:
        self.refresh_event_statuses(user_id)
        results: list[tuple[str, float]] = []
        meaningful_keywords = [kw for kw in keywords if kw.strip()]
        with self._get_conn() as conn:
            for kw in meaningful_keywords:
                rows = conn.execute(
                    """SELECT content, importance, access_count, memory_type, status, created_at FROM key_memories
                       WHERE user_id = ? AND content LIKE ?""",
                    (user_id, f"%{kw}%"),
                ).fetchall()
                for row in rows:
                    if not self._is_recallable_memory(row["memory_type"], row["status"]):
                        continue
                    if not self._memory_is_fresh(row["content"], row["created_at"]):
                        continue
                    imp = row["importance"] or 1
                    acc = row["access_count"] or 0
                    boost = imp * 0.5 + min(acc * 0.1, 1.5)
                    results.append((row["content"], 1.0 + boost))
                sum_rows = conn.execute(
                    "SELECT summary_text, created_at FROM summaries WHERE user_id = ? AND summary_text LIKE ?",
                    (user_id, f"%{kw}%"),
                ).fetchall()
                for row in sum_rows:
                    if not self._summary_is_fresh(row["summary_text"]):
                        continue
                    if not self._summary_is_relevant(self._strip_summary_header(row["summary_text"]), kw):
                        continue
                    results.append((self._strip_summary_header(row["summary_text"]), 2.0))

        seen = set()
        unique: list[str] = []
        for content, score in sorted(results, key=lambda x: x[1], reverse=True):
            if content not in seen:
                seen.add(content)
                unique.append(content)

        # Increment access_count for retrieved memories
        top = unique[:limit]
        if top:
            with self._get_conn() as conn:
                for content in top:
                    conn.execute(
                        "UPDATE key_memories SET access_count = access_count + 1, last_accessed_at = CURRENT_TIMESTAMP WHERE user_id = ? AND content = ?",
                        (user_id, content),
                    )
        return top

    def get_all_key_memories(self, user_id: int | None = None, include_inactive: bool = False) -> list[str]:
        if user_id is not None:
            self.refresh_event_statuses(user_id)
        with self._get_conn() as conn:
            filters = [] if include_inactive else ["NOT (memory_type = 'event' AND status IN ('completed', 'expired', 'suppressed'))", "NOT (memory_type = 'manifestation' AND status IN ('released', 'fulfilled', 'expired', 'suppressed'))", "status != 'suppressed'"]
            if user_id is not None:
                where = "WHERE user_id = ?"
                if filters:
                    where += " AND " + " AND ".join(filters)
                rows = conn.execute(
                    f"SELECT content FROM key_memories {where} ORDER BY created_at DESC",
                    (user_id,),
                ).fetchall()
            else:
                where = ""
                if filters:
                    where = "WHERE " + " AND ".join(filters)
                rows = conn.execute(f"SELECT content FROM key_memories {where} ORDER BY created_at DESC").fetchall()
        return [r["content"] for r in rows]

    def get_key_memories_with_meta(self, user_id: int, limit: int | None = None) -> list[dict]:
        self.refresh_event_statuses(user_id)
        sql = """SELECT id, content, memory_type, status, emotion_tags, entity_tags, importance, created_at
                 FROM key_memories WHERE user_id = ? ORDER BY created_at DESC"""
        params: tuple = (user_id,)
        if limit is not None:
            sql += " LIMIT ?"
            params = (user_id, limit)
        with self._get_conn() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def find_key_memories(self, user_id: int, query: str = "", limit: int = 10, include_inactive: bool = True) -> list[dict]:
        self.refresh_event_statuses(user_id)
        with self._get_conn() as conn:
            where = "user_id = ?"
            params: list = [user_id]
            if query:
                where += " AND content LIKE ?"
                params.append(f"%{query}%")
            if not include_inactive:
                where += " AND NOT (memory_type = 'event' AND status IN ('completed', 'expired'))"
                where += " AND NOT (memory_type = 'manifestation' AND status IN ('released', 'fulfilled', 'expired'))"
            params.append(limit)
            rows = conn.execute(
                f"""SELECT id, content, memory_type, status, importance, created_at
                    FROM key_memories WHERE {where}
                    ORDER BY importance DESC, created_at DESC LIMIT ?""",
                tuple(params),
            ).fetchall()
        return [dict(r) for r in rows]

    def delete_key_memories(self, user_id: int, query: str) -> list[str]:
        matches = self.find_key_memories(user_id, query, limit=20, include_inactive=True)
        if not matches:
            return []
        ids = [m["id"] for m in matches]
        placeholders = ",".join("?" for _ in ids)
        with self._get_conn() as conn:
            conn.execute(
                f"DELETE FROM key_memories WHERE user_id = ? AND id IN ({placeholders})",
                tuple([user_id] + ids),
            )
        return [m["content"] for m in matches]

    def update_key_memory_status(self, user_id: int, query: str, status: str, memory_type: str | None = None) -> list[str]:
        allowed = {"active", "ongoing", "completed", "expired", "suppressed"}
        if status not in allowed:
            raise ValueError(f"Invalid memory status: {status}")
        matches = self.find_key_memories(user_id, query, limit=20, include_inactive=True)
        if not matches:
            return []
        ids = [m["id"] for m in matches]
        placeholders = ",".join("?" for _ in ids)
        assignments = ["status = ?"]
        params: list = [status]
        if memory_type is not None:
            assignments.append("memory_type = ?")
            params.append(memory_type)
        with self._get_conn() as conn:
            conn.execute(
                f"UPDATE key_memories SET {', '.join(assignments)} WHERE user_id = ? AND id IN ({placeholders})",
                tuple(params + [user_id] + ids),
            )
        return [m["content"] for m in matches]

    def build_memory_overview(self, user_id: int, limit: int = 12) -> str:
        items = self.find_key_memories(user_id, limit=limit, include_inactive=True)
        if not items:
            return "我现在还没有记下什么长期信息。你可以说「记住：……」让我记住。"
        labels = {
            ("fact", "active"): "长期事实",
            ("preference", "active"): "偏好",
            ("boundary", "active"): "边界",
            ("emotional_pattern", "active"): "情绪模式",
            ("event", "ongoing"): "正在进行",
            ("event", "completed"): "已结束，不会主动提",
            ("event", "expired"): "过期计划，不会主动提",
            ("event", "suppressed"): "已隐藏",
            ("manifestation", "active"): "显化记忆",
        }
        groups: dict[str, list[dict]] = {}
        for item in items:
            label = labels.get((item["memory_type"], item["status"]), f"{item['memory_type']}/{item['status']}")
            groups.setdefault(label, []).append(item)

        preferred_order = [
            "长期事实", "偏好", "边界", "情绪模式", "正在进行", "显化记忆",
            "已结束，不会主动提", "过期计划，不会主动提", "已隐藏",
        ]
        lines = ["我现在记得这些："]
        for label in preferred_order + [label for label in groups if label not in preferred_order]:
            if label not in groups:
                continue
            lines.append(f"\n{label}")
            for item in groups[label]:
                lines.append(f"- #{item['id']} {item['content']}")
        lines.append("\n你可以说「忘掉：关键词」「这件事结束了：关键词」「以后别再提：关键词」「暂停主动关心」来整理。")
        return "\n".join(lines)

    @classmethod
    def _event_ttl_days(cls, content: str) -> int | None:
        if not any(marker in content for marker in cls.PLANNING_EVENT_MARKERS):
            return None
        if any(marker in content for marker in cls.SHORT_EVENT_MARKERS):
            return 3
        if any(marker in content for marker in cls.MEDIUM_EVENT_MARKERS):
            return 14
        return 30

    @classmethod
    def is_expired_ongoing_event(cls, content: str, created_at: str) -> bool:
        ttl_days = cls._event_ttl_days(content)
        if ttl_days is None:
            return False
        with sqlite3.connect(":memory:") as conn:
            row = conn.execute(
                "SELECT datetime(?) < datetime('now', '-' || ? || ' days') AS expired",
                (created_at, ttl_days),
            ).fetchone()
        return bool(row[0]) if row else False

    def refresh_event_statuses(self, user_id: int):
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT id, content, created_at FROM key_memories
                   WHERE user_id = ? AND memory_type = 'event' AND status = 'ongoing'""",
                (user_id,),
            ).fetchall()
            expired_ids = [r["id"] for r in rows if self.is_expired_ongoing_event(r["content"], r["created_at"])]
            for memory_id in expired_ids:
                conn.execute(
                    "UPDATE key_memories SET status = 'expired' WHERE id = ?",
                    (memory_id,),
                )

    def _extract_wish_title(self, content: str) -> str:
        text = content.replace("\n", " ").strip()
        for marker in ["愿望种子已种下", "你想显化的是", "愿望名称", "原始愿望", "显化愿望种子："]:
            if marker in text:
                after = text.split(marker, 1)[1].strip(" ：:，,。")
                if after:
                    text = after
                    break
        title = text[:40].strip(" ：:，,。")
        return title or "未命名愿望"

    def create_manifestation_wish(self, user_id: int, title: str, raw_content: str, status: str = "active") -> int:
        with self._get_conn() as conn:
            cur = conn.execute(
                """INSERT INTO manifestation_wishes (user_id, title, raw_content, status)
                   VALUES (?, ?, ?, ?)""",
                (user_id, title, raw_content, status),
            )
            return cur.lastrowid or 0

    def get_manifestation_wishes(self, user_id: int, statuses: list[str] | None = None, limit: int = 10) -> list[dict]:
        params: list = [user_id]
        where = "user_id = ?"
        if statuses:
            placeholders = ",".join("?" for _ in statuses)
            where += f" AND status IN ({placeholders})"
            params.extend(statuses)
        params.append(limit)
        with self._get_conn() as conn:
            rows = conn.execute(
                f"""SELECT id, title, raw_content, status, created_at, updated_at
                    FROM manifestation_wishes WHERE {where}
                    ORDER BY updated_at DESC, id DESC LIMIT ?""",
                tuple(params),
            ).fetchall()
        return [dict(r) for r in rows]

    def update_manifestation_wish_status(self, user_id: int, wish_id: int, status: str):
        allowed = {"active", "paused", "released", "fulfilled", "expired"}
        if status not in allowed:
            raise ValueError(f"Invalid manifestation wish status: {status}")
        with self._get_conn() as conn:
            conn.execute(
                """UPDATE manifestation_wishes
                   SET status = ?, updated_at = CURRENT_TIMESTAMP
                   WHERE user_id = ? AND id = ?""",
                (status, user_id, wish_id),
            )

    def add_manifestation_evidence(self, user_id: int, content: str, wish_id: int | None = None, evidence_type: str = "internal") -> int:
        with self._get_conn() as conn:
            cur = conn.execute(
                """INSERT INTO manifestation_evidence (user_id, wish_id, evidence_type, content)
                   VALUES (?, ?, ?, ?)""",
                (user_id, wish_id, evidence_type, content),
            )
            return cur.lastrowid or 0

    def get_manifestation_evidence(self, user_id: int, wish_id: int | None = None, limit: int = 20) -> list[dict]:
        params: list = [user_id]
        where = "user_id = ?"
        if wish_id is not None:
            where += " AND wish_id = ?"
            params.append(wish_id)
        params.append(limit)
        with self._get_conn() as conn:
            rows = conn.execute(
                f"""SELECT id, wish_id, evidence_type, content, created_at
                    FROM manifestation_evidence WHERE {where}
                    ORDER BY created_at DESC, id DESC LIMIT ?""",
                tuple(params),
            ).fetchall()
        return [dict(r) for r in rows]

    def build_manifestation_dashboard(self, user_id: int) -> str:
        wishes = self.get_manifestation_wishes(user_id, limit=8)
        evidence = self.get_manifestation_evidence(user_id, limit=12)
        if not wishes and not evidence:
            return "现在还没有显化愿望记录。你可以说「我想显化……」先种下一颗愿望种子。"

        lines = ["你的显化仪表盘："]
        active = [w for w in wishes if w["status"] == "active"]
        non_active = [w for w in wishes if w["status"] != "active"]
        if active:
            lines.append("\n正在显化：")
            for wish in active[:5]:
                related = [e for e in evidence if e["wish_id"] == wish["id"]]
                latest = related[0]["content"] if related else "还没有单独记录证据"
                lines.append(f"- #{wish['id']} {wish['title']}｜最近证据：{latest}")
        if non_active:
            lines.append("\n已暂停/放下/完成：")
            for wish in non_active[:5]:
                lines.append(f"- #{wish['id']} {wish['title']}｜状态：{wish['status']}")
        if evidence:
            lines.append("\n最近显化证据：")
            for item in evidence[:5]:
                lines.append(f"- {item['content']}")
        lines.append("\n你可以说「记录显化证据：……」或「愿望#1完成了 / 放下了 / 暂停」。")
        return "\n".join(lines)

    def save_manifestation_entry(self, user_id: int, entry_type: str, content: str, source_msg_id: int | None = None):
        label = {
            "manifest_seed": "显化愿望种子",
            "manifest_diary": "显化日记",
            "belief_rewrite": "显化信念改写",
            "obsession_downshift": "显化执念降频",
            "future_self": "显化未来自我",
        }.get(entry_type, "显化记录")
        self.add_key_memory(
            f"{label}：{content}",
            source_msg_id=source_msg_id,
            user_id=user_id,
            importance=5,
            memory_type="manifestation",
            status="active",
        )
        if entry_type == "manifest_seed":
            title = self._extract_wish_title(content)
            self.create_manifestation_wish(user_id, title, content)
        elif entry_type == "manifest_diary":
            self.add_manifestation_evidence(user_id, content, evidence_type="reflection")

    def get_manifestation_memories(self, user_id: int, limit: int = 8) -> list[str]:
        with self._get_conn() as conn:
            rows = conn.execute(
                """SELECT content FROM key_memories
                   WHERE user_id = ? AND memory_type = 'manifestation'
                   AND status NOT IN ('released', 'fulfilled', 'expired', 'suppressed')
                   ORDER BY created_at DESC LIMIT ?""",
                (user_id, limit),
            ).fetchall()
        return [r["content"] for r in rows]

    def count_key_memories(self, user_id: int | None = None) -> int:
        with self._get_conn() as conn:
            if user_id is not None:
                row = conn.execute("SELECT COUNT(*) AS cnt FROM key_memories WHERE user_id = ?", (user_id,)).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) AS cnt FROM key_memories").fetchone()
            return row["cnt"]

    def has_similar_memory(self, text: str, user_id: int, threshold: float = 0.6) -> bool:
        existing = self.get_all_key_memories(user_id, include_inactive=True)
        for mem in existing:
            shorter = text if len(text) <= len(mem) else mem
            longer = mem if len(text) <= len(mem) else text
            if shorter in longer:
                return True
            common = len(set(text) & set(mem))
            total = len(set(text) | set(mem))
            if total > 0 and common / total > threshold:
                return True
        return False

    def prune_stale_memories(self, user_id: int, min_importance: int = 2, days_unused: int = 30):
        """Delete low-importance memories that haven't been accessed in N days."""
        with self._get_conn() as conn:
            conn.execute(
                """DELETE FROM key_memories WHERE user_id = ?
                   AND importance < ?
                   AND last_accessed_at < datetime('now', ? || ' days')""",
                (user_id, min_importance, -days_unused),
            )

    def messages_since_last_extraction(self, user_id: int) -> int:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT last_msg_id FROM extraction_checkpoints WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            if row:
                last_id = row["last_msg_id"]
            else:
                legacy = conn.execute(
                    "SELECT COALESCE(MAX(source_msg_id), 0) AS last_msg_id FROM key_memories WHERE user_id = ?",
                    (user_id,),
                ).fetchone()
                last_id = legacy["last_msg_id"]
            count_row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM messages WHERE user_id = ? AND id > ?",
                (user_id, last_id),
            ).fetchone()
            return count_row["cnt"]

    # ── emotion + entity retrieval ─────────────────────────────

    EMOTION_KEYWORDS = EMOTION_KEYWORDS

    @staticmethod
    def detect_emotions(text: str) -> list[str]:
        return detect_emotions(text)

    def retrieve_memories_with_emotion(self, keywords: list[str], user_id: int,
                                        user_emotion_tags: list[str] | None = None,
                                        limit: int = 5) -> list[str]:
        """Retrieve memories with emotion-boosted scoring.

        Memories whose stored emotion tags overlap with the user's current
        emotional state get a score boost, simulating how humans more easily
        recall memories that match their current mood.
        """
        results: list[tuple[str, float]] = []
        user_emotions = user_emotion_tags or []
        meaningful_keywords = [kw for kw in keywords if kw.strip()]
        self.refresh_event_statuses(user_id)
        with self._get_conn() as conn:
            for kw in meaningful_keywords:
                rows = conn.execute(
                    """SELECT content, importance, access_count, emotion_tags, memory_type, status, created_at FROM key_memories
                       WHERE user_id = ? AND content LIKE ?""",
                    (user_id, f"%{kw}%"),
                ).fetchall()
                for row in rows:
                    if not self._is_recallable_memory(row["memory_type"], row["status"]):
                        continue
                    if not self._memory_is_fresh(row["content"], row["created_at"]):
                        continue
                    imp = row["importance"] or 1
                    acc = row["access_count"] or 0
                    boost = imp * 0.5 + min(acc * 0.1, 1.5)
                    # Emotion boost: overlap between user's current mood and memory's stored emotion
                    stored_emotions = (row["emotion_tags"] or "").split(",")
                    stored_emotions = [e.strip() for e in stored_emotions if e.strip()]
                    emotion_overlap = len(set(user_emotions) & set(stored_emotions))
                    if emotion_overlap > 0:
                        boost += emotion_overlap * 0.6
                    results.append((row["content"], 1.0 + boost))
                sum_rows = conn.execute(
                    "SELECT summary_text, created_at FROM summaries WHERE user_id = ? AND summary_text LIKE ?",
                    (user_id, f"%{kw}%"),
                ).fetchall()
                for row in sum_rows:
                    if not self._summary_is_fresh(row["summary_text"]):
                        continue
                    stripped = self._strip_summary_header(row["summary_text"])
                    if not self._summary_is_relevant(stripped, kw):
                        continue
                    results.append((stripped, 1.0))

        seen = set()
        unique: list[str] = []
        for content, score in sorted(results, key=lambda x: x[1], reverse=True):
            if content not in seen:
                seen.add(content)
                unique.append(content)

        top = unique[:limit]
        if top:
            with self._get_conn() as conn:
                for content in top:
                    conn.execute(
                        "UPDATE key_memories SET access_count = access_count + 1, last_accessed_at = CURRENT_TIMESTAMP WHERE user_id = ? AND content = ?",
                        (user_id, content),
                    )
        return top

    def get_entity_associated_memories(self, memory_contents: list[str], user_id: int,
                                        exclude_contents: set[str] | None = None,
                                        limit: int = 3) -> list[str]:
        """Find memories that share entity tags with the given ones.

        This mimics human associative recall: mentioning '年糕' the cat
        also brings up memories about the vet visit and the scratched sofa.
        """
        exclude = exclude_contents or set()
        exclude.update(memory_contents)
        if not exclude:
            return []

        entity_tags: set[str] = set()
        with self._get_conn() as conn:
            for content in memory_contents:
                rows = conn.execute(
                    "SELECT entity_tags FROM key_memories WHERE user_id = ? AND content = ?",
                    (user_id, content),
                ).fetchall()
                for row in rows:
                    if row["entity_tags"]:
                        for tag in row["entity_tags"].split(","):
                            tag = tag.strip()
                            if tag and len(tag) >= 2:
                                entity_tags.add(tag)

        if not entity_tags:
            return []

        scored: dict[str, int] = {}
        with self._get_conn() as conn:
            for tag in entity_tags:
                rows = conn.execute(
                    """SELECT content, importance, access_count, memory_type, status FROM key_memories
                       WHERE user_id = ? AND entity_tags LIKE ?""",
                    (user_id, f"%{tag}%"),
                ).fetchall()
                for row in rows:
                    if not self._is_recallable_memory(row["memory_type"], row["status"]):
                        continue
                    content = row["content"]
                    if content in exclude:
                        continue
                    score = (row["importance"] or 1) + (row["access_count"] or 0)
                    if content not in scored or score > scored[content]:
                        scored[content] = score

        ranked = sorted(scored.items(), key=lambda x: x[1], reverse=True)
        return [content for content, _ in ranked[:limit]]

    # ── active users ───────────────────────────────────────────

    def record_user_active(self, user_id: int):
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO active_users (user_id, last_message_at)
                   VALUES (?, CURRENT_TIMESTAMP)
                   ON CONFLICT(user_id) DO UPDATE SET last_message_at = CURRENT_TIMESTAMP""",
                (user_id,),
            )

    def get_active_user_ids(self) -> list[int]:
        with self._get_conn() as conn:
            rows = conn.execute("SELECT user_id FROM active_users ORDER BY last_message_at DESC").fetchall()
        return [r["user_id"] for r in rows]

    def get_last_active_time(self, user_id: int) -> str | None:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT last_message_at FROM active_users WHERE user_id = ?", (user_id,)
            ).fetchone()
        return row["last_message_at"] if row else None

    # ── proactive log ──────────────────────────────────────────

    def record_proactive_sent(self, user_id: int, content: str = "", topic_kind: str = ""):
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO proactive_log (user_id, content, topic_kind) VALUES (?, ?, ?)",
                (user_id, content, topic_kind),
            )

    def get_recent_proactive_content(self, user_id: int, limit: int = 3) -> list[str]:
        """Return the content of the most recent proactive messages sent to this user."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT content FROM proactive_log WHERE user_id = ? AND content != '' ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [r["content"] for r in rows]

    def get_recent_proactive_topics(self, user_id: int, limit: int = 5) -> list[str]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT topic_kind FROM proactive_log WHERE user_id = ? AND topic_kind != '' ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [r["topic_kind"] for r in rows]

    def count_proactive_topic_since(self, user_id: int, topic_kind: str, hours: int) -> int:
        with self._get_conn() as conn:
            row = conn.execute(
                """SELECT COUNT(*) AS cnt FROM proactive_log
                   WHERE user_id = ? AND topic_kind = ? AND sent_at >= datetime('now', '-' || ? || ' hours')""",
                (user_id, topic_kind, hours),
            ).fetchone()
        return row["cnt"] if row else 0

    def get_recent_summaries(self, user_id: int, limit: int = 3) -> list[str]:
        """Return the most recent conversation summaries for this user."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT summary_text FROM summaries WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [r["summary_text"] for r in rows]

    def get_last_proactive_time(self, user_id: int) -> str | None:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT sent_at FROM proactive_log WHERE user_id = ? ORDER BY id DESC LIMIT 1",
                (user_id,),
            ).fetchone()
        return row["sent_at"] if row else None

    def create_reminder(self, user_id: int, text: str, kind: str, timezone_name: str,
                        due_at: str | None, time_of_day: str | None, next_run_at: str) -> int:
        with self._get_conn() as conn:
            cur = conn.execute(
                """INSERT INTO reminders (user_id, text, kind, timezone, due_at, time_of_day, next_run_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (user_id, text, kind, timezone_name, due_at, time_of_day, next_run_at),
            )
            return int(cur.lastrowid)

    def get_due_reminders(self, now_iso: str, limit: int = 20) -> list[sqlite3.Row]:
        with self._get_conn() as conn:
            return conn.execute(
                """SELECT * FROM reminders
                   WHERE status = 'active' AND next_run_at <= ?
                   ORDER BY next_run_at ASC, id ASC
                   LIMIT ?""",
                (now_iso, limit),
            ).fetchall()

    def claim_reminder_occurrence(self, reminder_id: int, occurrence_key: str) -> bool:
        try:
            with self._get_conn() as conn:
                conn.execute(
                    """INSERT INTO reminder_deliveries (reminder_id, occurrence_key, status)
                       VALUES (?, ?, 'sending')""",
                    (reminder_id, occurrence_key),
                )
            return True
        except sqlite3.IntegrityError:
            return False

    def mark_reminder_sent(self, reminder_id: int, occurrence_key: str):
        with self._get_conn() as conn:
            conn.execute(
                """UPDATE reminder_deliveries
                   SET status = 'sent', sent_at = CURRENT_TIMESTAMP, error = ''
                   WHERE reminder_id = ? AND occurrence_key = ?""",
                (reminder_id, occurrence_key),
            )

    def mark_reminder_failed(self, reminder_id: int, occurrence_key: str, error: str):
        with self._get_conn() as conn:
            conn.execute(
                """UPDATE reminder_deliveries
                   SET status = 'failed', error = ?
                   WHERE reminder_id = ? AND occurrence_key = ?""",
                (error[:500], reminder_id, occurrence_key),
            )

    def complete_reminder(self, reminder_id: int):
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE reminders SET status = 'completed', updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (reminder_id,),
            )

    def advance_daily_reminder(self, reminder_id: int, next_run_at: str):
        with self._get_conn() as conn:
            conn.execute(
                "UPDATE reminders SET next_run_at = ?, updated_at = CURRENT_TIMESTAMP WHERE id = ?",
                (next_run_at, reminder_id),
            )

    def list_active_reminders(self, user_id: int) -> list[sqlite3.Row]:
        with self._get_conn() as conn:
            return conn.execute(
                "SELECT * FROM reminders WHERE user_id = ? AND status = 'active' ORDER BY next_run_at ASC, id ASC",
                (user_id,),
            ).fetchall()

    def cancel_reminder(self, user_id: int, reminder_id: int) -> bool:
        with self._get_conn() as conn:
            cur = conn.execute(
                """UPDATE reminders
                   SET status = 'cancelled', updated_at = CURRENT_TIMESTAMP
                   WHERE user_id = ? AND id = ? AND status = 'active'""",
                (user_id, reminder_id),
            )
            return cur.rowcount > 0

    def set_proactive_snooze(self, user_id: int, until: datetime, reason: str = ""):
        if until.tzinfo is not None:
            until = until.astimezone(timezone.utc).replace(tzinfo=None)
        with self._get_conn() as conn:
            conn.execute(
                """INSERT INTO proactive_snoozes (user_id, snooze_until, reason, updated_at)
                   VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                   ON CONFLICT(user_id) DO UPDATE SET
                       snooze_until = excluded.snooze_until,
                       reason = excluded.reason,
                       updated_at = CURRENT_TIMESTAMP""",
                (user_id, until.strftime("%Y-%m-%d %H:%M:%S"), reason),
            )

    def get_proactive_snooze_until(self, user_id: int) -> str | None:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT snooze_until FROM proactive_snoozes WHERE user_id = ? AND snooze_until > CURRENT_TIMESTAMP",
                (user_id,),
            ).fetchone()
        return row["snooze_until"] if row else None

    def clear_proactive_snooze(self, user_id: int):
        with self._get_conn() as conn:
            conn.execute("DELETE FROM proactive_snoozes WHERE user_id = ?", (user_id,))

    def clear_expired_proactive_snoozes(self):
        with self._get_conn() as conn:
            conn.execute("DELETE FROM proactive_snoozes WHERE snooze_until <= CURRENT_TIMESTAMP")

    def count_proactive_since_last_user_message(self, user_id: int) -> int:
        """Count proactive messages sent after the user's last message. Used for DND detection."""
        with self._get_conn() as conn:
            last_msg = conn.execute(
                "SELECT last_message_at FROM active_users WHERE user_id = ?", (user_id,)
            ).fetchone()
            if not last_msg or not last_msg["last_message_at"]:
                return 0
            row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM proactive_log WHERE user_id = ? AND sent_at > ?",
                (user_id, last_msg["last_message_at"]),
            ).fetchone()
            return row["cnt"]if row else 0
