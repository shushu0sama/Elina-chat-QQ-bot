import sqlite3
from pathlib import Path

from .turn_context import EMOTION_KEYWORDS, detect_emotions


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
            """)
            int_columns = [
                ("user_id", "messages"), ("user_id", "summaries"), ("user_id", "key_memories"),
                ("importance", "key_memories"), ("access_count", "key_memories"),
                ("user_id", "manifestation_wishes"), ("wish_id", "manifestation_evidence"),
            ]
            text_columns = [
                ("content", "proactive_log"),
                ("emotion_tags", "key_memories"), ("entity_tags", "key_memories"),
                ("memory_type", "key_memories"), ("status", "key_memories"),
                ("status", "manifestation_wishes"), ("raw_content", "manifestation_wishes"),
                ("evidence_type", "manifestation_evidence"),
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

    # ── messages ───────────────────────────────────────────────

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
                    "SELECT role, content FROM messages WHERE user_id = ? ORDER BY id DESC LIMIT ?",
                    (user_id, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?",
                    (limit,),
                ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def get_messages_since(self, when: str, user_id: int, limit: int = 500) -> list[dict]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT role, content, created_at FROM messages WHERE user_id = ? AND created_at >= ? ORDER BY id ASC LIMIT ?",
                (user_id, when, limit),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"], "time": r["created_at"]} for r in rows]

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

    def save_summary(self, summary_text: str, start_msg_id: int, end_msg_id: int, user_id: int):
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO summaries (summary_text, start_msg_id, end_msg_id, user_id) VALUES (?, ?, ?, ?)",
                (summary_text, start_msg_id, end_msg_id, user_id),
            )

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
        with self._get_conn() as conn:
            for kw in keywords:
                rows = conn.execute(
                    """SELECT content, importance, access_count, memory_type, status FROM key_memories
                       WHERE user_id = ? AND content LIKE ?""",
                    (user_id, f"%{kw}%"),
                ).fetchall()
                for row in rows:
                    if not self._is_recallable_memory(row["memory_type"], row["status"]):
                        continue
                    imp = row["importance"] or 1
                    acc = row["access_count"] or 0
                    boost = imp * 0.5 + min(acc * 0.1, 1.5)
                    results.append((row["content"], 1.0 + boost))
                sum_rows = conn.execute(
                    "SELECT summary_text FROM summaries WHERE user_id = ? AND summary_text LIKE ?",
                    (user_id, f"%{kw}%"),
                ).fetchall()
                for row in sum_rows:
                    results.append((row["summary_text"], 2.0))

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
        lines = ["我现在记得这些："]
        for item in items:
            lines.append(f"- #{item['id']} [{item['memory_type']}/{item['status']}] {item['content']}")
        lines.append("\n你可以说「忘掉：关键词」「这件事结束了：关键词」「以后别再提：关键词」来整理。")
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
        self.refresh_event_statuses(user_id)
        with self._get_conn() as conn:
            for kw in keywords:
                rows = conn.execute(
                    """SELECT content, importance, access_count, emotion_tags, memory_type, status FROM key_memories
                       WHERE user_id = ? AND content LIKE ?""",
                    (user_id, f"%{kw}%"),
                ).fetchall()
                for row in rows:
                    if not self._is_recallable_memory(row["memory_type"], row["status"]):
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
                    "SELECT summary_text FROM summaries WHERE user_id = ? AND summary_text LIKE ?",
                    (user_id, f"%{kw}%"),
                ).fetchall()
                for row in sum_rows:
                    results.append((row["summary_text"], 1.0))

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

    def record_proactive_sent(self, user_id: int, content: str = ""):
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO proactive_log (user_id, content) VALUES (?, ?)",
                (user_id, content),
            )

    def get_recent_proactive_content(self, user_id: int, limit: int = 3) -> list[str]:
        """Return the content of the most recent proactive messages sent to this user."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT content FROM proactive_log WHERE user_id = ? AND content != '' ORDER BY id DESC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        return [r["content"] for r in rows]

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
