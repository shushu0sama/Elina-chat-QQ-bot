import sqlite3
from pathlib import Path


class MemoryStore:
    """SQLite-backed memory with full per-user isolation."""

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
                    last_accessed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
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
            """)
            for col, table in [
                ("user_id", "messages"), ("user_id", "summaries"), ("user_id", "key_memories"),
                ("importance", "key_memories"), ("access_count", "key_memories"),
                ("content", "proactive_log"),
            ]:
                try:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} INTEGER")
                except Exception:
                    pass

    # ── messages ───────────────────────────────────────────────

    def save_message(self, role: str, content: str, user_id: int | None = None) -> int:
        with self._get_conn() as conn:
            cur = conn.execute(
                "INSERT INTO messages (role, content, user_id) VALUES (?, ?, ?)",
                (role, content, user_id),
            )
            return cur.lastrowid

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

    def add_key_memory(self, content: str, source_msg_id: int | None = None,
                       user_id: int | None = None, importance: int = 1):
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO key_memories (content, source_msg_id, user_id, importance) VALUES (?, ?, ?, ?)",
                (content, source_msg_id, user_id, importance),
            )

    def retrieve_memories(self, keywords: list[str], user_id: int, limit: int = 5) -> list[str]:
        results: list[tuple[str, float]] = []
        with self._get_conn() as conn:
            for kw in keywords:
                rows = conn.execute(
                    "SELECT content, importance, access_count FROM key_memories WHERE user_id = ? AND content LIKE ?",
                    (user_id, f"%{kw}%"),
                ).fetchall()
                for row in rows:
                    # Score: base 1 + importance boost (0-4) + access boost (log-scaled)
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

    def get_all_key_memories(self, user_id: int | None = None) -> list[str]:
        with self._get_conn() as conn:
            if user_id is not None:
                rows = conn.execute(
                    "SELECT content FROM key_memories WHERE user_id = ? ORDER BY created_at DESC",
                    (user_id,),
                ).fetchall()
            else:
                rows = conn.execute("SELECT content FROM key_memories ORDER BY created_at DESC").fetchall()
        return [r["content"] for r in rows]

    def count_key_memories(self, user_id: int | None = None) -> int:
        with self._get_conn() as conn:
            if user_id is not None:
                row = conn.execute("SELECT COUNT(*) AS cnt FROM key_memories WHERE user_id = ?", (user_id,)).fetchone()
            else:
                row = conn.execute("SELECT COUNT(*) AS cnt FROM key_memories").fetchone()
            return row["cnt"]

    def has_similar_memory(self, text: str, user_id: int, threshold: float = 0.6) -> bool:
        existing = self.get_all_key_memories(user_id)
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
                "SELECT COALESCE(MAX(source_msg_id), 0) AS last_msg_id FROM key_memories WHERE user_id = ?",
                (user_id,),
            ).fetchone()
            last_msg_id = row["last_msg_id"]
            count_row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM messages WHERE user_id = ? AND id > ?",
                (user_id, last_msg_id),
            ).fetchone()
            return count_row["cnt"]

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
