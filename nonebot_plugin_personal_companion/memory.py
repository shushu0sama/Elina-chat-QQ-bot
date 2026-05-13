import sqlite3
import json
from pathlib import Path
from datetime import datetime


class MemoryStore:
    """SQLite-backed memory with message logging, summaries, and keyword retrieval."""

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
                    start_msg_id INTEGER,
                    end_msg_id INTEGER,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS key_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    source_msg_id INTEGER,
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
                    sent_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            # Migration: add user_id column if it doesn't exist (for old DBs)
            try:
                conn.execute("ALTER TABLE messages ADD COLUMN user_id INTEGER")
            except Exception:
                pass  # Column already exists

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

    def message_count_since_last_summary(self) -> int:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(end_msg_id), 0) AS last_id FROM summaries"
            ).fetchone()
            last_id = row["last_id"]
            count_row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM messages WHERE id > ?", (last_id,)
            ).fetchone()
            return count_row["cnt"]

    def save_summary(self, summary_text: str, start_msg_id: int, end_msg_id: int):
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO summaries (summary_text, start_msg_id, end_msg_id) VALUES (?, ?, ?)",
                (summary_text, start_msg_id, end_msg_id),
            )

    def add_key_memory(self, content: str, source_msg_id: int | None = None):
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO key_memories (content, source_msg_id) VALUES (?, ?)",
                (content, source_msg_id),
            )

    def retrieve_memories(self, keywords: list[str], limit: int = 5) -> list[str]:
        """Retrieve relevant memories using keyword matching on summaries and key_memories."""
        results: list[tuple[str, int]] = []  # (content, score)

        with self._get_conn() as conn:
            for kw in keywords:
                # Search key_memories
                mem_rows = conn.execute(
                    "SELECT content FROM key_memories WHERE content LIKE ?",
                    (f"%{kw}%",),
                ).fetchall()
                for row in mem_rows:
                    results.append((row["content"], 1))

                # Search summaries
                sum_rows = conn.execute(
                    "SELECT summary_text FROM summaries WHERE summary_text LIKE ?",
                    (f"%{kw}%",),
                ).fetchall()
                for row in sum_rows:
                    results.append((row["summary_text"], 2))

        # Deduplicate, sort by score desc, take top N
        seen = set()
        unique: list[str] = []
        for content, score in sorted(results, key=lambda x: x[1], reverse=True):
            if content not in seen:
                seen.add(content)
                unique.append(content)
        return unique[:limit]

    def get_all_key_memories(self) -> list[str]:
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT content FROM key_memories ORDER BY created_at DESC"
            ).fetchall()
        return [r["content"] for r in rows]

    def count_key_memories(self) -> int:
        with self._get_conn() as conn:
            row = conn.execute("SELECT COUNT(*) AS cnt FROM key_memories").fetchone()
            return row["cnt"]

    def has_similar_memory(self, text: str, threshold: float = 0.6) -> bool:
        """Check if a similar memory already exists using simple substring overlap."""
        existing = self.get_all_key_memories()
        for mem in existing:
            # Jaccard-like: if either contains the other, or strong overlap
            shorter = text if len(text) <= len(mem) else mem
            longer = mem if len(text) <= len(mem) else text
            if shorter in longer:
                return True
            # Check shared characters as a simple overlap metric
            common = len(set(text) & set(mem))
            total = len(set(text) | set(mem))
            if total > 0 and common / total > threshold:
                return True
        return False

    def get_user_message_stats(self, user_id: int) -> dict:
        """Return message count, average length, and first-chat date for a user."""
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

    def get_messages_since(self, when: str, user_id: int, limit: int = 500) -> list[dict]:
        """Get messages for a user since a given ISO timestamp, ordered chronologically."""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT role, content, created_at FROM messages WHERE user_id = ? AND created_at >= ? ORDER BY id ASC LIMIT ?",
                (user_id, when, limit),
            ).fetchall()
        return [{"role": r["role"], "content": r["content"], "time": r["created_at"]} for r in rows]

    # ── active users ──────────────────────────────────────────

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
            rows = conn.execute(
                "SELECT user_id FROM active_users ORDER BY last_message_at DESC"
            ).fetchall()
        return [r["user_id"] for r in rows]

    def get_last_active_time(self, user_id: int) -> str | None:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT last_message_at FROM active_users WHERE user_id = ?",
                (user_id,),
            ).fetchone()
        return row["last_message_at"] if row else None

    def record_proactive_sent(self, user_id: int):
        with self._get_conn() as conn:
            conn.execute(
                "INSERT INTO proactive_log (user_id) VALUES (?)",
                (user_id,),
            )

    def get_last_proactive_time(self, user_id: int) -> str | None:
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT sent_at FROM proactive_log WHERE user_id = ? ORDER BY id DESC LIMIT 1",
                (user_id,),
            ).fetchone()
        return row["sent_at"] if row else None

    def messages_since_last_extraction(self) -> int:
        """Count messages since the last auto-extraction marker."""
        with self._get_conn() as conn:
            row = conn.execute(
                "SELECT COALESCE(MAX(source_msg_id), 0) AS last_msg_id FROM key_memories"
            ).fetchone()
            last_msg_id = row["last_msg_id"]
            count_row = conn.execute(
                "SELECT COUNT(*) AS cnt FROM messages WHERE id > ?", (last_msg_id,)
            ).fetchone()
            return count_row["cnt"]
