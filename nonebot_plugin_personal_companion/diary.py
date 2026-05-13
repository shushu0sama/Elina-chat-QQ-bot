from pathlib import Path
from datetime import datetime, timedelta

from .llm_client import LLMClient
from .memory import MemoryStore

DIARIES_DIR = Path(__file__).parent / "diaries"


class DiaryWriter:
    def __init__(self, llm: LLMClient, memory: MemoryStore):
        self.llm = llm
        self.memory = memory
        DIARIES_DIR.mkdir(parents=True, exist_ok=True)

    async def write_daily_diary(self):
        """Generate and save diary entries for all active users. Called by midnight cron."""
        today = datetime.now()
        yesterday = today - timedelta(days=1)
        diary_date = yesterday.strftime("%Y-%m-%d")
        since = yesterday.strftime("%Y-%m-%d 00:00:00")

        user_ids = self.memory.get_active_user_ids()
        if not user_ids:
            print("[Diary] No active users, skipping")
            return

        for uid in user_ids:
            try:
                await self._write_for_user(uid, diary_date, since)
            except Exception as e:
                print(f"[Diary] Error for user {uid}: {e}")

    async def _write_for_user(self, user_id: int, diary_date: str, since: str):
        messages = self.memory.get_messages_since(since, user_id, limit=300)

        if len(messages) < 3:
            print(f"[Diary] {diary_date} user {user_id}: only {len(messages)} msgs, skipping")
            return

        content = await self._generate(messages, diary_date)
        if not content:
            return

        self._save(user_id, diary_date, content)

    async def _generate(self, messages: list[dict], date_str: str) -> str | None:
        conversation = []
        for m in messages:
            role_label = "你" if m["role"] == "user" else "艾琳娜"
            content = m["content"]
            if len(content) > 300:
                content = content[:300] + "..."
            conversation.append(f"{role_label}: {content}")

        if not conversation:
            return None

        conversation_text = "\n".join(conversation)

        prompt = (
            f"以下是{date_str}你和一个朋友的完整对话记录。请你以朋友「艾琳娜」的口吻，"
            "为对方写一篇个人日记。\n\n"
            "要求：\n"
            "- 用第二人称「你」来写，像是在帮朋友记日记\n"
            "- 分三个小节：## 聊聊 / ## 今日心情 / ## 艾琳娜的碎碎念\n"
            "- 「聊聊」：总结今天聊了哪些话题，简洁但不遗漏重要内容\n"
            "- 「今日心情」：根据对话推测对方今天的情绪变化\n"
            "- 「艾琳娜的碎碎念」：以艾琳娜的视角写几句温柔的观察或感慨，1-3句话\n"
            "- 语气自然、温柔，不肉麻，不鸡汤\n"
            "- 300-500字，不要太长\n\n"
            f"[对话记录]\n{conversation_text}"
        )

        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": "你是一个温柔的朋友，帮对方写个人日记。语气自然，像朋友间的记录，不是正式文书。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1024,
            )
            if response and len(response) > 30:
                return response
        except Exception as e:
            print(f"[Diary] LLM error: {e}")

        return None

    def _save(self, user_id: int, date_str: str, content: str):
        user_dir = DIARIES_DIR / str(user_id)
        user_dir.mkdir(parents=True, exist_ok=True)

        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            date_heading = dt.strftime("%Y年%m月%d日")
        except ValueError:
            date_heading = date_str

        full_content = (
            f"# {date_heading} 日记\n\n"
            f"{content}\n\n"
            f"---\n"
            f"生成于 {datetime.now().strftime('%Y-%m-%d %H:%M')}\n"
        )

        filepath = user_dir / f"{date_str}.md"
        filepath.write_text(full_content, encoding="utf-8")
        print(f"[Diary] Saved: {filepath}")
