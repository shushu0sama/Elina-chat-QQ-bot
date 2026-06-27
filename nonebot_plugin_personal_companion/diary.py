import asyncio
from pathlib import Path
from datetime import datetime, timedelta

from nonebot import get_bot
from nonebot.log import logger

from .llm_client import LLMClient
from .memory import MemoryStore
from .config import Config
from .personality import BEIJING_TZ

DIARIES_DIR = Path(__file__).parent / "diaries"


class DiaryWriter:
    def __init__(self, llm: LLMClient, memory: MemoryStore, config: Config):
        self.llm = llm
        self.memory = memory
        self.config = config
        DIARIES_DIR.mkdir(parents=True, exist_ok=True)

    def _filter_allowed(self, user_ids: list[int]) -> list[int]:
        allow = self.config.proactive_allow_users.strip()
        if not allow:
            return user_ids
        allowed = set()
        for item in allow.split(","):
            item = item.strip()
            if item.isdigit():
                allowed.add(int(item))
            elif item:
                logger.warning(f"Ignoring invalid proactive_allow_users token: {item}")
        return [uid for uid in user_ids if uid in allowed]

    async def write_daily_diary(self):
        """Generate and save diary entries for all active users. Called by midnight cron."""
        today = datetime.now(BEIJING_TZ)
        yesterday = today - timedelta(days=1)
        diary_date = yesterday.strftime("%Y-%m-%d")
        since = yesterday.strftime("%Y-%m-%d 00:00:00")

        user_ids = self._filter_allowed(self.memory.get_active_user_ids())
        if not user_ids:
            logger.info("No active users, skipping")
            return

        for uid in user_ids:
            try:
                await self._write_for_user(uid, diary_date, since)
                await self._send_manifestation_diary_for_user(uid, diary_date, since)
            except Exception as e:
                logger.warning(f"Error for user {uid}: {e}")

    async def _write_for_user(self, user_id: int, diary_date: str, since: str):
        messages = self.memory.get_messages_since(since, user_id, limit=300)

        if len(messages) < 3:
            logger.info(f"{diary_date} user {user_id}: only {len(messages)} msgs, skipping")
            return

        timeline_entries = self.memory.get_timeline_entries_between(user_id, diary_date, diary_date, limit=20)
        content = await self._generate(messages, diary_date, timeline_entries)
        if not content:
            return

        self._save(user_id, diary_date, content)

    async def _send_manifestation_diary_for_user(self, user_id: int, diary_date: str, since: str):
        manifestation_memories = self.memory.get_manifestation_memories(user_id, limit=8)
        wishes = self.memory.get_manifestation_wishes(user_id, limit=8)
        recent_evidence = self.memory.get_manifestation_evidence(user_id, limit=12)
        messages = self.memory.get_messages_since(since, user_id, limit=300)
        if not manifestation_memories and not wishes and not self._has_manifestation_conversation(messages):
            return
        if len(messages) < 2 and not manifestation_memories:
            return

        content = await self._generate_manifestation_diary(messages, manifestation_memories, diary_date, wishes, recent_evidence)
        if not content:
            return

        self._save_manifestation(user_id, diary_date, content)
        try:
            bot = get_bot()
        except Exception:
            return

        message = f"今晚的显化日记来了。\n\n{content}"
        for chunk in LLMClient.chunk_text(message):
            await bot.send_private_msg(user_id=user_id, message=chunk)

    @staticmethod
    def _has_manifestation_conversation(messages: list[dict]) -> bool:
        markers = ["显化", "愿望", "信念", "执念", "未来自我", "显化证据", "小魔女"]
        return any(any(marker in m["content"] for marker in markers) for m in messages)

    async def _generate_manifestation_diary(self, messages: list[dict], manifestation_memories: list[str], date_str: str,
                                            wishes: list[dict] | None = None,
                                            recent_evidence: list[dict] | None = None) -> str | None:
        conversation = []
        for m in messages:
            role_label = "你" if m["role"] == "user" else "艾琳娜"
            content = m["content"]
            if len(content) > 240:
                content = content[:240] + "..."
            conversation.append(f"{role_label}: {content}")

        memory_text = "\n".join(f"- {m}" for m in manifestation_memories) if manifestation_memories else "（暂无显化记忆）"
        wish_text = "\n".join(f"- #{w['id']} {w['title']}｜状态：{w['status']}" for w in (wishes or [])) or "（暂无愿望生命周期记录）"
        evidence_text = "\n".join(f"- {e['content']}" for e in (recent_evidence or [])) or "（暂无累计证据链）"
        conversation_text = "\n".join(conversation) if conversation else "（今天没有明显聊天内容）"

        prompt = (
            f"请根据{date_str}的聊天内容、愿望生命周期和显化记忆，为用户写一份睡前显化日记。\n\n"
            "愿望生命周期：\n"
            f"{wish_text}\n\n"
            "累计显化证据链：\n"
            f"{evidence_text}\n\n"
            "显化记忆：\n"
            f"{memory_text}\n\n"
            "今日聊天：\n"
            f"{conversation_text}\n\n"
            "要求：\n"
            "- 用第二人称'你'，口吻是艾琳娜，温柔、清醒、有一点小魔女感\n"
            "- 分四节：## 今日状态 / ## 今日显化证据 / ## 今天可以放下 / ## 明日对齐行动\n"
            "- 如果有 active 愿望，围绕 active 愿望写；released/fulfilled/paused 的愿望不要写成还需要追逐\n"
            "- 显化证据优先写内部证据和行动证据，不要硬编外部结果\n"
            "- 不承诺结果，不说宇宙一定会给，不说没成功是频率不够\n"
            "- 如果今天聊天很少，就基于已有愿望和记忆写一份轻量版，不要假装知道用户今天发生了什么\n"
            "- 250-450字"
        )

        try:
            response = await asyncio.to_thread(
                self.llm.chat,
                messages=[
                    {"role": "system", "content": "你是艾琳娜的显化日记写作模块。你帮助用户澄清愿望、收集证据、放下执念、对齐行动，但不承诺结果。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1536,
            )
            if response and len(response) > 30:
                return response
        except Exception as e:
            logger.warning(f"Manifestation diary LLM error: {e}")

        return None

    def _save_manifestation(self, user_id: int, date_str: str, content: str):
        user_dir = DIARIES_DIR / str(user_id) / "manifestation"
        user_dir.mkdir(parents=True, exist_ok=True)

        try:
            dt = datetime.strptime(date_str, "%Y-%m-%d")
            date_heading = dt.strftime("%Y年%m月%d日")
        except ValueError:
            date_heading = date_str

        full_content = (
            f"# {date_heading} 显化日记\n\n"
            f"{content}\n\n"
            f"---\n"
            f"生成于 {datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M')}\n"
        )

        filepath = user_dir / f"{date_str}.md"
        filepath.write_text(full_content, encoding="utf-8")
        logger.info(f"Saved manifestation diary: {filepath}")

    async def _generate(self, messages: list[dict], date_str: str,
                        timeline_entries: list[dict] | None = None) -> str | None:
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
        timeline_text = "（无当天时间线事件）"
        if timeline_entries:
            lines = []
            for item in timeline_entries:
                event_time = f" {item['event_time']}" if item.get("event_time") else ""
                lines.append(f"- {item['event_date']}{event_time}：{item['content']}")
            timeline_text = "\n".join(lines)

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
            f"[当天时间线事件]\n{timeline_text}\n\n"
            f"[对话记录]\n{conversation_text}"
        )

        try:
            response = await asyncio.to_thread(
                self.llm.chat,
                messages=[
                    {"role": "system", "content": "你是一个温柔的朋友，帮对方写个人日记。语气自然，像朋友间的记录，不是正式文书。"},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2048,
            )
            if response and len(response) > 30:
                return response
        except Exception as e:
            logger.warning(f"LLM error: {e}")

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
            f"生成于 {datetime.now(BEIJING_TZ).strftime('%Y-%m-%d %H:%M')}\n"
        )

        filepath = user_dir / f"{date_str}.md"
        filepath.write_text(full_content, encoding="utf-8")
        logger.info(f"Saved: {filepath}")
