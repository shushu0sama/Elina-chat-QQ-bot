import random
from datetime import datetime

from nonebot import get_bot

from .config import Config
from .memory import MemoryStore
from .personality import build_system_prompt
from .llm_client import LLMClient
from .knowledge import KnowledgeBase, build_knowledge_prompt


class ProactiveChat:
    def __init__(self, memory: MemoryStore, llm: LLMClient, config: Config, kb: KnowledgeBase | None = None):
        self.memory = memory
        self.llm = llm
        self.config = config
        self.kb = kb

    def _filter_allowed(self, user_ids: list[int]) -> list[int]:
        allow = self.config.proactive_allow_users.strip()
        if not allow:
            return user_ids
        allowed = {int(x.strip()) for x in allow.split(",") if x.strip()}
        return [uid for uid in user_ids if uid in allowed]

    async def try_proactive(self):
        if not self.config.proactive_enabled:
            return

        now = datetime.now()
        if not (self.config.proactive_active_hours_start <= now.hour < self.config.proactive_active_hours_end):
            return

        user_ids = self._filter_allowed(self.memory.get_active_user_ids())
        if not user_ids:
            return

        for user_id in user_ids:
            if self._should_send(user_id, now):
                await self._send_to_user(user_id)

    def _should_send(self, user_id: int, now: datetime) -> bool:
        # DND: stop if user ignored 3+ consecutive proactive messages
        ignored = self.memory.count_proactive_since_last_user_message(user_id)
        if ignored >= 3:
            return False

        # Check cooldown: user must have been silent for COOLDOWN_MINUTES
        last_active = self.memory.get_last_active_time(user_id)
        if last_active:
            last_active_dt = datetime.fromisoformat(last_active)
            cooldown_min = (now - last_active_dt).total_seconds() / 60
            if cooldown_min < self.config.proactive_cooldown_minutes:
                return False

        # Check interval since last proactive message
        last_proactive = self.memory.get_last_proactive_time(user_id)
        if last_proactive:
            last_p_dt = datetime.fromisoformat(last_proactive)
            since_last = (now - last_p_dt).total_seconds() / 60
            if since_last < self.config.proactive_interval_minutes:
                return False

        # Add ~25% jitter: randomly skip ~1 in 4 eligible windows
        if random.random() < 0.25:
            return False

        return True

    async def _send_to_user(self, user_id: int):
        try:
            bot = get_bot()
        except Exception:
            return

        prompt = self._build_proactive_prompt(user_id)
        try:
            reply = self.llm.chat([{"role": "system", "content": prompt}])
        except Exception:
            return

        if not reply:
            return

        # Record BEFORE sending — interval timing uses this
        self.memory.record_proactive_sent(user_id, content=reply)

        for chunk in LLMClient.chunk_text(reply):
            await bot.send_private_msg(user_id=user_id, message=chunk)

        print(f"[Companion] Proactive message sent to user {user_id}")

    def _build_proactive_prompt(self, user_id: int) -> str:
        now = datetime.now()

        # Time-of-day context
        hour = now.hour
        if 6 <= hour <= 9:
            time_ctx = "现在是早晨。可以问候早安，关心一下对方今天的状态——带一点点书里的视角，轻轻的一句就够了。"
        elif 10 <= hour <= 12:
            time_ctx = "现在是上午。可以问问对方在做什么，或者分享一个轻松的想法。"
        elif 13 <= hour <= 14:
            time_ctx = "现在是午后。关心一下对方有没有午休，语气轻松。"
        elif 15 <= hour <= 17:
            time_ctx = "现在是下午。可以问问今天过得怎么样。"
        elif 18 <= hour <= 20:
            time_ctx = "现在是傍晚。可以关心一下晚饭，或者分享一句温柔的感慨。"
        else:
            time_ctx = "现在是晚上。语气柔和一点，关心对方今天的状态和心情。"

        # ── Conversation summaries (high-level view of what was discussed AND resolved) ──
        summaries = self.memory.get_recent_summaries(user_id, limit=3)
        summary_block = ""
        if summaries:
            summary_block = (
                "【最近对话摘要——这些话题已经聊过了，不要重复提起：】\n"
                + "\n---\n".join(f"摘要 {i+1}：{s}" for i, s in enumerate(summaries))
                + "\n\n请据此判断：如果某个话题在摘要中已被讨论并完结，就不要再提起。"
            )

        # ── User memories — split by type ──
        memories = self.memory.get_all_key_memories(user_id)
        memory_block = ""
        if memories:
            # Rough split: short factual memories vs longer event-style ones
            facts = [m for m in memories if len(m) < 30 and not any(
                kw in m for kw in ["今天", "昨天", "刚", "正在", "最近", "下午", "早上", "晚上"]
            )]
            events = [m for m in memories if m not in facts]

            parts = []
            if facts:
                parts.append("【关于对方的长期偏好/习惯——可以作为聊天背景：】\n" + "\n".join(f"- {f}" for f in facts[:8]))
            if events:
                parts.append("【已发生的事件——这些大多已经完结，除非特别相关否则不要追问：】\n" + "\n".join(f"- {e}" for e in events[:5]))
            if parts:
                memory_block = "\n\n".join(parts)
                memory_block += "\n\n重要：上面的事件是过去式。比如对方说过'吃完了馄饨'，说明这件事已经完结，不要再去问'馄饨吃完了吗'。"

        # ── Recent conversation ──
        recent = self.memory.get_recent_messages(limit=12, user_id=user_id)
        recent_block = ""
        if recent:
            recent_lines = []
            for m in recent:
                role_label = "对方" if m["role"] == "user" else "你"
                recent_lines.append(f"{role_label}: {m['content'][:200]}")
            recent_block = "【最近聊天记录——了解当前上下文：】\n" + "\n".join(recent_lines)

        # ── Proactive history — what the bot already sent proactively ──
        prev_proactive = self.memory.get_recent_proactive_content(user_id, limit=3)
        proactive_block = ""
        if prev_proactive:
            proactive_block = (
                "【你最近主动找对方时发的消息——这次绝对不能重复的话题：】\n"
                + "\n".join(f"- 第{i+1}次: 「{msg[:150]}」" for i, msg in enumerate(prev_proactive))
                + "\n\n以上话题已经说过了，这次必须换全新的切入点。如果最近一次的主动消息对方没回，"
                + "说明对方可能不感兴趣，不要再延续那个话题。"
            )

        # ── Philosophy knowledge ──
        kb_block = ""
        if self.kb and recent:
            last_user_msgs = [m["content"] for m in recent if m["role"] == "user"]
            if last_user_msgs:
                kb_block = build_knowledge_prompt(last_user_msgs[-1], self.kb)

        # ── Personality ──
        persona = build_system_prompt(user_id=user_id)

        # ── Time since last active ──
        last_active = self.memory.get_last_active_time(user_id)
        gap_hint = ""
        if last_active:
            last_dt = datetime.fromisoformat(last_active)
            gap_min = (now - last_dt).total_seconds() / 60
            if gap_min >= 120:
                gap_hint = f"对方已经{int(gap_min / 60)}小时多没和你说话了，可能是在忙。语气温和，不要催促。"
            elif gap_min >= 60:
                gap_hint = "对方有一阵子没和你说话了。简单问候就好，不要给对方压力。"

        # ── Assemble ──
        return "\n".join(filter(None, [
            "你需要主动给朋友发一条消息。",
            "",
            persona,
            "",
            time_ctx,
            "",
            "你的目标：",
            "- 只发一条消息，表达关心或分享一个小小的观察/想法",
            "- 保持自然、像真人朋友，不要像是在完成任务",
            "- 2-4句话即可，不要长篇大论",
            "- 不要问太多问题，至多问一个",
            "- 不要用'好久不见''好几天没聊了'这类话，除非给出的数据里确实很久没聊",
            "- 绝对不要以'作为XX'开头介绍自己是谁",
            "",
            "【核心规则——务必遵守：】",
            "- 不要重复你最近主动发过的任何话题（见下方'你最近主动发了什么'）",
            "- 不要追问已经完结的事件（见下方'已发生的事件'和'对话摘要'）",
            "- 如果找不到合适的新话题，就分享一个自己的小观察或当下的感受，而不是翻旧话题",
            "",
            summary_block,
            "",
            memory_block,
            "",
            recent_block,
            "",
            proactive_block,
            "",
            kb_block,
            "",
            gap_hint,
        ])).strip()
