import random
from datetime import datetime, timezone

from nonebot import get_bot
from nonebot.log import logger

from .config import Config
from .memory import MemoryStore
from .personality import BEIJING_TZ, build_system_prompt
from .llm_client import LLMClient
from .knowledge import KnowledgeBase, build_knowledge_prompt
from .turn_context import analyze_turn
from .manifestation_quotes import (
    build_frequency_first_aid_text,
    detect_frequency_support_category,
    pick_manifestation_quote,
)


class ProactiveChat:
    MANIFESTATION_MARKERS = ["显化", "愿望", "信念", "执念", "未来自我", "证据", "咒语", "小魔女"]
    FREQUENCY_MARKERS = ["过去", "以前", "旧事", "旧故事", "焦虑", "迷茫", "频率", "低频", "放不下", "后悔"]

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

        now = datetime.now(BEIJING_TZ)
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
            last_active_dt = datetime.fromisoformat(last_active).replace(tzinfo=timezone.utc)
            cooldown_min = (now - last_active_dt.astimezone(BEIJING_TZ)).total_seconds() / 60
            if cooldown_min < self.config.proactive_cooldown_minutes:
                return False

        # Respect explicit ending / busy / sleep signals from the latest user message.
        recent = self.memory.get_recent_messages(limit=6, user_id=user_id)
        last_user = next((m["content"] for m in reversed(recent) if m["role"] == "user"), "")
        if last_user:
            turn_ctx = analyze_turn(last_user, recent)
            if turn_ctx.should_end_softly:
                return False

        # Check interval since last proactive message
        last_proactive = self.memory.get_last_proactive_time(user_id)
        if last_proactive:
            last_p_dt = datetime.fromisoformat(last_proactive).replace(tzinfo=timezone.utc)
            since_last = (now - last_p_dt.astimezone(BEIJING_TZ)).total_seconds() / 60
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

        logger.info(f"Proactive message sent to user {user_id}")

    def _should_offer_manifestation_checkin(self, manifestation: list[str], recent: list[dict], prev_proactive: list[str], now: datetime) -> bool:
        if not manifestation:
            return False
        if any(any(marker in msg for marker in self.MANIFESTATION_MARKERS) for msg in prev_proactive):
            return False
        recent_user = [m["content"] for m in recent if m["role"] == "user"]
        if recent_user and any(any(marker in msg for marker in self.MANIFESTATION_MARKERS) for msg in recent_user[-3:]):
            return True
        return now.hour in {8, 9, 21, 22}

    def _build_manifestation_checkin_hint(self, manifestation: list[str], recent: list[dict], prev_proactive: list[str], now: datetime) -> str:
        if not self._should_offer_manifestation_checkin(manifestation, recent, prev_proactive, now):
            return ""

        if 6 <= now.hour <= 10:
            examples = [
                "早安。今天想不想给你的愿望设一个很小的对齐行动？",
                "早上好。今天先不检查结果，只选一个你愿意相信的小念头，好吗？",
            ]
            mode = "偏早晨设定：状态、信念、今日行动。"
        elif 20 <= now.hour <= 23:
            examples = [
                "今晚要不要收集一个小小的显化证据？哪怕只是你比昨天稳定了一点。",
                "睡前轻轻问一句：今天有没有哪个瞬间，说明你正在回到自己？",
            ]
            mode = "偏睡前复盘：显化证据、放下执念、自我照顾。"
        else:
            examples = [
                "我想起你之前种下的愿望。今天要不要只做一个很轻的状态校准？",
                "不检查结果，我们只看看：此刻的你需要回到哪个更稳定的状态？",
            ]
            mode = "偏轻量关心：一句话即可，不要开启长流程，除非对方主动要。"

        return (
            "【可选显化关心——本次可以主动发起，但要很轻，不要每次都用：】\n"
            + mode
            + "\n可参考的开场，不要照抄：\n"
            + "\n".join(f"- {e}" for e in examples)
            + "\n边界：不要承诺结果，不要说频率不够，不要追问旧计划，不要要求对方立刻做完整流程。"
        )

    def _build_frequency_first_aid_hint(self, recent: list[dict], prev_proactive: list[str], memory_items: list[dict], now: datetime) -> str:
        if any("小魔女降频提醒" in msg or "30秒练习" in msg for msg in prev_proactive):
            return ""

        recent_user = [m["content"] for m in recent if m["role"] == "user"]
        candidate = "\n".join(recent_user[-3:])
        category = detect_frequency_support_category(candidate)

        if category is None:
            memory_text = "\n".join(
                m["content"] for m in memory_items
                if (m.get("memory_type") in ["manifestation", "emotional_pattern"])
            )
            category = detect_frequency_support_category(memory_text)

        if category is None:
            return ""

        if 23 <= now.hour or now.hour <= 7:
            tone = "深夜/清晨只发很轻的安抚；可以给一个30秒身体回到当下的小练习，但不要刺激用户复盘创伤。"
        else:
            tone = "可以分享一句显化短句，再给一个30秒身体回到当下的小练习。"

        item = pick_manifestation_quote(category, prev_proactive)
        return (
            "【可选频率急救包——适合用户被过去、焦虑、低频或旧故事拉住时使用：】\n"
            f"{tone}\n"
            f"可用短句：“{item.quote}”\n"
            f"配套练习：{item.practice}\n"
            "边界：不要说用户频率太低，不要要求立刻开心，不要承诺结果；目标只是从焦虑降到中性。"
        )

    def _build_proactive_prompt(self, user_id: int) -> str:
        now = datetime.now(BEIJING_TZ)

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

        # ── User memories — split by type/status ──
        memory_items = self.memory.get_key_memories_with_meta(user_id, limit=20)
        memory_block = ""
        if memory_items:
            stable = [m["content"] for m in memory_items if (m["memory_type"] or "fact") in ["fact", "preference", "boundary", "emotional_pattern"]]
            manifestation = [m["content"] for m in memory_items if m["memory_type"] == "manifestation"]
            ongoing = [m["content"] for m in memory_items if m["memory_type"] == "event" and (m["status"] or "active") == "ongoing"]
            completed = [m["content"] for m in memory_items if m["memory_type"] == "event" and (m["status"] or "active") == "completed"]
            expired = [m["content"] for m in memory_items if m["memory_type"] == "event" and (m["status"] or "active") == "expired"]

            parts = []
            if stable:
                parts.append("【长期事实/偏好/边界——可以作为聊天背景：】\n" + "\n".join(f"- {f}" for f in stable[:8]))
            if manifestation:
                parts.append("【显化系统记忆——只在自然相关时轻轻陪伴，不要强行提起：】\n" + "\n".join(f"- {m}" for m in manifestation[:4]))
            if ongoing:
                parts.append("【可能仍在进行的事——只有自然相关时才轻轻问一句：】\n" + "\n".join(f"- {e}" for e in ongoing[:5]))
            if completed:
                parts.append(f"【已经结束的事：有{len(completed)}条旧事已隐藏——不要追问，也不要主动提起。】")
            if expired:
                parts.append(f"【时间已经过期的旧计划：有{len(expired)}条旧计划已隐藏——绝对不要当成今天/近期计划追问。】")
            if parts:
                memory_block = "\n\n".join(parts)
                memory_block += "\n\n重要：不要把记忆当任务清单。长期偏好可用来贴近对方；显化记忆只在对方最近正在聊相关主题时轻轻使用；进行中的事少问；已完成和已过期的旧事已隐藏，不要主动提起或追问。"

        # ── Structured manifestation lifecycle ──
        wish_items = self.memory.get_manifestation_wishes(user_id, limit=5)
        evidence_items = self.memory.get_manifestation_evidence(user_id, limit=8)
        lifecycle_block = ""
        if wish_items:
            wish_lines = [f"- #{w['id']} {w['title']}｜状态：{w['status']}" for w in wish_items]
            evidence_lines = [f"- {e['content']}" for e in evidence_items[:5]]
            lifecycle_block = "【显化愿望生命周期——优先围绕 active，released/fulfilled/paused 不要当成还要追：】\n" + "\n".join(wish_lines)
            if evidence_lines:
                lifecycle_block += "\n\n【最近显化证据链：】\n" + "\n".join(evidence_lines)

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
        pressure_hint = ""
        if prev_proactive:
            proactive_block = (
                "【你最近主动找对方时发的消息——这次绝对不能重复的话题：】\n"
                + "\n".join(f"- 第{i+1}次: 「{msg[:150]}」" for i, msg in enumerate(prev_proactive))
                + "\n\n以上话题已经说过了，这次必须换全新的切入点。如果最近一次的主动消息对方没回，"
                + "说明对方可能不感兴趣，不要再延续那个话题。"
            )
            pressure_hint = (
                "对方最近没有回应你主动发起的话题时，本次要更轻、更短、更没有压力。"
                "可以不问问题，只留一句自然的小观察或问候。"
            )

        # ── Proactive manifestation check-in ──
        manifestation_hint = self._build_manifestation_checkin_hint(manifestation if memory_items else [], recent, prev_proactive, now)

        # ── Proactive frequency first-aid ──
        frequency_hint = self._build_frequency_first_aid_hint(recent, prev_proactive, memory_items if memory_items else [], now)

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
            last_dt = datetime.fromisoformat(last_active).replace(tzinfo=timezone.utc)
            gap_min = (now - last_dt.astimezone(BEIJING_TZ)).total_seconds() / 60
            if gap_min >= 120:
                gap_hint = f"对方已经{int(gap_min / 60)}小时多没和你说话了，可能是在忙。语气温和，不要催促，也不要暗示他应该回复。"
            elif gap_min >= 60:
                gap_hint = "对方有一阵子没和你说话了。简单问候就好，不要给对方压力，也可以不问问题。"

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
            "- 1-3句话即可，不要长篇大论；有时一句话比三句话更像真人",
            "- 不要问太多问题，至多问一个，也可以完全不问",
            "- 不要总问'今天怎么样''在干嘛'，连续这样会显得像定时任务",
            "- 不要用'好久不见''好几天没聊了'这类话，除非给出的数据里确实很久没聊",
            "- 绝对不要以'作为XX'开头介绍自己是谁",
            "",
            "【核心规则——务必遵守：】",
            "- 不要重复你最近主动发过的任何话题（见下方'你最近主动发了什么'）",
            "- 不要追问已经完结或时间已经过期的事件（见下方'已发生的事件'和'时间可能已经过期的旧计划'）",
            "- 不要每次都从记忆里翻旧事；只有自然相关时才轻轻提起",
            "- 如果找不到合适的新话题，就分享一个自己的小观察或当下的感受，而不是翻旧话题",
            "",
            summary_block,
            "",
            memory_block,
            "",
            lifecycle_block,
            "",
            recent_block,
            "",
            proactive_block,
            "",
            pressure_hint,
            "",
            manifestation_hint,
            "",
            frequency_hint,
            "",
            kb_block,
            "",
            gap_hint,
        ])).strip()
