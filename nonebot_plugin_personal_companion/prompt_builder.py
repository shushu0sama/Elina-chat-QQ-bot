import random
from collections import Counter
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

from .knowledge import (
    KnowledgeBase,
    build_knowledge_prompt_personalized,
    build_manifestation_knowledge_prompt,
    is_manifestation_intent,
)
from .manifestation_quotes import build_frequency_first_aid_text
from .memory import MemoryStore
from .personality import build_system_prompt
from .relationship import RelationshipProfiler, build_relationship_prompt
from .turn_context import (
    FlowInviteChecker,
    analyze_turn,
    build_companion_context_prompt,
    build_reply_mode_prompt,
)


BEIJING_TZ = ZoneInfo("Asia/Shanghai")


def build_time_lock(now: datetime) -> str:
    weekday_names = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return (
        f"【当前真实时间校验——你只能以以下时间为准，放在最后确保你不会被旧数据干扰：】\n"
        f"当前真实时间是北京时间 {now.strftime('%Y年%m月%d日')}（{now.strftime('%Y-%m-%d')}）"
        f"{now.hour}:{now.minute:02d}，{weekday_names[now.weekday()]}。\n"
        f"如果用户问你时间、日期、今天几号、星期几，只认这个时间。\n"
        f"历史消息、记忆、摘要、日记、知识库里的日期都不是今天，不能当成现在。\n"
        f"【注意：以上时间仅供你内部参考。你的回复中绝对不能出现 '北京时间'、日期、时间等文字。】"
    )


def history_time_label(msg: dict) -> str:
    if msg.get("time_display"):
        return str(msg["time_display"])
    raw = msg.get("time") or msg.get("created_at")
    if not raw:
        return "时间未知"
    text = str(raw).replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError:
        try:
            dt = datetime.strptime(str(raw), "%Y-%m-%d %H:%M:%S")
        except ValueError:
            return str(raw)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(BEIJING_TZ).strftime("%Y-%m-%d %H:%M")


def format_history_message(msg: dict) -> dict:
    return {"role": msg["role"], "content": msg["content"]}


def format_timeline_entries(entries: list[dict]) -> str:
    if not entries:
        return ""
    lines = [
        "[时间线记忆——以下是过去某天发生/计划过的事，日期是事件发生日期，不代表今天。只有和本轮话题自然相关时使用；不要把旧日期当成今天，不要主动追问旧事。]"
    ]
    for entry in entries:
        event_time = f" {entry['event_time']}" if entry.get("event_time") else ""
        lines.append(f"- {entry['event_date']}{event_time}：{entry['content']}")
    return "\n".join(lines)


def build_anti_repeat(recent_messages: list[dict]) -> str:
    bot_replies = [m["content"] for m in recent_messages if m["role"] == "assistant"]
    if len(bot_replies) < 2:
        return ""

    recent_replies = bot_replies[-5:]
    lines = ["【重要：你最近几轮说过这些话，本次回复必须有所不同——不要重复相同的意思、相同的句式、相同的口头禅：】"]
    for i, reply in enumerate(recent_replies, 1):
        snippet = reply[:80].replace("\n", " ")
        lines.append(f"  {i}. {snippet}...")
    lines.append("请换个角度、换个语气、换个切入点。如果实在想不出不同的回应，就干脆换个话题。")
    return "\n".join(lines)


def build_thread_context(recent_messages: list[dict], current_msg: str, extract_keywords) -> str:
    if len(recent_messages) < 2:
        return ""

    recent_exchanges = recent_messages[-8:]
    user_msgs = [m["content"] for m in recent_exchanges if m["role"] == "user"]

    if not user_msgs:
        return ""

    all_keywords: list[str] = []
    for msg in user_msgs[-3:]:
        all_keywords.extend(extract_keywords(msg, top_n=3))

    kw_counts = Counter(all_keywords)
    recurring = [kw for kw, cnt in kw_counts.items() if cnt >= 2]

    lines = ["[当前对话线索——请在回复时保持话题连贯：]"]
    if recurring:
        lines.append(f"- 对方最近反复提到的词：{'、'.join(recurring[:3])}")

    if len(recent_exchanges) >= 2:
        last = recent_exchanges[-2:]
        summary_parts = []
        for m in last:
            role = "对方" if m["role"] == "user" else "你"
            snippet = m["content"][:60].replace("\n", " ")
            summary_parts.append(f"{role}：{snippet}")
        lines.append(f"- 上一轮：{' | '.join(summary_parts)}")

    if lines:
        lines.append("- 现在对方说：" + current_msg[:100])
        lines.append("- 请确保你的回应承接上一轮的话题。如果对方用了代词（'那个''它''这样'），要根据上下文理解具体指什么。")

    return "\n".join(lines)


class PromptBuilder:
    def __init__(
        self,
        memory: MemoryStore,
        config,
        knowledge_base: KnowledgeBase | None = None,
        manifestation_knowledge_base: KnowledgeBase | None = None,
        relationship_profiler: RelationshipProfiler | None = None,
        flow_manager: FlowInviteChecker | None = None,
        extract_keywords=None,
        now_provider=None,
    ):
        self.memory = memory
        self.config = config
        self.knowledge_base = knowledge_base
        self.manifestation_knowledge_base = manifestation_knowledge_base
        self.relationship_profiler = relationship_profiler
        self.flow_manager = flow_manager
        self.extract_keywords = extract_keywords
        self.now_provider = now_provider

    def build_messages(
        self,
        user_msg: str,
        retrieved_memories: list[str],
        user_id: int = 0,
        associated_memories: list[str] | None = None,
        turn_ctx=None,
        timeline_entries: list[dict] | None = None,
    ) -> list[dict]:
        messages: list[dict] = []

        if retrieved_memories:
            mems = retrieved_memories.copy()
            random.shuffle(mems)
            mems = mems[:3]
            memory_block = "[你记得以下关于用户的当前有效信息。只能在和用户本轮话题自然相关时轻轻使用；不要为了展示记忆而主动翻旧账：]\n"
            for mem in mems:
                memory_block += f"- {mem}\n"
            messages.append({"role": "system", "content": memory_block})

        if associated_memories:
            assoc = associated_memories[:2]
            assoc_block = "[你联想到了以下仍然有效的相关信息——只有话题自然关联到时才可以轻轻提起，绝对不要追问旧事：]\n"
            for mem in assoc:
                assoc_block += f"- {mem}\n"
            messages.append({"role": "system", "content": assoc_block})

        if self.knowledge_base:
            kb_prompt = build_knowledge_prompt_personalized(
                user_msg, self.knowledge_base, user_id, self.memory
            )
            if kb_prompt:
                messages.append({"role": "system", "content": kb_prompt})

        if self.manifestation_knowledge_base and is_manifestation_intent(user_msg):
            manifestation_prompt = build_manifestation_knowledge_prompt(
                user_msg, self.manifestation_knowledge_base, user_id, self.memory
            )
            if manifestation_prompt:
                messages.append({"role": "system", "content": manifestation_prompt})

        if self.relationship_profiler:
            rel_prompt = build_relationship_prompt(user_id, self.relationship_profiler)
            if rel_prompt:
                messages.append({"role": "system", "content": rel_prompt})

        recent = self.memory.get_recent_messages(self.config.max_recent_messages, user_id)

        turn_ctx = turn_ctx or analyze_turn(user_msg, recent, self.flow_manager)
        reply_mode_prompt = build_reply_mode_prompt(turn_ctx)
        if reply_mode_prompt:
            messages.append({"role": "system", "content": reply_mode_prompt})
        companion_ctx = build_companion_context_prompt(turn_ctx)
        if companion_ctx:
            messages.append({"role": "system", "content": companion_ctx})

        manifestation_memories = self.memory.get_manifestation_memories(user_id, limit=5)
        if manifestation_memories:
            manifest_block = (
                "[艾琳娜显化系统记忆——用于愿望澄清、信念改写、显化日记和执念降频：]\n"
                + "\n".join(f"- {m}" for m in manifestation_memories)
                + "\n使用规则：只在用户聊到显化、愿望、信念、执念、未来自我时自然使用；不要主动保证结果，不要把显化变成压力。"
            )
            messages.append({"role": "system", "content": manifest_block})

        recent_proactive = self.memory.get_recent_proactive_content(user_id, limit=3)
        first_aid = build_frequency_first_aid_text(user_msg, recent_proactive)
        if first_aid:
            messages.append({"role": "system", "content": (
                "[当前用户可能被过去、焦虑、低频或旧故事拉住。请优先温柔降频，再回应具体内容。]\n"
                + first_aid
                + "\n使用边界：不要说用户频率太低，不要要求立刻开心，不要承诺结果；目标只是从焦虑降到中性。"
            )})

        timeline_block = format_timeline_entries(timeline_entries or [])
        if timeline_block:
            messages.append({"role": "system", "content": timeline_block})

        if self.extract_keywords:
            thread_ctx = build_thread_context(recent, user_msg, self.extract_keywords)
            if thread_ctx:
                messages.append({"role": "system", "content": thread_ctx})

        anti_repeat = build_anti_repeat(recent)
        if anti_repeat:
            messages.append({"role": "system", "content": anti_repeat})

        personality_prompt = build_system_prompt(user_id=user_id, turn_context=turn_ctx)
        messages.append({"role": "system", "content": personality_prompt})

        if recent:
            messages.append({"role": "system", "content": "[最近聊天记录只作为历史上下文，请把它们当作过去发生的对话，不要把旧日期当成今天。]"})
        for msg in recent:
            messages.append(format_history_message(msg))

        now = self.now_provider(BEIJING_TZ) if self.now_provider else datetime.now(BEIJING_TZ)
        messages.append({"role": "system", "content": build_time_lock(now)})

        if not (recent and recent[-1]["role"] == "user" and recent[-1]["content"] == user_msg):
            messages.append({"role": "user", "content": user_msg})

        return messages
