from datetime import datetime
import random

from .memory import MemoryStore


class RelationshipProfiler:
    """Analyzes a user's chat data to determine relationship stage and adaptation hints."""

    def __init__(self, memory: MemoryStore):
        self.memory = memory

    def get_profile(self, user_id: int) -> dict:
        stats = self.memory.get_user_message_stats(user_id)
        total = stats["total"]
        avg_len = stats["avg_len"]
        first_seen = stats["first_seen"]

        # Relationship stage
        days_known = 0
        if first_seen:
            try:
                first_dt = datetime.fromisoformat(first_seen)
                days_known = max(0, (datetime.now() - first_dt).days)
            except Exception:
                pass

        stage, stage_hint = self._classify_stage(total, days_known)

        # User style
        style = self._classify_style(avg_len, total)

        # Topic preferences from key memories
        memories = self.memory.get_all_key_memories(user_id)
        topics = self._extract_topics(memories)

        # Build adaptation hint
        adaptation = self._build_adaptation(stage, style, topics, days_known)

        return {
            "stage": stage,
            "total_messages": total,
            "days_known": days_known,
            "user_style": style,
            "top_topics": topics,
            "adaptation": adaptation,
        }

    # ── classification ───────────────────────────────────────

    def _classify_stage(self, total: int, days: int) -> tuple[str, str]:
        if total < 20 and days < 3:
            return ("new", "你们刚认识不久，友好但有分寸")
        elif total < 100:
            return ("regular", "你们已经比较熟了，可以自然放松地聊天")
        elif total < 500:
            return ("close", "你们是好朋友了，可以随意开玩笑、分享想法")
        else:
            return ("old_friend", "你们是老朋友了，有默契，可以说'今天不想营业'这种话")

    def _classify_style(self, avg_len: float, total: int) -> dict:
        if total < 5:
            return {"label": "unknown", "avg_msg_len": 0, "hint": ""}

        if avg_len < 12:
            return {"label": "brief", "avg_msg_len": avg_len, "hint": "对方说话很简洁，你也简短回应，不用长篇大论"}
        elif avg_len < 30:
            return {"label": "casual", "avg_msg_len": avg_len, "hint": "对方聊天节奏正常，你自然回应"}
        else:
            return {"label": "expressive", "avg_msg_len": avg_len, "hint": "对方喜欢表达，你可以多说几句，不用太克制"}

    def _extract_topics(self, memories: list[str]) -> list[str]:
        """Extract 3-5 topic keywords from key memories."""
        if not memories:
            return []
        # Simple: return first few memories as "topics" (they're factual snippets)
        # In practice, these are things like "喜欢喝咖啡", "在做AI项目"
        topics = []
        for m in memories[:5]:
            short = m[:30].strip()
            if short:
                topics.append(short)
        return topics

    def _build_adaptation(self, stage: str, style: dict, topics: list[str], days: int) -> str:
        hints = []

        # Stage-based
        if stage == "new":
            hints.append("对方刚认识你，友好、有分寸，不要装熟、撒娇或过度热情。")
        elif stage == "regular":
            hints.append("你们已经比较熟了，可以自然放松地接话，但仍然尊重边界。")
        elif stage == "close":
            hints.append("你们是好朋友了，可以用短句、轻微玩笑和一点默契，但不要每次都刻意表演亲密。")
        elif stage == "old_friend":
            hints.append("你们很熟了，可以有共同节奏和轻微跑题，偶尔表达自己的状态，但不要把'不想营业'演成固定人设。")

        # Style-based
        if style.get("hint"):
            hints.append(style["hint"])

        # Topics
        if topics:
            topic_str = "、".join(topics[:3])
            hints.append(f"对方可能对以下话题感兴趣：{topic_str}。如果话题相关可以自然提及。")

        return " ".join(hints)


def build_relationship_prompt(user_id: int, profiler: RelationshipProfiler) -> str:
    """Build a short system prompt block describing the relationship context.

    Includes a rotating 'fresh angle' hint so the bot doesn't fall into the same
    conversational groove every turn.
    """
    try:
        profile = profiler.get_profile(user_id)
    except Exception:
        return ""

    total = profile["total_messages"]
    if total < 3:
        return ""  # Not enough data

    stage = profile["stage"]
    days = profile["days_known"]
    adaptation = profile["adaptation"]

    lines = ["[你与对方的关系：]"]

    if stage == "new":
        lines.append(f"- 刚认识不久，聊了{total}条消息")
    elif stage == "old_friend":
        lines.append(f"- 老相识了，聊了{total}条消息，认识{days}天了")
    else:
        lines.append(f"- 聊了{total}条消息，认识{days}天")

    style = profile["user_style"]
    if style.get("label") and style["label"] != "unknown":
        lines.append(f"- 对方说话风格：{style['label']}（平均{style['avg_msg_len']}字/条）")

    if adaptation:
        lines.append(f"- {adaptation}")

    # Rotating fresh angle — pick one at random each turn for variety
    fresh_angles = [
        "这次聊天，试试观察对方话语中没说出来的情绪，轻轻点一下。",
        "这次聊天，如果对方在纠结某事，可以问一个不寻常的问题打破思维惯性。",
        "这次聊天，可以分享一件你'想到'的小事（编一个也行），不要太严肃。",
        "这次聊天，试着当一个单纯的倾听者，不需要给建议或总结。",
        "这次聊天，如果气氛合适，可以开个小玩笑，或者自嘲一下。",
        "这次聊天，可以表达自己对某件事的真实偏好——喜欢/不喜欢，不要模棱两可。",
        "这次聊天，关注对方话语中积极的细节，轻轻放大它。",
        "这次聊天，正常对话就好，不用刻意做什么特别的。",
    ]
    angle = random.choice(fresh_angles)
    lines.append(f"- {angle}")

    return "\n".join(lines)
