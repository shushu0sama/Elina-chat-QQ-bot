from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class ManifestationQuote:
    category: str
    quote: str
    practice: str


PAST_RELEASE_TRIGGERS = ["过去", "以前", "从前", "旧事", "旧故事", "又想起", "放不下", "后悔"]
ANXIETY_TRIGGERS = ["焦虑", "慌", "迷茫", "害怕", "不安", "崩溃", "内耗", "频率", "低频"]
DETACHMENT_TRIGGERS = ["执念", "为什么还没", "显化失败", "反复确认", "控制", "等不到"]
SELF_CONCEPT_TRIGGERS = ["我不配", "不值得", "没人选", "不够好", "总是失败", "没办法拥有"]

QUOTES: list[ManifestationQuote] = [
    ManifestationQuote(
        category="past_release",
        quote="不要用过去的证据，审判未来的可能性。",
        practice="把手放在胸口，呼吸三次，然后只问：今天的我能不能先不继续扮演旧故事？",
    ),
    ManifestationQuote(
        category="past_release",
        quote="过去发生过，但它不是你今天必须继续携带的频率。",
        practice="对自己说：那是旧体验，不是我此刻唯一的身份。然后去喝一口水。",
    ),
    ManifestationQuote(
        category="anxiety_downshift",
        quote="不用从焦虑直接跳到快乐，先从焦虑走到中性就已经是在升频。",
        practice="看见身边三个物体，慢慢说出它们的名字，让注意力先回到现在。",
    ),
    ManifestationQuote(
        category="anxiety_downshift",
        quote="更好的念头不一定要很亮，只要比刚才轻一点。",
        practice="把“完了”改成“我现在还在这里，我可以先做下一件小事”。",
    ),
    ManifestationQuote(
        category="detachment",
        quote="放下不是放弃愿望，是停止用焦虑喂养旧故事。",
        practice="接下来30分钟不检查结果，只做一件能照顾自己的小事。",
    ),
    ManifestationQuote(
        category="detachment",
        quote="你不需要追逐那个属于你的东西，你需要先停止追逐感本身。",
        practice="把想确认的冲动写下来，但先不行动，等身体松一点再决定。",
    ),
    ManifestationQuote(
        category="self_concept",
        quote="新结果通常先从新自我概念开始，而不是从外界立刻改变态度开始。",
        practice="问自己：如果我是值得被认真对待的人，我现在会怎么对待自己？",
    ),
    ManifestationQuote(
        category="wish_fulfilled",
        quote="活在愿望已成真，不是骗自己结果来了，而是先练习那个已经稳定的你。",
        practice="用那个稳定版本的你，做一个很小的选择：慢一点、稳一点、少追一点。",
    ),
]


def detect_frequency_support_category(text: str) -> str | None:
    if any(trigger in text for trigger in SELF_CONCEPT_TRIGGERS):
        return "self_concept"
    if any(trigger in text for trigger in DETACHMENT_TRIGGERS):
        return "detachment"
    if any(trigger in text for trigger in PAST_RELEASE_TRIGGERS):
        return "past_release"
    if any(trigger in text for trigger in ANXIETY_TRIGGERS):
        return "anxiety_downshift"
    return None


def pick_manifestation_quote(category: str | None = None, recent_texts: list[str] | None = None) -> ManifestationQuote:
    pool = [q for q in QUOTES if q.category == category] if category else QUOTES
    if not pool:
        pool = QUOTES
    recent = "\n".join(recent_texts or [])
    fresh = [q for q in pool if q.quote not in recent]
    return random.choice(fresh or pool)


def build_frequency_first_aid_text(text: str, recent_texts: list[str] | None = None) -> str | None:
    category = detect_frequency_support_category(text)
    if category is None:
        return None
    item = pick_manifestation_quote(category, recent_texts)
    return (
        "今天的小魔女降频提醒：\n\n"
        f"“{item.quote}”\n\n"
        f"30秒练习：{item.practice}\n\n"
        "先不用急着变高频。能从焦虑走到中性，就已经是在回到自己。"
    )
