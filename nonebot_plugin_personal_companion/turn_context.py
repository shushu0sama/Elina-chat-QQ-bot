from dataclasses import dataclass
from typing import Protocol


class FlowInviteChecker(Protocol):
    def should_invite_process(self, user_msg: str) -> bool: ...
    def should_invite_appreciation(self, user_msg: str) -> bool: ...


EMOTION_KEYWORDS: dict[str, list[str]] = {
    "焦虑": ["焦虑", "压力", "担心", "害怕", "紧张", "不安", "慌", "忧心", "撑不住"],
    "难过": ["难过", "伤心", "痛苦", "崩溃", "绝望", "想哭", "难受", "心酸", "委屈", "心碎"],
    "烦躁": ["烦躁", "生气", "愤怒", "烦", "火大", "不爽", "讨厌", "气死", "烦人", "恼火"],
    "开心": ["开心", "高兴", "兴奋", "快乐", "幸福", "太棒", "顺利", "幸运", "欣喜"],
    "迷茫": ["迷茫", "困惑", "搞不懂", "看不清", "没方向", "不知所措", "纠结"],
    "疲惫": ["累", "疲惫", "困", "乏", "没精神", "不想动", "倦", "没力气"],
}

INTENSE_WORDS = ["崩溃", "受不了", "绝望", "撑不住", "扛不住", "死了", "完蛋"]
FACTUAL_MARKERS = ["怎么", "如何", "为什么", "是什么", "帮我", "教我", "解释", "代码", "报错", "查一下"]
SEARCH_MARKERS = ["查一下", "搜一下", "搜索", "最近", "现在", "新闻", "价格", "版本", "最新", "实时", "今天"]
ENDING_MARKERS = ["睡了", "先这样", "不聊了", "忙了", "回头说", "算了", "晚安"]
SHORT_ACKS = {"嗯", "嗯嗯", "好", "好吧", "哦", "行", "可以", "收到", "懂了", "哈哈", "哈哈哈"}
CELEBRATION_MARKERS = ["好消息", "拿到", "过了", "中了", "成功", "太棒", "终于"]


@dataclass(frozen=True)
class TurnContext:
    emotions: list[str]
    intent: str
    intensity: str
    reply_length: str
    allow_question: bool
    allow_web_search: bool
    flow_invite: str
    should_end_softly: bool
    recent_questions: int
    recently_invited_flow: bool


def detect_emotions(text: str) -> list[str]:
    emotions: list[str] = []
    for emotion, keywords in EMOTION_KEYWORDS.items():
        if any(kw in text for kw in keywords):
            emotions.append(emotion)
    return emotions if emotions else ["中性"]


def analyze_turn(
    user_msg: str,
    recent_messages: list[dict],
    flow_manager: FlowInviteChecker | None = None,
) -> TurnContext:
    text = user_msg.strip()
    emotions = detect_emotions(text)
    assistant_recent = [m["content"] for m in recent_messages[-5:] if m.get("role") == "assistant"]
    recent_questions = sum(1 for m in assistant_recent if m.rstrip().endswith(("?", "？")))
    recently_invited_flow = any("流程" in m for m in assistant_recent)

    short_ack = text in SHORT_ACKS or len(text) <= 3
    ending = any(w in text for w in ENDING_MARKERS)
    factual = any(w in text for w in FACTUAL_MARKERS)
    intense = any(w in text for w in INTENSE_WORDS)
    distressed = any(e in emotions for e in ["焦虑", "难过", "烦躁", "疲惫"]) or intense
    celebrating = "开心" in emotions or any(w in text for w in CELEBRATION_MARKERS)

    if ending:
        intent = "ending"
    elif short_ack:
        intent = "short_ack"
    elif distressed and not factual:
        intent = "venting"
    elif celebrating:
        intent = "celebrating"
    elif factual:
        intent = "factual"
    else:
        intent = "casual"

    if intense:
        intensity = "high"
    elif distressed or celebrating:
        intensity = "medium"
    else:
        intensity = "low"

    if intent in ["short_ack", "ending"]:
        reply_length = "short"
    elif intent == "factual":
        reply_length = "detailed"
    elif intent == "venting":
        reply_length = "supportive"
    elif len(text) > 120:
        reply_length = "normal_long"
    else:
        reply_length = "normal"

    allow_question = recent_questions < 2 and intent not in ["ending", "short_ack"]
    allow_web_search = intent == "factual" or any(w in text for w in SEARCH_MARKERS)

    flow_invite = "none"
    if flow_manager and not recently_invited_flow and not factual and not ending:
        if flow_manager.should_invite_process(text):
            flow_invite = "process"
        elif flow_manager.should_invite_appreciation(text):
            flow_invite = "appreciation"

    return TurnContext(
        emotions=emotions,
        intent=intent,
        intensity=intensity,
        reply_length=reply_length,
        allow_question=allow_question,
        allow_web_search=allow_web_search,
        flow_invite=flow_invite,
        should_end_softly=ending,
        recent_questions=recent_questions,
        recently_invited_flow=recently_invited_flow,
    )


def build_companion_context_prompt(ctx: TurnContext) -> str:
    lines = ["[本轮陪伴方式：]"]
    lines.append(f"- 对方当前情绪：{'、'.join(ctx.emotions)}。")

    if ctx.intent == "ending":
        lines.append("- 对方像是在收尾或需要离开，顺着他的节奏，不要追问或强行延长聊天。")
    elif ctx.intent == "short_ack":
        lines.append("- 对方只给了很短的回应，本轮可以只接一句、附和一下，别硬展开。")
    elif ctx.intent == "venting":
        lines.append("- 对方现在更需要被接住情绪，先像朋友一样站在他这边，不要马上分析、教育或列方案。")
        lines.append("- 回复以2-4句为宜，温柔但别肉麻，可以承认事情确实难。")
    elif ctx.intent == "celebrating":
        lines.append("- 对方在分享好事，先真诚替他开心，别急着分析或转成建议。")
        lines.append("- 回复可以轻快一点，像朋友一起庆祝。")
    elif ctx.intent == "factual":
        lines.append("- 对方在寻求解释或帮助，先直接回答问题；需要时可以稍微详细，但不要装作闲聊绕圈。")
    else:
        lines.append("- 这是普通聊天，保持自然承接，不要把每句话都处理成任务。")

    if ctx.allow_question:
        lines.append("- 如果要问问题，最多问一个；也可以完全不问。")
    else:
        lines.append("- 你最近已经连续问过问题或当前不适合追问，这轮尽量不要再追问，除非对方明确需要你问。")

    if ctx.flow_invite == "process":
        lines.append("- 如果气氛自然，可以只用一句很轻的话问要不要一起走个流程；不要强推。")
    elif ctx.flow_invite == "appreciation":
        lines.append("- 如果气氛自然，可以陪他停一下、感受这件好事；不要把庆祝变成说教。")
    elif ctx.recently_invited_flow:
        lines.append("- 最近已经提过流程了，这轮不要重复邀请。")

    lines.append("- 避免固定的'总结 + 建议 + 提问'套路，像真人朋友一样按当下反应。")
    return "\n".join(lines)


def choose_reply_max_tokens(ctx: TurnContext) -> int:
    return 2048
