import yaml
import random
from pathlib import Path
from datetime import datetime

PROMPTS_DIR = Path(__file__).parent / "prompts"
DEFAULT_PROMPT_PATH = PROMPTS_DIR / "default.yaml"

# Per-conversation state cache: {user_id: state_name}
# Reset each time build_system_prompt is called for a new conversation turn.
# A new state is rolled probabilistically: ~30% chance to re-roll each call.
_state_cache: dict[int, str] = {}


def _load_yaml(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _time_context(now: datetime) -> str:
    """Return a precise time-of-day context so the bot is aware of the current time."""
    hour = now.hour
    minute = now.minute
    time_str = f"{hour:02d}:{minute:02d}"
    weekday = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"][now.weekday()]

    if 23 <= hour or hour <= 4:
        period = "深夜"
        vibe = "大部分人已经睡了。如果对方还在线，可能是在熬夜。语气温柔一点，可以关心但不唠叨。对方提到睡觉相关的事，回复要符合深夜语境。"
    elif 5 <= hour <= 7:
        period = "清晨"
        vibe = "天刚亮不久。对方可能是早起的人。"
    elif 8 <= hour <= 10:
        period = "上午"
        vibe = "一天的开始。对方可能在准备工作或已经开始忙碌。"
    elif 11 <= hour <= 12:
        period = "午前"
        vibe = "快到午饭时间了。对方可能有点饿了或者准备休息。"
    elif 13 <= hour <= 14:
        period = "午后"
        vibe = "刚过午饭，人容易犯困。语气可以轻松随意一点。"
    elif 15 <= hour <= 17:
        period = "下午"
        vibe = "下午工作时间。对方可能在忙，或者快下班了。"
    elif 18 <= hour <= 19:
        period = "傍晚"
        vibe = "晚饭时间。对方可能刚下班/放学，在吃饭或者准备晚上的安排。注意：现在是傍晚不是睡觉时间，不要对睡觉相关话题说晚安。"
    elif 20 <= hour <= 22:
        period = "晚上"
        vibe = "晚上的休息时间。对方可能在放松、看东西、或者做自己的事。离睡觉还有一段时间。"
    else:
        period = "深夜前夕"
        vibe = "时间不早了但还不是深夜。对方可能在准备休息了。"

    return (
        f"现在是北京时间 {time_str}，{weekday}{period}。{vibe}\n"
        f"重要：你的所有回复必须与当前时间吻合。不要在不合理的时间说早安/晚安/午安。"
    )


def _roll_state(cfg: dict, user_id: int) -> dict:
    """Pick a random state, weighted. ~30% chance to re-roll from cache."""
    states = cfg.get("states", [])
    if not states:
        return {}

    # Re-roll with ~30% probability to create natural variation
    if user_id in _state_cache and random.random() > 0.3:
        name = _state_cache[user_id]
        for s in states:
            if s["name"] == name:
                return s

    # Weighted random selection
    total = sum(s.get("weight", 10) for s in states)
    r = random.randint(1, total)
    cumulative = 0
    for s in states:
        cumulative += s.get("weight", 10)
        if r <= cumulative:
            _state_cache[user_id] = s["name"]
            return s

    return states[0]


def build_system_prompt(persona_path: Path | None = None, user_id: int = 0) -> str:
    """Build the full system prompt with personality, state, and time context."""
    path = persona_path or DEFAULT_PROMPT_PATH
    cfg = _load_yaml(path)
    voice = cfg["voice"]

    # Roll state (mood)
    state = _roll_state(cfg, user_id)

    # Current time (prominent — bot must be time-aware)
    now = datetime.now()
    time_ctx = _time_context(now)

    lines = [
        time_ctx,
        "",
        f"你的名字是{voice['name']}。",
        voice["style"],
        "",
    ]

    # Core traits (just a few)
    if voice.get("traits"):
        lines.append("记住：")
        for trait in voice["traits"]:
            lines.append(f"- {trait}")
        lines.append("")

    # Current state (mood)
    if state:
        state_hints = state.get("hints", [])
        if state_hints:
            state_name = state["name"]
            lines.append(f"你现在处于「{state_name}」的状态。")
            for hint in state_hints:
                lines.append(hint)
            lines.append("")

    # Reply variety
    variety = voice.get("reply_variety", [])
    if variety:
        # Only include a random subset (2-3) each time for variation
        hints = random.sample(variety, min(3, len(variety)))
        for h in hints:
            lines.append(h)
        lines.append("")

    # Philosophy (lightweight)
    philosophy = voice.get("philosophy", [])
    if philosophy:
        lines.append("你的内在视角（偶尔自然会流露，不要刻意输出）：")
        for p in philosophy:
            lines.append(f"- {p}")
        lines.append("")

    # Forbidden (minimum)
    forbidden = voice.get("forbidden", [])
    if forbidden:
        lines.append("绝对不要：")
        for fb in forbidden:
            lines.append(f"- {fb}")
        lines.append("")

    return "\n".join(lines)
