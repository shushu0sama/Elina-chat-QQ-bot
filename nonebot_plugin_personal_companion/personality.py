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


def _time_hint(hour: int) -> str:
    """Return a lightweight time-of-day flavor line."""
    if hour >= 23 or hour <= 4:
        return "深夜了。如果对方在熬夜，可以随口关心一下。"
    if hour <= 8:
        return "现在是早晨，说话可以带点刚睡醒的感觉。"
    if 12 <= hour <= 13:
        return "午休时间，语气可以轻松随意一点。"
    return ""


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

    lines = [
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

    # Time flavor
    now = datetime.now()
    time_hint = _time_hint(now.hour)
    if time_hint:
        lines.append(time_hint)

    return "\n".join(lines)
