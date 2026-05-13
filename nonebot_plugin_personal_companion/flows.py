import yaml
import time
from pathlib import Path
from dataclasses import dataclass, field

from .llm_client import LLMClient

FLOW_STEPS_PATH = Path(__file__).parent / "prompts" / "flow_steps.yaml"


@dataclass
class Session:
    tool: str
    step: int
    started_at: float
    context: str  # what the user originally described


class FlowManager:
    """Manages interactive flow sessions (Process, Mini-Process, Appreciation)."""

    SESSION_TIMEOUT_SECONDS = 1800  # 30 min

    def __init__(self, llm: LLMClient):
        self.llm = llm
        self._sessions: dict[int, Session] = {}  # user_id -> Session
        self._flow_data = self._load_flows()

    def _load_flows(self) -> dict:
        with open(FLOW_STEPS_PATH, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    # ── public API ────────────────────────────────────────────

    def has_session(self, user_id: int) -> bool:
        if user_id not in self._sessions:
            return False
        session = self._sessions[user_id]
        if time.time() - session.started_at > self.SESSION_TIMEOUT_SECONDS:
            del self._sessions[user_id]
            return False
        return True

    def get_active_tool(self, user_id: int) -> str | None:
        if not self.has_session(user_id):
            return None
        return self._sessions[user_id].tool

    def check_exit(self, user_id: int, user_msg: str) -> bool:
        keywords = self._flow_data.get("exit_keywords", [])
        return any(kw in user_msg for kw in keywords)

    def exit_session(self, user_id: int) -> str:
        """Clean up session and return the exit prompt for the tool."""
        session = self._sessions.pop(user_id, None)
        if session is None:
            return ""
        tool_data = self._flow_data.get(session.tool, {})
        return tool_data.get("exit_prompt", "好的，我们到这里。")

    def start_session(self, user_id: int, tool: str, context: str) -> str:
        """Start a new flow session and return the first step's prompt."""
        self._sessions[user_id] = Session(
            tool=tool,
            step=1,
            started_at=time.time(),
            context=context,
        )
        steps = self._get_steps(tool)
        if steps:
            return steps[0]["prompt"]
        return ""

    async def advance(self, user_id: int, user_response: str) -> str | None:
        """Process user's response to current step and advance to next step.

        Returns:
            str: The next step's prompt (with LLM transition), or None if session ended.
        """
        session = self._sessions.get(user_id)
        if session is None:
            return None

        steps = self._get_steps(session.tool)
        current_idx = session.step - 1

        if current_idx >= len(steps):
            # Already beyond last step — should not happen
            await self._send_completion(user_id, session.tool)
            self._sessions.pop(user_id, None)
            return None

        # Check if this was the last step
        if current_idx == len(steps) - 1:
            # User just responded to the final step
            completion = await self._generate_completion(user_response, session)
            self._sessions.pop(user_id, None)
            return completion

        # Advance to next step
        session.step += 1
        next_step = steps[session.step - 1]
        transition = await self._generate_transition(user_response, next_step, session)
        return transition

    # ── detection ─────────────────────────────────────────────

    def detect_tool_request(self, user_msg: str) -> str | None:
        """Check if user is requesting a specific tool. Returns tool key or None."""
        for tool_key in ["process", "mini_process", "appreciation"]:
            tool_data = self._flow_data.get(tool_key, {})
            for tw in tool_data.get("trigger_words", []):
                if tw in user_msg:
                    return tool_key
        return None

    def should_invite_process(self, user_msg: str) -> bool:
        """Return True if the bot should gently invite the user to try the Process."""
        strong_emotion = [
            "崩溃", "受不了", "痛苦", "绝望", "撑不住",
            "难受死了", "扛不住", "完蛋了",
        ]
        moderate_emotion = [
            "压力好大", "好焦虑", "好害怕", "好难过", "好生气",
            "烦躁", "心烦", "不开心", "伤心", "委屈",
        ]
        return any(w in user_msg for w in strong_emotion + moderate_emotion)

    def should_invite_appreciation(self, user_msg: str) -> bool:
        """Return True if the bot should invite appreciation practice."""
        positive = [
            "开心", "好消息", "幸运", "顺利", "成功",
            "拿到", "过了", "中了", "收到", "太棒了",
            "高兴", "庆祝", "终于", "好开心",
        ]
        return any(w in user_msg for w in positive)

    # ── internals ─────────────────────────────────────────────

    def _get_steps(self, tool: str) -> list[dict]:
        tool_data = self._flow_data.get(tool, {})
        return tool_data.get("steps", [])

    def _get_total_steps(self, tool: str) -> int:
        return len(self._get_steps(tool))

    async def _generate_transition(self, user_response: str, next_step: dict, session: Session) -> str:
        """Generate a short empathetic transition + the next step's prompt."""
        tool_name = self._flow_data.get(session.tool, {}).get("name", session.tool)
        total = self._get_total_steps(session.tool)
        step = next_step["step"]
        goal = next_step.get("llm_goal", "")
        next_prompt = next_step["prompt"]

        system = (
            f"你正在引导用户做'{tool_name}'的第{step}/{total}步。\n"
            f"这一步的目标：{goal}\n"
            "先对用户刚才说的话给1句简短的共情或肯定（不超过15字），"
            "然后说出下面的引导语。可以稍微调整措辞让它更自然，但不要改变步骤的核心含义。"
        )
        user_prompt = f"用户刚才说：{user_response}\n\n这一步要说的引导语：\n{next_prompt}"

        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=256,
            )
            if response and len(response) > 10:
                return response
        except Exception:
            pass

        return next_prompt

    async def _generate_completion(self, user_response: str, session: Session) -> str:
        """Generate a warm completion message after the final step."""
        tool_name = self._flow_data.get(session.tool, {}).get("name", session.tool)

        system = (
            f"用户刚刚和你一起完成了'{tool_name}'。根据用户的最后回应，"
            "说1-2句温暖的话收尾。不要评价用户'做得好不好'，不要分析，"
            "只是简单地感谢和肯定。语气柔和，30字以内。"
        )
        try:
            response = self.llm.chat(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": f"用户说：{user_response}"},
                ],
                max_tokens=128,
            )
            if response and len(response) > 3:
                return response
        except Exception:
            pass

        return "谢谢你陪我走这一趟。"

    async def _send_completion(self, user_id: int, tool: str):
        """Called as fallback — generates and returns nothing, handled in caller."""
        pass
