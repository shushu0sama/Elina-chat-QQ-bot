from dataclasses import dataclass
from datetime import datetime
from typing import Callable
from zoneinfo import ZoneInfo

from .memory import MemoryStore


TextCommand = Callable[[str, int], str | None]
DateTimeHandler = Callable[[str, datetime], str | None]


@dataclass(frozen=True)
class CommandResult:
    handled: bool
    reply: str | None = None


class CommandRouter:
    def __init__(
        self,
        memory: MemoryStore,
        handle_memory_command: TextCommand,
        handle_manifestation_command: TextCommand,
        handle_date_time_question: DateTimeHandler,
        timezone_name: str = "Asia/Shanghai",
    ):
        self.memory = memory
        self.handle_memory_command = handle_memory_command
        self.handle_manifestation_command = handle_manifestation_command
        self.handle_date_time_question = handle_date_time_question
        self.timezone_name = timezone_name if isinstance(timezone_name, str) else "Asia/Shanghai"

    def route(self, user_msg: str, user_id: int, source_msg_id: int) -> CommandResult:
        memory_reply = self.handle_memory_command(user_msg, user_id)
        if memory_reply:
            return CommandResult(True, memory_reply)

        manifest_reply = self.handle_manifestation_command(user_msg, user_id)
        if manifest_reply:
            return CommandResult(True, manifest_reply)

        if user_msg.startswith("记住："):
            memory_text = user_msg[3:].strip()
            self.memory.add_key_memory(memory_text, source_msg_id=source_msg_id, user_id=user_id, importance=5)
            return CommandResult(True, f"记住了：{memory_text}")

        date_reply = self.handle_date_time_question(user_msg, datetime.now(ZoneInfo(self.timezone_name)))
        if date_reply:
            return CommandResult(True, date_reply)

        return CommandResult(False)
