from pydantic import BaseModel
from pathlib import Path


class Config(BaseModel):
    """Plugin configuration, loaded from .env and nonebot config."""

    # DeepSeek
    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com/v1"
    deepseek_model: str = "deepseek-v4-pro"

    # Memory
    memory_db_path: Path = Path("data/memory.db")
    max_recent_messages: int = 30
    summary_trigger_count: int = 20
    memory_injection_cap_tokens: int = 2000

    # Auto memory extraction
    auto_extract_interval: int = 10

    # Proactive chat
    proactive_enabled: bool = True
    proactive_interval_minutes: int = 30
    proactive_cooldown_minutes: int = 60
    proactive_active_hours_start: int = 9
    proactive_active_hours_end: int = 23

    # Content push (Bilibili)
    content_push_enabled: bool = True
    content_push_interval_hours: float = 6.0
    content_push_bili_categories: str = "0,36,188"
    content_push_max_per_push: int = 1

    # Bot identity
    bot_nickname: str = "艾琳娜"

    class Config:
        extra = "allow"
