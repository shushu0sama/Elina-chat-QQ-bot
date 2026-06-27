from dataclasses import dataclass

from .config import Config
from .content_fetcher import BilibiliFetcher
from .diary import DiaryWriter
from .feishu_calendar import FeishuCalendarClient
from .flows import FlowManager
from .knowledge import KnowledgeBase
from .llm_client import LLMClient
from .memory import MemoryStore
from .proactive import ProactiveChat
from .relationship import RelationshipProfiler
from .reminders import ReminderService


@dataclass
class AppServices:
    config: Config
    memory: MemoryStore
    llm: LLMClient
    proactive_chat: ProactiveChat
    knowledge_base: KnowledgeBase | None
    manifestation_knowledge_base: KnowledgeBase | None
    flow_manager: FlowManager | None
    bili_fetcher: BilibiliFetcher | None
    diary_writer: DiaryWriter | None
    relationship_profiler: RelationshipProfiler | None
    feishu_calendar_client: FeishuCalendarClient | None
    reminder_service: ReminderService | None
