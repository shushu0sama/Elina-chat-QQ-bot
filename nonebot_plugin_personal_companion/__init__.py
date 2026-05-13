import jieba
from pathlib import Path

from nonebot import on_message, get_driver
from nonebot.adapters.onebot.v11 import Bot, Event, PrivateMessageEvent
from nonebot.rule import Rule

from .config import Config
from .memory import MemoryStore
from .personality import build_system_prompt
from .llm_client import LLMClient
from .proactive import ProactiveChat
from .knowledge import KnowledgeBase, build_knowledge_prompt
from .flows import FlowManager
from .content_fetcher import BilibiliFetcher
from .diary import DiaryWriter

# Lazy-initialized globals — set in _startup()
plugin_config: Config | None = None
memory_store: MemoryStore | None = None
llm: LLMClient | None = None
proactive_chat: ProactiveChat | None = None
knowledge_base: KnowledgeBase | None = None
flow_manager: FlowManager | None = None
bili_fetcher: BilibiliFetcher | None = None
diary_writer: DiaryWriter | None = None


async def _is_private_chat(event: Event) -> bool:
    return isinstance(event, PrivateMessageEvent)


async def _not_self_sent(bot: Bot, event: PrivateMessageEvent) -> bool:
    return event.user_id != event.self_id


private_msg = on_message(
    rule=Rule(_is_private_chat, _not_self_sent),
    block=False,
)


@private_msg.handle()
async def handle_private_message(bot: Bot, event: PrivateMessageEvent):
    global memory_store, llm, plugin_config, flow_manager

    user_msg = event.get_plaintext().strip()

    # Record user activity (even for empty messages — they're online)
    memory_store.record_user_active(event.user_id)

    # ── Flow session: user is in the middle of an interactive tool ──
    if flow_manager and flow_manager.has_session(event.user_id):
        if not user_msg:
            return

        # Check if user wants to exit
        if flow_manager.check_exit(event.user_id, user_msg):
            exit_msg = flow_manager.exit_session(event.user_id)
            if exit_msg:
                for chunk in LLMClient.chunk_text(exit_msg):
                    await bot.send(event, chunk)
            return

        # Advance the flow
        next_msg = await flow_manager.advance(event.user_id, user_msg)
        if next_msg:
            for chunk in LLMClient.chunk_text(next_msg):
                await bot.send(event, chunk)
        else:
            # Session ended unexpectedly — ensure cleanup
            flow_manager.exit_session(event.user_id)
        return

    if not user_msg:
        return

    # ── Flow tool request: user explicitly asks for a tool ──
    if flow_manager:
        tool = flow_manager.detect_tool_request(user_msg)
        if tool:
            first_step = flow_manager.start_session(event.user_id, tool, user_msg)
            # Save user message first
            memory_store.save_message("user", user_msg, event.user_id)
            # Send the first step
            if first_step:
                for chunk in LLMClient.chunk_text(first_step):
                    await bot.send(event, chunk)
                memory_store.save_message("assistant", first_step, event.user_id)
            return

    # 1. Save user message
    msg_id = memory_store.save_message("user", user_msg, event.user_id)

    # 2. Check if user wants to save a memory explicitly
    if user_msg.startswith("记住："):
        memory_text = user_msg[3:].strip()
        memory_store.add_key_memory(memory_text, source_msg_id=msg_id)
        await bot.send(event, f"记住了：{memory_text}")
        return

    # 3. Retrieve relevant memories
    keywords = _extract_keywords(user_msg)
    retrieved = memory_store.retrieve_memories(keywords)

    # 4. Build the full message list for the LLM
    messages = _build_messages(user_msg, retrieved, event.user_id)

    # 5. Call LLM
    reply = llm.chat(messages)

    # 6. Save assistant reply
    memory_store.save_message("assistant", reply, event.user_id)

    # 7. Chunk and send reply
    for chunk in LLMClient.chunk_text(reply):
        await bot.send(event, chunk)

    # 8. Auto-extract memories (async, after reply sent)
    await _maybe_extract_memories(event.user_id)

    # 9. Check if we should generate a summary
    if memory_store.message_count_since_last_summary() >= plugin_config.summary_trigger_count:
        recent = memory_store.get_recent_messages(plugin_config.max_recent_messages, event.user_id)
        summary = llm.generate_summary(recent)
        if summary:
            memory_store.save_summary(summary, 0, 0)


async def _maybe_extract_memories(user_id: int):
    """Periodically auto-extract new facts about the user from recent conversation."""
    global memory_store, llm, plugin_config

    count = memory_store.messages_since_last_extraction()
    if count < plugin_config.auto_extract_interval:
        return

    recent = memory_store.get_recent_messages(limit=20, user_id=user_id)
    if len(recent) < 5:
        return

    existing = memory_store.get_all_key_memories()
    new_memories = llm.extract_memories(recent, existing)

    if not new_memories:
        # Mark as processed even if nothing found (avoid re-extracting same messages)
        return

    added = 0
    for mem in new_memories:
        if not memory_store.has_similar_memory(mem):
            memory_store.add_key_memory(mem)
            added += 1

    if added > 0:
        print(f"[Companion] Auto-extracted {added} new memories (total: {memory_store.count_key_memories()})")


def _extract_keywords(text: str, top_n: int = 5) -> list[str]:
    words = jieba.cut(text)
    filtered = [w.strip() for w in words if len(w.strip()) >= 2]
    filtered.sort(key=len, reverse=True)
    return filtered[:top_n]


def _build_messages(user_msg: str, retrieved_memories: list[str], user_id: int = 0) -> list[dict]:
    global knowledge_base
    messages: list[dict] = []

    # System: injected memories
    if retrieved_memories:
        memory_block = "[你记得以下关于用户的信息：]\n"
        for mem in retrieved_memories:
            memory_block += f"- {mem}\n"
        messages.append({"role": "system", "content": memory_block})

    # System: philosophy knowledge (retrieved by keywords from user message)
    if knowledge_base:
        kb_prompt = build_knowledge_prompt(user_msg, knowledge_base)
        if kb_prompt:
            messages.append({"role": "system", "content": kb_prompt})

    # System: personality
    personality_prompt = build_system_prompt(user_id=user_id)
    messages.append({"role": "system", "content": personality_prompt})

    # Conversation history
    recent = memory_store.get_recent_messages(plugin_config.max_recent_messages, user_id)
    for msg in recent:
        messages.append(msg)

    if not (messages and messages[-1]["role"] == "user" and messages[-1]["content"] == user_msg):
        messages.append({"role": "user", "content": user_msg})

    return messages


@get_driver().on_startup
async def _startup():
    global plugin_config, memory_store, llm, proactive_chat, knowledge_base, flow_manager, bili_fetcher, diary_writer

    # Import scheduler here — NoneBot is initialized by now
    from nonebot import require
    require("nonebot_plugin_apscheduler")
    from nonebot_plugin_apscheduler import scheduler

    plugin_config = Config.parse_obj(get_driver().config.dict())
    memory_store = MemoryStore(db_path=Path(plugin_config.memory_db_path))
    llm = LLMClient(
        api_key=plugin_config.deepseek_api_key,
        base_url=plugin_config.deepseek_base_url,
        model=plugin_config.deepseek_model,
    )
    knowledge_base = KnowledgeBase()
    proactive_chat = ProactiveChat(memory_store, llm, plugin_config, knowledge_base)
    flow_manager = FlowManager(llm)
    bili_fetcher = BilibiliFetcher(llm, memory_store, plugin_config)
    diary_writer = DiaryWriter(llm, memory_store)

    # Register proactive chat job
    scheduler.add_job(
        proactive_chat.try_proactive,
        trigger="interval",
        minutes=plugin_config.proactive_interval_minutes,
        jitter=60,
        id="proactive_chat",
        replace_existing=True,
    )

    # Register content push job
    scheduler.add_job(
        bili_fetcher.try_push,
        trigger="interval",
        hours=plugin_config.content_push_interval_hours,
        jitter=300,  # ±5 min jitter
        id="content_push",
        replace_existing=True,
    )

    # Register midnight diary job (cron: 0 0 * * *)
    scheduler.add_job(
        diary_writer.write_daily_diary,
        trigger="cron",
        hour=0,
        minute=0,
        jitter=120,  # ±2 min jitter to avoid peak
        id="daily_diary",
        replace_existing=True,
    )

    print(f"[Companion] Plugin loaded. DB: {plugin_config.memory_db_path}")
    print(f"[Companion] Model: {plugin_config.deepseek_model}")
    print(f"[Companion] Nickname: {plugin_config.bot_nickname}")
    print(f"[Companion] Knowledge base: {len(knowledge_base.concepts)} concepts loaded")
    print(f"[Companion] Flow tools: process ({len(flow_manager._get_steps('process'))} steps), "
          f"mini ({len(flow_manager._get_steps('mini_process'))} steps), "
          f"appreciation ({len(flow_manager._get_steps('appreciation'))} steps)")
    print(f"[Companion] Proactive chat: enabled={plugin_config.proactive_enabled}, "
          f"interval={plugin_config.proactive_interval_minutes}min, "
          f"cooldown={plugin_config.proactive_cooldown_minutes}min, "
          f"hours={plugin_config.proactive_active_hours_start}-{plugin_config.proactive_active_hours_end}")
    print(f"[Companion] Content push: enabled={plugin_config.content_push_enabled}, "
          f"interval={plugin_config.content_push_interval_hours}h, "
          f"categories={plugin_config.content_push_bili_categories}, "
          f"max_per_push={plugin_config.content_push_max_per_push}")
    print(f"[Companion] Daily diary: scheduled at 00:00 each day")
