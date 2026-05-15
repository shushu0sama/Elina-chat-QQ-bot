import json
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
from .relationship import RelationshipProfiler, build_relationship_prompt
from .web_search import search_web, format_search_results

# Web search tool definition for function calling
WEB_SEARCH_TOOL = {
    "type": "function",
    "function": {
        "name": "web_search",
        "description": (
            "当你不确定某个事实、需要最新信息、对方询问你知识范围外的问题、"
            "或想核实某个说法时，使用此工具搜索互联网获取实时信息。"
            "搜索结果会作为参考信息返回给你，请基于搜索结果用自然的语气回复。"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，提炼对方问题的核心，中英文均可。例如对方问'最近有什么好电影'可以搜索'2026年热门电影推荐'"
                }
            },
            "required": ["query"]
        }
    }
}

# Lazy-initialized globals — set in _startup()
plugin_config: Config | None = None
memory_store: MemoryStore | None = None
llm: LLMClient | None = None
proactive_chat: ProactiveChat | None = None
knowledge_base: KnowledgeBase | None = None
flow_manager: FlowManager | None = None
bili_fetcher: BilibiliFetcher | None = None
diary_writer: DiaryWriter | None = None
rel_profiler: RelationshipProfiler | None = None


async def _is_private_chat(event: Event) -> bool:
    return isinstance(event, PrivateMessageEvent)


async def _not_self_sent(bot: Bot, event: PrivateMessageEvent) -> bool:
    return event.user_id != event.self_id


private_msg = on_message(
    rule=Rule(_is_private_chat, _not_self_sent),
    block=False,
)


async def _extract_images(bot: Bot, event: PrivateMessageEvent) -> list[str]:
    """Extract downloadable image URLs from an event's message segments."""
    urls: list[str] = []
    for seg in event.message:
        if seg.type == "image":
            file_val = seg.data.get("file", "")
            # If already an HTTP URL, use directly
            if file_val.startswith("http"):
                urls.append(file_val)
            else:
                try:
                    info = await bot.get_image(file=file_val)
                    url = info.get("url", "")
                    if url:
                        urls.append(url)
                except Exception:
                    pass
    return urls


def _extract_faces(event: PrivateMessageEvent) -> list[str]:
    """Extract QQ face IDs from an event."""
    return [seg.data.get("id", "") for seg in event.message if seg.type == "face"]


@private_msg.handle()
async def handle_private_message(bot: Bot, event: PrivateMessageEvent):
    global memory_store, llm, plugin_config, flow_manager

    user_msg = event.get_plaintext().strip()
    image_urls = await _extract_images(bot, event)
    face_ids = _extract_faces(event)

    # Record user activity
    memory_store.record_user_active(event.user_id)

    # ── Flow session: user is in the middle of an interactive tool ──
    if flow_manager and flow_manager.has_session(event.user_id):
        if not user_msg and not image_urls:
            return

        # In flow session, images/faces get a simple acknowledgment
        if image_urls:
            await bot.send(event, "看到啦，不过现在我们还在走流程~ 要继续吗？还是先算了？")
            return
        if face_ids:
            return  # Silently ignore faces during flow

        if flow_manager.check_exit(event.user_id, user_msg):
            exit_msg = flow_manager.exit_session(event.user_id)
            if exit_msg:
                for chunk in LLMClient.chunk_text(exit_msg):
                    await bot.send(event, chunk)
            return

        next_msg = await flow_manager.advance(event.user_id, user_msg)
        if next_msg:
            for chunk in LLMClient.chunk_text(next_msg):
                await bot.send(event, chunk)
        else:
            flow_manager.exit_session(event.user_id)
        return

    # ── Image/face handling (outside flow sessions) ──
    if image_urls and not user_msg:
        # Pure image: use vision to understand and respond
        prompt = (
            "朋友给你发了一张图片。请用自然、朋友间聊天的语气回应这张图片的内容。"
            "可以描述你看到的、表达你的感受或想法。2-3句话即可，不要太长。"
            "如果你看不懂这张图（比如是纯色块或模糊的），就诚实地说看不清。"
        )
        reply = llm.chat_vision(prompt, image_urls[:3], max_tokens=384)
        if reply:
            memory_store.save_message("user", "[图片]", event.user_id)
            memory_store.save_message("assistant", reply, event.user_id)
            for chunk in LLMClient.chunk_text(reply):
                await bot.send(event, chunk)
        else:
            # Vision failed — natural fallback
            fallbacks = [
                "看到了~",
                "收到！",
                "哈哈哈这个是啥",
                "有意思",
            ]
            import random as _random
            fb = _random.choice(fallbacks)
            memory_store.save_message("user", "[图片]", event.user_id)
            memory_store.save_message("assistant", fb, event.user_id)
            await bot.send(event, fb)
        return

    if image_urls and user_msg:
        # Mixed: text + image — include both in context
        prompt = (
            f"朋友给你发了一张图片，同时说：「{user_msg}」。"
            "请结合图片内容回应对方。语气自然，像朋友聊天。2-3句话。"
            "如果你实在看不清图片，就根据对方的文字回应，顺便提一句图片没加载出来。"
        )
        reply = llm.chat_vision(prompt, image_urls[:3], max_tokens=384)
        if reply:
            memory_store.save_message("user", f"[图片] {user_msg}", event.user_id)
            memory_store.save_message("assistant", reply, event.user_id)
            for chunk in LLMClient.chunk_text(reply):
                await bot.send(event, chunk)
        else:
            # Vision failed, fall back to text-only
            user_msg_for_llm = f"（对方发了一张图片，说：{user_msg}。你看不到图片，根据文字回应。）"
            await _process_text_message(user_msg_for_llm, event, bot)
        return

    if face_ids and not user_msg:
        # Pure emoji/sticker — simple natural acknowledgment
        import random as _random
        face_replies = [
            "哈哈哈",
            "笑死",
            "好的",
            "嗯嗯",
            "收到了",
            "懂了",
        ]
        reply = _random.choice(face_replies)
        memory_store.save_message("user", "[表情]", event.user_id)
        memory_store.save_message("assistant", reply, event.user_id)
        await bot.send(event, reply)
        return

    if not user_msg:
        return

    # ── Flow tool request ──
    if flow_manager:
        tool = flow_manager.detect_tool_request(user_msg)
        if tool:
            first_step = flow_manager.start_session(event.user_id, tool, user_msg)
            memory_store.save_message("user", user_msg, event.user_id)
            if first_step:
                for chunk in LLMClient.chunk_text(first_step):
                    await bot.send(event, chunk)
                memory_store.save_message("assistant", first_step, event.user_id)
            return

    await _process_text_message(user_msg, event, bot)


async def _process_text_message(user_msg: str, event, bot):
    """Handle a regular text message (extracted from main handler for reuse)."""
    global memory_store, llm, plugin_config

    # 1. Save user message
    msg_id = memory_store.save_message("user", user_msg, event.user_id)

    # 2. Check if user wants to save a memory explicitly
    if user_msg.startswith("记住："):
        memory_text = user_msg[3:].strip()
        memory_store.add_key_memory(memory_text, source_msg_id=msg_id, user_id=event.user_id, importance=5)
        await bot.send(event, f"记住了：{memory_text}")
        return

    # 3. Retrieve relevant memories
    keywords = _extract_keywords(user_msg)
    retrieved = memory_store.retrieve_memories(keywords, event.user_id)

    # 4. Build the full message list for the LLM
    messages = _build_messages(user_msg, retrieved, event.user_id)

    # 5. Call LLM — with function calling if web search is enabled
    reply = ""
    if plugin_config and plugin_config.web_search_enabled:
        try:
            response = llm.chat_with_tools(messages, tools=[WEB_SEARCH_TOOL])

            if response is None:
                reply = llm.chat(messages)
            else:
                choice = response.choices[0]
                msg = choice.message

                if msg.tool_calls:
                    # LLM requested web search — execute and feed results back
                    assistant_msg = {
                        "role": "assistant",
                        "tool_calls": [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                }
                            }
                            for tc in msg.tool_calls
                        ],
                        "content": msg.content or "",
                    }
                    # DeepSeek thinking mode: must pass back reasoning_content
                    rc = getattr(msg, "reasoning_content", None)
                    if rc:
                        assistant_msg["reasoning_content"] = rc
                    messages.append(assistant_msg)

                    for tc in msg.tool_calls:
                        if tc.function.name == "web_search":
                            try:
                                args = json.loads(tc.function.arguments)
                                query = args.get("query", user_msg)
                                results = search_web(query)
                                formatted = format_search_results(results, query)
                            except Exception as e:
                                print(f"[Companion] Search error: {e}")
                                formatted = f"搜索「{tc.function.arguments}」时出错，请根据已有知识回答。"

                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": formatted,
                            })

                    # Second call: generate final response with search results baked in
                    reply = llm.chat(messages, max_tokens=2048)
                else:
                    reply = msg.content or ""
        except Exception as e:
            print(f"[Companion] Function calling failed, falling back to regular chat: {e}")
            reply = llm.chat(messages)
    else:
        # Web search disabled — use regular chat
        reply = llm.chat(messages)

    if not reply:
        reply = "嗯…我想了一下，不知道该怎么回。换个话题？"

    # 6. Save assistant reply
    memory_store.save_message("assistant", reply, event.user_id)

    # 7. Chunk and send reply
    for chunk in LLMClient.chunk_text(reply):
        await bot.send(event, chunk)

    # 8. Auto-extract memories
    await _maybe_extract_memories(event.user_id)

    # 9. Generate summary if needed
    if memory_store.message_count_since_last_summary(event.user_id) >= plugin_config.summary_trigger_count:
        recent = memory_store.get_recent_messages(plugin_config.max_recent_messages, event.user_id)
        summary = llm.generate_summary(recent)
        if summary:
            memory_store.save_summary(summary, 0, 0, event.user_id)


async def _maybe_extract_memories(user_id: int):
    """Periodically auto-extract new facts about the user from recent conversation."""
    global memory_store, llm, plugin_config

    count = memory_store.messages_since_last_extraction(user_id)
    if count < plugin_config.auto_extract_interval:
        return

    recent = memory_store.get_recent_messages(limit=20, user_id=user_id)
    if len(recent) < 5:
        return

    existing = memory_store.get_all_key_memories(user_id)
    new_memories = llm.extract_memories(recent, existing)

    if not new_memories:
        return

    added = 0
    for mem in new_memories:
        if not memory_store.has_similar_memory(mem, user_id):
            memory_store.add_key_memory(mem, user_id=user_id)
            added += 1

    if added > 0:
        print(f"[Companion] Auto-extracted {added} new memories (total: {memory_store.count_key_memories(user_id)})")

    # Clean stale low-priority memories every ~50 messages
    if count >= plugin_config.auto_extract_interval and count % 50 < plugin_config.auto_extract_interval:
        memory_store.prune_stale_memories(user_id, min_importance=2, days_unused=14)
        print(f"[Companion] Pruned stale memories for user {user_id}")


def _extract_keywords(text: str, top_n: int = 5) -> list[str]:
    words = jieba.cut(text)
    filtered = [w.strip() for w in words if len(w.strip()) >= 2]
    filtered.sort(key=len, reverse=True)
    return filtered[:top_n]


def _build_messages(user_msg: str, retrieved_memories: list[str], user_id: int = 0) -> list[dict]:
    global knowledge_base
    messages: list[dict] = []

    # System: injected memories (max 3, shuffled for variety, lower importance ones may be dropped)
    if retrieved_memories:
        import random as _random
        # Shuffle and cap at 3 — prevents the same memories dominating every turn
        mems = retrieved_memories.copy()
        _random.shuffle(mems)
        mems = mems[:3]
        memory_block = "[你记得以下关于用户的信息：]\n"
        for mem in mems:
            memory_block += f"- {mem}\n"
        messages.append({"role": "system", "content": memory_block})

    # System: philosophy knowledge (retrieved by keywords from user message)
    if knowledge_base:
        kb_prompt = build_knowledge_prompt(user_msg, knowledge_base)
        if kb_prompt:
            messages.append({"role": "system", "content": kb_prompt})

    # System: relationship context (per-user adaptation)
    if rel_profiler:
        rel_prompt = build_relationship_prompt(user_id, rel_profiler)
        if rel_prompt:
            messages.append({"role": "system", "content": rel_prompt})

    # System: conversation thread (keeps multi-turn coherence)
    recent = memory_store.get_recent_messages(plugin_config.max_recent_messages, user_id)
    thread_ctx = _build_thread_context(recent, user_msg)
    if thread_ctx:
        messages.append({"role": "system", "content": thread_ctx})

    # System: anti-repetition — show recent bot replies so model avoids repeating
    anti_repeat = _build_anti_repeat(recent)
    if anti_repeat:
        messages.append({"role": "system", "content": anti_repeat})

    # System: personality
    personality_prompt = build_system_prompt(user_id=user_id)
    messages.append({"role": "system", "content": personality_prompt})

    # Conversation history
    for msg in recent:
        messages.append(msg)

    if not (messages and messages[-1]["role"] == "user" and messages[-1]["content"] == user_msg):
        messages.append({"role": "user", "content": user_msg})

    return messages


def _build_anti_repeat(recent_messages: list[dict]) -> str:
    """Build a directive showing the bot's recent replies so it avoids repeating itself."""
    bot_replies = [m["content"] for m in recent_messages if m["role"] == "assistant"]
    if len(bot_replies) < 2:
        return ""

    # Take the last 3-5 bot replies
    recent_replies = bot_replies[-5:]
    lines = ["【重要：你最近几轮说过这些话，本次回复必须有所不同——不要重复相同的意思、相同的句式、相同的口头禅：】"]
    for i, reply in enumerate(recent_replies, 1):
        snippet = reply[:80].replace("\n", " ")
        lines.append(f"  {i}. {snippet}...")
    lines.append("请换个角度、换个语气、换个切入点。如果实在想不出不同的回应，就干脆换个话题。")
    return "\n".join(lines)


def _build_thread_context(recent_messages: list[dict], current_msg: str) -> str:
    """Build a compact context line summarizing the recent conversation thread."""
    if len(recent_messages) < 2:
        return ""

    # Take the last 3-4 exchanges (6-8 messages)
    recent_exchanges = recent_messages[-8:]
    user_msgs = [m["content"] for m in recent_exchanges if m["role"] == "user"]

    if not user_msgs:
        return ""

    # Extract keywords from recent user messages to detect topic continuity
    all_keywords: list[str] = []
    for msg in user_msgs[-3:]:
        all_keywords.extend(_extract_keywords(msg, top_n=3))

    # Find recurring keywords (appear in >= 2 recent messages)
    from collections import Counter
    kw_counts = Counter(all_keywords)
    recurring = [kw for kw, cnt in kw_counts.items() if cnt >= 2]

    lines = ["[当前对话线索——请在回复时保持话题连贯：]"]
    if recurring:
        lines.append(f"- 对方最近反复提到的词：{'、'.join(recurring[:3])}")

    # Summarize the last exchange
    if len(recent_exchanges) >= 2:
        last = recent_exchanges[-2:]
        summary_parts = []
        for m in last:
            role = "对方" if m["role"] == "user" else "你"
            snippet = m["content"][:60].replace("\n", " ")
            summary_parts.append(f"{role}：{snippet}")
        lines.append(f"- 上一轮：{' | '.join(summary_parts)}")

    if lines:
        lines.append("- 现在对方说：" + current_msg[:100])
        lines.append("- 请确保你的回应承接上一轮的话题。如果对方用了代词（'那个''它''这样'），要根据上下文理解具体指什么。")

    return "\n".join(lines)


@get_driver().on_startup
async def _startup():
    global plugin_config, memory_store, llm, proactive_chat, knowledge_base, flow_manager, bili_fetcher, diary_writer, rel_profiler

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
    diary_writer = DiaryWriter(llm, memory_store, plugin_config)
    rel_profiler = RelationshipProfiler(memory_store)

    # Initialize web search backend based on config
    try:
        from .web_search import set_backend, BingBackend, DuckDuckGoBackend
        if plugin_config.web_search_backend == "duckduckgo":
            set_backend(DuckDuckGoBackend())
        else:
            set_backend(BingBackend())
    except Exception as e:
        print(f"[Companion] Web search backend init failed: {e}")
        plugin_config.web_search_enabled = False

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
    print(f"[Companion] Web search: enabled={plugin_config.web_search_enabled}, "
          f"backend={plugin_config.web_search_backend}, "
          f"max_results={plugin_config.web_search_max_results}")
