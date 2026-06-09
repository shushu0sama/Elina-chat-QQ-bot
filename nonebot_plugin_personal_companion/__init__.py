import json
import random
import re
import jieba
from collections import Counter
from pathlib import Path

from nonebot import on_message, get_driver
from nonebot.adapters.onebot.v11 import Bot, Event, PrivateMessageEvent
from nonebot.plugin import PluginMetadata
from nonebot.rule import Rule

from .config import Config
from .memory import MemoryStore
from .personality import build_system_prompt
from .llm_client import LLMClient
from .proactive import ProactiveChat
from .knowledge import KnowledgeBase, build_knowledge_prompt_personalized
from .flows import FlowManager
from .content_fetcher import BilibiliFetcher
from .diary import DiaryWriter
from .relationship import RelationshipProfiler, build_relationship_prompt
from .turn_context import (
    analyze_turn,
    build_companion_context_prompt,
    choose_reply_max_tokens,
    detect_emotions,
)
from nonebot.log import logger

from .web_search import search_web, format_search_results
from .manifestation_quotes import build_frequency_first_aid_text

__plugin_meta__ = PluginMetadata(
    name="personal_companion",
    description="个人AI陪伴插件：人格系统、哲学知识库、交互式流程工具、长期记忆、主动推送、每日日记",
    usage="私聊直接对话；说「记住：XXX」保存记忆；说「陪我走流程」进入引导工具；说「算了」退出流程",
    config=Config,
    supported_adapters={"~onebot.v11"},
    extra={
        "author": "shushu0sama",
        "version": "0.1.0",
    },
)

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

# ── Error handling ─────────────────────────────────────────────

_LLM_FALLBACKS = [
    "我刚才没有生成出完整回复，你把上一句再发我一次好吗？",
    "我这边刚刚没接住这句，可以再说一遍吗？",
    "刚才那条回复没生成好，我想重新认真回你一次。",
]

def _llm_fallback(user_id: int | None = None) -> str:
    """Return a fallback message that avoids repeating recent fallback text."""
    if user_id is None or memory_store is None:
        return random.choice(_LLM_FALLBACKS)

    recent = memory_store.get_recent_messages(limit=20, user_id=user_id)
    recent_assistant = [m["content"] for m in recent if m["role"] == "assistant"]
    fresh = [fb for fb in _LLM_FALLBACKS if fb not in recent_assistant]
    if fresh:
        return random.choice(fresh)

    return "我这边刚刚没生成出可用的回复，你把上一句再发我一次好吗？"



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


async def _send_chunks(send_one, text: str) -> int:
    sent = 0
    last_error: Exception | None = None
    for chunk in LLMClient.chunk_text(text):
        for attempt in range(3):
            try:
                await send_one(chunk)
                sent += 1
                break
            except Exception as e:
                last_error = e
                logger.warning(f"Failed to send chunk attempt {attempt + 1}: {e}")
        else:
            raise last_error or RuntimeError("Failed to send message chunk")
    return sent


async def _send_event_reply(bot: Bot, event: PrivateMessageEvent, text: str) -> int:
    return await _send_chunks(lambda chunk: bot.send(event, chunk), text)


async def _send_private_reply(bot: Bot, user_id: int, text: str) -> int:
    return await _send_chunks(lambda chunk: bot.send_private_msg(user_id=user_id, message=chunk), text)


@private_msg.handle()
async def handle_private_message(bot: Bot, event: PrivateMessageEvent):
    global memory_store, llm, plugin_config, flow_manager
    # All globals are set during _startup() before any message is processed
    assert memory_store is not None and llm is not None and plugin_config is not None

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
                await _send_event_reply(bot, event, exit_msg)
            return

        active_tool = flow_manager.get_active_tool(event.user_id)
        next_msg = await flow_manager.advance(event.user_id, user_msg)
        if next_msg:
            await _send_event_reply(bot, event, next_msg)
            memory_store.save_message("user", user_msg, event.user_id)
            memory_store.save_message("assistant", next_msg, event.user_id)
            if active_tool and active_tool.startswith(("manifest_", "belief_", "obsession_", "future_")) and not flow_manager.has_session(event.user_id):
                memory_store.save_manifestation_entry(event.user_id, active_tool, next_msg)
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
            await _send_event_reply(bot, event, reply)
        else:
            fb = random.choice(["看到了~", "收到！", "哈哈哈这个是啥", "有意思"])
            memory_store.save_message("user", "[图片]", event.user_id)
            memory_store.save_message("assistant", fb, event.user_id)
            await _send_event_reply(bot, event, fb)
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
            await _send_event_reply(bot, event, reply)
        else:
            # Vision failed, fall back to text-only
            user_msg_for_llm = f"（对方发了一张图片，说：{user_msg}。你看不到图片，根据文字回应。）"
            await _process_text_message(user_msg_for_llm, event, bot)
        return

    if face_ids and not user_msg:
        # Pure emoji/sticker — simple natural acknowledgment
        face_replies = [
            "哈哈哈",
            "笑死",
            "好的",
            "嗯嗯",
            "收到了",
            "懂了",
        ]
        reply = random.choice(face_replies)
        memory_store.save_message("user", "[表情]", event.user_id)
        memory_store.save_message("assistant", reply, event.user_id)
        await _send_event_reply(bot, event, reply)
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
                await _send_event_reply(bot, event, first_step)
                memory_store.save_message("assistant", first_step, event.user_id)
            return

    await _process_text_message(user_msg, event, bot)


async def _process_text_message(user_msg: str, event, bot):
    """Handle a regular text message (extracted from main handler for reuse)."""
    global memory_store, llm, plugin_config
    assert memory_store is not None and llm is not None and plugin_config is not None

    # 1. Save user message
    msg_id = memory_store.save_message("user", user_msg, event.user_id)

    # 2. Explicit memory management commands
    memory_reply = _handle_memory_management_command(user_msg, event.user_id)
    if memory_reply:
        memory_store.save_message("assistant", memory_reply, event.user_id)
        await _send_event_reply(bot, event, memory_reply)
        return

    # 3. Manifestation dashboard and lifecycle commands
    manifest_reply = _handle_manifestation_command(user_msg, event.user_id)
    if manifest_reply:
        memory_store.save_message("assistant", manifest_reply, event.user_id)
        await _send_event_reply(bot, event, manifest_reply)
        return

    # 4. Check if user wants to save a memory explicitly
    if user_msg.startswith("记住："):
        memory_text = user_msg[3:].strip()
        memory_store.add_key_memory(memory_text, source_msg_id=msg_id, user_id=event.user_id, importance=5)
        await _send_event_reply(bot, event, f"记住了：{memory_text}")
        return

    # 4. Retrieve relevant memories (emotion-boosted + entity association)
    keywords = _extract_keywords(user_msg)
    user_emotions = detect_emotions(user_msg)
    retrieved = memory_store.retrieve_memories_with_emotion(keywords, event.user_id, user_emotions)

    # 4b. Entity association: find related memories via shared entities
    if retrieved:
        associated = memory_store.get_entity_associated_memories(
            retrieved[:3], event.user_id, set(retrieved)
        )
    else:
        associated = []

    # 5. Build the full message list for the LLM
    turn_ctx = analyze_turn(user_msg, memory_store.get_recent_messages(plugin_config.max_recent_messages, event.user_id), flow_manager)
    messages = _build_messages(user_msg, retrieved, event.user_id, associated, turn_ctx)
    max_tokens = choose_reply_max_tokens(turn_ctx)

    # 5. Call LLM — with budget check, function calling if web search is enabled
    reply = ""
    if plugin_config and plugin_config.web_search_enabled and turn_ctx.allow_web_search:
        try:
            response = llm.chat_with_tools(messages, tools=[WEB_SEARCH_TOOL])

            if response is None:
                reply = llm.chat(messages, max_tokens=max_tokens) or _llm_fallback(event.user_id)
            else:
                choice = response.choices[0]
                msg = choice.message

                if msg.tool_calls:
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
                                logger.warning(f"Search error: {e}")
                                formatted = f"搜索「{tc.function.arguments}」时出错，请根据已有知识回答。"

                            messages.append({
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": formatted,
                            })

                    reply = llm.chat(messages, max_tokens=max(1024, max_tokens)) or _llm_fallback(event.user_id)
                else:
                    reply = msg.content or ""
                    if choice.finish_reason == "length":
                        messages.append({"role": "assistant", "content": reply})
                        messages.append({
                            "role": "user",
                            "content": "继续刚才被截断的回复，只输出后续内容，不要重复已经说过的部分。",
                        })
                        reply += llm.chat(messages, max_tokens=max_tokens)
                    reply = llm.complete_if_needed(messages, reply, max_tokens=256)
        except Exception as e:
            logger.warning(f"Function calling failed, falling back to regular chat: {e}")
            reply = llm.chat(messages, max_tokens=max_tokens) or _llm_fallback(event.user_id)
    else:
        reply = llm.chat(messages, max_tokens=max_tokens) or _llm_fallback(event.user_id)

    reply = llm.complete_if_needed(messages, reply, max_tokens=256)

    if not reply:
        reply = _llm_fallback(event.user_id)

    # 6. Save assistant reply
    memory_store.save_message("assistant", reply, event.user_id)

    # 7. Chunk and send reply
    await _send_event_reply(bot, event, reply)

    # 8. Auto-extract memories
    await _maybe_extract_memories(event.user_id)

    # 9. Generate summary if needed
    summary_count = memory_store.message_count_since_last_summary(event.user_id)
    if summary_count >= plugin_config.summary_trigger_count:
        end_msg_id = memory_store.get_latest_message_id(event.user_id)
        start_msg_id = memory_store.get_oldest_message_id_after(event.user_id, end_msg_id - summary_count)
        recent = memory_store.get_recent_messages(plugin_config.max_recent_messages, event.user_id)
        summary = llm.generate_summary(recent)
        if summary:
            memory_store.save_summary(summary, start_msg_id, end_msg_id, event.user_id)


async def _maybe_extract_memories(user_id: int):
    """Periodically auto-extract new facts about the user from recent conversation."""
    global memory_store, llm, plugin_config
    assert memory_store is not None and llm is not None and plugin_config is not None

    count = memory_store.messages_since_last_extraction(user_id)
    if count < plugin_config.auto_extract_interval:
        return

    latest_msg_id = memory_store.get_latest_message_id(user_id)
    recent = memory_store.get_recent_messages(limit=20, user_id=user_id)
    if len(recent) < 5:
        memory_store.save_extraction_checkpoint(user_id, latest_msg_id)
        return

    existing = memory_store.get_all_key_memories(user_id)
    structured = llm.extract_memories_structured(recent, existing)

    if not structured:
        memory_store.save_extraction_checkpoint(user_id, latest_msg_id)
        return

    added = 0
    for item in structured:
        content = item["content"]
        if not memory_store.has_similar_memory(content, user_id):
            memory_store.add_key_memory(
                content,
                source_msg_id=latest_msg_id,
                user_id=user_id,
                emotion_tags=",".join(item.get("emotions", [])),
                entity_tags=",".join(item.get("entities", [])),
            )
            added += 1

    if added > 0:
        logger.info(f"Auto-extracted {added} new memories (total: {memory_store.count_key_memories(user_id)})")

    memory_store.save_extraction_checkpoint(user_id, latest_msg_id)

    # Clean stale low-priority memories every ~50 messages
    if count >= plugin_config.auto_extract_interval and count % 50 < plugin_config.auto_extract_interval:
        memory_store.prune_stale_memories(user_id, min_importance=2, days_unused=14)
        logger.info(f"Pruned stale memories for user {user_id}")


def _format_memory_update_result(action: str, matches: list[str]) -> str:
    if not matches:
        return "我没找到相关记忆。你可以说「我的记忆」看看我现在记得什么。"
    shown = "\n".join(f"- {m}" for m in matches[:5])
    suffix = f"\n另外还有 {len(matches) - 5} 条也已处理。" if len(matches) > 5 else ""
    return f"已{action}：\n{shown}{suffix}"


def _handle_memory_management_command(user_msg: str, user_id: int) -> str | None:
    assert memory_store is not None
    stripped = user_msg.strip()

    if stripped in {"我的记忆", "你记得我什么", "整理一下你记得我的什么", "查看记忆", "记忆列表"}:
        return memory_store.build_memory_overview(user_id)

    list_prefixes = ["搜索记忆：", "搜索记忆:", "查记忆：", "查记忆:"]
    for prefix in list_prefixes:
        if stripped.startswith(prefix):
            query = stripped[len(prefix):].strip()
            items = memory_store.find_key_memories(user_id, query=query, limit=10, include_inactive=True)
            if not items:
                return "我没找到相关记忆。"
            lines = [f"和「{query}」相关的记忆："]
            for item in items:
                lines.append(f"- #{item['id']} [{item['memory_type']}/{item['status']}] {item['content']}")
            return "\n".join(lines)

    forget_prefixes = ["忘掉：", "忘掉:", "忘记：", "忘记:", "删除记忆：", "删除记忆:"]
    for prefix in forget_prefixes:
        if stripped.startswith(prefix):
            query = stripped[len(prefix):].strip()
            if not query:
                return "可以说「忘掉：关键词」来删除相关记忆。"
            return _format_memory_update_result("忘掉", memory_store.delete_key_memories(user_id, query))

    ended_prefixes = ["这件事结束了：", "这件事结束了:", "这件事已经结束了：", "这件事已经结束了:", "标记结束：", "标记结束:"]
    for prefix in ended_prefixes:
        if stripped.startswith(prefix):
            query = stripped[len(prefix):].strip()
            if not query:
                return "可以说「这件事结束了：关键词」来把相关记忆标记为已结束。"
            matches = memory_store.update_key_memory_status(user_id, query, "completed", memory_type="event")
            return _format_memory_update_result("标记为已结束", matches)

    suppress_prefixes = ["以后别再提：", "以后别再提:", "不要再提：", "不要再提:", "别再提：", "别再提:"]
    for prefix in suppress_prefixes:
        if stripped.startswith(prefix):
            query = stripped[len(prefix):].strip()
            if not query:
                return "可以说「以后别再提：关键词」来隐藏相关记忆，但不删除它。"
            matches = memory_store.update_key_memory_status(user_id, query, "suppressed")
            return _format_memory_update_result("设为不再主动提起", matches)

    return None


def _handle_manifestation_command(user_msg: str, user_id: int) -> str | None:
    assert memory_store is not None
    stripped = user_msg.strip()

    dashboard_triggers = ["我的显化状态", "显化仪表盘", "我的愿望列表", "我的显化进度", "总结我的显化证据"]
    if any(trigger in stripped for trigger in dashboard_triggers):
        return memory_store.build_manifestation_dashboard(user_id)

    evidence_prefixes = ["记录显化证据：", "记录显化证据:", "显化证据：", "显化证据:"]
    for prefix in evidence_prefixes:
        if stripped.startswith(prefix):
            content = stripped[len(prefix):].strip()
            if not content:
                return "可以直接说：记录显化证据：我今天没有反复确认。"
            wishes = memory_store.get_manifestation_wishes(user_id, statuses=["active"], limit=1)
            wish_id = wishes[0]["id"] if wishes else None
            memory_store.add_manifestation_evidence(user_id, content, wish_id=wish_id)
            return "记下来了。这是一条显化证据，不管它多小，都说明你正在从旧版本里出来。"

    status_match = re.search(r"愿望#?(\d+).*(完成了|实现了|放下了|释放了|暂停|先停|过期|不重要了)", stripped)
    if status_match:
        wish_id = int(status_match.group(1))
        phrase = status_match.group(2)
        if phrase in ["完成了", "实现了"]:
            status = "fulfilled"
            text = "已把这颗愿望标记为完成。我们不只记结果，也记得你一路成为了更能承接它的人。"
        elif phrase in ["放下了", "释放了"]:
            status = "released"
            text = "已把这颗愿望标记为放下。放下不是失败，是把控制还给生命，把力量还给自己。"
        elif phrase in ["暂停", "先停"]:
            status = "paused"
            text = "已把这颗愿望标记为暂停。等你想重新滋养它时，再叫我。"
        else:
            status = "expired"
            text = "已把这颗愿望标记为过期。愿望变化也没关系，你可以继续选择更适合现在的自己。"
        memory_store.update_manifestation_wish_status(user_id, wish_id, status)
        return text

    if any(trigger in stripped for trigger in ["给我肯定句", "今日肯定句", "生成肯定句", "显化咒语"]):
        wishes = memory_store.get_manifestation_wishes(user_id, statuses=["active"], limit=1)
        if not wishes:
            return "今天的肯定句：我允许自己稳定下来，也允许好的事情用适合我的方式靠近。"
        title = wishes[0]["title"]
        return (
            f"围绕「{title}」，今天给你三档肯定句：\n\n"
            "我现在能相信的版本：我正在一点点靠近更好的状态。\n"
            "更高版本：我值得自然、清楚、稳定地拥有适合我的结果。\n"
            "已拥有版本：我已经在用新版本的自己生活、选择和行动。"
        )

    return None


def _extract_keywords(text: str, top_n: int = 5) -> list[str]:
    words = jieba.cut(text)
    filtered = [w.strip() for w in words if len(w.strip()) >= 2]
    filtered.sort(key=len, reverse=True)
    return filtered[:top_n]



def _build_messages(user_msg: str, retrieved_memories: list[str], user_id: int = 0,
                    associated_memories: list[str] | None = None,
                    turn_ctx=None) -> list[dict]:
    global knowledge_base
    assert memory_store is not None and plugin_config is not None

    messages: list[dict] = []

    # System: injected memories (max 3, shuffled for variety, lower importance ones may be dropped)
    if retrieved_memories:
        # Shuffle and cap at 3 — prevents the same memories dominating every turn
        mems = retrieved_memories.copy()
        random.shuffle(mems)
        mems = mems[:3]
        memory_block = "[你记得以下关于用户的当前有效信息。只能在和用户本轮话题自然相关时轻轻使用；不要为了展示记忆而主动翻旧账：]\n"
        for mem in mems:
            memory_block += f"- {mem}\n"
        messages.append({"role": "system", "content": memory_block})

    # System: entity-associated memories (associative recall — shared entities with primary memories)
    if associated_memories:
        assoc = associated_memories[:2]
        assoc_block = (
            "[你联想到了以下仍然有效的相关信息——只有话题自然关联到时才可以轻轻提起，绝对不要追问旧事：]\n"
        )
        for mem in assoc:
            assoc_block += f"- {mem}\n"
        messages.append({"role": "system", "content": assoc_block})

    # System: philosophy knowledge (retrieved by keywords from user message)
    if knowledge_base:
        kb_prompt = build_knowledge_prompt_personalized(
            user_msg, knowledge_base, user_id, memory_store
        )
        if kb_prompt:
            messages.append({"role": "system", "content": kb_prompt})

    # System: relationship context (per-user adaptation)
    if rel_profiler:
        rel_prompt = build_relationship_prompt(user_id, rel_profiler)
        if rel_prompt:
            messages.append({"role": "system", "content": rel_prompt})

    recent = memory_store.get_recent_messages(plugin_config.max_recent_messages, user_id)

    # System: current-turn companion context
    turn_ctx = turn_ctx or analyze_turn(user_msg, recent, flow_manager)
    companion_ctx = build_companion_context_prompt(turn_ctx)
    if companion_ctx:
        messages.append({"role": "system", "content": companion_ctx})

    # System: manifestation context
    manifestation_memories = memory_store.get_manifestation_memories(user_id, limit=5)
    if manifestation_memories:
        manifest_block = (
            "[艾琳娜显化系统记忆——用于愿望澄清、信念改写、显化日记和执念降频：]\n"
            + "\n".join(f"- {m}" for m in manifestation_memories)
            + "\n使用规则：只在用户聊到显化、愿望、信念、执念、未来自我时自然使用；不要主动保证结果，不要把显化变成压力。"
        )
        messages.append({"role": "system", "content": manifest_block})

    # System: frequency first-aid for manifestation anxiety and old stories
    recent_proactive = memory_store.get_recent_proactive_content(user_id, limit=3)
    first_aid = build_frequency_first_aid_text(user_msg, recent_proactive)
    if first_aid:
        messages.append({"role": "system", "content": (
            "[当前用户可能被过去、焦虑、低频或旧故事拉住。请优先温柔降频，再回应具体内容。]\n"
            + first_aid
            + "\n使用边界：不要说用户频率太低，不要要求立刻开心，不要承诺结果；目标只是从焦虑降到中性。"
        )})

    # System: conversation thread (keeps multi-turn coherence)
    thread_ctx = _build_thread_context(recent, user_msg)
    if thread_ctx:
        messages.append({"role": "system", "content": thread_ctx})

    # System: anti-repetition — show recent bot replies so model avoids repeating
    anti_repeat = _build_anti_repeat(recent)
    if anti_repeat:
        messages.append({"role": "system", "content": anti_repeat})

    # System: personality
    personality_prompt = build_system_prompt(user_id=user_id, turn_context=turn_ctx)
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
        logger.error(f"Web search backend init failed: {e}")
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

    logger.info(f"Plugin loaded. DB: {plugin_config.memory_db_path}")
    logger.info(f"Model: {plugin_config.deepseek_model}")
    logger.info(f"Nickname: {plugin_config.bot_nickname}")
    logger.info(f"Knowledge base: {len(knowledge_base.concepts)} concepts loaded")
    logger.info(f"Flow tools: process ({len(flow_manager._get_steps('process'))} steps), "
          f"mini ({len(flow_manager._get_steps('mini_process'))} steps), "
          f"appreciation ({len(flow_manager._get_steps('appreciation'))} steps)")
    logger.info(f"Proactive chat: enabled={plugin_config.proactive_enabled}, "
          f"interval={plugin_config.proactive_interval_minutes}min, "
          f"cooldown={plugin_config.proactive_cooldown_minutes}min, "
          f"hours={plugin_config.proactive_active_hours_start}-{plugin_config.proactive_active_hours_end}")
    logger.info(f"Content push: enabled={plugin_config.content_push_enabled}, "
          f"interval={plugin_config.content_push_interval_hours}h, "
          f"categories={plugin_config.content_push_bili_categories}, "
          f"max_per_push={plugin_config.content_push_max_per_push}")
    logger.info("Daily diary: scheduled at 00:00 each day")
    logger.info(f"Web search: enabled={plugin_config.web_search_enabled}, "
          f"backend={plugin_config.web_search_backend}, "
          f"max_results={plugin_config.web_search_max_results}")
