import asyncio
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Awaitable, Callable, Protocol
from zoneinfo import ZoneInfo

from nonebot.log import logger

from .command_router import CommandRouter
from .feishu_calendar import (
    format_calendar_event_confirmation,
    format_calendar_intent_confirmation,
    looks_like_calendar_request,
    parse_calendar_request,
    should_confirm_calendar_request,
)
from .llm_client import LLMClient
from .services import AppServices
from .turn_context import analyze_turn, choose_reply_max_tokens, detect_emotions
from .web_search import format_search_results, search_web


class EventLike(Protocol):
    user_id: int


SendReply = Callable[[object, EventLike, str], Awaitable[int]]
TextCommand = Callable[[str, int], str | None]
MemoryExtractor = Callable[[int], Awaitable[None]]
TimelineRetriever = Callable[[str, list[str], int], list[dict]]
KeywordExtractor = Callable[[str], list[str]]
DateTimeHandler = Callable[[str, datetime], str | None]
FallbackProvider = Callable[[int | None], str]
SpeakerPrefixStripper = Callable[[str], str]
SnoozeUpdater = Callable[[int, str], None]
MessageBuilder = Callable[[str, list[str], int, list[str] | None, object, list[dict] | None], list[dict]]


@dataclass
class MessageHandlerCallbacks:
    send_event_reply: SendReply
    handle_memory_command: TextCommand
    handle_manifestation_command: TextCommand
    maybe_extract_memories: MemoryExtractor
    extract_keywords: KeywordExtractor
    retrieve_timeline_for_turn: TimelineRetriever
    handle_date_time_question: DateTimeHandler
    llm_fallback: FallbackProvider
    strip_outgoing_speaker_prefix: SpeakerPrefixStripper
    maybe_snooze_proactive_for_user_message: SnoozeUpdater
    build_messages: MessageBuilder


class MessageHandler:
    def __init__(self, services: AppServices, callbacks: MessageHandlerCallbacks, web_search_tool: dict):
        self.services = services
        self.callbacks = callbacks
        self.web_search_tool = web_search_tool

    async def process_text(self, user_msg: str, event: EventLike, bot: object):
        memory = self.services.memory
        llm = self.services.llm
        config = self.services.config

        msg_id = memory.save_message("user", user_msg, event.user_id)
        self.callbacks.maybe_snooze_proactive_for_user_message(event.user_id, user_msg)
        memory.maybe_add_timeline_entry_from_message(event.user_id, user_msg, msg_id)

        if self.services.reminder_service and config.reminder_enabled:
            management_reply = self.services.reminder_service.handle_management_command(event.user_id, user_msg)
            if management_reply:
                memory.save_message("assistant", management_reply, event.user_id)
                await self.callbacks.send_event_reply(bot, event, management_reply)
                return
            reminder = self.services.reminder_service.try_parse(user_msg)
            if reminder:
                if reminder.ok:
                    self.services.reminder_service.create_from_parse(event.user_id, reminder)
                    reply = reminder.confirmation
                else:
                    reply = reminder.clarification
                memory.save_message("assistant", reply, event.user_id)
                await self.callbacks.send_event_reply(bot, event, reply)
                return

        if self.services.feishu_calendar_client and looks_like_calendar_request(user_msg):
            request = parse_calendar_request(
                user_msg,
                now=datetime.now(ZoneInfo(config.feishu_timezone)),
                timezone_name=config.feishu_timezone,
            )
            if should_confirm_calendar_request(user_msg):
                if not request.ok:
                    return
                reply = format_calendar_intent_confirmation(request)
                memory.save_message("assistant", reply, event.user_id)
                await self.callbacks.send_event_reply(bot, event, reply)
                return
            if not request.ok:
                memory.save_message("assistant", request.clarification, event.user_id)
                await self.callbacks.send_event_reply(bot, event, request.clarification)
                return
            try:
                await self.services.feishu_calendar_client.create_event_from_request(request)
                reply = format_calendar_event_confirmation(request)
            except Exception:
                logger.warning("Feishu calendar create failed", exc_info=True)
                reply = "我想帮你记到飞书日历，但刚才创建失败了。可能是日历权限或网络问题，你可以稍后再试一次。"
            memory.save_message("assistant", reply, event.user_id)
            await self.callbacks.send_event_reply(bot, event, reply)
            return

        command = CommandRouter(
            memory,
            self.callbacks.handle_memory_command,
            self.callbacks.handle_manifestation_command,
            self.callbacks.handle_date_time_question,
            config.feishu_timezone,
        ).route(user_msg, event.user_id, msg_id)
        if command.handled:
            if command.reply:
                memory.save_message("assistant", command.reply, event.user_id)
                await self.callbacks.send_event_reply(bot, event, command.reply)
            return

        keywords = self.callbacks.extract_keywords(user_msg)
        user_emotions = detect_emotions(user_msg)
        retrieved = memory.retrieve_memories_with_emotion(keywords, event.user_id, user_emotions)
        timeline_entries = self.callbacks.retrieve_timeline_for_turn(user_msg, keywords, event.user_id)

        if retrieved:
            associated = memory.get_entity_associated_memories(retrieved[:3], event.user_id, set(retrieved))
        else:
            associated = []

        now = datetime.now(ZoneInfo("Asia/Shanghai"))
        turn_ctx = analyze_turn(user_msg, memory.get_recent_messages(config.max_recent_messages, event.user_id), self.services.flow_manager)
        messages = self.callbacks.build_messages(user_msg, retrieved, event.user_id, associated, turn_ctx, timeline_entries)
        max_tokens = choose_reply_max_tokens(turn_ctx)

        reply = await self._call_llm(user_msg, event.user_id, messages, turn_ctx, max_tokens, llm, config.web_search_enabled)
        reply = await asyncio.to_thread(llm.complete_if_needed, messages, reply, 256)
        reply = self.callbacks.strip_outgoing_speaker_prefix(reply)

        if not reply:
            reply = self.callbacks.llm_fallback(event.user_id)

        reply = self.callbacks.strip_outgoing_speaker_prefix(reply)
        memory.save_message("assistant", reply, event.user_id)
        await self.callbacks.send_event_reply(bot, event, reply)

        await self.callbacks.maybe_extract_memories(event.user_id)

        summary_count = memory.message_count_since_last_summary(event.user_id)
        if summary_count >= config.summary_trigger_count:
            end_msg_id = memory.get_latest_message_id(event.user_id)
            start_msg_id = memory.get_oldest_message_id_after(event.user_id, end_msg_id - summary_count)
            recent = memory.get_recent_messages(config.max_recent_messages, event.user_id)
            summary = await asyncio.to_thread(llm.generate_summary, recent)
            if summary:
                memory.save_summary(summary, start_msg_id, end_msg_id, event.user_id)

    async def _call_llm(self, user_msg: str, user_id: int, messages: list[dict], turn_ctx, max_tokens: int, llm: LLMClient, web_search_enabled: bool) -> str:
        if web_search_enabled and turn_ctx.allow_web_search:
            try:
                response = await asyncio.to_thread(llm.chat_with_tools, messages, [self.web_search_tool])
                if response is None:
                    return await asyncio.to_thread(llm.chat, messages, 3, max_tokens) or self.callbacks.llm_fallback(user_id)

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
                                },
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
                                results = await asyncio.to_thread(search_web, query)
                                formatted = format_search_results(results, query)
                            except Exception as e:
                                logger.warning(f"Search error: {e}")
                                formatted = f"搜索「{tc.function.arguments}」时出错，请根据已有知识回答。"
                            messages.append({"role": "tool", "tool_call_id": tc.id, "content": formatted})
                    return await asyncio.to_thread(llm.chat, messages, 3, max(1024, max_tokens)) or self.callbacks.llm_fallback(user_id)

                reply = msg.content or ""
                if choice.finish_reason == "length":
                    messages.append({"role": "assistant", "content": reply})
                    messages.append({
                        "role": "user",
                        "content": "继续刚才被截断的回复，只输出后续内容，不要重复已经说过的部分。",
                    })
                    continuation = await asyncio.to_thread(llm.chat, messages, 3, max_tokens)
                    reply += continuation or ""
                return await asyncio.to_thread(llm.complete_if_needed, messages, reply, 256)
            except Exception as e:
                logger.warning(f"Function calling failed, falling back to regular chat: {e}")
                return await asyncio.to_thread(llm.chat, messages, 3, max_tokens) or self.callbacks.llm_fallback(user_id)

        return await asyncio.to_thread(llm.chat, messages, 3, max_tokens) or self.callbacks.llm_fallback(user_id)
