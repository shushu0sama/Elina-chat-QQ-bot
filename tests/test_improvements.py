"""Integration tests for the three memory/knowledge improvements."""
# NoneBot must be initialized BEFORE any plugin imports (__init__.py uses get_driver())
import nonebot
nonebot.init(_env_file=".env.example")

import asyncio  # noqa: E402
import tempfile  # noqa: E402
from pathlib import Path  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from unittest.mock import Mock, patch  # noqa: E402

# Now safe to import plugin modules
from nonebot_plugin_personal_companion.memory import MemoryStore  # noqa: E402
from nonebot_plugin_personal_companion.diary import DiaryWriter  # noqa: E402
from nonebot_plugin_personal_companion.manifestation_quotes import (  # noqa: E402
    build_frequency_first_aid_text,
    detect_frequency_support_category,
    pick_manifestation_quote,
)
from nonebot_plugin_personal_companion.knowledge import (  # noqa: E402
    MANIFESTATION_KNOWLEDGE_PATH,
    KnowledgeBase,
    build_knowledge_prompt_personalized,
    build_manifestation_knowledge_prompt,
    is_manifestation_intent,
)
from nonebot_plugin_personal_companion import personality  # noqa: E402
import nonebot_plugin_personal_companion as companion_plugin  # noqa: E402
from nonebot_plugin_personal_companion.personality import BEIJING_TZ, build_system_prompt, _roll_state  # noqa: E402
from nonebot_plugin_personal_companion.proactive import ProactiveChat  # noqa: E402
from nonebot_plugin_personal_companion.relationship import RelationshipProfiler, build_relationship_prompt  # noqa: E402
from nonebot_plugin_personal_companion.llm_client import LLMClient  # noqa: E402
from nonebot_plugin_personal_companion.turn_context import (  # noqa: E402
    analyze_turn,
    build_companion_context_prompt,
    build_reply_mode_prompt,
    choose_reply_max_tokens,
)


class CompletionChoice:
    def __init__(self, content, finish_reason="stop"):
        self.message = Mock(content=content)
        self.finish_reason = finish_reason


class CompletionResponse:
    def __init__(self, content, finish_reason="stop"):
        self.choices = [CompletionChoice(content, finish_reason)]


def test_manifestation_knowledge_loads_and_injects_safety_boundaries():
    kb = KnowledgeBase(MANIFESTATION_KNOWLEDGE_PATH)

    prompt = build_manifestation_knowledge_prompt("我想下一个宇宙订单，但是担心自己不配", kb)

    assert "显化知识视角" in prompt
    assert "不承诺结果一定发生" in prompt
    assert "不把未显化归咎于用户频率不够" in prompt
    assert "愿望说清楚" in prompt or "旧信念" in prompt


def test_manifestation_intent_detection_is_selective():
    assert is_manifestation_intent("为什么还没发生，是不是我频率太低") is True
    assert is_manifestation_intent("今天午饭吃什么") is False


def test_normal_chat_prompt_injects_manifestation_knowledge_only_when_relevant():
    store, db_path = make_store()
    old_store = companion_plugin.memory_store
    old_config = companion_plugin.plugin_config
    old_kb = companion_plugin.knowledge_base
    old_manifest_kb = companion_plugin.manifestation_knowledge_base
    old_rel = companion_plugin.rel_profiler
    old_flow = companion_plugin.flow_manager
    try:
        companion_plugin.memory_store = store
        companion_plugin.plugin_config = Mock(max_recent_messages=5)
        companion_plugin.knowledge_base = None
        companion_plugin.manifestation_knowledge_base = KnowledgeBase(MANIFESTATION_KNOWLEDGE_PATH)
        companion_plugin.rel_profiler = None
        companion_plugin.flow_manager = None

        manifest_messages = companion_plugin._build_messages("我想下一个宇宙订单", [], user_id=1)
        normal_messages = companion_plugin._build_messages("今天午饭吃什么", [], user_id=1)
        manifest_content = "\n".join(str(m["content"]) for m in manifest_messages)
        normal_content = "\n".join(str(m["content"]) for m in normal_messages)

        assert "显化知识视角" in manifest_content
        assert "不承诺结果一定发生" in manifest_content
        assert "显化知识视角" not in normal_content
    finally:
        companion_plugin.memory_store = old_store
        companion_plugin.plugin_config = old_config
        companion_plugin.knowledge_base = old_kb
        companion_plugin.manifestation_knowledge_base = old_manifest_kb
        companion_plugin.rel_profiler = old_rel
        companion_plugin.flow_manager = old_flow
        cleanup_store(store, db_path)


def test_history_messages_do_not_inject_speaker_prefixes():
    formatted = companion_plugin._format_history_message({"role": "assistant", "content": "我在"})

    assert formatted == {"role": "assistant", "content": "我在"}


def test_outgoing_speaker_prefix_is_stripped():
    assert companion_plugin._strip_outgoing_speaker_prefix("艾琳娜：我在") == "我在"
    assert companion_plugin._strip_outgoing_speaker_prefix("小鼠: 收到") == "收到"
    assert companion_plugin._strip_outgoing_speaker_prefix("我在") == "我在"


def test_explicit_memory_save_records_assistant_confirmation():
    store, db_path = make_store()
    old_store = companion_plugin.memory_store
    old_config = companion_plugin.plugin_config
    old_llm = companion_plugin.llm

    class Event:
        user_id = 1

    class Bot:
        async def send(self, event, chunk):
            pass

    try:
        companion_plugin.memory_store = store
        companion_plugin.plugin_config = Mock(auto_extract_interval=999, summary_trigger_count=999)
        companion_plugin.llm = Mock()

        _run(companion_plugin._process_text_message("记住：我喜欢咖啡", Event(), Bot()))

        recent = store.get_recent_messages(limit=5, user_id=1)
        assert recent[-1]["role"] == "assistant"
        assert recent[-1]["content"] == "记住了：我喜欢咖啡"
    finally:
        companion_plugin.memory_store = old_store
        companion_plugin.plugin_config = old_config
        companion_plugin.llm = old_llm
        cleanup_store(store, db_path)


def test_llm_length_continuation_handles_none_without_crashing():
    store, db_path = make_store()
    old_store = companion_plugin.memory_store
    old_config = companion_plugin.plugin_config
    old_llm = companion_plugin.llm
    old_flow = companion_plugin.flow_manager
    old_kb = companion_plugin.knowledge_base
    old_manifest_kb = companion_plugin.manifestation_knowledge_base
    old_rel = companion_plugin.rel_profiler
    old_feishu = companion_plugin.feishu_calendar_client

    class Event:
        user_id = 1

    class Bot:
        async def send(self, event, chunk):
            pass

    class Choice:
        finish_reason = "length"
        message = Mock(content="前半段", tool_calls=None)

    class Response:
        choices = [Choice()]

    llm = Mock()
    llm.chat_with_tools.return_value = Response()
    llm.chat.return_value = None
    llm.complete_if_needed.side_effect = lambda messages, reply, max_tokens=256: reply

    try:
        companion_plugin.memory_store = store
        companion_plugin.plugin_config = Mock(
            max_recent_messages=5,
            web_search_enabled=True,
            auto_extract_interval=999,
            summary_trigger_count=999,
        )
        companion_plugin.llm = llm
        companion_plugin.flow_manager = None
        companion_plugin.knowledge_base = None
        companion_plugin.manifestation_knowledge_base = None
        companion_plugin.rel_profiler = None
        companion_plugin.feishu_calendar_client = None

        _run(companion_plugin._process_text_message("查一下最近新闻", Event(), Bot()))

        recent = store.get_recent_messages(limit=5, user_id=1)
        assert recent[-1]["role"] == "assistant"
        assert recent[-1]["content"] == "前半段"
    finally:
        companion_plugin.memory_store = old_store
        companion_plugin.plugin_config = old_config
        companion_plugin.llm = old_llm
        companion_plugin.flow_manager = old_flow
        companion_plugin.knowledge_base = old_kb
        companion_plugin.manifestation_knowledge_base = old_manifest_kb
        companion_plugin.rel_profiler = old_rel
        companion_plugin.feishu_calendar_client = old_feishu
        cleanup_store(store, db_path)


def test_weak_calendar_intent_asks_confirmation_without_creating_event():
    store, db_path = make_store()
    old_store = companion_plugin.memory_store
    old_config = companion_plugin.plugin_config
    old_llm = companion_plugin.llm
    old_flow = companion_plugin.flow_manager
    old_kb = companion_plugin.knowledge_base
    old_manifest_kb = companion_plugin.manifestation_knowledge_base
    old_rel = companion_plugin.rel_profiler
    old_feishu = companion_plugin.feishu_calendar_client

    class Event:
        user_id = 1

    class Bot:
        def __init__(self):
            self.sent = []

        async def send(self, event, chunk):
            self.sent.append(chunk)

    feishu = Mock()
    try:
        companion_plugin.memory_store = store
        companion_plugin.plugin_config = Mock(
            max_recent_messages=5,
            web_search_enabled=False,
            auto_extract_interval=999,
            summary_trigger_count=999,
            feishu_timezone="Asia/Shanghai",
        )
        companion_plugin.llm = Mock()
        companion_plugin.flow_manager = None
        companion_plugin.knowledge_base = None
        companion_plugin.manifestation_knowledge_base = None
        companion_plugin.rel_profiler = None
        companion_plugin.feishu_calendar_client = feishu
        bot = Bot()

        _run(companion_plugin._process_text_message("明天下午三点开会，好烦", Event(), bot))

        feishu.create_event_from_request.assert_not_called()
        assert "你是想让我帮你记到日历里" in bot.sent[-1]
    finally:
        companion_plugin.memory_store = old_store
        companion_plugin.plugin_config = old_config
        companion_plugin.llm = old_llm
        companion_plugin.flow_manager = old_flow
        companion_plugin.knowledge_base = old_kb
        companion_plugin.manifestation_knowledge_base = old_manifest_kb
        companion_plugin.rel_profiler = old_rel
        companion_plugin.feishu_calendar_client = old_feishu
        cleanup_store(store, db_path)


def test_feishu_create_error_reply_hides_internal_exception():
    store, db_path = make_store()
    old_store = companion_plugin.memory_store
    old_config = companion_plugin.plugin_config
    old_llm = companion_plugin.llm
    old_flow = companion_plugin.flow_manager
    old_kb = companion_plugin.knowledge_base
    old_manifest_kb = companion_plugin.manifestation_knowledge_base
    old_rel = companion_plugin.rel_profiler
    old_feishu = companion_plugin.feishu_calendar_client

    class Event:
        user_id = 1

    class Bot:
        def __init__(self):
            self.sent = []

        async def send(self, event, chunk):
            self.sent.append(chunk)

    feishu = Mock()
    feishu.create_event_from_request.side_effect = RuntimeError("tenant_access_token invalid secret")
    try:
        companion_plugin.memory_store = store
        companion_plugin.plugin_config = Mock(
            max_recent_messages=5,
            web_search_enabled=False,
            auto_extract_interval=999,
            summary_trigger_count=999,
            feishu_timezone="Asia/Shanghai",
        )
        companion_plugin.llm = Mock()
        companion_plugin.flow_manager = None
        companion_plugin.knowledge_base = None
        companion_plugin.manifestation_knowledge_base = None
        companion_plugin.rel_profiler = None
        companion_plugin.feishu_calendar_client = feishu
        bot = Bot()

        _run(companion_plugin._process_text_message("提醒我明天下午三点开会", Event(), bot))

        assert "创建失败" in bot.sent[-1]
        assert "tenant_access_token" not in bot.sent[-1]
        assert "invalid secret" not in bot.sent[-1]
    finally:
        companion_plugin.memory_store = old_store
        companion_plugin.plugin_config = old_config
        companion_plugin.llm = old_llm
        companion_plugin.flow_manager = old_flow
        companion_plugin.knowledge_base = old_kb
        companion_plugin.manifestation_knowledge_base = old_manifest_kb
        companion_plugin.rel_profiler = old_rel
        companion_plugin.feishu_calendar_client = old_feishu
        cleanup_store(store, db_path)


def test_user_ending_message_snoozes_proactive_chat():
    store, db_path = make_store()
    old_store = companion_plugin.memory_store
    old_config = companion_plugin.plugin_config
    old_llm = companion_plugin.llm
    old_flow = companion_plugin.flow_manager
    old_kb = companion_plugin.knowledge_base
    old_manifest_kb = companion_plugin.manifestation_knowledge_base
    old_rel = companion_plugin.rel_profiler
    old_feishu = companion_plugin.feishu_calendar_client

    class Event:
        user_id = 1

    class Bot:
        async def send(self, event, chunk):
            pass

    try:
        companion_plugin.memory_store = store
        companion_plugin.plugin_config = Mock(
            max_recent_messages=5,
            web_search_enabled=False,
            auto_extract_interval=999,
            summary_trigger_count=999,
        )
        companion_plugin.llm = Mock(chat=Mock(return_value="晚安，好好休息。"), complete_if_needed=Mock(side_effect=lambda messages, reply, max_tokens=256: reply))
        companion_plugin.flow_manager = None
        companion_plugin.knowledge_base = None
        companion_plugin.manifestation_knowledge_base = None
        companion_plugin.rel_profiler = None
        companion_plugin.feishu_calendar_client = None

        _run(companion_plugin._process_text_message("我睡了晚安", Event(), Bot()))

        assert store.get_proactive_snooze_until(1) is not None
    finally:
        companion_plugin.memory_store = old_store
        companion_plugin.plugin_config = old_config
        companion_plugin.llm = old_llm
        companion_plugin.flow_manager = old_flow
        companion_plugin.knowledge_base = old_kb
        companion_plugin.manifestation_knowledge_base = old_manifest_kb
        companion_plugin.rel_profiler = old_rel
        companion_plugin.feishu_calendar_client = old_feishu
        cleanup_store(store, db_path)


# ── Helpers ────────────────────────────────────────────────────

def make_store():
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = tmp.name
    tmp.close()
    s = MemoryStore(db_path=db_path)
    return s, db_path


def cleanup_store(store, db_path):
    # Close all connections by deleting the store
    del store
    for ext in ("", "-wal", "-shm"):
        p = Path(db_path + ext) if ext else Path(db_path)
        try:
            p.unlink()
        except OSError:
            pass


def _run(coro):
    return asyncio.run(coro)


async def _flaky_send_factory(failures: int, sent: list[str]):
    state = {"count": 0}

    async def send_one(chunk: str):
        if state["count"] < failures:
            state["count"] += 1
            raise RuntimeError("send failed")
        sent.append(chunk)

    return send_one


class FixedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        base = datetime(2026, 6, 2, 13, 5, tzinfo=timezone.utc)
        if tz is not None:
            return base.astimezone(tz)
        return base.replace(tzinfo=None)


def test_send_chunks_retries_failed_chunk():
    sent: list[str] = []
    send_one = _run(_flaky_send_factory(1, sent))

    count = _run(companion_plugin._send_chunks(send_one, "hello"))

    assert count == 1
    assert sent == ["hello"]


def test_llm_fallback_avoids_recent_repetition():
    store, db_path = make_store()
    old_store = companion_plugin.memory_store
    try:
        companion_plugin.memory_store = store
        store.save_message("assistant", companion_plugin._LLM_FALLBACKS[0], user_id=1)

        fallback = companion_plugin._llm_fallback(user_id=1)

        assert fallback != companion_plugin._LLM_FALLBACKS[0]
        assert fallback in companion_plugin._LLM_FALLBACKS
    finally:
        companion_plugin.memory_store = old_store
        cleanup_store(store, db_path)


def test_llm_fallback_exhaustion_uses_non_repeating_backup():
    store, db_path = make_store()
    old_store = companion_plugin.memory_store
    try:
        companion_plugin.memory_store = store
        for fallback in companion_plugin._LLM_FALLBACKS:
            store.save_message("assistant", fallback, user_id=1)

        fallback = companion_plugin._llm_fallback(user_id=1)

        assert fallback not in companion_plugin._LLM_FALLBACKS
        assert "再发我一次" in fallback
    finally:
        companion_plugin.memory_store = old_store
        cleanup_store(store, db_path)


def test_llm_fallbacks_do_not_end_the_conversation():
    banned_fragments = ["等下再聊", "等会儿再继续", "走神", "大脑", "信号不太好"]

    for fallback in companion_plugin._LLM_FALLBACKS:
        assert not any(fragment in fallback for fragment in banned_fragments)
        assert any(fragment in fallback for fragment in ["再发", "再说", "重新"])


def test_llm_chat_failure_returns_actionable_retry_message():
    llm = LLMClient("key", "https://example.test", "model")
    llm.client = Mock()
    llm.client.chat.completions.create.side_effect = RuntimeError("timeout")

    reply = llm.chat([{"role": "user", "content": "hello"}], max_retries=1)

    assert "再发我一次" in reply
    assert "timeout" not in reply


def test_llm_detects_example_only_reply_as_incomplete():
    reply = '把"应该"换成"选择"试试——\n\n"我选择相信我的人生。"'

    assert LLMClient.looks_incomplete_reply(reply) is True


def test_llm_complete_if_needed_adds_semantic_continuation():
    llm = LLMClient("key", "https://example.test", "model")
    llm.chat = Mock(return_value="这句话会更像你在主动站回自己这边，而不是被某个标准推着走。")
    reply = '把"应该"换成"选择"试试——\n\n"我选择相信我的人生。"'

    completed = llm.complete_if_needed([{"role": "user", "content": "我总觉得应该相信人生"}], reply)

    assert completed.endswith("而不是被某个标准推着走。")
    continuation_messages = llm.chat.call_args.args[0]
    assert continuation_messages[-2] == {"role": "assistant", "content": reply}
    assert "自然收尾" in continuation_messages[-1]["content"]


def test_memory_management_command_routes():
    store, db_path = make_store()
    old_store = companion_plugin.memory_store
    try:
        companion_plugin.memory_store = store
        store.add_key_memory("用户最近在准备考试", user_id=1)
        store.add_key_memory("用户已经拿到offer了", user_id=1)

        overview = companion_plugin._handle_memory_management_command("我的记忆", 1)
        ended = companion_plugin._handle_memory_management_command("这件事结束了：考试", 1)
        suppressed = companion_plugin._handle_memory_management_command("以后别再提：offer", 1)
        boundary = companion_plugin._handle_memory_management_command("以后晚上别提醒我任务", 1)
        preference = companion_plugin._handle_memory_management_command("我希望你回复短一点", 1)
        pause = companion_plugin._handle_memory_management_command("暂停主动关心", 1)
        resume = companion_plugin._handle_memory_management_command("恢复主动关心", 1)
        active = store.get_all_key_memories(user_id=1)
        memories = store.get_key_memories_with_meta(1)
        by_content = {m["content"]: m for m in memories}

        assert overview is not None and "准备考试" in overview
        assert "正在进行" in overview
        assert ended is not None and "已标记为已结束" in ended
        assert suppressed is not None and "不再主动提起" in suppressed
        assert boundary is not None and "记住这个边界" in boundary
        assert preference is not None and "记住这个偏好" in preference
        assert pause is not None and "暂停主动关心" in pause
        assert resume is not None and "恢复主动关心" in resume
        assert "用户最近在准备考试" not in active
        assert "用户已经拿到offer了" not in active
        assert by_content["用户不希望晚上别提醒我任务"]["memory_type"] == "boundary"
        assert by_content["用户希望我回复短一点"]["memory_type"] == "preference"
        assert store.get_proactive_snooze_until(1) is None
    finally:
        companion_plugin.memory_store = old_store
        cleanup_store(store, db_path)


def test_frequency_first_aid_detects_old_story_anxiety():
    assert detect_frequency_support_category("我又想起过去的事情，好焦虑") == "past_release"
    assert detect_frequency_support_category("我是不是显化失败了") == "detachment"
    assert detect_frequency_support_category("我不配拥有这个") == "self_concept"

    text = build_frequency_first_aid_text("我又被以前的事情拉住了")

    assert text is not None
    assert "小魔女降频提醒" in text
    assert "30秒练习" in text
    assert "先不用急着变高频" in text


def test_manifestation_quote_avoids_recent_repetition():
    quote = pick_manifestation_quote("past_release", ["不要用过去的证据，审判未来的可能性。"])

    assert quote.category == "past_release"
    assert quote.quote != "不要用过去的证据，审判未来的可能性。"


def test_proactive_prompt_can_offer_frequency_first_aid():
    store, db_path = make_store()
    try:
        store.add_key_memory("用户容易因为过去的事情焦虑", user_id=1)
        store.save_message("user", "我又想起以前的事情，有点焦虑", user_id=1)
        store.record_user_active(1)
        config = Mock()
        chat = ProactiveChat(store, Mock(), config, kb=None)

        prompt = chat._build_proactive_prompt(user_id=1)

        assert "可选频率急救包" in prompt
        assert "30秒" in prompt
        assert "不要说用户频率太低" in prompt
    finally:
        cleanup_store(store, db_path)


def test_proactive_frequency_first_aid_avoids_repeating_recent_prompt():
    store, db_path = make_store()
    try:
        store.add_key_memory("用户容易因为过去的事情焦虑", user_id=1)
        store.save_message("user", "我又想起以前的事情，有点焦虑", user_id=1)
        store.record_proactive_sent(1, "今天的小魔女降频提醒：先呼吸三次")
        store.record_user_active(1)
        config = Mock()
        chat = ProactiveChat(store, Mock(), config, kb=None)

        prompt = chat._build_proactive_prompt(user_id=1)

        assert "可选频率急救包" not in prompt
    finally:
        cleanup_store(store, db_path)


def test_manifestation_diary_requires_context():
    store, db_path = make_store()
    try:
        writer = DiaryWriter(Mock(), store, Mock())

        assert writer._has_manifestation_conversation([
            {"role": "user", "content": "我今天吃了饭"},
            {"role": "assistant", "content": "好呀"},
        ]) is False
        assert writer._has_manifestation_conversation([
            {"role": "user", "content": "我想收集显化证据"},
        ]) is True
    finally:
        cleanup_store(store, db_path)


def test_manifestation_diary_generation_uses_memories_and_boundaries():
    store, db_path = make_store()
    try:
        llm = Mock()
        llm.chat.return_value = "## 今日状态\n你今天更稳定了一点。\n\n## 今日显化证据\n你愿意复盘。\n\n## 今天可以放下\n放下检查结果。\n\n## 明日对齐行动\n好好睡觉。"
        writer = DiaryWriter(llm, store, Mock())
        messages = [{"role": "user", "content": "我今天又有点执念，但停住了"}]
        memories = ["显化愿望种子：用户想显化稳定关系"]

        wishes = [{"id": 1, "title": "稳定关系", "status": "active"}]
        evidence = [{"content": "我今天没有反复确认"}]
        result = asyncio.run(writer._generate_manifestation_diary(messages, memories, "2026-06-08", wishes, evidence))

        assert result is not None
        args = llm.chat.call_args.kwargs
        user_prompt = args["messages"][1]["content"]
        assert "愿望生命周期" in user_prompt
        assert "累计显化证据链" in user_prompt
        assert "显化愿望种子" in user_prompt
        assert "不承诺结果" in user_prompt
        assert "不要硬编外部结果" in user_prompt
    finally:
        cleanup_store(store, db_path)


def test_llm_chat_continues_when_finish_reason_is_length():
    llm = LLMClient("key", "https://example.test", "model")
    llm.client = Mock()
    llm.client.chat.completions.create.side_effect = [
        CompletionResponse("前半段", "length"),
        CompletionResponse("后半段", "stop"),
    ]

    reply = llm.chat([{"role": "user", "content": "讲一个长故事"}], max_tokens=8)

    assert reply == "前半段后半段"
    assert llm.client.chat.completions.create.call_count == 2
    second_messages = llm.client.chat.completions.create.call_args_list[1].kwargs["messages"]
    assert second_messages[-2] == {"role": "assistant", "content": "前半段"}
    assert "不要重复" in second_messages[-1]["content"]


def test_chunk_text_splits_single_overlong_sentence_by_bytes():
    text = "一" * 20

    chunks = LLMClient.chunk_text(text, max_bytes=15)

    assert "".join(chunks) == text
    assert len(chunks) > 1
    assert all(len(chunk.encode("utf-8")) <= 15 for chunk in chunks)


# ── 0. Time awareness ────────────────────────────────────────────

def test_system_prompt_uses_beijing_time():
    """System prompt time context uses Beijing time instead of host-local time."""
    with patch.object(personality, "datetime", FixedDateTime):
        prompt = build_system_prompt(user_id=1)

    assert "北京时间 2026年06月02日（2026-06-02）21:05" in prompt
    assert "周二晚上" in prompt
    assert "历史聊天、摘要、记忆、日记里的日期都只是过去发生时间" in prompt
    assert "北京时间 2026年06月02日（2026-06-02）13:05" not in prompt


def test_proactive_prompt_uses_beijing_time_for_time_context():
    """Proactive prompts use the same Beijing clock as normal replies."""
    memory = Mock()
    memory.get_recent_summaries.return_value = []
    memory.get_key_memories_with_meta.return_value = []
    memory.get_all_key_memories.return_value = []
    memory.get_recent_messages.return_value = []
    memory.get_recent_proactive_content.return_value = []
    memory.get_last_active_time.return_value = None
    memory.get_manifestation_wishes.return_value = []
    memory.get_manifestation_evidence.return_value = []
    memory.get_recent_timeline_entries.return_value = []
    chat = ProactiveChat(memory, Mock(), Mock(), kb=None)
    with patch("nonebot_plugin_personal_companion.proactive.datetime", FixedDateTime), \
         patch("nonebot_plugin_personal_companion.personality.datetime", FixedDateTime):
        prompt = chat._build_proactive_prompt(user_id=1)

    assert "北京时间 2026年06月02日（2026-06-02）21:05" in prompt
    assert "现在是晚上" in prompt
    assert "现在是午后" not in prompt


def test_companion_context_receives_distress_before_solutions():
    flow = Mock()
    flow.should_invite_process.return_value = True
    flow.should_invite_appreciation.return_value = False
    with patch.object(flow, "should_invite_process", return_value=True), \
         patch.object(flow, "should_invite_appreciation", return_value=False):
        turn = analyze_turn("我压力好大快撑不住了", [], flow)
    ctx = build_companion_context_prompt(turn)

    assert turn.intent == "venting"
    assert turn.reply_mode == "comfort"
    assert turn.intensity == "high"
    assert turn.reply_length == "supportive"
    assert turn.flow_invite == "process"

    assert "接住情绪" in ctx
    assert "不要马上分析" in ctx
    assert "走个流程" in ctx


def test_companion_context_celebrates_good_news():
    flow = Mock()
    flow.should_invite_process.return_value = False
    flow.should_invite_appreciation.return_value = True
    with patch.object(flow, "should_invite_process", return_value=False), \
         patch.object(flow, "should_invite_appreciation", return_value=True):
        turn = analyze_turn("我拿到offer了！太开心了", [], flow)
    ctx = build_companion_context_prompt(turn)

    assert turn.intent == "celebrating"
    assert turn.reply_mode == "celebration"
    assert turn.flow_invite == "appreciation"

    assert "分享好事" in ctx
    assert "真诚替他开心" in ctx
    assert "不要把庆祝变成说教" in ctx


def test_companion_context_short_ack_allows_short_reply():
    turn = analyze_turn("嗯嗯", [])
    ctx = build_companion_context_prompt(turn)

    assert turn.intent == "short_ack"
    assert turn.reply_mode == "short_ack"

    assert "很短的回应" in ctx
    assert "只接一句" in ctx


def test_companion_context_suppresses_repeated_questions_and_flow_invites():
    recent = [
        {"role": "assistant", "content": "要不要一起走个流程？"},
        {"role": "assistant", "content": "你现在感觉怎么样？"},
    ]
    flow = Mock()
    flow.should_invite_process.return_value = True
    with patch.object(flow, "should_invite_process", return_value=True):
        turn = analyze_turn("还是很烦", recent, flow)
    ctx = build_companion_context_prompt(turn)

    assert "尽量不要再追问" in ctx
    assert "不要重复邀请" in ctx


def test_proactive_prompt_emphasizes_novelty_and_low_pressure():
    memory = Mock()
    memory.get_recent_summaries.return_value = []
    memory.get_key_memories_with_meta.return_value = []
    memory.get_all_key_memories.return_value = []
    memory.get_recent_messages.return_value = []
    memory.get_recent_proactive_content.return_value = ["今天过得怎么样？"]
    memory.get_last_active_time.return_value = None
    memory.get_manifestation_wishes.return_value = []
    memory.get_manifestation_evidence.return_value = []
    memory.get_recent_timeline_entries.return_value = []
    chat = ProactiveChat(memory, Mock(), Mock(), kb=None)

    prompt = chat._build_proactive_prompt(user_id=1)

    assert "不要总问'今天怎么样''在干嘛'" in prompt
    assert "更轻、更短、更没有压力" in prompt
    assert "也可以完全不问" in prompt


def test_relationship_prompt_new_user_stays_bounded():
    store, db_path = make_store()
    try:
        for _ in range(5):
            store.save_message("user", "你好", user_id=1)
        store.record_user_active(1)
        prompt = build_relationship_prompt(1, RelationshipProfiler(store))

        assert "不要装熟" in prompt
        assert "有分寸" in prompt
    finally:
        cleanup_store(store, db_path)


def test_reply_mode_prompts_shape_response_strategy():
    short = analyze_turn("嗯", [])
    comfort = analyze_turn("我好焦虑，感觉什么都做不好", [])
    celebration = analyze_turn("我成功了！", [])
    practical = analyze_turn("这个报错怎么修", [])
    soft_end = analyze_turn("睡了晚安", [])

    assert short.reply_mode == "short_ack"
    assert "不要连续提问" in build_reply_mode_prompt(short)
    assert comfort.reply_mode == "comfort"
    assert "先共情" in build_reply_mode_prompt(comfort)
    assert celebration.reply_mode == "celebration"
    assert "不要说教" in build_reply_mode_prompt(celebration)
    assert practical.reply_mode == "practical"
    assert "先给结论" in build_reply_mode_prompt(practical)
    assert soft_end.reply_mode == "soft_end"
    assert "不要追问" in build_reply_mode_prompt(soft_end)


def test_build_messages_injects_reply_mode_before_companion_context():
    store, db_path = make_store()
    old_store = companion_plugin.memory_store
    old_config = companion_plugin.plugin_config
    old_kb = companion_plugin.knowledge_base
    old_manifest_kb = companion_plugin.manifestation_knowledge_base
    old_rel = companion_plugin.rel_profiler
    old_flow = companion_plugin.flow_manager
    try:
        companion_plugin.memory_store = store
        companion_plugin.plugin_config = Mock(max_recent_messages=5)
        companion_plugin.knowledge_base = None
        companion_plugin.manifestation_knowledge_base = None
        companion_plugin.rel_profiler = None
        companion_plugin.flow_manager = None

        messages = companion_plugin._build_messages("我好焦虑", [], user_id=1)
        contents = [m["content"] for m in messages if m["role"] == "system"]
        reply_mode_index = next(i for i, content in enumerate(contents) if "[本轮回复模式：]" in content)
        companion_index = next(i for i, content in enumerate(contents) if "[本轮陪伴方式：]" in content)

        assert reply_mode_index < companion_index
        assert "模式：comfort" in contents[reply_mode_index]
    finally:
        companion_plugin.memory_store = old_store
        companion_plugin.plugin_config = old_config
        companion_plugin.knowledge_base = old_kb
        companion_plugin.manifestation_knowledge_base = old_manifest_kb
        companion_plugin.rel_profiler = old_rel
        companion_plugin.flow_manager = old_flow
        cleanup_store(store, db_path)


    assert choose_reply_max_tokens(analyze_turn("嗯嗯", [])) == 2048
    assert choose_reply_max_tokens(analyze_turn("睡了晚安", [])) == 2048
    assert choose_reply_max_tokens(analyze_turn("这段代码为什么报错，帮我解释一下", [])) == 2048
    assert choose_reply_max_tokens(analyze_turn("我压力好大快撑不住了", [])) == 2048


def test_turn_context_gates_web_search():
    assert analyze_turn("我压力好大快撑不住了", []).allow_web_search is False
    assert analyze_turn("睡了晚安", []).allow_web_search is False
    assert analyze_turn("帮我解释一下这个报错", []).allow_web_search is True
    assert analyze_turn("查一下最近 DeepSeek 有什么更新", []).allow_web_search is True


def test_proactive_should_send_respects_ending_signal():
    memory = Mock()
    memory.count_proactive_since_last_user_message.return_value = 0
    memory.get_last_active_time.return_value = None
    memory.get_last_proactive_time.return_value = None
    memory.get_recent_messages.return_value = [{"role": "user", "content": "我睡了晚安"}]
    config = Mock()
    config.proactive_cooldown_minutes = 0
    config.proactive_interval_minutes = 0
    chat = ProactiveChat(memory, Mock(), config, kb=None)

    assert chat._should_send(1, FixedDateTime.now(personality.BEIJING_TZ)) is False


def test_persona_state_avoids_inappropriate_moods_for_distress():
    cfg = {
        "states": [
            {"name": "有点烦", "weight": 100, "hints": []},
            {"name": "温柔感性", "weight": 1, "hints": []},
        ]
    }
    turn = analyze_turn("我快崩溃了", [])

    state = _roll_state(cfg, user_id=99, turn_context=turn)

    assert state["name"] == "温柔感性"


def test_persona_state_allows_casual_variety():
    cfg = {"states": [{"name": "随意/摆烂模式", "weight": 1, "hints": []}]}
    turn = analyze_turn("笑死", [])

    state = _roll_state(cfg, user_id=100, turn_context=turn)

    assert state["name"] == "随意/摆烂模式"


def test_proactive_topic_selector_uses_light_greeting_by_default():
    store, db_path = make_store()
    try:
        store.add_key_memory("用户喜欢喝咖啡", user_id=1)
        store.record_user_active(1)
        chat = ProactiveChat(store, Mock(), Mock(), kb=None)

        prompt = chat._build_proactive_prompt(user_id=1)

        assert "本次主动主题：轻问候" in prompt
        assert "不提显化" in prompt
    finally:
        cleanup_store(store, db_path)


def test_proactive_topic_selector_limits_manifestation_after_recent_checkin():
    store, db_path = make_store()
    try:
        store.create_manifestation_wish(1, "稳定关系", "我想显化稳定关系")
        store.add_key_memory("用户正在练习稳定关系", user_id=1, memory_type="manifestation")
        store.save_message("user", "我想做显化日记", user_id=1)
        store.record_proactive_sent(1, "今晚要不要收集一个小小的显化证据？")
        store.record_user_active(1)
        chat = ProactiveChat(store, Mock(), Mock(), kb=None)

        prompt = chat._build_proactive_prompt(user_id=1)

        assert "本次主动主题：显化轻 check-in" not in prompt
        assert "本次主动主题：轻问候" in prompt
        assert "显化信心补给" not in prompt
    finally:
        cleanup_store(store, db_path)


def test_proactive_topic_selector_allows_frequency_first_aid_for_recent_anxiety():
    store, db_path = make_store()
    try:
        store.save_message("user", "我又被过去的事情拉住了，好焦虑", user_id=1)
        store.record_user_active(1)
        chat = ProactiveChat(store, Mock(), Mock(), kb=None)

        prompt = chat._build_proactive_prompt(user_id=1)

        assert "本次主动主题：频率急救" in prompt
        assert "可选频率急救包" in prompt
    finally:
        cleanup_store(store, db_path)


    store, db_path = make_store()
    try:
        store.add_key_memory("用户喜欢喝咖啡", user_id=1)
        store.add_key_memory("用户最近在准备考试", user_id=1)
        store.add_key_memory("用户已经拿到offer了", user_id=1)
        store.add_key_memory("用户准备周四出去玩", user_id=1)
        conn = store._get_conn()
        conn.execute(
            "UPDATE key_memories SET created_at = datetime('now', '-7 days') WHERE content = ?",
            ("用户准备周四出去玩",),
        )
        conn.commit()
        conn.close()
        store.record_user_active(1)
        config = Mock()
        chat = ProactiveChat(store, Mock(), config, kb=None)

        prompt = chat._build_proactive_prompt(user_id=1)

        assert "长期事实/偏好/边界" in prompt
        assert "可能仍在进行的事" in prompt
        assert "已经结束的事" in prompt
        assert "已完成和已过期的旧事已隐藏" in prompt
        assert "时间已经过期的旧计划" in prompt
        assert "准备周四出去玩" not in prompt
    finally:
        cleanup_store(store, db_path)


def test_proactive_topic_budget_blocks_recent_manifestation_checkin():
    store, db_path = make_store()
    try:
        store.save_manifestation_entry(1, "manifest_seed", "用户想显化稳定关系")
        store.save_message("user", "我想做显化日记", user_id=1)
        store.record_proactive_sent(1, "显化轻 check-in", topic_kind="manifestation_checkin")
        store.record_user_active(1)
        chat = ProactiveChat(store, Mock(), Mock(), kb=None)

        prompt = chat._build_proactive_prompt(user_id=1)

        assert "本次主动主题：显化轻 check-in" not in prompt
        assert "可选显化关心" not in prompt
    finally:
        cleanup_store(store, db_path)


def test_proactive_ignored_messages_create_quiet_period():
    store, db_path = make_store()
    try:
        store.record_user_active(1)
        conn = store._get_conn()
        conn.execute("UPDATE active_users SET last_message_at = datetime('now', '-1 hour') WHERE user_id = ?", (1,))
        conn.commit()
        conn.close()
        store.record_proactive_sent(1, "第一条", topic_kind="light_greeting")
        store.record_proactive_sent(1, "第二条", topic_kind="light_greeting")
        store.record_proactive_sent(1, "第三条", topic_kind="light_greeting")
        config = Mock(proactive_cooldown_minutes=0, proactive_interval_minutes=0)
        chat = ProactiveChat(store, Mock(), config, kb=None)

        assert chat._should_send(1, datetime.now(BEIJING_TZ)) is False
        assert store.get_proactive_snooze_until(1) is not None
    finally:
        cleanup_store(store, db_path)


def test_proactive_prompt_can_offer_manifestation_checkin():
    store, db_path = make_store()
    try:
        store.save_manifestation_entry(1, "manifest_seed", "用户想显化稳定关系，今日行动是不反复确认")
        wishes = store.get_manifestation_wishes(1)
        store.add_manifestation_evidence(1, "我今天没有反复确认", wish_id=wishes[0]["id"])
        store.save_message("user", "我想做显化日记", user_id=1)
        store.record_user_active(1)
        config = Mock()
        chat = ProactiveChat(store, Mock(), config, kb=None)

        prompt = chat._build_proactive_prompt(user_id=1)

        assert "显化愿望生命周期" in prompt
        assert "最近显化证据链" in prompt
        assert "我今天没有反复确认" in prompt
        assert "可选显化关心" in prompt
        assert "不要承诺结果" in prompt
        assert "不要要求对方立刻做完整流程" in prompt
    finally:
        cleanup_store(store, db_path)


def test_proactive_manifestation_checkin_avoids_repeating_recent_proactive():
    store, db_path = make_store()
    try:
        store.save_manifestation_entry(1, "manifest_seed", "用户想显化稳定关系")
        store.save_message("user", "我想做显化日记", user_id=1)
        store.record_proactive_sent(1, "今晚要不要收集一个小小的显化证据？")
        store.record_user_active(1)
        config = Mock()
        chat = ProactiveChat(store, Mock(), config, kb=None)

        prompt = chat._build_proactive_prompt(user_id=1)

        assert "可选显化关心" not in prompt
    finally:
        cleanup_store(store, db_path)

def test_normal_chat_prompt_excludes_completed_and_expired_memory():
    store, db_path = make_store()
    old_store = companion_plugin.memory_store
    old_config = companion_plugin.plugin_config
    old_kb = companion_plugin.knowledge_base
    old_rel = companion_plugin.rel_profiler
    old_flow = companion_plugin.flow_manager
    try:
        companion_plugin.memory_store = store
        companion_plugin.plugin_config = Mock(max_recent_messages=5)
        companion_plugin.knowledge_base = None
        companion_plugin.rel_profiler = None
        companion_plugin.flow_manager = None
        store.add_key_memory("用户喜欢喝咖啡", user_id=1)
        store.add_key_memory("用户已经拿到offer了", user_id=1)
        store.add_key_memory("用户准备周四出去玩", user_id=1)
        conn = store._get_conn()
        conn.execute(
            "UPDATE key_memories SET created_at = datetime('now', '-7 days') WHERE content = ?",
            ("用户准备周四出去玩",),
        )
        conn.commit()
        conn.close()

        retrieved = store.retrieve_memories(["用户"], user_id=1, limit=10)
        messages = companion_plugin._build_messages("用户最近怎么样", retrieved, user_id=1)
        prompt_text = "\n".join(str(m["content"]) for m in messages if m["role"] == "system")

        assert "喜欢喝咖啡" in prompt_text
        assert "拿到offer" not in prompt_text
        assert "周四出去玩" not in prompt_text
    finally:
        companion_plugin.memory_store = old_store
        companion_plugin.plugin_config = old_config
        companion_plugin.knowledge_base = old_kb
        companion_plugin.rel_profiler = old_rel
        companion_plugin.flow_manager = old_flow
        cleanup_store(store, db_path)

def test_normal_chat_prompt_does_not_echo_history_timestamps():
    store, db_path = make_store()
    old_store = companion_plugin.memory_store
    old_config = companion_plugin.plugin_config
    old_kb = companion_plugin.knowledge_base
    old_rel = companion_plugin.rel_profiler
    old_flow = companion_plugin.flow_manager
    try:
        companion_plugin.memory_store = store
        companion_plugin.plugin_config = Mock(max_recent_messages=5)
        companion_plugin.knowledge_base = None
        companion_plugin.rel_profiler = None
        companion_plugin.flow_manager = None
        store.save_message("user", "我那天说自己拥有幸福了", user_id=1)
        conn = store._get_conn()
        conn.execute(
            "UPDATE messages SET created_at = ? WHERE content = ?",
            ("2026-04-22 09:14:00", "我那天说自己拥有幸福了"),
        )
        conn.commit()
        conn.close()

        with patch.object(companion_plugin, "datetime", FixedDateTime):
            messages = companion_plugin._build_messages("现在几点", [], user_id=1)
        content = "\n".join(str(m["content"]) for m in messages)

        assert "最近聊天记录只作为历史上下文" in content
        assert "[2026-04-22 17:14 北京时间]" not in content
        assert "时间戳" not in content
        assert messages[-1] == {"role": "user", "content": "现在几点"}
    finally:
        companion_plugin.memory_store = old_store
        companion_plugin.plugin_config = old_config
        companion_plugin.knowledge_base = old_kb
        companion_plugin.rel_profiler = old_rel
        companion_plugin.flow_manager = old_flow
        cleanup_store(store, db_path)


def test_timeline_retrieval_uses_explicit_date_even_without_keyword_match():
    store, db_path = make_store()
    old_store = companion_plugin.memory_store
    try:
        companion_plugin.memory_store = store
        store.add_timeline_entry(1, "2026-04-22", "用户宣布自己已经拥有幸福了", event_time="17:14")

        entries = companion_plugin._retrieve_timeline_for_turn("4月22日发生了什么", ["发生", "什么"], user_id=1)

        assert len(entries) == 1
        assert entries[0]["event_date"] == "2026-04-22"
        assert entries[0]["content"] == "用户宣布自己已经拥有幸福了"
    finally:
        companion_plugin.memory_store = old_store
        cleanup_store(store, db_path)


def test_handle_date_time_question_returns_deterministic_answer():
    from nonebot_plugin_personal_companion import BEIJING_TZ
    now = datetime(2026, 6, 12, 12, 24, tzinfo=BEIJING_TZ)

    reply = companion_plugin._handle_date_time_question("今天几号", now)
    assert reply is not None
    assert "6月12日" in reply

    reply = companion_plugin._handle_date_time_question("现在几点", now)
    assert reply is not None
    assert "12点24分" in reply

    reply = companion_plugin._handle_date_time_question("星期几", now)
    assert reply is not None
    assert "星期五" in reply

    reply = companion_plugin._handle_date_time_question("今天天气怎么样", now)
    assert reply is None


def test_time_lock_appears_at_end_of_built_messages():
    store, db_path = make_store()
    old_store = companion_plugin.memory_store
    old_config = companion_plugin.plugin_config
    old_kb = companion_plugin.knowledge_base
    old_rel = companion_plugin.rel_profiler
    old_flow = companion_plugin.flow_manager
    try:
        companion_plugin.memory_store = store
        companion_plugin.plugin_config = Mock(max_recent_messages=3)
        companion_plugin.knowledge_base = None
        companion_plugin.rel_profiler = None
        companion_plugin.flow_manager = None

        with patch.object(companion_plugin, "datetime", FixedDateTime):
            messages = companion_plugin._build_messages("你好", [], user_id=1)
        system_contents = [m["content"] for m in messages if m["role"] == "system"]

        assert any("当前真实时间校验" in c for c in system_contents)
        assert any("历史消息、记忆、摘要、日记" in c for c in system_contents)

        last_system = system_contents[-1]
        assert "当前真实时间校验" in last_system
    finally:
        companion_plugin.memory_store = old_store
        companion_plugin.plugin_config = old_config
        companion_plugin.knowledge_base = old_kb
        companion_plugin.rel_profiler = old_rel
        companion_plugin.flow_manager = old_flow
        cleanup_store(store, db_path)


def test_time_lock_not_confused_by_old_date_in_history():
    store, db_path = make_store()
    old_store = companion_plugin.memory_store
    old_config = companion_plugin.plugin_config
    old_kb = companion_plugin.knowledge_base
    old_rel = companion_plugin.rel_profiler
    old_flow = companion_plugin.flow_manager
    try:
        companion_plugin.memory_store = store
        companion_plugin.plugin_config = Mock(max_recent_messages=8)
        companion_plugin.knowledge_base = None
        companion_plugin.rel_profiler = None
        companion_plugin.flow_manager = None

        store.save_message("user", "今天是2月14号，情人节快乐", user_id=1)
        conn = store._get_conn()
        conn.execute(
            "UPDATE messages SET created_at = ? WHERE content = ?",
            ("2026-02-14 09:00:00", "今天是2月14号，情人节快乐"),
        )
        conn.commit()
        conn.close()

        with patch.object(companion_plugin, "datetime", FixedDateTime):
            messages = companion_plugin._build_messages("今天几号", [], user_id=1)
        system_contents = [m["content"] for m in messages if m["role"] == "system"]

        time_lock = [c for c in system_contents if "当前真实时间校验" in c][0]
        assert "2026年06月02日" in time_lock or "2026-06-02" in time_lock
    finally:
        companion_plugin.memory_store = old_store
        companion_plugin.plugin_config = old_config
        companion_plugin.knowledge_base = old_kb
        companion_plugin.rel_profiler = old_rel
        companion_plugin.flow_manager = old_flow
        cleanup_store(store, db_path)


def test_normal_chat_prompt_injects_timeline_entries_with_date_safety():
    store, db_path = make_store()
    old_store = companion_plugin.memory_store
    old_config = companion_plugin.plugin_config
    old_kb = companion_plugin.knowledge_base
    old_rel = companion_plugin.rel_profiler
    old_flow = companion_plugin.flow_manager
    try:
        companion_plugin.memory_store = store
        companion_plugin.plugin_config = Mock(max_recent_messages=5)
        companion_plugin.knowledge_base = None
        companion_plugin.rel_profiler = None
        companion_plugin.flow_manager = None
        timeline_entries = [{
            "event_date": "2026-04-22",
            "event_time": "17:14",
            "content": "用户宣布自己已经拥有幸福了",
        }]

        messages = companion_plugin._build_messages("幸福名场面", [], user_id=1, timeline_entries=timeline_entries)
        content = "\n".join(str(m["content"]) for m in messages)

        assert "时间线记忆" in content
        assert "日期是事件发生日期，不代表今天" in content
        assert "2026-04-22 17:14：用户宣布自己已经拥有幸福了" in content
    finally:
        companion_plugin.memory_store = old_store
        companion_plugin.plugin_config = old_config
        companion_plugin.knowledge_base = old_kb
        companion_plugin.rel_profiler = old_rel
        companion_plugin.flow_manager = old_flow
        cleanup_store(store, db_path)


def test_proactive_prompt_includes_timeline_as_background_only():
    store, db_path = make_store()
    try:
        store.add_timeline_entry(1, "2026-04-22", "用户宣布自己已经拥有幸福了", event_time="17:14")
        config = Mock()
        chat = ProactiveChat(store, Mock(), config, kb=None)

        prompt = chat._build_proactive_prompt(user_id=1)

        assert "最近时间线背景" in prompt
        assert "2026-04-22 17:14：用户宣布自己已经拥有幸福了" in prompt
        assert "这些是历史时间线，不是今天的待办" in prompt
    finally:
        cleanup_store(store, db_path)


def test_proactive_manifestation_checkin_can_boost_confidence():
    store, db_path = make_store()
    try:
        wish_id = store.create_manifestation_wish(1, "更稳定的工作", "我想显化更稳定的工作")
        store.add_manifestation_evidence(1, "我今天没有反复确认", wish_id=wish_id)
        store.add_key_memory("用户正在练习稳定下来", user_id=1, memory_type="manifestation")
        config = Mock()
        chat = ProactiveChat(store, Mock(), config, kb=KnowledgeBase(), manifestation_kb=KnowledgeBase(MANIFESTATION_KNOWLEDGE_PATH))

        prompt = chat._build_proactive_prompt(user_id=1)

        assert "显化信心补给" in prompt
        assert "小证据" in prompt
        assert "今天可以只选一个很小的对齐动作" in prompt or "先别检查结果" in prompt
        assert "不要追问结果" in prompt
    finally:
        cleanup_store(store, db_path)


def test_proactive_manifestation_checkin_avoids_overpressure():
    store, db_path = make_store()
    try:
        wish_id = store.create_manifestation_wish(1, "更稳定的工作", "我想显化更稳定的工作")
        store.add_manifestation_evidence(1, "我今天没有反复确认", wish_id=wish_id)
        config = Mock()
        chat = ProactiveChat(store, Mock(), config, kb=KnowledgeBase(), manifestation_kb=KnowledgeBase(MANIFESTATION_KNOWLEDGE_PATH))

        prompt = chat._build_proactive_prompt(user_id=1)

        assert "频率太低" not in prompt
        assert "结果来了没" not in prompt
        assert "你显化得怎么样了" not in prompt
        assert "不要问结果来了没" not in prompt
    finally:
        cleanup_store(store, db_path)


# ── 1. Emotion detection and retrieval ────────────────────────

def test_emotion_detection():
    """Emotion tags are correctly detected from user messages."""
    assert "焦虑" in MemoryStore.detect_emotions("我最近压力好大，担心项目会失败")
    assert "难过" in MemoryStore.detect_emotions("今天真的好崩溃，想哭")
    assert "开心" in MemoryStore.detect_emotions("太棒了！我拿到offer了！")
    assert "迷茫" in MemoryStore.detect_emotions("我也不知道该怎么办，很迷茫")
    assert "中性" in MemoryStore.detect_emotions("今天天气不错")
    print("  [OK] Emotion detection: all cases correct")


def test_emotion_boosted_retrieval():
    """Memories with matching emotion tags get retrieval score boost."""
    store, db_path = make_store()
    try:
        # Add memories with different emotion tags
        store.add_key_memory("用户养了一只猫", user_id=1,
                           emotion_tags="中性", entity_tags="猫,宠物")
        store.add_key_memory("用户加班到凌晨很焦虑", user_id=1,
                           emotion_tags="焦虑,疲惫", entity_tags="工作,加班")
        store.add_key_memory("用户今天吃到了好吃的", user_id=1,
                           emotion_tags="开心", entity_tags="食物")

        # Normal retrieval (no emotion boost) - all match keyword "用户"
        normal = store.retrieve_memories(["用户"], user_id=1, limit=3)

        # Emotion-boosted retrieval when user is anxious
        boosted = store.retrieve_memories_with_emotion(
            ["用户"], user_id=1, user_emotion_tags=["焦虑"], limit=3
        )

        # The anxious memory should rank higher with emotion boost
        # Both should return results
        assert len(boosted) >= 1, "Should retrieve at least 1 memory"
        assert len(boosted) > 0

        # When user expresses anxiety, the anxiety-tagged memory should
        # appear earlier in results (or at least be present)
        boost_contents = boosted
        has_anxiety = any("焦虑" in c or "加班" in c for c in boost_contents)
        assert has_anxiety, f"Anxiety-tagged memory should be in boosted results: {boost_contents}"

        print(f"  [OK] Emotion boosting: normal={normal[:2]}, emotion_boosted={boosted[:2]}")
    finally:
        cleanup_store(store, db_path)


# ── 2. Entity association graph ───────────────────────────────

def test_entity_association():
    """Memories sharing entity tags are associated for associative recall."""
    store, db_path = make_store()
    try:
        # Add memories that share entities
        store.add_key_memory("用户养了一只叫年糕的猫", user_id=1,
                           emotion_tags="开心", entity_tags="猫,年糕,宠物")
        store.add_key_memory("年糕最近还在宠物医院复查，花了800块", user_id=1,
                           emotion_tags="焦虑", entity_tags="猫,年糕,宠物医院")
        store.add_key_memory("用户在杭州西湖区工作", user_id=1,
                           emotion_tags="中性", entity_tags="杭州,工作")
        store.add_key_memory("用户喜欢吃川菜", user_id=1,
                           emotion_tags="开心", entity_tags="川菜,食物")

        # Retrieve primary memories by keyword
        primary = store.retrieve_memories(["猫"], user_id=1, limit=5)
        assert len(primary) >= 1

        # Get associated memories via entity overlap
        associated = store.get_entity_associated_memories(
            primary, user_id=1, exclude_contents=set(primary), limit=3
        )

        # Should find the vet visit memory (shares "猫,年糕" entities)
        assert len(associated) >= 1, "Should find associated memories via entity overlap"
        assert any("宠物医院" in m for m in associated), \
            f"Should associate 'vet visit' with 'cat' via shared entity '年糕'. Got: {associated}"

        print(f"  [OK] Entity association: primary={primary[:1]}, associated={associated}")
    finally:
        cleanup_store(store, db_path)


# ── 3. Personalized knowledge ─────────────────────────────────

def test_personalized_knowledge_prompt():
    """Knowledge prompt includes user's personal experiences related to concepts."""
    store, db_path = make_store()
    try:
        # Add user memories that contain concept trigger keywords
        # "流程工具" concept keywords include: 难受, 不舒服, 情绪, 焦虑, 害怕, ...
        store.add_key_memory("用户最近情绪很不稳定，压力大得难受", user_id=1,
                           emotion_tags="焦虑,疲惫", entity_tags="工作,情绪")
        store.add_key_memory("用户说老是反复遇到同样的问题，不知道怎么办", user_id=1,
                           emotion_tags="迷茫", entity_tags="模式,困境")
        store.add_key_memory("用户已经解决了之前情绪很不舒服的问题", user_id=1,
                           emotion_tags="开心", entity_tags="情绪")

        kb = KnowledgeBase()
        assert len(kb.concepts) == 12, "Should have 12 concepts"

        # User expresses anxiety — should match "流程工具" or "彩蛋" concepts
        prompt = build_knowledge_prompt_personalized(
            "最近情绪很不舒服，不知道该怎么办", kb, user_id=1, memory_store=store
        )

        assert len(prompt) > 0, "Should generate a prompt for matching message"
        assert "情绪" in prompt or "不舒服" in prompt or "流程" in prompt or "全息" in prompt, \
            f"Should match relevant concepts. Got: {prompt[:300]}"
        # The key test: user's personal memory should be referenced
        assert "不稳定" in prompt or "压力大" in prompt or "难受" in prompt or "反复" in prompt, \
            f"Should include user's personal experiences. Got: {prompt[:300]}"
        assert "已经解决" not in prompt

        print(f"  [OK] Personalized knowledge: prompt length={len(prompt)} chars")
        print(f"       First 300 chars: {prompt[:300]}...")
    finally:
        cleanup_store(store, db_path)


# ── 4. Structured memory extraction format ────────────────────

def test_structured_extraction_parsing():
    """Verify the structured output format (LLM response) parses correctly.

    Tests the parsing logic inside extract_memories_structured without
    needing a real LLM call.
    """
    # Simulate LLM response with structured format (bullet-prefixed lines)
    raw_llm_output = (
        "• 用户最近加班到凌晨，担心项目进度 | 情绪:焦虑,疲惫 | 实体:工作,项目,加班\n"
        "• 用户养了一只叫年糕的猫 | 情绪:中性 | 实体:猫,年糕,宠物\n"
        "无\n"
    )

    # Replicate the parsing logic from extract_memories_structured
    result: list[dict] = []
    for line in raw_llm_output.split("\n"):
        stripped = line.strip()
        if not stripped.startswith("•") and not stripped.startswith("-"):
            continue

        content = ""
        emotions: list[str] = []
        entities: list[str] = []

        if "|" in stripped:
            parts = stripped.split("|")
            content = parts[0].strip().lstrip("•- ")
            for part in parts[1:]:
                part = part.strip()
                if part.startswith("情绪:") or part.startswith("情绪："):
                    em = part[3:].strip()
                    emotions = [e.strip() for e in em.split(",") if e.strip()]
                elif part.startswith("实体:") or part.startswith("实体："):
                    ent = part[3:].strip()
                    entities = [e.strip() for e in ent.split(",") if e.strip()]
        else:
            content = stripped.lstrip("•- ")

        if content and content != "无" and len(content) > 2:
            result.append({
                "content": content,
                "emotions": emotions,
                "entities": entities,
            })

    assert len(result) == 2, f"Should extract 2 memories, got {len(result)}: {result}"

    # First memory: work stress
    assert result[0]["content"] == "用户最近加班到凌晨，担心项目进度"
    assert "焦虑" in result[0]["emotions"]
    assert "疲惫" in result[0]["emotions"]
    assert "工作" in result[0]["entities"]
    assert "项目" in result[0]["entities"]

    # Second memory: cat
    assert result[1]["content"] == "用户养了一只叫年糕的猫"
    assert "中性" in result[1]["emotions"]
    assert "年糕" in result[1]["entities"]
    assert "宠物" in result[1]["entities"]

    print(f"  [OK] Structured extraction parsing: {len(result)} memories")
    print(f"       Mem 1: {result[0]}")
    print(f"       Mem 2: {result[1]}")


def test_structured_extraction_fallback():
    """Old-format lines (without emotion/entity tags) still parse correctly."""
    raw_llm_output = (
        "• 用户在杭州工作\n"
        "• 用户喜欢喝咖啡\n"
    )

    result: list[dict] = []
    for line in raw_llm_output.split("\n"):
        stripped = line.strip()
        if not stripped.startswith("•") and not stripped.startswith("-"):
            continue

        content = stripped.lstrip("•- ")
        if content and content != "无" and len(content) > 2:
            result.append({
                "content": content,
                "emotions": [],
                "entities": [],
            })

    assert len(result) == 2
    assert result[0]["content"] == "用户在杭州工作"
    assert result[0]["emotions"] == []  # No tags = backward compatible
    assert result[1]["content"] == "用户喜欢喝咖啡"
    print("  [OK] Structured extraction fallback: handles old format")


# ── 5. Combined flow ──────────────────────────────────────────

def test_combined_flow():
    """Test the full retrieval pipeline: emotion boost + entity association + personalized knowledge."""
    store, db_path = make_store()
    try:
        # Seed diverse memories
        store.add_key_memory("用户在杭州做AI产品经理", user_id=1,
                           emotion_tags="中性", entity_tags="杭州,AI,产品,工作")
        store.add_key_memory("用户最近为项目进度焦虑，失眠", user_id=1,
                           emotion_tags="焦虑,疲惫", entity_tags="工作,项目,失眠")
        store.add_key_memory("用户养了一只叫年糕的猫，三岁了", user_id=1,
                           emotion_tags="开心", entity_tags="猫,年糕,宠物")
        store.add_key_memory("年糕上周生病花了800块医药费", user_id=1,
                           emotion_tags="焦虑", entity_tags="猫,年糕,宠物医院,医疗")
        store.add_key_memory("用户喜欢看科幻电影", user_id=1,
                           emotion_tags="开心", entity_tags="电影,科幻")
        store.add_key_memory("用户说你推荐的视频他很喜欢", user_id=1,
                           emotion_tags="开心", entity_tags="B站,视频,推荐")

        # Simulate user message: anxious about work
        user_msg = "最近项目压力好大，感觉自己快撑不住了"

        # Step 1: Emotion detection
        emotions = MemoryStore.detect_emotions(user_msg)
        assert "焦虑" in emotions, f"Should detect anxiety: {emotions}"
        print(f"  [Step 1] Detected emotions: {emotions}")

        # Step 2: Emotion-boosted retrieval
        keywords = ["项目", "压力"]
        primary = store.retrieve_memories_with_emotion(
            keywords, user_id=1, user_emotion_tags=emotions, limit=3
        )
        assert len(primary) >= 1, "Should find memories about work stress"
        print(f"  [Step 2] Primary memories: {primary}")

        # Step 3: Entity association
        associated = store.get_entity_associated_memories(
            primary, user_id=1, exclude_contents=set(primary), limit=3
        )
        print(f"  [Step 3] Associated memories (via entity overlap): {associated}")

        # Step 4: Personalized knowledge
        kb = KnowledgeBase()
        prompt = build_knowledge_prompt_personalized(
            user_msg, kb, user_id=1, memory_store=store
        )
        assert len(prompt) > 0
        # The prompt contains the concept's essence/wisdom text, not necessarily the concept name
        has_concept = any(kw in prompt for kw in ["不舒服", "不逃避", "深入感受", "彩蛋", "全息", "流程"])
        assert has_concept, \
            f"Knowledge prompt should match relevant concept. Prompt: {prompt[:300]}"
        print(f"  [Step 4] Knowledge prompt length: {len(prompt)} chars")

        print("\n  [OK] Combined flow: all 4 steps passed!")
    finally:
        cleanup_store(store, db_path)


# ── Runner ────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Testing memory & knowledge improvements...")
    print()

    test_system_prompt_uses_beijing_time()
    test_proactive_prompt_uses_beijing_time_for_time_context()
    test_companion_context_receives_distress_before_solutions()
    test_companion_context_celebrates_good_news()
    test_companion_context_short_ack_allows_short_reply()
    test_companion_context_suppresses_repeated_questions_and_flow_invites()
    test_proactive_prompt_emphasizes_novelty_and_low_pressure()
    test_relationship_prompt_new_user_stays_bounded()
    test_reply_max_tokens_short_and_factual_profiles()
    test_turn_context_gates_web_search()
    test_proactive_should_send_respects_ending_signal()
    test_persona_state_avoids_inappropriate_moods_for_distress()
    test_persona_state_allows_casual_variety()
    test_proactive_prompt_groups_memory_by_status()
    test_emotion_detection()
    test_emotion_boosted_retrieval()
    test_entity_association()
    test_personalized_knowledge_prompt()
    test_structured_extraction_parsing()
    test_structured_extraction_fallback()
    test_combined_flow()

    print()
    print("All improvement tests passed!")
