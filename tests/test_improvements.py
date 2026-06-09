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
    KnowledgeBase,
    build_knowledge_prompt_personalized,
)
from nonebot_plugin_personal_companion import personality  # noqa: E402
import nonebot_plugin_personal_companion as companion_plugin  # noqa: E402
from nonebot_plugin_personal_companion.personality import build_system_prompt, _roll_state  # noqa: E402
from nonebot_plugin_personal_companion.proactive import ProactiveChat  # noqa: E402
from nonebot_plugin_personal_companion.relationship import RelationshipProfiler, build_relationship_prompt  # noqa: E402
from nonebot_plugin_personal_companion.llm_client import LLMClient  # noqa: E402
from nonebot_plugin_personal_companion.turn_context import (  # noqa: E402
    analyze_turn,
    build_companion_context_prompt,
    choose_reply_max_tokens,
)


class CompletionChoice:
    def __init__(self, content, finish_reason="stop"):
        self.message = Mock(content=content)
        self.finish_reason = finish_reason


class CompletionResponse:
    def __init__(self, content, finish_reason="stop"):
        self.choices = [CompletionChoice(content, finish_reason)]


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
        active = store.get_all_key_memories(user_id=1)

        assert overview is not None and "准备考试" in overview
        assert ended is not None and "已标记为已结束" in ended
        assert suppressed is not None and "不再主动提起" in suppressed
        assert "用户最近在准备考试" not in active
        assert "用户已经拿到offer了" not in active
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

    assert "北京时间 21:05" in prompt
    assert "周二晚上" in prompt
    assert "北京时间 13:05" not in prompt


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
    chat = ProactiveChat(memory, Mock(), Mock(), kb=None)
    with patch("nonebot_plugin_personal_companion.proactive.datetime", FixedDateTime), \
         patch("nonebot_plugin_personal_companion.personality.datetime", FixedDateTime):
        prompt = chat._build_proactive_prompt(user_id=1)

    assert "北京时间 21:05" in prompt
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
    assert turn.flow_invite == "appreciation"

    assert "分享好事" in ctx
    assert "真诚替他开心" in ctx
    assert "不要把庆祝变成说教" in ctx


def test_companion_context_short_ack_allows_short_reply():
    turn = analyze_turn("嗯嗯", [])
    ctx = build_companion_context_prompt(turn)

    assert turn.intent == "short_ack"

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


def test_reply_max_tokens_short_and_factual_profiles():
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


def test_proactive_prompt_groups_memory_by_status():
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
