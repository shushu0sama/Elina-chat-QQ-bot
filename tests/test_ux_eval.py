from pathlib import Path
from unittest.mock import Mock

import yaml

from nonebot_plugin_personal_companion.feishu_calendar import (
    format_calendar_intent_confirmation,
    has_strong_calendar_intent,
    parse_calendar_request,
    should_confirm_calendar_request,
)
from nonebot_plugin_personal_companion.memory import MemoryStore
from nonebot_plugin_personal_companion.proactive import ProactiveChat
from nonebot_plugin_personal_companion.turn_context import analyze_turn, build_reply_mode_prompt
import nonebot_plugin_personal_companion as companion_plugin


CASES_PATH = Path(__file__).parent / "evals" / "ux_cases.yaml"


def load_cases(kind: str) -> list[dict]:
    payload = yaml.safe_load(CASES_PATH.read_text(encoding="utf-8"))
    return [case for case in payload["cases"] if case["kind"] == kind]


def make_store():
    import tempfile

    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = tmp.name
    tmp.close()
    return MemoryStore(db_path), db_path


def cleanup_store(store, db_path):
    del store
    for ext in ("", "-wal", "-shm"):
        path = Path(db_path + ext) if ext else Path(db_path)
        try:
            path.unlink()
        except OSError:
            pass


def assert_includes(text: str, needles: list[str]):
    for needle in needles:
        assert needle in text


def assert_excludes(text: str, needles: list[str]):
    for needle in needles:
        assert needle not in text


def test_reply_mode_ux_eval_cases():
    for case in load_cases("reply_mode"):
        turn = analyze_turn(case["user"], [])
        prompt = build_reply_mode_prompt(turn)
        expect = case["expect"]

        assert turn.reply_mode == expect["reply_mode"], case["name"]
        assert_includes(prompt, expect.get("prompt_must_include", []))
        assert_excludes(prompt, expect.get("prompt_must_not_include", []))


def test_calendar_ux_eval_cases():
    for case in load_cases("calendar"):
        expect = case["expect"]
        request = parse_calendar_request(case["user"])

        assert should_confirm_calendar_request(case["user"]) is expect["should_confirm_calendar"], case["name"]
        assert has_strong_calendar_intent(case["user"]) is expect["strong_intent"], case["name"]
        assert request.ok is expect["parsed_ok"], case["name"]
        if expect.get("confirmation_must_include"):
            assert_includes(format_calendar_intent_confirmation(request), expect["confirmation_must_include"])


def test_memory_command_ux_eval_cases():
    for case in load_cases("memory_command"):
        store, db_path = make_store()
        old_store = companion_plugin.memory_store
        try:
            companion_plugin.memory_store = store
            reply = companion_plugin._handle_memory_management_command(case["user"], 1)
            memories = store.get_key_memories_with_meta(1)
            expect = case["expect"]

            assert reply is not None, case["name"]
            assert_includes(reply, expect.get("reply_must_include", []))
            assert any(
                item["memory_type"] == expect["memory_type"]
                and expect["memory_content_must_include"] in item["content"]
                for item in memories
            ), case["name"]
            if expect.get("proactive_snoozed"):
                assert store.get_proactive_snooze_until(1) is not None
        finally:
            companion_plugin.memory_store = old_store
            cleanup_store(store, db_path)


def test_proactive_topic_ux_eval_cases():
    for case in load_cases("proactive_topic"):
        store, db_path = make_store()
        try:
            setup = case.get("setup", {})
            for message in setup.get("recent_user_messages", []):
                store.save_message("user", message, user_id=1)
            for memory in setup.get("memories", []):
                store.add_key_memory(
                    memory["content"],
                    user_id=1,
                    memory_type=memory.get("memory_type"),
                )
            store.record_user_active(1)
            prompt = ProactiveChat(store, Mock(), Mock(), kb=None)._build_proactive_prompt(1)

            assert_includes(prompt, case["expect"].get("prompt_must_include", []))
            assert_excludes(prompt, case["expect"].get("prompt_must_not_include", []))
        finally:
            cleanup_store(store, db_path)
