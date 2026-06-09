import pytest
import time
import tempfile
from pathlib import Path

from nonebot_plugin_personal_companion.memory import MemoryStore


@pytest.fixture
def store():
    """Create a temporary file-backed MemoryStore for isolated tests."""
    tmp = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
    db_path = tmp.name
    tmp.close()
    s = MemoryStore(db_path=db_path)
    yield s
    # Close WAL connections before cleanup
    del s
    for ext in ("", "-wal", "-shm"):
        p = Path(db_path + ext) if ext else Path(db_path)
        try:
            p.unlink()
        except OSError:
            pass


class TestInit:
    def test_tables_created(self, store):
        conn = store._get_conn()
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        names = [r["name"] for r in tables]
        for t in ("messages", "summaries", "key_memories", "active_users", "proactive_log", "manifestation_wishes", "manifestation_evidence"):
            assert t in names


class TestMessages:
    def test_save_and_get_recent(self, store):
        store.save_message("user", "hello", user_id=1)
        store.save_message("assistant", "hi there", user_id=1)

        recent = store.get_recent_messages(limit=10, user_id=1)
        assert len(recent) == 2
        assert recent[0] == {"role": "user", "content": "hello"}
        assert recent[1] == {"role": "assistant", "content": "hi there"}

    def test_user_isolation(self, store):
        store.save_message("user", "msg from A", user_id=1)
        store.save_message("user", "msg from B", user_id=2)

        assert len(store.get_recent_messages(limit=10, user_id=1)) == 1
        assert len(store.get_recent_messages(limit=10, user_id=2)) == 1
        assert store.get_recent_messages(limit=10, user_id=1)[0]["content"] == "msg from A"

    def test_limit_respected(self, store):
        for i in range(50):
            store.save_message("user", f"msg {i}", user_id=1)

        recent = store.get_recent_messages(limit=5, user_id=1)
        assert len(recent) == 5
        # Should return the most recent 5
        assert recent[-1]["content"] == "msg 49"

    def test_get_messages_since(self, store):
        store.save_message("user", "old", user_id=1)
        time.sleep(0.01)
        store.save_message("user", "recent", user_id=1)

        # Use a cutoff far in the past to get all messages
        result = store.get_messages_since("2020-01-01", user_id=1)
        assert len(result) >= 1

    def test_get_user_message_stats(self, store):
        store.save_message("user", "short", user_id=1)
        store.save_message("user", "a longer message here", user_id=1)
        store.record_user_active(1)

        stats = store.get_user_message_stats(user_id=1)
        assert stats["total"] == 2
        assert stats["avg_len"] > 0


class TestKeyMemories:
    def test_add_and_retrieve(self, store):
        store.add_key_memory("用户养了一只猫", user_id=1)
        store.add_key_memory("用户喜欢喝咖啡", user_id=1)

        result = store.retrieve_memories(["猫"], user_id=1)
        assert any("猫" in r for r in result)

    def test_retrieve_by_keyword(self, store):
        store.add_key_memory("用户在杭州工作", user_id=1)
        store.add_key_memory("用户是程序员", user_id=1)

        result = store.retrieve_memories(["杭州"], user_id=1)
        assert len(result) >= 1
        assert any("杭州" in r for r in result)

    def test_retrieve_no_match(self, store):
        store.add_key_memory("用户在杭州工作", user_id=1)
        result = store.retrieve_memories(["xyznotfound"], user_id=1)
        assert len(result) == 0

    def test_retrieve_user_isolation(self, store):
        store.add_key_memory("A的秘密", user_id=1)
        store.add_key_memory("B的秘密", user_id=2)

        result_a = store.retrieve_memories(["秘密"], user_id=1)
        result_b = store.retrieve_memories(["秘密"], user_id=2)
        assert len(result_a) == 1
        assert len(result_b) == 1
        assert result_a != result_b

    def test_get_all_key_memories(self, store):
        store.add_key_memory("m1", user_id=1)
        store.add_key_memory("m2", user_id=1)
        assert len(store.get_all_key_memories(user_id=1)) == 2

    def test_has_similar_memory_substring(self, store):
        store.add_key_memory("用户喜欢喝咖啡", user_id=1)
        assert store.has_similar_memory("喝咖啡", user_id=1) is True

    def test_has_similar_memory_jaccard(self, store):
        store.add_key_memory("用户在杭州西湖区工作", user_id=1)
        # Similar enough by Jaccard
        assert store.has_similar_memory("用户在杭州西湖区生活", user_id=1) is True

    def test_has_similar_memory_no_match(self, store):
        store.add_key_memory("用户喜欢喝咖啡", user_id=1)
        assert store.has_similar_memory("完全不同的内容XYZ", user_id=1) is False

    def test_manifestation_memory_helpers(self, store):
        store.save_manifestation_entry(1, "manifest_seed", "愿望种子内容")
        store.save_manifestation_entry(1, "belief_rewrite", "信念改写内容")
        store.add_key_memory("普通记忆", user_id=1)

        entries = store.get_manifestation_memories(1)
        assert len(entries) == 2
        assert any("显化愿望种子" in entry for entry in entries)
        assert any("信念改写内容" in entry for entry in entries)

        meta = store.get_key_memories_with_meta(1)
        manifest_items = [item for item in meta if item["memory_type"] == "manifestation"]
        assert len(manifest_items) == 2
        assert all(item["importance"] == 5 for item in manifest_items)

    def test_manifestation_wish_lifecycle_and_evidence(self, store):
        wish_id = store.create_manifestation_wish(1, "稳定关系", "原始愿望内容")
        store.add_manifestation_evidence(1, "我今天没有反复确认", wish_id=wish_id, evidence_type="action")

        wishes = store.get_manifestation_wishes(1)
        evidence = store.get_manifestation_evidence(1, wish_id=wish_id)

        assert wishes[0]["id"] == wish_id
        assert wishes[0]["status"] == "active"
        assert evidence[0]["content"] == "我今天没有反复确认"

        store.update_manifestation_wish_status(1, wish_id, "released")
        updated = store.get_manifestation_wishes(1)
        assert updated[0]["status"] == "released"

    def test_manifestation_entry_creates_wish_and_diary_evidence(self, store):
        store.save_manifestation_entry(1, "manifest_seed", "愿望种子已种下：稳定关系")
        store.save_manifestation_entry(1, "manifest_diary", "今日显化证据：我更稳定了")

        wishes = store.get_manifestation_wishes(1)
        evidence = store.get_manifestation_evidence(1)

        assert len(wishes) == 1
        assert "稳定关系" in wishes[0]["title"]
        assert any("我更稳定了" in e["content"] for e in evidence)

    def test_manifestation_dashboard(self, store):
        wish_id = store.create_manifestation_wish(1, "事业机会", "raw")
        store.add_manifestation_evidence(1, "投出一份简历", wish_id=wish_id)

        dashboard = store.build_manifestation_dashboard(1)

        assert "你的显化仪表盘" in dashboard
        assert "事业机会" in dashboard
        assert "投出一份简历" in dashboard

    def test_count_key_memories(self, store):
        store.add_key_memory("m1", user_id=1)
        store.add_key_memory("m2", user_id=1)
        assert store.count_key_memories(user_id=1) == 2

    def test_prune_stale_memories(self, store):
        store.add_key_memory("low importance stale", user_id=1, importance=1)
        store.add_key_memory("high importance", user_id=1, importance=5)

        # Set last_accessed_at to 60 days ago for the low importance one
        conn = store._get_conn()
        conn.execute(
            "UPDATE key_memories SET last_accessed_at = datetime('now', '-60 days') WHERE content = ?",
            ("low importance stale",),
        )
        conn.commit()
        conn.close()

        store.prune_stale_memories(user_id=1, min_importance=2, days_unused=30)
        remaining = store.get_all_key_memories(user_id=1)
        assert "high importance" in remaining
        assert "low importance stale" not in remaining


class TestSummaries:
    def test_save_and_get_summaries(self, store):
        store.save_summary("用户说今天心情不好", 1, 10, user_id=1)
        store.save_summary("用户分享了旅行经历", 11, 20, user_id=1)

        summaries = store.get_recent_summaries(user_id=1, limit=2)
        assert len(summaries) == 2
        assert "旅行" in summaries[0]  # most recent first

    def test_message_count_since_last_summary(self, store):
        assert store.message_count_since_last_summary(user_id=1) == 0

        store.save_message("user", "hello", user_id=1)
        store.save_message("user", "world", user_id=1)

        assert store.message_count_since_last_summary(user_id=1) == 2

        store.save_summary("test", 0, 2, user_id=1)
        assert store.message_count_since_last_summary(user_id=1) == 0

    def test_summary_watermark_advances(self, store):
        for i in range(5):
            store.save_message("user", f"msg {i}", user_id=1)

        assert store.message_count_since_last_summary(user_id=1) == 5
        start_id = store.get_oldest_message_id_after(user_id=1, after_id=0)
        end_id = store.get_latest_message_id(user_id=1)
        store.save_summary("summary", start_id, end_id, user_id=1)

        assert store.message_count_since_last_summary(user_id=1) == 0

    def test_extraction_checkpoint_advances(self, store):
        for i in range(10):
            store.save_message("user", f"msg {i}", user_id=1)

        assert store.messages_since_last_extraction(user_id=1) == 10
        latest_id = store.get_latest_message_id(user_id=1)
        store.add_key_memory("用户提到了测试消息", source_msg_id=latest_id, user_id=1)
        store.save_extraction_checkpoint(user_id=1, last_msg_id=latest_id)

        assert store.messages_since_last_extraction(user_id=1) == 0

    def test_expired_short_event_memory_status(self, store):
        store.add_key_memory("用户准备周四出去玩", user_id=1)
        conn = store._get_conn()
        conn.execute(
            "UPDATE key_memories SET created_at = datetime('now', '-7 days') WHERE content = ?",
            ("用户准备周四出去玩",),
        )
        conn.commit()
        conn.close()

        items = store.get_key_memories_with_meta(1)
        by_content = {item["content"]: item for item in items}

        assert by_content["用户准备周四出去玩"]["status"] == "expired"

    def test_recent_short_event_memory_stays_ongoing(self, store):
        store.add_key_memory("用户准备明天出去玩", user_id=1)

        items = store.get_key_memories_with_meta(1)
        by_content = {item["content"]: item for item in items}

        assert by_content["用户准备明天出去玩"]["status"] == "ongoing"

    def test_open_ended_ongoing_memory_does_not_expire_immediately(self, store):
        store.add_key_memory("用户最近在准备考试", user_id=1)
        conn = store._get_conn()
        conn.execute(
            "UPDATE key_memories SET created_at = datetime('now', '-7 days') WHERE content = ?",
            ("用户最近在准备考试",),
        )
        conn.commit()
        conn.close()

        items = store.get_key_memories_with_meta(1)
        by_content = {item["content"]: item for item in items}

        assert by_content["用户最近在准备考试"]["status"] == "ongoing"

    def test_memory_classification_metadata(self, store):
        store.add_key_memory("用户喜欢喝咖啡", user_id=1)
        store.add_key_memory("用户最近在准备考试", user_id=1)
        store.add_key_memory("用户已经拿到offer了", user_id=1)
        store.add_key_memory("用户不喜欢被催促", user_id=1)
        store.add_key_memory("用户今天去了医院", user_id=1)

        items = store.get_key_memories_with_meta(1)
        by_content = {item["content"]: item for item in items}

        assert by_content["用户喜欢喝咖啡"]["memory_type"] == "preference"
        assert by_content["用户最近在准备考试"]["status"] == "ongoing"
        assert by_content["用户已经拿到offer了"]["status"] == "completed"
        assert by_content["用户今天去了医院"]["status"] == "completed"
        assert by_content["用户不喜欢被催促"]["memory_type"] == "boundary"

    def test_completed_and_expired_events_are_not_recalled(self, store):
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

        recalled = store.retrieve_memories(["用户"], user_id=1, limit=10)
        all_active = store.get_all_key_memories(user_id=1)
        all_with_inactive = store.get_all_key_memories(user_id=1, include_inactive=True)

        assert "用户喜欢喝咖啡" in recalled
        assert "用户已经拿到offer了" not in recalled
        assert "用户准备周四出去玩" not in recalled
        assert "用户已经拿到offer了" not in all_active
        assert "用户准备周四出去玩" not in all_active
        assert "用户已经拿到offer了" in all_with_inactive
        assert "用户准备周四出去玩" in all_with_inactive

    def test_memory_management_helpers(self, store):
        store.add_key_memory("用户最近在准备考试", user_id=1)
        store.add_key_memory("用户不想再提offer", user_id=1)

        overview = store.build_memory_overview(1)
        ended = store.update_key_memory_status(1, "考试", "completed", memory_type="event")
        suppressed = store.update_key_memory_status(1, "offer", "suppressed")
        active = store.get_all_key_memories(user_id=1)
        deleted = store.delete_key_memories(1, "考试")

        assert "用户最近在准备考试" in overview
        assert ended == ["用户最近在准备考试"]
        assert suppressed == ["用户不想再提offer"]
        assert "用户最近在准备考试" not in active
        assert "用户不想再提offer" not in active
        assert deleted == ["用户最近在准备考试"]


class TestActiveUsers:
    def test_record_and_get_active(self, store):
        store.record_user_active(1)
        store.record_user_active(2)

        active = store.get_active_user_ids()
        assert 1 in active
        assert 2 in active

    def test_last_active_time(self, store):
        store.record_user_active(1)
        last = store.get_last_active_time(1)
        assert last is not None

    def test_last_active_time_none(self, store):
        assert store.get_last_active_time(999) is None


class TestProactiveLog:
    def test_record_and_get(self, store):
        store.record_proactive_sent(1, content="早安")
        store.record_proactive_sent(1, content="今天过得怎么样")

        recent = store.get_recent_proactive_content(1, limit=3)
        assert len(recent) == 2
        assert recent[0] == "今天过得怎么样"  # most recent first

    def test_get_last_proactive_time(self, store):
        assert store.get_last_proactive_time(1) is None
        store.record_proactive_sent(1)
        assert store.get_last_proactive_time(1) is not None

    def test_count_proactive_since_last_user_message(self, store):
        # Initially 0
        assert store.count_proactive_since_last_user_message(1) == 0

        store.record_proactive_sent(1)
        assert store.count_proactive_since_last_user_message(1) == 0  # No user activity recorded
