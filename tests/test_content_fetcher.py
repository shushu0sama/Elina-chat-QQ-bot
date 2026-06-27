import asyncio
from unittest.mock import AsyncMock, Mock, patch

from nonebot_plugin_personal_companion.content_fetcher import BilibiliFetcher, VideoInfo
from nonebot_plugin_personal_companion.diary import DiaryWriter


def _run(coro):
    return asyncio.run(coro)


def test_content_filter_allowed_ignores_non_numeric_tokens():
    config = Mock(proactive_allow_users="1, abc, 2 # comment")
    fetcher = BilibiliFetcher(Mock(), Mock(), config)

    assert fetcher._filter_allowed([1, 2, 3]) == [1]


def test_diary_filter_allowed_ignores_non_numeric_tokens():
    config = Mock(proactive_allow_users="1, abc, 2 # comment")
    writer = DiaryWriter(Mock(), Mock(), config)

    assert writer._filter_allowed([1, 2, 3]) == [1]


def test_parse_category_ids_ignores_non_numeric_tokens():
    config = Mock(content_push_bili_categories="0, 160, abc, 2 # comment")
    fetcher = BilibiliFetcher(Mock(), Mock(), config)

    assert fetcher._parse_category_ids() == [0, 160]


def test_filter_push_interval_skips_users_in_cooldown():
    config = Mock(content_push_interval_hours=6.0)
    fetcher = BilibiliFetcher(Mock(), Mock(), config)
    fetcher._last_push = {1: 1000.0, 2: 1000.0 - 7 * 3600}

    assert fetcher._filter_push_interval([1, 2, 3], now=1000.0) == [2, 3]


def test_try_push_does_not_fetch_when_all_users_in_cooldown():
    config = Mock(
        content_push_enabled=True,
        proactive_active_hours_start=0,
        proactive_active_hours_end=24,
        proactive_allow_users="",
        content_push_interval_hours=6.0,
    )
    memory = Mock()
    memory.get_active_user_ids.return_value = [1]
    fetcher = BilibiliFetcher(Mock(), memory, config)
    fetcher._last_push = {1: 9999999999.0}
    fetcher._fetch_videos = AsyncMock()

    _run(fetcher.try_push())

    fetcher._fetch_videos.assert_not_called()


def test_try_push_only_sends_to_users_outside_cooldown():
    config = Mock(
        content_push_enabled=True,
        proactive_active_hours_start=0,
        proactive_active_hours_end=24,
        proactive_allow_users="",
        content_push_interval_hours=6.0,
    )
    memory = Mock()
    memory.get_active_user_ids.return_value = [1, 2]
    memory.count_proactive_since_last_user_message.return_value = 0
    memory.get_all_key_memories.return_value = []
    memory.get_last_active_time.return_value = None
    fetcher = BilibiliFetcher(Mock(), memory, config)
    fetcher._last_push = {1: 9999999999.0, 2: 0.0}
    fetcher._fetch_videos = AsyncMock(return_value=[VideoInfo("title", "BV1", "author", 100, 10, 60, "desc", "cat")])
    fetcher._build_recommendation = AsyncMock(return_value="hello")
    bot = Mock()
    bot.send_private_msg = AsyncMock()

    with patch("nonebot.get_bot", return_value=bot):
        _run(fetcher.try_push())

    bot.send_private_msg.assert_awaited_once_with(user_id=2, message="hello")
    memory.record_proactive_sent.assert_called_once_with(2, content="hello")
