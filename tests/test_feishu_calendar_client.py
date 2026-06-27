import asyncio
from datetime import datetime
from zoneinfo import ZoneInfo
from unittest.mock import AsyncMock, Mock, patch

import nonebot
nonebot.init(_env_file=".env.example")

from nonebot_plugin_personal_companion.feishu_calendar import FeishuCalendarClient, ParsedCalendarRequest


def test_feishu_client_caches_tenant_token_and_creates_event():
    client = FeishuCalendarClient("app", "secret", "cal", "Asia/Shanghai")
    client._get_tenant_access_token = AsyncMock(return_value="tenant-token")
    mock_response = Mock()
    mock_response.raise_for_status = Mock()
    mock_response.json = Mock(return_value={"code": 0, "data": {"event_id": "evt_1"}})

    async def run():
        with patch("httpx.AsyncClient") as httpx_client:
            httpx_client.return_value.__aenter__.return_value.post = AsyncMock(return_value=mock_response)
            result = await client.create_event(
                "开会",
                datetime(2026, 6, 15, 15, 0, tzinfo=ZoneInfo("Asia/Shanghai")),
                datetime(2026, 6, 15, 15, 30, tzinfo=ZoneInfo("Asia/Shanghai")),
            )

        assert result["data"]["event_id"] == "evt_1"
        httpx_client.return_value.__aenter__.return_value.post.assert_awaited()

    asyncio.run(run())


def test_feishu_client_create_event_from_request_rejects_incomplete_request():
    client = FeishuCalendarClient("app", "secret", "cal")
    request = ParsedCalendarRequest(ok=False, clarification="need time")

    async def run():
        with pytest.raises(ValueError):
            await client.create_event_from_request(request)

    import pytest
    asyncio.run(run())
