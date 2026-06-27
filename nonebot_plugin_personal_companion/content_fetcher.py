import asyncio
import random
import jieba
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx
from nonebot.log import logger

from .llm_client import LLMClient
from .memory import MemoryStore
from .config import Config
from .personality import BEIJING_TZ


@dataclass
class VideoInfo:
    title: str
    bvid: str
    author: str
    views: int
    likes: int
    duration: int  # seconds
    desc: str
    category: str
    url: str = ""

    def __post_init__(self):
        self.url = f"https://www.bilibili.com/video/{self.bvid}"

    def keywords(self) -> list[str]:
        text = f"{self.title} {self.category} {self.desc}"
        words = jieba.cut(text)
        return [w.strip() for w in words if len(w.strip()) >= 2]


class BilibiliFetcher:
    """Fetches Bilibili popular/region videos and matches against user interests."""

    API_POPULAR = "https://api.bilibili.com/x/web-interface/popular"
    API_REGION = "https://api.bilibili.com/x/web-interface/dynamic/region"
    HEADERS = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/130.0.0.0 Safari/537.36"
        ),
        "Referer": "https://www.bilibili.com",
    }

    def __init__(self, llm: LLMClient, memory: MemoryStore, config: Config):
        self.llm = llm
        self.memory = memory
        self.config = config
        self._last_push: dict[int, float] = {}  # user_id -> last push timestamp

    def _filter_allowed(self, user_ids: list[int]) -> list[int]:
        allow = self.config.proactive_allow_users.strip()
        if not allow:
            return user_ids
        allowed = set()
        for item in allow.split(","):
            item = item.strip()
            if item.isdigit():
                allowed.add(int(item))
            elif item:
                logger.warning(f"Ignoring invalid proactive_allow_users token: {item}")
        return [uid for uid in user_ids if uid in allowed]

    def _filter_push_interval(self, user_ids: list[int], now: float) -> list[int]:
        eligible = []
        for uid in user_ids:
            last_push = self._last_push.get(uid)
            if last_push is None:
                eligible.append(uid)
                continue
            hours_since = (now - last_push) / 3600
            if hours_since >= self.config.content_push_interval_hours:
                eligible.append(uid)
        return eligible

    def _parse_category_ids(self) -> list[int]:
        rids = []
        for item in self.config.content_push_bili_categories.split(","):
            item = item.strip()
            if item.isdigit():
                rids.append(int(item))
            elif item:
                logger.warning(f"Ignoring invalid Bilibili category id: {item}")
        return rids

    # ── public API ────────────────────────────────────────────

    async def try_push(self):
        """Called by scheduler. Fetches, filters, and pushes content if appropriate."""
        if not self.config.content_push_enabled:
            return

        # Respect active hours (same as proactive chat)
        now_dt = datetime.now(BEIJING_TZ)
        hour = now_dt.hour
        if not (self.config.proactive_active_hours_start <= hour < self.config.proactive_active_hours_end):
            return

        user_ids = self._filter_allowed(self.memory.get_active_user_ids())
        if not user_ids:
            return

        # Check interval per user
        now = now_dt.timestamp()
        user_ids = self._filter_push_interval(user_ids, now)
        if not user_ids:
            return

        # Fetch videos
        videos = await self._fetch_videos()
        if not videos:
            logger.info("No videos fetched")
            return

        # Match against user interests and push to each user
        for uid in user_ids:
            # DND: skip if user ignored 3+ consecutive proactive messages
            ignored = self.memory.count_proactive_since_last_user_message(uid)
            if ignored >= 3:
                continue

            interests = self._get_user_interests(uid)
            top = self._filter_by_interest(videos, interests, limit=3)
            if not top:
                continue

            # LLM picks the best one and writes recommendation
            msg = await self._build_recommendation(top, uid)
            if not msg:
                continue

            try:
                from nonebot import get_bot
                bot = get_bot()
                for chunk in LLMClient.chunk_text(msg):
                    await bot.send_private_msg(user_id=uid, message=chunk)
                self.memory.record_proactive_sent(uid, content=msg)
                self._last_push[uid] = now
                logger.info(f"Sent to user {uid}: {top[0].title[:30]}")
            except Exception as e:
                logger.warning(f"Failed to send: {e}")

    # ── fetching ──────────────────────────────────────────────

    async def _fetch_videos(self) -> list[VideoInfo]:
        """Fetch videos from configured categories. Each category gets ~15 items."""
        rids = self._parse_category_ids()

        all_videos: list[VideoInfo] = []
        seen: set[str] = set()

        for rid in rids:
            per_rid = max(30 // max(len(rids), 1), 10)
            try:
                videos = await self._fetch_region(rid, per_rid)
                for v in videos:
                    if v.bvid not in seen:
                        seen.add(v.bvid)
                        all_videos.append(v)
            except Exception as e:
                logger.warning(f"Failed to fetch rid={rid}: {e}")

        return all_videos

    async def _fetch_region(self, rid: int, count: int = 30) -> list[VideoInfo]:
        """Fetch from region/ranking API. rid=0 uses popular API."""
        if rid == 0:
            url = f"{self.API_POPULAR}?ps={count}&pn=1"
            list_key = "list"
        else:
            url = f"{self.API_REGION}?ps={count}&rid={rid}"
            list_key = "archives"

        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(url, headers=self.HEADERS)
            data = resp.json()

        if data.get("code") != 0:
            return []

        items = data.get("data", {}).get(list_key, [])
        videos: list[VideoInfo] = []
        for item in items:
            try:
                videos.append(VideoInfo(
                    title=item.get("title", ""),
                    bvid=item.get("bvid", ""),
                    author=item.get("owner", {}).get("name", ""),
                    views=item.get("stat", {}).get("view", 0),
                    likes=item.get("stat", {}).get("like", 0),
                    duration=item.get("duration", 0),
                    desc=item.get("desc", "") or "",
                    category=item.get("tname", ""),
                ))
            except Exception:
                continue

        return videos

    # ── interest matching ─────────────────────────────────────

    def _get_user_interests(self, user_id: int) -> list[str]:
        """Extract interest keywords from user's key memories."""
        memories = self.memory.get_all_key_memories(user_id)
        if not memories:
            return []
        all_text = " ".join(memories)
        words = jieba.cut(all_text)
        return list(set(w.strip() for w in words if len(w.strip()) >= 2))

    def _filter_by_interest(self, videos: list[VideoInfo], interests: list[str], limit: int = 3) -> list[VideoInfo]:
        """Score videos by keyword overlap with user interests."""
        if not interests:
            # No known interests — return top viewed videos with some randomness
            ranked = sorted(videos, key=lambda v: v.views, reverse=True)[:10]
            return random.sample(ranked, min(limit, len(ranked)))

        interest_set = set(interests)
        scored: list[tuple[float, VideoInfo]] = []
        for v in videos:
            kw_set = set(v.keywords())
            common = len(kw_set & interest_set)
            total = len(kw_set | interest_set)
            score = common / max(total, 1)
            # Boost by engagement
            engagement_bonus = min((v.likes + v.views * 0.1) / 100000, 0.3)
            scored.append((score + engagement_bonus, v))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [v for _, v in scored[:limit]]

    # ── recommendation generation ─────────────────────────────

    async def _build_recommendation(self, candidates: list[VideoInfo], user_id: int) -> str | None:
        """Use LLM to write a natural recommendation for the top-matched video. Link is always appended."""
        if not candidates:
            return None

        # Pick the top candidate
        video = candidates[0]
        dur_min = video.duration // 60
        dur_str = f"{dur_min}分钟" if dur_min > 0 else f"{video.duration}秒"
        video_info = (
            f"视频：【{video.category}·{video.author}】{video.title}\n"
            f"时长{dur_str}，{video.views}播放·{video.likes}赞\n"
            f"简介：{video.desc[:120]}"
        )
        video_url = video.url

        # Get user context
        memories = self.memory.get_all_key_memories(user_id)
        memory_hint = ""
        if memories:
            memory_hint = "你记得关于对方的事：" + "；".join(memories[:5])

        last_active = self.memory.get_last_active_time(user_id)
        gap_hint = ""
        if last_active:
            last_dt = datetime.fromisoformat(last_active).replace(tzinfo=timezone.utc)
            gap_h = (datetime.now(BEIJING_TZ) - last_dt.astimezone(BEIJING_TZ)).total_seconds() / 3600
            if gap_h > 4:
                gap_hint = f"对方{int(gap_h)}小时没说话了，语气轻松自然。"

        prompt = (
            "你要给朋友推荐这个视频，写2-3句话的推荐语。\n"
            "要求：\n"
            "- 自然，像朋友分享，不是营销号\n"
            "- 提一下为什么觉得对方会喜欢（基于你对他的了解）\n"
            "- 不要写视频链接，链接会自动加上\n"
            "- 2-4句话，不要太长\n\n"
            f"{memory_hint}\n"
            f"{gap_hint}\n\n"
            f"{video_info}"
        )

        system = (
            "你是艾琳娜，一个温柔耐心的朋友。你正在给信任的朋友分享你觉得他会喜欢的视频。"
            "语气自然，不死板。"
        )

        try:
            reply = await asyncio.to_thread(
                self.llm.chat,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=384,
            )
            if reply and len(reply) > 10:
                return f"{reply.strip()}\n\n👉 {video_url}"
        except Exception:
            pass

        # Fallback
        return f"刚看到这个视频觉得挺有意思的——{video.title}，分享给你看看 👉 {video_url}"
