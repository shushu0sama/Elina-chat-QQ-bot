import time
import re
from copy import deepcopy
from typing import Any
from openai import OpenAI


class LLMClient:
    """DeepSeek API wrapper with retry logic and response chunking for QQ."""

    # QQ text message limit (~4500 bytes), leave headroom
    MAX_CHUNK_BYTES = 4000
    CHUNK_DELAY_SECONDS = 1.0

    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def chat(self, messages: Any, max_retries: int = 3, max_tokens: int = 1024) -> str:
        """Send a chat completion request with exponential backoff retry."""
        last_error = None
        continuation_messages = deepcopy(messages)
        parts: list[str] = []

        for continuation in range(3):
            for attempt in range(max_retries):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=continuation_messages,
                        temperature=0.8,
                        max_tokens=max_tokens,
                    )
                    content = response.choices[0].message.content or ""
                    parts.append(content)

                    if response.choices[0].finish_reason != "length":
                        return "".join(parts)

                    continuation_messages = self._build_continuation_messages(continuation_messages, content)
                    break
                except Exception as e:
                    last_error = e
                    if attempt < max_retries - 1:
                        delay = 2**attempt  # 1s, 2s, 4s
                        time.sleep(delay)
            else:
                return "我刚才没有生成出完整回复，你把上一句再发我一次好吗？"

        return "".join(parts)

    @staticmethod
    def _build_continuation_messages(messages: Any, partial_reply: str) -> Any:
        continuation_messages = deepcopy(messages)
        continuation_messages.append({"role": "assistant", "content": partial_reply})
        continuation_messages.append({
            "role": "user",
            "content": "继续刚才被截断的回复，只输出后续内容，不要重复已经说过的部分。",
        })
        return continuation_messages

    _CN_CONNECTORS = (
        "就", "也", "还", "然后", "所以", "但是", "不过", "而且",
        "或者", "还是", "就是", "只是", "尤其是", "特别是",
    )

    @staticmethod
    def _is_chinese_char(c: str) -> bool:
        cp = ord(c)
        return (0x4E00 <= cp <= 0x9FFF or 0x3400 <= cp <= 0x4DBF
                or 0xF900 <= cp <= 0xFAFF or 0x20000 <= cp <= 0x2A6DF)

    @classmethod
    def looks_incomplete_reply(cls, text: str) -> bool:
        stripped = text.strip()
        if not stripped:
            return False

        if stripped.endswith(("——", "—", "：", ":", "，", ",", "；", ";")):
            return True

        _lq = chr(0x201C)
        _rq = chr(0x201D)
        if stripped.count(chr(92) + chr(34)) % 2 == 1 or stripped.count(_lq) != stripped.count(_rq):
            return True

        for conn in cls._CN_CONNECTORS:
            if stripped.endswith(conn):
                return True


        _bs = chr(92)
        _pat = (
            "(试试|可以这样说|换成|改成)"
            + "[" + _bs + "s" + _bs + "S]{0,40}[——：:]"
            + _bs + "s*[" + _bs + "n" + _bs + "r]*" + _bs + "s*["
            + _bs + chr(34) + _lq + "][^"
            + _bs + chr(34) + _rq + "]{4,80}[。！？!?]?["
            + _bs + chr(34) + _rq + "]?" + _bs + "s*$"
        )
        example_only = re.search(_pat, stripped)
        return bool(example_only)

    @staticmethod
    def _build_semantic_continuation_messages(messages: Any, partial_reply: str) -> Any:
        continuation_messages = deepcopy(messages)
        continuation_messages.append({"role": "assistant", "content": partial_reply})
        continuation_messages.append({
            "role": "user",
            "content": "刚才这条回复看起来像只给了开头或例句。请补上一两句自然收尾，只输出补充内容，不要重复已经说过的部分。",
        })
        return continuation_messages

    def complete_if_needed(self, messages: Any, reply: str, max_tokens: int = 256) -> str:
        if not self.looks_incomplete_reply(reply):
            return reply
        continuation = self.chat(
            self._build_semantic_continuation_messages(messages, reply),
            max_tokens=max_tokens,
        )
        if not continuation or continuation.startswith("我刚才没有生成出完整回复"):
            return reply
        return reply + continuation

    def chat_with_tools(self, messages: Any, tools: Any,
                        max_retries: int = 2, max_tokens: int = 1024):
        """Send a chat request with tool definitions. Returns the raw response
        so the caller can inspect tool_calls. On failure, returns None."""
        for attempt in range(max_retries):
            try:
                return self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    temperature=0.8,
                    max_tokens=max_tokens,
                )
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                else:
                    return None

    def chat_vision(self, text: str, image_urls: list[str], system_prompt: str = "",
                    max_retries: int = 2, max_tokens: int = 512) -> str:
        """Send a vision request with images. Falls back gracefully on failure."""
        content: list[dict[str, Any]] = [{"type": "text", "text": text}]
        for url in image_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})

        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.8,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception:
                if attempt < max_retries - 1:
                    time.sleep(2**attempt)

        return ""

    @staticmethod
    def _error_hint(error: Exception | None) -> str:
        if error is None:
            return ""
        msg = str(error)
        if "rate" in msg.lower() or "429" in msg:
            return "频率限制"
        if "timeout" in msg.lower():
            return "超时"
        if "auth" in msg.lower() or "401" in msg or "403" in msg:
            return "认证失败，请检查API Key"
        return "未知错误"

    def generate_summary(self, messages: list[dict]) -> str:
        """Generate a concise summary of recent conversation."""
        prompt = (
            "请用3-5个要点总结以下对话中的关键信息（用户提到的事实、偏好、事件等），用中文。\n"
            "不要把历史内容写成当前正在发生。只输出要点本身，不要加时间标记。\n\n"
        )
        conversation = "\n".join(
            f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}"
            for m in messages
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个对话摘要助手。只总结事实信息，不要加时间标记。"},
                    {"role": "user", "content": prompt + conversation},
                ],
                temperature=0.3,
                max_tokens=512,
            )
            return response.choices[0].message.content or ""
        except Exception:
            return ""

    def extract_memories(self, recent_messages: list[dict], existing_memories: list[str]) -> list[str]:
        """Extract new facts, preferences, and patterns from recent conversation."""
        if not recent_messages:
            return []

        conversation = "\n".join(
            f"{"用户" if m["role"] == "user" else "艾琳娜"}: {m["content"]}"
            for m in recent_messages
        )
        existing_block = "\n".join(f"- {m}" for m in existing_memories) if existing_memories else "（暂无）"

        prompt = (
            "从以下对话中提取关于用户的新信息（之前没记录过的）。\n"
            "只提取事实性信息，例如：姓名、经历、偏好、习惯、正在做的事、关心的话题、情绪模式。\n"
            "每条一行，用• 开头。如果没有值得记录的新信息，回复‘无’。\n"
            "不要提取泛泛的闲聊内容。\n\n"
            f"[已记录的信息：]\n{existing_block}\n\n"
            f"[最近的对话：]\n{conversation}"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个信息提取助手。只提取新的、值得记住的事实。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=512,
            )
            text = response.choices[0].message.content or ""
        except Exception:
            return []

        memories: list[str] = []
        for line in text.split("\n"):
            stripped = line.strip()
            if stripped.startswith("•") or stripped.startswith("-"):
                content = stripped.lstrip("•- ").strip()
                if content and content != "无" and len(content) > 2:
                    memories.append(content)
        return memories

    def extract_memories_structured(self, recent_messages: list[dict],
                                      existing_memories: list[str]) -> list[dict]:
        """Extract memories with emotion tags and entity tags for richer retrieval.

        Returns a list of dicts: {content, emotions, entities}
        """
        if not recent_messages:
            return []

        conversation = "\n".join(
            f"{"用户" if m["role"] == "user" else "艾琳娜"}: {m["content"]}"
            for m in recent_messages
        )
        existing_block = "\n".join(f"- {m}" for m in existing_memories) if existing_memories else "（暂无）"

        prompt = (
            "从以下对话中提取关于用户的新信息（之前没记录过的）。\n"
            "只提取事实性信息，例如：姓名、经历、偏好、习惯、正在做的事、关心的话题、情绪模式。\n"
            "对每条信息，同时标注情绪标签和实体标签。\n\n"
            "情绪标签从以下选择：焦虑、难过、烦躁、开心、迷茫、疲惫、中性\n"
            "实体标签：信息中涉及的人/事/物/地点/话题（用逗号分隔）\n\n"
            "输出格式（每条一行）：\n"
            "• [事实内容] | 情绪:[标签] | 实体:[标签1,标签2,...]\n\n"
            "示例：\n"
            "• 用户最近加班到很晚，担心项目进度 | 情绪:焦虑 | 实体:工作,项目,加班\n"
            "• 用户养了一只叫年糕的猫 | 情绪:中性 | 实体:猫,年糕,宠物\n\n"
            "如果没有值得记录的新信息，回复‘无’。\n"
            "不要提取泛泛的闲聊内容。\n\n"
            f"[已记录的信息：]\n{existing_block}\n\n"
            f"[最近的对话：]\n{conversation}"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个信息提取助手。只提取新的、值得记住的事实，同时标注情绪和实体。"},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=512,
            )
            text = response.choices[0].message.content or ""
        except Exception:
            return []

        result: list[dict] = []
        for line in text.split("\n"):
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

        return result

    @classmethod
    def chunk_text(cls, text: str, max_bytes: int | None = None) -> list[str]:
        """Split text at sentence boundaries to fit QQ's message length limit."""
        limit = max_bytes or cls.MAX_CHUNK_BYTES
        if len(text.encode("utf-8")) <= limit:
            return [text]

        chunks: list[str] = []
        sentences = re.split(r"(?<=[。！？!?\n])", text)
        current = ""
        for sent in sentences:
            if len(sent.encode("utf-8")) > limit:
                if current:
                    chunks.append(current)
                    current = ""
                chunks.extend(cls._split_by_bytes(sent, limit))
                continue

            if len((current + sent).encode("utf-8")) > limit:
                if current:
                    chunks.append(current)
                current = sent
            else:
                current += sent
        if current:
            chunks.append(current)
        return chunks

    @staticmethod
    def _split_by_bytes(text: str, limit: int) -> list[str]:
        chunks: list[str] = []
        current = ""
        for char in text:
            if current and len((current + char).encode("utf-8")) > limit:
                chunks.append(current)
                current = char
            else:
                current += char
        if current:
            chunks.append(current)
        return chunks

