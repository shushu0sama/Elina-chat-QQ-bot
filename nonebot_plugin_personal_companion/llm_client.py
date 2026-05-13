import time
import re
from openai import OpenAI


class LLMClient:
    """DeepSeek API wrapper with retry logic and response chunking for QQ."""

    # QQ text message limit (~4500 bytes), leave headroom
    MAX_CHUNK_BYTES = 4000
    CHUNK_DELAY_SECONDS = 1.0

    def __init__(self, api_key: str, base_url: str, model: str):
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    def chat(self, messages: list[dict], max_retries: int = 3, max_tokens: int = 1024) -> str:
        """Send a chat completion request with exponential backoff retry."""
        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.8,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    delay = 2**attempt  # 1s, 2s, 4s
                    time.sleep(delay)

        return f"我好像连接不上大脑了，稍等一下再试试？({self._error_hint(last_error)})"

    def chat_vision(self, text: str, image_urls: list[str], system_prompt: str = "",
                    max_retries: int = 2, max_tokens: int = 512) -> str:
        """Send a vision request with images. Falls back gracefully on failure."""
        content: list[dict] = [{"type": "text", "text": text}]
        for url in image_urls:
            content.append({"type": "image_url", "image_url": {"url": url}})

        messages: list[dict] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": content})

        last_error = None
        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.8,
                    max_tokens=max_tokens,
                )
                return response.choices[0].message.content or ""
            except Exception as e:
                last_error = e
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
            "请用3-5个要点总结以下对话中的关键信息（用户提到的事实、偏好、事件等），用中文：\n\n"
        )
        conversation = "\n".join(
            f"{'用户' if m['role'] == 'user' else '助手'}: {m['content']}"
            for m in messages
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是一个对话摘要助手，只总结事实信息。"},
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
            f"{'用户' if m['role'] == 'user' else '艾琳娜'}: {m['content']}"
            for m in recent_messages
        )
        existing_block = "\n".join(f"- {m}" for m in existing_memories) if existing_memories else "（暂无）"

        prompt = (
            "从以下对话中提取关于用户的新信息（之前没记录过的）。\n"
            "只提取事实性信息，例如：姓名、经历、偏好、习惯、正在做的事、关心的话题、情绪模式。\n"
            "每条一行，用'• '开头。如果没有值得记录的新信息，回复'无'。\n"
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
            if len((current + sent).encode("utf-8")) > limit:
                if current:
                    chunks.append(current)
                current = sent
            else:
                current += sent
        if current:
            chunks.append(current)
        return chunks
