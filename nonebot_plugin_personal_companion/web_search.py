"""Web search with pluggable backends. Falls back gracefully on failure."""

import httpx
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from bs4 import BeautifulSoup


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


# ── Abstract backend ────────────────────────────────────────────

class SearchBackend(ABC):
    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        ...

    def fetch_page(self, url: str, timeout: float = 10.0) -> str:
        """Fetch and extract readable text from a web page."""
        try:
            r = httpx.get(
                url,
                timeout=timeout,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.5",
                },
                follow_redirects=True,
            )
            if r.status_code != 200:
                return ""
            soup = BeautifulSoup(r.text, "lxml")
            # Remove script/style/nav/footer
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            text = soup.get_text(separator="\n", strip=True)
            # Deduplicate blank lines and trim to reasonable length
            lines = [ln.strip() for ln in text.split("\n") if ln.strip()]
            return "\n".join(lines[:300])  # ~300 lines is plenty for context
        except Exception:
            return ""


# ── Bing CN backend ─────────────────────────────────────────────

class BingBackend(SearchBackend):
    """Uses cn.bing.com — works in China, clean HTML, no API key needed."""

    BASE_URL = "https://cn.bing.com/search"

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        try:
            r = httpx.get(
                self.BASE_URL,
                params={"q": query, "count": min(max_results, 15)},
                timeout=10.0,
                headers={
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "Accept": "text/html,application/xhtml+xml",
                    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.5",
                },
                follow_redirects=True,
            )
            if r.status_code != 200:
                return []

            soup = BeautifulSoup(r.text, "lxml")
            results: list[SearchResult] = []

            for li in soup.find_all("li", class_="b_algo"):
                h2 = li.find("h2")
                a_tag = h2.find("a") if h2 else None
                title = a_tag.get_text(strip=True) if a_tag else ""
                url = a_tag.get("href", "") if a_tag else ""

                caption = li.find("div", class_="b_caption")
                snippet = caption.get_text(strip=True) if caption else ""

                if title and url:
                    results.append(SearchResult(title=title, url=url, snippet=snippet))
                    if len(results) >= max_results:
                        break

            return results
        except Exception:
            return []


# ── DuckDuckGo backend (when network allows) ────────────────────

class DuckDuckGoBackend(SearchBackend):
    """Uses duckduckgo_search library. May not work behind GFW."""

    def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        try:
            from duckduckgo_search import DDGS

            raw = list(DDGS().text(query, max_results=max_results, backend="html"))
            results: list[SearchResult] = []
            for item in raw:
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("href", ""),
                    snippet=item.get("body", ""),
                ))
            return results
        except Exception:
            return []


# ── Top-level API ───────────────────────────────────────────────

_default_backend: SearchBackend | None = None


def get_backend() -> SearchBackend:
    global _default_backend
    if _default_backend is None:
        _default_backend = BingBackend()
    return _default_backend


def set_backend(backend: SearchBackend) -> None:
    global _default_backend
    _default_backend = backend


def search_web(query: str, max_results: int = 5) -> list[SearchResult]:
    """Search the web. Returns a list of SearchResult."""
    return get_backend().search(query, max_results=max_results)


def fetch_page(url: str, timeout: float = 10.0) -> str:
    """Fetch and extract readable text from a web page."""
    return get_backend().fetch_page(url, timeout=timeout)


def format_search_results(results: list[SearchResult], query: str) -> str:
    """Format search results into a compact string for LLM consumption."""
    if not results:
        return f'搜索「{query}」没有找到相关结果。'

    lines = [f'以下是你搜索「{query}」获得的结果：', '']
    for i, r in enumerate(results, 1):
        snippet = r.snippet[:250] if r.snippet else "(无摘要)"
        lines.append(f"{i}. **{r.title}**")
        lines.append(f"   {snippet}")
        lines.append(f"   来源: {r.url}")
        lines.append("")
    return "\n".join(lines)
