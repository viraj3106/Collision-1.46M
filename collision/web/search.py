import time
import urllib.parse
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import requests
from collision.web.schemas import WebSearchItem, WebSearchResult

class WebSearchProvider(ABC):
    """
    Abstract interface for web search providers.
    Ensures provider independence across the COLLISION system.
    """

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> WebSearchResult:
        pass

class MockWebSearchProvider(WebSearchProvider):
    """
    Deterministic offline mock search provider for test suites and benchmark evaluations.
    """

    def __init__(self, mock_database: Optional[Dict[str, List[Dict[str, str]]]] = None):
        self.mock_database: Dict[str, List[Dict[str, str]]] = mock_database or {}

    def add_mock_results(self, query: str, items: List[Dict[str, str]]):
        self.mock_database[query.lower().strip()] = items

    def search(self, query: str, max_results: int = 5) -> WebSearchResult:
        t0 = time.perf_counter()
        if not query or not str(query).strip():
            return WebSearchResult(query=query, results=[], total_found=0, latency_ms=0.0)

        q_norm = query.lower().strip()
        matched_items: List[WebSearchItem] = []

        raw_items = []
        if q_norm in self.mock_database:
            raw_items = list(self.mock_database[q_norm])
        else:
            # Score matches using content-word overlap
            import re
            from collision.rag.embeddings import LocalEmbeddingModel
            q_words = set(re.findall(r"\b[a-z0-9_]{2,}\b", q_norm)) - LocalEmbeddingModel.STOP_WORDS
            matches: List[tuple[float, int, List[Dict[str, str]]]] = []
            if q_words:
                for k, items in self.mock_database.items():
                    k_words = set(re.findall(r"\b[a-z0-9_]{2,}\b", k.lower())) - LocalEmbeddingModel.STOP_WORDS
                    overlap = len(q_words.intersection(k_words))
                    if overlap > 0:
                        score = overlap / min(max(1, len(k_words)), max(1, len(q_words)))
                        matches.append((score, overlap, items))

                matches.sort(key=lambda x: (x[0], x[1]), reverse=True)
                for score, overlap, items in matches[:max_results]:
                    if overlap >= 1:
                        raw_items.extend(items)

        # Remove duplicate URLs
        seen_urls = set()
        for item in raw_items:
            url = item.get("url", "https://example.com")
            if url in seen_urls:
                continue
            seen_urls.add(url)

            domain = urllib.parse.urlparse(url).netloc or "example.com"
            matched_items.append(WebSearchItem(
                title=item.get("title", "Web Title"),
                url=url,
                snippet=item.get("snippet", "Web snippet."),
                domain=domain
            ))
            if len(matched_items) >= max_results:
                break

        latency_ms = (time.perf_counter() - t0) * 1000.0
        return WebSearchResult(
            query=query,
            results=matched_items,
            total_found=len(matched_items),
            latency_ms=latency_ms,
            error=None if matched_items else "No results found"
        )

class LiveDuckDuckGoSearchProvider(WebSearchProvider):
    """
    Live web search provider using DuckDuckGo HTML / Instant Answers.
    Handles rate-limiting and connection errors gracefully without crashing.
    """

    def search(self, query: str, max_results: int = 5) -> WebSearchResult:
        t0 = time.perf_counter()
        if not query or not str(query).strip():
            return WebSearchResult(query=query, results=[], total_found=0, latency_ms=0.0)

        url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        try:
            resp = requests.get(url, headers=headers, timeout=4.0)
            if resp.status_code != 200:
                return WebSearchResult(query=query, results=[], total_found=0, latency_ms=(time.perf_counter()-t0)*1000.0, error=f"HTTP Error {resp.status_code}")

            from bs4 import BeautifulSoup
            soup = BeautifulSoup(resp.text, "html.parser")
            results = []
            seen_urls = set()

            for result_elem in soup.find_all("div", class_="result"):
                link_elem = result_elem.find("a", class_="result__url")
                title_elem = result_elem.find("a", class_="result__title")
                snippet_elem = result_elem.find("a", class_="result__snippet")

                if link_elem and link_elem.get("href"):
                    raw_url = link_elem.get("href").strip()
                    if raw_url.startswith("//duckduckgo.com/l/?uddg="):
                        parsed_url = urllib.parse.parse_qs(urllib.parse.urlparse(raw_url).query).get("uddg", [None])[0]
                        if parsed_url:
                            raw_url = parsed_url

                    if raw_url in seen_urls or not raw_url.startswith("http"):
                        continue
                    seen_urls.add(raw_url)

                    title = title_elem.get_text(strip=True) if title_elem else "Search Result"
                    snippet = snippet_elem.get_text(strip=True) if snippet_elem else ""
                    domain = urllib.parse.urlparse(raw_url).netloc

                    results.append(WebSearchItem(title=title, url=raw_url, snippet=snippet, domain=domain))
                    if len(results) >= max_results:
                        break

            latency_ms = (time.perf_counter() - t0) * 1000.0
            return WebSearchResult(query=query, results=results, total_found=len(results), latency_ms=latency_ms)

        except Exception as e:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return WebSearchResult(query=query, results=[], total_found=0, latency_ms=latency_ms, error=f"Search failed: {str(e)}")
