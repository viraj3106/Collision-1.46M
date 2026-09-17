import os
import time
import requests
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from rag.schemas import SearchItem, SearchResult

class BaseSearchProvider(ABC):
    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> SearchResult:
        pass

class MockSearchProvider(BaseSearchProvider):
    """
    Mock search provider for testing and deterministic benchmark evaluations.
    """
    def __init__(self, mock_database: Optional[Dict[str, List[Dict[str, str]]]] = None):
        self.mock_database = mock_database or {}

    def add_mock_results(self, query: str, results: List[Dict[str, str]]):
        self.mock_database[query.lower().strip()] = results

    def search(self, query: str, top_k: int = 5) -> SearchResult:
        t0 = time.perf_counter()
        q_norm = query.lower().strip()
        
        # Check exact or keyword partial match
        matched_items = []
        if q_norm in self.mock_database:
            raw_items = self.mock_database[q_norm]
        else:
            raw_items = []
            for key, items in self.mock_database.items():
                if any(word in key for word in q_norm.split()) or any(word in q_norm for word in key.split()):
                    raw_items.extend(items)
                    break
        
        for item in raw_items[:top_k]:
            matched_items.append(
                SearchItem(
                    title=item.get("title", "Mock Title"),
                    url=item.get("url", "https://example.com/mock"),
                    snippet=item.get("snippet", "Mock content snippet."),
                    source=item.get("source", "mock_web")
                )
            )
            
        latency = (time.perf_counter() - t0) * 1000.0
        return SearchResult(
            query=query,
            results=matched_items,
            total_found=len(matched_items),
            latency_ms=latency
        )

class WikipediaSearchProvider(BaseSearchProvider):
    """
    High-fidelity Wikipedia search provider using MediaWiki OpenSearch, REST Summary APIs,
    and smart entity query normalization with title proximity ranking. Requires no API keys.
    """
    def __init__(self, user_agent: str = "CollisionKnowledgeEngine/1.0 (https://github.com/viraj3106/Collision-1.46M)"):
        self.headers = {"User-Agent": user_agent}

    def _normalize_query_candidates(self, query: str) -> Tuple[str, List[str]]:
        import re
        q = query.strip()
        # Remove common question framing and conversational preambles
        clean_q = re.sub(
            r'^(?:can\s+you\s+)?(?:tell\s+me\s+(?:about|what\s+is|who\s+is)|explain(?:\s+to\s+me)?|describe|define|i\s+want\s+to\s+know\s+about|what\s+do\s+you\s+know\s+about|who|what|where|when|why|how)\s+(?:is|was|are|were|do|does|did|can|the|a|an|about)?\s*',
            '',
            q,
            flags=re.I
        ).strip(' ?.:;!"\'')
        
        # Strip trailing "work", "works", "mean", etc.
        clean_q = re.sub(r'\s+(?:and\s+how\s+(?:it|does\s+it)\s+works?|how\s+does\s+it\s+work|work|works|work\?|do|mean)\??$', '', clean_q, flags=re.I).strip(' ?.:;!"\'')
        clean_q = clean_q.strip(' ?.:;!"\'')

        candidates = []
        
        # Targeted extraction for relational queries ("capital of Japan" -> "Japan", "Capital of Japan")
        rel_match = re.search(r'\b(?:capital|president|prime\s+minister|founder|ceo|currency|population|history)\s+of\s+([a-zA-Z\s\-]+)', q, re.I)
        if rel_match:
            target_entity = rel_match.group(1).strip(' ?.:;!"\'')
            if len(target_entity) >= 2:
                candidates.append(target_entity)
                candidates.append(f"{clean_q}")
                candidates.append(f"Capital of {target_entity}")

        if clean_q and len(clean_q) >= 2 and clean_q.lower() != q.lower() and clean_q not in candidates:
            candidates.append(clean_q)
        if q.strip(' ?.:;!"\'') not in candidates:
            candidates.append(q.strip(' ?.:;!"\''))
        return clean_q, candidates

    def search(self, query: str, top_k: int = 5) -> SearchResult:
        t0 = time.perf_counter()
        results = []
        seen_titles = set()
        clean_q, candidates = self._normalize_query_candidates(query)
        collected_titles = []

        try:
            for q_term in candidates:
                opensearch_url = "https://en.wikipedia.org/w/api.php?action=opensearch&search=" + requests.utils.quote(q_term) + f"&limit={max(10, top_k*2)}&namespace=0&format=json"
                r = requests.get(opensearch_url, headers=self.headers, timeout=3.5)
                if r.status_code == 200:
                    data = r.json()
                    titles = data[1] if len(data) > 1 else []
                    urls = data[3] if len(data) > 3 else []
                    
                    for title, page_url in zip(titles, urls):
                        if title.lower() not in seen_titles:
                            seen_titles.add(title.lower())
                            collected_titles.append((title, page_url))

            # Rank titles by exactness to clean entity query
            def _rank_title(item):
                t, _ = item
                t_low = t.lower()
                q_low = (clean_q or query).lower()
                
                # Penalize mismatch traps
                if "punishment" in t_low and "punishment" not in q_low:
                    return (10, len(t))
                if "investment" in t_low and "investment" not in q_low:
                    return (9, len(t))
                if "corporation" in t_low and "corporation" not in q_low:
                    return (8, len(t))

                if t_low == q_low:
                    return (0, len(t))
                for cand in candidates:
                    if t_low == cand.lower():
                        return (1, len(t))
                if t_low.startswith(q_low):
                    return (2, len(t))
                if q_low in t_low:
                    return (3, len(t))
                return (4, len(t))

            collected_titles.sort(key=_rank_title)

            # Fetch extracts for top candidates
            for title, page_url in collected_titles[:top_k]:
                sum_url = "https://en.wikipedia.org/api/rest_v1/page/summary/" + requests.utils.quote(title)
                sr = requests.get(sum_url, headers=self.headers, timeout=3.0)
                if sr.status_code == 200:
                    sdata = sr.json()
                    extract = sdata.get("extract") or sdata.get("description", "")
                    if extract:
                        results.append(
                            SearchItem(
                                title=title,
                                url=page_url,
                                snippet=extract.strip(),
                                source="wikipedia"
                            )
                        )
                else:
                    results.append(
                        SearchItem(
                            title=title,
                            url=page_url,
                            snippet=f"Wikipedia article on {title}.",
                            source="wikipedia"
                        )
                    )

            # 3. Fallback to MediaWiki Full-Text search if OpenSearch yielded nothing
            if not results:
                import re
                url = "https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch=" + requests.utils.quote(query) + f"&srlimit={top_k}&format=json"
                r = requests.get(url, headers=self.headers, timeout=3.5)
                if r.status_code == 200:
                    rdata = r.json()
                    for item in rdata.get("query", {}).get("search", []):
                        title = item.get("title", "")
                        if title.lower() in seen_titles:
                            continue
                        seen_titles.add(title.lower())
                        clean_snippet = re.sub(r'<[^>]+>', '', item.get("snippet", ""))
                        page_url = f"https://en.wikipedia.org/wiki/{requests.utils.quote(title)}"
                        results.append(
                            SearchItem(
                                title=title,
                                url=page_url,
                                snippet=clean_snippet.strip(),
                                source="wikipedia"
                            )
                        )
                        if len(results) >= top_k:
                            break
        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000.0
            return SearchResult(query=query, results=[], total_found=0, latency_ms=latency, error=str(e))

        latency = (time.perf_counter() - t0) * 1000.0
        return SearchResult(query=query, results=results[:top_k], total_found=len(results), latency_ms=latency)




class DuckDuckGoSearchProvider(BaseSearchProvider):
    """
    Robust DuckDuckGo search provider combining Instant Answer API and HTML fallback.
    """
    def __init__(self):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
        }

    def search(self, query: str, top_k: int = 5) -> SearchResult:
        t0 = time.perf_counter()
        results = []
        try:
            # 1. Try DuckDuckGo Instant Answer API
            api_url = f"https://api.duckduckgo.com/?q={requests.utils.quote(query)}&format=json&no_html=1&skip_disambig=1"
            r = requests.get(api_url, headers=self.headers, timeout=3.0)
            if r.status_code == 200:
                data = r.json()
                abstract = data.get("AbstractText", "").strip()
                heading = data.get("Heading", "").strip()
                abstract_url = data.get("AbstractURL", "").strip()
                
                if abstract:
                    results.append(
                        SearchItem(
                            title=heading or f"DuckDuckGo: {query}",
                            url=abstract_url or "https://duckduckgo.com/?q=" + requests.utils.quote(query),
                            snippet=abstract,
                            source="duckduckgo"
                        )
                    )
                
                # Check related topics
                for topic in data.get("RelatedTopics", [])[:top_k]:
                    if isinstance(topic, dict) and topic.get("Text"):
                        results.append(
                            SearchItem(
                                title=topic.get("Text", "")[:60] + "...",
                                url=topic.get("FirstURL", "https://duckduckgo.com"),
                                snippet=topic.get("Text", ""),
                                source="duckduckgo"
                            )
                        )
            
            # 2. If API was empty, try HTML parse fallback
            if not results:
                html_url = "https://html.duckduckgo.com/html/"
                res = requests.post(html_url, data={"q": query}, headers=self.headers, timeout=3.5)
                if res.status_code == 200:
                    from bs4 import BeautifulSoup
                    soup = BeautifulSoup(res.text, "html.parser")
                    links = soup.find_all("a", class_="result__url", limit=top_k*2)
                    snippets = soup.find_all("a", class_="result__snippet", limit=top_k*2)
                    titles = soup.find_all("a", class_="result__a", limit=top_k*2)
                    
                    seen_urls = set()
                    for t, l, s in zip(titles, links, snippets):
                        raw_url = l.get("href", "").strip()
                        if raw_url.startswith("//"):
                            raw_url = "https:" + raw_url
                        elif not raw_url.startswith("http"):
                            raw_url = "https://" + raw_url
                            
                        if raw_url in seen_urls:
                            continue
                        seen_urls.add(raw_url)
                        
                        results.append(
                            SearchItem(
                                title=t.get_text(strip=True),
                                url=raw_url,
                                snippet=s.get_text(strip=True),
                                source="duckduckgo"
                            )
                        )
                        if len(results) >= top_k:
                            break
        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000.0
            return SearchResult(query=query, results=[], total_found=0, latency_ms=latency, error=str(e))
            
        latency = (time.perf_counter() - t0) * 1000.0
        return SearchResult(query=query, results=results[:top_k], total_found=len(results), latency_ms=latency)


class APIWebSearchProvider(BaseSearchProvider):
    """
    Configurable search provider using SEARCH_API_URL and SEARCH_API_KEY environment variables.
    """
    def __init__(self, api_url: Optional[str] = None, api_key: Optional[str] = None):
        self.api_url = api_url or os.environ.get("SEARCH_API_URL")
        self.api_key = api_key or os.environ.get("SEARCH_API_KEY")

    def search(self, query: str, top_k: int = 5) -> SearchResult:
        if not self.api_url:
            # Fallback to DuckDuckGo if no external custom search API is provided
            return DuckDuckGoSearchProvider().search(query, top_k=top_k)

        t0 = time.perf_counter()
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            res = requests.get(self.api_url, params={"q": query, "limit": top_k}, headers=headers, timeout=4.0)
            if res.status_code == 200:
                data = res.json()
                raw_results = data.get("results", data.get("organic_results", []))
                results = []
                for item in raw_results[:top_k]:
                    results.append(
                        SearchItem(
                            title=item.get("title", "Search Result"),
                            url=item.get("url", item.get("link", "")),
                            snippet=item.get("snippet", item.get("description", "")),
                            source="search_api"
                        )
                    )
                latency = (time.perf_counter() - t0) * 1000.0
                return SearchResult(query=query, results=results, total_found=len(results), latency_ms=latency)
            else:
                latency = (time.perf_counter() - t0) * 1000.0
                return SearchResult(query=query, results=[], total_found=0, latency_ms=latency, error=f"HTTP {res.status_code}")
        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000.0
            return SearchResult(query=query, results=[], total_found=0, latency_ms=latency, error=str(e))


class HybridSearchProvider(BaseSearchProvider):
    """
    Multi-source search provider combining Wikipedia, DuckDuckGo, and Custom Search API.
    Provides unmatched factual coverage across any domain.
    """
    def __init__(self, providers: Optional[List[BaseSearchProvider]] = None):
        self.providers = providers or [
            WikipediaSearchProvider(),
            DuckDuckGoSearchProvider()
        ]
        # Include API search provider if environment has SEARCH_API_URL
        if os.environ.get("SEARCH_API_URL"):
            self.providers.insert(0, APIWebSearchProvider())

    def search(self, query: str, top_k: int = 5) -> SearchResult:
        t0 = time.perf_counter()
        all_results = []
        seen_urls = set()
        seen_titles = set()
        errors = []

        for provider in self.providers:
            try:
                res = provider.search(query, top_k=top_k)
                if res.results:
                    for item in res.results:
                        norm_url = item.url.lower().rstrip("/")
                        norm_title = item.title.lower().strip()
                        if norm_url in seen_urls or norm_title in seen_titles:
                            continue
                        seen_urls.add(norm_url)
                        seen_titles.add(norm_title)
                        all_results.append(item)
                elif res.error:
                    errors.append(res.error)
            except Exception as pe:
                errors.append(str(pe))

            if len(all_results) >= top_k:
                break

        latency = (time.perf_counter() - t0) * 1000.0
        return SearchResult(
            query=query,
            results=all_results[:top_k],
            total_found=len(all_results),
            latency_ms=latency,
            error="; ".join(errors) if (not all_results and errors) else None
        )

