from __future__ import annotations

import html
import re
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import requests
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


SEARCH_URL = "https://html.duckduckgo.com/html/"
REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}
TRUSTED_DOMAIN_KEYWORDS = [
    ".gov",
    ".edu",
    "extension",
    "usda",
    "cabi.org",
    "plantwiseplusknowledgebank.org",
    "apsnet.org",
    "cimmyt.org",
    "fao.org",
]


@dataclass
class WebSource:
    title: str
    url: str
    snippet: str
    source_type: str = "web"
    score: float = 0.0


def _clean_text(text: str) -> str:
    text = html.unescape(text or "")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _extract_result_url(raw_href: str) -> str:
    if not raw_href:
        return ""

    parsed = urlparse(raw_href)
    if parsed.netloc.endswith("duckduckgo.com") and parsed.path == "/l/":
        uddg = parse_qs(parsed.query).get("uddg", [""])[0]
        return unquote(uddg)

    return raw_href


def _is_trusted_url(url: str) -> bool:
    hostname = urlparse(url).netloc.lower()
    return any(keyword in hostname for keyword in TRUSTED_DOMAIN_KEYWORDS)


def build_search_query(question: str) -> str:
    base = question.strip()
    advisory_terms = "crop disease management pesticide official extension"
    return f"{base} {advisory_terms}"


def search_trusted_sources(question: str, max_results: int = 6) -> list[WebSource]:
    query = build_search_query(question)
    response = requests.post(
        SEARCH_URL,
        data={"q": query},
        headers=REQUEST_HEADERS,
        timeout=20,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    sources: list[WebSource] = []

    for result in soup.select(".result"):
        link = result.select_one(".result__title a") or result.select_one("a.result__a")
        snippet_node = result.select_one(".result__snippet")
        if not link:
            continue

        url = _extract_result_url(link.get("href", ""))
        if not url or not _is_trusted_url(url):
            continue

        title = _clean_text(link.get_text(" ", strip=True))
        snippet = _clean_text(snippet_node.get_text(" ", strip=True) if snippet_node else "")
        if not title:
            continue

        sources.append(WebSource(title=title, url=url, snippet=snippet))
        if len(sources) >= max_results:
            break

    return sources


def fetch_page_passages(url: str, max_passages: int = 6) -> list[str]:
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=20)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "lxml")
    for tag in soup(["script", "style", "noscript", "header", "footer", "nav", "aside"]):
        tag.decompose()

    text_blocks: list[str] = []
    for node in soup.find_all(["p", "li"]):
        text = _clean_text(node.get_text(" ", strip=True))
        if 70 <= len(text) <= 450:
            text_blocks.append(text)

    unique_blocks: list[str] = []
    seen = set()
    for block in text_blocks:
        key = block.lower()
        if key in seen:
            continue
        seen.add(key)
        unique_blocks.append(block)
        if len(unique_blocks) >= max_passages:
            break

    return unique_blocks


def rank_passages(question: str, passages: Iterable[str], top_k: int = 5) -> list[tuple[str, float]]:
    passages = list(passages)
    if not passages:
        return []

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(passages + [question])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
    ranked_indexes = scores.argsort()[::-1][:top_k]

    ranked: list[tuple[str, float]] = []
    for index in ranked_indexes:
        score = float(scores[index])
        if score <= 0:
            continue
        ranked.append((passages[index], score))
    return ranked


def retrieve_web_evidence(question: str, max_sources: int = 4, max_passages: int = 5) -> dict:
    try:
        sources = search_trusted_sources(question, max_results=max_sources)
    except Exception as exc:
        return {
            "query": build_search_query(question),
            "sources": [],
            "passages": [],
            "error": f"Web search failed: {exc}",
        }

    source_payload = []
    evidence_passages = []

    for source in sources:
        try:
            passages = fetch_page_passages(source.url)
        except Exception:
            passages = []

        ranked = rank_passages(question, passages, top_k=2)
        top_passages = [text for text, _score in ranked][:2]
        if not top_passages and source.snippet:
            top_passages = [source.snippet]

        source_payload.append(
            {
                "title": source.title,
                "url": source.url,
                "snippet": source.snippet,
                "passages": top_passages,
                "source_type": "web",
            }
        )

        for passage in top_passages:
            evidence_passages.append(
                {
                    "title": source.title,
                    "url": source.url,
                    "text": passage,
                    "source_type": "web",
                }
            )
            if len(evidence_passages) >= max_passages:
                break

        if len(evidence_passages) >= max_passages:
            break

    return {
        "query": build_search_query(question),
        "sources": source_payload,
        "passages": evidence_passages,
        "error": None,
    }
