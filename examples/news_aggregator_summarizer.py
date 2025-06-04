"""Simple news aggregator and summarizer example.

This script fetches the top Hacker News headlines and generates short
summaries for the first two articles.  Results are printed to the console
and also written to ``news_summaries.txt``.
"""

import httpx
import re
from html.parser import HTMLParser
from swarms import Agent
from swarms.prompts.summaries_prompts import SUMMARIZE_PROMPT


def fetch_hackernews_headlines(limit: int = 5):
    """Fetch top headlines from Hacker News using its public API."""
    try:
        ids = httpx.get(
            "https://hacker-news.firebaseio.com/v0/topstories.json", timeout=10
        ).json()
    except Exception:
        return []
    headlines = []
    for story_id in ids[:limit]:
        try:
            item = httpx.get(
                f"https://hacker-news.firebaseio.com/v0/item/{story_id}.json",
                timeout=10,
            ).json()
        except Exception:
            continue
        headlines.append({"title": item.get("title", "No title"), "url": item.get("url", "")})
    return headlines


class _ParagraphExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.in_p = False
        self.text_parts = []

    def handle_starttag(self, tag, attrs):
        if tag == "p":
            self.in_p = True

    def handle_endtag(self, tag):
        if tag == "p":
            self.in_p = False

    def handle_data(self, data):
        if self.in_p:
            self.text_parts.append(data.strip())


def _extract_paragraph_text(html: str) -> str:
    parser = _ParagraphExtractor()
    parser.feed(html)
    parser.close()
    return " ".join(t for t in parser.text_parts if t)


def fetch_article_content(url: str) -> str:
    """Retrieve article content from a URL using httpx."""
    try:
        res = httpx.get(url, timeout=10)
        res.raise_for_status()
    except Exception:
        return ""
    text = _extract_paragraph_text(res.text)
    if not text:
        text = re.sub("<[^>]+>", " ", res.text)
    return text.strip()


summarizer = Agent(
    agent_name="News-Summarizer",
    system_prompt="You summarize news articles succinctly.",
    max_loops=1,
    model_name="gpt-4o-mini",
    output_type="final",
)


def summarize_article(text: str) -> str:
    prompt = f"{SUMMARIZE_PROMPT}\n\n{text}"
    return summarizer.run(prompt)


if __name__ == "__main__":
    headlines = fetch_hackernews_headlines()
    print("Top Headlines:\n")
    for idx, headline in enumerate(headlines, 1):
        print(f"{idx}. {headline['title']}")

    summaries = []
    for article in headlines:
        content = fetch_article_content(article["url"])
        summary = summarize_article(content)
        summaries.append({"title": article["title"], "summary": summary})

    print("\nArticle Summaries:\n")
    for s in summaries:
        print(f"{s['title']}\n{s['summary']}\n")

    with open("news_summaries.txt", "w") as f:
        for s in summaries:
            f.write(f"{s['title']}\n{s['summary']}\n\n")
