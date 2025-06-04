import requests
from bs4 import BeautifulSoup
from swarms import Agent
from swarms.prompts.summaries_prompts import SUMMARIZE_PROMPT


def fetch_hackernews_headlines(limit: int = 5):
    """Fetch top headlines from Hacker News."""
    resp = requests.get("https://news.ycombinator.com")
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")
    headlines = []
    for item in soup.select("tr.athing")[:limit]:
        link = item.select_one("span.titleline a")
        if link:
            headlines.append({"title": link.get_text(), "url": link["href"]})
    return headlines


def fetch_article_content(url: str) -> str:
    """Pull text content from an article URL."""
    try:
        res = requests.get(url, timeout=10)
        res.raise_for_status()
    except Exception:
        return ""
    soup = BeautifulSoup(res.text, "html.parser")
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()
    text = " ".join(p.get_text() for p in soup.find_all("p"))
    return text.strip()


summarizer = Agent(
    agent_name="News-Summarizer",
    system_prompt="You summarize news articles succinctly.",
    max_loops=1,
    model_name="gpt-4o-mini",
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
    for article in headlines[:2]:
        content = fetch_article_content(article["url"])
        summary = summarize_article(content)
        summaries.append({"title": article["title"], "summary": summary})

    print("\nArticle Summaries:\n")
    for s in summaries:
        print(f"{s['title']}\n{s['summary']}\n")

    with open("news_summaries.txt", "w") as f:
        for s in summaries:
            f.write(f"{s['title']}\n{s['summary']}\n\n")
