# pip install swarms bs4 requests

import re
from typing import Any, Dict
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

from swarms import Agent

load_dotenv()

# Custom system prompt for financial sentiment analysis
FINANCIAL_SENTIMENT_SYSTEM_PROMPT = """
You are an expert financial analyst specializing in sentiment analysis of financial news and content. Your task is to:

1. Analyze financial content for bullish or bearish sentiment
2. Provide a numerical sentiment score between 0.0 (extremely bearish) and 1.0 (extremely bullish) where:
   - 0.0-0.2: Extremely bearish (strong negative outlook)
   - 0.2-0.4: Bearish (negative outlook)
   - 0.4-0.6: Neutral (balanced or unclear outlook)
   - 0.6-0.8: Bullish (positive outlook)
   - 0.8-1.0: Extremely bullish (strong positive outlook)

3. Provide detailed rationale for your sentiment score by considering:
   - Market indicators and metrics mentioned
   - Expert opinions and quotes
   - Historical comparisons
   - Industry trends and context
   - Risk factors and potential challenges
   - Growth opportunities and positive catalysts
   - Overall market sentiment and broader economic factors

Your analysis should be:
- Objective and data-driven
- Based on factual information present in the content
- Free from personal bias or speculation
- Considering both explicit and implicit sentiment indicators
- Taking into account the broader market context

For each analysis, structure your response as a clear sentiment score backed by comprehensive reasoning that explains why you arrived at that specific rating.
"""


class ArticleExtractor:
    """Class to handle article content extraction and cleaning."""

    # Common financial news domains and their article content selectors
    DOMAIN_SELECTORS = {
        "seekingalpha.com": {"article": "div#SA-content"},
        "finance.yahoo.com": {"article": "div.caas-body"},
        "reuters.com": {
            "article": "div.article-body__content__17Yit"
        },
        "bloomberg.com": {"article": "div.body-content"},
        "marketwatch.com": {"article": "div.article__body"},
        # Add more domains and their selectors as needed
    }

    @staticmethod
    def get_domain(url: str) -> str:
        """Extract domain from URL."""
        return urlparse(url).netloc.lower()

    @staticmethod
    def clean_text(text: str) -> str:
        """Clean extracted text content."""
        # Remove extra whitespace
        text = re.sub(r"\s+", " ", text)
        # Remove special characters but keep basic punctuation
        text = re.sub(r"[^\w\s.,!?-]", "", text)
        # Remove multiple periods
        text = re.sub(r"\.{2,}", ".", text)
        return text.strip()

    @classmethod
    def extract_article_content(
        cls, html_content: str, domain: str
    ) -> str:
        """Extract article content using domain-specific selectors."""
        soup = BeautifulSoup(html_content, "html.parser")

        # Remove unwanted elements
        for element in soup.find_all(
            ["script", "style", "nav", "header", "footer", "iframe"]
        ):
            element.decompose()

        # Try domain-specific selector first
        if domain in cls.DOMAIN_SELECTORS:
            selector = cls.DOMAIN_SELECTORS[domain]["article"]
            content = soup.select_one(selector)
            if content:
                return cls.clean_text(content.get_text())

        # Fallback to common article containers
        article_containers = [
            "article",
            '[role="article"]',
            ".article-content",
            ".post-content",
            ".entry-content",
            "#main-content",
        ]

        for container in article_containers:
            content = soup.select_one(container)
            if content:
                return cls.clean_text(content.get_text())

        # Last resort: extract all paragraph text
        paragraphs = soup.find_all("p")
        if paragraphs:
            return cls.clean_text(
                " ".join(p.get_text() for p in paragraphs)
            )

        return cls.clean_text(soup.get_text())


def fetch_url_content(url: str) -> Dict[str, Any]:
    """
    Fetch and extract content from a financial news URL.

    Args:
        url (str): The URL of the financial news article

    Returns:
        Dict[str, Any]: Dictionary containing extracted content and metadata
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()

        domain = ArticleExtractor.get_domain(url)
        content = ArticleExtractor.extract_article_content(
            response.text, domain
        )

        # Extract title if available
        soup = BeautifulSoup(response.text, "html.parser")
        title = soup.title.string if soup.title else None

        return {
            "title": title,
            "content": content,
            "domain": domain,
            "url": url,
            "status": "success",
        }
    except Exception as e:
        return {
            "content": f"Error fetching URL content: {str(e)}",
            "status": "error",
            "url": url,
        }


tools = [
    {
        "type": "function",
        "function": {
            "name": "analyze_sentiment",
            "description": "Analyze the sentiment of financial content and provide a bullish/bearish rating with rationale.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sentiment_score": {
                        "type": "number",
                        "description": "A score from 0.0 (extremely bearish) to 1.0 (extremely bullish)",
                    },
                    "rationale": {
                        "type": "string",
                        "description": "Detailed explanation of the sentiment analysis",
                    },
                },
                "required": ["sentiment_score", "rationale"],
            },
        },
    }
]

# Initialize the agent
agent = Agent(
    agent_name="Financial-Sentiment-Analyst",
    agent_description="Expert financial sentiment analyzer that provides detailed bullish/bearish analysis of financial content",
    system_prompt=FINANCIAL_SENTIMENT_SYSTEM_PROMPT,
    max_loops=1,
    tools_list_dictionary=tools,
    output_type="final",
    model_name="gpt-4.1",
)


def run_sentiment_agent(url: str) -> Dict[str, Any]:
    """
    Run the sentiment analysis agent on a given URL.

    Args:
        url (str): The URL of the financial content to analyze

    Returns:
        Dict[str, Any]: Dictionary containing sentiment analysis results
    """
    article_data = fetch_url_content(url)

    if article_data["status"] == "error":
        return {"error": article_data["content"], "status": "error"}

    prompt = f"""
        Analyze the following financial article:
        Title: {article_data.get('title', 'N/A')}
        Source: {article_data['domain']}
        URL: {article_data['url']}

        Content:
        {article_data['content']}

        Please provide a detailed sentiment analysis with a score and explanation.
        """

    return agent.run(prompt)


if __name__ == "__main__":
    url = "https://finance.yahoo.com/"
    result = run_sentiment_agent(url)
    print(result)
