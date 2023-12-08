from ..tool import Tool
from typing import Any
import arxiv


def build_tool(config) -> Tool:
    tool = Tool(
        "Arxiv",
        "Look up for information from scientific articles on arxiv.org",
        name_for_model="Arxiv",
        description_for_model=(
            "Search information from Arxiv.org "
            "Useful for when you need to answer questions about Physics, Mathematics, "
            "Computer Science, Quantitative Biology, Quantitative Finance, Statistics, "
            "Electrical Engineering, and Economics "
            "from scientific articles on arxiv.org. "
            "Input should be a search query."
        ),
        logo_url="https://your-app-url.com/.well-known/logo.png",
        contact_email="hello@contact.com",
        legal_info_url="hello@legal.com",
    )

    arxiv_exceptions: Any  # :meta private:
    top_k_results: int = 3
    ARXIV_MAX_QUERY_LENGTH = 300
    doc_content_chars_max: int = 4000

    @tool.get("/get_arxiv_article_information")
    def get_arxiv_article_information(query: str):
        """Run Arxiv search and get the article meta information."""
        param = {"q": query}
        try:
            results = arxiv.Search(  # type: ignore
                query[:ARXIV_MAX_QUERY_LENGTH], max_results=top_k_results
            ).results()
        except arxiv_exceptions as ex:
            return f"Arxiv exception: {ex}"
        docs = [
            f"Published: {result.updated.date()}\nTitle: {result.title}\n"
            f"Authors: {', '.join(a.name for a in result.authors)}\n"
            f"Summary: {result.summary}"
            for result in results
        ]
        if docs:
            return "\n\n".join(docs)[:doc_content_chars_max]
        else:
            return "No good Arxiv Result was found"

    return tool
