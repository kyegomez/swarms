import requests
from bs4 import BeautifulSoup


def arxiv_search(query):
    """
    Performs a semantic search on arxiv.org for the given query.

    Args:
      query: The query to search for.

    Returns:
      A list of search results.
    """

    # Make a request to arxiv.org
    response = requests.get(
        "http://export.arxiv.org/api/query",
        params={"search_query": query, "start": 0, "max_results": 10},
    )

    # Parse the response
    soup = BeautifulSoup(response.content, "html.parser")

    # Extract the search results
    results = []
    for result in soup.find_all("entry"):
        results.append(
            {
                "title": result.find("title").text,
                "author": result.find("author").text,
                "abstract": result.find("summary").text,
                "link": result.find("link")["href"],
            }
        )

    return results


search = arxiv_search("quantum computing")
print(search)
