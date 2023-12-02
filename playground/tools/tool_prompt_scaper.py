from swarms.tools.tool import tool
from swarms.tools.tool_func_doc_scraper import scrape_tool_func_docs

# Define a tool by decorating a function with the tool decorator and providing a docstring


@tool(return_direct=True)
def search_api(query: str):
    """Search the web for the query

    Args:
        query (str): _description_

    Returns:
        _type_: _description_
    """
    return f"Search results for {query}"


# Scrape the tool func docs to prepare for injection into the agent prompt
out = scrape_tool_func_docs(search_api)
print(out)
