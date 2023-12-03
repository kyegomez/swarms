from swarms.tools.tool import tool
from swarms.tools.tool_func_doc_scraper import scrape_tool_func_docs


@tool
def search_api(query: str) -> str:
    """Search API

    Args:
        query (str): _description_

    Returns:
        str: _description_
    """
    print(f"Searching API for {query}")
    
    
tool_docs = scrape_tool_func_docs(search_api)
print(tool_docs)