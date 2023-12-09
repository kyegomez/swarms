# Arxiv Queries

Contributor: [Sihan Zhao](https://github.com/Sarah816)

## Tool Description
This Python-based tool offers a streamlined way to look up scientific articles on Arxiv.org. Named "Arxiv", this tool is particularly helpful when you need to answer questions about Physics, Mathematics, Computer Science, Quantitative Biology, Quantitative Finance, Statistics, Electrical Engineering, and Economics based on scientific articles from Arxiv.org.

### Tool Specifications

- **Name**: Arxiv
- **Purpose**: Look up for information from scientific articles on arxiv.org
- **Logo**: ![Arxiv Logo](https://your-app-url.com/.well-known/logo.png)
- **Contact Email**: hello@contact.com
- **Legal Information**: [Legal Information](hello@legal.com)

### Core Functionality

1. `get_arxiv_article_information`

    This method takes a search query and returns meta-information about the Arxiv articles that match this query. The method uses an API to search articles on Arxiv.org and returns details like the date of publication, title of the article, names of the authors, and the summary of the article.

    The method follows these steps:
    
    - It takes a query as a string input.
    - The query is passed to the Arxiv Search API.
    - The method fetches the top three results.
    - For each result, it collects information about the publication date, title, authors, and summary.
    - It returns this information as a string.
    
    If the search operation encounters an error, the method returns a message describing the Arxiv exception. If no suitable articles are found on Arxiv.org that match the query, it returns a message stating that no good Arxiv result was found.

### Constants

- **ARXIV_MAX_QUERY_LENGTH**: Maximum length of a query that can be passed to the Arxiv Search API. It's set to 300.
- **doc_content_chars_max**: Maximum characters of the Arxiv results to be returned. It's set to 4000.
- **top_k_results**: The maximum number of Arxiv Search results to be returned. It's set to 3.

Please note that the parameters can be optional and have their own default values. You should consult the method's documentation to understand the default behavior and the specific role of each parameter.