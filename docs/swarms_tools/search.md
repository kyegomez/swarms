# Search Tools Documentation

This documentation covers the search tools available in the `swarms-tools` package.

## Installation

```bash
pip3 install -U swarms-tools
```

## Environment Variables Required

Create a `.env` file in your project root with the following API keys:

```bash
# Bing Search API
BING_API_KEY=your_bing_api_key

# Google Search API
GOOGLE_API_KEY=your_google_api_key
GOOGLE_CX=your_google_cx_id
GEMINI_API_KEY=your_gemini_api_key

# Exa AI API
EXA_API_KEY=your_exa_api_key
```

## Tools Overview

### 1. Bing Search Tool

The Bing Search tool allows you to fetch web articles using the Bing Web Search API.

#### Function: `fetch_web_articles_bing_api`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| query | str | Yes | The search query to retrieve articles |

#### Example Usage:

```python
from swarms_tools.search import fetch_web_articles_bing_api

# Fetch articles about AI
results = fetch_web_articles_bing_api("swarms ai github")
print(results)
```

### 2. Exa AI Search Tool

The Exa AI tool is designed for searching research papers and academic content.

#### Function: `search_exa_ai`

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| query | str | Yes | "Latest developments in LLM capabilities" | Search query |
| num_results | int | No | 10 | Number of results to return |
| auto_prompt | bool | No | True | Whether to use auto-prompting |
| include_domains | List[str] | No | ["arxiv.org", "paperswithcode.com"] | Domains to include |
| exclude_domains | List[str] | No | [] | Domains to exclude |
| category | str | No | "research paper" | Category of search |

#### Example Usage:

```python
from swarms_tools.search import search_exa_ai

# Search for research papers
results = search_exa_ai(
    query="Latest developments in LLM capabilities",
    num_results=5,
    include_domains=["arxiv.org"]
)
print(results)
```

### 3. Google Search Tool

A comprehensive search tool that uses Google Custom Search API and includes content extraction and summarization using Gemini.

#### Class: `WebsiteChecker`

| Method | Parameters | Description |
|--------|------------|-------------|
| search | query: str | Main search function that fetches, processes, and summarizes results |

#### Example Usage:

```python
from swarms_tools.search import WebsiteChecker

# Initialize with an agent (required for summarization)
checker = WebsiteChecker(agent=your_agent_function)

# Perform search
async def search_example():
    results = await checker.search("who won elections 2024 us")
    print(results)

# For synchronous usage
from swarms_tools.search import search

results = search("who won elections 2024 us", agent=your_agent_function)
print(results)
```

## Features

- **Bing Search**: Fetch and parse web articles with structured output
- **Exa AI**: Specialized academic and research paper search
- **Google Search**: 
  - Custom search with content extraction
  - Concurrent URL processing
  - Content summarization using Gemini
  - Progress tracking
  - Automatic retry mechanisms
  - Results saved to JSON

## Dependencies

The tools automatically handle dependency installation, but here are the main requirements:

```python
aiohttp
asyncio
beautifulsoup4
google-generativeai
html2text
playwright
python-dotenv
rich
tenacity
```

## Error Handling

All tools include robust error handling:
- Automatic retries for failed requests
- Timeout handling
- Rate limiting consideration
- Detailed error messages

## Output Format

Each tool provides structured output:

- **Bing Search**: Returns formatted string with article details
- **Exa AI**: Returns JSON response with search results
- **Google Search**: Returns summarized content with sections:
  - Key Findings
  - Important Details
  - Sources

## Best Practices

1. Always store API keys in environment variables
2. Use appropriate error handling
3. Consider rate limits of the APIs
4. Cache results when appropriate
5. Monitor API usage and costs

## Limitations

- Bing Search: Limited to 4 articles per query
- Exa AI: Focused on academic content
- Google Search: Requires Gemini API for summarization

## Support

For issues and feature requests, please visit the [GitHub repository](https://github.com/swarms-tools). 