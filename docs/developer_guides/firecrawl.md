# Firecrawl Tool


The Firecrawl tool is a powerful web crawling utility that integrates seamlessly with Swarms agents to extract, analyze, and process content from websites. It leverages the Firecrawl API to crawl entire websites, extract structured data, and provide comprehensive content analysis for various use cases including marketing, research, content creation, and data analysis.

### Key Features

| Feature                     | Description                                                                                   |
|-----------------------------|-----------------------------------------------------------------------------------------------|
| **Complete Site Crawling**  | Crawl entire websites and extract content from multiple pages                                 |
| **Structured Data Extraction** | Automatically parse and structure web content                                              |
| **Agent Integration**       | Works seamlessly with Swarms agents for intelligent content processing                        |
| **Marketing Copy Analysis** | Specialized for analyzing and improving marketing content                                     |
| **Content Optimization**    | Identify and enhance key value propositions and calls-to-action                              |

## Prerequisites

Before getting started, you'll need:

1. **Python 3.8+** installed on your system
2. **Firecrawl API Key** from [firecrawl.dev/app](https://www.firecrawl.dev/app)
3. **OpenAI API Key** for agent functionality

## Install

```bash
pip3 install -U swarms swarms-tools
```

## ENV

```txt
FIRECRAWL_API_KEY=""
OPENAI_API_KEY=""
```

## Usage

```python
from swarms_tools import crawl_entire_site_firecrawl

from swarms import Agent

agent = Agent(
    agent_name="Marketing Copy Improver",
    model_name="gpt-4.1",
    tools=[crawl_entire_site_firecrawl],
    dynamic_context_window=True,
    dynamic_temperature_enabled=True,
    max_loops=1,
    system_prompt=(
        "You are a world-class marketing copy improver. "
        "Given a website URL, your job is to crawl the entire site, analyze all marketing copy, "
        "and rewrite it to maximize clarity, engagement, and conversion. "
        "Return the improved marketing copy in a structured, easy-to-read format. "
        "Be concise, persuasive, and ensure the tone matches the brand. "
        "Highlight key value propositions and calls to action."
    ),
)

out = agent.run(
    "Crawl 2-3 pages of swarms.ai and improve the marketing copy found on those pages. Return the improved copy in a structured format."
)
print(out)
```