# Web Scraper Agents

Web scraper agents are specialized AI agents that can automatically extract and process information from websites. These agents combine the power of large language models with web scraping tools to intelligently gather, analyze, and structure data from the web.

Web scraper agents are AI-powered tools that can:

| Capability                                                                 | Description                                                                                   |
|----------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------|
| **Automatically navigate websites**                                        | Extract relevant information from web pages                                                   |
| **Parse and structure data**                                               | Convert HTML content into readable, structured formats                                        |
| **Handle dynamic content**                                                 | Process JavaScript-rendered pages and dynamic website elements                                |
| **Provide intelligent summaries and analysis**                             | Generate summaries and analyze the scraped content                                            |
| **Scale to multiple websites simultaneously**                              | Scrape and process data from several websites at once for comprehensive research              |


## Install

```bash
pip3 install -U swarms swarms-tools
```

## Environment Setup

```bash
OPENAI_API_KEY="your_openai_api_key_here"
```

## Basic Usage

Here's a simple example of how to create a web scraper agent:

```python
from swarms import Agent
from swarms_tools import scrape_and_format_sync

agent = Agent(
    agent_name="Web Scraper Agent",
    model_name="gpt-4o-mini",
    tools=[scrape_and_format_sync],
    dynamic_context_window=True,
    dynamic_temperature_enabled=True,
    max_loops=1,
    system_prompt="You are a web scraper agent. You are given a URL and you need to scrape the website and return the data in a structured format. The format type should be full",
)

out = agent.run(
    "Scrape swarms.ai website and provide a full report of the company  does. The format type should be full."
)
print(out)
```

## Scraping Multiple Sites

For comprehensive research, you can scrape multiple websites simultaneously using batch execution:

```python
from swarms.structs.multi_agent_exec import batched_grid_agent_execution
from swarms_tools import scrape_and_format_sync
from swarms import Agent

agent = Agent(
    agent_name="Web Scraper Agent",
    model_name="gpt-4o-mini",
    tools=[scrape_and_format_sync],
    dynamic_context_window=True,
    dynamic_temperature_enabled=True,
    max_loops=1,
    system_prompt="You are a web scraper agent. You are given a URL and you need to scrape the website and return the data in a structured format. The format type should be full",
)

out = batched_grid_agent_execution(
    agents=[agent, agent],
    tasks=[
        "Scrape swarms.ai website and provide a full report of the company's mission, products, and team. The format type should be full.",
        "Scrape langchain.com website and provide a full report of the company's mission, products, and team. The format type should be full.",
    ],
)

print(out)
```

## Conclusion

Web scraper agents combine AI with advanced automation to efficiently gather and process web data at scale. As you master the basics, explore features like batch processing and custom tools to unlock the full power of AI-driven web scraping.
