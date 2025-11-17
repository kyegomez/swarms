# Web Search with Exa


Exa is a powerful web search API that provides real-time access to current web information. It allows AI agents to search the internet and retrieve up-to-date information on any topic, making it an essential tool for agents that need current knowledge beyond their training data.

Key features of Exa:

| Feature                  | Description                                                        |
|--------------------------|--------------------------------------------------------------------|
| **Real-time search**     | Access the latest information from the web                         |
| **Semantic search**      | Find relevant results using natural language queries                |
| **Comprehensive coverage** | Search across billions of web pages                              |
| **Structured results**   | Get clean, formatted search results for easy processing            |
| **API integration**      | Simple REST API for seamless integration with AI applications       |

## Install

```bash
pip3 install -U swarms swarms-tools
```

## ENV

```txt
# Get your API key from exa
EXA_SEARCH_API=""

OPENAI_API_KEY=""

WORKSPACE_DIR=""
```

## Code

```python
from swarms import Agent
from swarms_tools import exa_search


agent = Agent(
    name="Exa Search Agent",
    llm="gpt-4o-mini",
    tools=[exa_search],
)

out = agent.run("What are the latest experimental treatments for diabetes?")
print(out)
```