# Advanced Research

An enhanced implementation of the orchestrator-worker pattern from Anthropic's paper, "How we built our multi-agent research system", built on top of the bleeding-edge multi-agent framework [swarms](https://github.com/kyegomez/swarms). Our implementation of this advanced research system leverages parallel execution, LLM-as-judge evaluation, and professional report generation with export capabilities.

**Repository**: [AdvancedResearch](https://github.com/The-Swarm-Corporation/AdvancedResearch)

## Installation

```bash
pip3 install -U advanced-research

# uv pip install -U advanced-research
```

## Environment Variables

```txt
# Exa Search API Key (Required for web search functionality)
EXA_API_KEY="your_exa_api_key_here"

# Anthropic API Key (For Claude models)
ANTHROPIC_API_KEY="your_anthropic_api_key_here"

# OpenAI API Key (For GPT models)  
OPENAI_API_KEY="your_openai_api_key_here"

# Worker Agent Configuration
WORKER_MODEL_NAME="gpt-4.1"
WORKER_MAX_TOKENS=8000

# Exa Search Configuration
EXA_SEARCH_NUM_RESULTS=2
EXA_SEARCH_MAX_CHARACTERS=100
```

**Note**: At minimum, you need `EXA_API_KEY` for web search functionality. For LLM functionality, you need either `ANTHROPIC_API_KEY` or `OPENAI_API_KEY`.

## Quick Start

### Basic Usage

```python
from advanced_research import AdvancedResearch

# Initialize the research system
research_system = AdvancedResearch(
    name="AI Research Team",
    description="Specialized AI research system",
    max_loops=1,
)

# Run research and get results
result = research_system.run(
    "What are the latest developments in quantum computing?"
)
print(result)
```