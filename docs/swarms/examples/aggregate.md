# Aggregate Multi-Agent Responses

The `aggregate` function allows you to run multiple agents concurrently on the same task and then synthesize their responses using an intelligent aggregator agent. This is useful for getting diverse perspectives on a problem and then combining them into a comprehensive analysis.

## Installation

You can get started by first installing swarms with the following command, or [click here for more detailed installation instructions](https://docs.swarms.world/en/latest/swarms/install/install/):

```bash
pip3 install -U swarms
``` 

## Environment Variables

```txt
OPENAI_API_KEY=""
ANTHROPIC_API_KEY=""
```

## Function Parameters

### `workers: List[Callable]` (Required)

A list of Agent instances that will work on the task concurrently. Each agent should be a callable object (typically an Agent instance).

### `task: str` (Required)

The task or question that all agents will work on simultaneously. This should be a clear, specific prompt that allows for diverse perspectives.

### `type: HistoryOutputType` (Optional, Default: "all")

Controls the format of the returned conversation history. Available options:

| Type | Description |
|------|-------------|
| **"all"** | Returns the complete conversation including all agent responses and the final aggregation |
| **"list"** | Returns the conversation as a list format |
| **"dict"** or **"dictionary"** | Returns the conversation as a dictionary format |
| **"string"** or **"str"** | Returns only the final aggregated response as a string |
| **"final"** or **"last"** | Returns only the final aggregated response |
| **"json"** | Returns the conversation in JSON format |
| **"yaml"** | Returns the conversation in YAML format |
| **"xml"** | Returns the conversation in XML format |
| **"dict-all-except-first"** | Returns dictionary format excluding the first message |
| **"str-all-except-first"** | Returns string format excluding the first message |
| **"basemodel"** | Returns the conversation as a base model object |
| **"dict-final"** | Returns dictionary format with only the final response |

### `aggregator_model_name: str` (Optional, Default: "anthropic/claude-3-sonnet-20240229")

The model to use for the aggregator agent that synthesizes all the individual agent responses. This should be a model capable of understanding and summarizing complex multi-agent conversations.

## How It Works

1. **Concurrent Execution**: All agents in the `workers` list run the same task simultaneously
2. **Response Collection**: Individual agent responses are collected into a conversation
3. **Intelligent Aggregation**: A specialized aggregator agent analyzes all responses and creates a comprehensive synthesis
4. **Formatted Output**: The final result is returned in the specified format

## Code Example

```python
from swarms.structs.agent import Agent
from swarms.structs.ma_blocks import aggregate


# Create specialized agents for different perspectives
agents = [
    Agent(
        agent_name="Sector-Financial-Analyst",
        agent_description="Senior financial analyst at BlackRock.",
        system_prompt="You are a financial analyst tasked with optimizing asset allocations for a $50B portfolio. Provide clear, quantitative recommendations for each sector.",
        max_loops=1,
        model_name="gpt-4o-mini",
        max_tokens=3000,
    ),
    Agent(
        agent_name="Sector-Risk-Analyst",
        agent_description="Expert risk management analyst.",
        system_prompt="You are a risk analyst responsible for advising on risk allocation within a $50B portfolio. Provide detailed insights on risk exposures for each sector.",
        max_loops=1,
        model_name="gpt-4o-mini",
        max_tokens=3000,
    ),
    Agent(
        agent_name="Tech-Sector-Analyst",
        agent_description="Technology sector analyst.",
        system_prompt="You are a tech sector analyst focused on capital and risk allocations. Provide data-backed insights for the tech sector.",
        max_loops=1,
        model_name="gpt-4o-mini",
        max_tokens=3000,
    ),
]

# Run the aggregate function
result = aggregate(
    workers=agents,
    task="What is the best sector to invest in?",
    type="all",  # Get complete conversation history
    aggregator_model_name="anthropic/claude-3-sonnet-20240229"
)

print(result)
```

## Code Example



## Use Cases

| Use Case | Description |
|----------|-------------|
| **Investment Analysis** | Get multiple financial perspectives on investment decisions |
| **Research Synthesis** | Combine insights from different research agents |
| **Problem Solving** | Gather diverse approaches to complex problems |
| **Content Creation** | Generate comprehensive content from multiple specialized agents |
| **Decision Making** | Get balanced recommendations from different expert perspectives |

## Error Handling

The function includes validation for:

- Required parameters (`task` and `workers`)

- Proper data types (workers must be a list of callable objects)

- Agent compatibility

## Performance Considerations

- All agents run concurrently, so total execution time is limited by the slowest agent

- The aggregator agent processes all responses, so consider response length and complexity

- Memory usage scales with the number of agents and their response sizes
