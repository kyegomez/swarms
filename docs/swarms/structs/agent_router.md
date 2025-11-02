# AgentRouter Documentation

The `AgentRouter` class is an embedding-based routing system that intelligently matches tasks to the most appropriate specialized agent using cosine similarity on embeddings. It uses LiteLLM's embedding models to generate vector representations of both agents and tasks, enabling semantic matching for optimal agent selection.

Full Path: `from swarms.structs.agent_router import AgentRouter`

## Overview

The `AgentRouter` uses embedding-based semantic matching to route tasks to specialized agents. Each agent is represented by an embedding vector generated from its name, description, and system prompt. When a task is submitted, the router:

1. Generates an embedding vector for the task
2. Calculates cosine similarity between the task embedding and all agent embeddings
3. Returns the agent with the highest similarity score
4. Optionally updates agent embeddings with interaction history for improved matching over time

## Initialization Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_model` | `str` | `"text-embedding-ada-002"` | The embedding model to use for generating embeddings. Supports various models like `text-embedding-3-small`, `text-embedding-3-large`, `cohere/embed-english-v3.0`, `huggingface/microsoft/codebert-base`, etc. |
| `n_agents` | `int` | `1` | Number of agents to return in queries (currently supports returning the best match) |
| `api_key` | `Optional[str]` | `None` | API key for the embedding service. If not provided, will use environment variables. |
| `api_base` | `Optional[str]` | `None` | Custom API base URL for the embedding service. |
| `agents` | `Optional[List[AgentType]]` | `None` | List of agents to initialize the router with. Each agent should have `name`, `description`, and `system_prompt` attributes. |

## Methods

### `__init__`

Initialize the AgentRouter with embedding model configuration.

```python
router = AgentRouter(
    embedding_model="text-embedding-ada-002",
    n_agents=1,
    api_key=None,
    api_base=None,
    agents=None
)
```

### `add_agent(agent: AgentType) -> None`

Add a single agent to the router. The agent will be embedded using its name, description, and system prompt.

**Parameters:**

- `agent` (`AgentType`): The agent to add. Must have `name`, `description`, and `system_prompt` attributes.

**Raises:**

- `Exception`: If there's an error generating the embedding or adding the agent.

**Example:**

```python
from swarms.structs.agent import Agent
from swarms.structs.agent_router import AgentRouter

router = AgentRouter()

symptom_checker = Agent(
    agent_name="Symptom Checker",
    agent_description="Expert agent for initial triage and identifying possible causes based on symptom input.",
    system_prompt="You are a medical symptom checker agent..."
)

router.add_agent(symptom_checker)
```

### `add_agents(agents: List[Union[AgentType, Callable, Any]]) -> None`

Add multiple agents to the router at once.

**Parameters:**

- `agents` (`List[Union[AgentType, Callable, Any]]`): List of agents to add.

**Example:**

```python
agents = [agent1, agent2, agent3]
router.add_agents(agents)
```

### `find_best_agent(task: str, *args, **kwargs) -> Optional[AgentType]`

Find the best matching agent for a given task using cosine similarity on embeddings.

**Parameters:**

| Parameter   | Type    | Description                                                             |
|-------------|---------|-------------------------------------------------------------------------|
| `task`      | `str`   | The task description to match against agents.                           |
| `*args`     |         | Additional arguments (unused, kept for compatibility).                  |
| `**kwargs`  |         | Additional keyword arguments (unused, kept for compatibility).          |

**Returns:**

- `Optional[AgentType]`: The best matching agent if found, `None` otherwise.

**Raises:**

- `Exception`: If there's an error finding the best agent.

**Example:**

```python
best_agent = router.find_best_agent(
    "I have a headache, fever, and cough. What could be wrong?"
)

if best_agent:
    result = best_agent.run("I have a headache, fever, and cough. What could be wrong?")
    print(result)
```

### `run(task: str) -> Optional[AgentType]`

Convenience method that calls `find_best_agent`. Run the agent router on a given task.

**Parameters:**

- `task` (`str`): The task description to match against agents.

**Returns:**

- `Optional[AgentType]`: The best matching agent if found, `None` otherwise.

**Example:**

```python
result = router.run("I have a headache, fever, and cough. What could be wrong?")
print(result.agent_name)
```

### `update_agent_history(agent_name: str) -> None`

Update the agent's embedding in the router with its interaction history. This allows the router to learn from past interactions and improve matching over time.

**Parameters:**

- `agent_name` (`str`): The name of the agent to update.

**Note:** This method updates the agent's embedding to include its conversation history, which can improve future routing decisions based on what the agent has learned or discussed.

**Example:**

```python
# After an agent has processed a task and has conversation history
best_agent = router.run("Analyze patient symptoms")
result = best_agent.run("Analyze patient symptoms")

# Update the agent's embedding with its interaction history
router.update_agent_history(best_agent.name)
```

### `_generate_embedding(text: str) -> List[float]`

Internal method to generate embedding for the given text using the specified embedding model.

**Parameters:**

- `text` (`str`): The text to generate embedding for.

**Returns:**

- `List[float]`: The embedding vector as a list of floats.

### `_cosine_similarity(vec1: List[float], vec2: List[float]) -> float`

Internal method to calculate cosine similarity between two vectors.

**Parameters:**

- `vec1` (`List[float]`): First vector.

- `vec2` (`List[float]`): Second vector.

**Returns:**

- `float`: Cosine similarity score between -1 and 1.

## Usage Examples

### Basic Medical Use Case

```python
from swarms.structs.agent import Agent
from swarms.structs.agent_router import AgentRouter

# Initialize the router
agent_router = AgentRouter(
    embedding_model="text-embedding-ada-002",
    n_agents=1,
    agents=[
        Agent(
            agent_name="Symptom Checker",
            agent_description="Expert agent for initial triage and identifying possible causes based on symptom input.",
            system_prompt=(
                "You are a medical symptom checker agent. Ask clarifying questions "
                "about the patient's symptoms, duration, severity, and related risk factors. "
                "Provide a list of possible conditions and next diagnostic steps, but do not make a final diagnosis."
            ),
        ),
        Agent(
            agent_name="Diagnosis Synthesizer",
            agent_description="Agent specializing in synthesizing diagnostic possibilities from patient information and medical history.",
            system_prompt=(
                "You are a medical diagnosis assistant. Analyze the patient's reported symptoms, medical history, and any test results. "
                "Provide a differential diagnosis, and highlight the most likely conditions a physician should consider."
            ),
        ),
        Agent(
            agent_name="Lab Interpretation Expert",
            agent_description="Specializes in interpreting laboratory and imaging results for diagnostic support.",
            system_prompt=(
                "You are a medical lab and imaging interpretation agent. Take the patient's test results, imaging findings, and vitals, "
                "and interpret them in context of their symptoms. Suggest relevant follow-up diagnostics or considerations for the physician."
            ),
        ),   
    ],
)

# Route a task to the best agent
result = agent_router.run(
    "I have a headache, fever, and cough. What could be wrong?"
)

# Use the selected agent
if result:
    print(f"Selected agent: {result.agent_name}")
    response = result.run("I have a headache, fever, and cough. What could be wrong?")
    print(response)
    
    # Update agent history after interaction
    agent_router.update_agent_history(result.name)
```

### Finance Analysis Use Case

```python
from swarms.structs.agent import Agent
from swarms.structs.agent_router import AgentRouter

# Define specialized finance agents
finance_agents = [
    Agent(
        agent_name="Market Analyst",
        agent_description="Analyzes market trends and provides trading insights",
        system_prompt="You are a financial market analyst specializing in market data analysis and trading insights."
    ),
    Agent(
        agent_name="Risk Assessor",
        agent_description="Evaluates financial risks and compliance requirements",
        system_prompt="You are a risk assessment specialist focusing on financial risk analysis and compliance."
    ),
    Agent(
        agent_name="Investment Strategist",
        agent_description="Provides investment strategies and portfolio management",
        system_prompt="You are an investment strategy specialist developing long-term financial planning strategies."
    )
]

# Initialize router
finance_router = AgentRouter(
    embedding_model="text-embedding-ada-002",
    agents=finance_agents
)

# Route tasks
market_task = finance_router.run("Analyze current market conditions for technology sector")
if market_task:
    market_analysis = market_task.run("Analyze current market conditions for technology sector")
    
risk_task = finance_router.run("Assess the risk profile for a cryptocurrency investment")
if risk_task:
    risk_assessment = risk_task.run("Assess the risk profile for a cryptocurrency investment")
```

### Custom Embedding Model

```python
from swarms.structs.agent import Agent
from swarms.structs.agent_router import AgentRouter

# Use a different embedding model
router = AgentRouter(
    embedding_model="text-embedding-3-large",  # OpenAI's newer model
    n_agents=1,
    api_key="your-api-key",  # Optional: specify API key
    api_base=None  # Optional: use custom API base URL
)

# Or use Cohere embeddings
cohere_router = AgentRouter(
    embedding_model="cohere/embed-english-v3.0",
    api_key="your-cohere-api-key"
)
```

### Dynamic Agent Addition

```python
from swarms.structs.agent import Agent
from swarms.structs.agent_router import AgentRouter

# Initialize empty router
router = AgentRouter(embedding_model="text-embedding-ada-002")

# Add agents dynamically
agent1 = Agent(
    agent_name="Data Extractor",
    agent_description="Extracts structured data from documents",
    system_prompt="You are a data extraction specialist..."
)
router.add_agent(agent1)

agent2 = Agent(
    agent_name="Document Summarizer",
    agent_description="Creates concise summaries of documents",
    system_prompt="You are a document summarization expert..."
)
router.add_agent(agent2)

# Now use the router
best_agent = router.run("Extract key information from this contract")
```

## Best Practices

### 1. Agent Descriptions

Provide clear, specific descriptions for agents to improve matching accuracy:

```python
# Good: Specific description
Agent(
    agent_name="Medical Diagnostician",
    agent_description="Specializes in analyzing patient symptoms, medical history, and test results to provide differential diagnoses for common medical conditions.",
    system_prompt="..."
)

# Poor: Vague description
Agent(
    agent_name="Doctor",
    agent_description="Helps with medical stuff",
    system_prompt="..."
)
```

### 2. Task Descriptions

When routing tasks, provide detailed descriptions that match the agent's specialization:

```python
# Good: Detailed task description
task = """
Analyze the following patient case:
- Symptoms: Persistent headache, fever (38.5°C), cough for 3 days
- Medical history: No known allergies, non-smoker
- Recent travel: None
Provide a differential diagnosis with likelihood ratings.
"""

# Poor: Too vague
task = "Help with medical issue"
```

### 3. Update Agent History

After agents interact with tasks, update their embeddings to improve future routing:

```python
best_agent = router.run(task)
result = best_agent.run(task)

# Update the agent's embedding with interaction history
router.update_agent_history(best_agent.name)
```

### 4. Error Handling

Always handle cases where no agent is found:

```python
best_agent = router.run(task)

if best_agent:
    result = best_agent.run(task)
    print(f"Task completed by {best_agent.name}: {result}")
else:
    print("No suitable agent found for this task")
```

### 5. Embedding Model Selection

Choose embedding models based on your needs:

| Provider      | Model Name(s)                                             | Recommended Use Case/Notes                     |
|---------------|-----------------------------------------------------------|------------------------------------------------|
| **OpenAI**    | `text-embedding-ada-002`<br>`text-embedding-3-small`<br>`text-embedding-3-large` | Ada-002 is default, cost-effective; 3-small/large offer higher accuracy |
| **Cohere**    | `cohere/embed-english-v3.0`                              | Excellent for English text                     |
| **HuggingFace** | `huggingface/microsoft/codebert-base`                  | Best for code-related tasks                    |

## Performance Considerations

1. **Embedding Generation**: The first time agents are added, embeddings are generated which can take a few seconds per agent. Consider pre-generating embeddings for frequently used agents.

2. **API Rate Limits**: Be aware of rate limits when using embedding APIs, especially when adding many agents or processing many tasks.

3. **Caching**: The router doesn't cache task embeddings. If you're routing the same task multiple times, consider caching the result.

4. **Batch Processing**: For processing multiple tasks, consider batching or using concurrent execution where appropriate.

## Troubleshooting

### No Agent Found

If `find_best_agent` returns `None`:

1. Check that agents have been added: `len(router.agents) > 0`
2. Verify agent descriptions are clear and specific
3. Ensure the task description is detailed enough to match an agent
4. Check logs for embedding generation errors

### Embedding Errors

If you encounter embedding errors:

1. Verify your API key is set correctly (environment variable or passed to constructor)
2. Check that the embedding model name is correct and supported by LiteLLM
3. Ensure network connectivity to the embedding API
4. Check API rate limits and quotas

### Low Similarity Scores

If agents are being matched but with low similarity scores:

1. Improve agent descriptions to be more specific
2. Enhance system prompts with more relevant keywords
3. Consider using a different embedding model
4. Update agent history after interactions to improve matching
