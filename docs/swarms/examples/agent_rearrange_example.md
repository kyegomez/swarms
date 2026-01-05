# AgentRearrange Examples

The `AgentRearrange` architecture enables flexible orchestration of multiple agents using custom flow patterns. You can define sequential execution (using `->`) and concurrent execution (using `,`) within the same workflow, allowing for sophisticated multi-agent task processing.

## Prerequisites

- Python 3.7+
- OpenAI API key or other supported LLM provider keys
- Swarms library

## Installation

```bash
pip3 install -U swarms
```

## Environment Variables

```plaintext
WORKSPACE_DIR="agent_workspace"
OPENAI_API_KEY=""
ANTHROPIC_API_KEY=""
GROQ_API_KEY=""
```

## Understanding Flow Patterns

AgentRearrange uses a simple but powerful syntax to define agent execution:

- **Sequential execution**: Use `->` to chain agents sequentially
  - Example: `"Researcher -> Writer -> Editor"`
  
- **Concurrent execution**: Use `,` to run agents in parallel
  - Example: `"Researcher1, Researcher2 -> Writer"`
  
- **Mixed patterns**: Combine both for complex workflows
  - Example: `"Researcher -> Writer1, Writer2 -> Editor"`

## Basic Usage

### 1. Simple Sequential Flow

```python
from swarms import Agent, AgentRearrange

# Create specialized agents
researcher = Agent(
    agent_name="Researcher",
    system_prompt="""You are a research specialist. Your tasks include:
    1. Gathering information from reliable sources
    2. Analyzing data and identifying key insights
    3. Synthesizing findings into clear summaries
    4. Verifying facts and checking sources
    5. Organizing information logically""",
    model_name="gpt-4.1",
    max_loops=1,
    temperature=0.7,
)

writer = Agent(
    agent_name="Writer",
    system_prompt="""You are a professional writer. Your responsibilities include:
    1. Creating clear and engaging content
    2. Structuring information logically
    3. Ensuring proper grammar and style
    4. Adapting tone for the target audience
    5. Incorporating research findings effectively""",
    model_name="gpt-4.1",
    max_loops=1,
    temperature=0.7,
)

editor = Agent(
    agent_name="Editor",
    system_prompt="""You are an expert editor. Your focus areas include:
    1. Reviewing content for clarity and accuracy
    2. Correcting grammar and spelling errors
    3. Improving sentence structure and flow
    4. Ensuring consistency in style and tone
    5. Verifying facts and checking sources""",
    model_name="gpt-4.1",
    max_loops=1,
    temperature=0.7,
)

# Define sequential flow: Researcher -> Writer -> Editor
flow = "Researcher -> Writer -> Editor"

# Create the AgentRearrange system
workflow = AgentRearrange(
    name="content-creation-workflow",
    agents=[researcher, writer, editor],
    flow=flow,
    max_loops=1,
    verbose=True,
)

# Execute the workflow
result = workflow.run(
    "Research and write a comprehensive article about the impact of AI on healthcare"
)
print(result)
```

### 2. Parallel Execution

```python
# Create multiple research agents
researcher1 = Agent(
    agent_name="Market-Researcher",
    system_prompt="Research market trends and competitive landscape",
    model_name="gpt-4.1",
    max_loops=1,
)

researcher2 = Agent(
    agent_name="Financial-Researcher",
    system_prompt="Research financial data and performance metrics",
    model_name="gpt-4.1",
    max_loops=1,
)

analyst = Agent(
    agent_name="Analyst",
    system_prompt="Synthesize research findings into actionable insights",
    model_name="gpt-4.1",
    max_loops=1,
)

# Define flow: Both researchers run in parallel, then analyst processes results
flow = "Market-Researcher, Financial-Researcher -> Analyst"

workflow = AgentRearrange(
    name="parallel-research-workflow",
    agents=[researcher1, researcher2, analyst],
    flow=flow,
    max_loops=1,
)

result = workflow.run("Analyze Tesla (TSLA) from market and financial perspectives")
```

### 3. Mixed Sequential and Parallel Flow

```python
# Create agents for a complex workflow
researcher = Agent(
    agent_name="Researcher",
    system_prompt="Conduct comprehensive research on the topic",
    model_name="gpt-4.1",
    max_loops=1,
)

writer1 = Agent(
    agent_name="Technical-Writer",
    system_prompt="Write technical content based on research",
    model_name="gpt-4.1",
    max_loops=1,
)

writer2 = Agent(
    agent_name="Marketing-Writer",
    system_prompt="Write marketing content based on research",
    model_name="gpt-4.1",
    max_loops=1,
)

editor = Agent(
    agent_name="Editor",
    system_prompt="Review and edit all written content",
    model_name="gpt-4.1",
    max_loops=1,
)

# Complex flow: Sequential -> Parallel -> Sequential
flow = "Researcher -> Technical-Writer, Marketing-Writer -> Editor"

workflow = AgentRearrange(
    name="mixed-workflow",
    agents=[researcher, writer1, writer2, editor],
    flow=flow,
    max_loops=1,
)

result = workflow.run("Create technical and marketing content about cloud computing")
```

## Sequential Awareness Feature

The `team_awareness` feature provides agents with context about their position in the workflow, enabling better coordination and task understanding.

### How Sequential Awareness Works

When enabled, agents automatically receive information about:
- **Agent ahead**: The agent that completed their task before them
- **Agent behind**: The agent that will receive their output next

This helps agents:
- Reference previous work more effectively
- Prepare output suitable for the next agent
- Understand their role in the larger workflow

### Example with Sequential Awareness

```python
from swarms import Agent, AgentRearrange

# Create agents
researcher = Agent(
    agent_name="Researcher",
    system_prompt="Research the topic thoroughly and provide comprehensive findings",
    model_name="gpt-4.1",
    max_loops=1,
)

writer = Agent(
    agent_name="Writer",
    system_prompt="Write content based on the research provided. Consider what the editor will need.",
    model_name="gpt-4.1",
    max_loops=1,
)

editor = Agent(
    agent_name="Editor",
    system_prompt="Edit the written content for clarity, accuracy, and style",
    model_name="gpt-4.1",
    max_loops=1,
)

# Create workflow with team awareness enabled
workflow = AgentRearrange(
    name="aware-workflow",
    agents=[researcher, writer, editor],
    flow="Researcher -> Writer -> Editor",
    team_awareness=True,  # Enable sequential awareness
    time_enabled=True,     # Enable timestamps
    message_id_on=True,   # Enable message IDs
    verbose=True,
)

# Get flow structure information
flow_structure = workflow.get_sequential_flow_structure()
print("Flow Structure:", flow_structure)

# Get awareness for specific agent
writer_awareness = workflow.get_agent_sequential_awareness("Writer")
print("Writer Awareness:", writer_awareness)

# Execute workflow
result = workflow.run("Research and write about quantum computing")
```

**What happens automatically:**
- **Researcher** runs first (no awareness info needed)
- **Writer** receives: "Sequential awareness: Agent ahead: Researcher | Agent behind: Editor"
- **Editor** receives: "Sequential awareness: Agent ahead: Writer"

## Advanced Features

### Output Formatting Options

The workflow supports multiple output formats:

- **"all"**: Complete conversation history (default)
- **"final"**: Only the final agent's output
- **"list"**: List of individual agent outputs
- **"dict"**: Dictionary with agent names as keys

```python
workflow = AgentRearrange(
    agents=[researcher, writer, editor],
    flow="Researcher -> Writer -> Editor",
    output_type="dict",  # Get structured dictionary output
)
```

### Human-in-the-Loop

Add human review points in your workflow:

```python
def human_reviewer(input_text: str) -> str:
    """Custom human-in-the-loop function"""
    print(f"\nReview this content:\n{input_text}\n")
    response = input("Enter your feedback or 'approve' to continue: ")
    return response if response else "approved"

workflow = AgentRearrange(
    agents=[researcher, writer, editor],
    flow="Researcher -> Writer -> H -> Editor",  # H = Human review point
    human_in_the_loop=True,
    custom_human_in_the_loop=human_reviewer,
)
```

### Memory System Integration

Integrate with memory systems for persistent storage:

```python
from swarms.memory import QdrantVectorDatabase

# Initialize memory system
memory = QdrantVectorDatabase(
    collection_name="agent_workflow_memory",
    path="./memory"
)

workflow = AgentRearrange(
    agents=[researcher, writer, editor],
    flow="Researcher -> Writer -> Editor",
    memory_system=memory,
    autosave=True,
)
```

### Custom Rules

Add system-wide rules that apply to all agents:

```python
rules = """
1. Always cite sources for factual claims
2. Maintain professional tone throughout
3. Check for accuracy before proceeding
4. Highlight uncertainties when present
"""

workflow = AgentRearrange(
    agents=[researcher, writer, editor],
    flow="Researcher -> Writer -> Editor",
    rules=rules,
)
```

### Batch Processing

Process multiple tasks efficiently:

```python
tasks = [
    "Research and write about renewable energy",
    "Research and write about electric vehicles",
    "Research and write about battery technology",
]

# batch_size: how many tasks to process at the same time (2 = process 2 tasks, then next 2, etc.)
results = workflow.batch_run(tasks, batch_size=2)
for i, result in enumerate(results):
    print(f"Task {i+1} Result:", result)
```

### Concurrent Task Processing

Process multiple tasks concurrently:

```python
tasks = [
    "Analyze Q1 financial performance",
    "Analyze Q2 financial performance",
    "Analyze Q3 financial performance",
]

# max_workers: how many tasks to run at the same time (3 = process 3 tasks simultaneously)
results = workflow.concurrent_run(tasks, max_workers=3)
```

## Example Implementations

### Content Creation Pipeline

```python
from swarms import Agent, AgentRearrange

# Create content creation agents
researcher = Agent(
    agent_name="Researcher",
    system_prompt="""You are a research specialist. Conduct thorough research on the given topic,
    gather information from multiple sources, and provide comprehensive findings with citations.""",
    model_name="gpt-4.1",
    max_loops=1,
)

outline_creator = Agent(
    agent_name="Outline-Creator",
    system_prompt="""You are an outline specialist. Create detailed outlines based on research findings.
    Structure content logically with clear sections and subsections.""",
    model_name="gpt-4.1",
    max_loops=1,
)

writer = Agent(
    agent_name="Writer",
    system_prompt="""You are a professional writer. Write engaging content based on the outline and research.
    Ensure clarity, proper grammar, and engaging style.""",
    model_name="gpt-4.1",
    max_loops=1,
)

editor = Agent(
    agent_name="Editor",
    system_prompt="""You are an expert editor. Review content for accuracy, clarity, grammar, and style.
    Ensure consistency and professional quality.""",
    model_name="gpt-4.1",
    max_loops=1,
)

# Create workflow with sequential awareness
workflow = AgentRearrange(
    name="content-pipeline",
    agents=[researcher, outline_creator, writer, editor],
    flow="Researcher -> Outline-Creator -> Writer -> Editor",
    team_awareness=True,
    output_type="dict",
    verbose=True,
)

# Execute content creation
result = workflow.run(
    "Create a comprehensive guide on implementing machine learning in production systems"
)

# Display results
for agent_name, output in result.items():
    print(f"\n{agent_name} Output:")
    print("-" * 50)
    print(output)
```

### Multi-Perspective Analysis

```python
# Create analysis agents with different perspectives
technical_analyst = Agent(
    agent_name="Technical-Analyst",
    system_prompt="Analyze from a technical perspective focusing on implementation details",
    model_name="gpt-4.1",
    max_loops=1,
)

business_analyst = Agent(
    agent_name="Business-Analyst",
    system_prompt="Analyze from a business perspective focusing on ROI and market impact",
    model_name="gpt-4.1",
    max_loops=1,
)

risk_analyst = Agent(
    agent_name="Risk-Analyst",
    system_prompt="Analyze from a risk perspective focusing on potential issues and mitigation",
    model_name="gpt-4.1",
    max_loops=1,
)

synthesizer = Agent(
    agent_name="Synthesizer",
    system_prompt="Synthesize all analysis perspectives into a comprehensive report",
    model_name="gpt-4.1",
    max_loops=1,
)

# Parallel analysis followed by synthesis
workflow = AgentRearrange(
    name="multi-perspective-analysis",
    agents=[technical_analyst, business_analyst, risk_analyst, synthesizer],
    flow="Technical-Analyst, Business-Analyst, Risk-Analyst -> Synthesizer",
    output_type="dict",
)

result = workflow.run(
    "Analyze the adoption of blockchain technology in financial services"
)
```

### Document Processing Workflow

```python
# Create document processing agents
extractor = Agent(
    agent_name="Data-Extractor",
    system_prompt="Extract key data points and information from documents",
    model_name="gpt-4.1",
    max_loops=1,
)

validator = Agent(
    agent_name="Data-Validator",
    system_prompt="Validate extracted data for accuracy and completeness",
    model_name="gpt-4.1",
    max_loops=1,
)

formatter = Agent(
    agent_name="Data-Formatter",
    system_prompt="Format validated data into structured output",
    model_name="gpt-4.1",
    max_loops=1,
)

# Sequential document processing
workflow = AgentRearrange(
    name="document-processor",
    agents=[extractor, validator, formatter],
    flow="Data-Extractor -> Data-Validator -> Data-Formatter",
    team_awareness=True,
    output_type="final",
)

result = workflow.run(
    "Extract, validate, and format financial data from the quarterly report"
)
```

## Using AgentRearrange with SwarmRouter

You can also use AgentRearrange through SwarmRouter for a unified interface:

```python
from swarms import Agent, SwarmRouter

# Create agents
researcher = Agent(agent_name="Researcher", ...)
writer = Agent(agent_name="Writer", ...)
editor = Agent(agent_name="Editor", ...)

# Use SwarmRouter with AgentRearrange
router = SwarmRouter(
    name="content-router",
    agents=[researcher, writer, editor],
    swarm_type="AgentRearrange",
    rearrange_flow="Researcher -> Writer -> Editor",  # Required for AgentRearrange
    max_loops=1,
)

result = router.run("Create content about AI ethics")
```


## Common Issues

**Flow validation errors:**
- Ensure agent names in flow match `agent_name` exactly
- Check for typos in agent names
- Verify all agents in flow are included in agents list

**Agent execution failures:**
- Check agent configurations (model names, API keys)
- Verify system prompts are clear and appropriate
- Enable verbose mode to see detailed error messages

**Memory issues:**
- Clear conversation history for large batch jobs
- Use `output_type="final"` if you don't need full history
- Consider disabling autosave for very large workflows

