# AgentRearrange Examples

This page provides simple, practical examples of how to use `AgentRearrange` for various real-world scenarios.

## Basic Example: Content Creation Pipeline

```python
from swarms import Agent, AgentRearrange

# Create specialized content creation agents
researcher = Agent(
    agent_name="Researcher",
    agent_description="Expert in research, data gathering, and information synthesis",
    system_prompt="""You are a research specialist with expertise in:
    - Gathering information from reliable sources
    - Analyzing data and identifying key insights
    - Synthesizing findings into clear summaries
    - Verifying facts and checking sources
    - Organizing information logically""",
    model_name="gpt-4.1",
    max_loops=1,
)

writer = Agent(
    agent_name="Writer",
    agent_description="Professional writer specializing in engaging content creation",
    system_prompt="""You are a professional writer with expertise in:
    - Creating clear and engaging content
    - Structuring information logically
    - Ensuring proper grammar and style
    - Adapting tone for the target audience
    - Incorporating research findings effectively""",
    model_name="gpt-4.1",
    max_loops=1,
)

editor = Agent(
    agent_name="Editor",
    agent_description="Expert editor specializing in content review and refinement",
    system_prompt="""You are an expert editor with expertise in:
    - Reviewing content for clarity and accuracy
    - Correcting grammar and spelling errors
    - Improving sentence structure and flow
    - Ensuring consistency in style and tone
    - Verifying facts and checking sources""",
    model_name="gpt-4.1",
    max_loops=1,
)

# Define sequential flow: Researcher -> Writer -> Editor
flow = "Researcher -> Writer -> Editor"

# Create workflow
workflow = AgentRearrange(
    name="Content-Creation-Pipeline",
    agents=[researcher, writer, editor],
    flow=flow,
    max_loops=1,
    verbose=True,
)

# Execute workflow
task = "Research and write a comprehensive article about the impact of AI on healthcare"
result = workflow.run(task=task)
print(result)
```

## Parallel Analysis Example

```python
from swarms import Agent, AgentRearrange

# Create specialized analysis agents
market_researcher = Agent(
    agent_name="Market-Researcher",
    agent_description="Expert in market research and trend analysis",
    system_prompt="Research market trends, competitive landscape, and market opportunities",
    model_name="gpt-4.1",
    max_loops=1,
)

financial_researcher = Agent(
    agent_name="Financial-Researcher",
    agent_description="Specialist in financial data analysis",
    system_prompt="Research financial data, performance metrics, and financial trends",
    model_name="gpt-4.1",
    max_loops=1,
)

analyst = Agent(
    agent_name="Analyst",
    agent_description="Expert in synthesizing research findings",
    system_prompt="Synthesize research findings from multiple sources into actionable insights",
    model_name="gpt-4.1",
    max_loops=1,
)

# Define flow: Both researchers run in parallel, then analyst processes results
flow = "Market-Researcher, Financial-Researcher -> Analyst"

# Create workflow
workflow = AgentRearrange(
    name="Parallel-Research-Pipeline",
    agents=[market_researcher, financial_researcher, analyst],
    flow=flow,
    max_loops=1,
    verbose=True,
)

# Execute workflow
task = "Analyze Tesla (TSLA) from market and financial perspectives"
result = workflow.run(task=task)
print(result)
```

## Mixed Sequential and Parallel Flow Example

```python
from swarms import Agent, AgentRearrange

# Create agents for a complex workflow
researcher = Agent(
    agent_name="Researcher",
    agent_description="Expert in comprehensive research",
    system_prompt="Conduct comprehensive research on the topic and provide detailed findings",
    model_name="gpt-4.1",
    max_loops=1,
)

technical_writer = Agent(
    agent_name="Technical-Writer",
    agent_description="Specialist in technical content writing",
    system_prompt="Write technical content based on research findings",
    model_name="gpt-4.1",
    max_loops=1,
)

marketing_writer = Agent(
    agent_name="Marketing-Writer",
    agent_description="Expert in marketing content creation",
    system_prompt="Write marketing content based on research findings",
    model_name="gpt-4.1",
    max_loops=1,
)

editor = Agent(
    agent_name="Editor",
    agent_description="Expert editor for content review",
    system_prompt="Review and edit all written content for quality and consistency",
    model_name="gpt-4.1",
    max_loops=1,
)

# Complex flow: Sequential -> Parallel -> Sequential
flow = "Researcher -> Technical-Writer, Marketing-Writer -> Editor"

# Create workflow
workflow = AgentRearrange(
    name="Mixed-Content-Pipeline",
    agents=[researcher, technical_writer, marketing_writer, editor],
    flow=flow,
    max_loops=1,
    verbose=True,
)

# Execute workflow
task = "Create technical and marketing content about cloud computing"
result = workflow.run(task=task)
print(result)
```

## Sequential Awareness Example

```python
from swarms import Agent, AgentRearrange

# Create agents with sequential awareness
researcher = Agent(
    agent_name="Researcher",
    agent_description="Expert in research and information gathering",
    system_prompt="Research the topic thoroughly and provide comprehensive findings",
    model_name="gpt-4.1",
    max_loops=1,
)

writer = Agent(
    agent_name="Writer",
    agent_description="Professional writer",
    system_prompt="Write content based on the research provided. Consider what the editor will need.",
    model_name="gpt-4.1",
    max_loops=1,
)

editor = Agent(
    agent_name="Editor",
    agent_description="Expert editor",
    system_prompt="Edit the written content for clarity, accuracy, and style",
    model_name="gpt-4.1",
    max_loops=1,
)

# Create workflow with team awareness enabled
workflow = AgentRearrange(
    name="Aware-Content-Pipeline",
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
task = "Research and write about quantum computing"
result = workflow.run(task=task)
print(result)
```

## Document Processing Workflow Example

```python
from swarms import Agent, AgentRearrange

# Create document processing agents
extractor = Agent(
    agent_name="Data-Extractor",
    agent_description="Expert in extracting data from documents",
    system_prompt="Extract key data points and information from documents accurately",
    model_name="gpt-4.1",
    max_loops=1,
)

validator = Agent(
    agent_name="Data-Validator",
    agent_description="Specialist in data validation",
    system_prompt="Validate extracted data for accuracy, completeness, and consistency",
    model_name="gpt-4.1",
    max_loops=1,
)

formatter = Agent(
    agent_name="Data-Formatter",
    agent_description="Expert in data formatting",
    system_prompt="Format validated data into structured output suitable for downstream processing",
    model_name="gpt-4.1",
    max_loops=1,
)

# Sequential document processing
workflow = AgentRearrange(
    name="Document-Processor",
    agents=[extractor, validator, formatter],
    flow="Data-Extractor -> Data-Validator -> Data-Formatter",
    team_awareness=True,
    output_type="final",
    verbose=True,
)

# Execute workflow
task = "Extract, validate, and format financial data from the quarterly report"
result = workflow.run(task=task)
print(result)
```

## Batch Processing

```python
from swarms import Agent, AgentRearrange

# Create analysis agents
market_agent = Agent(
    agent_name="Market-Analyst",
    agent_description="Expert in market analysis and trends",
    model_name="gpt-4.1",
    max_loops=1,
)

technical_agent = Agent(
    agent_name="Technical-Analyst",
    agent_description="Specialist in technical analysis and patterns",
    model_name="gpt-4.1",
    max_loops=1,
)

synthesis_agent = Agent(
    agent_name="Synthesis-Agent",
    agent_description="Expert in synthesizing analysis results",
    model_name="gpt-4.1",
    max_loops=1,
)

# Create workflow
workflow = AgentRearrange(
    name="Analysis-Pipeline",
    agents=[market_agent, technical_agent, synthesis_agent],
    flow="Market-Analyst, Technical-Analyst -> Synthesis-Agent",
    max_loops=1,
    verbose=True,
)

# Execute multiple tasks
tasks = [
    "Analyze Apple (AAPL) stock performance",
    "Evaluate Microsoft (MSFT) market position",
    "Assess Google (GOOGL) competitive landscape"
]

# batch_size: how many tasks to process at the same time (2 = process 2 tasks, then next 2, etc.)
results = workflow.batch_run(tasks, batch_size=2)
for i, result in enumerate(results):
    print(f"Task {i+1} Result:", result)
```

## Human-in-the-Loop Example

Add human review points in your workflow for quality control:

```python
from swarms import Agent, AgentRearrange

# Create agents
researcher = Agent(
    agent_name="Researcher",
    agent_description="Expert in research and information gathering",
    system_prompt="Research the topic thoroughly and provide comprehensive findings",
    model_name="gpt-4.1",
    max_loops=1,
)

writer = Agent(
    agent_name="Writer",
    agent_description="Professional writer",
    system_prompt="Write content based on the research provided",
    model_name="gpt-4.1",
    max_loops=1,
)

editor = Agent(
    agent_name="Editor",
    agent_description="Expert editor",
    system_prompt="Edit the written content for clarity, accuracy, and style",
    model_name="gpt-4.1",
    max_loops=1,
)

# Custom human reviewer function
def human_reviewer(input_text: str) -> str:
    """Custom human-in-the-loop function"""
    print(f"\n{'='*60}")
    print("HUMAN REVIEW REQUIRED")
    print(f"{'='*60}")
    print(f"\nContent to review:\n{input_text}\n")
    response = input("Enter your feedback or 'approve' to continue: ")
    return response if response else "approved"

# Create workflow with human review point
workflow = AgentRearrange(
    name="Human-Reviewed-Pipeline",
    agents=[researcher, writer, editor],
    flow="Researcher -> Writer -> H -> Editor",  # H = Human review point
    human_in_the_loop=True,
    custom_human_in_the_loop=human_reviewer,
    verbose=True,
)

# Execute workflow
task = "Research and write about quantum computing"
result = workflow.run(task=task)
print(result)
```

## Memory System Integration Example

Integrate with memory systems for persistent storage and context:

```python
from swarms import Agent, AgentRearrange
from swarms.memory import QdrantVectorDatabase

# Initialize memory system
memory = QdrantVectorDatabase(
    collection_name="agent_workflow_memory",
    path="./memory"
)

# Create agents
researcher = Agent(
    agent_name="Researcher",
    agent_description="Expert in research",
    system_prompt="Research the topic and store findings in memory",
    model_name="gpt-4.1",
    max_loops=1,
)

writer = Agent(
    agent_name="Writer",
    agent_description="Professional writer",
    system_prompt="Write content based on research from memory",
    model_name="gpt-4.1",
    max_loops=1,
)

# Create workflow with memory system
workflow = AgentRearrange(
    name="Memory-Enabled-Pipeline",
    agents=[researcher, writer],
    flow="Researcher -> Writer",
    memory_system=memory,
    autosave=True,
    verbose=True,
)

# Execute workflow
task = "Research and write about renewable energy trends"
result = workflow.run(task=task)
print(result)
```

## Custom Rules Example

Add system-wide rules that apply to all agents:

```python
from swarms import Agent, AgentRearrange

# Create agents
researcher = Agent(
    agent_name="Researcher",
    agent_description="Expert in research",
    system_prompt="Research the topic thoroughly",
    model_name="gpt-4.1",
    max_loops=1,
)

writer = Agent(
    agent_name="Writer",
    agent_description="Professional writer",
    system_prompt="Write content based on research",
    model_name="gpt-4.1",
    max_loops=1,
)

editor = Agent(
    agent_name="Editor",
    agent_description="Expert editor",
    system_prompt="Edit content for quality",
    model_name="gpt-4.1",
    max_loops=1,
)

# Define system-wide rules
rules = """
1. Always cite sources for factual claims
2. Maintain professional tone throughout
3. Check for accuracy before proceeding
4. Highlight uncertainties when present
5. Ensure all content is fact-checked
"""

# Create workflow with custom rules
workflow = AgentRearrange(
    name="Rules-Based-Pipeline",
    agents=[researcher, writer, editor],
    flow="Researcher -> Writer -> Editor",
    rules=rules,
    verbose=True,
)

# Execute workflow
task = "Research and write about climate change impacts"
result = workflow.run(task=task)
print(result)
```

## Concurrent Task Processing

Process multiple tasks concurrently for better performance:

```python
from swarms import Agent, AgentRearrange

# Create analysis agents
market_agent = Agent(
    agent_name="Market-Analyst",
    agent_description="Expert in market analysis and trends",
    model_name="gpt-4.1",
    max_loops=1,
)

technical_agent = Agent(
    agent_name="Technical-Analyst",
    agent_description="Specialist in technical analysis and patterns",
    model_name="gpt-4.1",
    max_loops=1,
)

synthesis_agent = Agent(
    agent_name="Synthesis-Agent",
    agent_description="Expert in synthesizing analysis results",
    model_name="gpt-4.1",
    max_loops=1,
)

# Create workflow
workflow = AgentRearrange(
    name="Analysis-Pipeline",
    agents=[market_agent, technical_agent, synthesis_agent],
    flow="Market-Analyst, Technical-Analyst -> Synthesis-Agent",
    max_loops=1,
    verbose=True,
)

# Execute multiple tasks concurrently
tasks = [
    "Analyze Apple (AAPL) stock performance",
    "Evaluate Microsoft (MSFT) market position",
    "Assess Google (GOOGL) competitive landscape"
]

# max_workers: how many tasks to run at the same time (3 = process 3 tasks simultaneously)
results = workflow.concurrent_run(tasks, max_workers=3)
for i, result in enumerate(results):
    print(f"Task {i+1} Result:", result)
```

## Output Type Variations

Use different output types based on your needs:

```python
from swarms import Agent, AgentRearrange

# Create agents
researcher = Agent(
    agent_name="Researcher",
    agent_description="Expert in research",
    system_prompt="Research the topic thoroughly",
    model_name="gpt-4.1",
    max_loops=1,
)

writer = Agent(
    agent_name="Writer",
    agent_description="Professional writer",
    system_prompt="Write content based on research",
    model_name="gpt-4.1",
    max_loops=1,
)

# Example 1: Get dictionary output with all agent results
workflow_dict = AgentRearrange(
    name="Dict-Output-Pipeline",
    agents=[researcher, writer],
    flow="Researcher -> Writer",
    output_type="dict",  # Returns dictionary with agent names as keys
    verbose=True,
)
result_dict = workflow_dict.run("Research and write about AI")
print(result_dict)  # {"Researcher": "...", "Writer": "..."}

# Example 2: Get only final agent's output
workflow_final = AgentRearrange(
    name="Final-Output-Pipeline",
    agents=[researcher, writer],
    flow="Researcher -> Writer",
    output_type="final",  # Returns only the last agent's output
    verbose=True,
)
result_final = workflow_final.run("Research and write about AI")
print(result_final)  # Only Writer's output

# Example 3: Get list of all outputs
workflow_list = AgentRearrange(
    name="List-Output-Pipeline",
    agents=[researcher, writer],
    flow="Researcher -> Writer",
    output_type="list",  # Returns list of outputs in order
    verbose=True,
)
result_list = workflow_list.run("Research and write about AI")
print(result_list)  # [researcher_output, writer_output]
```

## Using AgentRearrange with SwarmRouter

You can also use AgentRearrange through SwarmRouter for a unified interface:

```python
from swarms import Agent, SwarmRouter

# Create agents
researcher = Agent(
    agent_name="Researcher",
    agent_description="Research specialist",
    system_prompt="Research the topic thoroughly",
    model_name="gpt-4o-mini",
    max_loops=1,
)

writer = Agent(
    agent_name="Writer",
    agent_description="Professional writer",
    system_prompt="Write content based on research",
    model_name="gpt-4o-mini",
    max_loops=1,
)

editor = Agent(
    agent_name="Editor",
    agent_description="Expert editor",
    system_prompt="Edit content for quality",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Use SwarmRouter with AgentRearrange
router = SwarmRouter(
    name="Content-Router",
    agents=[researcher, writer, editor],
    swarm_type="AgentRearrange",
    rearrange_flow="Researcher -> Writer -> Editor",  # Required for AgentRearrange
    max_loops=1,
)

# Execute workflow
task = "Create content about AI ethics"
result = router.run(task=task)
print(result)
```

## Key Takeaways

1. **Flow Pattern Syntax**: Use `->` for sequential execution and `,` for parallel execution
2. **Sequential Awareness**: Enable `team_awareness=True` to give agents context about their position in the workflow
3. **Parallel Execution**: Use comma-separated agents for parallel processing when tasks are independent
4. **Mixed Patterns**: Combine sequential and parallel execution for complex workflows
5. **Output Types**: Choose appropriate output types (`"all"`, `"final"`, `"list"`, `"dict"`) based on your needs
6. **Batch Processing**: Use `batch_run()` for processing multiple tasks efficiently
7. **SwarmRouter Integration**: Use SwarmRouter with `swarm_type="AgentRearrange"` for a unified interface

For more detailed information about the `AgentRearrange` API and advanced usage patterns, see the [main documentation](../structs/agent_rearrange.md).
