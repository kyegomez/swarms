# GraphWorkflow with Rustworkx: 3-Step Quickstart Guide

GraphWorkflow provides a powerful workflow orchestration system that creates directed graphs of agents for complex multi-agent collaboration. The new **Rustworkx integration** delivers 5-10x faster performance for large-scale workflows.

## Overview

| Feature | Description |
|---------|-------------|
| **Directed Graph Structure** | Nodes are agents, edges define data flow |
| **Dual Backend Support** | NetworkX (compatibility) or Rustworkx (performance) |
| **Parallel Execution** | Multiple agents run simultaneously within layers |
| **Automatic Compilation** | Optimizes workflow structure for efficient execution |
| **5-10x Performance** | Rustworkx backend for high-throughput workflows |

---

## Step 1: Install and Import

Install Swarms and Rustworkx for high-performance workflows:

```bash
pip install swarms rustworkx
```

```python
from swarms import Agent, GraphWorkflow
```

---

## Step 2: Create the Workflow with Rustworkx Backend

Create agents and build a workflow using the high-performance Rustworkx backend:

```python
# Create specialized agents
research_agent = Agent(
    agent_name="ResearchAgent",
    model_name="gpt-4o-mini",
    system_prompt="You are a research specialist. Gather and analyze information.",
    max_loops=1
)

analysis_agent = Agent(
    agent_name="AnalysisAgent",
    model_name="gpt-4o-mini",
    system_prompt="You are an analyst. Process research findings and extract insights.",
    max_loops=1
)

# Create workflow with rustworkx backend for better performance
workflow = GraphWorkflow(
    name="Research-Analysis-Pipeline",
    backend="rustworkx",  # Use rustworkx for 5-10x faster performance
    verbose=True
)

# Add agents as nodes using batch processing
workflow.add_nodes([research_agent, analysis_agent])

# Connect agents with edges
workflow.add_edge("ResearchAgent", "AnalysisAgent")
```

---

## Step 3: Execute the Workflow

Run the workflow and get results:

```python
# Execute the workflow
results = workflow.run("What are the latest trends in renewable energy technology?")

# Print results
print(results)
```

---

## Complete Example

Here's a complete parallel processing workflow:

```python
from swarms import Agent, GraphWorkflow

# Step 1: Create specialized agents
data_collector = Agent(
    agent_name="DataCollector",
    model_name="gpt-4o-mini",
    system_prompt="You collect and organize data from various sources.",
    max_loops=1
)

technical_analyst = Agent(
    agent_name="TechnicalAnalyst",
    model_name="gpt-4o-mini",
    system_prompt="You perform technical analysis on data.",
    max_loops=1
)

market_analyst = Agent(
    agent_name="MarketAnalyst",
    model_name="gpt-4o-mini",
    system_prompt="You analyze market trends and conditions.",
    max_loops=1
)

synthesis_agent = Agent(
    agent_name="SynthesisAgent",
    model_name="gpt-4o-mini",
    system_prompt="You synthesize insights from multiple analysts into a cohesive report.",
    max_loops=1
)

# Step 2: Build workflow with rustworkx backend
workflow = GraphWorkflow(
    name="Market-Analysis-Pipeline",
    backend="rustworkx",  # High-performance backend
    verbose=True
)

# Add all agents at once using batch processing (faster than individual add_node calls)
workflow.add_nodes([data_collector, technical_analyst, market_analyst, synthesis_agent])

# Create fan-out pattern: data collector feeds both analysts
workflow.add_edges_from_source(
    "DataCollector",
    ["TechnicalAnalyst", "MarketAnalyst"]
)

# Create fan-in pattern: both analysts feed synthesis agent
workflow.add_edges_to_target(
    ["TechnicalAnalyst", "MarketAnalyst"],
    "SynthesisAgent"
)

# Step 3: Execute and get results
results = workflow.run("Analyze Bitcoin market trends for Q4 2024")

print("=" * 60)
print("WORKFLOW RESULTS:")
print("=" * 60)
print(results)

# Get compilation status
status = workflow.get_compilation_status()
print(f"\nLayers: {status['cached_layers_count']}")
print(f"Max workers: {status['max_workers']}")
```

---

## NetworkX vs Rustworkx Backend

| Graph Size | Recommended Backend | Performance |
|------------|-------------------|-------------|
| < 100 nodes | NetworkX | Minimal overhead |
| 100-1000 nodes | Either | Both perform well |
| 1000+ nodes | **Rustworkx** | 5-10x faster |
| 10k+ nodes | **Rustworkx** | Essential |

```python
# NetworkX backend (default, maximum compatibility)
workflow = GraphWorkflow(backend="networkx")

# Rustworkx backend (high performance)
workflow = GraphWorkflow(backend="rustworkx")
```

---

## Edge Patterns

### Fan-Out (One-to-Many)

```python
# One agent feeds multiple agents
workflow.add_edges_from_source(
    "DataCollector",
    ["Analyst1", "Analyst2", "Analyst3"]
)
```

### Fan-In (Many-to-One)

```python
# Multiple agents feed one agent
workflow.add_edges_to_target(
    ["Analyst1", "Analyst2", "Analyst3"],
    "SynthesisAgent"
)
```

### Parallel Chain (Many-to-Many)

```python
# Full mesh connection
workflow.add_parallel_chain(
    ["Source1", "Source2"],
    ["Target1", "Target2", "Target3"]
)
```

---

## Using from_spec for Quick Setup

Create workflows quickly with the `from_spec` class method:

```python
from swarms import Agent, GraphWorkflow

# Create agents
agent1 = Agent(agent_name="Researcher", model_name="gpt-4o-mini", max_loops=1)
agent2 = Agent(agent_name="Analyzer", model_name="gpt-4o-mini", max_loops=1)
agent3 = Agent(agent_name="Reporter", model_name="gpt-4o-mini", max_loops=1)

# Create workflow from specification
workflow = GraphWorkflow.from_spec(
    agents=[agent1, agent2, agent3],
    edges=[
        ("Researcher", "Analyzer"),
        ("Analyzer", "Reporter"),
    ],
    task="Analyze climate change data",
    backend="rustworkx"  # Use high-performance backend
)

results = workflow.run()
```

---

## Visualization

Generate visual representations of your workflow:

```python
# Create visualization (requires graphviz)
output_file = workflow.visualize(
    format="png",
    view=True,
    show_summary=True
)
print(f"Visualization saved to: {output_file}")

# Simple text visualization
text_viz = workflow.visualize_simple()
print(text_viz)
```

---

## Serialization

Save and load workflows:

```python
# Save workflow with conversation history
workflow.save_to_file(
    "my_workflow.json",
    include_conversation=True,
    include_runtime_state=True
)

# Load workflow later
loaded_workflow = GraphWorkflow.load_from_file(
    "my_workflow.json",
    restore_runtime_state=True
)

# Continue execution
results = loaded_workflow.run("Follow-up analysis")
```

---

## Large-Scale Example with Rustworkx

```python
from swarms import Agent, GraphWorkflow

# Create workflow for large-scale processing
workflow = GraphWorkflow(
    name="Large-Scale-Pipeline",
    backend="rustworkx",  # Essential for large graphs
    verbose=True
)

# Create many processing agents
processors = []
for i in range(50):
    agent = Agent(
        agent_name=f"Processor{i}",
        model_name="gpt-4o-mini",
        max_loops=1
    )
    processors.append(agent)
    workflow.add_node(agent)

# Create layered connections
for i in range(0, 40, 10):
    sources = [f"Processor{j}" for j in range(i, i+10)]
    targets = [f"Processor{j}" for j in range(i+10, min(i+20, 50))]
    if targets:
        workflow.add_parallel_chain(sources, targets)

# Compile and execute
workflow.compile()
status = workflow.get_compilation_status()
print(f"Compiled: {status['cached_layers_count']} layers")

results = workflow.run("Process dataset in parallel")
```

---

## Next Steps

- Explore [GraphWorkflow Reference](../swarms/structs/graph_workflow.md) for complete API details
- See [Multi-Agentic Patterns with GraphWorkflow](./graphworkflow_rustworkx_patterns.md) for advanced patterns
- Learn about [Visualization Options](../swarms/structs/graph_workflow.md#visualization-methods) for debugging workflows

