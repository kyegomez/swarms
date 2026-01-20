# GraphWorkflow: Complete Guide

A comprehensive guide to using `GraphWorkflow` for orchestrating complex multi-agent workflows with directed graphs.

## Overview

GraphWorkflow provides a powerful workflow orchestration system that creates directed graphs of agents for complex multi-agent collaboration. The **Rustworkx integration** delivers 5-10x faster performance for large-scale workflows.

| Feature | Description |
|---------|-------------|
| **Directed Graph Structure** | Nodes are agents, edges define data flow |
| **Dual Backend Support** | NetworkX (compatibility) or Rustworkx (performance) |
| **Parallel Execution** | Multiple agents run simultaneously within layers |
| **Automatic Compilation** | Optimizes workflow structure for efficient execution |
| **5-10x Performance** | Rustworkx backend for high-throughput workflows |

---

## Quick Start

### Step 1: Install and Import

```bash
pip install swarms rustworkx
```

```python
from swarms import Agent, GraphWorkflow
```

### Step 2: Create the Workflow with Rustworkx Backend

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

### Step 3: Execute the Workflow

```python
# Execute the workflow
results = workflow.run("What are the latest trends in renewable energy technology?")

# Print results
print(results)
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

## Basic Example: Research and Analysis Pipeline

```python
from swarms import Agent, GraphWorkflow

# Create specialized research and analysis agents
research_agent = Agent(
    agent_name="ResearchAgent",
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

analysis_agent = Agent(
    agent_name="AnalysisAgent",
    agent_description="Specialist in data analysis and insight extraction",
    system_prompt="""You are an analyst with expertise in:
    - Processing research findings
    - Extracting key insights and patterns
    - Identifying trends and implications
    - Preparing actionable recommendations""",
    model_name="gpt-4.1",
    max_loops=1,
)

synthesis_agent = Agent(
    agent_name="SynthesisAgent",
    agent_description="Expert in combining insights into cohesive reports",
    system_prompt="""You are a synthesis specialist with expertise in:
    - Combining insights from multiple sources
    - Creating cohesive reports
    - Ensuring consistency and clarity
    - Highlighting key takeaways""",
    model_name="gpt-4.1",
    max_loops=1,
)

# Create workflow
workflow = GraphWorkflow(
    name="Research-Analysis-Pipeline",
    verbose=True
)

# Add agents as nodes
workflow.add_node(research_agent)
workflow.add_node(analysis_agent)
workflow.add_node(synthesis_agent)

# Connect agents sequentially
workflow.add_edge("ResearchAgent", "AnalysisAgent")
workflow.add_edge("AnalysisAgent", "SynthesisAgent")

# Execute workflow
task = "Research and analyze the latest trends in renewable energy technology"
result = workflow.run(task=task)
print(result)
```

## Parallel Analysis Example

```python
from swarms import Agent, GraphWorkflow

# Create specialized analysis agents
data_collector = Agent(
    agent_name="DataCollector",
    agent_description="Expert in collecting and organizing data from various sources",
    system_prompt="You are a data collection specialist. Collect and organize data from various sources, ensuring accuracy and completeness.",
    model_name="gpt-4.1",
    max_loops=1,
)

technical_analyst = Agent(
    agent_name="TechnicalAnalyst",
    agent_description="Specialist in technical analysis and data processing",
    system_prompt="You are a technical analyst. Perform technical analysis on data, identify patterns, and extract technical insights.",
    model_name="gpt-4.1",
    max_loops=1,
)

market_analyst = Agent(
    agent_name="MarketAnalyst",
    agent_description="Expert in market trends and competitive analysis",
    system_prompt="You are a market analyst. Analyze market trends, competitive landscape, and market conditions.",
    model_name="gpt-4.1",
    max_loops=1,
)

synthesis_agent = Agent(
    agent_name="SynthesisAgent",
    agent_description="Expert in synthesizing multiple perspectives into cohesive reports",
    system_prompt="You are a synthesis specialist. Combine insights from multiple analysts into a cohesive, comprehensive report.",
    model_name="gpt-4.1",
    max_loops=1,
)

# Create workflow
workflow = GraphWorkflow(
    name="Parallel-Analysis-Pipeline",
    verbose=True
)

# Add all agents
for agent in [data_collector, technical_analyst, market_analyst, synthesis_agent]:
    workflow.add_node(agent)

# Fan-out: one agent feeds multiple agents (parallel execution)
workflow.add_edges_from_source(
    "DataCollector",
    ["TechnicalAnalyst", "MarketAnalyst"]
)

# Fan-in: multiple agents feed one agent
workflow.add_edges_to_target(
    ["TechnicalAnalyst", "MarketAnalyst"],
    "SynthesisAgent"
)

# Execute workflow
task = "Analyze Bitcoin market trends for Q4 2024"
result = workflow.run(task=task)
print(result)
```

## Content Creation Pipeline Example

```python
from swarms import Agent, GraphWorkflow

# Create content creation agents
researcher = Agent(
    agent_name="Researcher",
    agent_description="Expert in comprehensive research and information gathering",
    system_prompt="""You are a research specialist. Conduct thorough research on the given topic,
    gather information from multiple sources, and provide comprehensive findings with citations.""",
    model_name="gpt-4.1",
    max_loops=1,
)

outline_creator = Agent(
    agent_name="OutlineCreator",
    agent_description="Specialist in creating structured content outlines",
    system_prompt="""You are an outline specialist. Create detailed outlines based on research findings.
    Structure content logically with clear sections and subsections.""",
    model_name="gpt-4.1",
    max_loops=1,
)

writer = Agent(
    agent_name="Writer",
    agent_description="Professional writer specializing in engaging content creation",
    system_prompt="""You are a professional writer. Write engaging content based on the outline and research.
    Ensure clarity, proper grammar, and engaging style.""",
    model_name="gpt-4.1",
    max_loops=1,
)

editor = Agent(
    agent_name="Editor",
    agent_description="Expert editor specializing in content review and refinement",
    system_prompt="""You are an expert editor. Review content for accuracy, clarity, grammar, and style.
    Ensure consistency and professional quality.""",
    model_name="gpt-4.1",
    max_loops=1,
)

# Create workflow
workflow = GraphWorkflow(
    name="Content-Creation-Pipeline",
    verbose=True
)

# Add agents
for agent in [researcher, outline_creator, writer, editor]:
    workflow.add_node(agent)

# Create sequential flow
workflow.add_edge("Researcher", "OutlineCreator")
workflow.add_edge("OutlineCreator", "Writer")
workflow.add_edge("Writer", "Editor")

# Execute content creation
task = "Create a comprehensive guide on implementing machine learning in production systems"
result = workflow.run(task=task)
print(result)
```

## Multi-Perspective Analysis Example

```python
from swarms import Agent, GraphWorkflow

# Create analysis agents with different perspectives
technical_analyst = Agent(
    agent_name="TechnicalAnalyst",
    agent_description="Expert in technical analysis focusing on implementation details",
    system_prompt="Analyze from a technical perspective focusing on implementation details, architecture, and technical feasibility.",
    model_name="gpt-4.1",
    max_loops=1,
)

business_analyst = Agent(
    agent_name="BusinessAnalyst",
    agent_description="Specialist in business analysis focusing on ROI and market impact",
    system_prompt="Analyze from a business perspective focusing on ROI, market impact, revenue potential, and business strategy.",
    model_name="gpt-4.1",
    max_loops=1,
)

risk_analyst = Agent(
    agent_name="RiskAnalyst",
    agent_description="Expert in risk assessment and mitigation strategies",
    system_prompt="Analyze from a risk perspective focusing on potential issues, threats, and mitigation strategies.",
    model_name="gpt-4.1",
    max_loops=1,
)

synthesizer = Agent(
    agent_name="Synthesizer",
    agent_description="Expert in synthesizing multiple perspectives into comprehensive reports",
    system_prompt="Synthesize all analysis perspectives into a comprehensive report that integrates technical, business, and risk considerations.",
    model_name="gpt-4.1",
    max_loops=1,
)

# Create workflow with rustworkx backend for better performance
workflow = GraphWorkflow(
    name="Multi-Perspective-Analysis",
    backend="rustworkx",
    verbose=True
)

# Add agents
for agent in [technical_analyst, business_analyst, risk_analyst, synthesizer]:
    workflow.add_node(agent)

# Parallel analysis followed by synthesis
workflow.add_edges_to_target(
    ["TechnicalAnalyst", "BusinessAnalyst", "RiskAnalyst"],
    "Synthesizer"
)

# Execute workflow
task = "Analyze the adoption of blockchain technology in financial services"
result = workflow.run(task=task)
print(result)
```

## Using from_spec for Quick Setup

```python
from swarms import Agent, GraphWorkflow

# Create agents
researcher = Agent(
    agent_name="Researcher",
    agent_description="Research specialist",
    system_prompt="Research the topic thoroughly and provide comprehensive findings",
    model_name="gpt-4o-mini",
    max_loops=1
)

analyzer = Agent(
    agent_name="Analyzer",
    agent_description="Data analyst",
    system_prompt="Analyze research findings and extract key insights",
    model_name="gpt-4o-mini",
    max_loops=1
)

reporter = Agent(
    agent_name="Reporter",
    agent_description="Report writer",
    system_prompt="Create final report based on analysis",
    model_name="gpt-4o-mini",
    max_loops=1
)

# Create workflow from specification
workflow = GraphWorkflow.from_spec(
    agents=[researcher, analyzer, reporter],
    edges=[
        ("Researcher", "Analyzer"),
        ("Analyzer", "Reporter"),
    ],
    task="Analyze climate change data and create a report",
    backend="rustworkx"
)

# Execute workflow
results = workflow.run()
print(results)
```

## Batch Processing

```python
from swarms import Agent, GraphWorkflow

# Create analysis agents
market_agent = Agent(
    agent_name="MarketAnalyst",
    agent_description="Expert in market analysis and trends",
    model_name="gpt-4.1",
    max_loops=1,
)

technical_agent = Agent(
    agent_name="TechnicalAnalyst",
    agent_description="Specialist in technical analysis and patterns",
    model_name="gpt-4.1",
    max_loops=1,
)

synthesis_agent = Agent(
    agent_name="SynthesisAgent",
    agent_description="Expert in synthesizing analysis results",
    model_name="gpt-4.1",
    max_loops=1,
)

# Create workflow
workflow = GraphWorkflow(
    name="Analysis-Pipeline",
    verbose=True
)

# Add agents
for agent in [market_agent, technical_agent, synthesis_agent]:
    workflow.add_node(agent)

# Create edges
workflow.add_edges_to_target(
    ["MarketAnalyst", "TechnicalAnalyst"],
    "SynthesisAgent"
)

# Execute multiple tasks
tasks = [
    "Analyze Apple (AAPL) stock performance",
    "Evaluate Microsoft (MSFT) market position",
    "Assess Google (GOOGL) competitive landscape"
]

results = workflow.batch_run(tasks=tasks)
for i, result in enumerate(results):
    print(f"Task {i+1} Result:", result)
```

## Visualizing Workflow Structure

You can visualize the structure of your workflow before executing tasks using the `visualize()` method:

```python
from swarms import Agent, GraphWorkflow

# Create specialized agents
research_agent = Agent(
    agent_name="ResearchAgent",
    agent_description="Specialized in comprehensive research and data gathering",
    model_name="gpt-4o-mini",
    max_loops=1,
)

analysis_agent = Agent(
    agent_name="AnalysisAgent",
    agent_description="Expert in data analysis and pattern recognition",
    model_name="gpt-4o-mini",
    max_loops=1,
)

synthesis_agent = Agent(
    agent_name="SynthesisAgent",
    agent_description="Specialized in synthesizing insights into reports",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Create workflow
workflow = GraphWorkflow(
    name="Research-Pipeline",
    verbose=True
)

# Add agents
for agent in [research_agent, analysis_agent, synthesis_agent]:
    workflow.add_node(agent)

# Create edges
workflow.add_edge("ResearchAgent", "AnalysisAgent")
workflow.add_edge("AnalysisAgent", "SynthesisAgent")

# Visualize the workflow structure
# Requires graphviz: pip install graphviz
output_file = workflow.visualize(
    format="png",
    view=True,
    show_summary=True
)
print(f"Visualization saved to: {output_file}")

# Or use simple text visualization
text_viz = workflow.visualize_simple()
print(text_viz)
```

## Saving and Loading Workflows

You can save workflows for reuse and load them later:

```python
from swarms import Agent, GraphWorkflow

# Create and configure workflow
workflow = GraphWorkflow(name="My-Workflow")
# ... add agents and edges ...

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

## Key Takeaways

1. **Node-Based Architecture**: Agents are nodes, edges define data flow and execution order
2. **Parallel Execution**: Use fan-out patterns (`add_edges_from_source`) for parallel processing
3. **Backend Selection**: Use `backend="rustworkx"` for large workflows (1000+ nodes) for 5-10x better performance
4. **Automatic Compilation**: Workflows are automatically compiled for optimal execution layers
5. **Visualization**: Use `visualize()` to see your workflow structure before execution
6. **Persistence**: Save and load workflows for reuse with `save_to_file()` and `load_from_file()`
7. **Quick Setup**: Use `from_spec()` for rapid workflow creation from agent and edge specifications

For more detailed information about the `GraphWorkflow` API and advanced usage patterns, see the [main documentation](../structs/graph_workflow.md).
