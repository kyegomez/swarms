# BatchedGridWorkflow: Complete Guide

A comprehensive guide to using `BatchedGridWorkflow` for parallel multi-agent task processing.

## Overview

`BatchedGridWorkflow` enables parallel execution of multiple tasks across multiple agents, creating a grid-like execution pattern where each agent can process different tasks simultaneously.

| Feature | Description |
|---------|-------------|
| **Parallel Execution** | Tasks run simultaneously across agents |
| **Flexible Configuration** | Easy to customize names, descriptions, and loop counts |
| **Error Handling** | Built-in error handling and logging |
| **Scalable** | Works with any number of agents and tasks |

---

## Quick Start

### Basic Example

```python
from swarms import Agent
from swarms.structs.batched_grid_workflow import BatchedGridWorkflow

# Create two basic agents
agent1 = Agent(model="gpt-4")
agent2 = Agent(model="gpt-4")

# Create workflow with default settings
workflow = BatchedGridWorkflow(
    agents=[agent1, agent2]
)

# Define simple tasks
tasks = [
    "What is the capital of France?",
    "Explain photosynthesis in simple terms"
]

# Run the workflow
result = workflow.run(tasks)
```

---

## Basic Examples

### Named Workflow Example

```python
# Create agents
writer = Agent(model="gpt-4")
analyst = Agent(model="gpt-4")

# Create named workflow
workflow = BatchedGridWorkflow(
    name="Content Analysis Workflow",
    description="Analyze and write content in parallel",
    agents=[writer, analyst]
)

# Content tasks
tasks = [
    "Write a short paragraph about renewable energy",
    "Analyze the benefits of solar power"
]

# Execute workflow
result = workflow.run(tasks)
```

### Multi-Loop Example

```python
# Create agents
agent1 = Agent(model="gpt-4")
agent2 = Agent(model="gpt-4")

# Create workflow with multiple loops
workflow = BatchedGridWorkflow(
    agents=[agent1, agent2],
    max_loops=3
)

# Tasks for iterative processing
tasks = [
    "Generate ideas for a mobile app",
    "Evaluate the feasibility of each idea"
]

# Run with multiple loops
result = workflow.run(tasks)
```

### Three Agent Example

```python
# Create three agents
researcher = Agent(model="gpt-4")
writer = Agent(model="gpt-4")
editor = Agent(model="gpt-4")

# Create workflow
workflow = BatchedGridWorkflow(
    name="Research and Writing Pipeline",
    agents=[researcher, writer, editor]
)

# Three different tasks
tasks = [
    "Research the history of artificial intelligence",
    "Write a summary of the research findings",
    "Review and edit the summary for clarity"
]

# Execute workflow
result = workflow.run(tasks)
```

---

## Advanced Examples

### Custom Conversation Configuration

```python
from swarms import Agent
from swarms.structs.batched_grid_workflow import BatchedGridWorkflow

# Create agents with specific roles
researcher = Agent(
    model="gpt-4",
    system_prompt="You are a research specialist who conducts thorough investigations."
)

writer = Agent(
    model="gpt-4",
    system_prompt="You are a technical writer who creates clear, comprehensive documentation."
)

reviewer = Agent(
    model="gpt-4",
    system_prompt="You are a quality reviewer who ensures accuracy and completeness."
)

# Create workflow with custom conversation settings
workflow = BatchedGridWorkflow(
    id="custom-research-workflow",
    name="Custom Research Pipeline",
    description="Research, writing, and review pipeline with custom conversation tracking",
    agents=[researcher, writer, reviewer],
    conversation_args={
        "message_id_on": True,
        "conversation_id": "research-pipeline-001"
    },
    max_loops=2
)

# Research and documentation tasks
tasks = [
    "Research the latest developments in artificial intelligence safety",
    "Write comprehensive documentation for a new API endpoint",
    "Review and validate the technical specifications document"
]

# Execute with custom configuration
result = workflow.run(tasks)
```

### Iterative Refinement Workflow

```python
# Create refinement agents
initial_creator = Agent(
    model="gpt-4",
    system_prompt="You are a creative content creator who generates initial ideas and drafts."
)

detail_enhancer = Agent(
    model="gpt-4",
    system_prompt="You are a detail specialist who adds depth and specificity to content."
)

polish_expert = Agent(
    model="gpt-4",
    system_prompt="You are a polish expert who refines content for maximum impact and clarity."
)

# Create workflow with multiple refinement loops
workflow = BatchedGridWorkflow(
    name="Iterative Content Refinement",
    description="Multi-stage content creation with iterative improvement",
    agents=[initial_creator, detail_enhancer, polish_expert],
    max_loops=4
)

# Content creation tasks
tasks = [
    "Create an initial draft for a product launch announcement",
    "Add detailed specifications and technical details to the content",
    "Polish and refine the content for maximum engagement"
]

# Execute iterative refinement
result = workflow.run(tasks)
```

### Specialized Domain Workflow

```python
# Create domain-specific agents
medical_expert = Agent(
    model="gpt-4",
    system_prompt="You are a medical expert specializing in diagnostic procedures and treatment protocols."
)

legal_advisor = Agent(
    model="gpt-4",
    system_prompt="You are a legal advisor specializing in healthcare regulations and compliance."
)

technology_architect = Agent(
    model="gpt-4",
    system_prompt="You are a technology architect specializing in healthcare IT systems and security."
)

# Create specialized workflow
workflow = BatchedGridWorkflow(
    name="Healthcare Technology Assessment",
    description="Multi-domain assessment of healthcare technology solutions",
    agents=[medical_expert, legal_advisor, technology_architect],
    max_loops=1
)

# Domain-specific assessment tasks
tasks = [
    "Evaluate the medical efficacy and safety of a new diagnostic AI system",
    "Assess the legal compliance and regulatory requirements for the system",
    "Analyze the technical architecture and security implications of implementation"
]

# Execute specialized assessment
result = workflow.run(tasks)
```

### Parallel Analysis Workflow

```python
# Create analysis agents
market_analyst = Agent(
    model="gpt-4",
    system_prompt="You are a market analyst who evaluates business opportunities and market trends."
)

financial_analyst = Agent(
    model="gpt-4",
    system_prompt="You are a financial analyst who assesses investment potential and financial viability."
)

risk_assessor = Agent(
    model="gpt-4",
    system_prompt="You are a risk assessor who identifies potential threats and mitigation strategies."
)

# Create parallel analysis workflow
workflow = BatchedGridWorkflow(
    name="Comprehensive Business Analysis",
    description="Parallel analysis of market, financial, and risk factors",
    agents=[market_analyst, financial_analyst, risk_assessor],
    max_loops=2
)

# Analysis tasks
tasks = [
    "Analyze the market opportunity for a new fintech product",
    "Evaluate the financial projections and investment requirements",
    "Assess the regulatory and operational risks associated with the venture"
]

# Execute parallel analysis
result = workflow.run(tasks)
```

---

## Key Points

- **Simple Setup**: Minimal configuration required for basic usage
- **Parallel Execution**: Tasks run simultaneously across agents
- **Flexible Configuration**: Easy to customize names, descriptions, and loop counts
- **Error Handling**: Built-in error handling and logging
- **Scalable**: Works with any number of agents and tasks

## Use Cases

- **Content Creation**: Multiple writers working on different topics
- **Research Tasks**: Different researchers investigating various aspects
- **Analysis Work**: Multiple analysts processing different datasets
- **Educational Content**: Different instructors creating materials for various subjects
- **Quality Assurance**: Comprehensive testing across multiple domains
- **Domain Specialization**: Multi-domain assessment and evaluation

## Best Practices

1. **Agent Specialization**: Create agents with specific roles and expertise
2. **Task Alignment**: Ensure tasks match agent capabilities
3. **Loop Configuration**: Use multiple loops for iterative processes
4. **Error Monitoring**: Monitor logs for execution issues
5. **Resource Management**: Consider computational requirements for multiple agents