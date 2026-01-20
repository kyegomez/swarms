# AutoSwarmBuilder: Complete Guide

A comprehensive guide to using AutoSwarmBuilder for automatic multi-agent team generation and orchestration.

## Overview

**AutoSwarmBuilder** is an intelligent system that automatically designs, creates, and orchestrates multi-agent teams based on natural language task descriptions. It uses a sophisticated "boss" agent that analyzes your requirements and generates specialized agents with comprehensive system prompts, distinct personalities, and appropriate swarm architectures.

| Feature | Description |
|---------|-------------|
| **Automatic Agent Generation** | Creates agents with roles, personalities, and expertise based on task |
| **Intelligent Architecture Selection** | Chooses optimal swarm type (Sequential, Concurrent, Hierarchical, etc.) |
| **Comprehensive System Prompts** | Generates detailed prompts with decision-making frameworks |
| **Flexible Execution** | Returns agents, swarm router config, or agent objects |
| **Scalable Teams** | Generates from 2 to dozens of specialized agents |

```
Your Task Description
        │
        ▼
   AutoSwarmBuilder
   (Boss System Prompt)
        │
        ▼
┌───────────────────────┐
│ Auto-Generated Team   │
│  - Agent Roles        │
│  - Personalities      │
│  - System Prompts     │
│  - Architecture Type  │
└───────────────────────┘
        │
        ▼
    Ready to Run
```

### When to Use AutoSwarmBuilder

**Best For:**
- Rapid prototyping of multi-agent systems
- Exploring different team compositions
- Tasks where optimal agent design isn't obvious
- Creating specialized teams for one-time use
- Learning about multi-agent architectures

**Not Ideal For:**
- Production systems requiring precise agent tuning
- Tasks with well-established agent patterns
- When you need complete control over every detail

---

## Installation

```bash
pip install swarms
```

---

## Quick Start

### Step 1: Import and Create

```python
from swarms.structs.auto_swarm_builder import AutoSwarmBuilder

# Initialize the builder
swarm_builder = AutoSwarmBuilder(
    name="Marketing-Team-Builder",
    description="Builds marketing teams automatically",
    model_name="gpt-4o",  # Boss agent model
    max_loops=1,
    execution_type="return-agents",  # or "return-swarm-router-config", "return-agents-objects"
    verbose=True
)
```

### Step 2: Generate and Run

```python
# Describe what you need
task = "Create a marketing team with 4 agents: market researcher, content strategist, copywriter, and social media specialist. They should collaborate on launching a new AI product."

# Auto-generate the team
result = swarm_builder.run(task=task)

# The builder creates:
# - 4 agents with specialized roles
# - Comprehensive system prompts for each
# - Appropriate swarm architecture
# - Ready-to-use configuration

print(result)
```

---

## Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | Required | Name of the builder instance |
| `description` | `str` | Required | Purpose/description of the builder |
| `model_name` | `str` | `"gpt-4o"` | Model for boss agent |
| `max_loops` | `int` | `1` | Loops for generation process |
| `execution_type` | `str` | `"return-agents"` | Output format type |
| `verbose` | `bool` | `False` | Enable detailed logging |

### Execution Types

| Type | Returns | Use Case |
|------|---------|----------|
| `"return-agents"` | List of agent dictionaries | Inspect and customize agents |
| `"return-swarm-router-config"` | Complete SwarmRouter configuration | Ready-to-use swarm |
| `"return-agents-objects"` | List of Agent objects | Direct execution |

---

## Complete Examples

### Example 1: Content Creation Team

```python
from swarms.structs.auto_swarm_builder import AutoSwarmBuilder
import json

# Create builder
swarm = AutoSwarmBuilder(
    name="Content-Creation-Team",
    description="Builds content creation teams",
    model_name="gpt-4o",
    max_loops=1,
    execution_type="return-agents",
    verbose=True
)

# Define task
task = """
Create a content creation team with 4 agents:
- Researcher: Gathers information and data
- Writer: Creates compelling narratives
- Editor: Refines and polishes content
- SEO Specialist: Optimizes for search engines

They should collaborate on creating high-quality blog posts.
"""

# Generate team
team_config = swarm.run(task=task)

# View generated team
print(json.dumps(team_config, indent=2))
```

### Example 2: Financial Analysis Team

```python
swarm = AutoSwarmBuilder(
    name="Financial-Analysis-Builder",
    description="Creates specialized financial analysis teams",
    model_name="gpt-4o",
    execution_type="return-agents",
    verbose=True
)

task = """
Create a comprehensive financial analysis team with 6 highly specialized agents:

1. **Equity Analyst**: Expert in stock market analysis, company valuations, and sector trends
2. **Fixed Income Analyst**: Specialist in bonds, credit ratings, and debt instruments
3. **Quantitative Analyst**: Builds mathematical models and trading algorithms
4. **Risk Manager**: Assesses portfolio risk, VaR calculations, and hedging strategies
5. **Macro Economist**: Analyzes macroeconomic trends, monetary policy, and global markets
6. **Portfolio Manager**: Oversees overall strategy, asset allocation, and rebalancing

The team should work together to analyze investment opportunities and manage a diversified portfolio.
Make each agent's system prompt extremely detailed with specific methodologies and frameworks.
"""

team = swarm.run(task=task)

# Each agent will have:
# - Specific role and responsibilities
# - Distinct personality and approach
# - Comprehensive system prompt (often 500+ words)
# - Clear capabilities and limitations
# - Collaboration guidelines

for agent in team:
    print(f"\nAgent: {agent['agent_name']}")
    print(f"Description: {agent['description']}")
    print(f"Prompt Length: {len(agent['system_prompt'])} characters")
```

### Example 3: Software Development Team

```python
swarm = AutoSwarmBuilder(
    name="Dev-Team-Builder",
    model_name="gpt-4o",
    execution_type="return-swarm-router-config",  # Get complete swarm
)

task = """
Create an enterprise software development team:

- **Tech Lead**: Architecture decisions, code reviews, technical strategy
- **Backend Developer**: API design, database optimization, server-side logic
- **Frontend Developer**: UI components, state management, responsive design
- **DevOps Engineer**: CI/CD, infrastructure, monitoring, deployments
- **QA Engineer**: Testing strategy, automation, quality assurance
- **Security Engineer**: Security audits, vulnerability assessment, compliance

Team should use Agile methodology and collaborate on building a SaaS platform.
Include detailed technical specifications and best practices in each agent's prompt.
"""

swarm_config = swarm.run(task=task)

# swarm_config is ready to use with SwarmRouter
# It includes the complete architecture and agent specifications
```

### Example 4: Research Team

```python
swarm = AutoSwarmBuilder(
    name="Research-Team-Builder",
    model_name="gpt-4o",
    execution_type="return-agents-objects",  # Get Agent objects
)

task = """
Create a scientific research team for a clinical study:

- **Principal Investigator**: Leads research, designs studies, interprets results
- **Biostatistician**: Statistical analysis, study design, data interpretation
- **Data Scientist**: Machine learning, data mining, predictive modeling
- **Literature Reviewer**: Systematic reviews, meta-analysis, evidence synthesis
- **Research Coordinator**: Project management, participant recruitment, data collection

Team should collaborate on a longitudinal study analyzing treatment efficacy.
"""

agents = swarm.run(task=task)

# agents is a list of Agent objects
# Ready to use immediately
for agent in agents:
    result = agent.run("Analyze the study protocol")
    print(f"{agent.agent_name}: {result[:100]}...")
```

---

## Use Cases

| Scenario | Team Description |
|----------|------------------|
| **Content Creation** | "Writers, editors, SEO specialists for blog content" |
| **Software Development** | "Full-stack developers, QA engineers, DevOps for microservices" |
| **Financial Analysis** | "Financial analysts, risk managers, compliance officers for investment portfolio" |
| **Customer Support** | "Support agents, escalation specialists, quality reviewers for customer service" |
| **Research** | "Researchers, data scientists, literature reviewers for scientific study" |

### Use Case: Accounting Team

```python
task = """
Create an accounting team to analyze cryptocurrency transactions with 5 agents:

1. **Tax Accountant**: Cryptocurrency tax implications, capital gains, reporting requirements
2. **Forensic Accountant**: Transaction tracing, fraud detection, blockchain analysis
3. **Compliance Officer**: Regulatory compliance, AML/KYC requirements, reporting standards
4. **Financial Auditor**: Accuracy verification, reconciliation, audit trails
5. **Risk Assessor**: Risk analysis, exposure assessment, mitigation strategies

Each agent needs extremely extensive and comprehensive system prompts with specific frameworks,
regulations, and methodologies for handling crypto transactions.
"""

swarm = AutoSwarmBuilder(
    name="Crypto-Accounting-Team",
    model_name="gpt-4o",
    execution_type="return-agents"
)

team = swarm.run(task=task)
```

---

## How It Works

1. **Task Analysis**: Boss agent analyzes your requirements
2. **Agent Design**: Creates agents with:
   - Unique roles and purposes
   - Distinct personalities
   - Comprehensive system prompts
   - Specific capabilities and limitations
3. **Architecture Selection**: Chooses optimal swarm type
4. **Configuration Generation**: Outputs ready-to-use configuration
5. **Return**: Provides agents in requested format

---

## Boss System Prompt

The boss agent uses a sophisticated system prompt that includes:

### Core Design Principles
- Comprehensive task analysis and decomposition
- Agent design excellence with distinct personalities
- Multi-agent coordination architecture
- Quality assurance and governance

### Agent Design Framework
For each agent, the boss defines:
- Role & Purpose
- Personality Profile
- Expertise Matrix
- Communication Protocol
- Decision-Making Framework
- Limitations & Boundaries
- Collaboration Strategy

### Supported Swarm Architectures
The boss can select from 14+ swarm architectures:
- AgentRearrange
- MixtureOfAgents
- SpreadSheetSwarm
- SequentialWorkflow
- ConcurrentWorkflow
- GroupChat
- MultiAgentRouter
- HierarchicalSwarm
- MajorityVoting
- And more...

---

## Output Structure

### Format: return-agents

```json
[
  {
    "agent_name": "Market-Researcher",
    "description": "Expert market researcher specializing in...",
    "system_prompt": "You are a market research specialist... [comprehensive prompt]"
  },
  {
    "agent_name": "Data-Analyst",
    "description": "Data analysis expert focusing on...",
    "system_prompt": "You are a data analyst... [comprehensive prompt]"
  }
]
```

### Format: return-swarm-router-config

Includes complete SwarmRouter configuration with:
- All agent specifications
- Swarm type selection
- Architecture parameters
- Ready to instantiate and run

---

## Best Practices

1. **Be Extremely Specific**: Provide detailed role descriptions
2. **Request Comprehensive Prompts**: Ask for "extremely detailed" and "comprehensive" prompts
3. **Define Team Size**: Specify exact number of agents needed
4. **Describe Collaboration**: Explain how agents should work together
5. **Use Powerful Models**: gpt-4o or claude-sonnet-4 for best results
6. **Review and Customize**: Always review generated agents before production use
7. **Iterate**: Run multiple times with refined descriptions if needed

---

## Common Patterns

### Pattern 1: Specialized Expertise Teams

```python
# Create domain expert teams
task = "Create 5 medical specialists: cardiologist, neurologist, oncologist, radiologist, pathologist"
```

### Pattern 2: Workflow Teams

```python
# Create process-oriented teams
task = "Create workflow team: intake specialist, processor, quality checker, approver, notifier"
```

### Pattern 3: Cross-Functional Teams

```python
# Create diverse skill teams
task = "Create cross-functional team: technical, business, creative, operational, strategic perspectives"
```

---

## Related Architectures

| Architecture | Relationship |
|--------------|--------------|
| **[SwarmRouter](./swarm_router.md)** | Can use AutoSwarmBuilder output |
| **[HierarchicalSwarm](./hierarchical_swarm_example.md)** | One possible architecture AutoSwarmBuilder might choose |
| **[SequentialWorkflow](./sequential_example.md)** | Another architecture option |

---

## Next Steps

- See [GitHub Examples](https://github.com/kyegomez/swarms/tree/master/examples/multi_agent/asb)
- Learn about [Agent Design Principles](../concept/agent_design.md)
- Try [SwarmRouter](./swarm_router.md) for task routing
