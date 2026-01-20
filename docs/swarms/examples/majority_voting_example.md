# MajorityVoting: Complete Guide

A comprehensive guide to using the MajorityVoting architecture for robust multi-agent consensus building and decision making.

## Overview

The **MajorityVoting** system is a multi-agent architecture that enables sophisticated consensus building through concurrent agent execution and intelligent synthesis. Multiple agents independently analyze a task from their unique perspectives, and a specialized consensus agent evaluates, ranks, and synthesizes their outputs into a comprehensive final answer.

| Feature | Description |
|---------|-------------|
| **Concurrent Processing** | All agents execute simultaneously using ThreadPoolExecutor for maximum efficiency |
| **Intelligent Consensus** | Dedicated consensus agent evaluates responses across multiple dimensions |
| **Multi-Loop Refinement** | Supports iterative consensus building where agents refine responses based on previous rounds |
| **Memory Retention** | Maintains full conversation history across all loops for context-aware refinement |
| **Flexible Output** | Supports multiple output formats (dict, str, list) for different use cases |
| **Comprehensive Evaluation** | Assesses accuracy, depth, relevance, clarity, unique perspectives, and innovation |

```
Agent A ─┐
Agent B ─┼──> Concurrent Execution
Agent C ─┘
    │
    ▼
 Consensus Agent
    │
    ▼
Synthesized Final Answer
```

### When to Use MajorityVoting

**Best For:**
- Complex decision-making requiring multiple perspectives
- Situations where diverse expert opinions improve outcomes
- Analysis tasks benefiting from different specialized viewpoints
- Research synthesis where comprehensive coverage is critical
- Strategic planning requiring balanced consideration of alternatives

**Not Ideal For:**
- Simple queries with straightforward answers
- Time-critical tasks requiring minimal processing
- Tasks requiring sequential agent collaboration
- Single-perspective analysis

---

## Installation

```bash
pip install swarms
```

---

## Quick Start

### Step 1: Create Agents

```python
from swarms import Agent, MajorityVoting

# Create specialized agents
agent1 = Agent(
    agent_name="Financial-Analysis-Agent-1",
    system_prompt="You are a conservative financial advisor focused on risk management and long-term stability.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

agent2 = Agent(
    agent_name="Financial-Analysis-Agent-2",
    system_prompt="You are a growth-oriented financial advisor focused on high-potential opportunities.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

agent3 = Agent(
    agent_name="Financial-Analysis-Agent-3",
    system_prompt="You are a balanced financial advisor focused on diversification and risk-adjusted returns.",
    model_name="gpt-4o-mini",
    max_loops=1,
)
```

### Step 2: Create the Voting System

```python
# Create the majority voting system
voting_system = MajorityVoting(
    agents=[agent1, agent2, agent3],
    max_loops=1,
    verbose=True
)
```

### Step 3: Run the Voting System

```python
# Define the task
task = "Create a table of super high growth opportunities for AI. I have $40k to invest in ETFs, index funds, and more. Please create a table in markdown."

# Run the voting system
result = voting_system.run(task=task)

# Print the result
print(result)
```

---

## Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agents` | `List[Agent]` | Required | List of Agent instances participating in the voting system |
| `max_loops` | `int` | `1` | Number of consensus rounds for iterative refinement |
| `output_type` | `str` | `"dict"` | Output format: "dict", "str", or "list" |
| `verbose` | `bool` | `False` | Enable detailed logging and progress tracking |
| `consensus_agent_prompt` | `str` | Default prompt | Custom system prompt for the consensus agent |
| `consensus_agent_name` | `str` | `"Consensus-Agent"` | Name of the consensus agent |
| `consensus_agent_model_name` | `str` | `"gpt-4.1"` | Model name for the consensus agent |
| `additional_consensus_agent_kwargs` | `dict` | `{}` | Additional keyword arguments for consensus agent initialization |
| `autosave` | `bool` | `False` | Automatically save conversation history |

### Output Types

| Value | Description |
|-------|-------------|
| `"dict"` | Conversation history as dictionary with roles and content |
| `"str"` | All messages formatted as a single string |
| `"list"` | Messages as a list of dictionaries |

---

## Complete Examples

### Example 1: Basic Financial Analysis

```python
from swarms import Agent, MajorityVoting

# Create agents with different perspectives
agents = [
    Agent(
        agent_name="Conservative-Analyst",
        system_prompt="You are a conservative financial advisor focused on risk management and long-term stability.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="Growth-Analyst",
        system_prompt="You are a growth-oriented financial advisor focused on high-potential opportunities.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="Balanced-Analyst",
        system_prompt="You are a balanced financial advisor focused on diversification and risk-adjusted returns.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
]

# Create the majority voting system
voting_system = MajorityVoting(
    agents=agents,
    max_loops=1,  # Single round of voting
    output_type="dict",  # Return as dictionary
    verbose=True
)

# Run a task
task = "What are the top 3 AI investment opportunities for 2024?"
result = voting_system.run(task=task)

print("=" * 60)
print("VOTING RESULT:")
print("=" * 60)
print(result)
```

### Example 2: Multi-Loop Consensus

Multi-loop voting enables iterative refinement where agents see the consensus from previous rounds and refine their responses:

```python
from swarms import Agent, MajorityVoting

# Create specialized research agents
researchers = [
    Agent(
        agent_name="Medical-Researcher",
        system_prompt=(
            "You are a medical research specialist focused on clinical evidence, "
            "safety profiles, and therapeutic efficacy. Always cite peer-reviewed sources."
        ),
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="Data-Scientist",
        system_prompt=(
            "You are a data scientist specialized in analyzing clinical trial data, "
            "statistical significance, and population health outcomes."
        ),
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="Ethics-Specialist",
        system_prompt=(
            "You are a bioethics specialist focused on patient autonomy, "
            "informed consent, equity in healthcare access, and ethical implications."
        ),
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
]

# Multi-loop voting system
voting_system = MajorityVoting(
    agents=researchers,
    max_loops=3,  # 3 rounds of refinement
    output_type="dict",
    verbose=True
)

# Complex research question
result = voting_system.run(
    task=(
        "Evaluate the efficacy and safety of mRNA vaccines for COVID-19. "
        "Include analysis of clinical trial data, real-world effectiveness, "
        "potential side effects, and ethical considerations for global distribution."
    )
)

# The system will run 3 loops:
# Loop 1: Initial independent analysis
# Loop 2: Agents refine based on consensus from Loop 1
# Loop 3: Final refinement based on consensus from Loop 2
```

### Example 3: Custom Consensus Agent

Customize the consensus agent for domain-specific evaluation:

```python
from swarms import Agent, MajorityVoting

# Custom consensus prompt for technical evaluation
TECHNICAL_CONSENSUS_PROMPT = """
You are a Senior Technical Architect responsible for evaluating and synthesizing technical recommendations from your engineering team.

**Evaluation Framework:**

1. **Technical Accuracy**: Assess correctness of technical claims and architectural patterns
2. **Scalability**: Evaluate solutions for performance at scale
3. **Maintainability**: Consider long-term code quality and technical debt
4. **Security**: Identify potential security vulnerabilities
5. **Cost**: Analyze infrastructure and operational costs
6. **Innovation**: Recognize novel approaches and creative solutions

**Synthesis Process:**
- Compare technical approaches objectively
- Identify trade-offs between solutions
- Provide clear recommendations with justification
- Highlight areas of agreement and disagreement
- Synthesize the best elements into a comprehensive solution

**Output Format:**
For each agent: [Agent Name]: [Technical Evaluation]
Comparative Analysis: [Architecture Comparison]
Recommended Approach: [Synthesized Solution]
Implementation Roadmap: [Next Steps]
"""

# Create technical agents
tech_agents = [
    Agent(
        agent_name="Backend-Architect",
        system_prompt="You are a backend architecture specialist focused on API design, database optimization, and server infrastructure.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="Frontend-Architect",
        system_prompt="You are a frontend architecture specialist focused on user experience, performance optimization, and accessibility.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="DevOps-Engineer",
        system_prompt="You are a DevOps specialist focused on CI/CD, infrastructure as code, monitoring, and deployment strategies.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
]

# Voting system with custom consensus
voting_system = MajorityVoting(
    agents=tech_agents,
    max_loops=2,
    consensus_agent_prompt=TECHNICAL_CONSENSUS_PROMPT,
    consensus_agent_name="Tech-Lead",
    consensus_agent_model_name="gpt-4o",  # Use more powerful model for consensus
    verbose=True
)

result = voting_system.run(
    task="Design a scalable architecture for a real-time collaborative document editing application with 1M+ concurrent users."
)
```

---

## Use Cases

### Use Case 1: Investment Strategy

```python
from swarms import Agent, MajorityVoting

# Different investment philosophies
agents = [
    Agent(
        agent_name="Value-Investor",
        system_prompt="You follow value investing principles: focus on undervalued stocks, fundamental analysis, and margin of safety.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="Momentum-Trader",
        system_prompt="You follow momentum investing: identify trending stocks, technical analysis, and market psychology.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="Dividend-Investor",
        system_prompt="You focus on dividend investing: steady income, dividend growth, and sustainable payout ratios.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
]

voting = MajorityVoting(agents=agents, max_loops=2)
result = voting.run("Create a diversified portfolio strategy for $200k with 10-year horizon")
```

### Use Case 2: Product Development

```python
from swarms import Agent, MajorityVoting

# Different product perspectives
agents = [
    Agent(
        agent_name="UX-Designer",
        system_prompt="You prioritize user experience, usability, and design aesthetics in product decisions.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="Product-Manager",
        system_prompt="You balance business goals, market fit, and customer needs in product decisions.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="Tech-Lead",
        system_prompt="You evaluate technical feasibility, scalability, and implementation complexity.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
]

voting = MajorityVoting(agents=agents, max_loops=1)
result = voting.run("Should we build a native mobile app or responsive web app for our new product?")
```

### Use Case 3: Batch Processing

```python
from swarms import Agent, MajorityVoting

voting_system = MajorityVoting(
    agents=[agent1, agent2, agent3],
    max_loops=1
)

# Multiple tasks
tasks = [
    "What are the best cloud computing stocks?",
    "Should I invest in renewable energy ETFs?",
    "What's the outlook for semiconductor companies?",
]

# Process all tasks
results = voting_system.batch_run(tasks)

for task, result in zip(tasks, results):
    print(f"\nTask: {task}")
    print(f"Result: {result[:200]}...")
```

---

## How It Works

1. **Concurrent Execution**: All agents run simultaneously on the same task
2. **Independent Analysis**: Each agent provides its own perspective based on its system prompt
3. **Consensus Evaluation**: The consensus agent:
   - Evaluates each agent's response on multiple dimensions
   - Compares and contrasts different viewpoints
   - Identifies the strongest arguments
   - Synthesizes a comprehensive final answer
4. **Iterative Refinement** (if max_loops > 1): Agents see the consensus and refine their responses

---

## Best Practices

### 1. Agent Diversity

Create agents with truly different perspectives, not just different names:

```python
# Good - Distinct perspectives
agent1 = Agent(system_prompt="Conservative risk-averse approach...")
agent2 = Agent(system_prompt="Aggressive growth-focused approach...")
agent3 = Agent(system_prompt="Balanced diversified approach...")

# Bad - Too similar
agent1 = Agent(system_prompt="Financial advisor...")
agent2 = Agent(system_prompt="Finance expert...")
agent3 = Agent(system_prompt="Investment specialist...")
```

### 2. Appropriate Loop Count

Choose the right number of loops for your use case:

- `max_loops=1`: Quick consensus for straightforward tasks
- `max_loops=2`: Balanced refinement for most use cases
- `max_loops=3+`: Deep iterative refinement for complex decisions

### 3. Output Type Selection

Choose the output format that fits your needs:

```python
# Dict - Best for programmatic access to individual responses
voting = MajorityVoting(agents=agents, output_type="dict")

# String - Best for readable output and logging
voting = MajorityVoting(agents=agents, output_type="str")

# List - Best for sequential processing
voting = MajorityVoting(agents=agents, output_type="list")
```

### 4. Consensus Agent Configuration

Use more powerful models for consensus when quality is critical:

```python
voting = MajorityVoting(
    agents=agents,
    consensus_agent_model_name="gpt-4o",  # More powerful for synthesis
    # While agent models can be gpt-4o-mini
)
```

### 5. Verbose Mode for Debugging

Enable verbose mode during development to see the full process:

```python
voting = MajorityVoting(
    agents=agents,
    verbose=True  # See agent execution and consensus process
)
```

---

## Common Patterns

### Pattern 1: Expert Panel

Simulate an expert panel discussion:

```python
experts = [
    Agent(agent_name="Expert-1", system_prompt="Domain expert in area A..."),
    Agent(agent_name="Expert-2", system_prompt="Domain expert in area B..."),
    Agent(agent_name="Expert-3", system_prompt="Domain expert in area C..."),
]

panel = MajorityVoting(agents=experts, max_loops=2)
```

### Pattern 2: Pros and Cons Analysis

Create agents that focus on different aspects:

```python
agents = [
    Agent(system_prompt="Identify all advantages and positive aspects..."),
    Agent(system_prompt="Identify all disadvantages and risks..."),
    Agent(system_prompt="Provide balanced perspective and trade-offs..."),
]

analysis = MajorityVoting(agents=agents, max_loops=1)
```

---

## Related Architectures

| Architecture | When to Use Instead |
|--------------|---------------------|
| **[LLMCouncil](./llm_council_examples.md)** | When you want agents to rank each other's responses and a chairman to synthesize |
| **[CouncilAsAJudge](./council_as_judge_example.md)** | When you need multi-dimensional evaluation (accuracy, helpfulness, etc.) with specialized judges |
| **[DebateWithJudge](../examples/debate_quickstart.md)** | When you want adversarial debate between two opposing positions |
| **[ConcurrentWorkflow](./concurrent_workflow.md)** | When agents work on different tasks rather than the same task |
| **[SequentialWorkflow](./sequential_example.md)** | When tasks need to flow sequentially through agents |

---

## Next Steps

- See [GitHub Examples](https://github.com/kyegomez/swarms/tree/master/examples/multi_agent/majority_voting) for more use cases
- Learn about [Consensus Mechanisms](../concept/consensus_mechanisms.md) in multi-agent systems
- Try [CouncilAsAJudge](./council_as_judge_example.md) for multi-dimensional evaluation
