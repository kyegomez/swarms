# MajorityVoting: Practical Tutorial

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
pip install -U swarms
```

---

## Basic Example

```python
from swarms import Agent, MajorityVoting

# Create agents with different perspectives
agent1 = Agent(
    agent_name="Conservative-Analyst",
    system_prompt=(
        "You are a conservative financial advisor focused on "
        "risk management, capital preservation, and long-term stability."
    ),
    model_name="gpt-4o-mini",
    max_loops=1,
)

agent2 = Agent(
    agent_name="Growth-Analyst",
    system_prompt=(
        "You are a growth-oriented financial advisor focused on "
        "high-potential opportunities, emerging markets, and innovation."
    ),
    model_name="gpt-4o-mini",
    max_loops=1,
)

agent3 = Agent(
    agent_name="Balanced-Analyst",
    system_prompt=(
        "You are a balanced financial advisor focused on "
        "diversification, risk-adjusted returns, and portfolio optimization."
    ),
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Initialize majority voting system
voting_system = MajorityVoting(
    agents=[agent1, agent2, agent3],
    max_loops=1,
    output_type="dict",
    verbose=True
)

# Run the voting system
result = voting_system.run(
    task="Create a comprehensive investment strategy for a 35-year-old with $100k to invest. Consider risk tolerance, time horizon, and diversification."
)

# Access the result
print(result)
```

### Output

The output will contain the conversation history including:
1. Each agent's independent analysis
2. The consensus agent's comprehensive evaluation and synthesis
3. Final recommendations based on all perspectives

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

---

## Advanced Example 1: Multi-Loop Consensus

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

---

## Advanced Example 2: Custom Consensus Agent

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

## Advanced Example 3: Batch Processing with Concurrent Execution

Process multiple tasks efficiently:

```python
from swarms import Agent, MajorityVoting

# Create agent team
agents = [
    Agent(agent_name="Analyst-1", system_prompt="Financial analyst specializing in equities", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Analyst-2", system_prompt="Financial analyst specializing in fixed income", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Analyst-3", system_prompt="Financial analyst specializing in alternative investments", model_name="gpt-4o-mini", max_loops=1),
]

voting_system = MajorityVoting(
    agents=agents,
    max_loops=1,
    verbose=False  # Disable verbose for batch processing
)

# Multiple analysis tasks
tasks = [
    "Analyze the outlook for technology sector in 2024",
    "Evaluate the impact of interest rate changes on bond markets",
    "Assess cryptocurrency as an alternative investment",
    "Review ESG investment opportunities",
]

# Sequential batch processing
results = voting_system.batch_run(tasks)

# Concurrent batch processing (faster)
concurrent_results = voting_system.run_concurrently(tasks)

# Display results
for task, result in zip(tasks, results):
    print(f"\n{'='*60}")
    print(f"Task: {task}")
    print(f"{'='*60}")
    print(result)
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

### Use Case 3: Research Synthesis

```python
from swarms import Agent, MajorityVoting

# Different research perspectives
agents = [
    Agent(
        agent_name="Quantitative-Researcher",
        system_prompt="You focus on quantitative analysis, statistical methods, and numerical data.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="Qualitative-Researcher",
        system_prompt="You focus on qualitative analysis, thematic patterns, and contextual understanding.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="Literature-Reviewer",
        system_prompt="You synthesize existing research, identify gaps, and connect findings across studies.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
]

voting = MajorityVoting(agents=agents, max_loops=2)
result = voting.run("Synthesize current research on the effectiveness of remote work on employee productivity")
```

### Use Case 4: Legal Analysis

```python
from swarms import Agent, MajorityVoting

# Different legal perspectives
agents = [
    Agent(
        agent_name="Corporate-Lawyer",
        system_prompt="You analyze from corporate law perspective: contracts, governance, and regulatory compliance.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="IP-Lawyer",
        system_prompt="You analyze from intellectual property perspective: patents, trademarks, and licensing.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="Privacy-Lawyer",
        system_prompt="You analyze from privacy law perspective: data protection, GDPR, and user consent.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
]

voting = MajorityVoting(agents=agents, max_loops=1)
result = voting.run("Analyze legal considerations for launching a new AI-powered SaaS product in the EU")
```

### Use Case 5: Strategic Business Decision

```python
from swarms import Agent, MajorityVoting

# Different business perspectives
agents = [
    Agent(
        agent_name="CFO-Perspective",
        system_prompt="You analyze from financial perspective: ROI, cash flow, and financial risk.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="CMO-Perspective",
        system_prompt="You analyze from marketing perspective: brand impact, market positioning, and customer acquisition.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="CTO-Perspective",
        system_prompt="You analyze from technology perspective: technical feasibility, scalability, and innovation.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
]

voting = MajorityVoting(agents=agents, max_loops=2)
result = voting.run("Should we acquire a smaller competitor or build the technology in-house?")
```

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

### Pattern 3: Multi-Perspective Research

Different research methodologies:

```python
researchers = [
    Agent(system_prompt="Empirical research approach with data focus..."),
    Agent(system_prompt="Theoretical research approach with conceptual models..."),
    Agent(system_prompt="Practical research approach with real-world applications..."),
]

research_team = MajorityVoting(agents=researchers, max_loops=2)
```

---

## Related Architectures

| Architecture | When to Use Instead |
|--------------|---------------------|
| **[LLMCouncil](./llm_council_examples.md)** | When you want agents to rank each other's responses and a chairman to synthesize |
| **[CouncilAsAJudge](./council_as_judge_example.md)** | When you need multi-dimensional evaluation (accuracy, helpfulness, etc.) with specialized judges |
| **[DebateWithJudge](./debate_quickstart.md)** | When you want adversarial debate between two opposing positions |
| **[ConcurrentWorkflow](./concurrent_workflow.md)** | When agents work on different tasks rather than the same task |
| **[SequentialWorkflow](./sequential_example.md)** | When tasks need to flow sequentially through agents |

---

## API Reference

### MajorityVoting Class

```python
class MajorityVoting:
    def __init__(
        self,
        id: str = swarm_id(),
        name: str = "MajorityVoting",
        description: str = "A multi-loop majority voting system for agents",
        agents: List[Agent] = None,
        autosave: bool = False,
        verbose: bool = False,
        max_loops: int = 1,
        output_type: str = "dict",
        consensus_agent_prompt: str = CONSENSUS_AGENT_PROMPT,
        consensus_agent_name: str = "Consensus-Agent",
        consensus_agent_description: str = "An agent that uses consensus to generate a final answer.",
        consensus_agent_model_name: str = "gpt-4.1",
        additional_consensus_agent_kwargs: dict = {},
    )

    def run(self, task: str) -> Any:
        """Run the majority voting system and return the consensus."""

    def batch_run(self, tasks: List[str]) -> List[Any]:
        """Run multiple tasks sequentially."""

    def run_concurrently(self, tasks: List[str]) -> List[Any]:
        """Run multiple tasks concurrently using ThreadPoolExecutor."""
```

---

## Troubleshooting

### Issue: Agents producing too similar responses

**Solution**: Ensure agents have truly distinct system prompts and perspectives.

### Issue: Consensus agent not synthesizing well

**Solution**: Use a more powerful model for the consensus agent or customize the consensus prompt.

### Issue: Slow performance

**Solution**:
- Reduce `max_loops`
- Use lighter models for agents (keep gpt-4o for consensus)
- Use `run_concurrently()` for batch tasks

### Issue: Output format not as expected

**Solution**: Verify `output_type` parameter matches your needs ("dict", "str", or "list").

---

## Next Steps

- Explore [MajorityVoting Quickstart](../../examples/majority_voting_quickstart.md)
- See [GitHub Examples](https://github.com/kyegomez/swarms/tree/master/examples/multi_agent/majority_voting)
- Learn about [Multi-Agent Architectures](../../examples/multi_agent_architectures_overview.md)
- Try [CouncilAsAJudge](./council_as_judge_example.md) for multi-dimensional evaluation
