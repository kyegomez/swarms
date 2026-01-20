# SocialAlgorithms: Complete Guide

A comprehensive guide to creating custom multi-agent communication patterns using the SocialAlgorithms framework.

## Overview

**SocialAlgorithms** is a flexible framework that lets you define custom communication patterns between agents as Python callables. Instead of being limited to pre-built architectures, you can create any interaction pattern that suits your specific use case - from simple sequential pipelines to complex negotiation protocols.

| Feature | Description |
|---------|-------------|
| **Custom Communication Patterns** | Define any algorithm as a Python function |
| **Complete Control** | Specify exact order, flow, and logic of agent interactions |
| **Communication Tracking** | Optional logging of all agent-to-agent messages |
| **Timeout Protection** | Configurable execution time limits prevent runaway processes |
| **Structured Results** | SocialAlgorithmResult with execution metrics and history |
| **Flexible Output** | Support for dict, list, and string output formats |

### When to Use SocialAlgorithms

**Best For:**
- Custom communication patterns not available in pre-built architectures
- Complex multi-stage workflows with specific logic
- Experimental or novel agent coordination strategies
- Domain-specific interaction protocols

**Not Ideal For:**
- Simple use cases where pre-built architectures suffice
- When you need battle-tested, optimized patterns
- Real-time performance-critical applications

---

## Installation

```bash
pip install swarms
```

---

## Quick Start

### Step 1: Define Your Algorithm

```python
def research_analysis_synthesis(agents, task, **kwargs):
    """
    Custom algorithm: Research → Analysis → Synthesis
    """
    # Agent 0: Research
    research = agents[0].run(f"Research the topic: {task}")

    # Agent 1: Analyze the research
    analysis = agents[1].run(f"Analyze this research: {research}")

    # Agent 2: Synthesize findings
    synthesis = agents[2].run(
        f"Synthesize these findings:\nResearch: {research}\nAnalysis: {analysis}"
    )

    return {
        "research": research,
        "analysis": analysis,
        "synthesis": synthesis
    }
```

### Step 2: Create and Run

```python
from swarms import Agent, SocialAlgorithms

# Create agents
researcher = Agent(
    agent_name="Researcher",
    system_prompt="You are a research specialist. Gather comprehensive information.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

analyst = Agent(
    agent_name="Analyst",
    system_prompt="You are an analyst. Interpret data and identify insights.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

synthesizer = Agent(
    agent_name="Synthesizer",
    system_prompt="You synthesize information into clear recommendations.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Create social algorithm
social_alg = SocialAlgorithms(
    name="Research-Analysis-Synthesis",
    agents=[researcher, analyst, synthesizer],
    social_algorithm=research_analysis_synthesis,
    verbose=True
)

# Run
result = social_alg.run("The impact of AI on healthcare")
print(result.final_outputs)
```

---

## Basic Example

```python
from swarms import Agent, SocialAlgorithms

# Define a custom algorithm
def research_pipeline(agents, task, **kwargs):
    """Research → Analysis → Synthesis pipeline"""
    # Step 1: Research
    research_result = agents[0].run(f"Research: {task}")

    # Step 2: Analyze
    analysis_result = agents[1].run(f"Analyze: {research_result}")

    # Step 3: Synthesize
    synthesis_result = agents[2].run(
        f"Synthesize findings:\nResearch: {research_result}\nAnalysis: {analysis_result}"
    )

    return {
        "research": research_result,
        "analysis": analysis_result,
        "synthesis": synthesis_result
    }

# Create agents
agents = [
    Agent(agent_name="Researcher", system_prompt="Research specialist", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Analyst", system_prompt="Analysis specialist", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Synthesizer", system_prompt="Synthesis specialist", model_name="gpt-4o-mini", max_loops=1),
]

# Create and run algorithm
social_alg = SocialAlgorithms(
    name="Research-Pipeline",
    agents=agents,
    social_algorithm=research_pipeline,
    verbose=True
)

result = social_alg.run("Impact of quantum computing on cybersecurity")
print(result.final_outputs)
```

---

## Key Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `agents` | `List[Agent]` | Required | Agents participating in the algorithm |
| `social_algorithm` | `Callable` | Required | Function defining communication pattern |
| `max_execution_time` | `float` | `300.0` | Maximum execution time in seconds |
| `output_type` | `str` | `"dict"` | Output format: "dict", "list", "str" |
| `verbose` | `bool` | `False` | Enable detailed logging |
| `enable_communication_logging` | `bool` | `False` | Track all communications |
| `algorithm_id` | `str` | Auto-generated UUID | Unique identifier |

---

## Advanced Examples

### Example 1: Consensus Building

```python
from swarms import Agent, SocialAlgorithms

def consensus_building(agents, task, **kwargs):
    """Multi-round consensus building with voting"""
    max_rounds = kwargs.get("max_rounds", 3)
    consensus_threshold = kwargs.get("consensus_threshold", 0.75)

    stakeholders = agents[:-1]  # All but last agent
    facilitator = agents[-1]  # Last agent facilitates

    # Round 1: Initial positions
    positions = {}
    for agent in stakeholders:
        position = agent.run(f"Your position on: {task}")
        positions[agent.agent_name] = position

    # Rounds 2-N: Discussion and convergence
    for round_num in range(1, max_rounds):
        # Facilitator summarizes
        summary = facilitator.run(
            f"Summarize these positions and identify common ground: {positions}"
        )

        # Agents revise positions
        for agent in stakeholders:
            revised = agent.run(
                f"Given this summary: {summary}\nRevise your position on: {task}"
            )
            positions[agent.agent_name] = revised

    # Final consensus
    final_consensus = facilitator.run(
        f"Create final consensus from: {positions}"
    )

    return {
        "positions": positions,
        "consensus": final_consensus,
        "rounds": max_rounds
    }

# Implementation
agents = [
    Agent(agent_name="Tech-Lead", system_prompt="Technical perspective", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Business-Lead", system_prompt="Business perspective", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="UX-Lead", system_prompt="User experience perspective", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Facilitator", system_prompt="Neutral facilitator", model_name="gpt-4o-mini", max_loops=1),
]

social_alg = SocialAlgorithms(
    name="Consensus-Builder",
    agents=agents,
    social_algorithm=consensus_building,
)

result = social_alg.run(
    "Should we rebuild our app in React or Vue?",
    algorithm_args={"max_rounds": 3, "consensus_threshold": 0.75}
)
```

### Example 2: Hierarchical Decision Making

```python
def hierarchical_decision(agents, task, **kwargs):
    """Manager → Workers → Manager pattern"""
    manager = agents[0]
    workers = agents[1:]

    # Manager breaks down task
    breakdown = manager.run(
        f"Break down this task into subtasks for {len(workers)} workers: {task}"
    )

    # Workers execute subtasks
    worker_results = {}
    for i, worker in enumerate(workers):
        subtask_result = worker.run(f"Execute subtask {i+1}: {breakdown}")
        worker_results[worker.agent_name] = subtask_result

    # Manager synthesizes results
    final_decision = manager.run(
        f"Synthesize worker results into final decision:\n{worker_results}"
    )

    return {
        "task_breakdown": breakdown,
        "worker_results": worker_results,
        "final_decision": final_decision
    }

agents = [
    Agent(agent_name="Manager", system_prompt="Strategic manager", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Worker-1", system_prompt="Implementation specialist", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Worker-2", system_prompt="Quality specialist", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="Worker-3", system_prompt="Testing specialist", model_name="gpt-4o-mini", max_loops=1),
]

social_alg = SocialAlgorithms(
    name="Hierarchical-Team",
    agents=agents,
    social_algorithm=hierarchical_decision,
)

result = social_alg.run("Design and implement a new authentication system")
```

### Example 3: Peer Review System

```python
def peer_review_algorithm(agents, task, **kwargs):
    """Round-robin peer review with final synthesis"""
    reviewers = agents[:-1]
    synthesizer = agents[-1]

    # Each reviewer evaluates independently
    reviews = {}
    for reviewer in reviewers:
        review = reviewer.run(f"Review this proposal: {task}")
        reviews[reviewer.agent_name] = review

    # Peer review: Each reviewer comments on others' reviews
    peer_comments = {}
    for reviewer in reviewers:
        other_reviews = {k: v for k, v in reviews.items() if k != reviewer.agent_name}
        comments = reviewer.run(
            f"Comment on these peer reviews: {other_reviews}"
        )
        peer_comments[reviewer.agent_name] = comments

    # Final synthesis
    synthesis = synthesizer.run(
        f"Synthesize all reviews and peer comments:\nReviews: {reviews}\nPeer Comments: {peer_comments}"
    )

    return {
        "reviews": reviews,
        "peer_comments": peer_comments,
        "synthesis": synthesis
    }
```

---

## Algorithm Pattern Library

### Pattern 1: Sequential Pipeline

```python
def sequential_pipeline(agents, task, **kwargs):
    """A → B → C → D..."""
    result = task
    for agent in agents:
        result = agent.run(result)
    return result
```

### Pattern 2: Parallel Then Aggregate

```python
def parallel_aggregate(agents, task, **kwargs):
    """All work in parallel, then aggregate"""
    workers = agents[:-1]
    aggregator = agents[-1]

    # Parallel execution
    results = [worker.run(task) for worker in workers]

    # Aggregate
    aggregate = aggregator.run(f"Aggregate these results: {results}")

    return {"individual": results, "aggregate": aggregate}
```

### Pattern 3: Debate

```python
def debate_algorithm(agents, task, **kwargs):
    """Agents debate over multiple rounds"""
    rounds = kwargs.get("rounds", 3)
    discussion = []

    for round_num in range(rounds):
        for agent in agents:
            context = "\n".join(discussion[-10:])  # Last 10 messages
            response = agent.run(
                f"Round {round_num+1}: {task}\n\nDiscussion so far:\n{context}"
            )
            discussion.append(f"[{agent.agent_name}]: {response}")

    return {"discussion": discussion}
```

### Pattern 4: Auction/Bidding

```python
def auction_algorithm(agents, task, **kwargs):
    """Agents bid, highest bid wins"""
    bidders = agents[:-1]
    auctioneer = agents[-1]

    # Collect bids
    bids = {}
    for bidder in bidders:
        bid = bidder.run(f"Submit your bid for: {task}")
        bids[bidder.agent_name] = bid

    # Auctioneer selects winner
    winner = auctioneer.run(f"Select winning bid from: {bids}")

    return {"bids": bids, "winner": winner}
```

---

## Use Cases

### Use Case 1: Multi-Stage Content Creation

```python
def content_creation_pipeline(agents, task, **kwargs):
    research_agent, outline_agent, writer_agent, editor_agent = agents

    research = research_agent.run(f"Research: {task}")
    outline = outline_agent.run(f"Create outline from: {research}")
    draft = writer_agent.run(f"Write based on: {outline}")
    final = editor_agent.run(f"Edit and polish: {draft}")

    return {
        "research": research,
        "outline": outline,
        "draft": draft,
        "final": final
    }
```

### Use Case 2: Code Review Workflow

```python
def code_review_workflow(agents, task, **kwargs):
    security_reviewer, performance_reviewer, style_reviewer, lead_reviewer = agents

    security = security_reviewer.run(f"Security review: {task}")
    performance = performance_reviewer.run(f"Performance review: {task}")
    style = style_reviewer.run(f"Style review: {task}")

    final_review = lead_reviewer.run(
        f"Final review incorporating:\nSecurity: {security}\nPerformance: {performance}\nStyle: {style}"
    )

    return {
        "security_review": security,
        "performance_review": performance,
        "style_review": style,
        "final_review": final_review
    }
```

---

## Best Practices

1. **Structure Your Algorithm**: Break complex algorithms into clear phases
2. **Handle Errors**: Wrap agent calls in try/except blocks
3. **Return Structured Data**: Use dictionaries with meaningful keys
4. **Use Kwargs**: Make algorithms configurable via **kwargs
5. **Enable Logging**: Use `verbose=True` and `enable_communication_logging=True` during development
6. **Set Timeouts**: Always set appropriate `max_execution_time`
7. **Document Your Algorithm**: Add docstrings explaining the communication pattern

---

## Communication Logging

```python
social_alg = SocialAlgorithms(
    name="My-Algorithm",
    agents=agents,
    social_algorithm=my_algorithm,
    enable_communication_logging=True,
    verbose=True
)

result = social_alg.run("Task...")

# Access detailed communication history
for step in result.communication_history:
    print(f"[{step.timestamp}] {step.sender_agent} → {step.receiver_agent}")
    print(f"Message: {step.message[:100]}...")
    print(f"Metadata: {step.metadata}\n")

# Check execution metrics
print(f"Execution time: {result.execution_time:.2f}s")
print(f"Total steps: {result.total_steps}")
print(f"Successful: {result.successful_steps}")
print(f"Failed: {result.failed_steps}")
```

---

## Related Architectures

| Architecture | Relationship |
|--------------|--------------|
| **[SequentialWorkflow](./sequential_example.md)** | Pre-built sequential pattern |
| **[ConcurrentWorkflow](./concurrent_workflow.md)** | Pre-built parallel pattern |
| **[RoundRobinSwarm](./roundrobin_example.md)** | Pre-built randomized discussion pattern |

---

## Next Steps

- Explore [SocialAlgorithms Quickstart](../../examples/social_algorithms_quickstart.md)
- See [12+ Algorithm Examples](https://github.com/kyegomez/swarms/tree/master/examples/multi_agent/social_algorithms_examples)
- Study patterns: Consensus, Negotiation, Peer Review, Auction, Hierarchical
- Design your own custom communication protocols
