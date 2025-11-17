# Introduction to Multi-Agent Collaboration

---

## Benefits of Multi-Agent Collaboration

### Why Multi-Agent Architectures?

Multi-agent systems unlock new levels of intelligence, reliability, and efficiency by enabling agents to work together. Here are the core benefits:

| **Benefit**                       | **Description**                                                                                                      |
|------------------------------------|----------------------------------------------------------------------------------------------------------------------|
| **Reduction of Hallucination**     | Cross-verification between agents ensures more accurate, reliable outputs by reducing hallucination.                 |
| **Extended Memory**                | Agents share knowledge and task history, achieving collective long-term memory for smarter, more adaptive responses. |
| **Specialization & Task Distribution** | Delegating tasks to specialized agents boosts efficiency and quality.                                           |
| **Parallel Processing**            | Multiple agents work simultaneously, greatly increasing speed and throughput.                                        |
| **Scalability & Adaptability**     | Systems can dynamically scale and adapt, maintaining efficiency as demands change.                                  |

---

## Multi-Agent Architectures For Production Deployments

`swarms` provides a variety of powerful, pre-built multi-agent architectures enabling you to orchestrate agents in various ways. Choose the right structure for your specific problem to build efficient and reliable production systems.

| **Architecture** | **Description** | **Best For** |
|---|---|---|
| **[SequentialWorkflow](https://docs.swarms.world/en/latest/swarms/structs/sequential_workflow/)** | Agents execute tasks in a linear chain; one agent's output is the next one's input. | Step-by-step processes like data transformation pipelines, report generation. |
| **[ConcurrentWorkflow](https://docs.swarms.world/en/latest/swarms/structs/concurrent_workflow/)** | Agents run tasks simultaneously for maximum efficiency. | High-throughput tasks like batch processing, parallel data analysis. |
| **[AgentRearrange](https://docs.swarms.world/en/latest/swarms/structs/agent_rearrange/)** | Dynamically maps complex relationships (e.g., `a -> b, c`) between agents. | Flexible and adaptive workflows, task distribution, dynamic routing. |
| **[GraphWorkflow](https://docs.swarms.world/en/latest/swarms/structs/graph_workflow/)** | Orchestrates agents as nodes in a Directed Acyclic Graph (DAG). | Complex projects with intricate dependencies, like software builds. |
| **[MixtureOfAgents (MoA)](https://docs.swarms.world/en/latest/swarms/structs/moa/)** | Utilizes multiple expert agents in parallel and synthesizes their outputs. | Complex problem-solving, achieving state-of-the-art performance through collaboration. |
| **[GroupChat](https://docs.swarms.world/en/latest/swarms/structs/group_chat/)** | Agents collaborate and make decisions through a conversational interface. | Real-time collaborative decision-making, negotiations, brainstorming. |
| **[ForestSwarm](https://docs.swarms.world/en/latest/swarms/structs/forest_swarm/)** | Dynamically selects the most suitable agent or tree of agents for a given task. | Task routing, optimizing for expertise, complex decision-making trees. |
| **[SpreadSheetSwarm](https://docs.swarms.world/en/latest/swarms/structs/spreadsheet_swarm/)** | Manages thousands of agents concurrently, tracking tasks and outputs in a structured format. | Massive-scale parallel operations, large-scale data generation and analysis. |
| **[SwarmRouter](https://docs.swarms.world/en/latest/swarms/structs/swarm_router/)** | Universal orchestrator that provides a single interface to run any type of swarm with dynamic selection. | Simplifying complex workflows, switching between swarm strategies, unified multi-agent management. |
| **[HierarchicalSwarm](https://docs.swarms.world/en/latest/swarms/structs/hierarchical_swarm/)** | Director agent coordinates specialized worker agents in a hierarchy. | Complex, multi-stage tasks, iterative refinement, enterprise workflows. |
| **[Hybrid Hierarchical-Cluster Swarm (HHCS)](https://docs.swarms.world/en/latest/swarms/structs/hhcs/)** | Router agent distributes tasks to specialized swarms for parallel, hierarchical processing. | Enterprise-scale, multi-domain, and highly complex workflows. |

---

### HierarchicalSwarm Example

Hierarchical architectures enable structured, iterative, and scalable problem-solving by combining a director (or router) agent with specialized worker agents or swarms. Below are two key patterns:


```python
from swarms import Agent
from swarms.structs.hiearchical_swarm import HierarchicalSwarm

# Create specialized agents
research_agent = Agent(
    agent_name="Research-Specialist",
    agent_description="Expert in market research and analysis",
    model_name="gpt-4.1",
)
financial_agent = Agent(
    agent_name="Financial-Analyst",
    agent_description="Specialist in financial analysis and valuation",
    model_name="gpt-4.1",
)

# Initialize the hierarchical swarm
swarm = HierarchicalSwarm(
    name="Financial-Analysis-Swarm",
    description="A hierarchical swarm for comprehensive financial analysis",
    agents=[research_agent, financial_agent],
    max_loops=2,
    verbose=True,
)

# Execute a complex task
result = swarm.run(task="Analyze the market potential for Tesla (TSLA) stock")
print(result)
```

[Full HierarchicalSwarm Documentation →](https://docs.swarms.world/en/latest/swarms/structs/hierarchical_swarm/)



### SequentialWorkflow

A `SequentialWorkflow` executes tasks in a strict order, forming a pipeline where each agent builds upon the work of the previous one. `SequentialWorkflow` is Ideal for processes that have clear, ordered steps. This ensures that tasks with dependencies are handled correctly.

```python
from swarms import Agent, SequentialWorkflow

# Initialize agents for a 3-step process
# 1. Generate an idea
idea_generator = Agent(agent_name="IdeaGenerator", system_prompt="Generate a unique startup idea.", model_name="gpt-4o-mini")
# 2. Validate the idea
validator = Agent(agent_name="Validator", system_prompt="Take this startup idea and analyze its market viability.", model_name="gpt-4o-mini")
# 3. Create a pitch
pitch_creator = Agent(agent_name="PitchCreator", system_prompt="Write a 3-sentence elevator pitch for this validated startup idea.", model_name="gpt-4o-mini")

# Create the sequential workflow
workflow = SequentialWorkflow(agents=[idea_generator, validator, pitch_creator])

# Run the workflow
elevator_pitch = workflow.run()
print(elevator_pitch)
```

### ConcurrentWorkflow (with `SpreadSheetSwarm`)

A concurrent workflow runs multiple agents simultaneously. `SpreadSheetSwarm` is a powerful implementation that can manage thousands of concurrent agents and log their outputs to a CSV file. Use this architecture for high-throughput tasks that can be performed in parallel, drastically reducing execution time.

```python
from swarms import Agent, SpreadSheetSwarm

# Define a list of tasks (e.g., social media posts to generate)
platforms = ["Twitter", "LinkedIn", "Instagram"]

# Create an agent for each task
agents = [
    Agent(
        agent_name=f"{platform}-Marketer",
        system_prompt=f"Generate a real estate marketing post for {platform}.",
        model_name="gpt-4o-mini",
    )
    for platform in platforms
]

# Initialize the swarm to run these agents concurrently
swarm = SpreadSheetSwarm(
    agents=agents,
    autosave=True,
    save_file_path="marketing_posts.csv",
)

# Run the swarm with a single, shared task description
property_description = "A beautiful 3-bedroom house in sunny California."
swarm.run(task=f"Generate a post about: {property_description}")
# Check marketing_posts.csv for the results!
```

### AgentRearrange

Inspired by `einsum`, `AgentRearrange` lets you define complex, non-linear relationships between agents using a simple string-based syntax. [Learn more](https://docs.swarms.world/en/latest/swarms/structs/agent_rearrange/). This architecture is Perfect for orchestrating dynamic workflows where agents might work in parallel, sequence, or a combination of both.

```python
from swarms import Agent, AgentRearrange

# Define agents
researcher = Agent(agent_name="researcher", model_name="gpt-4o-mini")
writer = Agent(agent_name="writer", model_name="gpt-4o-mini")
editor = Agent(agent_name="editor", model_name="gpt-4o-mini")

# Define a flow: researcher sends work to both writer and editor simultaneously
# This is a one-to-many relationship
flow = "researcher -> writer, editor"

# Create the rearrangement system
rearrange_system = AgentRearrange(
    agents=[researcher, writer, editor],
    flow=flow,
)

# Run the system
# The researcher will generate content, and then both the writer and editor
# will process that content in parallel.
outputs = rearrange_system.run("Analyze the impact of AI on modern cinema.")
print(outputs)
```

---

### SwarmRouter: The Universal Swarm Orchestrator

The `SwarmRouter` simplifies building complex workflows by providing a single interface to run any type of swarm. Instead of importing and managing different swarm classes, you can dynamically select the one you need just by changing the `swarm_type` parameter. [Read the full documentation](https://docs.swarms.world/en/latest/swarms/structs/swarm_router/)

This makes your code cleaner and more flexible, allowing you to switch between different multi-agent strategies with ease. Here's a complete example that shows how to define agents and then use `SwarmRouter` to execute the same task using different collaborative strategies.

```python
from swarms import Agent
from swarms.structs.swarm_router import SwarmRouter, SwarmType

# Define a few generic agents
writer = Agent(agent_name="Writer", system_prompt="You are a creative writer.", model_name="gpt-4o-mini")
editor = Agent(agent_name="Editor", system_prompt="You are an expert editor for stories.", model_name="gpt-4o-mini")
reviewer = Agent(agent_name="Reviewer", system_prompt="You are a final reviewer who gives a score.", model_name="gpt-4o-mini")

# The agents and task will be the same for all examples
agents = [writer, editor, reviewer]
task = "Write a short story about a robot who discovers music."

# --- Example 1: SequentialWorkflow ---
# Agents run one after another in a chain: Writer -> Editor -> Reviewer.
print("Running a Sequential Workflow...")
sequential_router = SwarmRouter(swarm_type=SwarmType.SequentialWorkflow, agents=agents)
sequential_output = sequential_router.run(task)
print(f"Final Sequential Output:\n{sequential_output}\n")

# --- Example 2: ConcurrentWorkflow ---
# All agents receive the same initial task and run at the same time.
print("Running a Concurrent Workflow...")
concurrent_router = SwarmRouter(swarm_type=SwarmType.ConcurrentWorkflow, agents=agents)
concurrent_outputs = concurrent_router.run(task)
# This returns a dictionary of each agent's output
for agent_name, output in concurrent_outputs.items():
    print(f"Output from {agent_name}:\n{output}\n")

# --- Example 3: MixtureOfAgents ---
# All agents run in parallel, and a special 'aggregator' agent synthesizes their outputs.
print("Running a Mixture of Agents Workflow...")
aggregator = Agent(
    agent_name="Aggregator",
    system_prompt="Combine the story, edits, and review into a final document.",
    model_name="gpt-4o-mini"
)
moa_router = SwarmRouter(
    swarm_type=SwarmType.MixtureOfAgents,
    agents=agents,
    aggregator_agent=aggregator, # MoA requires an aggregator
)
aggregated_output = moa_router.run(task)
print(f"Final Aggregated Output:\n{aggregated_output}\n")
```


The `SwarmRouter` is a powerful tool for simplifying multi-agent orchestration. It provides a consistent and flexible way to deploy different collaborative strategies, allowing you to build more sophisticated applications with less code.

---

### MixtureOfAgents (MoA)

The `MixtureOfAgents` architecture processes tasks by feeding them to multiple "expert" agents in parallel. Their diverse outputs are then synthesized by an aggregator agent to produce a final, high-quality result. [Learn more here](https://docs.swarms.world/en/latest/swarms/examples/moa_example/)

```python
from swarms import Agent, MixtureOfAgents

# Define expert agents
financial_analyst = Agent(agent_name="FinancialAnalyst", system_prompt="Analyze financial data.", model_name="gpt-4o-mini")
market_analyst = Agent(agent_name="MarketAnalyst", system_prompt="Analyze market trends.", model_name="gpt-4o-mini")
risk_analyst = Agent(agent_name="RiskAnalyst", system_prompt="Analyze investment risks.", model_name="gpt-4o-mini")

# Define the aggregator agent
aggregator = Agent(
    agent_name="InvestmentAdvisor",
    system_prompt="Synthesize the financial, market, and risk analyses to provide a final investment recommendation.",
    model_name="gpt-4o-mini"
)

# Create the MoA swarm
moa_swarm = MixtureOfAgents(
    agents=[financial_analyst, market_analyst, risk_analyst],
    aggregator_agent=aggregator,
)

# Run the swarm
recommendation = moa_swarm.run("Should we invest in NVIDIA stock right now?")
print(recommendation)
```

---

### GroupChat

`GroupChat` creates a conversational environment where multiple agents can interact, discuss, and collaboratively solve a problem. You can define the speaking order or let it be determined dynamically. This architecture is ideal for tasks that benefit from debate and multi-perspective reasoning, such as contract negotiation, brainstorming, or complex decision-making.

```python
from swarms import Agent, GroupChat

# Define agents for a debate
tech_optimist = Agent(agent_name="TechOptimist", system_prompt="Argue for the benefits of AI in society.", model_name="gpt-4o-mini")
tech_critic = Agent(agent_name="TechCritic", system_prompt="Argue against the unchecked advancement of AI.", model_name="gpt-4o-mini")

# Create the group chat
chat = GroupChat(
    agents=[tech_optimist, tech_critic],
    max_loops=4, # Limit the number of turns in the conversation
)

# Run the chat with an initial topic
conversation_history = chat.run(
    "Let's discuss the societal impact of artificial intelligence."
)

# Print the full conversation
for message in conversation_history:
    print(f"[{message['agent_name']}]: {message['content']}")
```

--

## Connect With Us

Join our community of agent engineers and researchers for technical support, cutting-edge updates, and exclusive access to world-class agent engineering insights!

| Platform | Description | Link |
|----------|-------------|------|
| Documentation | Official documentation and guides | [docs.swarms.world](https://docs.swarms.world) |
| Blog | Latest updates and technical articles | [Medium](https://medium.com/@kyeg) |
| Discord | Live chat and community support | [Join Discord](https://discord.gg/EamjgSaEQf) |
| Twitter | Latest news and announcements | [@kyegomez](https://twitter.com/kyegomez) |
| LinkedIn | Professional network and updates | [The Swarm Corporation](https://www.linkedin.com/company/the-swarm-corporation) |
| YouTube | Tutorials and demos | [Swarms Channel](https://www.youtube.com/channel/UC9yXyXyitkbU_WSy7bd_41SqQ) |
| Events | Join our community events | [Sign up here](https://lu.ma/swarms_calendar) |
| Onboarding Session | Get onboarded with Kye Gomez, creator and lead maintainer of Swarms | [Book Session](https://cal.com/swarms/swarms-onboarding-session) |

---

