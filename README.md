<div align="left">
  <a href="https://swarms.world">
    <img src="https://github.com/kyegomez/swarms/blob/master/images/new_logo.png" style="margin: 15px; max-width: 350px" width="70%" alt="Logo">
  </a>
</div>


<p align="left">
  <!-- Main Navigation Links -->
  <a href="https://swarms.ai">Swarms Website</a>
  <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
  <a href="https://docs.swarms.world">Documentation</a>
  <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
  <a href="https://swarms.world">Swarms Marketplace</a>
</p>


<p align="left">
  <a href="https://pypi.org/project/swarms/" target="_blank">
    <picture>
      <source srcset="https://img.shields.io/pypi/v/swarms?style=for-the-badge&color=3670A0" media="(prefers-color-scheme: dark)">
      <img alt="Version" src="https://img.shields.io/pypi/v/swarms?style=for-the-badge&color=3670A0">
    </picture>
  </a>
  <a href="https://pypi.org/project/swarms/" target="_blank">
    <picture>
      <source srcset="https://img.shields.io/pypi/dm/swarms?style=for-the-badge&color=3670A0" media="(prefers-color-scheme: dark)">
      <img alt="Downloads" src="https://img.shields.io/pypi/dm/swarms?style=for-the-badge&color=3670A0">
    </picture>
  </a>
  <a href="https://twitter.com/swarms_corp/">
    <picture>
      <source srcset="https://img.shields.io/badge/Twitter-Follow-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white" media="(prefers-color-scheme: dark)">
      <img src="https://img.shields.io/badge/Twitter-Follow-1DA1F2?style=for-the-badge&logo=twitter&logoColor=white" alt="Twitter">
    </picture>
  </a>
  <a href="https://discord.gg/EamjgSaEQf">
    <picture>
      <source srcset="https://img.shields.io/badge/Discord-Join-5865F2?style=for-the-badge&logo=discord&logoColor=white" media="(prefers-color-scheme: dark)">
      <img src="https://img.shields.io/badge/Discord-Join-5865F2?style=for-the-badge&logo=discord&logoColor=white" alt="Discord">
    </picture>
  </a>
</p>

## Overview

>
> Swarms, The Enterprise-Grade Production-Ready Multi-Agent Orchestration Framework 

Swarms is the most reliable, scalable, and adaptive multi-agent orchestration framework available today. We provide a comprehensive suite of production-ready, prebuilt multi-agent architectures, including sequential, concurrent, and hierarchical systems. Additionally, Swarms offers backward compatibility with leading agent frameworks and interoperability with protocols such as MCP, x402, skills, and much more.


## Install

### Using pip

```bash
$ pip3 install -U swarms
```

### Using uv (Recommended)

[uv](https://github.com/astral-sh/uv) is a fast Python package installer and resolver, written in Rust.

```bash
$ uv pip install swarms
```

### Using poetry

```bash
$ poetry add swarms
```

### From source

```bash
# Clone the repository
$ git clone https://github.com/kyegomez/swarms.git
$ cd swarms
$ pip install -r requirements.txt
```

<!-- ### Using Docker

The easiest way to get started with Swarms is using our pre-built Docker image:

```bash
# Pull and run the latest image
$ docker pull kyegomez/swarms:latest
$ docker run --rm kyegomez/swarms:latest python -c "import swarms; print('Swarms is ready!')"

# Run interactively for development
$ docker run -it --rm -v $(pwd):/app kyegomez/swarms:latest bash

# Using docker-compose (recommended for development)
$ docker-compose up -d
```

For more Docker options and advanced usage, see our [Docker documentation](/scripts/docker/DOCKER.md). -->

---

## Environment Configuration

[Learn more about the environment configuration here](https://docs.swarms.world/environment-setup)

```
OPENAI_API_KEY=""
WORKSPACE_DIR="agent_workspace"
ANTHROPIC_API_KEY=""
GROQ_API_KEY=""
```


### Your First Agent

An **Agent** is the fundamental building block of a swarm—an autonomous entity powered by an LLM + Tools + Memory. [Learn more Here](https://docs.swarms.world/api/agent)

```python
from swarms import Agent

# Initialize a new agent
agent = Agent(
    model_name="gpt-5.4", # Specify the LLM
    max_loops="auto",              # Set the number of interactions
    interactive=True,         # Enable interactive mode for real-time feedback
)

# Run the agent with a task
agent.run("What are the key benefits of using a multi-agent system?")
```

### Autonomous Agent with `max_loops="auto"`

Setting `max_loops="auto"` lets the agent decide for itself when the task is complete — it keeps reasoning and acting until it reaches a stopping condition, rather than halting after a fixed number of iterations. This is the recommended mode for open-ended, multi-step tasks where the number of steps isn't known in advance.

```python
from swarms import Agent

agent = Agent(
    agent_name="Autonomous-Research-Agent",
    agent_description="An autonomous agent that conducts multi-step research independently.",
    system_prompt=(
        "You are an autonomous research agent. Break down complex tasks into steps, "
        "execute each step thoroughly, and signal completion only when the full task is done."
    ),
    model_name="gpt-5.4",
    max_loops="auto",       # Agent decides when it's done — no fixed iteration cap
    autosave=True,
    verbose=True,
)

# The agent will keep looping — planning, executing, and reflecting — until it
# determines the task is fully complete.
result = agent.run(
    "Research the current state of quantum computing, identify the top three "
    "hardware approaches, and summarize the key challenges each faces."
)
print(result)
```

**When to use `max_loops="auto"`:**
- Open-ended research or analysis tasks
- Tasks that require iterative refinement (e.g., write → review → revise)
- Any workflow where the number of steps depends on intermediate results

**When to use a fixed `max_loops` value:**
- Latency-sensitive or cost-sensitive production pipelines
- Tasks with a well-defined, bounded number of steps

### Your First Swarm: Multi-Agent Collaboration

A **Swarm** consists of multiple agents working together. This simple example creates a two-agent workflow for researching and writing a blog post. [Learn More About SequentialWorkflow](https://docs.swarms.world/api/sequential-workflow)

```python
from swarms import Agent, SequentialWorkflow

# Agent 1: The Researcher
researcher = Agent(
    agent_name="Researcher",
    system_prompt="Your job is to research the provided topic and provide a detailed summary.",
    model_name="gpt-5.4",
)

# Agent 2: The Writer
writer = Agent(
    agent_name="Writer",
    system_prompt="Your job is to take the research summary and write a beautiful, engaging blog post about it.",
    model_name="gpt-5.4",
)

# Create a sequential workflow where the researcher's output feeds into the writer's input
workflow = SequentialWorkflow(agents=[researcher, writer])

# Run the workflow on a task
final_post = workflow.run("The history and future of artificial intelligence")
print(final_post)

```

-----

## Available Multi-Agent Architectures

`swarms` provides a variety of powerful, pre-built multi-agent architectures enabling you to orchestrate agents in various ways. Choose the right structure for your specific problem to build efficient and reliable production systems.

| **Architecture** | **Description** | **Best For** |
|---|---|---|
| **[SequentialWorkflow](https://docs.swarms.world/api/sequential-workflow)** | Agents execute tasks in a linear chain; the output of one agent becomes the input for the next. | Step-by-step processes such as data transformation pipelines and report generation. |
| **[ConcurrentWorkflow](https://docs.swarms.world/api/concurrent-workflow)** | Agents run tasks simultaneously for maximum efficiency. | High-throughput tasks such as batch processing and parallel data analysis. |
| **[AgentRearrange](https://docs.swarms.world/api/agent-rearrange)** | Dynamically maps complex relationships (e.g., `a -> b, c`) between agents. | Flexible and adaptive workflows, task distribution, and dynamic routing. |
| **[GraphWorkflow](https://docs.swarms.world/api/graph-workflow)** | Orchestrates agents as nodes in a Directed Acyclic Graph (DAG). | Complex projects with intricate dependencies, such as software builds. |
| **[MixtureOfAgents (MoA)](https://docs.swarms.world/api/mixture-of-agents)** | Utilizes multiple expert agents in parallel and synthesizes their outputs. | Complex problem-solving and achieving state-of-the-art performance through collaboration. |
| **[GroupChat](https://docs.swarms.world/api/group-chat)** | Agents collaborate and make decisions through a conversational interface. | Real-time collaborative decision-making, negotiations, and brainstorming. |
| **[ForestSwarm](https://docs.swarms.world/api/forest-swarm)** | Dynamically selects the most suitable agent or tree of agents for a given task. | Task routing, optimizing for expertise, and complex decision-making trees. |
| **[HierarchicalSwarm](https://docs.swarms.world/api/hierarchical-swarm)** | Orchestrates agents with a director who creates plans and distributes tasks to specialized worker agents. | Complex project management, team coordination, and hierarchical decision-making with feedback loops. |
| **[HeavySwarm](https://docs.swarms.world/api/heavy-swarm)** | Implements a five-phase workflow with specialized agents (Research, Analysis, Alternatives, Verification) for comprehensive task analysis. | Complex research and analysis tasks, financial analysis, strategic planning, and comprehensive reporting. |
| **[SwarmRouter](https://docs.swarms.world/api/swarm-router)** | A universal orchestrator that provides a single interface to run any type of swarm with dynamic selection. | Simplifying complex workflows, switching between swarm strategies, and unified multi-agent management. |

Learn more about all of the 60+ Multi-Agent Structures we have available [here](/docs/MULTI_AGENT_STRUCTURES.md)

-----

### SequentialWorkflow

A `SequentialWorkflow` executes tasks in a strict order, forming a pipeline where each agent builds upon the work of the previous one. `SequentialWorkflow` is Ideal for processes that have clear, ordered steps. This ensures that tasks with dependencies are handled correctly.

```python
from swarms import Agent, SequentialWorkflow

# Agent 1: The Researcher
researcher = Agent(
    agent_name="Researcher",
    system_prompt="Your job is to research the provided topic and provide a detailed summary.",
    model_name="gpt-5.4",
)

# Agent 2: The Writer
writer = Agent(
    agent_name="Writer",
    system_prompt="Your job is to take the research summary and write a beautiful, engaging blog post about it.",
    model_name="gpt-5.4",
)

# Create a sequential workflow where the researcher's output feeds into the writer's input
workflow = SequentialWorkflow(agents=[researcher, writer])

# Run the workflow on a task
final_post = workflow.run("The history and future of artificial intelligence")
print(final_post)
```

-----


### ConcurrentWorkflow

A `ConcurrentWorkflow` runs multiple agents simultaneously, allowing for parallel execution of tasks. This architecture drastically reduces execution time for tasks that can be performed in parallel, making it ideal for high-throughput scenarios where agents work on similar tasks concurrently.

```python
from swarms import Agent, ConcurrentWorkflow

# Create agents for different analysis tasks
market_analyst = Agent(
    agent_name="Market-Analyst",
    system_prompt="Analyze market trends and provide insights on the given topic.",
    model_name="gpt-5.4",
    max_loops=1,
)

financial_analyst = Agent(
    agent_name="Financial-Analyst", 
    system_prompt="Provide financial analysis and recommendations on the given topic.",
    model_name="gpt-5.4",
    max_loops=1,
)

risk_analyst = Agent(
    agent_name="Risk-Analyst",
    system_prompt="Assess risks and provide risk management strategies for the given topic.",
    model_name="gpt-5.4", 
    max_loops=1,
)

# Create concurrent workflow
concurrent_workflow = ConcurrentWorkflow(
    agents=[market_analyst, financial_analyst, risk_analyst],
    max_loops=1,
)

# Run all agents concurrently on the same task
results = concurrent_workflow.run(
    "Analyze the potential impact of AI technology on the healthcare industry"
)

print(results)
```

---

### AgentRearrange

Inspired by `einsum`, `AgentRearrange` lets you define complex, non-linear relationships between agents using a simple string-based syntax. [Learn more](https://docs.swarms.world/api/agent-rearrange). This architecture is perfect for orchestrating dynamic workflows where agents might work in parallel, in sequence, or in any combination you choose.

```python
from swarms import Agent, AgentRearrange

# Define agents
researcher = Agent(agent_name="researcher", model_name="gpt-5.4")
writer = Agent(agent_name="writer", model_name="gpt-5.4")
editor = Agent(agent_name="editor", model_name="gpt-5.4")

# Define a flow: researcher sends work to both writer and editor simultaneously
# This is a one-to-many relationship
flow = "researcher -> writer, editor"

# Create the rearrangement system
rearrange_system = AgentRearrange(
    agents=[researcher, writer, editor],
    flow=flow,
)

# Run the swarm
outputs = rearrange_system.run("Analyze the impact of AI on modern cinema.")
print(outputs)
```


### GraphWorkflow

`GraphWorkflow` orchestrates agents as nodes in a Directed Acyclic Graph (DAG). Each node is an agent and each edge declares a dependency, so a node only runs after every upstream node has finished. A topological sort guarantees correct execution order, while independent branches run in parallel automatically. 

This makes `GraphWorkflow` the right choice when your workflow has fan-out / fan-in patterns, conditional dependencies, or any structure that doesn't fit a strict line or a flat parallel batch. [Learn more about GraphWorkflow](https://docs.swarms.world/api/graph-workflow)

```python
from swarms import Agent, GraphWorkflow, Node, Edge, NodeType

# Define agents
researcher = Agent(agent_name="Researcher", system_prompt="Research the given topic and produce key findings.", model_name="gpt-5.4")
writer     = Agent(agent_name="Writer",     system_prompt="Write a clear article from the research provided.", model_name="gpt-5.4")
reviewer   = Agent(agent_name="Reviewer",   system_prompt="Review the article for accuracy and clarity.",      model_name="gpt-5.4")
publisher  = Agent(agent_name="Publisher",  system_prompt="Format the final reviewed article for publication.", model_name="gpt-5.4")

# Build the graph: Researcher -> Writer -> Reviewer -> Publisher
workflow = GraphWorkflow()
workflow.add_node(Node(id="researcher", type=NodeType.AGENT, agent=researcher))
workflow.add_node(Node(id="writer",     type=NodeType.AGENT, agent=writer))
workflow.add_node(Node(id="reviewer",   type=NodeType.AGENT, agent=reviewer))
workflow.add_node(Node(id="publisher",  type=NodeType.AGENT, agent=publisher))

workflow.add_edge(Edge(source="researcher", target="writer"))
workflow.add_edge(Edge(source="writer",     target="reviewer"))
workflow.add_edge(Edge(source="reviewer",   target="publisher"))

workflow.set_entry_points(["researcher"])
workflow.set_end_points(["publisher"])

# Run the graph
results = workflow.run("Produce a short article on the rise of small language models.")
print(results)
```

`GraphWorkflow` excels at:
- **Complex Dependencies**: Express any DAG, including fan-out, fan-in, and diamond patterns
- **Automatic Parallelism**: Independent branches execute concurrently without extra configuration
- **Per-node Observability**: Hook into node completion via callbacks for streaming and progress tracking

----

### SwarmRouter: The Universal Swarm Orchestrator

The `SwarmRouter` simplifies building complex workflows by providing a single interface to run any type of swarm. Instead of importing and managing different swarm classes, you can dynamically select the one you need just by changing the `swarm_type` parameter. [Read the full documentation](https://docs.swarms.world/api/swarm-router)

This makes your code cleaner and more flexible, allowing you to switch between different multi-agent strategies with ease. Here's a complete example that shows how to define agents and then use `SwarmRouter` to execute the same task using different collaborative strategies.

```python
from swarms import Agent, SwarmRouter, SwarmType

# Define a few generic agents
writer = Agent(agent_name="Writer", system_prompt="You are a creative writer.", model_name="gpt-5.4")
editor = Agent(agent_name="Editor", system_prompt="You are an expert editor for stories.", model_name="gpt-5.4")
reviewer = Agent(agent_name="Reviewer", system_prompt="You are a final reviewer who gives a score.", model_name="gpt-5.4")

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
    model_name="gpt-5.4"
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

-------

### AutoSwarmBuilder: Autonomous Agent Generation

The `AutoSwarmBuilder` automatically generates specialized agents and their workflows based on your task description. Simply describe what you need, and it will create a complete multi-agent system with detailed prompts and optimal agent configurations. [Learn more about AutoSwarmBuilder](https://docs.swarms.world/api/auto-swarm-builder)

```python
from swarms import AutoSwarmBuilder
import json

# Initialize the AutoSwarmBuilder
swarm = AutoSwarmBuilder(
    name="My Swarm",
    description="A swarm of agents",
    verbose=True,
    max_loops=1,
    return_agents=True,
    model_name="gpt-5.4",
)

# Let the builder automatically create agents and workflows
result = swarm.run(
    task="Create an accounting team to analyze crypto transactions, "
         "there must be 5 agents in the team with extremely extensive prompts. "
         "Make the prompts extremely detailed and specific and long and comprehensive. "
         "Make sure to include all the details of the task in the prompts."
)

# The result contains the generated agents and their configurations
print(json.dumps(result, indent=4))
```

The `AutoSwarmBuilder` provides:

- **Automatic Agent Generation**: Creates specialized agents based on task requirements
- **Intelligent Prompt Engineering**: Generates comprehensive, detailed prompts for each agent
- **Optimal Workflow Design**: Determines the best agent interactions and workflow structure
- **Production-Ready Configurations**: Returns fully configured agents ready for deployment
- **Flexible Architecture**: Supports various swarm types and agent specializations

This feature is perfect for rapid prototyping, complex task decomposition, and creating specialized agent teams without manual configuration.

-------

### MixtureOfAgents (MoA)

The `MixtureOfAgents` architecture processes tasks by feeding them to multiple "expert" agents in parallel. Their diverse outputs are then synthesized by an aggregator agent to produce a final, high-quality result. [Learn more here](https://docs.swarms.world/examples/mixture-of-agents-example)

```python
from swarms import Agent, MixtureOfAgents

# Define expert agents
financial_analyst = Agent(agent_name="FinancialAnalyst", system_prompt="Analyze financial data.", model_name="gpt-5.4")
market_analyst = Agent(agent_name="MarketAnalyst", system_prompt="Analyze market trends.", model_name="gpt-5.4")
risk_analyst = Agent(agent_name="RiskAnalyst", system_prompt="Analyze investment risks.", model_name="gpt-5.4")

# Define the aggregator agent
aggregator = Agent(
    agent_name="InvestmentAdvisor",
    system_prompt="Synthesize the financial, market, and risk analyses to provide a final investment recommendation.",
    model_name="gpt-5.4"
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

----

### GroupChat

`GroupChat` creates a conversational environment where multiple agents can interact, discuss, and collaboratively solve a problem. You can define the speaking order or let it be determined dynamically. This architecture is ideal for tasks that benefit from debate and multi-perspective reasoning, such as contract negotiation, brainstorming, or complex decision-making.

```python
from swarms import Agent, GroupChat

# Define agents for a debate
tech_optimist = Agent(agent_name="TechOptimist", system_prompt="Argue for the benefits of AI in society.", model_name="gpt-5.4")
tech_critic = Agent(agent_name="TechCritic", system_prompt="Argue against the unchecked advancement of AI.", model_name="gpt-5.4")

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

----

### HierarchicalSwarm

`HierarchicalSwarm` implements a director-worker pattern where a central director agent creates comprehensive plans and distributes specific tasks to specialized worker agents. The director evaluates results and can issue new orders in feedback loops, making it ideal for complex project management and team coordination scenarios.

```python
from swarms import Agent, HierarchicalSwarm

# Define specialized worker agents
content_strategist = Agent(
    agent_name="Content-Strategist",
    system_prompt="You are a senior content strategist. Develop comprehensive content strategies, editorial calendars, and content roadmaps.",
    model_name="gpt-5.4"
)

creative_director = Agent(
    agent_name="Creative-Director", 
    system_prompt="You are a creative director. Develop compelling advertising concepts, visual directions, and campaign creativity.",
    model_name="gpt-5.4"
)

seo_specialist = Agent(
    agent_name="SEO-Specialist",
    system_prompt="You are an SEO expert. Conduct keyword research, optimize content, and develop organic growth strategies.",
    model_name="gpt-5.4"
)

brand_strategist = Agent(
    agent_name="Brand-Strategist",
    system_prompt="You are a brand strategist. Develop brand positioning, identity systems, and market differentiation strategies.",
    model_name="gpt-5.4"
)

# Create the hierarchical swarm with a director
marketing_swarm = HierarchicalSwarm(
    name="Marketing-Team-Swarm",
    description="A comprehensive marketing team with specialized agents coordinated by a director",
    agents=[content_strategist, creative_director, seo_specialist, brand_strategist],
    max_loops=2,  # Allow for feedback and refinement
    verbose=True
)

# Run the swarm on a complex marketing challenge
result = marketing_swarm.run(
    "Develop a comprehensive marketing strategy for a new SaaS product launch. "
    "The product is a project management tool targeting small to medium businesses. "
    "Coordinate the team to create content strategy, creative campaigns, SEO optimization, "
    "and brand positioning that work together cohesively."
)

print(result)
```

The `HierarchicalSwarm` excels at:
- **Complex Project Management**: Breaking down large tasks into specialized subtasks
- **Team Coordination**: Ensuring all agents work toward unified goals
- **Quality Control**: Director provides feedback and refinement loops
- **Scalable Workflows**: Easy to add new specialized agents as needed

---

### HeavySwarm

`HeavySwarm` implements a sophisticated 5-phase workflow inspired by X.AI's Grok heavy implementation. It uses specialized agents (Research, Analysis, Alternatives, Verification) to provide comprehensive task analysis through intelligent question generation, parallel execution, and synthesis. This architecture excels at complex research and analysis tasks requiring thorough investigation and multiple perspectives.

```python
from swarms import HeavySwarm

# Pip install swarms-tools
from swarms_tools import exa_search

swarm = HeavySwarm(
    name="Gold ETF Research Team",
    description="A team of agents that research the best gold ETFs",
    worker_model_name="claude-sonnet-4-20250514",
    show_dashboard=True,
    question_agent_model_name="gpt-4.1",
    loops_per_agent=1,
    agent_prints_on=False,
    worker_tools=[exa_search],
    random_loops_per_agent=True,
)

prompt = (
    "Find the best 3 gold ETFs. For each ETF, provide the ticker symbol, "
    "full name, current price, expense ratio, assets under management, and "
    "a brief explanation of why it is considered among the best. Present the information "
    "in a clear, structured format suitable for investors. Scrape the data from the web. "
)

out = swarm.run(prompt)
print(out)

```

The `HeavySwarm` provides:

- **5-Phase Analysis**: Question generation, research, analysis, alternatives, and verification

- **Specialized Agents**: Each phase uses purpose-built agents for optimal results

- **Comprehensive Coverage**: Multiple perspectives and thorough investigation

- **Real-time Dashboard**: Optional visualization of the analysis process

- **Structured Output**: Well-organized and actionable results

This architecture is perfect for financial analysis, strategic planning, research reports, and any task requiring deep, multi-faceted analysis. [Learn more about HeavySwarm](https://docs.swarms.world/api/heavy-swarm)

---

### Social Algorithms

**Social Algorithms** provide a flexible framework for defining custom communication patterns between agents. You can upload any arbitrary social algorithm as a callable that defines the sequence of communication, enabling agents to talk to each other in sophisticated ways. [Learn more about Social Algorithms](https://docs.swarms.world/api/social-algorithms)

```python
from swarms import Agent, SocialAlgorithms

# Define a custom social algorithm
def research_analysis_synthesis_algorithm(agents, task, **kwargs):
    # Agent 1 researches the topic
    research_result = agents[0].run(f"Research: {task}")
    
    # Agent 2 analyzes the research
    analysis = agents[1].run(f"Analyze this research: {research_result}")
    
    # Agent 3 synthesizes the findings
    synthesis = agents[2].run(f"Synthesize: {research_result} + {analysis}")
    
    return {
        "research": research_result,
        "analysis": analysis,
        "synthesis": synthesis
    }

# Create agents
researcher = Agent(
  agent_name="Researcher",
  agent_description="Expert in comprehensive research and information gathering.",
  model_name="gpt-4.1"
)
analyst = Agent(
  agent_name="Analyst",
  agent_description="Specialist in analyzing and interpreting data.",
  model_name="gpt-4.1"
)
synthesizer = Agent(
  agent_name="Synthesizer",
  agent_description="Focused on synthesizing and integrating research insights.",
  model_name="gpt-4.1"
)

# Create social algorithm
social_alg = SocialAlgorithms(
    name="Research-Analysis-Synthesis",
    agents=[researcher, analyst, synthesizer],
    social_algorithm=research_analysis_synthesis_algorithm,
    verbose=True
)

# Run the algorithm
result = social_alg.run("The impact of AI on healthcare")
print(result.final_outputs)
```

Perfect for implementing complex multi-agent workflows, collaborative problem-solving, and custom communication protocols.

---

### Agent Orchestration Protocol (AOP)

The **Agent Orchestration Protocol (AOP)** is a powerful framework for deploying and managing agents as distributed services. AOP enables agents to be discovered, managed, and executed through a standardized protocol, making it perfect for building scalable multi-agent systems. [Learn more about AOP](https://docs.swarms.world/api/aop)

```python
from swarms import Agent, AOP

# Create specialized agents
research_agent = Agent(
    agent_name="Research-Agent",
    agent_description="Expert in research and data collection",
    model_name="anthropic/claude-sonnet-4-5",
    max_loops=1,
    tags=["research", "data-collection", "analysis"],
    capabilities=["web-search", "data-gathering", "report-generation"],
    role="researcher"
)

analysis_agent = Agent(
    agent_name="Analysis-Agent", 
    agent_description="Expert in data analysis and insights",
    model_name="anthropic/claude-sonnet-4-5",
    max_loops=1,
    tags=["analysis", "data-processing", "insights"],
    capabilities=["statistical-analysis", "pattern-recognition", "visualization"],
    role="analyst"
)

# Create AOP server
deployer = AOP(
    server_name="ResearchCluster",
    port=8000,
    verbose=True
)

# Add agents to the server
deployer.add_agent(
    agent=research_agent,
    tool_name="research_tool",
    tool_description="Research and data collection tool",
    timeout=30,
    max_retries=3
)

deployer.add_agent(
    agent=analysis_agent,
    tool_name="analysis_tool", 
    tool_description="Data analysis and insights tool",
    timeout=30,
    max_retries=3
)

# List all registered agents
print("Registered agents:", deployer.list_agents())

# Start the AOP server
deployer.run()
```

Perfect for deploying large scale multi-agent systems. [Read the complete AOP documentation](https://docs.swarms.world/api/aop)

---

## Documentation

The full documentation lives at **[docs.swarms.world](https://docs.swarms.world)**. Below are the resources most useful when building with Swarms — both for humans and for AI coding assistants.

| Resource | Link | What it's for |
|---|---|---|
| **Main documentation** | [docs.swarms.world](https://docs.swarms.world) | Guides, API reference, tutorials |
| **`llms.txt` (LLM-ingestible docs)** | [docs.swarms.world/llms.txt](https://docs.swarms.world/llms.txt) | A single, machine-readable index of the entire documentation, formatted for LLMs and AI IDEs (Cursor, Claude Code, etc.) to consume in one fetch |
| **MCP integration guide** | [docs.swarms.world/mcp](https://docs.swarms.world/mcp) | How to connect a Swarms `Agent` to any [Model Context Protocol](https://modelcontextprotocol.io) server, auto-discover its tools, and call them from a swarm |
| **API reference** | [docs.swarms.world/api](https://docs.swarms.world/api) | Per-class reference for `Agent`, `SequentialWorkflow`, `ConcurrentWorkflow`, `AgentRearrange`, `GraphWorkflow`, `SwarmRouter`, and every multi-agent architecture |
| **Environment setup** | [docs.swarms.world/environment-setup](https://docs.swarms.world/environment-setup) | API keys, model providers, and configuration options |

> **Tip for AI coding assistants:** point your tool (Claude Code, Cursor, Windsurf, Continue, etc.) at `https://docs.swarms.world/llms.txt`. It will pull the entire docs index in one shot and write idiomatic Swarms code without per-question lookups.

---

## Using Swarms with AI Coding Assistants

The repo ships with a [`CLAUDE.md`](./CLAUDE.md) at the root — a focused guide that teaches Claude Code, Cursor, and other AI coding assistants how to build with Swarms. It covers the `Agent` primitive, every multi-agent architecture (`SequentialWorkflow`, `ConcurrentWorkflow`, `AgentRearrange`, `GraphWorkflow`, `MixtureOfAgents`, `HierarchicalSwarm`, `SwarmRouter`, and more), tools, streaming, memory, MCP integration, and the patterns to reach for in each situation.

Drop `CLAUDE.md` (or symlink it as `AGENTS.md` / `.cursorrules`) into any project that depends on `swarms` and your assistant will write idiomatic Swarms code on the first try — no extra prompting required.

---

## Features

Swarms delivers a comprehensive, enterprise-grade multi-agent infrastructure platform designed for production-scale deployments and seamless integration with existing systems. [Learn more about the swarms feature set here](https://docs.swarms.world/community/features)

| Category | Features | Benefits |
|----------|----------|-----------|
| **Enterprise Architecture** | • Production-Ready Infrastructure<br>• High Availability Systems<br>• Modular Microservices Design<br>• Comprehensive Observability<br>• Backwards Compatibility | • 99.9%+ Uptime Guarantee<br>• Reduced Operational Overhead<br>• Seamless Legacy Integration<br>• Enhanced System Monitoring<br>• Risk-Free Migration Path |
| **Multi-Agent Orchestration** | • Hierarchical Agent Swarms<br>• Parallel Processing Pipelines<br>• Sequential Workflow Orchestration<br>• Graph-Based Agent Networks<br>• Dynamic Agent Composition<br>• Agent Registry Management | • Complex Business Process Automation<br>• Scalable Task Distribution<br>• Flexible Workflow Adaptation<br>• Optimized Resource Utilization<br>• Centralized Agent Governance<br>• Enterprise-Grade Agent Lifecycle Management |
| **Enterprise Integration** | • Multi-Model Provider Support<br>• Custom Agent Development Framework<br>• Extensive Enterprise Tool Library<br>• Multiple Memory Systems<br>• Backwards Compatibility with LangChain, AutoGen, CrewAI<br>• Standardized API Interfaces | • Vendor-Agnostic Architecture<br>• Custom Solution Development<br>• Extended Functionality Integration<br>• Enhanced Knowledge Management<br>• Seamless Framework Migration<br>• Reduced Integration Complexity |
| **Enterprise Scalability** | • Concurrent Multi-Agent Processing<br>• Intelligent Resource Management<br>• Load Balancing & Auto-Scaling<br>• Horizontal Scaling Capabilities<br>• Performance Optimization<br>• Capacity Planning Tools | • High-Throughput Processing<br>• Cost-Effective Resource Utilization<br>• Elastic Scaling Based On Demand<br>• Linear Performance Scaling<br>• Optimized Response Times<br>• Predictable Growth Planning |
| **Developer Experience** | • Intuitive Enterprise API<br>• Comprehensive Documentation<br>• Active Enterprise Community<br>• CLI & SDK Tools<br>• IDE Integration Support<br>• Code Generation Templates | • Accelerated Development Cycles<br>• Reduced Learning Curve<br>• Expert Community Support<br>• Rapid Deployment Capabilities<br>• Enhanced Developer Productivity<br>• Standardized Development Patterns |


## Supported Protocols & Integrations

Swarms seamlessly integrates with industry-standard protocols and open specifications, unlocking powerful capabilities for tool integration, payment processing, distributed agent orchestration, and model interoperability.

| Protocol | Description | Documentation |
|----------|-------------|---------------|
| **[MCP (Model Context Protocol)](https://docs.swarms.world/integrations/mcp)** | Standardized protocol for AI agents to interact with external tools and services through MCP servers. Enables dynamic tool discovery and execution. | [MCP Integration Guide](https://docs.swarms.world/integrations/mcp) |
| **[X402](https://docs.swarms.world/examples/integrations/x402-payment)** | Cryptocurrency payment protocol for API endpoints. Enables monetization of agents with pay-per-use models. | [X402 Quickstart](https://docs.swarms.world/examples/integrations/x402-payment) |
| **[AOP (Agent Orchestration Protocol)](https://docs.swarms.world/examples/multi-agent/aop-medical)** | Framework for deploying and managing agents as distributed services. Enables agent discovery, management, and execution through standardized protocols. | [AOP Reference](https://docs.swarms.world/api/aop) |
| **[Swarms Marketplace](https://swarms.world)** | Platform for discovering and sharing production-ready prompts, agents, and tools. Enables automatic prompt loading from the marketplace and publishing your own prompts directly from code. | [Marketplace Tutorial](https://docs.swarms.world/integrations/marketplace) |
| **[Open Responses](https://www.openresponses.org/)** | Open-source specification and ecosystem for multi-provider, interoperable LLM interfaces based on the OpenAI Responses API. Provides a unified schema and tooling for calling language models, streaming results, and composing agentic workflows—independent of provider. | [Open Responses Website](https://www.openresponses.org/) |
| **[Agent Skills](https://docs.swarms.world/agents/agent-skills)** | Lightweight, markdown-based format for defining modular, reusable agent capabilities introduced by Anthropic. Enables specialization of agents without modifying code by loading skill definitions from simple SKILL.md files. | [Agent Skills Documentation](https://docs.swarms.world/agents/agent-skills) |


---

## Examples

Explore comprehensive examples and tutorials to learn how to use Swarms effectively.

| Category | Example | Description | Link |
|----------|---------|-------------|------|
| **Basic Examples** | Basic Agent | Simple agent setup and usage | [Basic Agent](https://docs.swarms.world/examples/basic-agent) |
| **Basic Examples** | Agent with Tools | Using agents with various tools | [Agent with Tools](https://docs.swarms.world/examples/agent-with-tools) |
| **Basic Examples** | Agent with Structured Outputs | Working with structured data outputs | [Structured Outputs](https://docs.swarms.world/agents/structured-outputs) |
| **Basic Examples** | Agent with MCP Integration | Model Context Protocol integration | [MCP Integration](https://docs.swarms.world/integrations/mcp) |
| **Basic Examples** | Vision Processing | Agents with image processing capabilities | [Vision Processing](https://docs.swarms.world/examples/vision-agent) |
| **Basic Examples** | Multiple Images | Working with multiple images | [Multiple Images](https://docs.swarms.world/examples/vision-agent) |
| **Basic Examples** | Vision and Tools | Combining vision with tool usage | [Vision and Tools](https://docs.swarms.world/examples/vision-agent) |
| **Basic Examples** | Agent Streaming | Real-time agent output streaming | [Agent Streaming](https://docs.swarms.world/examples/agents/agent-streaming) |
| **Basic Examples** | Agent Output Types | Different output formats and types | [Output Types](https://docs.swarms.world/agents/structured-outputs) |
| **Basic Examples** | Gradio Chat Interface | Building interactive chat interfaces | [Gradio UI](https://docs.swarms.world/examples/basic-agent) |
| **Model Providers** | Model Providers Overview | Complete guide to supported models | [Model Providers](https://docs.swarms.world/integrations/model-providers) |
| **Model Providers** | OpenAI | OpenAI model integration | [OpenAI Examples](https://docs.swarms.world/integrations/model-providers) |
| **Model Providers** | Anthropic | Claude model integration | [Anthropic Examples](https://docs.swarms.world/integrations/model-providers) |
| **Model Providers** | Groq | Groq model integration | [Groq Examples](https://docs.swarms.world/integrations/model-providers) |
| **Model Providers** | Cohere | Cohere model integration | [Cohere Examples](https://docs.swarms.world/integrations/model-providers) |
| **Model Providers** | DeepSeek | DeepSeek model integration | [DeepSeek Examples](https://docs.swarms.world/integrations/model-providers) |
| **Model Providers** | Ollama | Local Ollama model integration | [Ollama Examples](https://docs.swarms.world/integrations/model-providers) |
| **Model Providers** | OpenRouter | OpenRouter model integration | [OpenRouter Examples](https://docs.swarms.world/integrations/model-providers) |
| **Model Providers** | XAI | XAI model integration | [XAI Examples](https://docs.swarms.world/integrations/model-providers) |
| **Model Providers** | Llama4 | Llama4 model integration | [Llama4 Examples](https://docs.swarms.world/integrations/model-providers) |
| **Multi-Agent Architecture** | HierarchicalSwarm | Hierarchical agent orchestration | [HierarchicalSwarm Examples](https://docs.swarms.world/examples/hierarchical-swarm-example) |
| **Multi-Agent Architecture** | Hybrid Hierarchical-Cluster Swarm | Advanced hierarchical patterns | [HHCS Examples](https://docs.swarms.world/api/hhcs) |
| **Multi-Agent Architecture** | GroupChat | Multi-agent conversations | [GroupChat Examples](https://docs.swarms.world/examples/group-chat-example) |
| **Multi-Agent Architecture** | Sequential Workflow | Step-by-step agent workflows | [Sequential Examples](https://docs.swarms.world/examples/sequential-workflow-example) |
| **Multi-Agent Architecture** | SwarmRouter | Universal swarm orchestration | [SwarmRouter Examples](https://docs.swarms.world/architectures/swarm-router) |
| **Multi-Agent Architecture** | MultiAgentRouter | Minimal router example | [MultiAgentRouter Examples](https://docs.swarms.world/api/multi-agent-router) |
| **Multi-Agent Architecture** | ConcurrentWorkflow | Parallel agent execution | [Concurrent Examples](https://docs.swarms.world/examples/concurrent-workflow-example) |
| **Multi-Agent Architecture** | Mixture of Agents | Expert agent collaboration | [MoA Examples](https://docs.swarms.world/examples/mixture-of-agents-example) |
| **Multi-Agent Architecture** | Unique Swarms | Specialized swarm patterns | [Unique Swarms](https://docs.swarms.world/architectures/overview) |
| **Multi-Agent Architecture** | Agents as Tools | Using agents as tools in workflows | [Agents as Tools](https://docs.swarms.world/architectures/overview) |
| **Multi-Agent Architecture** | Aggregate Responses | Combining multiple agent outputs | [Aggregate Examples](https://docs.swarms.world/architectures/mixture-of-agents) |
| **Multi-Agent Architecture** | Interactive GroupChat | Real-time agent interactions | [Interactive GroupChat](https://docs.swarms.world/examples/group-chat-example) |
| **Deployment Solutions** | Agent Orchestration Protocol (AOP) | Deploy agents as distributed services with discovery and management | [AOP Reference](https://docs.swarms.world/api/aop) |
| **Applications** | Advanced Research System | Multi-agent research system inspired by Anthropic's research methodology | [AdvancedResearch](https://github.com/The-Swarm-Corporation/AdvancedResearch) |
| **Applications** | Hospital Simulation | Healthcare simulation system using multi-agent architecture | [HospitalSim](https://github.com/The-Swarm-Corporation/HospitalSim) |
| **Applications** | Browser Agents | Web automation with agents | [Browser Agents](https://docs.swarms.world/examples/integrations/browser-use) |
| **Applications** | Medical Analysis | Healthcare applications | [Medical Examples](https://docs.swarms.world/examples/multi-agent/aop-medical) |
| **Applications** | Finance Analysis | Financial applications | [Finance Examples](https://docs.swarms.world/examples/use-cases/financial-analysis) |
| **Cookbook & Templates** | Examples Overview | Complete examples directory | [Examples Index](https://docs.swarms.world/examples/) |
| **Cookbook & Templates** | Cookbook Index | Curated example collection | [Cookbook](https://docs.swarms.world/examples/overviews/cookbook) |
| **Cookbook & Templates** | Paper Implementations | Research paper implementations | [Paper Implementations](https://docs.swarms.world/examples/overviews/paper-implementations) |
| **Cookbook & Templates** | Templates & Applications | Reusable templates | [Templates](https://docs.swarms.world/examples/overviews/templates) |

---

## Contribute to Swarms

Swarms is an open-source, community-driven framework aiming to accelerate a fully autonomous world by providing robust infrastructure for deploying and orchestrating millions of agents. By contributing, you can help advance multi-agent AI, collaborate with passionate peers, shape the agent economy, and enhance your expertise. 

Learn more about how you can make a meaningful impact in our [Contributor's Guide](https://docs.swarms.world/community/contributing).

### How to Contribute

We've made it easy to start contributing. Here's how you can help:

1. **Find an Issue to Tackle:** The best way to begin is by visiting our [**contributing project board**](https://github.com/users/kyegomez/projects/1). Look for issues tagged with `good first issue`—these are specifically selected for new contributors.

2. **Report a Bug or Request a Feature:** Have a new idea or found something that isn't working right? We'd love to hear from you. Please [**file a Bug Report or Feature Request**](https://github.com/kyegomez/swarms/issues) on our GitHub Issues page.

3. **Understand Our Workflow and Standards:** Before submitting your work, please review our complete [**Contribution Guidelines**](https://github.com/kyegomez/swarms/blob/master/CONTRIBUTING.md). To help maintain code quality, we also encourage you to read our guide on [**Code Cleanliness**](https://docs.swarms.world/community/contributing-to-docs).

4. **Join the Discussion:** To participate in roadmap discussions and connect with other developers, join our community on [**Discord**](https://discord.gg/EamjgSaEQf).

### Thank You to Our Contributors

Thank you for contributing to swarms. Your work is extremely appreciated and recognized.

<a href="https://github.com/kyegomez/swarms/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=kyegomez/swarms" />
</a>


## We're Hiring

Swarms is hiring. We're building the infrastructure for a world of autonomous agents, and we're looking for engineers, researchers, and operators who want to ship at the frontier of multi-agent AI.

- **Open roles:** [swarms.ai/hiring](https://swarms.ai/hiring)
- **Get in touch:** email [kye@swarms.world](mailto:kye@swarms.world) to learn more


## Join the Discord

Join thousands of agent builders and AI engineers in the **[Swarms Discord](https://discord.gg/EamjgSaEQf)** for technical support, project showcases, collaboration, and the latest swarms ecosystem updates.

[Join the Swarms Discord →](https://discord.gg/EamjgSaEQf)

-----

## Join the Swarms Community!

Join our community of agent engineers and researchers for technical support, cutting-edge updates, and exclusive access to world-class agent engineering insights!

| Platform | Description | Link |
|----------|-------------|------|
| Documentation | Official documentation and guides | [docs.swarms.world](https://docs.swarms.world) |
| Blog | Latest updates and technical articles | [Medium](https://medium.com/@kyeg) |
| Discord | Live chat and community support | [Join Discord](https://discord.gg/EamjgSaEQf) |
| Twitter | Latest news and announcements | [@swarms_corp](https://twitter.com/swarms_corp) |
| LinkedIn | Professional network and updates | [The Swarm Corporation](https://www.linkedin.com/company/the-swarm-corporation) |
| YouTube | Tutorials and demos | [Swarms Channel](https://www.youtube.com/channel/UC9yXyitkbU_WSy7bd_41SqQ) |
| Events | Join our community events | [Sign up here](https://lu.ma/swarms_calendar) |
| Onboarding Session | Get onboarded with Kye Gomez, creator and lead maintainer of Swarms | [Book Session](https://cal.com/swarms/swarms-onboarding-session) |

------

## Citation

If you use **swarms** in your research, please cite the project by referencing the metadata in [CITATION.cff](./CITATION.cff).

```bibtex
@misc{SWARMS_2022,
  author  = {Kye Gomez and Pliny and Zack Bradshaw and Ilumn and Harshal and the Swarms Community},
  title   = {{Swarms: Production-Grade Multi-Agent Infrastructure Platform}},
  year    = {2022},
  howpublished = {\url{https://github.com/kyegomez/swarms}},
  note    = {Documentation available at \url{https://docs.swarms.world}},
  version = {latest}
```

---

# License

Swarms is licensed under the Apache License 2.0. [Learn more here](./LICENSE)
