<div align="left">
  <a href="https://swarms.world">
    <img src="https://github.com/kyegomez/swarms/blob/master/images/new_logo.png" style="margin: 15px; max-width: 350px" width="70%" alt="Logo">
  </a>
</div>


<p align="left">
  <!-- Main Navigation Links -->
  <a href="https://swarms.ai">Swarms 官网</a>
  <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
  <a href="https://docs.swarms.world">文档</a>
  <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
  <a href="https://swarms.world">Swarms 市场</a>
  <span>&nbsp;&nbsp;•&nbsp;&nbsp;</span>
  <a href="./README.md">English</a>
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

## 概述

>
> Swarms：企业级、可直接投入生产的多智能体编排框架

Swarms 是当今最可靠、最具扩展性、最具适应性的多智能体编排框架。我们提供一整套可直接投入生产的预置多智能体架构，包括顺序、并发和分层系统。此外，Swarms 向后兼容主流智能体框架，并可与 MCP、x402、skills 等协议互操作。


## 安装

### 使用 pip

```bash
$ pip3 install -U swarms
```

### 使用 uv（推荐）

[uv](https://github.com/astral-sh/uv) 是一个用 Rust 编写的高速 Python 包安装器与依赖解析器。

```bash
$ uv pip install swarms
```

### 使用 poetry

```bash
$ poetry add swarms
```

### 从源码安装

```bash
# Clone the repository
$ git clone https://github.com/kyegomez/swarms.git
$ cd swarms
$ pip install -r requirements.txt
```

---

## 环境配置

[点击这里了解更多环境配置信息](https://docs.swarms.world/environment-setup)

```
OPENAI_API_KEY=""
WORKSPACE_DIR="agent_workspace"
ANTHROPIC_API_KEY=""
GROQ_API_KEY=""
```


### 你的第一个智能体

**Agent（智能体）** 是 swarm 的基本构建单元，是一个由 LLM + 工具 + 记忆驱动的自主实体。[点击这里了解更多](https://docs.swarms.world/api/agent)

```python
from swarms import Agent

# Initialize a new agent
agent = Agent(
    model_name="gpt-5.4", # Specify the LLM
    max_loops="auto",              # Set the number of interactions
    interactive=True,         # Enable interactive mode for real-time feedback
    temperature=None,
)

# Run the agent with a task
agent.run("What are the key benefits of using a multi-agent system?")
```

### 使用 `max_loops="auto"` 的自主智能体

设置 `max_loops="auto"` 后，智能体会自行判断任务何时完成：它会持续推理和行动，直到满足停止条件，而不是在固定的迭代次数后停止。对于步骤数无法预先确定的开放式、多步骤任务，推荐使用这种模式。

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

**何时使用 `max_loops="auto"`：**
- 开放式的研究或分析任务
- 需要反复打磨的任务（例如：撰写 → 审阅 → 修改）
- 步骤数取决于中间结果的任何工作流

**何时使用固定的 `max_loops` 值：**
- 对延迟或成本敏感的生产流水线
- 步骤数明确且有限的任务

## MCP 集成

[模型上下文协议（MCP）](https://modelcontextprotocol.io) 让智能体只需指向一个 MCP 服务器 URL，即可轻松访问外部工具和数据，所需工具会自动提供给智能体。通过设置 `mcp_url` 或 `mcp_urls`，智能体即可启用 MCP，并且无需手动配置就能使用一个或多个服务器上的工具。像 [DeepWiki](https://mcp.deepwiki.com/mcp) 这样免费、公开的 MCP 服务器开箱即用，可立即为智能体提供实用工具。

```python
from swarms import Agent

agent = Agent(
    agent_name="MCP-Agent",
    model_name="claude-sonnet-5",
    mcp_url="https://mcp.deepwiki.com/mcp",
    max_loops=1,
    temperature=None,
    max_tokens=16_000,
    reasoning_effort=None,
)

print(
    agent.run(
        "Use your tools to explain what the kyegomez/swarms repository does."
    )
)
```

### 将智能体作为 MCP 服务器提供

反过来也可以。`MCPDeployer` 能把任意智能体或任意 swarm 变成一个 MCP 服务器，供其他智能体和 MCP 宿主调用，并在前面加上一层鉴权。每个目标对应一个工具；传入列表或字典即可在一个服务器上提供多个目标。[查看 MCPDeployer 示例](examples/mcp/mcp_deployer/)

```python
from swarms import Agent, MCPDeployer

researcher = Agent(
    agent_name="Researcher",
    agent_description="Answers research questions with a short summary.",
    model_name="gpt-5.4",
    max_loops=1,
)

# Serves http://127.0.0.1:8000/mcp as the tool "researcher".
MCPDeployer(researcher, api_keys=["sk-local-dev"], port=8000).run()
```

任何其他智能体只需指向该 URL 并附上密钥即可使用它：

```python
from swarms import Agent
from swarms.schemas.mcp_schemas import MCPConnection

client = Agent(
    agent_name="Client",
    model_name="gpt-5.4",
    mcp_url=MCPConnection(url="http://127.0.0.1:8000/mcp", api_key="sk-local-dev"),
    max_loops=2,
)
client.run("Use the researcher tool to summarise the state of solid-state batteries.")
```

鉴权方式可以是静态 API 密钥、你自己编写的读取请求头的 `auth` 可调用对象，或带有必需 scope 的 `mcp` `TokenVerifier`。未配置任何鉴权的服务器会拒绝启动，除非显式传入 `allow_anonymous=True`。支持的传输方式：可流式 HTTP（默认）、SSE，以及面向桌面 MCP 宿主的 stdio。

| 示例 | 展示内容 |
|---|---|
| [single_agent_api_key.py](examples/mcp/mcp_deployer/single_agent_api_key.py) | 一个智能体，使用静态密钥保护 |
| [multiple_agents_one_server.py](examples/mcp/mcp_deployer/multiple_agents_one_server.py) | 两个智能体、一个 `SequentialWorkflow` 和两个函数，各自作为独立工具 |
| [custom_auth_per_tenant.py](examples/mcp/mcp_deployer/custom_auth_per_tenant.py) | 自定义的异步鉴权可调用对象，读取 `x-tenant` 请求头 |
| [token_verifier_with_scopes.py](examples/mcp/mcp_deployer/token_verifier_with_scopes.py) | 带必需 scope 的 `TokenVerifier` |
| [background_server_and_client_agent.py](examples/mcp/mcp_deployer/background_server_and_client_agent.py) | 在同一进程中启动服务、由第二个智能体调用、然后停止 |
| [全部 MCPDeployer 示例](examples/mcp/mcp_deployer/) | 覆盖每种目标类型、鉴权方式和传输方式 |

### 你的第一个 Swarm：多智能体协作

一个 **Swarm** 由多个协同工作的智能体组成。下面这个简单示例创建了一个双智能体工作流，用于研究并撰写一篇博客文章。[了解更多关于 SequentialWorkflow 的信息](https://docs.swarms.world/api/sequential-workflow)

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

## 可用的多智能体架构

`swarms` 提供了多种强大的预置多智能体架构，让你能够以不同方式编排智能体。针对具体问题选择合适的结构，即可构建高效、可靠的生产系统。

| **架构** | **说明** | **适用场景** |
|---|---|---|
| **[SequentialWorkflow](https://docs.swarms.world/api/sequential-workflow)** | 智能体按线性链条执行任务；前一个智能体的输出作为下一个智能体的输入。 | 数据转换流水线、报告生成等按步骤进行的流程。 |
| **[ConcurrentWorkflow](https://docs.swarms.world/api/concurrent-workflow)** | 智能体同时运行任务，以获得最高效率。 | 批处理、并行数据分析等高吞吐量任务。 |
| **[AgentRearrange](https://docs.swarms.world/api/agent-rearrange)** | 动态映射智能体之间的复杂关系（例如 `a -> b, c`）。 | 灵活、自适应的工作流，任务分发和动态路由。 |
| **[GraphWorkflow](https://docs.swarms.world/api/graph-workflow)** | 将智能体编排为有向无环图（DAG）中的节点。 | 依赖关系复杂的项目，例如软件构建。 |
| **[MixtureOfAgents (MoA)](https://docs.swarms.world/api/mixture-of-agents)** | 并行调用多个专家智能体，并综合它们的输出。 | 复杂问题求解，通过协作达到最先进的效果。 |
| **[GroupChat](https://docs.swarms.world/api/group-chat)** | 智能体通过对话界面协作并做出决策。 | 实时协作决策、谈判和头脑风暴。 |
| **[ForestSwarm](https://docs.swarms.world/api/forest-swarm)** | 为给定任务动态选择最合适的智能体或智能体树。 | 任务路由、按专长优化，以及复杂的决策树。 |
| **[HierarchicalSwarm](https://docs.swarms.world/api/hierarchical-swarm)** | 由一个主管（director）制定计划并将任务分发给专门的工作智能体。 | 复杂的项目管理、团队协调，以及带反馈循环的分层决策。 |
| **[HeavySwarm](https://docs.swarms.world/api/heavy-swarm)** | 通过专门的智能体（研究、分析、备选方案、验证）实现五阶段工作流，进行全面的任务分析。 | 复杂的研究与分析任务、金融分析、战略规划和综合报告。 |
| **[SwarmRouter](https://docs.swarms.world/api/swarm-router)** | 通用编排器，提供单一接口以动态选择并运行任意类型的 swarm。 | 简化复杂工作流、在 swarm 策略之间切换，以及统一的多智能体管理。 |

我们提供了 60 多种多智能体结构，[点击这里](/docs/MULTI_AGENT_STRUCTURES.md)了解全部内容。

-----

### SequentialWorkflow

`SequentialWorkflow` 按严格顺序执行任务，形成一条流水线，每个智能体都在前一个智能体的工作基础上继续。`SequentialWorkflow` 非常适合步骤清晰、有先后顺序的流程，能确保带依赖关系的任务得到正确处理。

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

`ConcurrentWorkflow` 同时运行多个智能体，实现任务的并行执行。对于可并行完成的任务，这种架构能大幅缩短执行时间，非常适合多个智能体并发处理相似任务的高吞吐量场景。

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

受 `einsum` 启发，`AgentRearrange` 让你可以用简单的字符串语法定义智能体之间复杂的非线性关系。[了解更多](https://docs.swarms.world/api/agent-rearrange)。这种架构非常适合编排动态工作流，智能体可以并行、串行，或以你选择的任意组合方式工作。

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

`GraphWorkflow` 将智能体编排为有向无环图（DAG）中的节点。每个节点是一个智能体，每条边声明一个依赖关系，因此一个节点只有在所有上游节点完成后才会运行。拓扑排序保证了正确的执行顺序，而相互独立的分支会自动并行运行。

当你的工作流包含扇出/扇入模式、条件依赖，或任何无法用一条直线或一个扁平并行批次表达的结构时，`GraphWorkflow` 是正确的选择。[了解更多关于 GraphWorkflow 的信息](https://docs.swarms.world/api/graph-workflow)

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

`GraphWorkflow` 的优势：
- **复杂依赖**：可表达任意 DAG，包括扇出、扇入和菱形模式
- **自动并行**：相互独立的分支无需额外配置即可并发执行
- **节点级可观测性**：通过回调钩住节点完成事件，用于流式输出和进度跟踪

----

### SwarmRouter：通用 Swarm 编排器

`SwarmRouter` 提供单一接口来运行任意类型的 swarm，从而简化复杂工作流的构建。你不必导入和管理不同的 swarm 类，只需修改 `swarm_type` 参数即可动态选择所需的类型。[阅读完整文档](https://docs.swarms.world/api/swarm-router)

这让你的代码更简洁、更灵活，可以轻松在不同的多智能体策略之间切换。下面是一个完整示例，展示了如何定义智能体，然后使用 `SwarmRouter` 以不同的协作策略执行同一个任务。

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


`SwarmRouter` 是简化多智能体编排的强大工具。它以一致而灵活的方式部署不同的协作策略，让你用更少的代码构建更复杂的应用。

-------

### AutoSwarmBuilder：自动生成智能体

`AutoSwarmBuilder` 会根据你的任务描述自动生成专门的智能体及其工作流。只需描述你的需求，它就会创建一个完整的多智能体系统，包含详细的提示词和最优的智能体配置。[了解更多关于 AutoSwarmBuilder 的信息](https://docs.swarms.world/api/auto-swarm-builder)

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

`AutoSwarmBuilder` 提供：

- **自动生成智能体**：根据任务需求创建专门的智能体
- **智能提示词工程**：为每个智能体生成全面、详细的提示词
- **最优工作流设计**：确定最佳的智能体交互方式和工作流结构
- **可直接投入生产的配置**：返回配置完整、可随时部署的智能体
- **灵活的架构**：支持多种 swarm 类型和智能体专长

这一功能非常适合快速原型开发、复杂任务分解，以及无需手动配置即可创建专门的智能体团队。

-------

### MixtureOfAgents (MoA)

`MixtureOfAgents` 架构将任务并行交给多个"专家"智能体处理，然后由一个聚合智能体综合它们各不相同的输出，得到最终的高质量结果。[点击这里了解更多](https://docs.swarms.world/examples/mixture-of-agents-example)

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

`GroupChat` 是一个异步、自选择的群聊。所有智能体并行监听；对于每条广播消息，其他每个智能体都会执行一次强制的 `respond(score, message)` 函数调用来决定是否发言，得分高于 `threshold` 的回复会被广播。当已发布的消息数达到 `max_loops`，或在 `idle_timeout` 秒内没有新消息时，群聊结束。没有固定的发言顺序：多个智能体可以同时对同一条消息作出反应，而选择沉默的智能体则保持沉默。

```python
from swarms import Agent, GroupChat, RESPOND_TOOL

# Every agent MUST carry RESPOND_TOOL so the chat can ask it whether to speak.
tech_optimist = Agent(
    agent_name="TechOptimist",
    system_prompt="Argue for the benefits of AI in society.",
    model_name="gpt-5.4",
    max_loops=1,
    persistent_memory=False,
    tools_list_dictionary=[RESPOND_TOOL],
)
tech_critic = Agent(
    agent_name="TechCritic",
    system_prompt="Argue against the unchecked advancement of AI.",
    model_name="gpt-5.4",
    max_loops=1,
    persistent_memory=False,
    tools_list_dictionary=[RESPOND_TOOL],
)

chat = GroupChat(
    agents=[tech_optimist, tech_critic],
    max_loops=10,       # hard cap on total messages posted
    threshold=0.5,      # min decision score (0..1) to publish a reply
    idle_timeout=8.0,   # seconds of silence before stopping
)

result = chat.run("Let's discuss the societal impact of artificial intelligence.")
print(result)
```

----

### HierarchicalSwarm

`HierarchicalSwarm` 实现了主管-工作者（director-worker）模式：一个中心主管智能体制定全面的计划，并将具体任务分发给专门的工作智能体。主管会评估结果，并可在反馈循环中下达新的指令，因此非常适合复杂的项目管理和团队协调场景。

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

`HierarchicalSwarm` 的优势：
- **复杂的项目管理**：将大任务拆解为专门的子任务
- **团队协调**：确保所有智能体朝着统一目标努力
- **质量控制**：主管提供反馈和迭代打磨的循环
- **可扩展的工作流**：可按需轻松加入新的专门智能体

---

### HeavySwarm

`HeavySwarm` 实现了一个精巧的五阶段工作流，灵感来自 X.AI 的 Grok heavy 实现。它使用专门的智能体（研究、分析、备选方案、验证），通过智能的问题生成、并行执行和综合，提供全面的任务分析。这种架构擅长需要深入调查和多角度视角的复杂研究与分析任务。

```python
from swarms import HeavySwarm

# Pip install swarms-tools
from swarms_tools import exa_search

swarm = HeavySwarm(
    name="Gold ETF Research Team",
    description="A team of agents that research the best gold ETFs",
    worker_model_name="claude-sonnet-4-20250514",
    show_dashboard=True,
    question_agent_model_name="gpt-5.4",
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

`HeavySwarm` 提供：

- **五阶段分析**：问题生成、研究、分析、备选方案和验证

- **专门的智能体**：每个阶段使用专门构建的智能体，以获得最佳结果

- **全面覆盖**：多角度视角与深入调查

- **实时仪表盘**：可选的分析过程可视化

- **结构化输出**：条理清晰、可直接采取行动的结果

这种架构非常适合金融分析、战略规划、研究报告，以及任何需要深入、多维度分析的任务。[了解更多关于 HeavySwarm 的信息](https://docs.swarms.world/api/heavy-swarm)

---

### 社交算法（Social Algorithms）

**社交算法** 提供了一个灵活的框架，用于定义智能体之间的自定义通信模式。你可以将任意社交算法作为可调用对象上传，由它定义通信顺序，让智能体以精巧的方式相互交流。[了解更多关于社交算法的信息](https://docs.swarms.world/api/social-algorithms)

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
  model_name="gpt-5.4"
)
analyst = Agent(
  agent_name="Analyst",
  agent_description="Specialist in analyzing and interpreting data.",
  model_name="gpt-5.4"
)
synthesizer = Agent(
  agent_name="Synthesizer",
  agent_description="Focused on synthesizing and integrating research insights.",
  model_name="gpt-5.4"
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

非常适合实现复杂的多智能体工作流、协作式问题求解和自定义通信协议。

---

## 文档

完整文档位于 **[docs.swarms.world](https://docs.swarms.world)**。下面是使用 Swarms 进行开发时最有用的资源，既适合人类阅读，也适合 AI 编程助手使用。

| 资源 | 链接 | 用途 |
|---|---|---|
| **主文档** | [docs.swarms.world](https://docs.swarms.world) | 指南、API 参考、教程 |
| **`llms.txt`（可供 LLM 摄取的文档）** | [docs.swarms.world/llms.txt](https://docs.swarms.world/llms.txt) | 整套文档的单一机器可读索引，专为 LLM 和 AI IDE（Cursor、Claude Code 等）一次性获取而格式化 |
| **MCP 集成指南** | [docs.swarms.world/mcp](https://docs.swarms.world/mcp) | 如何将 Swarms `Agent` 连接到任意 [模型上下文协议](https://modelcontextprotocol.io) 服务器、自动发现其工具，并在 swarm 中调用 |
| **API 参考** | [docs.swarms.world/api](https://docs.swarms.world/api) | `Agent`、`SequentialWorkflow`、`ConcurrentWorkflow`、`AgentRearrange`、`GraphWorkflow`、`SwarmRouter` 以及每一种多智能体架构的逐类参考 |
| **环境设置** | [docs.swarms.world/environment-setup](https://docs.swarms.world/environment-setup) | API 密钥、模型提供商和配置选项 |

> **给 AI 编程助手的提示：** 将你的工具（Claude Code、Cursor、Windsurf、Continue 等）指向 `https://docs.swarms.world/llms.txt`。它会一次性拉取整个文档索引，无需逐个问题查询即可写出地道的 Swarms 代码。

---

## 在 AI 编程助手中使用 Swarms

本仓库根目录附带了一份 [`CLAUDE.md`](./CLAUDE.md)，这是一份精炼的指南，教 Claude Code、Cursor 和其他 AI 编程助手如何使用 Swarms 进行开发。它涵盖了 `Agent` 原语、每一种多智能体架构（`SequentialWorkflow`、`ConcurrentWorkflow`、`AgentRearrange`、`GraphWorkflow`、`MixtureOfAgents`、`HierarchicalSwarm`、`SwarmRouter` 等）、工具、流式输出、记忆、MCP 集成，以及各种场景下应采用的模式。

把 `CLAUDE.md` 放进任何依赖 `swarms` 的项目（或将其软链接为 `AGENTS.md` / `.cursorrules`），你的助手就能一次写出地道的 Swarms 代码，无需额外提示。

---

## 功能特性

Swarms 提供了一个全面的企业级多智能体基础设施平台，专为生产规模部署和与现有系统的无缝集成而设计。[点击这里了解更多 swarms 功能](https://docs.swarms.world/community/features)

| 类别 | 功能 | 收益 |
|----------|----------|-----------|
| **企业级架构** | • 可直接投入生产的基础设施<br>• 高可用系统<br>• 模块化微服务设计<br>• 全面的可观测性<br>• 向后兼容 | • 99.9%+ 的正常运行时间保证<br>• 降低运维开销<br>• 无缝集成遗留系统<br>• 增强的系统监控<br>• 零风险迁移路径 |
| **多智能体编排** | • 分层智能体 swarm<br>• 并行处理流水线<br>• 顺序工作流编排<br>• 基于图的智能体网络<br>• 动态智能体组合<br>• 智能体注册表管理 | • 复杂业务流程自动化<br>• 可扩展的任务分发<br>• 灵活的工作流适配<br>• 优化的资源利用<br>• 集中化的智能体治理<br>• 企业级智能体生命周期管理 |
| **企业级集成** | • 多模型提供商支持<br>• 自定义智能体开发框架<br>• 丰富的企业工具库<br>• 多种记忆系统<br>• 向后兼容 LangChain、AutoGen、CrewAI<br>• 标准化的 API 接口 | • 与供应商无关的架构<br>• 定制化解决方案开发<br>• 扩展功能集成<br>• 增强的知识管理<br>• 无缝的框架迁移<br>• 降低集成复杂度 |
| **企业级可扩展性** | • 并发多智能体处理<br>• 智能资源管理<br>• 负载均衡与自动扩缩容<br>• 水平扩展能力<br>• 性能优化<br>• 容量规划工具 | • 高吞吐量处理<br>• 高性价比的资源利用<br>• 按需弹性伸缩<br>• 线性的性能扩展<br>• 优化的响应时间<br>• 可预测的增长规划 |
| **开发者体验** | • 直观的企业级 API<br>• 全面的文档<br>• 活跃的企业社区<br>• CLI 与 SDK 工具<br>• IDE 集成支持<br>• 代码生成模板 | • 加速开发周期<br>• 降低学习曲线<br>• 专家社区支持<br>• 快速部署能力<br>• 提升开发者生产力<br>• 标准化的开发模式 |


## 支持的协议与集成

Swarms 与行业标准协议和开放规范无缝集成，为工具集成、支付处理、分布式智能体编排和模型互操作提供强大能力。

| 协议 | 说明 | 文档 |
|----------|-------------|---------------|
| **[MCP（模型上下文协议）](https://docs.swarms.world/integrations/mcp)** | 供 AI 智能体通过 MCP 服务器与外部工具和服务交互的标准化协议。支持动态工具发现与执行。 | [MCP 集成指南](https://docs.swarms.world/integrations/mcp) |
| **[X402](https://docs.swarms.world/examples/integrations/x402-payment)** | 面向 API 端点的加密货币支付协议。支持以按次付费模式将智能体商业化。 | [X402 快速入门](https://docs.swarms.world/examples/integrations/x402-payment) |
| **[Swarms 市场](https://swarms.world)** | 用于发现和分享可直接投入生产的提示词、智能体和工具的平台。支持从市场自动加载提示词，也支持直接从代码发布你自己的提示词。 | [市场教程](https://docs.swarms.world/integrations/marketplace) |
| **[Open Responses](https://www.openresponses.org/)** | 基于 OpenAI Responses API 的多提供商、可互操作 LLM 接口的开源规范与生态。提供统一的 schema 和工具，用于调用语言模型、流式返回结果和组合智能体工作流，与具体提供商无关。 | [Open Responses 官网](https://www.openresponses.org/) |
| **[Agent Skills](https://docs.swarms.world/agents/agent-skills)** | 由 Anthropic 提出的轻量级、基于 markdown 的格式，用于定义模块化、可复用的智能体能力。通过从简单的 SKILL.md 文件加载技能定义，无需修改代码即可让智能体专业化。 | [Agent Skills 文档](https://docs.swarms.world/agents/agent-skills) |


---

## 示例

[点击这里](examples/README.md)浏览全面的示例和教程，学习如何高效使用 Swarms。

---

## 为 Swarms 做贡献

Swarms 是一个开源、社区驱动的框架，旨在通过为部署和编排数百万智能体提供强大的基础设施，加速实现完全自主的世界。通过贡献，你可以推动多智能体 AI 的发展，与志同道合的伙伴协作，塑造智能体经济，并提升自己的专业能力。

在我们的[贡献者指南](https://docs.swarms.world/community/contributing)中了解如何产生有意义的影响。

### 如何贡献

我们让参与贡献变得很简单。你可以这样开始：

1. **找一个要解决的 Issue：** 最好的起点是访问我们的[**贡献项目看板**](https://github.com/users/kyegomez/projects/1)。留意带有 `good first issue` 标签的 issue，它们是专门为新贡献者挑选的。

2. **报告 Bug 或提出功能需求：** 有新想法，或发现了不正常的地方？我们很乐意听取你的意见。请在 GitHub Issues 页面[**提交 Bug 报告或功能需求**](https://github.com/kyegomez/swarms/issues)。

3. **了解我们的工作流和规范：** 在提交你的工作之前，请阅读完整的[**贡献指南**](https://github.com/kyegomez/swarms/blob/master/CONTRIBUTING.md)。为了保持代码质量，我们也建议你阅读[**代码整洁**](https://docs.swarms.world/community/contributing-to-docs)指南。

4. **加入讨论：** 想参与路线图讨论并与其他开发者交流，请加入我们的 [**Discord**](https://discord.gg/EamjgSaEQf) 社区。

5. **所有提交、PR 和 Issue 都必须使用 WARP：** 无论是人还是 AI 智能体，所有贡献者的提交信息、PR 标题和 Issue 标题都必须采用 WARP（Warp Speed Protocol）速记格式：`[TYPE][Function/FileName][Short Description]`，例如 `[FIX][Agent._run][Raise AgentLLMError after retry exhaustion]`。完整规范见 [**WARP Git Message Skill**](https://swarms.world/prompt/32d1e7b4-34da-4035-bc05-d18f8e71a2f1)。未使用 WARP 的 Issue 和 PR 会排在使用了 WARP 的之后处理，因此不使用会有延迟。

### 感谢我们的贡献者

感谢你为 swarms 做出的贡献。你的工作备受重视和认可。

<a href="https://github.com/kyegomez/swarms/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=kyegomez/swarms" />
</a>


## 我们在招聘

Swarms 正在招聘。我们正在为一个自主智能体的世界构建基础设施，寻找希望在多智能体 AI 前沿交付成果的工程师、研究人员和运营人员。

- **在招职位：** [swarms.ai/hiring](https://swarms.ai/hiring)
- **联系我们：** 发送邮件至 [kye@swarms.world](mailto:kye@swarms.world) 了解更多


## 加入 Discord

加入 **[Swarms Discord](https://discord.gg/EamjgSaEQf)**，与数千名智能体开发者和 AI 工程师一起获取技术支持、展示项目、开展协作，并了解 swarms 生态的最新动态。

[加入 Swarms Discord →](https://discord.gg/EamjgSaEQf)

-----

## 加入 Swarms 社区！

加入我们的智能体工程师和研究者社区，获取技术支持、前沿动态，以及世界级智能体工程洞见的独家访问权限！

| 平台 | 说明 | 链接 |
|----------|-------------|------|
| 文档 | 官方文档与指南 | [docs.swarms.world](https://docs.swarms.world) |
| 博客 | 最新动态与技术文章 | [Medium](https://medium.com/@kyeg) |
| Discord | 实时聊天与社区支持 | [加入 Discord](https://discord.gg/EamjgSaEQf) |
| Twitter | 最新消息与公告 | [@swarms_corp](https://twitter.com/swarms_corp) |
| LinkedIn | 职业网络与动态 | [The Swarm Corporation](https://www.linkedin.com/company/the-swarm-corporation) |
| YouTube | 教程与演示 | [Swarms 频道](https://www.youtube.com/channel/UC9yXyitkbU_WSy7bd_41SqQ) |
| 活动 | 参加我们的社区活动 | [在此报名](https://lu.ma/swarms_calendar) |
| 入门指导 | 由 Swarms 创始人兼首席维护者 Kye Gomez 带你入门 | [预约指导](https://cal.com/swarms/swarms-onboarding-session) |

------

## 引用

如果你在研究中使用了 **swarms**，请参考 [CITATION.cff](./CITATION.cff) 中的元数据引用本项目。

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

# 许可证

Swarms 基于 Apache License 2.0 许可证发布。[点击这里了解更多](./LICENSE)
