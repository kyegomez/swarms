# HeavySwarm

HeavySwarm is a sophisticated multi-agent orchestration system that decomposes complex tasks into specialized questions and executes them using specialized agents in parallel. The results are then synthesized into a comprehensive response.

HeavySwarm supports three agent architectures:

- **Default mode** — 5 task-phase agents (Research, Analysis, Alternatives, Verification, Synthesis)
- **Grok mode** (`use_grok_agents=True`) — 4 thinking-style agents inspired by the [Grok 4.20 Heavy architecture](https://x.com/elonmusk/status/2034710771075273177) (Captain Swarm, Harper, Benjamin, Lucas)
- **Grok Heavy mode** (`use_grok_heavy=True`) — 16 domain-specialist agents inspired by the [Grok 4.20 Heavy 16-agent system](https://x.com/xai/status/2024121909286494435) (Grok captain + 15 specialists)

## Installation

```bash
pip install swarms
```

## Quick Start

```python
from swarms.structs.heavy_swarm import HeavySwarm

# Initialize HeavySwarm
swarm = HeavySwarm(
    name="Research Team",
    description="Multi-agent analysis system",
    worker_model_name="gpt-5.4",
    show_dashboard=True
)

# Run analysis
result = swarm.run("Analyze the impact of AI on healthcare")
print(result)
```

## Architecture

The HeavySwarm follows a structured workflow:

1. **Task Decomposition**: Complex tasks are broken down into specialized questions
2. **Question Generation**: AI-powered generation of role-specific questions
3. **Parallel Execution**: Specialized agents work concurrently
4. **Result Collection**: Outputs are gathered and validated
5. **Synthesis**: Integration into a comprehensive final response

### Default Mode — Task-Phase Agents

The default mode decomposes work by **task phase** — each agent handles a different stage of analysis.

| Agent | Role/Function |
|-------|---------------|
| **Research Agent** | Comprehensive information gathering and synthesis |
| **Analysis Agent** | Pattern recognition and statistical analysis |
| **Alternatives Agent** | Creative problem-solving and strategic options |
| **Verification Agent** | Validation, feasibility assessment, and quality assurance |
| **Synthesis Agent** | Multi-perspective integration and executive reporting |

**Flow:** LiteLLM question generator → 4 workers in parallel → Synthesis Agent

### Grok Mode — Thinking-Style Agents (`use_grok_agents=True`)

The Grok mode is inspired by the [Grok 4.20 Heavy architecture](https://x.com/elonmusk/status/2034710771075273177) and decomposes work by **thinking style** — each agent applies a fundamentally different cognitive approach.

| Agent | Specialization | Thinking Style |
|-------|---------------|----------------|
| **Captain Swarm** | Orchestration, task decomposition, conflict resolution, final synthesis | Structured, managerial |
| **Harper** | Research and facts, evidence gathering, source verification, data grounding | Empirical, evidence-based |
| **Benjamin** | Logic, math, and code — rigorous reasoning, numerical verification, stress-testing | Rigorous, analytical |
| **Lucas** | Creative and divergent thinking, contrarian analysis, blind-spot detection, bias identification | Divergent, lateral |

**Flow:** Captain Swarm decomposes → Harper, Benjamin, Lucas work in parallel → Captain Swarm mediates conflicts and synthesizes

### Grok Heavy Mode — Domain-Specialist Agents (`use_grok_heavy=True`)

The Grok Heavy mode mirrors the [Grok 4.20 Heavy 16-agent system](https://x.com/xai/status/2024121909286494435) and decomposes work by **domain expertise** — each agent is a specialist in a distinct field.

| Agent | Domain | Specialization |
|-------|--------|----------------|
| **Grok** (Captain) | Orchestration | Leads, coordinates, synthesizes all 15 specialist outputs |
| **Harper** | Creative Writing | Storytelling, narrative framing, communication strategy |
| **Benjamin** | Finance | Data analysis, financial modeling, economic forecasting |
| **Lucas** | Technology | Coding, programming, software architecture, technical builds |
| **Olivia** | Culture | Literature, arts, cultural context, humanistic perspectives |
| **James** | History | Historical analysis, political science, philosophical frameworks |
| **Charlotte** | Mathematics | Statistical analysis, formal logic, mathematical proofs |
| **Henry** | Engineering | Robotics, hardware, innovation, systems engineering |
| **Mia** | Medicine | Biology, health sciences, clinical research, public health |
| **William** | Business | Strategy, entrepreneurship, market analysis, competitive intelligence |
| **Sebastian** | Physics | Astronomy, hard sciences, fundamental research |
| **Jack** | Psychology | Human behavior, cognitive science, decision-making |
| **Owen** | Environment | Sustainability, climate science, global systems |
| **Luna** | Space | Space exploration, futurism, emerging frontiers |
| **Elizabeth** | Ethics | Policy analysis, ethical frameworks, critical thinking |
| **Noah** | Systems | Long-term innovation, systems thinking, civilizational impact |

**Flow:** Grok decomposes task into 15 domain-specific questions → 15 specialists work in parallel → Grok synthesizes with cross-domain convergence analysis

> **Note:** Harper, Benjamin, and Lucas have different roles in Grok Heavy mode compared to Grok mode. In Grok mode they represent thinking styles (facts/logic/creativity); in Grok Heavy mode they represent domain expertise (creative writing/finance/technology).

#### Key Differences Across All Three Modes

| Aspect | Default Mode | Grok Mode | Grok Heavy Mode |
|--------|-------------|-----------|-----------------|
| **Decomposition** | By task phase (research → analyze → verify) | By thinking style (facts vs logic vs creativity) | By domain expertise (15 specialist fields) |
| **Agent count** | 5 (4 workers + 1 synthesizer) | 4 (3 specialists + 1 leader) | 16 (15 specialists + 1 captain) |
| **Synthesis** | Generic integration agent | Captain Swarm mediates conflicts | Grok synthesizes with cross-domain convergence |
| **Contrarian check** | Not present | Lucas challenges assumptions | Elizabeth provides ethical/critical analysis |
| **Question schema** | 4 questions | 3 questions | 15 questions |
| **Best for** | Systematic, phased analysis | Debate-style reasoning | Deep multi-domain analysis |
| **Flag** | (default) | `use_grok_agents=True` | `use_grok_heavy=True` |

## API Reference

### HeavySwarm Class

A sophisticated multi-agent orchestration system for complex task analysis.

#### Constructor

```python
HeavySwarm(
    name: str = "HeavySwarm",
    description: str = "A swarm of agents that can analyze a task and generate specialized questions for each agent role",
    timeout: int = 300,
    aggregation_strategy: str = "synthesis",
    loops_per_agent: int = 1,
    question_agent_model_name: str = "gpt-5.4",
    worker_model_name: str = "gpt-5.4",
    verbose: bool = False,
    max_workers: int = int(os.cpu_count() * 0.9),
    show_dashboard: bool = False,
    agent_prints_on: bool = False,
    output_type: str = "dict-all-except-first",
    worker_tools: Optional[tool_type] = None,
    random_loops_per_agent: bool = False,
    max_loops: int = 1,
    use_grok_agents: bool = False,
    use_grok_heavy: bool = False,
)
```

##### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `name` | `str` | `"HeavySwarm"` | Identifier name for the swarm instance |
| `description` | `str` | `"A swarm of agents that can analyze a task and generate specialized questions for each agent role"` | Description of the swarm's purpose and capabilities |
| `timeout` | `int` | `300` | Maximum execution time per agent in seconds |
| `aggregation_strategy` | `str` | `"synthesis"` | Strategy for result aggregation (currently only 'synthesis' is supported) |
| `loops_per_agent` | `int` | `1` | Number of execution loops each agent should perform |
| `question_agent_model_name` | `str` | `"gpt-5.4"` | Language model for question generation |
| `worker_model_name` | `str` | `"gpt-5.4"` | Language model for specialized worker agents |
| `verbose` | `bool` | `False` | Enable detailed logging and debug output |
| `max_workers` | `int` | `int(os.cpu_count() * 0.9)` | Maximum concurrent workers for parallel execution |
| `show_dashboard` | `bool` | `False` | Enable rich dashboard with progress visualization |
| `agent_prints_on` | `bool` | `False` | Enable individual agent output printing |
| `output_type` | `str` | `"dict-all-except-first"` | Output formatting type for conversation history |
| `worker_tools` | `Optional[tool_type]` | `None` | Tools available to worker agents for enhanced functionality |
| `random_loops_per_agent` | `bool` | `False` | Enable random number of loops per agent (1-10 range) |
| `max_loops` | `int` | `1` | Maximum number of execution loops for iterative refinement |
| `use_grok_agents` | `bool` | `False` | Enable Grok 4-agent mode with Captain Swarm, Harper, Benjamin, and Lucas agents instead of the default Research/Analysis/Alternatives/Verification agents |
| `use_grok_heavy` | `bool` | `False` | Enable Grok Heavy 16-agent mode with Grok captain and 15 domain specialists (Harper, Benjamin, Lucas, Olivia, James, Charlotte, Henry, Mia, William, Sebastian, Jack, Owen, Luna, Elizabeth, Noah). Mutually exclusive with `use_grok_agents` |

##### Raises

- `ValueError`: If `loops_per_agent` is 0 or negative

- `ValueError`: If `worker_model_name` is None

- `ValueError`: If `question_agent_model_name` is None

- `ValueError`: If both `use_grok_agents` and `use_grok_heavy` are True (mutually exclusive)

##### Example

```python
swarm = HeavySwarm(
    name="AdvancedAnalysisSwarm",
    description="Comprehensive multi-agent analysis system",
    timeout=600,
    loops_per_agent=2,
    question_agent_model_name="gpt-5.4",
    worker_model_name="gpt-5.4",
    verbose=True,
    max_workers=8,
    show_dashboard=True,
    agent_prints_on=True,
    output_type="dict-all-except-first",
    worker_tools=None,
    random_loops_per_agent=False
)
```

#### Methods

##### `run(task: str, img: Optional[str] = None) -> str`

Execute the complete HeavySwarm orchestration flow.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `task` | `str` | Yes | - | The main task to analyze and decompose |
| `img` | `Optional[str]` | No | `None` | Image input for visual analysis tasks |

**Returns:**

- `str`: Comprehensive final analysis from synthesis agent

**Example:**

```python
result = swarm.run("Develop a go-to-market strategy for a new SaaS product")
print(result)
```

##### `get_questions_only(task: str) -> Dict[str, str]`

Generate and extract only the specialized questions without metadata or execution.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `task` | `str` | Yes | - | The main task or query to analyze and decompose into specialized questions |

**Returns:**

- `Dict[str, str]`: Clean dictionary containing only the questions.

  **Default mode keys:**
  - `research_question` (str): Question for comprehensive information gathering
  - `analysis_question` (str): Question for pattern analysis and insights
  - `alternatives_question` (str): Question for exploring creative solutions
  - `verification_question` (str): Question for validation and feasibility

  **Grok mode keys** (`use_grok_agents=True`):
  - `harper_question` (str): Question for evidence-based research and fact verification
  - `benjamin_question` (str): Question for logical reasoning and mathematical verification
  - `lucas_question` (str): Question for creative/contrarian analysis and blind-spot detection

  **Grok Heavy mode keys** (`use_grok_heavy=True`):
  - `harper_question` (str): Question for creative writing and storytelling analysis
  - `benjamin_question` (str): Question for data, finance, and economics analysis
  - `lucas_question` (str): Question for coding, programming, and technical analysis
  - `olivia_question` (str): Question for literature, arts, and cultural analysis
  - `james_question` (str): Question for historical, political, and philosophical analysis
  - `charlotte_question` (str): Question for mathematical, statistical, and logical analysis
  - `henry_question` (str): Question for engineering, robotics, and innovation analysis
  - `mia_question` (str): Question for biology, health, and medical analysis
  - `william_question` (str): Question for business strategy and entrepreneurship analysis
  - `sebastian_question` (str): Question for physics, astronomy, and hard sciences analysis
  - `jack_question` (str): Question for psychology and human behavior analysis
  - `owen_question` (str): Question for environment, sustainability, and global systems analysis
  - `luna_question` (str): Question for space exploration and futurism analysis
  - `elizabeth_question` (str): Question for ethics, policy, and critical thinking analysis
  - `noah_question` (str): Question for long-term innovation and systems thinking analysis

  - `error` (str): Error message if question generation fails (on error only)

**Example:**

```python
# Default mode
questions = swarm.get_questions_only("Analyze market trends for EVs")
print(questions['research_question'])
print(questions['analysis_question'])

# Grok mode
grok_swarm = HeavySwarm(use_grok_agents=True, worker_model_name="gpt-5.4", question_agent_model_name="gpt-5.4")
questions = grok_swarm.get_questions_only("Analyze market trends for EVs")
print(questions['harper_question'])
print(questions['benjamin_question'])
print(questions['lucas_question'])

# Grok Heavy mode
heavy_swarm = HeavySwarm(use_grok_heavy=True, worker_model_name="gpt-5.4", question_agent_model_name="gpt-5.4")
questions = heavy_swarm.get_questions_only("Analyze market trends for EVs")
for key, question in questions.items():
    print(f"{key}: {question}")
```

##### `get_questions_as_list(task: str) -> List[str]`

Generate specialized questions and return them as an ordered list.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `task` | `str` | Yes | - | The main task or query to decompose into specialized questions |

**Returns:**

- `List[str]`: Ordered list of specialized questions.

  **Default mode** (4 questions):
  - `[0]` Research question for comprehensive information gathering
  - `[1]` Analysis question for pattern analysis and insights
  - `[2]` Alternatives question for exploring creative solutions
  - `[3]` Verification question for validation and feasibility assessment

  **Grok mode** (3 questions):
  - `[0]` Harper question for evidence-based research and fact verification
  - `[1]` Benjamin question for logical reasoning and mathematical verification
  - `[2]` Lucas question for creative/contrarian analysis and blind-spot detection

  **Grok Heavy mode** (15 questions):
  - `[0]` Harper — creative writing and storytelling
  - `[1]` Benjamin — data, finance, and economics
  - `[2]` Lucas — coding, programming, and technology
  - `[3]` Olivia — literature, arts, and culture
  - `[4]` James — history, politics, and philosophy
  - `[5]` Charlotte — math, statistics, and logic
  - `[6]` Henry — engineering, robotics, and innovation
  - `[7]` Mia — biology, health, and medicine
  - `[8]` William — business strategy and entrepreneurship
  - `[9]` Sebastian — physics, astronomy, and hard sciences
  - `[10]` Jack — psychology and human behavior
  - `[11]` Owen — environment, sustainability, and global systems
  - `[12]` Luna — space exploration and futurism
  - `[13]` Elizabeth — ethics, policy, and critical thinking
  - `[14]` Noah — long-term innovation and systems thinking

  - Single-item list containing error message (on error)

**Example:**

```python
questions = swarm.get_questions_as_list("Optimize supply chain efficiency")
for i, question in enumerate(questions):
    print(f"Agent {i+1}: {question}")
```

##### `show_swarm_info() -> None`

Display comprehensive swarm configuration information in a rich dashboard format.

**Parameters:**

- None

**Returns:**

- `None`: This method only displays output and has no return value.

**Example:**

```python
swarm.show_swarm_info()  # Displays configuration dashboard
```

##### `reliability_check() -> None`

Perform comprehensive reliability and configuration validation checks.

**Parameters:**

- None

**Returns:**

- `None`: This method only performs validation and has no return value.

**Raises:**

| Exception    | Condition                                                                 |
|--------------|---------------------------------------------------------------------------|
| `ValueError` | If `loops_per_agent` is 0 or negative (agents won't execute)              |
| `ValueError` | If `worker_model_name` is None (agents can't be created)                  |
| `ValueError` | If `question_agent_model_name` is None (questions can't be generated)     |
| `ValueError` | If both `use_grok_agents` and `use_grok_heavy` are True (mutually exclusive) |

**Example:**

```python
try:
    swarm.reliability_check()  # Automatically called in __init__
    print("All checks passed!")
except ValueError as e:
    print(f"Configuration error: {e}")
```

##### `create_agents() -> Dict[str, Agent]`

Create and cache specialized agents with detailed role-specific prompts. The agents created depend on the `use_grok_agents` and `use_grok_heavy` settings.

**Parameters:**

- None

**Returns (default mode):**

| Key             | Agent Instance         | Description                                 |
|-----------------|-----------------------|---------------------------------------------|
| `'research'`    | Research Agent        | Research Agent instance                     |
| `'analysis'`    | Analysis Agent        | Analysis Agent instance                     |
| `'alternatives'`| Alternatives Agent    | Alternatives Agent instance                 |
| `'verification'`| Verification Agent    | Verification Agent instance                 |
| `'synthesis'`   | Synthesis Agent       | Synthesis Agent instance                    |

**Returns (Grok mode, `use_grok_agents=True`):**

| Key         | Agent Instance   | Description                                          |
|-------------|-----------------|------------------------------------------------------|
| `'captain'` | Captain Swarm   | Leader and orchestrator — handles decomposition and synthesis |
| `'harper'`  | Harper          | Research and facts specialist                        |
| `'benjamin'`| Benjamin        | Logic, math, and code specialist                     |
| `'lucas'`   | Lucas           | Creative and divergent thinking specialist           |

**Returns (Grok Heavy mode, `use_grok_heavy=True`):**

| Key           | Agent Instance | Description                                         |
|---------------|---------------|-----------------------------------------------------|
| `'captain'`   | Grok          | Captain — leads, coordinates, and synthesizes all specialist outputs |
| `'harper'`    | Harper        | Creative writing and storytelling specialist         |
| `'benjamin'`  | Benjamin      | Data, finance, and economics specialist              |
| `'lucas'`     | Lucas         | Coding, programming, and technical builds specialist |
| `'olivia'`    | Olivia        | Literature, arts, and culture specialist             |
| `'james'`     | James         | History, politics, and philosophy specialist         |
| `'charlotte'` | Charlotte     | Math, statistics, and logic specialist               |
| `'henry'`     | Henry         | Engineering, robotics, and innovation specialist     |
| `'mia'`       | Mia           | Biology, health, and medicine specialist             |
| `'william'`   | William       | Business strategy and entrepreneurship specialist    |
| `'sebastian'` | Sebastian     | Physics, astronomy, and hard sciences specialist     |
| `'jack'`      | Jack          | Psychology and human behavior specialist             |
| `'owen'`      | Owen          | Environment, sustainability, and global systems specialist |
| `'luna'`      | Luna          | Space exploration and futurism specialist            |
| `'elizabeth'` | Elizabeth     | Ethics, policy, and critical thinking specialist     |
| `'noah'`      | Noah          | Long-term innovation and systems thinking specialist |

**Example:**

```python
# Default mode
agents = swarm.create_agents()
research_agent = agents['research']
print(f"Research agent name: {research_agent.agent_name}")

# Grok mode
grok_swarm = HeavySwarm(use_grok_agents=True, worker_model_name="gpt-5.4", question_agent_model_name="gpt-5.4")
agents = grok_swarm.create_agents()
print(f"Captain: {agents['captain'].agent_name}")
print(f"Harper: {agents['harper'].agent_name}")

# Grok Heavy mode
heavy_swarm = HeavySwarm(use_grok_heavy=True, worker_model_name="gpt-5.4", question_agent_model_name="gpt-5.4")
agents = heavy_swarm.create_agents()
print(f"Total agents: {len(agents)}")  # 16
print(f"Captain: {agents['captain'].agent_name}")  # Grok
for key in ['harper', 'benjamin', 'lucas', 'olivia', 'james', 'charlotte',
            'henry', 'mia', 'william', 'sebastian', 'jack', 'owen',
            'luna', 'elizabeth', 'noah']:
    print(f"{key}: {agents[key].agent_name}")
```

## Examples

### Grok Mode — Quick Start

```python
from swarms.structs.heavy_swarm import HeavySwarm

# Enable Grok 4.20 Heavy architecture
swarm = HeavySwarm(
    name="Grok Analysis Team",
    description="Multi-agent analysis with Grok-style agents",
    worker_model_name="gpt-5.4",
    question_agent_model_name="gpt-5.4",
    use_grok_agents=True,
    show_dashboard=True,
)

# Captain Swarm decomposes the task, Harper/Benjamin/Lucas
# work in parallel, then Captain synthesizes with conflict resolution
result = swarm.run("Evaluate whether Tesla should expand into commercial trucking")
print(result)
```

### Grok Mode — Medical Research

```python
from swarms.structs.heavy_swarm import HeavySwarm

swarm = HeavySwarm(
    name="Medical Research Team",
    worker_model_name="gpt-5.4",
    question_agent_model_name="gpt-5.4",
    use_grok_agents=True,
    loops_per_agent=2,
    show_dashboard=True,
)

# Harper gathers clinical evidence, Benjamin verifies
# statistical claims, Lucas challenges treatment assumptions
result = swarm.run(
    "Analyze the latest research on GLP-1 receptor agonists "
    "for treating obesity: efficacy, long-term safety, cost-effectiveness, "
    "and potential off-label applications"
)
print(result)
```

### Grok Mode — Strategic Analysis with Tools

```python
from swarms.structs.heavy_swarm import HeavySwarm

def web_search(query: str) -> str:
    """Search the web for information."""
    # Your search implementation
    ...

swarm = HeavySwarm(
    name="Strategic Intel Team",
    worker_model_name="gpt-5.4",
    question_agent_model_name="gpt-5.4",
    use_grok_agents=True,
    worker_tools=[web_search],
    show_dashboard=True,
    timeout=600,
)

# Harper uses tools for real-time fact-gathering,
# Benjamin stress-tests the strategy mathematically,
# Lucas identifies blind spots and contrarian opportunities
result = swarm.run(
    "Should our company enter the Southeast Asian market in 2026? "
    "Consider regulatory environment, competitive landscape, "
    "logistics costs, and consumer behavior patterns."
)
print(result)
```

### Grok Mode via SwarmRouter

```python
from swarms.structs.swarm_router import SwarmRouter

router = SwarmRouter(
    name="GrokRouter",
    description="Router with Grok Heavy agents",
    swarm_type="HeavySwarm",
    heavy_swarm_worker_model_name="gpt-5.4",
    heavy_swarm_question_agent_model_name="gpt-5.4",
    heavy_swarm_use_grok_agents=True,
)

result = router.run("Analyze the future of autonomous vehicles")
print(result)
```

### Grok Mode — Question Preview

```python
from swarms.structs.heavy_swarm import HeavySwarm

swarm = HeavySwarm(
    worker_model_name="gpt-5.4",
    question_agent_model_name="gpt-5.4",
    use_grok_agents=True,
)

# Preview what Captain Swarm will ask each specialist
questions = swarm.get_questions_only("Should we migrate from PostgreSQL to CockroachDB?")
print(f"Harper (facts): {questions['harper_question']}")
print(f"Benjamin (logic): {questions['benjamin_question']}")
print(f"Lucas (creative): {questions['lucas_question']}")

# Or as a list
question_list = swarm.get_questions_as_list("Should we migrate from PostgreSQL to CockroachDB?")
for i, q in enumerate(question_list):
    print(f"Agent {i+1}: {q}")
```

### Grok Heavy Mode — Quick Start

```python
from swarms import HeavySwarm

# Enable 16-agent Grok Heavy mode
swarm = HeavySwarm(
    name="Grok Heavy Research Team",
    description="16-agent deep multi-domain analysis",
    worker_model_name="grok-4",
    question_agent_model_name="grok-4",
    use_grok_heavy=True,
    show_dashboard=True,
    loops_per_agent=1,
)

# Grok decomposes into 15 domain-specific questions,
# all 15 specialists work in parallel, then Grok synthesizes
result = swarm.run(
    "What are the most transformative technologies that will reshape "
    "civilization over the next 50 years? Analyze from all relevant dimensions."
)
print(result)
```

### Grok Heavy Mode — Comprehensive Policy Analysis

```python
from swarms import HeavySwarm

swarm = HeavySwarm(
    name="Policy Analysis Team",
    worker_model_name="gpt-5.4",
    question_agent_model_name="gpt-5.4",
    use_grok_heavy=True,
    show_dashboard=True,
    loops_per_agent=2,
)

# All 15 specialists contribute their domain expertise:
# - James analyzes historical precedents and political feasibility
# - Charlotte validates statistical claims and economic models
# - Mia assesses public health implications
# - Owen evaluates environmental impact
# - Elizabeth examines ethical and policy dimensions
# - William models business and market effects
# ... and 9 more specialists
result = swarm.run(
    "Should the US implement a universal basic income? Analyze the economic "
    "feasibility, social impact, political viability, and long-term consequences."
)
print(result)
```

### Grok Heavy Mode — Question Preview

```python
from swarms import HeavySwarm

swarm = HeavySwarm(
    worker_model_name="gpt-5.4",
    question_agent_model_name="gpt-5.4",
    use_grok_heavy=True,
)

# Preview the 15 domain-specific questions Grok will generate
questions = swarm.get_questions_only("Analyze the future of space colonization")
for key, question in questions.items():
    print(f"{key}: {question}\n")

# Or as a list
question_list = swarm.get_questions_as_list("Analyze the future of space colonization")
print(f"Total questions: {len(question_list)}")  # 15
for i, q in enumerate(question_list):
    print(f"Specialist {i+1}: {q}")
```

### Basic Usage (Default Mode)

```python
from swarms.structs.heavy_swarm import HeavySwarm

# Initialize HeavySwarm
swarm = HeavySwarm(
    name="Analysis Team",
    description="Multi-agent analysis system",
    worker_model_name="gpt-5.4",
    show_dashboard=True
)

# Run analysis
result = swarm.run("Analyze the impact of AI on healthcare")
print(result)
```

### Financial Research with Tools

```python
from swarms.structs.heavy_swarm import HeavySwarm
from swarms_tools import exa_search

# Initialize with tools
swarm = HeavySwarm(
    name="Gold ETF Research Team",
    description="Financial research with web scraping",
    worker_model_name="gpt-5.4",
    question_agent_model_name="gpt-5.4",
    show_dashboard=True,
    worker_tools=[exa_search],
    timeout=300
)

# Research prompt
prompt = """
Find the best 3 gold ETFs. For each ETF, provide:
- Ticker symbol and full name
- Current price and expense ratio
- Assets under management
- Why it's considered among the best
"""

result = swarm.run(prompt)
print(result)
```

### Advanced Market Analysis

```python
from swarms.structs.heavy_swarm import HeavySwarm

# Advanced configuration
swarm = HeavySwarm(
    name="AdvancedMarketAnalysis",
    description="Deep market analysis with multiple iterations",
    worker_model_name="gpt-5.4",
    question_agent_model_name="gpt-5.4",
    show_dashboard=True,
    loops_per_agent=3,  # Multiple iterations for depth
    timeout=600,
    max_workers=8,
    verbose=True
)

# Complex analysis prompt
complex_prompt = """
Conduct comprehensive analysis of renewable energy sector:
1. Market size and growth projections
2. Technology landscape and adoption rates
3. Investment opportunities and risks
4. Regulatory environment
5. Competitive analysis
"""

result = swarm.run(complex_prompt)
print(result)
```

### Question Generation Only

```python
from swarms.structs.heavy_swarm import HeavySwarm

# Configure for question generation
swarm = HeavySwarm(
    name="QuestionGenerator",
    question_agent_model_name="gpt-5.4",
    worker_model_name="gpt-5.4",
    show_dashboard=False
)

# Generate questions without execution
task = "Develop digital transformation strategy for manufacturing"
questions = swarm.get_questions_only(task)

print("Generated Questions:")
for key, question in questions.items():
    if key != 'error':
        print(f"\n{key.upper()}:")
        print(question)
```

### Batch Processing

```python
from swarms.structs.heavy_swarm import HeavySwarm

# Configure for batch processing
swarm = HeavySwarm(
    name="BatchProcessor",
    worker_model_name="gpt-5.4",
    question_agent_model_name="gpt-5.4",
    show_dashboard=False,  # Disable for batch efficiency
    loops_per_agent=1,
    timeout=300,
    verbose=False
)

# Batch tasks
tasks = [
    "Analyze remote work impact on commercial real estate",
    "Evaluate electric vehicles in taxi industry",
    "Assess IoT cybersecurity risks",
    "Explore blockchain in supply chain"
]

# Process batch
results = {}
for task in tasks:
    try:
        result = swarm.run(task)
        results[task] = result
        print(f"✅ Completed: {task[:50]}...")
    except Exception as e:
        results[task] = f"Error: {str(e)}"
        print(f"❌ Failed: {task[:50]}...")

print(f"\nBatch complete: {len(results)} tasks processed")
```

### Error Handling

```python
from swarms.structs.heavy_swarm import HeavySwarm

# Test configuration validation
try:
    swarm = HeavySwarm(
        name="TestSwarm",
        worker_model_name="gpt-5.4",
        question_agent_model_name="gpt-5.4"
    )
    print("✅ Configuration valid")
except ValueError as e:
    print(f"❌ Configuration error: {e}")

# Test question generation with error handling
swarm = HeavySwarm(
    name="ErrorTest",
    worker_model_name="gpt-5.4",
    question_agent_model_name="gpt-5.4"
)

try:
    questions = swarm.get_questions_only("Test task")
    if 'error' in questions:
        print(f"❌ Question generation failed: {questions['error']}")
    else:
        print("✅ Questions generated successfully")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
```

### Custom Tools Integration

```python
from swarms.structs.heavy_swarm import HeavySwarm
from swarms_tools import exa_search, calculator

# Initialize with custom tools
swarm = HeavySwarm(
    name="CustomToolsSwarm",
    description="HeavySwarm with enhanced tool capabilities",
    worker_model_name="gpt-5.4",
    worker_tools=[exa_search, calculator],
    show_dashboard=True,
    timeout=600
)

# Task that benefits from tools
analysis_prompt = """
Conduct competitive intelligence analysis:
1. Research competitor strategies using web search
2. Calculate market share percentages
3. Analyze financial metrics
4. Provide strategic recommendations
"""

result = swarm.run(analysis_prompt)
print(result)
```

### Enterprise Configuration

```python
from swarms.structs.heavy_swarm import HeavySwarm

# Enterprise-grade configuration
swarm = HeavySwarm(
    name="EnterpriseAnalysis",
    description="High-performance enterprise analysis",
    worker_model_name="gpt-5.4",  # Highest quality
    question_agent_model_name="gpt-5.4",
    show_dashboard=True,
    loops_per_agent=5,  # Maximum depth
    timeout=1800,  # 30 minutes
    max_workers=16,  # Maximum parallelization
    verbose=True
)

# Enterprise-level analysis
enterprise_prompt = """
Conduct enterprise strategic analysis:
- Executive summary and positioning
- Financial analysis and projections
- Operational assessment
- Market intelligence
- Strategic recommendations with implementation roadmap
"""

result = swarm.run(enterprise_prompt)
print(result)
```

### Random Loops for Diverse Perspectives

```python
from swarms.structs.heavy_swarm import HeavySwarm

# Enable random loops for varied analysis depth
swarm = HeavySwarm(
    name="DiverseAnalysis",
    description="Analysis with random iteration depth",
    worker_model_name="gpt-5.4",
    question_agent_model_name="gpt-5.4",
    show_dashboard=True,
    loops_per_agent=1,  # Base loops
    random_loops_per_agent=True,  # Enable randomization (1-10 loops)
    timeout=900
)

# Analysis that benefits from multiple perspectives
prompt = """
Analyze healthcare policy impacts on:
1. Access to care and underserved populations
2. Cost implications and spending trends
3. Quality of care metrics
4. Healthcare workforce dynamics
5. Technology integration
"""

result = swarm.run(prompt)
print(result)
```

## Notes

| Aspect              | Recommendation/Description                                                                                   |
|---------------------|-------------------------------------------------------------------------------------------------------------|
| **Performance**     | Use `max_workers` based on your CPU cores for optimal parallel execution                                     |
| **Cost**            | Higher model versions provide better analysis but increase costs. Grok mode uses 4 agents vs 5, slightly reducing cost per run. Grok Heavy mode uses 16 agents, significantly increasing cost but providing the deepest analysis |
| **Timeouts**        | Complex tasks may require longer `timeout` values                                                           |
| **Tools**           | Integrate domain-specific tools for enhanced analysis capabilities                                           |
| **Dashboard**       | Enable `show_dashboard=True` for visual progress tracking                                                   |
| **Batch Processing**| Disable dashboard and verbose logging for efficient batch operations                                        |
| **When to use Default mode** | Best for systematic, phased analysis where each step builds on the previous one (e.g., research reports, market studies, compliance audits). The dedicated Verification Agent provides stronger validation coverage. |
| **When to use Grok mode** | Best for tasks requiring debate-style analysis where facts, logic, and creative thinking need to be weighed against each other (e.g., strategic decisions, investment analysis, policy evaluation). Captain Swarm's conflict mediation is especially valuable when different perspectives might disagree. |
| **When to use Grok Heavy mode** | Best for complex, multi-domain tasks that benefit from deep specialist expertise across many fields (e.g., civilizational impact analysis, comprehensive policy evaluation, cross-disciplinary research). The 15 domain specialists provide unmatched breadth and depth, with Grok synthesizing cross-domain convergences. |
| **Mutual exclusion** | `use_grok_agents` and `use_grok_heavy` cannot both be `True`. Set only one to enable the desired mode. |
