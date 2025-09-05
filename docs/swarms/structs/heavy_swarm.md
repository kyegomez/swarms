# HeavySwarm

HeavySwarm is a sophisticated multi-agent orchestration system that decomposes complex tasks into specialized questions and executes them using four specialized agents: Research, Analysis, Alternatives, and Verification. The results are then synthesized into a comprehensive response.

Inspired by X.AI's Grok 4 heavy implementation, HeavySwarm provides robust task analysis through intelligent question generation, parallel execution, and comprehensive synthesis with real-time progress monitoring.

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
    worker_model_name="gpt-4o-mini",
    show_dashboard=True
)

# Run analysis
result = swarm.run("Analyze the impact of AI on healthcare")
print(result)
```

## Architecture

The HeavySwarm follows a structured 5-phase workflow:

1. **Task Decomposition**: Complex tasks are broken down into specialized questions
2. **Question Generation**: AI-powered generation of role-specific questions
3. **Parallel Execution**: Four specialized agents work concurrently
4. **Result Collection**: Outputs are gathered and validated
5. **Synthesis**: Integration into a comprehensive final response

### Agent Specialization

| Agent | Role/Function |
|-------|---------------|
| **Research Agent** | Comprehensive information gathering and synthesis |
| **Analysis Agent** | Pattern recognition and statistical analysis |
| **Alternatives Agent** | Creative problem-solving and strategic options |
| **Verification Agent** | Validation, feasibility assessment, and quality assurance |
| **Synthesis Agent** | Multi-perspective integration and executive reporting |

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
    question_agent_model_name: str = "gpt-4o-mini",
    worker_model_name: str = "gpt-4o-mini",
    verbose: bool = False,
    max_workers: int = int(os.cpu_count() * 0.9),
    show_dashboard: bool = False,
    agent_prints_on: bool = False,
    output_type: str = "dict-all-except-first",
    worker_tools: Optional[tool_type] = None,
    random_loops_per_agent: bool = False,
    max_loops: int = 1,
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
| `question_agent_model_name` | `str` | `"gpt-4o-mini"` | Language model for question generation |
| `worker_model_name` | `str` | `"gpt-4o-mini"` | Language model for specialized worker agents |
| `verbose` | `bool` | `False` | Enable detailed logging and debug output |
| `max_workers` | `int` | `int(os.cpu_count() * 0.9)` | Maximum concurrent workers for parallel execution |
| `show_dashboard` | `bool` | `False` | Enable rich dashboard with progress visualization |
| `agent_prints_on` | `bool` | `False` | Enable individual agent output printing |
| `output_type` | `str` | `"dict-all-except-first"` | Output formatting type for conversation history |
| `worker_tools` | `Optional[tool_type]` | `None` | Tools available to worker agents for enhanced functionality |
| `random_loops_per_agent` | `bool` | `False` | Enable random number of loops per agent (1-10 range) |
| `max_loops` | `int` | `1` | Maximum number of loops when using random_loops_per_agent |

##### Raises

- `ValueError`: If `loops_per_agent` is 0 or negative

- `ValueError`: If `worker_model_name` is None

- `ValueError`: If `question_agent_model_name` is None

##### Example

```python
swarm = HeavySwarm(
    name="AdvancedAnalysisSwarm",
    description="Comprehensive multi-agent analysis system",
    timeout=600,
    loops_per_agent=2,
    question_agent_model_name="gpt-4o",
    worker_model_name="gpt-4o",
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

- `Dict[str, str]`: Clean dictionary containing only the questions:
  - `research_question` (str): Question for comprehensive information gathering
  - `analysis_question` (str): Question for pattern analysis and insights
  - `alternatives_question` (str): Question for exploring creative solutions
  - `verification_question` (str): Question for validation and feasibility
  - `error` (str): Error message if question generation fails (on error only)

**Example:**

```python
questions = swarm.get_questions_only("Analyze market trends for EVs")
print(questions['research_question'])
print(questions['analysis_question'])
```

##### `get_questions_as_list(task: str) -> List[str]`

Generate specialized questions and return them as an ordered list.

**Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `task` | `str` | Yes | - | The main task or query to decompose into specialized questions |

**Returns:**

- `List[str]`: Ordered list of 4 specialized questions:
  - `[0]` Research question for comprehensive information gathering
  - `[1]` Analysis question for pattern analysis and insights
  - `[2]` Alternatives question for exploring creative solutions
  - `[3]` Verification question for validation and feasibility assessment
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

**Example:**

```python
try:
    swarm.reliability_check()  # Automatically called in __init__
    print("All checks passed!")
except ValueError as e:
    print(f"Configuration error: {e}")
```

##### `create_agents() -> Dict[str, Agent]`

Create and cache the 4 specialized agents with detailed role-specific prompts.

**Parameters:**

- None

**Returns:**

| Key             | Agent Instance         | Description                                 |
|-----------------|-----------------------|---------------------------------------------|
| `'research'`    | Research Agent        | Research Agent instance                     |
| `'analysis'`    | Analysis Agent        | Analysis Agent instance                     |
| `'alternatives'`| Alternatives Agent    | Alternatives Agent instance                 |
| `'verification'`| Verification Agent    | Verification Agent instance                 |
| `'synthesis'`   | Synthesis Agent       | Synthesis Agent instance                    |

**Example:**

```python
agents = swarm.create_agents()
research_agent = agents['research']
print(f"Research agent name: {research_agent.agent_name}")
```

## Examples

### Basic Usage

```python
from swarms.structs.heavy_swarm import HeavySwarm

# Initialize HeavySwarm
swarm = HeavySwarm(
    name="Analysis Team",
    description="Multi-agent analysis system",
    worker_model_name="gpt-4o-mini",
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
    worker_model_name="gpt-4o-mini",
    question_agent_model_name="gpt-4o-mini",
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
    worker_model_name="gpt-4o",
    question_agent_model_name="gpt-4o",
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
    question_agent_model_name="gpt-4o",
    worker_model_name="gpt-4o-mini",
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
    worker_model_name="gpt-4o-mini",
    question_agent_model_name="gpt-4o-mini",
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
        worker_model_name="gpt-4o-mini",
        question_agent_model_name="gpt-4o-mini"
    )
    print("✅ Configuration valid")
except ValueError as e:
    print(f"❌ Configuration error: {e}")

# Test question generation with error handling
swarm = HeavySwarm(
    name="ErrorTest",
    worker_model_name="gpt-4o-mini",
    question_agent_model_name="gpt-4o-mini"
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
    worker_model_name="gpt-4o",
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
    worker_model_name="gpt-4o",  # Highest quality
    question_agent_model_name="gpt-4o",
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
    worker_model_name="gpt-4o",
    question_agent_model_name="gpt-4o-mini",
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
| **Cost**            | Higher model versions (`gpt-4o`) provide better analysis but increase costs                                 |
| **Timeouts**        | Complex tasks may require longer `timeout` values                                                           |
| **Tools**           | Integrate domain-specific tools for enhanced analysis capabilities                                           |
| **Dashboard**       | Enable `show_dashboard=True` for visual progress tracking                                                   |
| **Batch Processing**| Disable dashboard and verbose logging for efficient batch operations                                        |
