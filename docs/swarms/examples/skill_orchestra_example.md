# SkillOrchestra Examples

Examples demonstrating skill-aware agent routing with `SkillOrchestra`. For API reference and architecture details, see the [SkillOrchestra Reference](../structs/skill_orchestra.md).

## Table of Contents
- [Basic Usage](#basic-usage)
- [Custom Skill Handbook](#custom-skill-handbook)
- [Multi-Agent Selection](#multi-agent-selection)
- [Learning from Execution](#learning-from-execution)
- [Software Development Team](#software-development-team)
- [Financial Analysis Team](#financial-analysis-team)

## Basic Usage

The simplest way to use SkillOrchestra — it auto-generates a skill handbook from agent descriptions.

```python
from swarms import Agent, SkillOrchestra

# Define agents with distinct specializations
code_agent = Agent(
    agent_name="CodeExpert",
    description="Expert Python developer who writes clean, efficient, production-ready code",
    system_prompt="You are an expert Python developer. Write clean, well-documented, production-ready code with proper error handling and type hints.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

writer_agent = Agent(
    agent_name="TechWriter",
    description="Technical writing specialist who creates clear documentation and tutorials",
    system_prompt="You are a technical writing specialist. Write clear, comprehensive documentation with examples and proper formatting.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

researcher_agent = Agent(
    agent_name="Researcher",
    description="Research analyst who gathers, synthesizes, and compares information",
    system_prompt="You are a research analyst. Provide thorough, well-structured analysis with comparisons, trade-offs, and actionable recommendations.",
    model_name="gpt-4o-mini",
    max_loops=1,
)

# Create SkillOrchestra — auto-generates skill handbook from agent descriptions
orchestra = SkillOrchestra(
    name="DevTeamOrchestra",
    agents=[code_agent, writer_agent, researcher_agent],
    model="gpt-4o-mini",
    top_k_agents=1,          # Select the single best agent per task
    learning_enabled=False,  # Disable profile updates for simplicity
    output_type="final",     # Return only the final agent output
)

# Inspect the auto-generated skill handbook
print("Generated Skill Handbook:")
for skill in orchestra.skill_handbook.skills:
    print(f"  - {skill.name}: {skill.description}")

# Run a task — routes to the most competent agent automatically
result = orchestra.run("Write a Python function to parse and validate JSON config files")
print(result)
```

## Custom Skill Handbook

You can provide a pre-built skill handbook for full control over skill definitions and agent profiles.

```python
from swarms import Agent
from swarms.structs.skill_orchestra import (
    SkillOrchestra,
    SkillHandbook,
    SkillDefinition,
    AgentProfile,
    AgentSkillProfile,
)

# Define skills manually
skills = [
    SkillDefinition(
        name="python_coding",
        description="Writing Python code with proper patterns and error handling",
        category="engineering",
    ),
    SkillDefinition(
        name="api_design",
        description="Designing RESTful APIs with proper structure and documentation",
        category="engineering",
    ),
    SkillDefinition(
        name="technical_writing",
        description="Writing clear technical documentation and tutorials",
        category="documentation",
    ),
    SkillDefinition(
        name="data_analysis",
        description="Analyzing datasets and producing insights",
        category="analysis",
    ),
]

# Define agent profiles with explicit competence and cost
agent_profiles = [
    AgentProfile(
        agent_name="CodeExpert",
        skill_profiles=[
            AgentSkillProfile(skill_name="python_coding", competence=0.95, cost=1.2),
            AgentSkillProfile(skill_name="api_design", competence=0.9, cost=1.0),
            AgentSkillProfile(skill_name="technical_writing", competence=0.4, cost=1.5),
            AgentSkillProfile(skill_name="data_analysis", competence=0.6, cost=1.3),
        ],
    ),
    AgentProfile(
        agent_name="TechWriter",
        skill_profiles=[
            AgentSkillProfile(skill_name="python_coding", competence=0.3, cost=1.5),
            AgentSkillProfile(skill_name="api_design", competence=0.5, cost=1.2),
            AgentSkillProfile(skill_name="technical_writing", competence=0.95, cost=0.8),
            AgentSkillProfile(skill_name="data_analysis", competence=0.4, cost=1.4),
        ],
    ),
    AgentProfile(
        agent_name="DataAnalyst",
        skill_profiles=[
            AgentSkillProfile(skill_name="python_coding", competence=0.7, cost=1.0),
            AgentSkillProfile(skill_name="api_design", competence=0.3, cost=1.5),
            AgentSkillProfile(skill_name="technical_writing", competence=0.5, cost=1.2),
            AgentSkillProfile(skill_name="data_analysis", competence=0.95, cost=0.9),
        ],
    ),
]

handbook = SkillHandbook(skills=skills, agent_profiles=agent_profiles)

# Create agents
code_agent = Agent(agent_name="CodeExpert", description="Expert Python developer", model_name="gpt-4o-mini", max_loops=1)
writer_agent = Agent(agent_name="TechWriter", description="Technical writing specialist", model_name="gpt-4o-mini", max_loops=1)
data_agent = Agent(agent_name="DataAnalyst", description="Data analysis expert", model_name="gpt-4o-mini", max_loops=1)

# Pass the custom handbook
orchestra = SkillOrchestra(
    agents=[code_agent, writer_agent, data_agent],
    skill_handbook=handbook,
    auto_generate_skills=False,
)

# This routes to DataAnalyst (highest competence on data_analysis)
result = orchestra.run("Analyze the sales dataset and identify trends for Q3")
```

## Multi-Agent Selection

Select multiple agents per task by setting `top_k_agents > 1`. Selected agents execute concurrently.

```python
from swarms import Agent, SkillOrchestra

agents = [
    Agent(agent_name="BackendDev", description="Backend API developer with database expertise", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="FrontendDev", description="Frontend developer specializing in React and UI/UX", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="SecurityEngineer", description="Security expert for vulnerability assessment and secure coding", model_name="gpt-4o-mini", max_loops=1),
]

orchestra = SkillOrchestra(
    agents=agents,
    top_k_agents=2,  # Select the top 2 agents for each task
    output_type="dict",
)

# Both BackendDev and SecurityEngineer likely selected for this task
result = orchestra.run("Design a secure authentication system with JWT tokens and refresh token rotation")
```

## Learning from Execution

Enable learning to update agent skill profiles over time based on execution quality.

```python
from swarms import Agent, SkillOrchestra

agents = [
    Agent(agent_name="GeneralistA", description="General-purpose assistant", model_name="gpt-4o-mini", max_loops=1),
    Agent(agent_name="GeneralistB", description="General-purpose assistant", model_name="gpt-4o-mini", max_loops=1),
]

orchestra = SkillOrchestra(
    agents=agents,
    learning_enabled=True,   # Enable EMA profile updates
    learning_rate=0.1,       # How fast profiles adapt (higher = faster)
    max_loops=2,             # Run 2 loops: execute, learn, refine, learn
)

# After each execution, the LLM evaluates output quality and updates
# the agent's competence scores via exponential moving average:
#   new_competence = old_competence * (1 - lr) + quality_score * lr
result = orchestra.run("Summarize the key findings from this research paper")

# Inspect updated profiles
handbook = orchestra.get_handbook()
for profile in handbook["agent_profiles"]:
    print(f"\n{profile['agent_name']}:")
    for sp in profile["skill_profiles"]:
        print(f"  {sp['skill_name']}: competence={sp['competence']:.3f}, executions={sp['execution_count']}")
```

## Software Development Team

```python
from swarms import Agent, SkillOrchestra

# Define a full dev team
agents = [
    Agent(
        agent_name="ArchitectAgent",
        description="Software architect who designs system architecture, selects technology stacks, and creates scalable solutions",
        system_prompt="You are a software architect. Design clean, scalable system architectures with clear component boundaries and data flow.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="BackendAgent",
        description="Backend developer specializing in APIs, databases, and server-side logic",
        system_prompt="You are a backend developer. Write robust server-side code with proper API design, database queries, and error handling.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="FrontendAgent",
        description="Frontend developer specializing in React, CSS, and responsive UI components",
        system_prompt="You are a frontend developer. Build responsive, accessible UI components with clean React code and modern CSS.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="DevOpsAgent",
        description="DevOps engineer handling CI/CD pipelines, Docker, Kubernetes, and cloud infrastructure",
        system_prompt="You are a DevOps engineer. Design and implement CI/CD pipelines, containerization, and cloud infrastructure.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="QAAgent",
        description="QA engineer who writes test plans, test cases, and automated tests",
        system_prompt="You are a QA engineer. Write comprehensive test plans, unit tests, integration tests, and end-to-end tests.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
]

orchestra = SkillOrchestra(
    name="DevTeam",
    agents=agents,
    model="gpt-4o-mini",
    top_k_agents=1,
    learning_enabled=True,
    output_type="final",
)

# Each task automatically routes to the best-matched agent
tasks = [
    "Design the microservices architecture for an e-commerce platform",        # → ArchitectAgent
    "Write a REST API endpoint for user registration with input validation",   # → BackendAgent
    "Build a responsive product card component with image carousel",           # → FrontendAgent
    "Set up a GitHub Actions CI/CD pipeline with Docker and Kubernetes",       # → DevOpsAgent
    "Write a test plan and pytest test cases for the user registration API",   # → QAAgent
]

# Run all tasks sequentially
results = orchestra.batch_run(tasks)

# Or run concurrently for independent tasks
results = orchestra.concurrent_batch_run(tasks)
```

## Financial Analysis Team

```python
from swarms import Agent, SkillOrchestra

agents = [
    Agent(
        agent_name="QuantAnalyst",
        description="Quantitative analyst specializing in statistical modeling, risk metrics, and portfolio optimization",
        system_prompt="You are a quantitative analyst. Build statistical models, compute risk metrics (VaR, Sharpe, etc.), and optimize portfolios.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="FundamentalAnalyst",
        description="Fundamental analyst who evaluates company financials, earnings reports, and intrinsic valuations",
        system_prompt="You are a fundamental analyst. Analyze financial statements, compute valuation metrics, and assess company health.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
    Agent(
        agent_name="ComplianceOfficer",
        description="Regulatory compliance specialist for financial regulations, reporting requirements, and audit procedures",
        system_prompt="You are a compliance officer. Ensure regulatory compliance, review reporting requirements, and flag potential violations.",
        model_name="gpt-4o-mini",
        max_loops=1,
    ),
]

orchestra = SkillOrchestra(
    name="FinanceTeam",
    agents=agents,
    model="gpt-4o-mini",
    competence_weight=0.8,  # Prioritize competence over cost
    cost_weight=0.2,
    output_type="final",
)

# Routes to QuantAnalyst
result = orchestra.run("Calculate the Value-at-Risk for a portfolio with 60% equities, 30% bonds, 10% alternatives")

# Routes to FundamentalAnalyst
result = orchestra.run("Analyze Apple's Q3 earnings report and estimate fair value using DCF model")

# Routes to ComplianceOfficer
result = orchestra.run("Review our trading desk procedures for MiFID II compliance gaps")
```

View the source on [GitHub](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/skill_orchestra_examples/skill_orchestra_example.py).
