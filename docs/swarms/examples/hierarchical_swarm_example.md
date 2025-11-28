# Hierarchical Swarm Examples

This page provides simple, practical examples of how to use the `HierarchicalSwarm` for various real-world scenarios.

## Basic Example: Financial Analysis

```python
from swarms import Agent
from swarms.structs.hiearchical_swarm import HierarchicalSwarm

# Create specialized financial analysis agents
market_research_agent = Agent(
    agent_name="Market-Research-Specialist",
    agent_description="Expert in market research, trend analysis, and competitive intelligence",
    system_prompt="""You are a senior market research specialist with expertise in:
    - Market trend analysis and forecasting
    - Competitive landscape assessment
    - Consumer behavior analysis
    - Industry report generation
    - Market opportunity identification
    - Risk assessment and mitigation strategies""",
    model_name="gpt-4.1",
)

financial_analyst_agent = Agent(
    agent_name="Financial-Analysis-Expert",
    agent_description="Specialist in financial statement analysis, valuation, and investment research",
    system_prompt="""You are a senior financial analyst with deep expertise in:
    - Financial statement analysis (income statement, balance sheet, cash flow)
    - Valuation methodologies (DCF, comparable company analysis, precedent transactions)
    - Investment research and due diligence
    - Financial modeling and forecasting
    - Risk assessment and portfolio analysis
    - ESG (Environmental, Social, Governance) analysis""",
    model_name="gpt-4.1",
)

# Initialize the hierarchical swarm
financial_analysis_swarm = HierarchicalSwarm(
    name="Financial-Analysis-Hierarchical-Swarm",
    description="A hierarchical swarm for comprehensive financial analysis with specialized agents",
    agents=[market_research_agent, financial_analyst_agent],
    max_loops=2,
    verbose=True,
)

# Execute financial analysis
task = "Conduct a comprehensive analysis of Tesla (TSLA) stock including market position, financial health, and investment potential"
result = financial_analysis_swarm.run(task=task)
print(result)
```

## Development Team Example

```python
from swarms import Agent
from swarms.structs.hiearchical_swarm import HierarchicalSwarm

# Create specialized development agents
frontend_developer_agent = Agent(
    agent_name="Frontend-Developer",
    agent_description="Senior frontend developer expert in modern web technologies and user experience",
    system_prompt="""You are a senior frontend developer with expertise in:
    - Modern JavaScript frameworks (React, Vue, Angular)
    - TypeScript and modern ES6+ features
    - CSS frameworks and responsive design
    - State management (Redux, Zustand, Context API)
    - Web performance optimization
    - Accessibility (WCAG) and SEO best practices""",
    model_name="gpt-4.1",
)

backend_developer_agent = Agent(
    agent_name="Backend-Developer",
    agent_description="Senior backend developer specializing in server-side development and API design",
    system_prompt="""You are a senior backend developer with expertise in:
    - Server-side programming languages (Python, Node.js, Java, Go)
    - Web frameworks (Django, Flask, Express, Spring Boot)
    - Database design and optimization (SQL, NoSQL)
    - API design and REST/GraphQL implementation
    - Authentication and authorization systems
    - Microservices architecture and containerization""",
    model_name="gpt-4.1",
)

# Initialize the development swarm
development_department_swarm = HierarchicalSwarm(
    name="Autonomous-Development-Department",
    description="A fully autonomous development department with specialized agents",
    agents=[frontend_developer_agent, backend_developer_agent],
    max_loops=3,
    verbose=True,
)

# Execute development project
task = "Create a simple web app that allows users to upload a file and then download it. The app should be built with React and Node.js."
result = development_department_swarm.run(task=task)
print(result)
```

## Single Step Execution

```python
from swarms import Agent
from swarms.structs.hiearchical_swarm import HierarchicalSwarm

# Create analysis agents
market_agent = Agent(
    agent_name="Market-Analyst",
    agent_description="Expert in market analysis and trends",
    model_name="gpt-4.1",
)

technical_agent = Agent(
    agent_name="Technical-Analyst",
    agent_description="Specialist in technical analysis and patterns",
    model_name="gpt-4.1",
)

# Initialize the swarm
swarm = HierarchicalSwarm(
    name="Analysis-Swarm",
    description="A hierarchical swarm for comprehensive analysis",
    agents=[market_agent, technical_agent],
    max_loops=1,
    verbose=True,
)

# Execute a single step
task = "Analyze the current market trends for electric vehicles"
feedback = swarm.step(task=task)
print("Director Feedback:", feedback)
```

## Batch Processing

```python
from swarms import Agent
from swarms.structs.hiearchical_swarm import HierarchicalSwarm

# Create analysis agents
market_agent = Agent(
    agent_name="Market-Analyst",
    agent_description="Expert in market analysis and trends",
    model_name="gpt-4.1",
)

technical_agent = Agent(
    agent_name="Technical-Analyst",
    agent_description="Specialist in technical analysis and patterns",
    model_name="gpt-4.1",
)

# Initialize the swarm
swarm = HierarchicalSwarm(
    name="Analysis-Swarm",
    description="A hierarchical swarm for comprehensive analysis",
    agents=[market_agent, technical_agent],
    max_loops=2,
    verbose=True,
)

# Execute multiple tasks
tasks = [
    "Analyze Apple (AAPL) stock performance",
    "Evaluate Microsoft (MSFT) market position",
    "Assess Google (GOOGL) competitive landscape"
]

results = swarm.batched_run(tasks=tasks)
for i, result in enumerate(results):
    print(f"Task {i+1} Result:", result)
```

## Research Team Example

```python
from swarms import Agent
from swarms.structs.hiearchical_swarm import HierarchicalSwarm

# Create specialized research agents
research_manager = Agent(
    agent_name="Research-Manager",
    agent_description="Manages research operations and coordinates research tasks",
    system_prompt="You are a research manager responsible for overseeing research projects and coordinating research efforts.",
    model_name="gpt-4.1",
)

data_analyst = Agent(
    agent_name="Data-Analyst",
    agent_description="Analyzes data and generates insights",
    system_prompt="You are a data analyst specializing in processing and analyzing data to extract meaningful insights.",
    model_name="gpt-4.1",
)

research_assistant = Agent(
    agent_name="Research-Assistant",
    agent_description="Assists with research tasks and data collection",
    system_prompt="You are a research assistant who helps gather information and support research activities.",
    model_name="gpt-4.1",
)

# Initialize the research swarm
research_swarm = HierarchicalSwarm(
    name="Research-Team-Swarm",
    description="A hierarchical swarm for comprehensive research projects",
    agents=[research_manager, data_analyst, research_assistant],
    max_loops=2,
    verbose=True,
)

# Execute research project
task = "Conduct a comprehensive market analysis for a new AI-powered productivity tool"
result = research_swarm.run(task=task)
print(result)
```

## Visualizing Swarm Hierarchy

You can visualize the hierarchical structure of your swarm before executing tasks using the `display_hierarchy()` method:

```python
from swarms import Agent
from swarms.structs.hiearchical_swarm import HierarchicalSwarm

# Create specialized agents
research_agent = Agent(
    agent_name="Research-Analyst",
    agent_description="Specialized in comprehensive research and data gathering",
    model_name="gpt-4o-mini",
)

analysis_agent = Agent(
    agent_name="Data-Analyst",
    agent_description="Expert in data analysis and pattern recognition",
    model_name="gpt-4o-mini",
)

strategy_agent = Agent(
    agent_name="Strategy-Consultant",
    agent_description="Specialized in strategic planning and recommendations",
    model_name="gpt-4o-mini",
)

# Create hierarchical swarm
swarm = HierarchicalSwarm(
    name="Swarms Corporation Operations",
    description="Enterprise-grade hierarchical swarm for complex task execution",
    agents=[research_agent, analysis_agent, strategy_agent],
    max_loops=1,
    director_model_name="claude-haiku-4-5",
)

# Display the hierarchy visualization
swarm.display_hierarchy()
```

This will output a visual tree structure showing the Director and all worker agents, making it easy to understand the swarm's organizational structure before executing tasks.

## Key Takeaways

1. **Agent Specialization**: Create agents with specific, well-defined expertise areas
2. **Clear Task Descriptions**: Provide detailed, actionable task descriptions
3. **Appropriate Loop Count**: Set `max_loops` based on task complexity (1-3 for most tasks)
4. **Verbose Logging**: Enable verbose mode during development for debugging
5. **Context Preservation**: The swarm maintains full conversation history automatically
6. **Hierarchy Visualization**: Use `display_hierarchy()` to visualize swarm structure before execution

For more detailed information about the `HierarchicalSwarm` API and advanced usage patterns, see the [main documentation](hierarchical_swarm.md). 