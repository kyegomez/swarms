# MultiAgentRouter Documentation

The MultiAgentRouter is a sophisticated task routing system that efficiently delegates tasks to specialized AI agents. It uses a "boss" agent to analyze incoming tasks and route them to the most appropriate specialized agent based on their capabilities and expertise.

## Table of Contents
- [Installation](#installation)
- [Key Components](#key-components)
- [Arguments](#arguments)
- [Methods](#methods)
- [Usage Examples](#usage-examples)
  - [Healthcare](#healthcare-example)
  - [Finance](#finance-example)
  - [Legal](#legal-example)
  - [Research](#research-example)

## Installation

```bash
pip install swarms
```

## Key Components

### Arguments Table

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| name | str | "swarm-router" | Name identifier for the router instance |
| description | str | "Routes tasks..." | Description of the router's purpose |
| agents | List[Agent] | [] | List of available specialized agents |
| model | str | "gpt-4o-mini" | Base language model for the boss agent |
| temperature | float | 0.1 | Temperature parameter for model outputs |
| shared_memory_system | callable | None | Optional shared memory system |
| output_type | Literal["json", "string"] | "json" | Format of agent outputs |
| execute_task | bool | True | Whether to execute routed tasks |

### Methods Table

| Method | Arguments | Returns | Description |
|--------|-----------|---------|-------------|
| route_task | task: str | dict | Routes a single task to appropriate agent |
| batch_run | tasks: List[str] | List[dict] | Sequentially routes multiple tasks |
| concurrent_batch_run | tasks: List[str] | List[dict] | Concurrently routes multiple tasks |
| query_ragent | task: str | str | Queries the research agent |
| find_agent_in_list | agent_name: str | Optional[Agent] | Finds agent by name |

## Production Examples

### Healthcare Example

```python
from swarms import Agent, MultiAgentRouter

# Define specialized healthcare agents
agents = [
    Agent(
        agent_name="DiagnosisAgent",
        description="Specializes in preliminary symptom analysis and diagnostic suggestions",
        system_prompt="""You are a medical diagnostic assistant. Analyze symptoms and provide 
        evidence-based diagnostic suggestions, always noting this is for informational purposes 
        only and recommending professional medical consultation.""",
        model_name="openai/gpt-4o"
    ),
    Agent(
        agent_name="TreatmentPlanningAgent",
        description="Assists in creating treatment plans and medical documentation",
        system_prompt="""You are a treatment planning assistant. Help create structured 
        treatment plans based on confirmed diagnoses, following medical best practices 
        and guidelines.""",
        model_name="openai/gpt-4o"
    ),
    Agent(
        agent_name="MedicalResearchAgent",
        description="Analyzes medical research papers and clinical studies",
        system_prompt="""You are a medical research analyst. Analyze and summarize medical 
        research papers, clinical trials, and scientific studies, providing evidence-based 
        insights.""",
        model_name="openai/gpt-4o"
    )
]

# Initialize router
healthcare_router = MultiAgentRouter(
    name="Healthcare-Router",
    description="Routes medical and healthcare-related tasks to specialized agents",
    agents=agents,
    model="gpt-4.1",
    temperature=0.1
)

# Example usage
try:
    # Process medical case
    case_analysis = healthcare_router.route_task(
        """Patient presents with: 
        - Persistent dry cough for 3 weeks
        - Mild fever (38.1°C)
        - Fatigue
        Analyze symptoms and suggest potential diagnoses for healthcare provider review."""
    )
    
    # Research treatment options
    treatment_research = healthcare_router.route_task(
        """Find recent clinical studies on treatment efficacy for community-acquired 
        pneumonia in adult patients, focusing on outpatient care."""
    )
    
    # Process multiple cases concurrently
    cases = [
        "Case 1: Patient symptoms...",
        "Case 2: Patient symptoms...",
        "Case 3: Patient symptoms..."
    ]
    concurrent_results = healthcare_router.concurrent_batch_run(cases)
    
except Exception as e:
    logger.error(f"Error in healthcare processing: {str(e)}")
```

### Finance Example

```python
# Define specialized finance agents
finance_agents = [
    Agent(
        agent_name="MarketAnalysisAgent",
        description="Analyzes market trends and provides trading insights",
        system_prompt="""You are a financial market analyst. Analyze market data, trends, 
        and indicators to provide evidence-based market insights and trading suggestions.""",
        model_name="openai/gpt-4o"
    ),
    Agent(
        agent_name="RiskAssessmentAgent",
        description="Evaluates financial risks and compliance requirements",
        system_prompt="""You are a risk assessment specialist. Analyze financial data 
        and operations for potential risks, ensuring regulatory compliance and suggesting 
        risk mitigation strategies.""",
        model_name="openai/gpt-4o"
    ),
    Agent(
        agent_name="InvestmentAgent",
        description="Provides investment strategies and portfolio management",
        system_prompt="""You are an investment strategy specialist. Develop and analyze 
        investment strategies, portfolio allocations, and provide long-term financial 
        planning guidance.""",
        model_name="openai/gpt-4o"
    )
]

# Initialize finance router
finance_router = MultiAgentRouter(
    name="Finance-Router",
    description="Routes financial analysis and investment tasks",
    agents=finance_agents
)

# Example tasks
tasks = [
    """Analyze current market conditions for technology sector, focusing on:
    - AI/ML companies
    - Semiconductor manufacturers
    - Cloud service providers
    Provide risk assessment and investment opportunities.""",
    
    """Develop a diversified portfolio strategy for a conservative investor with:
    - Investment horizon: 10 years
    - Risk tolerance: Low to medium
    - Initial investment: $500,000
    - Monthly contribution: $5,000""",
    
    """Conduct risk assessment for a fintech startup's crypto trading platform:
    - Regulatory compliance requirements
    - Security measures
    - Operational risks
    - Market risks"""
]

# Process tasks concurrently
results = finance_router.concurrent_batch_run(tasks)
```

### Legal Example

```python
# Define specialized legal agents
legal_agents = [
    Agent(
        agent_name="ContractAnalysisAgent",
        description="Analyzes legal contracts and documents",
        system_prompt="""You are a legal document analyst. Review contracts and legal 
        documents for key terms, potential issues, and compliance requirements.""",
        model_name="openai/gpt-4o"
    ),
    Agent(
        agent_name="ComplianceAgent",
        description="Ensures regulatory compliance and updates",
        system_prompt="""You are a legal compliance specialist. Monitor and analyze 
        regulatory requirements, ensuring compliance and suggesting necessary updates 
        to policies and procedures.""",
        model_name="openai/gpt-4o"
    ),
    Agent(
        agent_name="LegalResearchAgent",
        description="Conducts legal research and case analysis",
        system_prompt="""You are a legal researcher. Research relevant cases, statutes, 
        and regulations, providing comprehensive legal analysis and citations.""",
        model_name="openai/gpt-4o"
    )
]

# Initialize legal router
legal_router = MultiAgentRouter(
    name="Legal-Router",
    description="Routes legal analysis and compliance tasks",
    agents=legal_agents
)

# Example usage for legal department
contract_analysis = legal_router.route_task(
    """Review the following software licensing agreement:
    [contract text]
    
    Analyze for:
    1. Key terms and conditions
    2. Potential risks and liabilities
    3. Compliance with current regulations
    4. Suggested modifications"""
)
```

## Error Handling and Best Practices

1. Always use try-except blocks for task routing:
```python
try:
    result = router.route_task(task)
except Exception as e:
    logger.error(f"Task routing failed: {str(e)}")
```

2. Monitor agent performance:
```python
if result["execution"]["execution_time"] > 5.0:
    logger.warning(f"Long execution time for task: {result['task']['original']}")
```

3. Implement rate limiting for concurrent tasks:
```python
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=5) as executor:
    results = router.concurrent_batch_run(tasks)
```

4. Regular agent validation:
```python
for agent in router.agents.values():
    if not agent.validate():
        logger.error(f"Agent validation failed: {agent.name}")
```

## Performance Considerations

1. Task Batching

- Group similar tasks together

- Use concurrent_batch_run for independent tasks

- Monitor memory usage with large batches

2. Model Selection

- Choose appropriate models based on task complexity

- Balance speed vs. accuracy requirements

- Consider cost implications

3. Response Caching

- Implement caching for frequently requested analyses

- Use shared memory system for repeated queries

- Regular cache invalidation for time-sensitive data

## Security Considerations

1. Data Privacy

- Implement data encryption

- Handle sensitive information appropriately

- Regular security audits

2. Access Control

- Implement role-based access

- Audit logging

- Regular permission reviews

## Monitoring and Logging

1. Performance Metrics

- Response times

- Success rates

- Error rates

- Resource utilization

2. Logging

- Use structured logging

- Implement log rotation

- Regular log analysis

3. Alerts

- Set up alerting for critical errors

- Monitor resource usage

- Track API rate limits