# Advanced BatchedGridWorkflow Examples

This example demonstrates advanced usage patterns and configurations of the `BatchedGridWorkflow` for complex multi-agent scenarios.

## Custom Conversation Configuration

```python
from swarms import Agent
from swarms.structs.batched_grid_workflow import BatchedGridWorkflow

# Create agents with specific roles
researcher = Agent(
    model="gpt-4",
    system_prompt="You are a research specialist who conducts thorough investigations."
)

writer = Agent(
    model="gpt-4",
    system_prompt="You are a technical writer who creates clear, comprehensive documentation."
)

reviewer = Agent(
    model="gpt-4",
    system_prompt="You are a quality reviewer who ensures accuracy and completeness."
)

# Create workflow with custom conversation settings
workflow = BatchedGridWorkflow(
    id="custom-research-workflow",
    name="Custom Research Pipeline",
    description="Research, writing, and review pipeline with custom conversation tracking",
    agents=[researcher, writer, reviewer],
    conversation_args={
        "message_id_on": True,
        "conversation_id": "research-pipeline-001"
    },
    max_loops=2
)

# Research and documentation tasks
tasks = [
    "Research the latest developments in artificial intelligence safety",
    "Write comprehensive documentation for a new API endpoint",
    "Review and validate the technical specifications document"
]

# Execute with custom configuration
result = workflow.run(tasks)
```

## Iterative Refinement Workflow

```python
# Create refinement agents
initial_creator = Agent(
    model="gpt-4",
    system_prompt="You are a creative content creator who generates initial ideas and drafts."
)

detail_enhancer = Agent(
    model="gpt-4",
    system_prompt="You are a detail specialist who adds depth and specificity to content."
)

polish_expert = Agent(
    model="gpt-4",
    system_prompt="You are a polish expert who refines content for maximum impact and clarity."
)

# Create workflow with multiple refinement loops
workflow = BatchedGridWorkflow(
    name="Iterative Content Refinement",
    description="Multi-stage content creation with iterative improvement",
    agents=[initial_creator, detail_enhancer, polish_expert],
    max_loops=4
)

# Content creation tasks
tasks = [
    "Create an initial draft for a product launch announcement",
    "Add detailed specifications and technical details to the content",
    "Polish and refine the content for maximum engagement"
]

# Execute iterative refinement
result = workflow.run(tasks)
```

## Specialized Domain Workflow

```python
# Create domain-specific agents
medical_expert = Agent(
    model="gpt-4",
    system_prompt="You are a medical expert specializing in diagnostic procedures and treatment protocols."
)

legal_advisor = Agent(
    model="gpt-4",
    system_prompt="You are a legal advisor specializing in healthcare regulations and compliance."
)

technology_architect = Agent(
    model="gpt-4",
    system_prompt="You are a technology architect specializing in healthcare IT systems and security."
)

# Create specialized workflow
workflow = BatchedGridWorkflow(
    name="Healthcare Technology Assessment",
    description="Multi-domain assessment of healthcare technology solutions",
    agents=[medical_expert, legal_advisor, technology_architect],
    max_loops=1
)

# Domain-specific assessment tasks
tasks = [
    "Evaluate the medical efficacy and safety of a new diagnostic AI system",
    "Assess the legal compliance and regulatory requirements for the system",
    "Analyze the technical architecture and security implications of implementation"
]

# Execute specialized assessment
result = workflow.run(tasks)
```

## Parallel Analysis Workflow

```python
# Create analysis agents
market_analyst = Agent(
    model="gpt-4",
    system_prompt="You are a market analyst who evaluates business opportunities and market trends."
)

financial_analyst = Agent(
    model="gpt-4",
    system_prompt="You are a financial analyst who assesses investment potential and financial viability."
)

risk_assessor = Agent(
    model="gpt-4",
    system_prompt="You are a risk assessor who identifies potential threats and mitigation strategies."
)

# Create parallel analysis workflow
workflow = BatchedGridWorkflow(
    name="Comprehensive Business Analysis",
    description="Parallel analysis of market, financial, and risk factors",
    agents=[market_analyst, financial_analyst, risk_assessor],
    max_loops=2
)

# Analysis tasks
tasks = [
    "Analyze the market opportunity for a new fintech product",
    "Evaluate the financial projections and investment requirements",
    "Assess the regulatory and operational risks associated with the venture"
]

# Execute parallel analysis
result = workflow.run(tasks)
```

## Creative Collaboration Workflow

```python
# Create creative agents
visual_artist = Agent(
    model="gpt-4",
    system_prompt="You are a visual artist who creates compelling imagery and visual concepts."
)

music_composer = Agent(
    model="gpt-4",
    system_prompt="You are a music composer who creates original compositions and soundscapes."
)

story_writer = Agent(
    model="gpt-4",
    system_prompt="You are a story writer who crafts engaging narratives and character development."
)

# Create creative collaboration workflow
workflow = BatchedGridWorkflow(
    name="Multi-Media Creative Project",
    description="Collaborative creation across visual, audio, and narrative elements",
    agents=[visual_artist, music_composer, story_writer],
    max_loops=3
)

# Creative tasks
tasks = [
    "Design visual concepts for a fantasy adventure game",
    "Compose background music that enhances the game's atmosphere",
    "Write character backstories and dialogue for the main protagonists"
]

# Execute creative collaboration
result = workflow.run(tasks)
```

## Quality Assurance Workflow

```python
# Create QA agents
functional_tester = Agent(
    model="gpt-4",
    system_prompt="You are a functional tester who validates system behavior and user workflows."
)

security_tester = Agent(
    model="gpt-4",
    system_prompt="You are a security tester who identifies vulnerabilities and security issues."
)

performance_tester = Agent(
    model="gpt-4",
    system_prompt="You are a performance tester who evaluates system speed and resource usage."
)

# Create QA workflow
workflow = BatchedGridWorkflow(
    name="Comprehensive Quality Assurance",
    description="Multi-faceted testing across functional, security, and performance domains",
    agents=[functional_tester, security_tester, performance_tester],
    max_loops=1
)

# QA tasks
tasks = [
    "Test the user registration and authentication workflows",
    "Conduct security analysis of the payment processing system",
    "Evaluate the performance characteristics under high load conditions"
]

# Execute QA workflow
result = workflow.run(tasks)
```

## Advanced Features Demonstrated

- **Custom Conversation Tracking**: Advanced conversation management with custom IDs
- **Iterative Refinement**: Multiple loops for content improvement
- **Domain Specialization**: Agents with specific expertise areas
- **Parallel Analysis**: Simultaneous evaluation from multiple perspectives
- **Creative Collaboration**: Multi-modal creative content generation
- **Quality Assurance**: Comprehensive testing across multiple domains

## Best Practices

1. **Agent Specialization**: Create agents with specific roles and expertise
2. **Task Alignment**: Ensure tasks match agent capabilities
3. **Loop Configuration**: Use multiple loops for iterative processes
4. **Error Monitoring**: Monitor logs for execution issues
5. **Resource Management**: Consider computational requirements for multiple agents
