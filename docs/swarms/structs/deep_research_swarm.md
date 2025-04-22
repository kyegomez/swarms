# Deep Research Swarm

!!! abstract "Overview"
    The Deep Research Swarm is a powerful, production-grade research system that conducts comprehensive analysis across multiple domains using parallel processing and advanced AI agents.

    Key Features:
    
    - Parallel search processing
    
    - Multi-agent research coordination
    
    - Advanced information synthesis
    
    - Automated query generation
    
    - Concurrent task execution

## Getting Started

!!! tip "Quick Installation"
    ```bash
    pip install swarms
    ```

=== "Basic Usage"
    ```python
    from swarms.structs import DeepResearchSwarm

    # Initialize the swarm
    swarm = DeepResearchSwarm(
        name="MyResearchSwarm",
        output_type="json",
        max_loops=1
    )

    # Run a single research task
    results = swarm.run("What are the latest developments in quantum computing?")
    ```

=== "Batch Processing"
    ```python
    # Run multiple research tasks in parallel
    tasks = [
        "What are the environmental impacts of electric vehicles?",
        "How is AI being used in drug discovery?",
    ]
    batch_results = swarm.batched_run(tasks)
    ```

## Configuration

!!! info "Constructor Arguments"
    | Parameter | Type | Default | Description |
    |-----------|------|---------|-------------|
    | `name` | str | "DeepResearchSwarm" | Name identifier for the swarm |
    | `description` | str | "A swarm that conducts..." | Description of the swarm's purpose |
    | `research_agent` | Agent | research_agent | Custom research agent instance |
    | `max_loops` | int | 1 | Maximum number of research iterations |
    | `nice_print` | bool | True | Enable formatted console output |
    | `output_type` | str | "json" | Output format ("json" or "string") |
    | `max_workers` | int | CPU_COUNT * 2 | Maximum concurrent threads |
    | `token_count` | bool | False | Enable token counting |
    | `research_model_name` | str | "gpt-4o-mini" | Model to use for research |

## Core Methods

### Run
!!! example "Single Task Execution"
    ```python
    results = swarm.run("What are the latest breakthroughs in fusion energy?")
    ```

### Batched Run
!!! example "Parallel Task Execution"
    ```python
    tasks = [
        "What are current AI safety initiatives?",
        "How is CRISPR being used in agriculture?",
    ]
    results = swarm.batched_run(tasks)
    ```

### Step
!!! example "Single Step Execution"
    ```python
    results = swarm.step("Analyze recent developments in renewable energy storage")
    ```

## Domain-Specific Examples

=== "Scientific Research"
    ```python
    science_swarm = DeepResearchSwarm(
        name="ScienceSwarm",
        output_type="json",
        max_loops=2  # More iterations for thorough research
    )

    results = science_swarm.run(
        "What are the latest experimental results in quantum entanglement?"
    )
    ```

=== "Market Research"
    ```python
    market_swarm = DeepResearchSwarm(
        name="MarketSwarm",
        output_type="json"
    )

    results = market_swarm.run(
        "What are the emerging trends in electric vehicle battery technology market?"
    )
    ```

=== "News Analysis"
    ```python
    news_swarm = DeepResearchSwarm(
        name="NewsSwarm",
        output_type="string"  # Human-readable output
    )

    results = news_swarm.run(
        "What are the global economic impacts of recent geopolitical events?"
    )
    ```

=== "Medical Research"
    ```python
    medical_swarm = DeepResearchSwarm(
        name="MedicalSwarm",
        max_loops=2
    )

    results = medical_swarm.run(
        "What are the latest clinical trials for Alzheimer's treatment?"
    )
    ```

## Advanced Features

??? note "Custom Research Agent"
    ```python
    from swarms import Agent

    custom_agent = Agent(
        agent_name="SpecializedResearcher",
        system_prompt="Your specialized prompt here",
        model_name="gpt-4"
    )

    swarm = DeepResearchSwarm(
        research_agent=custom_agent,
        max_loops=2
    )
    ```

??? note "Parallel Processing Control"
    ```python
    swarm = DeepResearchSwarm(
        max_workers=8,  # Limit to 8 concurrent threads
        nice_print=False  # Disable console output for production
    )
    ```

## Best Practices

!!! success "Recommended Practices"
    1. **Query Formulation**: Be specific and clear in your research queries
    2. **Resource Management**: Adjust `max_workers` based on your system's capabilities
    3. **Output Handling**: Use appropriate `output_type` for your use case
    4. **Error Handling**: Implement try-catch blocks around swarm operations
    5. **Model Selection**: Choose appropriate models based on research complexity

## Limitations

!!! warning "Known Limitations"
    
    - Requires valid API keys for external services
    
    - Performance depends on system resources
    
    - Rate limits may apply to external API calls
    
    - Token limits apply to model responses

