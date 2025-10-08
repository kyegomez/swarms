# MixtureOfAgents Examples

The MixtureOfAgents architecture combines multiple specialized agents with an aggregator agent to process complex tasks. This architecture is particularly effective for tasks requiring diverse expertise and consensus-building among different specialists.

## Prerequisites

- Python 3.7+
- OpenAI API key or other supported LLM provider keys
- Swarms library

## Installation

```bash
pip3 install -U swarms
```

## Environment Variables

```plaintext
WORKSPACE_DIR="agent_workspace"
OPENAI_API_KEY=""
ANTHROPIC_API_KEY=""
GROQ_API_KEY=""
```

## Basic Usage

### 1. Initialize Specialized Agents

```python
from swarms import Agent, MixtureOfAgents

# Initialize specialized agents
legal_expert = Agent(
    agent_name="Legal-Expert",
    system_prompt="""You are a legal expert specializing in contract law. Your responsibilities include:
    1. Analyzing legal documents and contracts
    2. Identifying potential legal risks
    3. Ensuring regulatory compliance
    4. Providing legal recommendations
    5. Drafting and reviewing legal documents""",
    model_name="gpt-4.1",
    max_loops=1,
)

financial_expert = Agent(
    agent_name="Financial-Expert",
    system_prompt="""You are a financial expert specializing in business finance. Your tasks include:
    1. Analyzing financial implications
    2. Evaluating costs and benefits
    3. Assessing financial risks
    4. Providing financial projections
    5. Recommending financial strategies""",
    model_name="gpt-4.1",
    max_loops=1,
)

business_expert = Agent(
    agent_name="Business-Expert",
    system_prompt="""You are a business strategy expert. Your focus areas include:
    1. Analyzing business models
    2. Evaluating market opportunities
    3. Assessing competitive advantages
    4. Providing strategic recommendations
    5. Planning business development""",
    model_name="gpt-4.1",
    max_loops=1,
)

# Initialize aggregator agent
aggregator = Agent(
    agent_name="Decision-Aggregator",
    system_prompt="""You are a decision aggregator responsible for:
    1. Synthesizing input from multiple experts
    2. Resolving conflicting viewpoints
    3. Prioritizing recommendations
    4. Providing coherent final decisions
    5. Ensuring comprehensive coverage of all aspects""",
    model_name="gpt-4.1",
    max_loops=1,
)
```

### 2. Create and Run MixtureOfAgents

```python
# Create list of specialist agents
specialists = [legal_expert, financial_expert, business_expert]

# Initialize the mixture of agents
moa = MixtureOfAgents(
    agents=specialists,
    aggregator_agent=aggregator,
    layers=3,
)

# Run the analysis
result = moa.run(
    "Analyze the proposed merger between Company A and Company B, considering legal, financial, and business aspects."
)
```

## Advanced Usage

### 1. Custom Configuration with System Prompts

```python
# Initialize MixtureOfAgents with custom aggregator prompt
moa = MixtureOfAgents(
    agents=specialists,
    aggregator_agent=aggregator,
    aggregator_system_prompt="""As the decision aggregator, synthesize the analyses from all specialists into a coherent recommendation:
    1. Summarize key points from each specialist
    2. Identify areas of agreement and disagreement
    3. Weigh different perspectives
    4. Provide a balanced final recommendation
    5. Highlight key risks and opportunities""",
    layers=3,
)

result = moa.run("Evaluate the potential acquisition of StartupX")
```

### 2. Error Handling and Validation

```python
try:
    moa = MixtureOfAgents(
        agents=specialists,
        aggregator_agent=aggregator,
        layers=3,
        verbose=True,
    )
    
    result = moa.run("Complex analysis task")
    
    # Validate and process results
    if result:
        print("Analysis complete:")
        print(result)
    else:
        print("Analysis failed to produce results")
        
except Exception as e:
    print(f"Error in analysis: {str(e)}")
```

## Best Practices

1. Agent Selection and Configuration:
   - Choose specialists with complementary expertise
   - Configure appropriate system prompts
   - Set suitable model parameters

2. Aggregator Configuration:
   - Define clear aggregation criteria
   - Set appropriate weights for different opinions
   - Configure conflict resolution strategies

3. Layer Management:
   - Set appropriate number of layers
   - Monitor layer effectiveness
   - Adjust based on task complexity

4. Quality Control:
   - Implement validation checks
   - Monitor agent performance
   - Ensure comprehensive coverage

## Example Implementation

Here's a complete example showing how to use MixtureOfAgents for a comprehensive business analysis:

```python
import os
from swarms import Agent, MixtureOfAgents

# Initialize specialist agents
market_analyst = Agent(
    agent_name="Market-Analyst",
    system_prompt="""You are a market analysis specialist focusing on:
    1. Market size and growth
    2. Competitive landscape
    3. Customer segments
    4. Market trends
    5. Entry barriers""",
    model_name="gpt-4.1",
    max_loops=1,
)

financial_analyst = Agent(
    agent_name="Financial-Analyst",
    system_prompt="""You are a financial analysis expert specializing in:
    1. Financial performance
    2. Valuation metrics
    3. Cash flow analysis
    4. Investment requirements
    5. ROI projections""",
    model_name="gpt-4.1",
    max_loops=1,
)

risk_analyst = Agent(
    agent_name="Risk-Analyst",
    system_prompt="""You are a risk assessment specialist focusing on:
    1. Market risks
    2. Operational risks
    3. Financial risks
    4. Regulatory risks
    5. Strategic risks""",
    model_name="gpt-4.1",
    max_loops=1,
)

# Initialize aggregator
aggregator = Agent(
    agent_name="Strategic-Aggregator",
    system_prompt="""You are a strategic decision aggregator responsible for:
    1. Synthesizing specialist analyses
    2. Identifying key insights
    3. Evaluating trade-offs
    4. Making recommendations
    5. Providing action plans""",
    model_name="gpt-4.1",
    max_loops=1,
)

# Create and configure MixtureOfAgents
try:
    moa = MixtureOfAgents(
        agents=[market_analyst, financial_analyst, risk_analyst],
        aggregator_agent=aggregator,
        aggregator_system_prompt="""Synthesize the analyses from all specialists to provide:
        1. Comprehensive situation analysis
        2. Key opportunities and risks
        3. Strategic recommendations
        4. Implementation considerations
        5. Success metrics""",
        layers=3,
        verbose=True,
    )
    
    # Run the analysis
    result = moa.run(
        """Evaluate the business opportunity for expanding into the electric vehicle market:
        1. Market potential and competition
        2. Financial requirements and projections
        3. Risk assessment and mitigation strategies"""
    )
    
    # Process and display results
    print("\nComprehensive Analysis Results:")
    print("=" * 50)
    print(result)
    print("=" * 50)
    
except Exception as e:
    print(f"Error during analysis: {str(e)}")
```

This comprehensive guide demonstrates how to effectively use the MixtureOfAgents architecture for complex analysis tasks requiring multiple expert perspectives and consensus-building. 