# LLM Council: 3-Step Quickstart Guide

The LLM Council enables collaborative decision-making with multiple AI agents through peer review and synthesis. Inspired by Andrej Karpathy's llm-council, it creates a council of specialized agents that respond independently, review each other's anonymized responses, and have a Chairman synthesize the best elements into a final answer.

## Overview

| Feature | Description |
|---------|-------------|
| **Multiple Perspectives** | Each council member provides unique insights from different viewpoints |
| **Peer Review** | Members evaluate and rank each other's responses anonymously |
| **Synthesis** | Chairman combines the best elements from all responses |
| **Transparency** | See both individual responses and evaluation rankings |

---

## Step 1: Install and Import

First, ensure you have Swarms installed and import the LLMCouncil class:

```bash
pip install swarms
```

```python
from swarms.structs.llm_council import LLMCouncil
```

---

## Step 2: Create the Council

Create an LLM Council with default council members (GPT-5.1, Gemini 3 Pro, Claude Sonnet 4.5, and Grok-4):

```python
# Create the council with default members
council = LLMCouncil(
    name="Decision Council",
    verbose=True,
    output_type="dict-all-except-first"
)
```

---

## Step 3: Run a Query

Execute a query and get the synthesized response:

```python
# Run a query
result = council.run("What are the key factors to consider when choosing a cloud provider for enterprise applications?")

# Access the final synthesized answer
print(result["final_response"])

# View individual member responses
print(result["original_responses"])

# See how members ranked each other
print(result["evaluations"])
```

---

## Complete Example

Here's a complete working example:

```python
from swarms.structs.llm_council import LLMCouncil

# Step 1: Create the council
council = LLMCouncil(
    name="Strategy Council",
    description="A council for strategic decision-making",
    verbose=True,
    output_type="dict-all-except-first"
)

# Step 2: Run a strategic query
result = council.run(
    "Should a B2B SaaS startup prioritize product-led growth or sales-led growth? "
    "Consider factors like market size, customer acquisition costs, and scalability."
)

# Step 3: Process results
print("=" * 50)
print("FINAL SYNTHESIZED ANSWER:")
print("=" * 50)
print(result["final_response"])
```

---

## Custom Council Members

For specialized domains, create custom council members:

```python
from swarms import Agent
from swarms.structs.llm_council import LLMCouncil, get_gpt_councilor_prompt

# Create specialized agents
finance_expert = Agent(
    agent_name="Finance-Councilor",
    system_prompt="You are a financial analyst specializing in market analysis and investment strategies...",
    model_name="gpt-4.1",
    max_loops=1,
)

tech_expert = Agent(
    agent_name="Technology-Councilor", 
    system_prompt="You are a technology strategist specializing in digital transformation...",
    model_name="gpt-4.1",
    max_loops=1,
)

risk_expert = Agent(
    agent_name="Risk-Councilor",
    system_prompt="You are a risk management expert specializing in enterprise risk assessment...",
    model_name="gpt-4.1",
    max_loops=1,
)

# Create council with custom members
council = LLMCouncil(
    council_members=[finance_expert, tech_expert, risk_expert],
    chairman_model="gpt-4.1",
    verbose=True
)

result = council.run("Evaluate the risk-reward profile of investing in AI infrastructure")
```

---

## CLI Usage

Run LLM Council directly from the command line:

```bash
swarms llm-council --task "What is the best approach to implement microservices architecture?"
```

With verbose output:

```bash
swarms llm-council --task "Analyze the pros and cons of remote work" --verbose
```

---

## Use Cases

| Domain | Example Query |
|--------|---------------|
| **Business Strategy** | "Should we expand internationally or focus on domestic growth?" |
| **Technology** | "Which database architecture best suits our high-throughput requirements?" |
| **Finance** | "Evaluate investment opportunities in the renewable energy sector" |
| **Healthcare** | "What treatment approaches should be considered for this patient profile?" |
| **Legal** | "What are the compliance implications of this data processing policy?" |

---

## Next Steps

- Explore [LLM Council Examples](./llm_council_examples.md) for domain-specific implementations
- Learn about [LLM Council Reference Documentation](../swarms/structs/llm_council.md) for complete API details
- Try the [CLI Reference](../swarms/cli/cli_reference.md) for DevOps integration

