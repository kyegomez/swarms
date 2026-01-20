# LLM Council: Complete Guide

A comprehensive guide to using the LLM Council for collaborative decision-making with multiple AI agents through peer review and synthesis.

## Overview

The LLM Council enables collaborative decision-making with multiple AI agents through peer review and synthesis. Inspired by Andrej Karpathy's llm-council, it creates a council of specialized agents that respond independently, review each other's anonymized responses, and have a Chairman synthesize the best elements into a final answer.

| Feature | Description |
|---------|-------------|
| **Multiple Perspectives** | Each council member provides unique insights from different viewpoints |
| **Peer Review** | Members evaluate and rank each other's responses anonymously |
| **Synthesis** | Chairman combines the best elements from all responses |
| **Transparency** | See both individual responses and evaluation rankings |

---

## Installation

```bash
pip install swarms
```

---

## Quick Start

### Step 1: Import and Create

```python
from swarms.structs.llm_council import LLMCouncil

# Create the council with default members
council = LLMCouncil(
    name="Decision Council",
    verbose=True,
    output_type="dict-all-except-first"
)
```

### Step 2: Run a Query

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

## Council Members

The default council consists of:

| Council Member                 | Description                   |
|-------------------------------|-------------------------------|
| **GPT-5.1-Councilor**         | Analytical and comprehensive  |
| **Gemini-3-Pro-Councilor**    | Concise and well-processed    |
| **Claude-Sonnet-4.5-Councilor** | Thoughtful and balanced     |
| **Grok-4-Councilor**          | Creative and innovative       |

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

## Example Files

All LLM Council examples are located in the [`examples/multi_agent/llm_council_examples/`](https://github.com/kyegomez/swarms/tree/master/examples/multi_agent/llm_council_examples) directory.

### Marketing & Business

- **[marketing_strategy_council.py](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/llm_council_examples/marketing_strategy_council.py)** - Marketing strategy analysis and recommendations
- **[business_strategy_council.py](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/llm_council_examples/business_strategy_council.py)** - Comprehensive business strategy development

### Finance & Investment

- **[finance_analysis_council.py](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/llm_council_examples/finance_analysis_council.py)** - Financial analysis and investment recommendations
- **[etf_stock_analysis_council.py](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/llm_council_examples/etf_stock_analysis_council.py)** - ETF and stock analysis with portfolio recommendations

### Medical & Healthcare

- **[medical_treatment_council.py](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/llm_council_examples/medical_treatment_council.py)** - Medical treatment recommendations and care plans
- **[medical_diagnosis_council.py](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/llm_council_examples/medical_diagnosis_council.py)** - Diagnostic analysis based on symptoms

### Technology & Research

- **[technology_assessment_council.py](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/llm_council_examples/technology_assessment_council.py)** - Technology evaluation and implementation strategy
- **[research_analysis_council.py](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/llm_council_examples/research_analysis_council.py)** - Comprehensive research analysis on complex topics

### Legal

- **[legal_analysis_council.py](https://github.com/kyegomez/swarms/blob/master/examples/multi_agent/llm_council_examples/legal_analysis_council.py)** - Legal implications and compliance analysis

## Running Examples

Run any example directly:

```bash
python examples/multi_agent/llm_council_examples/marketing_strategy_council.py
python examples/multi_agent/llm_council_examples/finance_analysis_council.py
python examples/multi_agent/llm_council_examples/medical_diagnosis_council.py
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

## Basic Usage Pattern

All examples follow the same pattern:

```python
from swarms.structs.llm_council import LLMCouncil

# Create the council
council = LLMCouncil(verbose=True)

# Run a query
result = council.run("Your query here")

# Access results
print(result["final_response"])  # Chairman's synthesized answer
print(result["original_responses"])  # Individual member responses
print(result["evaluations"])  # How members ranked each other
```

---

## Customization

You can create custom council members:

```python
from swarms import Agent
from swarms.structs.llm_council import LLMCouncil, get_gpt_councilor_prompt

custom_agent = Agent(
    agent_name="Custom-Councilor",
    system_prompt=get_gpt_councilor_prompt(),
    model_name="gpt-4.1",
    max_loops=1,
)

council = LLMCouncil(
    council_members=[custom_agent, ...],
    chairman_model="gpt-5.1",
    verbose=True
)
```

---

## Next Steps

- Learn about [LLM Council Reference Documentation](../structs/llm_council.md) for complete API details
- Try the [CLI Reference](../cli/cli_reference.md) for DevOps integration
- See [GitHub Examples](https://github.com/kyegomez/swarms/tree/master/examples/multi_agent/llm_council_examples)
