# LLM Council Examples

This directory contains examples demonstrating the LLM Council pattern, inspired by Andrej Karpathy's llm-council implementation. The LLM Council uses multiple specialized AI agents that:

1. Each respond independently to queries
2. Review and rank each other's anonymized responses
3. Have a Chairman synthesize all responses into a final comprehensive answer

## Examples

### Marketing & Business
- **marketing_strategy_council.py** - Marketing strategy analysis and recommendations
- **business_strategy_council.py** - Comprehensive business strategy development

### Finance & Investment
- **finance_analysis_council.py** - Financial analysis and investment recommendations
- **etf_stock_analysis_council.py** - ETF and stock analysis with portfolio recommendations

### Medical & Healthcare
- **medical_treatment_council.py** - Medical treatment recommendations and care plans
- **medical_diagnosis_council.py** - Diagnostic analysis based on symptoms

### Technology & Research
- **technology_assessment_council.py** - Technology evaluation and implementation strategy
- **research_analysis_council.py** - Comprehensive research analysis on complex topics

### Legal
- **legal_analysis_council.py** - Legal implications and compliance analysis

## Usage

Each example follows the same pattern:

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

## Running Examples

Run any example directly:

```bash
python examples/multi_agent/llm_council_examples/marketing_strategy_council.py
python examples/multi_agent/llm_council_examples/finance_analysis_council.py
python examples/multi_agent/llm_council_examples/medical_diagnosis_council.py
```

## Key Features

- **Multiple Perspectives**: Each council member (GPT-5.1, Gemini, Claude, Grok) provides unique insights
- **Peer Review**: Members evaluate and rank each other's responses anonymously
- **Synthesis**: Chairman combines the best elements from all responses
- **Transparency**: See both individual responses and evaluation rankings

## Council Members

The default council consists of:
- **GPT-5.1-Councilor**: Analytical and comprehensive
- **Gemini-3-Pro-Councilor**: Concise and well-processed
- **Claude-Sonnet-4.5-Councilor**: Thoughtful and balanced
- **Grok-4-Councilor**: Creative and innovative

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

