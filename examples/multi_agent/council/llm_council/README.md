# LLM Council Examples

This directory contains examples demonstrating the LLM Council pattern, inspired by Andrej Karpathy's llm-council implementation. The LLM Council uses multiple specialized AI agents that:

1. Each respond independently to the task
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
from swarms import LLMCouncil

# Create the council
council = LLMCouncil(verbose=True, output_type="final")

# Run a task
result = council.run(task="Your task here")

# The chairman's synthesized answer
print(result)
```

`run` takes a single required `task` string and raises `ValueError` if it is empty. The `query` parameter has been removed; use `task` instead.

`output_type` controls what `run` returns: `"final"` gives only the chairman's answer, while the default `"dict-all-except-first"` returns every message — each member's response, each member's evaluation, and the chairman's synthesis — as a list of `{"role", "content"}` dicts.

With `verbose=True`, the council logs one line per stage through `loguru`:

```
[LLM Council] Initialized with 4 members: GPT-5.1-Councilor, Gemini-3-Pro-Councilor, Claude-Sonnet-4.5-Councilor, Grok-4-Councilor
[LLM Council] Collecting responses from 4 members
[LLM Council] Members ranking the anonymized responses
[LLM Council] Chairman synthesizing the final answer
[LLM Council] Session complete
```

## Running Examples

Run any example directly:

```bash
python examples/multi_agent/council/llm_council/marketing_strategy_council.py
python examples/multi_agent/council/llm_council/finance_analysis_council.py
python examples/multi_agent/council/llm_council/medical_diagnosis_council.py
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
from swarms import Agent, LLMCouncil
from swarms.prompts.llm_council_prompts import get_gpt_councilor_prompt

custom_agent = Agent(
    agent_name="Custom-Councilor",
    system_prompt=get_gpt_councilor_prompt(),
    model_name="gpt-5.4",
    max_loops=1,
)

council = LLMCouncil(
    council_members=[custom_agent, ...],
    chairman_model="gpt-5.1",
    verbose=True
)
```

All council prompts live in `swarms/prompts/llm_council_prompts.py`: the four councilor personas (`get_gpt_councilor_prompt`, `get_gemini_councilor_prompt`, `get_claude_councilor_prompt`, `get_grok_councilor_prompt`), the chairman (`get_chairman_prompt`), and the templates for peer evaluation (`get_evaluation_prompt`) and final synthesis (`get_synthesis_prompt`).

