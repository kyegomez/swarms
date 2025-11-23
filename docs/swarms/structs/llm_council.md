# LLM Council Class Documentation

```mermaid
flowchart TD
    A[User Query] --> B[LLM Council Initialization]
    B --> C{Council Members Provided?}
    C -->|No| D[Create Default Council]
    C -->|Yes| E[Use Provided Members]
    D --> F[Step 1: Parallel Response Generation]
    E --> F
    
    subgraph "Default Council Members"
        G1[GPT-5.1-Councilor<br/>Analytical & Comprehensive]
        G2[Gemini-3-Pro-Councilor<br/>Concise & Structured]
        G3[Claude-Sonnet-4.5-Councilor<br/>Thoughtful & Balanced]
        G4[Grok-4-Councilor<br/>Creative & Innovative]
    end
    
    F --> G1
    F --> G2
    F --> G3
    F --> G4
    
    G1 --> H[Collect All Responses]
    G2 --> H
    G3 --> H
    G4 --> H
    
    H --> I[Step 2: Anonymize Responses]
    I --> J[Assign Anonymous IDs: A, B, C, D...]
    
    J --> K[Step 3: Parallel Evaluation]
    
    subgraph "Evaluation Phase"
        K --> L1[Member 1 Evaluates All]
        K --> L2[Member 2 Evaluates All]
        K --> L3[Member 3 Evaluates All]
        K --> L4[Member 4 Evaluates All]
    end
    
    L1 --> M[Collect Evaluations & Rankings]
    L2 --> M
    L3 --> M
    L4 --> M
    
    M --> N[Step 4: Chairman Synthesis]
    N --> O[Chairman Agent]
    O --> P[Final Synthesized Response]
    
    P --> Q[Return Results Dictionary]
    
    style A fill:#e1f5ff
    style P fill:#c8e6c9
    style Q fill:#c8e6c9
    style O fill:#fff9c4
```

The `LLMCouncil` class orchestrates multiple specialized LLM agents to collaboratively answer queries through a structured peer review and synthesis process. Inspired by Andrej Karpathy's llm-council implementation, this architecture demonstrates how different models evaluate and rank each other's work, often selecting responses from other models as superior to their own.

## Workflow Overview

The LLM Council follows a four-step process:

1. **Parallel Response Generation**: All council members independently respond to the user query
2. **Anonymization**: Responses are anonymized with random IDs (A, B, C, D, etc.) to ensure objective evaluation
3. **Peer Review**: Each member evaluates and ranks all responses (including potentially their own)
4. **Synthesis**: The Chairman agent synthesizes all responses and evaluations into a final comprehensive answer

## Class Definition

### LLMCouncil

```python
class LLMCouncil:
```

### Attributes

| Attribute | Type | Description | Default |
|-----------|------|-------------|---------|
| `council_members` | `List[Agent]` | List of Agent instances representing council members | `None` (creates default council) |
| `chairman` | `Agent` | The Chairman agent responsible for synthesizing responses | Created during initialization |
| `verbose` | `bool` | Whether to print progress and intermediate results | `True` |

## Methods

### `__init__`

Initializes the LLM Council with council members and a Chairman agent.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `council_members` | `Optional[List[Agent]]` | `None` | List of Agent instances representing council members. If `None`, creates default council with GPT-5.1, Gemini 3 Pro, Claude Sonnet 4.5, and Grok-4. |
| `chairman_model` | `str` | `"gpt-5.1"` | Model name for the Chairman agent that synthesizes responses. |
| `verbose` | `bool` | `True` | Whether to print progress and intermediate results. |

#### Returns

| Type | Description |
|------|-------------|
| `LLMCouncil` | Initialized LLM Council instance. |

#### Description

Creates an LLM Council instance with specialized council members. If no members are provided, it creates a default council consisting of:
- **GPT-5.1-Councilor**: Analytical and comprehensive responses
- **Gemini-3-Pro-Councilor**: Concise and well-processed responses  
- **Claude-Sonnet-4.5-Councilor**: Thoughtful and balanced responses
- **Grok-4-Councilor**: Creative and innovative responses

The Chairman agent is automatically created with a specialized prompt for synthesizing responses.

#### Example Usage

```python
from swarms.structs.llm_council import LLMCouncil

# Create council with default members
council = LLMCouncil(verbose=True)

# Create council with custom members
from swarms import Agent
custom_members = [
    Agent(agent_name="Expert-1", model_name="gpt-4", max_loops=1),
    Agent(agent_name="Expert-2", model_name="claude-3-opus", max_loops=1),
]
council = LLMCouncil(
    council_members=custom_members,
    chairman_model="gpt-4",
    verbose=True
)
```

---

### `run`

Executes the full LLM Council workflow: parallel responses, anonymization, peer review, and synthesis.

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `query` | `str` | Required | The user's query to process through the council. |

#### Returns

| Type | Description |
|------|-------------|
| `Dict` | Dictionary containing the following keys: |

#### Return Dictionary Structure

| Key | Type | Description |
|-----|------|-------------|
| `query` | `str` | The original user query. |
| `original_responses` | `Dict[str, str]` | Dictionary mapping council member names to their original responses. |
| `evaluations` | `Dict[str, str]` | Dictionary mapping evaluator names to their evaluation texts (rankings and reasoning). |
| `final_response` | `str` | The Chairman's synthesized final answer combining all perspectives. |
| `anonymous_mapping` | `Dict[str, str]` | Mapping from anonymous IDs (A, B, C, D) to member names for reference. |

#### Description

Executes the complete LLM Council workflow:

1. **Dispatch Phase**: Sends the query to all council members in parallel using `run_agents_concurrently`
2. **Collection Phase**: Collects all responses and maps them to member names
3. **Anonymization Phase**: Creates anonymous IDs (A, B, C, D, etc.) and shuffles them to ensure anonymity
4. **Evaluation Phase**: Each member evaluates and ranks all anonymized responses using `batched_grid_agent_execution`
5. **Synthesis Phase**: The Chairman agent synthesizes all responses and evaluations into a final comprehensive answer

The method provides verbose output by default, showing progress at each stage.

#### Example Usage

```python
from swarms.structs.llm_council import LLMCouncil

council = LLMCouncil(verbose=True)

query = "What are the top five best energy stocks across nuclear, solar, gas, and other energy sources?"

result = council.run(query)

# Access the final synthesized response
print(result["final_response"])

# Access individual member responses
for name, response in result["original_responses"].items():
    print(f"{name}: {response[:200]}...")

# Access evaluation rankings
for evaluator, evaluation in result["evaluations"].items():
    print(f"{evaluator} evaluation:\n{evaluation[:300]}...")

# Check anonymous mapping
print("Anonymous IDs:", result["anonymous_mapping"])
```

---

### `_create_default_council`

Creates default council members with specialized prompts and models.

#### Parameters

None (internal method).

#### Returns

| Type | Description |
|------|-------------|
| `List[Agent]` | List of Agent instances configured as council members. |

#### Description

Internal method that creates the default council configuration with four specialized agents:

- **GPT-5.1-Councilor** (`model_name="gpt-5.1"`): Analytical and comprehensive, temperature=0.7
- **Gemini-3-Pro-Councilor** (`model_name="gemini-2.5-flash"`): Concise and structured, temperature=0.7
- **Claude-Sonnet-4.5-Councilor** (`model_name="anthropic/claude-sonnet-4-5"`): Thoughtful and balanced, temperature=0.0
- **Grok-4-Councilor** (`model_name="x-ai/grok-4"`): Creative and innovative, temperature=0.8

Each agent is configured with:
- Specialized system prompts matching their role
- `max_loops=1` for single-response generation
- `verbose=False` to reduce noise during parallel execution
- Appropriate temperature settings for their style

---

## Helper Functions

### `get_gpt_councilor_prompt()`

Returns the system prompt for GPT-5.1 councilor agent.

#### Returns

| Type | Description |
|------|-------------|
| `str` | System prompt string emphasizing analytical thinking and comprehensive coverage. |

---

### `get_gemini_councilor_prompt()`

Returns the system prompt for Gemini 3 Pro councilor agent.

#### Returns

| Type | Description |
|------|-------------|
| `str` | System prompt string emphasizing concise, well-processed, and structured responses. |

---

### `get_claude_councilor_prompt()`

Returns the system prompt for Claude Sonnet 4.5 councilor agent.

#### Returns

| Type | Description |
|------|-------------|
| `str` | System prompt string emphasizing thoughtful, balanced, and nuanced responses. |

---

### `get_grok_councilor_prompt()`

Returns the system prompt for Grok-4 councilor agent.

#### Returns

| Type | Description |
|------|-------------|
| `str` | System prompt string emphasizing creative, innovative, and unique perspectives. |

---

### `get_chairman_prompt()`

Returns the system prompt for the Chairman agent.

#### Returns

| Type | Description |
|------|-------------|
| `str` | System prompt string for synthesizing responses and evaluations into a final answer. |

---

### `get_evaluation_prompt(query, responses, evaluator_name)`

Creates evaluation prompt for council members to review and rank responses.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | The original user query. |
| `responses` | `Dict[str, str]` | Dictionary mapping anonymous IDs to response texts. |
| `evaluator_name` | `str` | Name of the agent doing the evaluation. |

#### Returns

| Type | Description |
|------|-------------|
| `str` | Formatted evaluation prompt string with instructions for ranking responses. |

---

### `get_synthesis_prompt(query, original_responses, evaluations, id_to_member)`

Creates synthesis prompt for the Chairman.

#### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `query` | `str` | Original user query. |
| `original_responses` | `Dict[str, str]` | Dictionary mapping member names to their responses. |
| `evaluations` | `Dict[str, str]` | Dictionary mapping evaluator names to their evaluation texts. |
| `id_to_member` | `Dict[str, str]` | Mapping from anonymous IDs to member names. |

#### Returns

| Type | Description |
|------|-------------|
| `str` | Formatted synthesis prompt for the Chairman agent. |

---

## Use Cases

The LLM Council is ideal for scenarios requiring:

- **Multi-perspective Analysis**: When you need diverse viewpoints on complex topics
- **Quality Assurance**: When peer review and ranking can improve response quality
- **Transparent Decision Making**: When you want to see how different models evaluate each other
- **Synthesis of Expertise**: When combining multiple specialized perspectives is valuable

### Common Applications

- **Medical Diagnosis**: Multiple medical AI agents provide diagnoses, evaluate each other, and synthesize recommendations
- **Financial Analysis**: Different financial experts analyze investments and rank each other's assessments
- **Legal Analysis**: Multiple legal perspectives evaluate compliance and risk
- **Business Strategy**: Diverse strategic viewpoints are synthesized into comprehensive plans
- **Research Analysis**: Multiple research perspectives are combined for thorough analysis

## Examples

For comprehensive examples demonstrating various use cases, see the [LLM Council Examples](../../../examples/multi_agent/llm_council_examples/) directory:

- **Medical**: `medical_diagnosis_council.py`, `medical_treatment_council.py`
- **Finance**: `finance_analysis_council.py`, `etf_stock_analysis_council.py`
- **Business**: `business_strategy_council.py`, `marketing_strategy_council.py`
- **Technology**: `technology_assessment_council.py`, `research_analysis_council.py`
- **Legal**: `legal_analysis_council.py`

### Quick Start Example

```python
from swarms.structs.llm_council import LLMCouncil

# Create the council
council = LLMCouncil(verbose=True)

# Example query
query = "What are the top five best energy stocks across nuclear, solar, gas, and other energy sources?"

# Run the council
result = council.run(query)

# Print final response
print(result["final_response"])

# Optionally print evaluations
print("\n\n" + "="*80)
print("EVALUATIONS")
print("="*80)
for name, evaluation in result["evaluations"].items():
    print(f"\n{name}:")
    print(evaluation[:500] + "..." if len(evaluation) > 500 else evaluation)
```

## Customization

### Creating Custom Council Members

You can create custom council members with specialized roles:

```python
from swarms import Agent
from swarms.structs.llm_council import LLMCouncil, get_gpt_councilor_prompt

# Create custom councilor
custom_agent = Agent(
    agent_name="Domain-Expert-Councilor",
    agent_description="Specialized domain expert for specific analysis",
    system_prompt=get_gpt_councilor_prompt(),  # Or create custom prompt
    model_name="gpt-4",
    max_loops=1,
    verbose=False,
    temperature=0.7,
)

# Create council with custom members
council = LLMCouncil(
    council_members=[custom_agent, ...],  # Add your custom agents
    chairman_model="gpt-4",
    verbose=True
)
```

### Custom Chairman Model

You can specify a different model for the Chairman:

```python
council = LLMCouncil(
    chairman_model="claude-3-opus",  # Use Claude as Chairman
    verbose=True
)
```

## Architecture Benefits

1. **Diversity**: Multiple models provide varied perspectives and approaches
2. **Quality Control**: Peer review ensures responses are evaluated objectively
3. **Synthesis**: Chairman combines the best elements from all responses
4. **Transparency**: Full visibility into individual responses and evaluation rankings
5. **Scalability**: Easy to add or remove council members
6. **Flexibility**: Supports custom agents and models

## Performance Considerations

- **Parallel Execution**: Both response generation and evaluation phases run in parallel for efficiency
- **Anonymization**: Responses are anonymized to prevent bias in evaluation
- **Model Selection**: Different models can be used for different roles based on their strengths
- **Verbose Mode**: Can be disabled for production use to reduce output

## Related Documentation

- [Multi-Agent Architectures Overview](overview.md)
- [Council of Judges](council_of_judges.md) - Similar peer review pattern
- [Agent Class Reference](agent.md) - Understanding individual agents
- [Multi-Agent Execution Utilities](various_execution_methods.md) - Underlying execution methods

