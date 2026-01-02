# SelfMoASeq (Self-MoA-Seq: Sequential Self-Mixture of Agents)

**Purpose**: Ensemble method that generates multiple candidate responses from a single high-performing model and synthesizes them sequentially using a sliding window approach. This keeps context within bounds while leveraging diversity across samples for a high-quality final output.

- **Phase 1**: Generate `num_samples` responses using a proposer agent.

- **Phase 2**: Aggregate responses in windows with an aggregator agent, biasing toward the current best.

- **Phase 3**: Iterate until all samples are processed or `max_loops` is reached.

---

## Example

The snippet below shows how to construct and run `SelfMoASeq` on a task.

```python
from swarms.structs.self_moa_seq import SelfMoASeq

# Initialize
moa_seq = SelfMoASeq(
    model_name="gpt-4o-mini",
    temperature=0.7,
    window_size=6,
    verbose=True,
    num_samples=4,
)

# Run
task = (
    "Describe an effective treatment plan for a patient with a broken rib. "
    "Include immediate care, pain management, expected recovery timeline, and potential complications to watch for."
)

result = moa_seq.run(task)
print(result)
```

---

## Constructor: __init__

Create a new `SelfMoASeq` instance.

### Parameters

| Name | Type | Default | Required | Description |
|---|---|---|---|---|
| name | str | "SelfMoASeq" | No | Human-readable name for this orchestrator. |
| description | str | "Self-MoA-Seq: Sequential Self-Mixture of Agents" | No | Short description of the orchestrator. |
| model_name | str | "gpt-4o-mini" | No | Base model used when specific proposer/aggregator models are not provided. |
| temperature | float | 0.7 | No | Sampling temperature for the proposer; must be in [0, 2]. |
| window_size | int | 6 | No | Total window size used during aggregation. Must be ≥ 2. |
| reserved_slots | int | 3 | No | Number of slots reserved for the current best (and possibly other fixed items) in the window. Must be < `window_size`. |
| max_loops | int | 10 | No | Maximum aggregation loops. Must be ≥ 1. |
| max_tokens | int | 2000 | No | Token budget to consider for downstream consumers. Not enforced internally. |
| num_samples | int | 30 | No | Number of candidate responses to generate overall; must be ≥ 2. |
| enable_logging | bool | True | No | Enable internal logging via `loguru`. |
| log_level | str | "INFO" | No | Log level string. |
| verbose | bool | True | No | If True, prints a run summary after completion. |
| proposer_model_name | Optional[str] | None | No | Overrides the model for the proposer agent; falls back to `model_name` if not provided. |
| aggregator_model_name | Optional[str] | None | No | Overrides the model for the aggregator agent; falls back to `model_name` if not provided. |
| max_retries | int | 3 | No | Max retry attempts for key operations (`_generate_samples`, `_aggregate_window`, `run`). Must be ≥ 0. |
| retry_delay | float | 1.0 | No | Initial delay before first retry (seconds). Must be ≥ 0. |
| retry_backoff_multiplier | float | 2.0 | No | Exponential backoff multiplier; must be ≥ 1. |
| retry_max_delay | float | 60.0 | No | Maximum backoff delay (seconds); must be ≥ `retry_delay`. |

### Raises

- `ValueError` for invalid parameter ranges (e.g., `window_size < 2`, `reserved_slots >= window_size`, temperature outside [0, 2], etc.).

---

## Method Overview

- **_generate_samples(task, num_samples) -> List[str]**: Generate multiple samples via the proposer agent with retries.
- **_format_aggregation_prompt(task, samples, best_so_far) -> str**: Build the aggregation prompt including current best and candidate responses.
- **_aggregate_window(task, window_samples, best_so_far) -> str**: Aggregate a window of samples with the aggregator agent, returning a synthesized output.
- **run(task) -> Dict[str, Any]**: Orchestrates the full Self-MoA-Seq process and returns the final result bundle.
- **_log_summary(result) -> None**: If `verbose`, logs a summary of the run (metrics, timings, output length).
- **get_metrics() -> Dict[str, Any]**: Returns a snapshot of internal metrics counters.

Note: Methods prefixed with `_` are internal but documented here for completeness.

---

## _generate_samples

Generate `num_samples` candidate responses using the proposer agent.

**Signature**: `_generate_samples(task: str, num_samples: int) -> List[str]`

### Parameters

| Name | Type | Required | Description |
|---|---|---|---|
| task | str | Yes | The task description to pass to the proposer agent. |
| num_samples | int | Yes | Number of samples to generate. |

### Returns

| Type | Description |
|---|---|
| List[str] | The generated samples in generation order. |

### Raises

- Propagates any exceptions from the underlying agent call; retried according to the instance retry configuration.

---

## _format_aggregation_prompt

Create the prompt that the aggregator agent will receive for a given window.

**Signature**: `_format_aggregation_prompt(task: str, samples: List[str], best_so_far: Optional[str] = None) -> str`

### Parameters

| Name | Type | Required | Description |
|---|---|---|---|
| task | str | Yes | The original task string. |
| samples | List[str] | Yes | Window of candidate responses to synthesize. |
| best_so_far | Optional[str] | No | Previously synthesized best output, if any. |

### Returns

| Type | Description |
|---|---|
| str | Aggregation prompt text to be sent to the aggregator agent. |

---

## _aggregate_window

Aggregate a window of samples using the aggregator agent, biased by `best_so_far`.

**Signature**: `_aggregate_window(task: str, window_samples: List[str], best_so_far: Optional[str] = None) -> str`

### Parameters

| Name | Type | Required | Description |
|---|---|---|---|
| task | str | Yes | The original task string. |
| window_samples | List[str] | Yes | Current window: typically `[best_output] + current_window`. |
| best_so_far | Optional[str] | No | Current best aggregation to bias the synthesizer. |

### Returns

| Type | Description |
|---|---|
| str | The synthesized output for this window. |

### Raises

- Propagates any exceptions from the underlying agent call; retried according to the instance retry configuration.

---

## run

Execute the full Self-MoA-Seq process: sample generation, sliding-window aggregation, and final synthesis.

**Signature**: `run(task: str) -> Dict[str, Any]`

### Parameters

| Name | Type | Required | Description |
|---|---|---|---|
| task | str | Yes | The task to process; must be a non-empty string. |

### Returns

| Key | Type | Description |
|---|---|---|
| final_output | str | The final synthesized best response. |
| all_samples | List[str] | All generated candidate responses. |
| aggregation_steps | int | Number of aggregation iterations executed. |
| metrics | Dict[str, Any] | Snapshot of performance metrics for this run. |
| task | str | Echoes the original task. |
| timestamp | str | ISO 8601 timestamp of completion. |

### Raises

- `ValueError` if `task` is not a non-empty string.
- Propagates any exceptions from generation/aggregation; both are retried according to the instance retry configuration.

---

## _log_summary

Log a brief summary of the run when `verbose=True`.

**Signature**: `_log_summary(result: Dict[str, Any]) -> None`

### Parameters

| Name | Type | Required | Description |
|---|---|---|---|
| result | Dict[str, Any] | Yes | The run result bundle returned by `run`. |

### Returns

| Type | Description |
|---|---|
| None | This method logs via `loguru` and returns nothing. |

---

## get_metrics

Get a snapshot of the internal metrics counters.

**Signature**: `get_metrics() -> Dict[str, Any]`

### Parameters

None

### Returns

| Type | Description |
|---|---|
| Dict[str, Any] | A copy of the current metrics dictionary. |

---


## Examples

### Example 1: Medical Diagnosis with High Sample Count

This example demonstrates using SelfMoASeq for a complex medical diagnosis task with a larger number of samples for comprehensive analysis.

```python
from swarms.structs.self_moa_seq import SelfMoASeq

# Initialize with medical-focused configuration
medical_moa = SelfMoASeq(
    model_name="gpt-4o",
    temperature=0.8,  # Higher creativity for diverse medical perspectives
    window_size=8,    # Larger window for complex medical reasoning
    num_samples=12,   # More samples for comprehensive analysis
    max_loops=15,
    verbose=True,
    proposer_model_name="gpt-4o",  # Use same model for consistency
    aggregator_model_name="gpt-4o"
)

# Complex medical case
medical_case = """
Patient: 45-year-old female
Symptoms: 
- Chest pain for 3 days, worse with deep breathing
- Shortness of breath
- Low-grade fever (100.2°F)
- Recent travel to Southeast Asia
- History of smoking (quit 5 years ago)

Vital signs: BP 140/90, HR 95, RR 22, O2 sat 94% on room air
Physical exam: Decreased breath sounds in right lower lobe, no JVD, no peripheral edema

Lab results pending: CBC, CMP, D-dimer, troponin
Chest X-ray: Small pleural effusion on right side

Provide a comprehensive differential diagnosis, immediate management plan, 
and follow-up recommendations.
"""

result = medical_moa.run(medical_case)
print("Medical Diagnosis Analysis:")
print("=" * 50)
print(result["final_output"])
print(f"\nGenerated {len(result['all_samples'])} samples in {result['aggregation_steps']} iterations")
print(f"Execution time: {result['metrics']['execution_time_seconds']:.2f} seconds")
```

### Example 2: Creative Writing with Different Models

This example shows how to use different models for the proposer and aggregator, with a creative writing task.

```python
from swarms.structs.self_moa_seq import SelfMoASeq

# Initialize with different models for proposer and aggregator
creative_moa = SelfMoASeq(
    model_name="gpt-4o-mini",  # Base model (fallback)
    temperature=1.2,  # High creativity for writing
    window_size=4,    # Smaller window for focused synthesis
    num_samples=6,    # Moderate number of samples
    max_loops=8,
    verbose=True,
    proposer_model_name="gpt-4o",      # Creative model for generation
    aggregator_model_name="gpt-4o-mini", # Efficient model for synthesis
    max_retries=2,
    retry_delay=0.5
)

# Creative writing prompt
writing_prompt = """
Write a compelling opening chapter for a science fiction novel set in 2150. 
The story should involve:
- A protagonist who discovers they can manipulate time in small ways
- A mysterious organization that's been monitoring them
- A world where AI and humans coexist but tensions are rising
- An unexpected twist that challenges the reader's assumptions

Focus on:
- Strong character development
- Immersive world-building
- A hook that makes readers want to continue
- Balance between action and character moments
"""

result = creative_moa.run(writing_prompt)
print("Creative Writing Result:")
print("=" * 40)
print(result["final_output"])
print(f"\nWriting samples generated: {len(result['all_samples'])}")
print(f"Synthesis iterations: {result['aggregation_steps']}")
```


## Retry Behavior

Key methods (`_generate_samples`, `_aggregate_window`, and `run`) are wrapped with an instance-configurable retry policy:

- **stop**: Up to `max_retries + 1` total attempts.
- **wait**: Exponential backoff with `retry_backoff_multiplier`, min `retry_delay`, and max `retry_max_delay` seconds.
- **retry on**: Any `Exception`.
- **before sleep**: Logs a warning before each retry.

---

## Conclusion

`SelfMoASeq` represents a powerful approach to ensemble-based text generation that addresses the fundamental challenge of context length limitations while maximizing the benefits of model diversity. By generating multiple candidate responses and synthesizing them through a sliding window approach, this method produces high-quality outputs that often surpass single-model performance.


### When to Choose SelfMoASeq

Choose `SelfMoASeq` when you need:

- High-quality outputs that benefit from multiple perspectives

- To work within context length constraints

- Reliable, production-ready ensemble methods

- Fine-grained control over the generation and synthesis process

- Comprehensive observability and error handling

For simpler tasks or when context limits aren't a concern, consider using single-agent approaches or other ensemble methods like `MixtureOfAgents` for more straightforward aggregation strategies.
