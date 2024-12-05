# Self-Healing Agent Documentation

The Self-Healing Agent is a specialized agent designed to automatically detect, analyze, and fix runtime errors in your code using LLM-based analysis. It provides structured error analysis and generates fixes in JSON format.

## Overview

The Self-Healing Agent uses advanced language models to:
1. Analyze runtime errors and exceptions
2. Understand the error context and root cause
3. Generate potential fixes in a structured format
4. Apply fixes automatically when possible

## Installation

The Self-Healing Agent is included in the main swarms package:

```bash
pip install -U swarms
```

## Basic Usage

```python
from swarms.agents import SelfHealingAgent

# Initialize the agent
agent = SelfHealingAgent(
    model_name="gpt-4",  # The LLM model to use
    max_retries=3,       # Maximum number of fix attempts
    verbose=True         # Enable detailed logging
)

# Example usage with error handling
try:
    result = some_function_that_might_fail()
except Exception as e:
    fix = agent.analyze_and_fix(e)
    print(f"Error Analysis: {fix['analysis']}")
    print(f"Proposed Fix: {fix['solution']}")
```

## API Reference

### SelfHealingAgent Class

#### Constructor Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| model_name | str | "gpt-4" | The name of the LLM model to use |
| max_retries | int | 3 | Maximum number of fix attempts |
| verbose | bool | False | Enable detailed logging |
| system_prompt | str | None | Custom system prompt for the LLM |
| temperature | float | 0.7 | Temperature for LLM responses |

#### Methods

##### analyze_and_fix(error: Exception) -> dict
Analyzes an error and generates potential fixes.

**Parameters:**
- error (Exception): The caught exception to analyze

**Returns:**
A dictionary containing:
```json
{
    "error_type": "str",      // Type of the error
    "analysis": "str",        // Detailed error analysis
    "context": "str",         // Error context
    "solution": "str",        // Proposed fix in code form
    "confidence": float,      // Confidence score (0-1)
    "metadata": {}            // Additional error metadata
}
```

##### apply_fix(fix: dict) -> bool
Attempts to apply a generated fix.

**Parameters:**
- fix (dict): The fix dictionary returned by analyze_and_fix()

**Returns:**
- bool: True if fix was successfully applied, False otherwise

## Error Output Format

The agent provides structured error analysis in JSON format:

```json
{
    "error_type": "ZeroDivisionError",
    "analysis": "Attempted division by zero in calculation",
    "context": "Error occurred in calculate_average() at line 45",
    "solution": "Add a check for zero denominator:\nif denominator != 0:\n    result = numerator/denominator\nelse:\n    raise ValueError('Denominator cannot be zero')",
    "confidence": 0.95,
    "metadata": {
        "file": "calculator.py",
        "line": 45,
        "function": "calculate_average"
    }
}
```

## Best Practices

1. **Error Context**: Always provide as much context as possible when catching errors
2. **Validation**: Review proposed fixes before applying them automatically
3. **Logging**: Enable verbose mode during development for detailed insights
4. **Model Selection**: Use GPT-4 for complex errors, GPT-3.5-turbo for simpler cases
5. **Retry Strategy**: Configure max_retries based on your use case

## Examples

### Basic Error Handling

```python
from swarms.agents import SelfHealingAgent

agent = SelfHealingAgent()

def process_data(data):
    try:
        result = data['key']['nested_key']
        return result
    except Exception as e:
        fix = agent.analyze_and_fix(e)
        if fix['confidence'] > 0.8:
            print(f"Applying fix: {fix['solution']}")
            return agent.apply_fix(fix)
        else:
            print(f"Low confidence fix, manual review needed: {fix['analysis']}")
            return None
```

### Custom System Prompt

```python
agent = SelfHealingAgent(
    system_prompt="""You are an expert Python developer specializing in
    fixing data processing errors. Focus on data validation and type checking
    in your solutions."""
)
```

### Batch Error Processing

```python
def process_batch(items):
    results = []
    errors = []
    
    for item in items:
        try:
            result = process_item(item)
            results.append(result)
        except Exception as e:
            errors.append((item, e))
    
    # Process all errors at once
    if errors:
        fixes = [agent.analyze_and_fix(e) for _, e in errors]
        return results, fixes
```

## Error Types Handled

The Self-Healing Agent can handle various types of runtime errors including:

- Syntax Errors
- Type Errors
- Index Errors
- Key Errors
- Attribute Errors
- Value Errors
- Import Errors
- And more...

## Contributing

We welcome contributions to improve the Self-Healing Agent! Please see our [Contributing Guidelines](../../CONTRIBUTING.md) for more information. 