# Self-Healing Agent Tutorial

This tutorial will guide you through using the Self-Healing Agent to automatically detect and fix runtime errors in your Python code.

## Introduction

The Self-Healing Agent is a powerful tool that uses LLM-based analysis to:
- Detect runtime errors
- Analyze their root causes
- Generate and apply fixes automatically
- Provide detailed error analysis in structured JSON format

## Prerequisites

- Python 3.7+
- swarms package installed (`pip install -U swarms`)
- OpenAI API key set in environment variables

## Basic Setup

First, let's set up a basic project with some error-prone code:

```python
# error_prone.py
def divide_numbers(a, b):
    return a / b

def access_nested_dict(data):
    return data['user']['name']

def process_list(items):
    return [item.upper() for item in items]
```

Now let's add error handling with the Self-Healing Agent:

```python
# main.py
from swarms.agents import SelfHealingAgent
from error_prone import divide_numbers, access_nested_dict, process_list

# Initialize the agent
agent = SelfHealingAgent(
    model_name="gpt-4o-mini",
    max_retries=3,
    verbose=True
)

# Test cases that will cause errors
def run_tests():
    test_cases = [
        lambda: divide_numbers(10, 0),
        lambda: access_nested_dict({}),
        lambda: process_list([1, 2, "three"])
    ]
    
    for test in test_cases:
        try:
            result = test()
            print(f"Success: {result}")
        except Exception as e:
            print(f"\nError caught: {type(e).__name__}")
            fix = agent.analyze_and_fix(e)
            
            print("\nError Analysis:")
            print(f"Type: {fix['error_type']}")
            print(f"Analysis: {fix['analysis']}")
            print(f"Proposed Fix: {fix['solution']}")
            print(f"Confidence: {fix['confidence']}")
            
            if fix['confidence'] > 0.8:
                print("\nApplying fix automatically...")
                success = agent.apply_fix(fix)
                if success:
                    print("Fix applied successfully!")
            else:
                print("\nLow confidence fix - manual review needed")

if __name__ == "__main__":
    run_tests()
```

## Step-by-Step Examples

### 1. Handling Division by Zero

```python
def safe_division():
    try:
        result = divide_numbers(10, 0)
    except Exception as e:
        fix = agent.analyze_and_fix(e)
        print(f"Original error: {e}")
        print(f"Fix suggestion: {fix['solution']}")
        
        # The agent might suggest something like:
        """
        def divide_numbers(a, b):
            if b == 0:
                raise ValueError("Cannot divide by zero")
            return a / b
        """
```

### 2. Handling Dictionary Access

```python
def safe_dict_access():
    data = {}
    try:
        name = access_nested_dict(data)
    except Exception as e:
        fix = agent.analyze_and_fix(e)
        print(f"Original error: {e}")
        print(f"Fix suggestion: {fix['solution']}")
        
        # The agent might suggest something like:
        """
        def access_nested_dict(data):
            return data.get('user', {}).get('name', None)
        """
```

### 3. Type Error Handling

```python
def safe_list_processing():
    items = [1, 2, "three"]
    try:
        result = process_list(items)
    except Exception as e:
        fix = agent.analyze_and_fix(e)
        print(f"Original error: {e}")
        print(f"Fix suggestion: {fix['solution']}")
        
        # The agent might suggest something like:
        """
        def process_list(items):
            return [str(item).upper() for item in items]
        """
```

## Advanced Usage

### Custom Error Handlers

```python
class CustomErrorHandler:
    def __init__(self, agent):
        self.agent = agent
        self.fix_history = []
    
    def handle_error(self, error, context=None):
        fix = self.agent.analyze_and_fix(error)
        self.fix_history.append(fix)
        
        if fix['confidence'] > 0.9:
            return self.agent.apply_fix(fix)
        elif fix['confidence'] > 0.7:
            return self.request_human_review(fix)
        else:
            return self.fallback_handler(error)
    
    def request_human_review(self, fix):
        print("Please review the following fix:")
        print(f"Error: {fix['error_type']}")
        print(f"Solution: {fix['solution']}")
        response = input("Apply fix? (y/n): ")
        return response.lower() == 'y'
    
    def fallback_handler(self, error):
        print(f"Unable to fix error: {error}")
        return False

# Usage
handler = CustomErrorHandler(agent)
try:
    result = some_risky_operation()
except Exception as e:
    handler.handle_error(e)
```

### Batch Processing with Progress Tracking

```python
from tqdm import tqdm

def batch_process_with_healing(items, process_func):
    results = []
    errors = []
    fixes_applied = 0
    
    for item in tqdm(items, desc="Processing items"):
        try:
            result = process_func(item)
            results.append(result)
        except Exception as e:
            fix = agent.analyze_and_fix(e)
            if fix['confidence'] > 0.8:
                if agent.apply_fix(fix):
                    fixes_applied += 1
                    # Retry with fix applied
                    try:
                        result = process_func(item)
                        results.append(result)
                        continue
                    except Exception as retry_e:
                        errors.append((item, retry_e))
            errors.append((item, e))
    
    print(f"\nProcessing complete:")
    print(f"- Successful items: {len(results)}")
    print(f"- Failed items: {len(errors)}")
    print(f"- Fixes applied: {fixes_applied}")
    
    return results, errors
```

## Best Practices

1. **Error Context**
   ```python
   try:
       result = risky_operation()
   except Exception as e:
       # Provide additional context
       fix = agent.analyze_and_fix(e, context={
           'function': 'risky_operation',
           'input_data': str(input_data)[:100],  # First 100 chars
           'expected_output': 'list of strings'
       })
   ```

2. **Confidence Thresholds**
   ```python
   def apply_fix_with_threshold(fix, threshold=0.8):
       if fix['confidence'] >= threshold:
           return agent.apply_fix(fix)
       return False
   ```

3. **Logging and Monitoring**
   ```python
   import logging

   logging.basicConfig(level=logging.INFO)
   logger = logging.getLogger('self_healing')

   def monitored_execution(func):
       try:
           return func()
       except Exception as e:
           logger.error(f"Error in {func.__name__}: {e}")
           fix = agent.analyze_and_fix(e)
           logger.info(f"Fix generated: {fix['solution']}")
           if agent.apply_fix(fix):
               logger.info("Fix applied successfully")
               return func()  # Retry
           logger.warning("Fix application failed")
           raise
   ```

## Common Pitfalls

1. **Avoid Infinite Loops**
   ```python
   max_retries = 3
   retry_count = 0
   
   while retry_count < max_retries:
       try:
           result = risky_operation()
           break
       except Exception as e:
           retry_count += 1
           fix = agent.analyze_and_fix(e)
           if not agent.apply_fix(fix):
               break
   ```

2. **Handle Nested Errors**
   ```python
   def safe_execute(func):
       try:
           return func()
       except Exception as outer_e:
           try:
               fix = agent.analyze_and_fix(outer_e)
           except Exception as inner_e:
               logger.error(f"Error in error handler: {inner_e}")
               raise outer_e
   ```

3. **Resource Cleanup**
   ```python
   def safe_file_operation(filename):
       file = None
       try:
           file = open(filename, 'r')
           return process_file(file)
       except Exception as e:
           fix = agent.analyze_and_fix(e)
           # Handle fix
       finally:
           if file:
               file.close()
   ```

## Conclusion

The Self-Healing Agent is a powerful tool for automating error handling and fixes in your Python code. By following this tutorial and best practices, you can significantly reduce the time spent debugging common runtime errors and improve your code's resilience.

Remember to:
- Always validate fixes before applying them in production
- Set appropriate confidence thresholds
- Maintain good logging and monitoring
- Handle edge cases and cleanup properly

For more information, see the [API Documentation](../swarms/agents/self_healing_agent.md). 