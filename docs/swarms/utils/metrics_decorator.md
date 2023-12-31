# metrics_decorator

This documentation explains the use and functionality of the `metrics_decorator` function in the LLM (Large Language Models). 

The `metrics_decorator` function is a standard Python decorator that augments a specific function by wrapping extra functionality around it. It is commonly used for things like timing, logging or memoization. 
-- 
The `metrics_decorator` in LLM is specially designed to measure and calculate three key performance metrics when generating language models:

1. `Time to First Token`: Measures the elapsed time from the start of function execution until the generation of the first token. 
2. `Generation Latency`: It measures the total time taken for a complete run.
3. `Throughput`: Calculates the rate of production of tokens per unit of time.

```python
def metrics_decorator(func: Callable):
    """

    Metrics decorator for LLM

    Args:
        func (Callable): The function to be decorated.

    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        """
        An inner function that wraps the decorated function. It calculates 'Time to First Token',
        'Generation Latency' and 'Throughput' metrics.

        Args:
            self : The object instance.
            *args : Variable length argument list of the decorated function.
            **kwargs : Arbitrary keyword arguments of the decorated function.
        """

        # Measure Time to First Token
        start_time = time.time()
        result = func(self, *args, **kwargs)
        first_token_time = time.time()

        # Measure Generation Latency
        end_time = time.time()

        # Calculate Throughput (assuming the function returns a list of tokens)
        throughput = len(result) / (end_time - start_time)

        return f"""
                 Time to First Token: {first_token_time - start_time}
                 Generation Latency: {end_time - start_time}
                 Throughput: {throughput}
               """

    return wrapper
```
## Example Usage
Now let's discuss the usage of the `metrics_decorator` function with an example.

Assuming that we have a language generation function called `text_generator()` that generates a list of tokens.

```python
@metrics_decorator
def text_generator(self, text: str):
    """
    Args:
        text (str): The input text.

    Returns:
        A list of tokens generated from the input text.
    """
    # language generation implementation goes here
    return tokens

# Instantiate the class and call the decorated function
obj = ClassName()
obj.text_generator("Hello, world!")
```

When the decorated `text_generator()` function is called, it will measure and return:

- Time elapsed until the first token is generated.
- The total execution time of the function.
- The rate of tokens generation per unit time.

This example provides a basic overview of how a function can be decorated with the `metrics_decorator`. The provided `func` argument could be any method from any class, as long as it complies with the structure defined in `metrics_decorator`. It is worth noting that the decorated function must return a list of tokens for the `Throughput` metric to work correctly.

Remember, applying the `metrics_decorator` does not affect the original functionality of the decorated function, it just adds additional measurement and logging capabilities to it. It's a great utility for tracking and optimizing the performance of your language models.
