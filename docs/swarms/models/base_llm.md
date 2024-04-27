# Language Model Interface Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Abstract Language Model](#abstract-language-model)
   - [Initialization](#initialization)
   - [Attributes](#attributes)
   - [Methods](#methods)
3. [Implementation](#implementation)
4. [Usage Examples](#usage-examples)
5. [Additional Features](#additional-features)
6. [Performance Metrics](#performance-metrics)
7. [Logging and Checkpoints](#logging-and-checkpoints)
8. [Resource Utilization Tracking](#resource-utilization-tracking)
9. [Conclusion](#conclusion)

---

## 1. Introduction <a name="introduction"></a>

The Language Model Interface (`BaseLLM`) is a flexible and extensible framework for working with various language models. This documentation provides a comprehensive guide to the interface, its attributes, methods, and usage examples. Whether you're using a pre-trained language model or building your own, this interface can help streamline the process of text generation, chatbots, summarization, and more.

## 2. Abstract Language Model <a name="abstract-language-model"></a>

### Initialization <a name="initialization"></a>

The `BaseLLM` class provides a common interface for language models. It can be initialized with various parameters to customize model behavior. Here are the initialization parameters:

| Parameter              | Description                                                                                     | Default Value |
|------------------------|-------------------------------------------------------------------------------------------------|---------------|
| `model_name`           | The name of the language model to use.                                                         | None          |
| `max_tokens`           | The maximum number of tokens in the generated text.                                              | None          |
| `temperature`          | The temperature parameter for controlling randomness in text generation.                        | None          |
| `top_k`                | The top-k parameter for filtering words in text generation.                                      | None          |
| `top_p`                | The top-p parameter for filtering words in text generation.                                      | None          |
| `system_prompt`        | A system-level prompt to set context for generation.                                             | None          |
| `beam_width`           | The beam width for beam search.                                                                 | None          |
| `num_return_sequences` | The number of sequences to return in the output.                                                 | None          |
| `seed`                 | The random seed for reproducibility.                                                            | None          |
| `frequency_penalty`    | The frequency penalty parameter for promoting word diversity.                                    | None          |
| `presence_penalty`     | The presence penalty parameter for discouraging repetitions.                                     | None          |
| `stop_token`           | A stop token to indicate the end of generated text.                                              | None          |
| `length_penalty`       | The length penalty parameter for controlling the output length.                                   | None          |
| `role`                 | The role of the language model (e.g., assistant, user, etc.).                                    | None          |
| `max_length`           | The maximum length of generated sequences.                                                       | None          |
| `do_sample`            | Whether to use sampling during text generation.                                                  | None          |
| `early_stopping`       | Whether to use early stopping during text generation.                                            | None          |
| `num_beams`            | The number of beams to use in beam search.                                                       | None          |
| `repition_penalty`     | The repetition penalty parameter for discouraging repeated tokens.                                | None          |
| `pad_token_id`         | The token ID for padding.                                                                       | None          |
| `eos_token_id`         | The token ID for the end of a sequence.                                                         | None          |
| `bos_token_id`         | The token ID for the beginning of a sequence.                                                   | None          |
| `device`               | The device to run the model on (e.g., 'cpu' or 'cuda').                                          | None          |

### Attributes <a name="attributes"></a>

- `model_name`: The name of the language model being used.
- `max_tokens`: The maximum number of tokens in generated text.
- `temperature`: The temperature parameter controlling randomness.
- `top_k`: The top-k parameter for word filtering.
- `top_p`: The top-p parameter for word filtering.
- `system_prompt`: A system-level prompt for context.
- `beam_width`: The beam width for beam search.
- `num_return_sequences`: The number of output sequences.
- `seed`: The random seed for reproducibility.
- `frequency_penalty`: The frequency penalty parameter.
- `presence_penalty`: The presence penalty parameter.
- `stop_token`: The stop token to indicate text end.
- `length_penalty`: The length penalty parameter.
- `role`: The role of the language model.
- `max_length`: The maximum length of generated sequences.
- `do_sample`: Whether to use sampling during generation.
- `early_stopping`: Whether to use early stopping.
- `num_beams`: The number of beams in beam search.
- `repition_penalty`: The repetition penalty parameter.
- `pad_token_id`: The token ID for padding.
- `eos_token_id`: The token ID for the end of a sequence.
- `bos_token_id`: The token ID for the beginning of a sequence.
- `device`: The device used for model execution.
- `history`: A list of conversation history.

### Methods <a name="methods"></a>

The `BaseLLM` class defines several methods for working with language models:

- `run(task: Optional[str] = None, *args, **kwargs) -> str`: Generate text using the language model. This method is abstract and must be implemented by subclasses.

- `arun(task: Optional[str] = None, *args, **kwargs)`: An asynchronous version of `run` for concurrent text generation.

- `batch_run(tasks: List[str], *args, **kwargs)`: Generate text for a batch of tasks.

- `abatch_run(tasks: List[str], *args, **kwargs)`: An asynchronous version of `batch_run` for concurrent batch generation.

- `chat(task: str, history: str = "") -> str`: Conduct a chat with the model, providing a conversation history.

- `__call__(task: str) -> str`: Call the model to generate text.

- `_tokens_per_second() -> float`: Calculate tokens generated per second.

- `_num_tokens(text: str) -> int`: Calculate the number of tokens in a text.

- `_time_for_generation(task: str) -> float`: Measure the time taken for text generation.

- `generate_summary(text: str) -> str`: Generate a summary of the provided text.

- `set_temperature(value: float)`: Set the temperature parameter.

- `set_max_tokens(value: int)`: Set the maximum number of tokens.

- `clear_history()`: Clear the conversation history.

- `enable_logging(log_file: str = "model.log")`: Initialize logging for the model.

- `log_event(message: str)`: Log an event.

- `save_checkpoint(checkpoint_dir: str = "checkpoints")`: Save the model state as a checkpoint.

- `load_checkpoint(checkpoint_path: str)`: Load the model state from a checkpoint.

- `toggle_creative_mode(enable: bool)`: Toggle creative mode for the model.

- `track_resource_utilization()`: Track and report resource utilization.

- `

get_generation_time() -> float`: Get the time taken for text generation.

- `set_max_length(max_length: int)`: Set the maximum length of generated sequences.

- `set_model_name(model_name: str)`: Set the model name.

- `set_frequency_penalty(frequency_penalty: float)`: Set the frequency penalty parameter.

- `set_presence_penalty(presence_penalty: float)`: Set the presence penalty parameter.

- `set_stop_token(stop_token: str)`: Set the stop token.

- `set_length_penalty(length_penalty: float)`: Set the length penalty parameter.

- `set_role(role: str)`: Set the role of the model.

- `set_top_k(top_k: int)`: Set the top-k parameter.

- `set_top_p(top_p: float)`: Set the top-p parameter.

- `set_num_beams(num_beams: int)`: Set the number of beams.

- `set_do_sample(do_sample: bool)`: Set whether to use sampling.

- `set_early_stopping(early_stopping: bool)`: Set whether to use early stopping.

- `set_seed(seed: int)`: Set the random seed.

- `set_device(device: str)`: Set the device for model execution.

## 3. Implementation <a name="implementation"></a>

The `BaseLLM` class serves as the base for implementing specific language models. Subclasses of `BaseLLM` should implement the `run` method to define how text is generated for a given task. This design allows flexibility in integrating different language models while maintaining a common interface.

## 4. Usage Examples <a name="usage-examples"></a>

To demonstrate how to use the `BaseLLM` interface, let's create an example using a hypothetical language model. We'll initialize an instance of the model and generate text for a simple task.

```python
# Import the BaseLLM class
from swarms.models import BaseLLM

# Create an instance of the language model
language_model = BaseLLM(
    model_name="my_language_model",
    max_tokens=50,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    device="cuda",
)

# Generate text for a task
task = "Translate the following English text to French: 'Hello, world.'"
generated_text = language_model.run(task)

# Print the generated text
print(generated_text)
```

In this example, we've created an instance of our hypothetical language model, configured its parameters, and used the `run` method to generate text for a translation task.

## 5. Additional Features <a name="additional-features"></a>

The `BaseLLM` interface provides additional features for customization and control:

- `batch_run`: Generate text for a batch of tasks efficiently.
- `arun` and `abatch_run`: Asynchronous versions of `run` and `batch_run` for concurrent text generation.
- `chat`: Conduct a conversation with the model by providing a history of the conversation.
- `__call__`: Allow the model to be called directly to generate text.

These features enhance the flexibility and utility of the interface in various applications, including chatbots, language translation, and content generation.

## 6. Performance Metrics <a name="performance-metrics"></a>

The `BaseLLM` class offers methods for tracking performance metrics:

- `_tokens_per_second`: Calculate tokens generated per second.
- `_num_tokens`: Calculate the number of tokens in a text.
- `_time_for_generation`: Measure the time taken for text generation.

These metrics help assess the efficiency and speed of text generation, enabling optimizations as needed.

## 7. Logging and Checkpoints <a name="logging-and-checkpoints"></a>

Logging and checkpointing are crucial for tracking model behavior and ensuring reproducibility:

- `enable_logging`: Initialize logging for the model.
- `log_event`: Log events and activities.
- `save_checkpoint`: Save the model state as a checkpoint.
- `load_checkpoint`: Load the model state from a checkpoint.

These capabilities aid in debugging, monitoring, and resuming model experiments.

## 8. Resource Utilization Tracking <a name="resource-utilization-tracking"></a>

The `track_resource_utilization` method is a placeholder for tracking and reporting resource utilization, such as CPU and memory usage. It can be customized to suit specific monitoring needs.

## 9. Conclusion <a name="conclusion"></a>

The Language Model Interface (`BaseLLM`) is a versatile framework for working with language models. Whether you're using pre-trained models or developing your own, this interface provides a consistent and extensible foundation. By following the provided guidelines and examples, you can integrate and customize language models for various natural language processing tasks.