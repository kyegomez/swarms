# Models Documentation

====================

## Language Models

---------------

Language models are the driving force of our agents. They are responsible for generating text based on a given prompt. We currently support two types of language models: Anthropic and HuggingFace.

### Anthropic

The `Anthropic` class is a wrapper for the Anthropic large language models.

#### Initialization

```

Anthropic(model="claude-2", max_tokens_to_sample=256, temperature=None, top_k=None, top_p=None, streaming=False, default_request_timeout=None)

```



##### Parameters

- `model` (str, optional): The name of the model to use. Default is "claude-2".

- `max_tokens_to_sample` (int, optional): The maximum number of tokens to sample. Default is 256.

- `temperature` (float, optional): The temperature to use for the generation. Higher values result in more random outputs.

- `top_k` (int, optional): The number of top tokens to consider for the generation.

- `top_p` (float, optional): The cumulative probability of parameter highest probability vocabulary tokens to keep for nucleus sampling.

- `streaming` (bool, optional): Whether to use streaming mode. Default is False.

- `default_request_timeout` (int, optional): The default request timeout in seconds. Default is 600.

##### Example

```

anthropic = Anthropic(model="claude-2", max_tokens_to_sample=100, temperature=0.8)

```



#### Generation

```

anthropic.generate(prompt, stop=None)

```



##### Parameters

- `prompt` (str): The prompt to use for the generation.

- `stop` (list, optional): A list of stop sequences. The generation will stop if one of these sequences is encountered.

##### Returns

- `str`: The generated text.

##### Example

```

prompt = "Once upon a time"

stop = ["The end"]

print(anthropic.generate(prompt, stop))

```



### HuggingFaceLLM

The `HuggingFaceLLM` class is a wrapper for the HuggingFace language models.

#### Initialization

```

HuggingFaceLLM(model_id: str, device: str = None, max_length: int = 20, quantize: bool = False, quantization_config: dict = None)

```



##### Parameters

- `model_id` (str): The ID of the model to use.

- `device` (str, optional): The device to use for the generation. Default is "cuda" if available, otherwise "cpu".

- `max_length` (int, optional): The maximum length of the generated text. Default is 20.

- `quantize` (bool, optional): Whether to quantize the model. Default is False.

- `quantization_config` (dict, optional): The configuration for the quantization.

##### Example

```

huggingface = HuggingFaceLLM(model_id="gpt2", device="cpu", max_length=50)

```



#### Generation

```

huggingface.generate(prompt_text: str, max_length: int = None)

```



##### Parameters

- `prompt_text` (str): The prompt to use for the generation.

- `max_length` (int, optional): The maximum length of the generated text. If not provided, the default value specified during initialization is used.

##### Returns

- `str`: The generated text.

##### Example

```

prompt = "Once upon a time"

print(huggingface.generate(prompt))

```


### Full Examples

```python
# Import the necessary classes

from swarms.models import Anthropic, HuggingFaceLLM

# Create an instance of the Anthropic class

anthropic = Anthropic(model="claude-2", max_tokens_to_sample=100, temperature=0.8)

# Use the Anthropic instance to generate text

prompt = "Once upon a time"

stop = ["The end"]

print("Anthropic output:")

print(anthropic.generate(prompt, stop))

# Create an instance of the HuggingFaceLLM class

huggingface = HuggingFaceLLM(model_id="gpt2", device="cpu", max_length=50)

# Use the HuggingFaceLLM instance to generate text

prompt = "Once upon a time"

print("\nHuggingFaceLLM output:")

print(huggingface.generate(prompt))

```

