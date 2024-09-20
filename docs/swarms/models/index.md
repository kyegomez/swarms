# Swarm Models


```bash
$ pip3 install -U swarm-models
```

Welcome to the documentation for the llm section of the swarms package, designed to facilitate seamless integration with various AI language models and APIs. This package empowers developers, end-users, and system administrators to interact with AI models from different providers, such as OpenAI, Hugging Face, Google PaLM, and Anthropic.

### Table of Contents
1. [OpenAI](#openai)
2. [HuggingFace](#huggingface)
3. [Google PaLM](#google-palm)
4. [Anthropic](#anthropic)

### 1. OpenAI (swarm_models.OpenAI)

The OpenAI class provides an interface to interact with OpenAI's language models. It allows both synchronous and asynchronous interactions.

**Constructor:**
```python
OpenAI(api_key: str, system: str = None, console: bool = True, model: str = None, params: dict = None, save_messages: bool = True)
```

**Attributes:**
- `api_key` (str): Your OpenAI API key.

- `system` (str, optional): A system message to be used in conversations.

- `console` (bool, default=True): Display console logs.

- `model` (str, optional): Name of the language model to use.

- `params` (dict, optional): Additional parameters for model interactions.

- `save_messages` (bool, default=True): Save conversation messages.

**Methods:**

- `run(message: str, **kwargs) -> str`: Generate a response using the OpenAI model.

- `generate_async(message: str, **kwargs) -> str`: Generate a response asynchronously.

- `ask_multiple(ids: List[str], question_template: str) -> List[str]`: Query multiple IDs simultaneously.

- `stream_multiple(ids: List[str], question_template: str) -> List[str]`: Stream multiple responses.

**Usage Example:**
```python
import asyncio

from swarm_models import OpenAI

chat = OpenAI(api_key="YOUR_OPENAI_API_KEY")

response = chat.run("Hello, how can I assist you?")
print(response)

ids = ["id1", "id2", "id3"]
async_responses = asyncio.run(chat.ask_multiple(ids, "How is {id}?"))
print(async_responses)
```

### 2. HuggingFace (swarm_models.HuggingFaceLLM)

The HuggingFaceLLM class allows interaction with language models from Hugging Face.

**Constructor:**
```python
HuggingFaceLLM(model_id: str, device: str = None, max_length: int = 20, quantize: bool = False, quantization_config: dict = None)
```

**Attributes:**

- `model_id` (str): ID or name of the Hugging Face model.

- `device` (str, optional): Device to run the model on (e.g., 'cuda', 'cpu').

- `max_length` (int, default=20): Maximum length of generated text.

- `quantize` (bool, default=False): Apply model quantization.

- `quantization_config` (dict, optional): Configuration for quantization.

**Methods:**

- `run(prompt_text: str, max_length: int = None) -> str`: Generate text based on a prompt.

**Usage Example:**
```python
from swarm_models import HuggingFaceLLM

model_id = "gpt2"
hugging_face_model = HuggingFaceLLM(model_id=model_id)

prompt = "Once upon a time"
generated_text = hugging_face_model.run(prompt)
print(generated_text)
```

### 3. Anthropic (swarm_models.Anthropic)

The Anthropic class enables interaction with Anthropic's large language models.

**Constructor:**
```python
Anthropic(model: str = "claude-2", max_tokens_to_sample: int = 256, temperature: float = None, top_k: int = None, top_p: float = None, streaming: bool = False, default_request_timeout: int = None)
```

**Attributes:**

- `model` (str): Name of the Anthropic model.

- `max_tokens_to_sample` (int, default=256): Maximum tokens to sample.

- `temperature` (float, optional): Temperature for text generation.

- `top_k` (int, optional): Top-k sampling value.

- `top_p` (float, optional): Top-p sampling value.

- `streaming` (bool, default=False): Enable streaming mode.

- `default_request_timeout` (int, optional): Default request timeout.

**Methods:**

- `run(prompt: str, stop: List[str] = None) -> str`: Generate text based on a prompt.

**Usage Example:**
```python
from swarm_models import Anthropic

anthropic = Anthropic()
prompt = "Once upon a time"
generated_text = anthropic.run(prompt)
print(generated_text)
```

This concludes the documentation for the "models" folder, providing you with tools to seamlessly integrate with various language models and APIs. Happy coding!