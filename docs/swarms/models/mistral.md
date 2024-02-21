# `Mistral` Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Overview](#overview)
3. [Class Definition](#class-definition)
   - [Mistral Class](#mistral-class)
   - [Initialization Parameters](#initialization-parameters)
4. [Functionality and Usage](#functionality-and-usage)
   - [Loading the Model](#loading-the-model)
   - [Running the Model](#running-the-model)
   - [Chatting with the Agent](#chatting-with-the-agent)
5. [Additional Information](#additional-information)
6. [Examples](#examples)
   - [Example 1: Initializing Mistral](#example-1-initializing-mistral)
   - [Example 2: Running a Task](#example-2-running-a-task)
   - [Example 3: Chatting with the Agent](#example-3-chatting-with-the-agent)
7. [References and Resources](#references-and-resources)

---

## 1. Introduction <a name="introduction"></a>

Welcome to the documentation for Mistral, a powerful language model-based AI agent. Mistral leverages the capabilities of large language models to generate text-based responses to queries and tasks. This documentation provides a comprehensive guide to understanding and using the Mistral AI agent.

### 1.1 Purpose

Mistral is designed to assist users by generating coherent and contextually relevant text based on user inputs or tasks. It can be used for various natural language understanding and generation tasks, such as chatbots, text completion, question answering, and content generation.

### 1.2 Key Features

- Utilizes large pre-trained language models.
- Supports GPU acceleration for faster processing.
- Provides an easy-to-use interface for running tasks and engaging in chat-based conversations.
- Offers fine-grained control over response generation through temperature and maximum length settings.

---

## 2. Overview <a name="overview"></a>

Before diving into the details of the Mistral AI agent, let's provide an overview of its purpose and functionality.

Mistral is built on top of powerful language models, such as GPT-3. It allows you to:

- Generate text-based responses to tasks and queries.
- Control the temperature of response generation for creativity.
- Set a maximum length for generated responses.
- Engage in chat-based conversations with the AI agent.
- Utilize GPU acceleration for faster inference.

In the following sections, we will explore the class definition, its initialization parameters, and how to use Mistral effectively.

---

## 3. Class Definition <a name="class-definition"></a>

Mistral consists of a single class, the `Mistral` class. This class provides methods for initializing the agent, loading the pre-trained model, and running tasks.

### 3.1 Mistral Class <a name="mistral-class"></a>

```python
class Mistral:
    """
    Mistral

    model = Mistral(device="cuda", use_flash_attention=True, temperature=0.7, max_length=200)
    task = "My favorite condiment is"
    result = model.run(task)
    print(result)
    """

    def __init__(
        self,
        ai_name: str = "Node Model Agent",
        system_prompt: str = None,
        model_name: str = "mistralai/Mistral-7B-v0.1",
        device: str = "cuda",
        use_flash_attention: bool = False,
        temperature: float = 1.0,
        max_length: int = 100,
        do_sample: bool = True,
    ):
        """
        Initializes the Mistral AI agent.

        Parameters:
        - ai_name (str): The name or identifier of the AI agent. Default: "Node Model Agent".
        - system_prompt (str): A system-level prompt for context (e.g., conversation history). Default: None.
        - model_name (str): The name of the pre-trained language model to use. Default: "mistralai/Mistral-7B-v0.1".
        - device (str): The device for model inference, such as "cuda" or "cpu". Default: "cuda".
        - use_flash_attention (bool): If True, enables flash attention for faster inference. Default: False.
        - temperature (float): A value controlling the creativity of generated text. Default: 1.0.
        - max_length (int): The maximum length of generated text. Default: 100.
        - do_sample (bool): If True, uses sampling for text generation. Default: True.
        """
```

### 3.2 Initialization Parameters <a name="initialization-parameters"></a>

- `ai_name` (str): The name or identifier of the AI agent. This name can be used to distinguish between different agents if multiple instances are used. The default value is "Node Model Agent".

- `system_prompt` (str): A system-level prompt that provides context for the AI agent. This can be useful for maintaining a conversation history or providing context for the current task. By default, it is set to `None`.

- `model_name` (str): The name of the pre-trained language model to use. The default value is "mistralai/Mistral-7B-v0.1", which points to a specific version of the Mistral model.

- `device` (str): The device on which the model should perform inference. You can specify "cuda" for GPU acceleration or "cpu" for CPU-only inference. The default is "cuda", assuming GPU availability.

- `use_flash_attention` (bool): If set to `True`, Mistral uses flash attention for faster inference. This is beneficial when low-latency responses are required. The default is `False`.

- `temperature` (float): The temperature parameter controls the creativity of the generated text. Higher values (e.g., 1.0) produce more random output, while lower values (e.g., 0.7) make the output more focused and deterministic. The default value is 1.0.

- `max_length` (int): This parameter sets the maximum length of the generated text. It helps control the length of responses. The default value is 100.

- `do_sample` (bool): If set to `True`, Mistral uses sampling during text generation. Sampling introduces randomness into the generated text. The default is `True`.

---

## 4. Functionality and Usage <a name="functionality-and-usage"></a>

Now that we've introduced the Mistral class and its parameters, let's explore how to use Mistral for various tasks.

### 4.1 Loading the Model <a name="loading-the-model"></a>

The `Mistral` class handles the loading of the pre-trained language model during initialization. You do not need to explicitly load the model. Simply create an instance of `Mistral`, and it will take care of loading the model into memory.

### 4.2 Running the Model <a name="running-the-model"></a>

Mistral provides two methods for running the model:

#### 4.2.1 `run` Method

The `run` method is used to generate text-based responses to a given task or input. It takes a single string parameter, `task`, and returns the generated text as a string.

```python
def run(self, task: str) -> str:
    """
    Run the model on a given task.

    Parameters:
    - task (str): The task or query for which to generate a response.

    Returns:
    - str: The generated text response.
    """
```

Example:

```python
from swarms.models import Mistral

model = Mistral()
task = "Translate the following English text to French: 'Hello, how are you?'"
result = model.run(task)
print(result)
```

#### 4.2.2 `__call__` Method

The `__call__` method provides a more concise way to run the model on a given task. You can use it by simply calling the `Mistral` instance with a task string.

Example:

```python
model = Mistral()
task = "Generate a summary of the latest research paper on AI ethics."
result = model(task)
print(result)
```

### 4.3 Chatting with the Agent <a name="chatting-with-the-agent"></a>

Mistral supports chat-based interactions with the AI agent. You can send a series of messages to the agent, and it will respond accordingly. The `chat` method handles these interactions.

#### `chat` Method

The `chat` method allows you to engage in chat-based conversations with the AI agent. You can send messages to the agent, and it will respond with text-based messages.

```python
def chat(self, msg: str = None, streaming: bool = False) -> str:
    """
    Run a chat conversation with the agent.

    Parameters:
    - msg (str, optional): The message to send to the agent. Defaults to None.
    - streaming (bool, optional): Whether to stream the response token by token. Defaults to False.

    Returns:
    - str: The response from the agent.
    """
```

Example:

```python
model = Mistral()
conversation = [
    "Tell me a joke.",
    "What's the weather like today?",
    "Translate 'apple' to Spanish.",
]
for user_message in conversation:
    response = model.chat(user_message)
    print(f"User: {user_message}")
    print(f"Agent: {response}")
```

---

## 5. Additional Information <a name="additional-information"></a>

Here are some additional tips and information for using Mistral effectively:

- Mistral uses a specific pre-trained model ("mistralai/Mistral-7B-v0.1" by default). You can explore other available models and choose one that best suits your task.

- The `temperature` parameter controls the randomness of generated text. Experiment with different values to achieve the desired level of creativity in responses.

- Be cautious with `max_length`, especially if you set it to a very high value, as it may lead to excessively long responses.

- Ensure that you have the required libraries, such as `torch` and `transformers`, installed to use Mistral successfully.

- Consider providing a system-level prompt when engaging in chat-based conversations to provide context for the agent.

---

## 6. Examples <a name="examples"></a>

In this section, we provide practical examples to illustrate how to use Mistral for various tasks.

### 6.1 Example 1: Initializing Mistral <a name="example-1-initializing-mistral"></a>

In this example, we initialize the Mistral AI agent with custom settings:

```python
from swarms.models import Mistral

model = Mistral(
    ai_name="My AI Assistant",
    device="cpu",
    temperature=0.8,
    max_length=150,
)
```

### 6.2 Example 2: Running a Task <a name="example-2-running-a-task"></a>

Here, we run a text generation task using Mistral:

```python
model = Mistral()
task = "Summarize the main findings of the recent climate change report."
result = model.run(task)
print(result)
```

### 6.3 Example 3: Chatting with the Agent <a name="example-3-chatting-with-the-agent"></a>

Engage in a chat-based conversation with Mistral:

```python
model = Mistral()
conversation = [
    "Tell me a joke.",
    "What's the latest news?",
    "Translate 'cat' to French.",
]
for user_message in conversation:
    response = model.chat(user_message)
    print(f"User: {user_message}")
    print(f"Agent: {response}")
```

---

## 7. References and Resources <a name="references-and-resources"></a>

Here are some references and resources for further information on Mistral and related topics:

- [Mistral GitHub Repository](https://github.com/mistralai/mistral): Official Mistral repository for updates and contributions.
- [Hugging Face Transformers](https://huggingface.co/transformers/): Documentation and models for various transformers, including Mistral's parent models.
- [PyTorch Official Website](https://pytorch.org/): Official website for PyTorch, the deep learning framework used in Mistral.

This concludes the documentation for the Mistral AI agent. You now have a comprehensive understanding of how to use Mistral for text generation and chat-based interactions. If you have any further questions or need assistance, please refer to the provided references and resources. Happy AI modeling!