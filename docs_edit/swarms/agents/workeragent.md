# WorkerClass Documentation

## Overview

The Worker class represents an autonomous agent that can perform tasks through function calls or by running a chat. It can be used to create applications that demand effective user interactions like search engines, human-like conversational bots, or digital assistants.

The `Worker` class is part of the `swarms.agents` codebase. This module is largely used in Natural Language Processing (NLP) projects where the agent undertakes conversations and other language-specific operations.

## Class Definition

The class `Worker` has the following arguments:

| Argument              | Type          | Default Value                    | Description                                        |
|-----------------------|---------------|----------------------------------|----------------------------------------------------|
| name                  | str           | "Worker"                     | Name of the agent.                                 |
| role                  | str           | "Worker in a swarm"              | Role of the agent.                                 |
| external_tools        | list          | None                             | List of external tools available to the agent.     |
| human_in_the_loop     | bool          | False                            | Determines whether human interaction is required.   |
| temperature           | float         | 0.5                              | Temperature for the autonomous agent.              |
| llm                   | None          | None                             | Language model.                                    |
| openai_api_key        | str           | None                             | OpenAI API key.                                    |
| tools                 | List[Any]    | None                             | List of tools available to the agent.              |
| embedding_size        | int           | 1536                             | Size of the word embeddings.                        |
| search_kwargs         | dict          | {"k": 8}                         | Search parameters.                                 |
| args                  | Multiple      |                                  | Additional arguments that can be passed.           |
| kwargs                | Multiple      |                                  | Additional keyword arguments that can be passed.   |
## Usage

#### Example 1: Creating and Running an Agent

```python
from swarms import Worker

worker = Worker(
    name="My Worker",
    role="Worker",
    external_tools=[MyTool1(), MyTool2()],
    human_in_the_loop=False,
    temperature=0.5,
    llm=some_language_model,
    openai_api_key="my_key",
)
worker.run("What's the weather in Miami?")
```

#### Example 2: Receiving and Sending Messages

```python
worker.receieve("User", "Hello there!")
worker.receieve("User", "Can you tell me something about history?")
worker.send()
```

#### Example 3: Setting up Tools

```python
external_tools = [MyTool1(), MyTool2()]
worker = Worker(
    name="My Worker",
    role="Worker",
    external_tools=external_tools,
    human_in_the_loop=False,
    temperature=0.5,
)
```

## Additional Information and Tips

- The class allows the setting up of tools for the worker to operate effectively. It provides setup facilities for essential computing infrastructure, such as the agent's memory and language model.
- By setting the `human_in_the_loop` parameter to True, interactions with the worker can be made more user-centric.
- The `openai_api_key` argument can be provided for leveraging the OpenAI infrastructure and services.
- A qualified language model can be passed as an instance of the `llm` object, which can be useful when integrating with state-of-the-art text generation engines.

## References and Resources

- [OpenAI APIs](https://openai.com)
- [Models and Languages at HuggingFace](https://huggingface.co/models)
- [Deep Learning and Language Modeling at the Allen Institute for AI](https://allenai.org)
