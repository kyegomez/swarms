Swarms Documentation

====================

Worker Node

-----------

The `WorkerNode` class is a powerful component of the Swarms framework. It is designed to spawn an autonomous agent instance as a worker to accomplish complex tasks. It can search the internet, spawn child multi-modality models to process and generate images, text, audio, and so on.

### WorkerNodeInitializer

The `WorkerNodeInitializer` class is used to initialize a worker node.

#### Initialization

```

WorkerNodeInitializer(openai_api_key: str,

llm: Optional[Union[InMemoryDocstore, ChatOpenAI]] = None,

tools: Optional[List[Tool]] = None,

worker_name: Optional[str] = "Swarm Worker AI Assistant",

worker_role: Optional[str] = "Assistant",

human_in_the_loop: Optional[bool] = False,

search_kwargs: dict = {},

verbose: Optional[bool] = False,

chat_history_file: str = "chat_history.txt")

```

Copy code

##### Parameters

- `openai_api_key` (str): The OpenAI API key.

- `llm` (Union[InMemoryDocstore, ChatOpenAI], optional): The language model to use. Default is `ChatOpenAI`.

- `tools` (List[Tool], optional): The tools to use.

- `worker_name` (str, optional): The name of the worker. Default is "Swarm Worker AI Assistant".

- `worker_role` (str, optional): The role of the worker. Default is "Assistant".

- `human_in_the_loop` (bool, optional): Whether to include a human in the loop. Default is False.

- `search_kwargs` (dict, optional): The keyword arguments for the search.

- `verbose` (bool, optional): Whether to print verbose output. Default is False.

- `chat_history_file` (str, optional): The file to store the chat history. Default is "chat_history.txt".

##### Example

```

from swarms.tools.autogpt import DuckDuckGoSearchRun

worker_node_initializer = WorkerNodeInitializer(openai_api_key="your_openai_api_key",

tools=[DuckDuckGoSearchRun()],

worker_name="My Worker",

worker_role="Assistant",

human_in_the_loop=True)

```

Copy code

### WorkerNode

The `WorkerNode` class is used to create a worker node.

#### Initialization

```

WorkerNode(openai_api_key: str,

temperature: int,

llm: Optional[Union[InMemoryDocstore, ChatOpenAI]] = None,

tools: Optional[List[Tool]] = None,

worker_name: Optional[str] = "Swarm Worker AI Assistant",

worker_role: Optional[str] = "Assistant",

human_in_the_loop: Optional[bool] = False,

search_kwargs: dict = {},

verbose: Optional[bool] = False,

chat_history_file: str = "chat_history.txt")

```

Copy code

##### Parameters

- `openai_api_key` (str): The OpenAI API key.

- `temperature` (int): The temperature for the language model.

- `llm` (Union[InMemoryDocstore, ChatOpenAI], optional): The language model to use. Default is `ChatOpenAI`.

- `tools` (List[Tool], optional): The tools to use.

- `worker_name` (str, optional): The name of the worker. Default is "Swarm Worker AI Assistant".

- `worker_role` (str, optional): The role of the worker. Default is "Assistant".

- `human_in_the_loop` (bool, optional): Whether to include a human in the loop. Default is False.

- `search_kwargs` (dict, optional): The keyword arguments for the search.

- `verbose` (bool, optional): Whether to print verbose output. Default is False.

- `chat_history_file` (str, optional): The file to store the chat history. Default is "chat_history.txt".

##### Example

```

worker_node = WorkerNode(openai_api_key="your_openai_api_key",

temperature=0.8,

tools=[DuckDuckGoSearchRun()],

worker_name="My Worker",

worker_role="As```

tools=[DuckDuckGoSearchRun()],

worker_name="My Worker",

worker_role="Assistant",

human_in_the_loop=True)

# Create a worker node

worker_node = WorkerNode(openai_api_key="your_openai_api_key",

temperature=0.8,

tools=[DuckDuckGoSearchRun()],

worker_name="My Worker",

worker_role="Assistant",

human_in_the_loop=True)

# Add a tool to the worker node

worker_node_initializer.add_tool(DuckDuckGoSearchRun())

# Initialize the language model and tools for the worker node

worker_node.initialize_llm(ChatOpenAI, temperature=0.8)

worker_node.initialize_tools(ChatOpenAI)

# Create the worker node

worker_node.create_worker_node(worker_name="My Worker Node",

worker_role="Assistant",

human_in_the_loop=True,

llm_class=ChatOpenAI,

search_kwargs={})

# Run the worker node

`worker_node.run("Hello, world!")`

In this example, we first initialize a `WorkerNodeInitializer` and a `WorkerNode`. We then add a tool to the `WorkerNodeInitializer` and initialize the language model and tools for the `WorkerNode`. Finally, we create the worker node and run it with a given prompt.

This example shows how you can use the `WorkerNode` and `WorkerNodeInitializer` classes to create a worker node, add tools to it, initialize its language model and tools, and run it with a given prompt. The parameters of these classes can be customized to suit your specific needs.

Thanks for becoming an alpha build user, email kye@apac.ai with all complaintssistant",

human_in_the_loop=True)

```

Copy code

### Full Example

Here is a full example of how to use the `WorkerNode` and `WorkerNodeInitializer` classes:

```python

from swarms.tools.autogpt import DuckDuckGoSearchRun

from swarms.worker_node import WorkerNode, WorkerNodeInitializer

# Initialize a worker node

worker_node_initializer = WorkerNodeInitializer(openai_api_key="your_openai_api_key",

tools=[DuckDuckGoSearchRun()],

worker_name="My Worker",

worker_role="Assistant",

human_in_the_loop=True)

# Create a worker node

worker_node = WorkerNode(openai_api_key="your_openai_api_key",

temperature=0.8,

tools=[DuckDuckGoSearchRun()],

worker_name="My Worker",

worker_role="Assistant",

human_in_the_loop=True)

# Add a tool to the worker node

worker_node_initializer.add_tool(DuckDuckGoSearchRun())

# Initialize the language model and tools for the worker node

worker_node.initialize_llm(ChatOpenAI, temperature=0.8)

worker_node.initialize_tools(ChatOpenAI)

# Create the worker node

worker_node.create_worker_node(worker_name="My Worker Node",

worker_role="Assistant",

human_in_the_loop=True,

llm_class=ChatOpenAI,

search_kwargs={})

# Run the worker node

worker_node.run("Hello, world!")

```

In this example, we first initialize a `WorkerNodeInitializer` and a `WorkerNode`. We then add a tool to the `WorkerNodeInitializer` and initialize the language model and tools for the `WorkerNode`. Finally, we create the worker node and run it with a given prompt.

This example shows how you can use the `WorkerNode` and `WorkerNodeInitializer` classes to create a worker node, add tools to it, initialize its language model and tools, and run it with a given prompt. The parameters of these classes can be customized to suit your specific needs.