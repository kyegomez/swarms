# Module/Class Name: OmniModalAgent

The `OmniModalAgent` class is a module that operates based on the Language Model (LLM) aka Language Understanding Model, Plans, Tasks, and Tools. It is designed to be a multi-modal chatbot which uses various AI-based capabilities for fulfilling user requests. 

It has the following architecture:
1. Language Model (LLM).
2. Chat Planner - Plans
3. Task Executor - Tasks
4. Tools - Tools

![OmniModalAgent](https://source.unsplash.com/random)

---

### Usage
    from swarms import OmniModalAgent, OpenAIChat
    
    llm = OpenAIChat()
    agent = OmniModalAgent(llm)
    response = agent.run("Hello, how are you? Create an image of how your are doing!")

---

---

### Initialization

The constructor of `OmniModalAgent` class takes two main parameters:
- `llm`: A `BaseLanguageModel` that represents the language model
- `tools`: A List of `BaseTool` instances that are used by the agent for fulfilling different requests.

```python
def __init__(
    self,
    llm: BaseLanguageModel,
    # tools: List[BaseTool]
):
```

---

### Methods

The class has two main methods:
1. `run`: This method takes an input string and executes various plans and tasks using the provided tools. Ultimately, it generates a response based on the user's input and returns it.
   - Parameters:
     - `input`: A string representing the user's input text.
   - Returns:
     - A string representing the response.
   
   Usage:
   ```python
   response = agent.run("Hello, how are you? Create an image of how your are doing!")
   ```

2. `chat`: This method is used to simulate a chat dialog with the agent. It can take user's messages and return the response (or stream the response word-by-word if required).
   - Parameters:
     - `msg` (optional): A string representing the message to send to the agent.
     - `streaming` (optional): A boolean specifying whether to stream the response.
   - Returns:
     - A string representing the response from the agent.

   Usage:
   ```python
   response = agent.chat("Hello")
   ```

---

### Streaming Response

The class provides a method `_stream_response` that can be used to get the response token by token (i.e. word by word). It yields individual tokens from the response.
   
Usage:
```python
for token in _stream_response(response):
    print(token)
```

