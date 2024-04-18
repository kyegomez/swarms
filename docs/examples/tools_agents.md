# **Equipping Autonomous Agents with Tools**
==========================================

Tools play a crucial role in enhancing the capabilities of AI agents. Swarms, a powerful open-source framework, provides a robust and flexible environment for building and integrating tools with AI agents. In this comprehensive guide, we'll explore the process of creating tools in Swarms, including the 3-step process, tool decorator, adding types and doc strings, and integrating them into the Agent class.

## **Introduction to Swarms**
--------------------------

Swarms is a Python-based framework that simplifies the development and deployment of AI agents. It provides a seamless integration with large language models (LLMs) and offers a wide range of tools and utilities to streamline the agent development process. One of the core features of Swarms is the ability to create and integrate custom tools, which can significantly extend the capabilities of AI agents.

Learn more here:

[**GitHub - kyegomez/swarms: Build, Deploy, and Scale Reliable Swarms of Autonomous Agents for...**](https://github.com/kyegomez/swarms?source=post_page-----49d146bcbf9e--------------------------------)
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

### [Build, Deploy, and Scale Reliable Swarms of Autonomous Agents for Workflow Automation. Join our Community...](https://github.com/kyegomez/swarms?source=post_page-----49d146bcbf9e--------------------------------)

[github.com](https://github.com/kyegomez/swarms?source=post_page-----49d146bcbf9e--------------------------------)

And, join our community for real-time support and conversations with friends!

[**Join the Agora Discord Server!**](https://discord.gg/A8DrG5nj?source=post_page-----49d146bcbf9e--------------------------------)
-----------------------------------------------------------------------------------------------------------------------------------

### [Advancing Humanity through open source AI research. | 6319 members](https://discord.gg/A8DrG5nj?source=post_page-----49d146bcbf9e--------------------------------)

[discord.gg](https://discord.gg/A8DrG5nj?source=post_page-----49d146bcbf9e--------------------------------)

**Installation**
================

First, download swarms with the following command. If you have any questions please refer to this video or ask us in the discord!

```bash
pip3 install -U swarms
```

**Necessary Imports**
---------------------

Before we dive into the process of creating tools in Swarms, let's familiarize ourselves with the necessary imports:

```python
from swarms import Agent, Anthropic, tool
import subprocess
```

-   These imports provide access to the core components of the Swarms framework, including the `Agent` class, the `Anthropic` language model, and the `tool` decorator for creating custom tools.

-   `import subprocess`: This import allows us to interact with the system's terminal and execute shell commands, which can be useful for certain types of tools.

With these imports in place, we're ready to explore the process of creating tools in Swarms.

### **The 3-Step Process**
======================

Creating tools in Swarms follows a straightforward 3-step process:

1\. Define the Tool Function\
2\. Decorate the Function with `[@tool](http://twitter.com/tool)` and add documentation with type hints.\
3\. Add the Tool to the `Agent` Instance

Let's go through each step in detail, accompanied by code examples.

### **Step 1: Define the Tool Function**
------------------------------------

The first step in creating a tool is to define a Python function that encapsulates the desired functionality. This function will serve as the core logic for your tool. Here's an example of a tool function that allows you to execute code in the terminal:

```python
def terminal(code: str) -> str:
    """
    Run code in the terminal.

    Args:
        code (str): The code to run in the terminal.

    Returns:
        str: The output of the code.
    """
    out = subprocess.run(
        code, shell=True, capture_output=True, text=True
    ).stdout
    return str(out)
```

In this example, the `terminal` function takes a string `code` as input and uses the `subprocess` module to execute the provided code in the system's terminal. The output of the code is captured and returned as a string.

### **Let's break down the components of this function:**
-----------------------------------------------------

-   **Function Signature:** The function signature `def terminal(code: str) -> str:` defines the function name (`terminal`), the parameter name and type (`code: str`), and the return type (`-> str`). This adheres to Python's type hinting conventions.

-   **Docstring**: The multiline string enclosed in triple quotes (`"""` ... `"""`) is a docstring, which provides a brief description of the function, its parameters, and its return value. Docstrings are essential for documenting your code and making it easier for the agent to understand and use your tools.

-   **Function Body**: The body of the function contains the actual logic for executing the code in the terminal. It uses the `subprocess.run` function to execute the provided `code` in the shell, capturing the output (`capture_output=True`), and returning the output as text (`text=True`). The `stdout` attribute of the `CompletedProcess` object contains the captured output, which is converted to a string and returned.

This is a simple example, but it demonstrates the key components of a tool function: a well-defined signature with type hints, a descriptive docstring, and the core logic encapsulated within the function body.

### **Step 2: Decorate the Function with `**[**@tool**](http://twitter.com/tool)**`**
---------------------------------------------------------------------------------

After defining the tool function, the next step is to decorate it with the `[@tool](http://twitter.com/tool)` decorator provided by Swarms. This decorator registers the function as a tool, allowing it to be used by AI agents within the Swarms framework.

Here's how you would decorate the `terminal` function from the previous example:

```python
@tool
def terminal(code: str) -> str:
    """
    Run code in the terminal.

    Args:
        code (str): The code to run in the terminal.

    Returns:
        str: The output of the code.
    """
    out = subprocess.run(
        code, shell=True, capture_output=True, text=True
    ).stdout
    return str(out)
```

The `[@tool](http://twitter.com/tool)` decorator is placed directly above the function definition. This decorator performs the necessary registration and configuration steps to integrate the tool with the Swarms framework.

### **Step 3: Add the Tool to the `Agent` Instance**
------------------------------------------------

The final step in creating a tool is to add it to the `Agent` instance. The `Agent` class is a core component of the Swarms framework and represents an AI agent capable of interacting with humans and other agents, as well as utilizing the available tools.

Here's an example of how to create an `Agent` instance and add the `terminal` tool:

```python
# Model
llm = Anthropic(
    temperature=0.1,
)

# Agent
agent = Agent(
    agent_name="Devin",
    system_prompt=(
        "Autonomous agent that can interact with humans and other"
        " agents. Be Helpful and Kind. Use the tools provided to"
        " assist the user. Return all code in markdown format."
    ),
    llm=llm,
    max_loops="auto",
    autosave=True,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
    interactive=True,
    tools=[terminal],
    code_interpreter=True,
)
```

In this example, we first create an instance of the `Anthropic` language model, which will be used by the agent for natural language

**The Necessity of Documentation**
----------------------------------

Before creating tools, it's essential to understand the importance of documentation. Clear and concise documentation ensures that your code is easily understandable and maintainable, not only for yourself but also for other developers who may work with your codebase in the future.

Effective documentation serves several purposes:

1.  Code Clarity: Well-documented code is easier to read and understand, making it more accessible for both developers and non-technical stakeholders.

2.  Collaboration: When working in a team or contributing to open-source projects, proper documentation facilitates collaboration and knowledge sharing.

3.  Onboarding: Comprehensive documentation can significantly streamline the onboarding process for new team members or contributors, reducing the time required to familiarize themselves with the codebase.

4.  Future Maintenance: As projects evolve and requirements change, well-documented code becomes invaluable for future maintenance and updates.

In the context of creating tools in Swarms, documentation plays a vital role in ensuring that your tools are easily discoverable, understandable, and usable by other developers and AI agents.

**Type Handling**
-----------------

Python is a dynamically-typed language, which means that variables can hold values of different types during runtime. While this flexibility can be advantageous in certain scenarios, it can also lead to potential errors and inconsistencies, especially in larger codebases.

Type hints, introduced in Python 3.5, provide a way to explicitly annotate the expected types of variables, function parameters, and return values. By incorporating type hints into your code, you can:

1.  Improve Code Readability: Type hints make it easier for developers to understand the expected data types, reducing the risk of introducing bugs due to type-related errors.

2.  Enable Static Type Checking: With tools like mypy, you can perform static type checking, catching potential type-related issues before running the code.

3.  Enhance Code Completion and Tooling: Modern IDEs and code editors can leverage type hints to provide better code completion, refactoring capabilities, and inline documentation.

In the context of creating tools in Swarms, type hints are crucial for ensuring that your tools are used correctly by AI agents and other developers. By clearly defining the expected input and output types, you can reduce the likelihood of runtime errors and improve the overall reliability of your tools.

Now, let's continue with other tool examples!

### **Additional Tool Examples**
============================

To further illustrate the process of creating tools in Swarms, let's explore a few more examples of tool functions with varying functionalities.

**Browser Tool**

```python
@tool
def browser(query: str) -> str:
    """
    Search the query in the browser with the `browser` tool.
    Args:
        query (str): The query to search in the browser.
    Returns:
        str: The search results.
    """
    import webbrowser
    url = f"https://www.google.com/search?q={query}"
    webbrowser.open(url)
    return f"Searching for {query} in the browser."
```

The `browser` tool allows the agent to perform web searches by opening the provided `query` in the default web browser. It leverages the `webbrowser` module to construct the search URL and open it in the browser. The tool returns a string indicating that the search is being performed.

**File Creation Tool**

```python
@tool
def create_file(file_path: str, content: str) -> str:
    """
    Create a file using the file editor tool.
    Args:
        file_path (str): The path to the file.
        content (str): The content to write to the file.
    Returns:
        str: The result of the file creation operation.
    """
    with open(file_path, "w") as file:
        file.write(content)
    return f"File {file_path} created successfully."
```

The `create_file` tool allows the agent to create a new file at the specified `file_path` with the provided `content`. It uses Python's built-in `open` function in write mode (`"w"`) to create the file and write the content to it. The tool returns a string indicating the successful creation of the file.

**File Editor Tool**

```python
@tool
def file_editor(file_path: str, mode: str, content: str) -> str:
    """
    Edit a file using the file editor tool.
    Args:
        file_path (str): The path to the file.
        mode (str): The mode to open the file in.
        content (str): The content to write to the file.
    Returns:
        str: The result of the file editing operation.
    """
    with open(file_path, mode) as file:
        file.write(content)
    return f"File {file_path} edited successfully."
```

The `file_editor` tool is similar to the `create_file` tool but provides more flexibility by allowing the agent to specify the mode in which the file should be opened (e.g., `"w"` for write, `"a"` for append). It writes the provided `content` to the file at the specified `file_path` and returns a string indicating the successful editing of the file.

These examples demonstrate the versatility of tools in Swarms and how they can be designed to perform a wide range of tasks, from executing terminal commands to interacting with files and performing web searches.

### **Plugging Tools into the Agent**
=================================

After defining and decorating your tool functions, the next step is to integrate them into the `Agent` instance. This process involves passing the tools as a list to the `tools` parameter when creating the `Agent` instance.

```python
# Model
llm = Anthropic(temperature=0.1)
# Tools
tools = [terminal, browser, create_file, file_editor]
# Agent
agent = Agent(
    agent_name="Devin",
    system_prompt=(
        "Autonomous agent that can interact with humans and other"
        " agents. Be Helpful and Kind. Use the tools provided to"
        " assist the user. Return all code in markdown format."
    ),
    llm=llm,
    max_loops="auto",
    autosave=True,
    dashboard=False,
    streaming_on=True,
    verbose=True,
    stopping_token="<DONE>",
    interactive=True,
    tools=tools,
    code_interpreter=True,
)
```

In this example, we create a list `tools` containing the `terminal`, `browser`, `create_file`, and `file_editor` tools. This list is then passed to the `tools` parameter when creating the `Agent` instance.

Once the `Agent` instance is created with the provided tools, it can utilize these tools to perform various tasks and interact with external systems. The agent can call these tools as needed, passing the required arguments and receiving the corresponding return values.

### **Using Tools in Agent Interactions**
=====================================

After integrating the tools into the `Agent` instance, you can utilize them in your agent's interactions with humans or other agents. Here's an example of how an agent might use the `terminal` tool:

```
out = agent("Run the command 'ls' in the terminal.")
print(out)
```

In this example, the human user instructs the agent to run the `"ls"` command in the terminal. The agent processes this request and utilizes the `terminal` tool to execute the command, capturing and returning the output.

Similarly, the agent can leverage other tools, such as the `browser` tool for web searches or the `file_editor` tool for creating and modifying files, based on the user's instructions.

### **Conclusion**
==============

Creating tools in Swarms is a powerful way to extend the capabilities of AI agents and enable them to interact with external systems and perform a wide range of tasks. By following the 3-step process of defining the tool function, decorating it with `@tool`, and adding it to the `Agent` instance, you can seamlessly integrate custom tools into your AI agent's workflow.

Throughout this blog post, we explored the importance of documentation and type handling, which are essential for maintaining code quality, facilitating collaboration, and ensuring the correct usage of your tools by other developers and AI agents.

We also covered the necessary imports and provided detailed code examples for various types of tools, such as executing terminal commands, performing web searches, and creating and editing files. These examples demonstrated the flexibility and versatility of tools in Swarms, allowing you to tailor your tools to meet your specific project requirements.

By leveraging the power of tools in Swarms, you can empower your AI agents with diverse capabilities, enabling them to tackle complex tasks, interact with external systems, and provide more comprehensive and intelligent solutions.
