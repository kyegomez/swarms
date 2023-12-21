# Conversation Module Documentation

## Table of Contents

1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Class: Conversation](#class-conversation)
   - [Attributes](#attributes)
   - [Methods](#methods)
4. [Usage Examples](#usage-examples)
   - [Example 1: Creating a Conversation](#example-1-creating-a-conversation)
   - [Example 2: Adding Messages](#example-2-adding-messages)
   - [Example 3: Displaying and Exporting Conversation](#example-3-displaying-and-exporting-conversation)
   - [Example 4: Counting Messages by Role](#example-4-counting-messages-by-role)
   - [Example 5: Loading and Searching](#example-5-loading-and-searching)
5. [Additional Information](#additional-information)
6. [References](#references)

---

## 1. Introduction <a name="introduction"></a>

The Conversation module provides a versatile and extensible structure for managing and analyzing text-based conversations. Whether you're developing a chatbot, analyzing customer support interactions, or conducting research on dialogues, this module simplifies the process of handling conversation data.

With the Conversation module, you can add, delete, update, query, and search for messages within a conversation. You can also display, export, and import conversation history, making it an essential tool for various applications.

## 2. Installation <a name="installation"></a>

To use the Conversation module, you need to have Python installed on your system. Additionally, you can install the required dependencies using pip:

```bash
pip install termcolor
```

Once you have the dependencies installed, you can import the Conversation module into your Python code.

```python
from swarms.structs.conversation import Conversation
```

## 3. Class: Conversation <a name="class-conversation"></a>

The Conversation class is the core of this module. It allows you to create and manipulate conversation histories. Below are the attributes and methods provided by this class.

### Attributes <a name="attributes"></a>

- `time_enabled` (bool): Indicates whether timestamps are enabled for messages in the conversation.
- `conversation_history` (list): A list that stores the conversation history as a collection of messages.

### Methods <a name="methods"></a>

The Conversation class provides the following methods:

- `add(role: str, content: str, *args, **kwargs)`: Adds a message to the conversation history.
- `delete(index: str)`: Deletes a message from the conversation history.
- `update(index: str, role, content)`: Updates a message in the conversation history.
- `query(index: str)`: Queries a message in the conversation history.
- `search(keyword: str)`: Searches for messages containing a specific keyword.
- `display_conversation(detailed: bool = False)`: Displays the conversation history.
- `export_conversation(filename: str)`: Exports the conversation history to a file.
- `import_conversation(filename: str)`: Imports a conversation history from a file.
- `count_messages_by_role()`: Counts the number of messages by role.
- `return_history_as_string()`: Returns the conversation history as a string.
- `save_as_json(filename: str)`: Saves the conversation history as a JSON file.
- `load_from_json(filename: str)`: Loads the conversation history from a JSON file.
- `search_keyword_in_conversation(keyword: str)`: Searches for a keyword in the conversation history.
- `pretty_print_conversation(messages)`: Pretty prints the conversation history.

## 4. Usage Examples <a name="usage-examples"></a>

In this section, we'll provide practical examples of how to use the Conversation module to manage and analyze conversation data.

### Example 1: Creating a Conversation <a name="example-1-creating-a-conversation"></a>

Let's start by creating a Conversation object and enabling timestamps for messages:

```python
conversation = Conversation(time_enabled=True)
```

### Example 2: Adding Messages <a name="example-2-adding-messages"></a>

You can add messages to the conversation using the `add` method. Here's how to add a user message and an assistant response:

```python
conversation.add("user", "Hello, how can I help you?")
conversation.add("assistant", "Hi there! I'm here to assist you.")
```

### Example 3: Displaying and Exporting Conversation <a name="example-3-displaying-and-exporting-conversation"></a>

You can display the conversation history and export it to a file. Let's see how to do this:

```python
# Display the conversation
conversation.display_conversation()

# Export the conversation to a file
conversation.export_conversation("conversation_history.txt")
```

### Example 4: Counting Messages by Role <a name="example-4-counting-messages-by-role"></a>

You can count the number of messages by role (e.g., user, assistant, system) using the `count_messages_by_role` method:

```python
message_counts = conversation.count_messages_by_role()
print(message_counts)
```

### Example 5: Loading and Searching <a name="example-5-loading-and-searching"></a>

You can load a conversation from a file and search for messages containing a specific keyword:

```python
# Load conversation from a file
conversation.load_from_json("saved_conversation.json")

# Search for messages containing the keyword "help"
results = conversation.search("help")
print(results)
```

## 5. Additional Information <a name="additional-information"></a>

- The Conversation module is designed to provide flexibility and ease of use for managing and analyzing text-based conversations.
- You can extend the module by adding custom functionality or integrating it into your chatbot or natural language processing applications.

## 6. References <a name="references"></a>

For more information on the Conversation module and its usage, refer to the official documentation and examples.

