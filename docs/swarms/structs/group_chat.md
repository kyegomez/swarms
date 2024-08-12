# GroupChat

The `GroupChat` class is designed to manage a group chat session involving multiple agents. This class handles initializing the conversation, selecting the next speaker, resetting the chat, and executing the chat rounds, providing a structured approach to managing a dynamic and interactive conversation.

### Key Concepts

- **Agents**: Entities participating in the group chat.
- **Conversation Management**: Handling the flow of conversation, selecting speakers, and maintaining chat history.
- **Round-based Execution**: Managing the chat in predefined rounds.

## Attributes

### Arguments

| Argument            | Type                 | Default     | Description |
|---------------------|----------------------|-------------|-------------|
| `agents`            | `List[Agent]`        | `None`      | List of agents participating in the group chat. |
| `max_rounds`        | `int`                | `10`        | Maximum number of chat rounds. |
| `admin_name`        | `str`                | `"Admin"`   | Name of the admin user. |
| `group_objective`   | `str`                | `None`      | Objective of the group chat. |
| `selector_agent`    | `Agent`              | `None`      | Agent responsible for selecting the next speaker. |
| `rules`             | `str`                | `None`      | Rules for the group chat. |
| `*args`             |                      |             | Variable length argument list. |
| `**kwargs`          |                      |             | Arbitrary keyword arguments. |

### Attributes

| Attribute           | Type                 | Description |
|---------------------|----------------------|-------------|
| `agents`            | `List[Agent]`        | List of agents participating in the group chat. |
| `max_rounds`        | `int`                | Maximum number of chat rounds. |
| `admin_name`        | `str`                | Name of the admin user. |
| `group_objective`   | `str`                | Objective of the group chat. |
| `selector_agent`    | `Agent`              | Agent responsible for selecting the next speaker. |
| `messages`          | `Conversation`       | Conversation object for storing the chat messages. |

## Methods

### __init__

Initializes the group chat with the given parameters.

**Examples:**

```python
agents = [Agent(name="Agent 1"), Agent(name="Agent 2")]
group_chat = GroupChat(agents=agents, max_rounds=5, admin_name="GroupAdmin")
```

### agent_names

Returns the names of the agents in the group chat.

**Returns:**

| Return Type | Description |
|-------------|-------------|
| `List[str]` | List of agent names. |

**Examples:**

```python
names = group_chat.agent_names
print(names)  # Output: ['Agent 1', 'Agent 2']
```

### reset

Resets the group chat by clearing the message history.

**Examples:**

```python
group_chat.reset()
```

### agent_by_name

Finds an agent whose name is contained within the given name string.

**Arguments:**

| Parameter | Type   | Description |
|-----------|--------|-------------|
| `name`    | `str`  | Name string to search for. |

**Returns:**

| Return Type | Description |
|-------------|-------------|
| `Agent`     | Agent object with a name contained in the given name string. |

**Raises:**

- `ValueError`: If no agent is found with a name contained in the given name string.

**Examples:**

```python
agent = group_chat.agent_by_name("Agent 1")
print(agent.agent_name)  # Output: 'Agent 1'
```

### next_agent

Returns the next agent in the list.

**Arguments:**

| Parameter | Type   | Description |
|-----------|--------|-------------|
| `agent`   | `Agent`| Current agent. |

**Returns:**

| Return Type | Description |
|-------------|-------------|
| `Agent`     | Next agent in the list. |

**Examples:**

```python
current_agent = group_chat.agents[0]
next_agent = group_chat.next_agent(current_agent)
print(next_agent.agent_name)  # Output: Name of the next agent
```

### select_speaker_msg

Returns the message for selecting the next speaker.

**Returns:**

| Return Type | Description |
|-------------|-------------|
| `str`       | Prompt message for selecting the next speaker. |

**Examples:**

```python
message = group_chat.select_speaker_msg()
print(message)
```

### select_speaker

Selects the next speaker.

**Arguments:**

| Parameter            | Type   | Description |
|----------------------|--------|-------------|
| `last_speaker_agent` | `Agent`| Last speaker in the conversation. |
| `selector_agent`     | `Agent`| Agent responsible for selecting the next speaker. |

**Returns:**

| Return Type | Description |
|-------------|-------------|
| `Agent`     | Next speaker. |

**Examples:**

```python
next_speaker = group_chat.select_speaker(last_speaker_agent, selector_agent)
print(next_speaker.agent_name)
```

### _participant_roles

Returns the roles of the participants.

**Returns:**

| Return Type | Description |
|-------------|-------------|
| `str`       | Participant roles. |

**Examples:**

```python
roles = group_chat._participant_roles()
print(roles)
```

### __call__

Executes the group chat as a function.

**Arguments:**

| Parameter | Type   | Description |
|-----------|--------|-------------|
| `task`    | `str`  | Task to be performed. |

**Returns:**

| Return Type | Description |
|-------------|-------------|
| `str`       | Reply from the last speaker. |

**Examples:**

```python
response = group_chat(task="Discuss the project plan")
print(response)
```

### Additional Examples

#### Example 1: Initializing and Running a Group Chat

```python
agents = [Agent(name="Agent 1"), Agent(name="Agent 2"), Agent(name="Agent 3")]
selector_agent = Agent(name="Selector")
group_chat = GroupChat(agents=agents, selector_agent=selector_agent, max_rounds=3, group_objective="Discuss the quarterly goals.")

response = group_chat(task="Let's start the discussion on quarterly goals.")
print(response)
```

#### Example 2: Resetting the Group Chat

```python
group_chat.reset()
```

#### Example 3: Selecting the Next Speaker

```python
last_speaker = group_chat.agents[0]
next_speaker = group_chat.select_speaker(last_speaker_agent=last_speaker, selector_agent=selector_agent)
print(next_speaker.agent_name)
```

## Summary

The `GroupChat` class offers a structured approach to managing a group chat involving multiple agents. With functionalities for initializing conversations, selecting speakers, and handling chat rounds, it provides a robust framework for dynamic and interactive discussions. This makes it an essential tool for applications requiring coordinated communication among multiple agents.