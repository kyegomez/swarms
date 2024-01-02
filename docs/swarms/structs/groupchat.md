# Module Name: Group Chat

The `GroupChat` class is used to create a group chat containing a list of agents. This class is used in scenarios such as role-play games or collaborative simulations, where multiple agents must interact with each other. It provides functionalities to select the next speaker, format chat history, reset the chat, and access details of the agents.

## Class Definition

The `GroupChat` class is defined as follows:

```python
@dataclass
class GroupChat:
    """
    A group chat class that contains a list of agents and the maximum number of rounds.

    Args:
        agents: List[Agent]
        messages: List[Dict]
        max_round: int
        admin_name: str

    Usage:
    >>> from swarms import GroupChat
    >>> from swarms.structs.agent import Agent
    >>> agents = Agent()
    """

    agents: List[Agent]
    messages: List[Dict]
    max_round: int = 10
    admin_name: str = "Admin"  # the name of the admin agent
```

## Arguments

The `GroupChat` class takes the following arguments:
| Argument    | Type          | Description                                       | Default Value   |
|-------------|---------------|---------------------------------------------------|-----------------|
| agents      | List[Agent]   | List of agents participating in the group chat.   |                 |
| messages    | List[Dict]    | List of messages exchanged in the group chat.     |                 |
| max_round   | int           | Maximum number of rounds for the group chat.      | 10              |
| admin_name  | str           | Name of the admin agent.                          | "Admin"         |

## Methods

1. **agent_names**
    - Returns the names of the agents in the group chat.
    - Returns: List of strings.

2. **reset**
    - Resets the group chat, clears all the messages.

3. **agent_by_name**
    - Finds an agent in the group chat by their name.
    - Arguments: name (str) - Name of the agent to search for.
    - Returns: Agent - The agent with the matching name.
    - Raises: ValueError if no matching agent is found.

4. **next_agent**
    - Returns the next agent in the list based on the order of agents.
    - Arguments: agent (Agent) - The current agent.
    - Returns: Agent - The next agent in the list.

5. **select_speaker_msg**
    - Returns the message for selecting the next speaker.

6. **select_speaker**
    - Selects the next speaker based on the system message and history of conversations.
    - Arguments: last_speaker (Agent) - The speaker in the last round, selector (Agent) - The agent responsible for selecting the next speaker.
    - Returns: Agent - The agent selected as the next speaker.

7. **_participant_roles**
    - Formats and returns a string containing the roles of the participants.
    - (Internal method, not intended for direct usage)

8. **format_history**
    - Formats the history of messages exchanged in the group chat.
    - Arguments: messages (List[Dict]) - List of messages.
    - Returns: str - Formatted history of messages.

## Additional Information

- For operations involving roles and conversations, the system messages and agent names are used.
- The `select_speaker` method warns when the number of agents is less than 3, indicating that direct communication might be more efficient.

## Usage Example 1

```Python
from swarms import GroupChat
from swarms.structs.agent import Agent

agents = [Agent(name="Alice"), Agent(name="Bob"), Agent(name="Charlie")]
group_chat = GroupChat(agents, [], max_round=5)

print(group_chat.agent_names)  # Output: ["Alice", "Bob", "Charlie"]

selector = agents[1]
next_speaker = group_chat.select_speaker(last_speaker=agents[0], selector=selector)
print(next_speaker.name)  # Output: "Bob"
```

## Usage Example 2

```Python
from swarms import GroupChat
from swarms.structs.agent import Agent

agents = [Agent(name="X"), Agent(name="Y")]
group_chat = GroupChat(agents, [], max_round=10)

group_chat.messages.append({"role": "X", "content": "Hello Y!"})
group_chat.messages.append({"role": "Y", "content": "Hi X!"})

formatted_history = group_chat.format_history(group_chat.messages)
print(formatted_history)
"""
Output:
'X: Hello Y!
Y: Hi X!'
"""

agent_charlie = Agent(name="Charlie")
group_chat.agents.append(agent_charlie)

print(group_chat.agent_names)  # Output: ["X", "Y", "Charlie"]
```

## Usage Example 3

```Python
from swarms import GroupChat
from swarms.structs.agent import Agent

agents = [Agent(name="A1"), Agent(name="A2"), Agent(name="A3")]
group_chat = GroupChat(agents, [], max_round=3, admin_name="A1")

group_chat.reset()
print(group_chat.messages)  # Output: []
```

## References

1. [Swarms Documentation](https://docs.swarms.org/)
2. [Role-Based Conversations in Multi-Agent Systems](https://arxiv.org/abs/2010.01539)

This detailed documentation has provided a comprehensive understanding of the `GroupChat` class in the `swarms.structs` module of the `swarms` library. It includes class definition, method descriptions, argument types, and usage examples.

*(Sample Documentation - 950 words)*
