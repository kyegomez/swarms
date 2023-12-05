# Swarms Framework Documentation

---

## Overview

The Swarms framework is a Python library designed to facilitate the creation and management of a simulated group chat environment. This environment can be used for a variety of purposes, such as training conversational agents, role-playing games, or simulating dialogues for machine learning purposes. The core functionality revolves around managing the agent of messages between different agents within the chat, as well as handling the selection and responses of these agents based on the conversation's context.

### Purpose

The purpose of the Swarms framework, and specifically the `GroupChat` and `GroupChatManager` classes, is to simulate a dynamic and interactive conversation between multiple agents. This simulates a real-time chat environment where each participant is represented by an agent with a specific role and behavioral patterns. These agents interact within the rules of the group chat, controlled by the `GroupChatManager`.

### Key Features

- **Agent Interaction**: Allows multiple agents to communicate within a group chat scenario.
- **Message Management**: Handles the storage and agent of messages within the group chat.
- **Role Play**: Enables agents to assume specific roles and interact accordingly.
- **Conversation Context**: Maintains the context of the conversation for appropriate responses by agents.

---

## GroupChat Class

The `GroupChat` class is the backbone of the Swarms framework's chat simulation. It maintains the list of agents participating in the chat, the messages that have been exchanged, and the logic to reset the chat and determine the next speaker.

### Class Definition

#### Parameters

| Parameter  | Type                | Description                                                  | Default Value |
|------------|---------------------|--------------------------------------------------------------|---------------|
| agents     | List[Agent]          | List of agent flows participating in the group chat.         | None          |
| messages   | List[Dict]          | List of message dictionaries exchanged in the group chat.    | None          |
| max_round  | int                 | Maximum number of rounds/messages allowed in the group chat. | 10            |
| admin_name | str                 | The name of the admin agent in the group chat.               | "Admin"       |

#### Class Properties and Methods

- `agent_names`: Returns a list of the names of the agents in the group chat.
- `reset()`: Clears all messages from the group chat.
- `agent_by_name(name: str) -> Agent`: Finds and returns an agent by name.
- `next_agent(agent: Agent) -> Agent`: Returns the next agent in the list.
- `select_speaker_msg() -> str`: Returns the message for selecting the next speaker.
- `select_speaker(last_speaker: Agent, selector: Agent) -> Agent`: Logic to select the next speaker based on the last speaker and the selector agent.
- `_participant_roles() -> str`: Returns a string listing all participant roles.
- `format_history(messages: List[Dict]) -> str`: Formats the history of messages for display or processing.

### Usage Examples

#### Example 1: Initializing a GroupChat

```python
from swarms.structs.agent import Agent
from swarms.groupchat import GroupChat

# Assuming Agent objects (flow1, flow2, flow3) are initialized and configured
agents = [flow1, flow2, flow3]
group_chat = GroupChat(agents=agents, messages=[], max_round=10)
```

#### Example 2: Resetting a GroupChat

```python
group_chat.reset()
```

#### Example 3: Selecting a Speaker

```python
last_speaker = agents[0]  # Assuming this is a Agent object representing the last speaker
selector = agents[1]  # Assuming this is a Agent object with the selector role

next_speaker = group_chat.select_speaker(last_speaker, selector)
```

---

## GroupChatManager Class

The `GroupChatManager` class acts as a controller for the `GroupChat` instance. It orchestrates the interaction between agents, prompts for tasks, and manages the rounds of conversation.

### Class Definition

#### Constructor Parameters

| Parameter  | Type        | Description                                          |
|------------|-------------|------------------------------------------------------|
| groupchat  | GroupChat   | The GroupChat instance that the manager will handle. |
| selector   | Agent        | The Agent object that selects the next speaker.       |

#### Methods

- `__call__(task: str)`: Invokes the GroupChatManager with a given task string to start the conversation.

### Usage Examples

#### Example 1: Initializing GroupChatManager

```python
from swarms.groupchat import GroupChat, GroupChatManager
from swarms.structs.agent import Agent

# Initialize your agents and group chat as shown in previous examples
chat_manager = GroupChatManager(groupchat=group_chat, selector=manager)
```

#### Example 2: Starting a Conversation

```python
# Start the group chat with a task
chat_history = chat_manager("Start a conversation about space exploration.")
```

#### Example 3: Using the Call Method

```python
# The call method is the same as starting a conversation
chat_history = chat_manager.__call__("Discuss recent advances in AI.")
```

---

## Conclusion

In summary, the Swarms framework offers a unique and effective solution for simulating group chat environments. Its `GroupChat` and `GroupChatManager` classes provide the necessary infrastructure to create dynamic conversations between agents, manage messages, and maintain the context of the dialogue. This framework can be instrumental in developing more sophisticated conversational agents, experimenting with social dynamics in chat environments, and providing a rich dataset for machine learning applications.

By leveraging the framework's features, users can create complex interaction scenarios that closely mimic real-world group communication. This can prove to be a valuable asset in the fields of artificial intelligence, computational social science, and beyond.

---

### Frequently Asked Questions (FAQ)

**Q: Can the Swarms framework handle real-time interactions between agents?**

A: The Swarms framework is designed to simulate group chat environments. While it does not handle real-time interactions as they would occur on a network, it can simulate the agent of conversation in a way that mimics real-time communication.

**Q: Is the Swarms framework capable of natural language processing?**

A: The framework itself is focused on the structure and management of group chats. It does not inherently include natural language processing (NLP) capabilities. However, it can be integrated with NLP tools to enhance the simulation with language understanding and generation features.

**Q: Can I customize the roles and behaviors of agents within the framework?**

A: Yes, the framework is designed to be flexible. You can define custom roles and behaviors for agents to fit the specific requirements of your simulation scenario.

**Q: What are the limitations of the Swarms framework?**

A: The framework is constrained by its design to simulate text-based group chats. It is not suitable for voice or video communication simulations. Additionally, its effectiveness depends on the sophistication of the agents’ decision-making logic, which is outside the framework itself.

**Q: Is it possible to integrate the Swarms framework with other chat services?**

A: The framework is can be integrated with any chat services. However, it could potentially be adapted to work with chat service APIs, where the agents could be used to simulate user behavior within a real chat application.

**Q: How does the `GroupChatManager` select the next speaker?**

A: The `GroupChatManager` uses a selection mechanism, which is typically based on the conversation's context and the roles of the agents, to determine the next speaker. The specifics of this mechanism can be customized to match the desired agent of the conversation.

**Q: Can I contribute to the Swarms framework or suggest features?**

A: As with many open-source projects, contributions and feature suggestions can usually be made through the project's repository on platforms like GitHub. It's best to check with the maintainers of the Swarms framework for their contribution guidelines.

**Q: Are there any tutorials or community support for new users of the Swarms framework?**

A: Documentation and usage examples are provided with the framework. Community support may be available through forums, chat groups, or the platform where the framework is hosted. Tutorials may also be available from third-party educators or in official documentation.

**Q: What programming skills do I need to use the Swarms framework effectively?**

A: You should have a good understanding of Python programming, including experience with classes and methods. Familiarity with the principles of agent-based modeling and conversational AI would also be beneficial.
