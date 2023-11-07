# Group Chat Module Documentation

## Introduction

The Swarms library is designed to orchestrate conversational agents powered by machine learning models, particularly large language models (LLMs) like the ones provided by OpenAI. These agents can engage in various types of interactions such as casual dialogues, problem-solving, and creative tasks like riddle generation. This documentation covers the `Flow` and `GroupChat` modules, which are fundamental parts of the Swarms library, allowing users to create and manage the flow of conversations between multiple agents and end-users.

The `Flow` class serves as a wrapper around the OpenAI's LLM, adding additional context and control to the interaction process. On the other hand, the `GroupChat` and `GroupChatManager` classes are responsible for managing a multi-agent conversation environment where each agent can contribute to the discussion based on its specialized role or functionality.

---

## Module Overview


### GroupChat Class

The `GroupChat` class orchestrates a conversation involving multiple agents. It allows for the aggregation of messages from different agents and manages the conversational rounds.

#### Parameters for GroupChat

| Parameter   | Type          | Description                                         | Default Value |
|-------------|---------------|-----------------------------------------------------|---------------|
| agents      | list of Flow  | A list of `Flow` instances participating in the chat.| Required      |
| messages    | list of str   | A list of pre-existing messages in the conversation.| []            |
| max_round   | int           | The maximum number of rounds in the group chat.     | 10            |

---

### GroupChatManager Class

The `GroupChatManager` handles the interaction between the group chat and an external agent, in this case, a manager. It determines which agent should respond in the group chat.

#### Parameters for GroupChatManager

| Parameter   | Type          | Description                                         | Default Value |
|-------------|---------------|-----------------------------------------------------|---------------|
| groupchat   | GroupChat     | An instance of the `GroupChat` class to be managed. | Required      |
| selector    | Flow          | The manager agent that oversees the group chat.     | Required      |

---

## Detailed Functionality and Usage

### Flow Class Functionality

The `Flow` class encapsulates the behavior of a single conversational agent. It interfaces with an OpenAI LLM to generate responses based on the `system_message` which sets the context, personality, or role for the agent. The `max_loops` parameter limits the number of times the agent will interact within a single conversational instance. The `name` parameter serves as an identifier, and the `dashboard` parameter allows for monitoring the conversation flow visually when set to `True`.

#### Usage Example for Flow

```python
from swarms import Flow, OpenAI

# Initialize the OpenAI instance with the required API key and parameters
llm = OpenAI(
    openai_api_key="your-api-key",
    temperature=0.5,
    max_tokens=3000,
)

# Create a Flow instance representing a "silly" character
silly_flow = Flow(
    llm=llm,
    max_loops=1,
    system_message="YOU ARE SILLY, YOU OFFER NOTHING OF VALUE",
    name='silly',
    dashboard=True,
)

# Use the Flow instance to get a response
response = silly_flow("Tell me a joke")
print(response)
```

### GroupChat Class Functionality

The `GroupChat` class manages a collection of `Flow` instances representing different agents. It orchestrates the conversation flow, ensuring each agent contributes to the conversation based on the `max_round` limit. The `messages` parameter can hold a historical list of all messages that have been part of the group chat conversation.

#### Usage Example for GroupChat

```python
from swarms import Flow, OpenAI
from swarms.swarms.groupchat import GroupChat

# Assuming llm and Flow instances have been initialized as shown above

# Initialize the group chat

 with multiple agents
group_chat = GroupChat(agents=[silly_flow, detective_flow, riddler_flow], messages=[], max_round=10)

# Simulate a round of conversation
for _ in range(10):
    message = input("User: ")
    response = group_chat.respond(message)
    print(f"Group Chat: {response}")
```

### GroupChatManager Class Functionality

The `GroupChatManager` class takes a `GroupChat` instance and a managing `Flow` instance (selector) and controls the selection process of which agent should respond at each step of the group chat conversation.

#### Usage Example for GroupChatManager

```python
from swarms import Flow, OpenAI
from swarms.swarms.groupchat import GroupChat, GroupChatManager

# Initialize the llm, Flow, and GroupChat instances as previously described

# Create the GroupChatManager instance
chat_manager = GroupChatManager(groupchat=group_chat, selector=manager_flow)

# Start the group chat managed conversation
chat_history = chat_manager.start_conversation("Write me a riddle")
print(chat_history)
```

---

## Additional Information and Tips

- When creating `Flow` instances, make sure the `system_message` aligns with the intended behavior or personality of the agent to ensure consistent interactions.
- The `max_loops` parameter in `Flow` and `max_round` in `GroupChat` are critical for controlling the length of interactions and should be set according to the expected conversation complexity.
- The `dashboard` parameter in `Flow` is useful for debugging and visualizing the agent's behavior during development but might not be needed in a production environment.
- Keep the `openai_api_key` secure and do not expose it in publicly accessible code.

---

## References and Further Reading

- OpenAI API documentation: [OpenAI API](https://beta.openai.com/docs/)
- PyTorch documentation for modules and classes: [PyTorch Docs](https://pytorch.org/docs/stable/index.html)

---

Please note that the given examples are simplified and the actual implementation details like error handling, API request management, and concurrency considerations need to be addressed in production code. The provided API key should be kept confidential and used in accordance with the terms of service of the API provider.