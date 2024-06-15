# GroupChatManager
Documentation:

The `GroupChatManager` class is designed for managing group chat interactions between agents. It allows you to create and manage group chats among multiple agents. The `GroupChatManager` requires two main arguments - the `groupchat` of type `GroupChat` which indicates the actual group chat object and `selector` of type `Agent` which specifies the agent who is the selector or the initiator of the chat.

This class provides a variety of features and functions such as maintaining and appending messages, managing the communication rounds, interacting between different agents and extracting replies.

Args:

| Parameter | Type         | Description                                      |
|-----------|--------------|--------------------------------------------------|
| groupchat | `GroupChat`  | The group chat object where the conversation occurs. |
| selector  | `Agent`      | The agent who is the selector or the initiator of the chat. |
Usage:

```python
from swarms import GroupChatManager
from swarms.structs.agent import Agent

# Create an instance of Agent
agents = Agent()

# Initialize GroupChatManager with an existing GroupChat instance and an agent
manager = GroupChatManager(groupchat, selector)

# Call the group chat manager passing a specific chat task
result = manager("Discuss the agenda for the upcoming meeting")
```

Explanation:

1. First, you import the `GroupChatManager` class and the `Agent` class from the `swarms` library.

2. Then, you create an instance of the `Agent`.

3. After that, you initialize the `GroupChatManager` with an existing `GroupChat` instance and an agent.

4. Finally, you call the group chat manager, passing a specific chat task and receive the response.

Source Code:

```python
class GroupChatManager:
    """
    GroupChatManager

    Args:
        groupchat: GroupChat
        selector: Agent

    Usage:
    >>> from swarms import GroupChatManager
    >>> from swarms.structs.agent import Agent
    >>> agents = Agent()
    """

    def __init__(self, groupchat: GroupChat, selector: Agent):
        self.groupchat = groupchat
        self.selector = selector

    def __call__(self, task: str):
        """Call 'GroupChatManager' instance as a function.

        Args:
            task (str): The task to be performed during the group chat.

        Returns:
            str: The response from the group chat.
        """
        self.groupchat.messages.append({"role": self.selector.name, "content": task})
        for i in range(self.groupchat.max_round):
            speaker = self.groupchat.select_speaker(
                last_speaker=self.selector, selector=self.selector
            )
            reply = speaker.generate_reply(
                self.groupchat.format_history(self.groupchat.messages)
            )
            self.groupchat.messages.append(reply)
            print(reply)
            if i == self.groupchat.max_round - 1:
                break

        return reply
```

The `GroupChatManager` class has an `__init__` method which takes `groupchat` and `selector` as arguments to initialize the class properties. It also has a `__call__` method to perform the group chat task and provide the appropriate response.

In the `__call__` method, it appends the message with the speaker’s role and their content. It then iterates over the communication rounds, selects speakers, generates replies and appends messages to the group chat. Finally, it returns the response.

The above example demonstrates how to use the `GroupChatManager` class to manage group chat interactions. You can further customize this class based on specific requirements and extend its functionality as needed.
