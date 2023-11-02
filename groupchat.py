# from swarms.structs import Flow
# from swarms.models import OpenAIChat
# from swarms.swarms.groupchat import GroupChat
# from swarms.agents import SimpleAgent

# api_key = ""

# llm = OpenAIChat(
#     openai_api_key=api_key,
# )

# agent1 = SimpleAgent("Captain Price", Flow(llm=llm, max_loops=4))
# agent2 = SimpleAgent("John Mactavis", Flow(llm=llm, max_loops=4))

# # Create a groupchat with the 2 agents
# chat = GroupChat([agent1, agent2])

# # Assign duties to the agents
# chat.assign_duty(agent1.name, "Buy the groceries")
# chat.assign_duty(agent2.name, "Clean the house")

# # Initate a chat
# response = chat.run("Captain Price", "Hello, how are you John?")
# print(response)


from swarms.models import OpenAIChat
from swarms.structs import Flow
import random

api_key = ""  # Your API Key here


class GroupChat:
    """
    GroupChat class that facilitates agent-to-agent communication using multiple instances of the Flow class.
    """

    def __init__(self, agents: list):
        self.agents = {f"agent_{i}": agent for i, agent in enumerate(agents)}
        self.message_log = []

    def add_agent(self, agent: Flow):
        agent_id = f"agent_{len(self.agents)}"
        self.agents[agent_id] = agent

    def remove_agent(self, agent_id: str):
        if agent_id in self.agents:
            del self.agents[agent_id]

    def send_message(self, sender_id: str, recipient_id: str, message: str):
        if sender_id not in self.agents or recipient_id not in self.agents:
            raise ValueError("Invalid sender or recipient ID.")
        formatted_message = f"{sender_id} to {recipient_id}: {message}"
        self.message_log.append(formatted_message)
        recipient_agent = self.agents[recipient_id]
        recipient_agent.run(message)

    def broadcast_message(self, sender_id: str, message: str):
        for agent_id, agent in self.agents.items():
            if agent_id != sender_id:
                self.send_message(sender_id, agent_id, message)

    def get_message_log(self):
        return self.message_log


class EnhancedGroupChatV2(GroupChat):
    def __init__(self, agents: list):
        super().__init__(agents)

    def multi_round_conversation(self, rounds: int = 5):
        """
        Initiate a multi-round conversation between agents.

        Args:
            rounds (int): The number of rounds of conversation.
        """
        for _ in range(rounds):
            # Randomly select a sender and recipient agent for the conversation
            sender_id = random.choice(list(self.agents.keys()))
            recipient_id = random.choice(list(self.agents.keys()))
            while recipient_id == sender_id:  # Ensure the recipient is not the sender
                recipient_id = random.choice(list(self.agents.keys()))

            # Generate a message (for simplicity, a generic message is used)
            message = f"Hello {recipient_id}, how are you today?"
            self.send_message(sender_id, recipient_id, message)


# Sample usage with EnhancedGroupChatV2
# Initialize the language model
llm = OpenAIChat(
    openai_api_key=api_key,
    temperature=0.5,
    max_tokens=3000,
)

# Initialize two Flow agents
agent1 = Flow(llm=llm, max_loops=5, dashboard=True)
agent2 = Flow(llm=llm, max_loops=5, dashboard=True)

# Create an enhanced group chat with the two agents
enhanced_group_chat_v2 = EnhancedGroupChatV2(agents=[agent1, agent2])

# Simulate multi-round agent to agent communication
enhanced_group_chat_v2.multi_round_conversation(rounds=5)

enhanced_group_chat_v2.get_message_log()  # Get the conversation log
