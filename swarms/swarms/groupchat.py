from dataclasses import dataclass
import sys
from typing import Dict, List, Optional, Union
import logging

from .. import Flow

logger = logging.getLogger(__name__)
from swarms.agents import SimpleAgent
from termcolor import colored


class GroupChat:
    """A group chat class that contains a list of agents and the maximum number of rounds."""

    agents: List[Flow]
    messages: List[Dict]
    max_round: int = 10
    admin_name: str = "Admin"  # the name of the admin agent

    @property
    def agent_names(self) -> List[str]:
        """Return the names of the agents in the group chat."""
        return [agent.name for agent in self.agents]

    def reset(self):
        """Reset the group chat."""
        self.messages.clear()

    def agent_by_name(self, name: str) -> Flow:
        """Find the next speaker based on the message."""
        return self.agents[self.agent_names.index(name)]

    def next_agent(self, agent: Flow) -> Flow:
        """Return the next agent in the list."""
        return self.agents[(self.agent_names.index(agent.name) + 1) % len(self.agents)]

    def select_speaker_msg(self):
        """Return the message for selecting the next speaker."""
        return f"""You are in a role play game. The following roles are available:
{self._participant_roles()}.

Read the following conversation.
Then select the next role from {self.agent_names} to play. Only return the role."""

    def select_speaker(self, last_speaker: Flow, selector: Flow):
        """Select the next speaker."""
        selector.update_system_message(self.select_speaker_msg())

        # Warn if GroupChat is underpopulated, without established changing behavior
        n_agents = len(self.agent_names)
        if n_agents < 3:
            logger.warning(
                f"GroupChat is underpopulated with {n_agents} agents. Direct communication would be more efficient."
            )

        final, name = selector.generate_oai_reply(
            self.messages
            + [
                {
                    "role": "system",
                    "content": f"Read the above conversation. Then select the next role from {self.agent_names} to play. Only return the role.",
                }
            ]
        )
        if not final:
            # i = self._random.randint(0, len(self._agent_names) - 1)  # randomly pick an id
            return self.next_agent(last_speaker)
        try:
            return self.agent_by_name(name)
        except ValueError:
            return self.next_agent(last_speaker)

    def _participant_roles(self):
        return "\n".join([f"{agent.name}: {agent.system_message}" for agent in self.agents])


class GroupChatManager(Flow):
    """(In preview) A chat manager agent that can manage a group chat of multiple agents."""

    def __init__(
        self,
        groupchat: GroupChat,
        name: Optional[str] = "chat_manager",
        # unlimited consecutive auto reply by default
        max_consecutive_auto_reply: Optional[int] = sys.maxsize,
        human_input_mode: Optional[str] = "NEVER",
        system_message: Optional[str] = "Group chat manager.",
        # seed: Optional[int] = 4,
        **kwargs,
    ):
        super().__init__(
            name=name,
            max_consecutive_auto_reply=max_consecutive_auto_reply,
            human_input_mode=human_input_mode,
            system_message=system_message,
            **kwargs,
        )
        self.register_reply(Flow, GroupChatManager.run_chat, config=groupchat, reset_config=GroupChat.reset)
        # self._random = random.Random(seed)

    def run_chat(
        self,
        messages: Optional[List[Dict]] = None,
        sender: Optional[Flow] = None,
        config: Optional[GroupChat] = None,
    ) -> Union[str, Dict, None]:
        """Run a group chat."""
        if messages is None:
            messages = self._oai_messages[sender]
        message = messages[-1]
        speaker = sender
        groupchat = config
        for i in range(groupchat.max_round):
            # set the name to speaker's name if the role is not function
            if message["role"] != "function":
                message["name"] = speaker.name
            groupchat.messages.append(message)
            # broadcast the message to all agents except the speaker
            for agent in groupchat.agents:
                if agent != speaker:
                    self.send(message, agent, request_reply=False, silent=True)
            if i == groupchat.max_round - 1:
                # the last round
                break
            try:
                # select the next speaker
                speaker = groupchat.select_speaker(speaker, self)
                # let the speaker speak
                reply = speaker.generate_reply(sender=self)
            except KeyboardInterrupt:
                # let the admin agent speak if interrupted
                if groupchat.admin_name in groupchat.agent_names:
                    # admin agent is one of the participants
                    speaker = groupchat.agent_by_name(groupchat.admin_name)
                    reply = speaker.generate_reply(sender=self)
                else:
                    # admin agent is not found in the participants
                    raise
            if reply is None:
                break
            # The speaker sends the message without requesting a reply
            speaker.send(reply, self, request_reply=False)
            message = self.last_message(speaker)
        return True, None
    """
    Groupchat

    Args:
        agents (list): List of agents
        dashboard (bool): Whether to print a dashboard or not

    Example:
    >>> from swarms.structs import Flow
    >>> from swarms.models import OpenAIChat
    >>> from swarms.swarms.groupchat import GroupChat
    >>> from swarms.agents import SimpleAgent
    >>> api_key = ""
    >>> llm = OpenAIChat()
    >>> agent1 = SimpleAgent("Captain Price", Flow(llm=llm, max_loops=4))
    >>> agent2 = SimpleAgent("John Mactavis", Flow(llm=llm, max_loops=4))
    >>> chat = GroupChat([agent1, agent2])
    >>> chat.assign_duty(agent1.name, "Buy the groceries")
    >>> chat.assign_duty(agent2.name, "Clean the house")
    >>> response = chat.run("Captain Price", "Hello, how are you John?")
    >>> print(response)



    """

    def __init__(self, agents, dashboard: bool = False):
        # Ensure that all provided agents are instances of simpleagents
        if not all(isinstance(agent, SimpleAgent) for agent in agents):
            raise ValueError("All agents must be instances of SimpleAgent")
        self.agents = {agent.name: agent for agent in agents}

        # Dictionary to store duties for each agent
        self.duties = {}

        # Dictionary to store roles for each agent
        self.roles = {}

        self.dashboard = dashboard

    def assign_duty(self, agent_name, duty):
        """Assigns duty to the agent"""
        if agent_name not in self.agents:
            raise ValueError(f"No agent named {agent_name} found.")

    def assign_role(self, agent_name, role):
        """Assigns a role to the specified agent"""
        if agent_name not in self.agents:
            raise ValueError(f"No agent named {agent_name} found")

        self.roles[agent_name] = role

    def run(self, sender_name: str, message: str):
        """Runs the groupchat"""
        if self.dashboard:
            metrics = print(
                colored(
                    f"""
            
            Groupchat Configuration:
            ------------------------
                                    
            Agents: {self.agents}
            Message: {message}
            Sender: {sender_name}
            """,
                    "red",
                )
            )

            print(metrics)

        responses = {}
        for agent_name, agent in self.agents.items():
            if agent_name != sender_name:
                if agent_name in self.duties:
                    message += f"Your duty is {self.duties[agent_name]}"
                if agent_name in self.roles:
                    message += (
                        f"You are the {self.roles[agent_name]} in this conversation"
                    )

                responses[agent_name] = agent.run(message)
        return responses
