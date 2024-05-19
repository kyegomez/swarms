from dataclasses import dataclass, field
from typing import List
from swarms.structs.conversation import Conversation
from swarms.utils.loguru_logger import logger
from swarms.structs.agent import Agent
from swarms.structs.base_swarm import BaseSwarm


@dataclass
class GroupChat(BaseSwarm):
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

    agents: List[Agent] = field(default_factory=list)
    max_round: int = 10
    admin_name: str = "Admin"  # the name of the admin agent
    group_objective: str = field(default_factory=str)

    def __post_init__(self):
        self.messages = Conversation(
            system_prompt=self.group_objective,
            time_enabled=True,
            user=self.admin_name,
        )

    @property
    def agent_names(self) -> List[str]:
        """Return the names of the agents in the group chat."""
        return [agent.agent_name for agent in self.agents]

    def reset(self):
        """Reset the group chat."""
        logger.info("Resetting Groupchat")
        self.messages.clear()

    def agent_by_name(self, name: str) -> Agent:
        """Find an agent whose name is contained within the given 'name' string."""
        for agent in self.agents:
            if agent.agent_name in name:
                return agent
        raise ValueError(
            f"No agent found with a name contained in '{name}'."
        )

    def next_agent(self, agent: Agent) -> Agent:
        """Return the next agent in the list."""
        return self.agents[
            (self.agent_names.index(agent.agent_name) + 1)
            % len(self.agents)
        ]

    def select_speaker_msg(self):
        """Return the message for selecting the next speaker."""
        return f"""
        You are in a role play game. The following roles are available:
        {self._participant_roles()}.

        Read the following conversation.
        Then select the next role from {self.agent_names} to play. Only return the role.
        """

    # @try_except_wrapper
    def select_speaker(self, last_speaker: Agent, selector: Agent):
        """Select the next speaker."""
        logger.info("Selecting a New Speaker")
        selector.system_prompt = self.select_speaker_msg()

        # Warn if GroupChat is underpopulated, without established changing behavior
        n_agents = len(self.agent_names)
        if n_agents < 3:
            logger.warning(
                f"GroupChat is underpopulated with {n_agents} agents."
                " Direct communication would be more efficient."
            )

        self.messages.add(
            role=self.admin_name,
            content=f"Read the above conversation. Then select the next most suitable role from {self.agent_names} to play. Only return the role.",
        )

        name = selector.run(self.messages.return_history_as_string())
        try:
            name = self.agent_by_name(name)
            print(name)
            return name
        except ValueError:
            return self.next_agent(last_speaker)

    def _participant_roles(self):
        """Print the roles of the participants.

        Returns:
            _type_: _description_
        """
        return "\n".join(
            [
                f"{agent.agent_name}: {agent.system_prompt}"
                for agent in self.agents
            ]
        )


@dataclass
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

    groupchat: GroupChat
    selector: Agent

    # @try_except_wrapper
    def __call__(self, task: str):
        """Call 'GroupChatManager' instance as a function.

        Args:
            task (str): _description_

        Returns:
            _type_: _description_
        """
        logger.info(
            f"Activating Groupchat with {len(self.groupchat.agents)} Agents"
        )

        self.groupchat.messages.add(self.selector.agent_name, task)

        for i in range(self.groupchat.max_round):
            speaker = self.groupchat.select_speaker(
                last_speaker=self.selector, selector=self.selector
            )
            reply = speaker.run(
                self.groupchat.messages.return_history_as_string()
            )
            self.groupchat.messages.add(speaker.agent_name, reply)
            print(reply)
            if i == self.groupchat.max_round - 1:
                break

        return reply
