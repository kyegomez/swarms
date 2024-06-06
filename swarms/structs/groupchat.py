from typing import List
from swarms.structs.conversation import Conversation
from swarms.utils.loguru_logger import logger
from swarms.structs.agent import Agent
from swarms.structs.base_swarm import BaseSwarm


class GroupChat(BaseSwarm):
    """Manager class for a group chat.

    This class handles the management of a group chat, including initializing the conversation,
    selecting the next speaker, resetting the chat, and executing the chat rounds.

    Args:
        agents (List[Agent], optional): List of agents participating in the group chat. Defaults to None.
        max_rounds (int, optional): Maximum number of chat rounds. Defaults to 10.
        admin_name (str, optional): Name of the admin user. Defaults to "Admin".
        group_objective (str, optional): Objective of the group chat. Defaults to None.
        selector_agent (Agent, optional): Agent responsible for selecting the next speaker. Defaults to None.
        rules (str, optional): Rules for the group chat. Defaults to None.
        *args: Variable length argument list.
        **kwargs: Arbitrary keyword arguments.

    Attributes:
        agents (List[Agent]): List of agents participating in the group chat.
        max_rounds (int): Maximum number of chat rounds.
        admin_name (str): Name of the admin user.
        group_objective (str): Objective of the group chat.
        selector_agent (Agent): Agent responsible for selecting the next speaker.
        messages (Conversation): Conversation object for storing the chat messages.

    """

    def __init__(
        self,
        agents: List[Agent] = None,
        max_rounds: int = 10,
        admin_name: str = "Admin",
        group_objective: str = None,
        selector_agent: Agent = None,
        rules: str = None,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.agents = agents
        self.max_rounds = max_rounds
        self.admin_name = admin_name
        self.group_objective = group_objective
        self.selector_agent = selector_agent

        # Initialize the conversation
        self.message_history = Conversation(
            system_prompt=self.group_objective,
            time_enabled=True,
            user=self.admin_name,
            rules=rules,
            *args,
            **kwargs,
        )

        # Check to see if the agents is not None:
        if agents is None:
            raise ValueError(
                "Agents may not be empty please try again, add more agents!"
            )

    @property
    def agent_names(self) -> List[str]:
        """Return the names of the agents in the group chat."""
        return [agent.agent_name for agent in self.agents]

    def reset(self):
        """Reset the group chat."""
        logger.info("Resetting Groupchat")
        self.message_history.clear()

    def agent_by_name(self, name: str) -> Agent:
        """Find an agent whose name is contained within the given 'name' string.

        Args:
            name (str): Name string to search for.

        Returns:
            Agent: Agent object with a name contained in the given 'name' string.

        Raises:
            ValueError: If no agent is found with a name contained in the given 'name' string.

        """
        for agent in self.agents:
            if agent.agent_name in name:
                return agent
        raise ValueError(
            f"No agent found with a name contained in '{name}'."
        )

    def next_agent(self, agent: Agent) -> Agent:
        """Return the next agent in the list.

        Args:
            agent (Agent): Current agent.

        Returns:
            Agent: Next agent in the list.

        """
        return self.agents[
            (self.agent_names.index(agent.agent_name) + 1)
            % len(self.agents)
        ]

    def select_speaker_msg(self):
        """Return the message for selecting the next speaker."""
        prompt = f"""
        You are in a role play game. The following roles are available:
        {self._participant_roles()}.

        Read the following conversation.
        Then select the next role from {self.agent_names} to play. Only return the role.
        """
        return prompt

    # @try_except_wrapper
    def select_speaker(
        self, last_speaker_agent: Agent, selector_agent: Agent
    ):
        """Select the next speaker.

        Args:
            last_speaker_agent (Agent): Last speaker in the conversation.
            selector_agent (Agent): Agent responsible for selecting the next speaker.

        Returns:
            Agent: Next speaker.

        """
        logger.info("Selecting a New Speaker")
        selector_agent.system_prompt = self.select_speaker_msg()

        # Warn if GroupChat is underpopulated, without established changing behavior
        n_agents = len(self.agent_names)
        if n_agents < 3:
            logger.warning(
                f"GroupChat is underpopulated with {n_agents} agents."
                " Direct communication would be more efficient."
            )

        self.message_history.add(
            role=self.admin_name,
            content=f"Read the above conversation. Then select the next most suitable role from {self.agent_names} to play. Only return the role.",
        )

        name = selector_agent.run(
            self.message_history.return_history_as_string()
        )
        try:
            name = self.agent_by_name(name)
            print(name)
            return name
        except ValueError:
            return self.next_agent(last_speaker_agent)

    def _participant_roles(self):
        """Print the roles of the participants.

        Returns:
            str: Participant roles.

        """
        return "\n".join(
            [
                f"{agent.agent_name}: {agent.system_prompt}"
                for agent in self.agents
            ]
        )

    def __call__(self, task: str, *args, **kwargs):
        """Call 'GroupChatManager' instance as a function.

        Args:
            task (str): Task to be performed.

        Returns:
            str: Reply from the last speaker.

        """
        try:
            logger.info(
                f"Activating Groupchat with {len(self.agents)} Agents"
            )

            # Message History
            self.message_history.add(self.selector_agent.agent_name, task)

            # Message
            for i in range(self.max_rounds):
                speaker_agent = self.select_speaker(
                    last_speaker_agent=self.selector_agent,
                    selector_agent=self.selector_agent,
                )

                logger.info(
                    f"Next speaker selected: {speaker_agent.agent_name}"
                )

                # Reply back to the input prompt
                reply = speaker_agent.run(
                    self.message_history.return_history_as_string(),
                    *args,
                    **kwargs,
                )

                # Message History
                self.message_history.add(speaker_agent.agent_name, reply)
                print(reply)

                if i == self.max_rounds - 1:
                    break

            return reply
        except Exception as error:
            logger.error(
                f"Error detected: {error} Try optimizing the inputs and then submit an issue into the swarms github, so we can help and assist you."
            )
            raise error
