from typing import List

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.logger import logger
from swarms.structs.base_multiagent_structure import (
    BaseMultiAgentStructure,
)


class StackOverflowSwarm(BaseMultiAgentStructure):
    """
    Represents a swarm of agents that work together to solve a problem or answer a question on Stack Overflow.

    Attributes:
        agents (List[Agent]): The list of agents in the swarm.
        autosave (bool): Flag indicating whether to automatically save the conversation.
        verbose (bool): Flag indicating whether to display verbose output.
        save_filepath (str): The filepath to save the conversation.
        conversation (Conversation): The conversation object for storing the interactions.

    Examples:
    >>> from swarms.structs.agent import Agent
    >>> from swarms.structs.stack_overflow_swarm import StackOverflowSwarm
    """

    def __init__(
        self,
        agents: List[Agent],
        autosave: bool = False,
        verbose: bool = False,
        save_filepath: str = "stack_overflow_swarm.json",
        eval_agent: Agent = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.agents = agents
        self.autosave = autosave
        self.verbose = verbose
        self.save_filepath = save_filepath
        self.eval_agent = eval_agent

        # Configure conversation
        self.conversation = Conversation(
            time_enabled=True,
            autosave=autosave,
            save_filepath=save_filepath,
            *args,
            **kwargs,
        )

        # Counter for the number of upvotes per post
        self.upvotes = 0
        
        # Counter for the number of downvotes per post
        self.downvotes = 0
        
        # Forum for the agents to interact
        self.forum = []
        
        

    def run(self, task: str, *args, **kwargs):
        """
        Run the swarm to solve a problem or answer a question like stack overflow

        Args:
            task (str): The task to be performed by the agents.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            List[str]: The conversation history.
        """
        # Add the task to the conversation
        self.conversation.add("Human", task)
        logger.info(f"Task: {task} Added to the Forum.")

        # Run the agents and get their responses and append to the conversation
        for agent in self.agents:
            response = agent.run(
                self.conversation.return_history_as_string(),
                *args,
                **kwargs,
            )
            # Add to the conversation
            self.conversation.add(
                agent.ai_name, f"{response}"
            )
            logger.info(f"[{agent.ai_name}]: [{response}]")
            
        return self.conversation.return_history_as_string()
