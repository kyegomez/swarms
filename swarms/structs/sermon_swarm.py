from typing import Union, Sequence, List, Callable
from swarms.structs.agent import Agent
from swarms.structs.base_swarm import BaseSwarm


class SermonSwarm(BaseSwarm):
    """
    Represents a swarm of agents that communicate through sermons.

    Args:
        priest (Agent): The priest agent responsible for generating sermons.
        agents (Sequence[Agent]): The list of agents in the swarm.
        max_loops (int, optional): The maximum number of loops to run the agents. Defaults to 5.
        stop_condition (Union[str, List[str]], optional): The condition(s) that can stop the agents.
            Defaults to "stop".
        stop_function (Union[None, Callable], optional): The function to apply to the sermons before
            checking the stop condition. Defaults to None.
    """

    def __init__(
        self,
        priest: Agent,
        agents: Sequence[Agent],
        max_loops: int = 5,
        stop_condition: Union[str, List[str]] = "stop",
        stop_function: Union[None, Callable] = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.priest = priest
        self.agents = agents
        self.max_loops = max_loops
        self.stop_condition = stop_condition
        self.stop_function = stop_function

    def run(self, task: str, *args, **kwargs):
        """
        Runs the swarm by generating sermons from the priest and executing the task on each agent.

        Args:
            task (str): The task to be executed by the agents.
            *args: Additional positional arguments for the task.
            **kwargs: Additional keyword arguments for the task.
        """
        sermon = self.priest(task, *args, **kwargs)

        # Add the sermon to the memory of all agents
        for agent in self.agents:
            agent.add_message_to_memory(sermon)

        # Then run the agents
        loop = 0
        # for _ in range(self.max_loops):
        while loop < self.max_loops:
            for agent in self.agents:
                preach = agent.run(task, *args, **kwargs)

                if self.stop_function:
                    preach = self.stop_function(preach)

                if self.stop_condition in preach:
                    if self.stop_condition is True:
                        break

                    elif self.stop_condition in preach:
                        break

            loop += 1
            return preach
