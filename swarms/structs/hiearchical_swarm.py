import json
from typing import List

from beartype import beartype

from swarms.structs.agent import Agent
from swarms.structs.base_swarm import BaseSwarm
from swarms.utils.loguru_logger import logger


class HiearchicalSwarm(BaseSwarm):
    
    @beartype
    def __init__(
        self,
        director: Agent = None,
        agents: List[Agent] = None,
        max_loops: int = 1,
        long_term_memory_system: BaseSwarm = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.director = director
        self.agents = agents
        self.max_loops = max_loops
        self.long_term_memory_system = long_term_memory_system
        
        # Set the director to max_one loop
        self.director.max_loops = 1
        
        # Set the long term memory system of every agent to long term memory system
        if long_term_memory_system is True:
            for agent in agents:
                agent.long_term_memory = long_term_memory_system
                
        

    def parse_function_activate_agent(
        self, json_data: str = None, *args, **kwargs
    ):
        """
        Parse the JSON data and activate the selected agent.

        Args:
            json_data (str): The JSON data containing the agent name and task.

        Returns:
            str: The response from the activated agent.

        Raises:
            json.JSONDecodeError: If the JSON data is invalid.

        """
        try:
            data = json.loads(json_data)
            name = data.get("name")
            task = data.get("task")

            response = self.select_agent_and_send_task(
                name, task, *args, **kwargs
            )

            return response
        except json.JSONDecodeError:
            logger.error("Invalid JSON data, try again.")
            raise json.JSONDecodeError

    @beartype
    def select_agent_and_send_task(
        self, name: str = None, task: str = None, *args, **kwargs
    ):
        """
        Select an agent from the list and send a task to them.

        Args:
            name (str): The name of the agent to send the task to.
            task (str): The task to send to the agent.

        Returns:
            str: The response from the agent.

        Raises:
            KeyError: If the agent name is not found in the list of agents.

        """
        try:
            # Check to see if the agent name is in the list of agents
            if name in self.agents:
                agent = self.agents[name]
            else:
                return "Invalid agent name. Please select 'Account Management Agent' or 'Product Support Agent'."

            response = agent.run(task, *args, **kwargs)

            return response
        except Exception as e:
            logger.error(f"Error: {e}")
            raise e

    @beartype
    def run(self, task: str = None, *args, **kwargs):
        """
        Run the hierarchical swarm.

        Args:
            task (str): The task to send to the director agent.

        Returns:
            str: The response from the director agent.

        Raises:
            Exception: If an error occurs while running the swarm.

        """
        try:
            loop = 0
            
            # While the loop is less than max loops
            while loop < self.max_loops:
                # Run the director
                response = self.director.run(task, *args, **kwargs)

                # Run agents
                response = self.parse_function_activate_agent(response)

                loop += 1

            return response
        except Exception as e:
            logger.error(f"Error: {e}")
            raise e
