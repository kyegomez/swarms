"""

Boss -> json containig orders in JSON -> list of agents -> send orders to every agent


# Requirements
- Boss needs to know which agents are available [PROMPTING]
- Boss needs to output json commands sending tasks to every agent with the task and name
- Worker agents need to return a response to the boss
-> Boss returns the final output to the user
"""

import json
from typing import List
from swarms.structs.agent import Agent
from swarms.structs.base_swarm import BaseSwarm
from swarms.utils.loguru_logger import logger
from pydantic import BaseModel, Field
from swarms.structs.conversation import Conversation


class HiearchicalRequestDict(BaseModel):
    task: str = Field(
        None,
        title="Task",
        description="The task to send to the director agent.",
    )
    agent_name: str = Field(
        None,
        title="Agent Name",
        description="The name of the agent to send the task to.",
    )

    class Config:
        schema_extra = {
            "example": {
                "task": "task",
                "agent_name": "agent_name",
            }
        }


class HiearchicalSwarm(BaseSwarm):
    """
    A class representing a hierarchical swarm.

    Attributes:
        name (str): The name of the hierarchical swarm.
        description (str): The description of the hierarchical swarm.
        director (Agent): The director agent of the hierarchical swarm.
        agents (List[Agent]): The list of agents in the hierarchical swarm.
        max_loops (int): The maximum number of loops to run the swarm.
        long_term_memory_system (BaseSwarm): The long term memory system of the swarm.
        custom_parse_function (callable): A custom parse function for the swarm.

    Methods:
        swarm_initialization(*args, **kwargs): Initializes the hierarchical swarm.
        find_agent_by_name(agent_name: str = None, *args, **kwargs): Finds an agent in the swarm by name.
        parse_function_activate_agent(json_data: str = None, *args, **kwargs): Parses JSON data and activates the selected agent.
        select_agent_and_send_task(name: str = None, task: str = None, *args, **kwargs): Selects an agent and sends a task to them.
        run(task: str = None, *args, **kwargs): Runs the hierarchical swarm.

    """

    def __init__(
        self,
        name: str = None,
        description: str = None,
        director: Agent = None,
        agents: List[Agent] = None,
        max_loops: int = 1,
        long_term_memory_system: BaseSwarm = None,
        custom_parse_function: callable = None,
        rules: str = None,
        custom_director_prompt: str = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.name = name
        self.description = description
        self.director = director
        self.agents = agents
        self.max_loops = max_loops
        self.long_term_memory_system = long_term_memory_system
        self.custom_parse_function = custom_parse_function
        self.rules = rules
        self.custom_director_prompt = custom_director_prompt

        # Check to see agents is not empty
        self.agent_error_handling_check()

        # Set the director to max_one loop
        if self.director.max_loops > 1:
            self.director.max_loops = 1

        # Set the long term memory system of every agent to long term memory system
        if long_term_memory_system is True:
            for agent in agents:
                agent.long_term_memory = long_term_memory_system

        # Initialize the swarm
        self.swarm_initialization()

        # Initialize the conversation message pool
        self.swarm_history = Conversation(
            time_enabled=True, *args, **kwargs
        )

        # Set the worker agents as tools for the director
        for agent in self.agents:
            self.director.add_tool(agent)

        # Set the has prompt for the director,
        if custom_director_prompt is not None:
            self.director.system_prompt = custom_director_prompt
        else:
            self.director.system_prompt = self.has_sop()

    def swarm_initialization(self, *args, **kwargs):
        """
        Initializes the hierarchical swarm.

        Args:
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            None

        """
        logger.info(f"Initializing the hierarchical swarm: {self.name}")
        logger.info(f"Purpose of this swarm: {self.description}")

        # Now log number of agnets and their names
        logger.info(f"Number of agents: {len(self.agents)}")
        logger.info(
            f"Agent names: {[agent.name for agent in self.agents]}"
        )

        # Now see if agents is not empty
        if len(self.agents) == 0:
            logger.info("No agents found. Please add agents to the swarm.")
            return None

        # Now see if director is not empty
        if self.director is None:
            logger.info(
                "No director found. Please add a director to the swarm."
            )
            return None

        logger.info(
            f"Initialization complete for the hierarchical swarm: {self.name}"
        )

    def agent_error_handling_check(self):
        """
        Check if the agents list is not empty.

        Returns:
            None

        Raises:
            ValueError: If the agents list is empty.

        """
        if len(self.agents) == 0:
            raise ValueError(
                "No agents found. Please add agents to the swarm."
            )
        return None

    def find_agent_by_name(self, agent_name: str = None, *args, **kwargs):
        """
        Finds an agent in the swarm by name.

        Args:
            agent_name (str): The name of the agent to find.

        Returns:
            Agent: The agent with the specified name, or None if not found.

        """
        for agent in self.agents:
            if agent.name == agent_name:
                return agent
        return None

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

            # Check if the data is a list of agent task pairs
            if isinstance(data, list):
                responses = []
                # Iterate over the list of agent task pairs
                for agent_task in data:
                    name = agent_task.get("name")
                    task = agent_task.get("task")

                    response = self.select_agent_and_send_task(
                        name, task, *args, **kwargs
                    )

                    responses.append(response)
                return responses
            else:
                name = data.get("name")
                task = data.get("task")

                response = self.select_agent_and_send_task(
                    name, task, *args, **kwargs
                )

                return response
        except json.JSONDecodeError:
            logger.error("Invalid JSON data, try again.")
            raise json.JSONDecodeError

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

                # Log the director's response
                self.swarm_history.add(self.director.agent_name, response)

                # Run agents
                if self.custom_parse_function is not None:
                    response = self.custom_parse_function(response)
                else:
                    response = self.parse_function_activate_agent(response)

                loop += 1

                task = response

            return response
        except Exception as e:
            logger.error(f"Error: {e}")
            raise e

    def run_worker_agent(
        self, name: str = None, task: str = None, *args, **kwargs
    ):
        """
        Run the worker agent.

        Args:
            name (str): The name of the worker agent.
            task (str): The task to send to the worker agent.

        Returns:
            str: The response from the worker agent.

        Raises:
            Exception: If an error occurs while running the worker agent.

        """
        try:
            # Find the agent by name
            agent = self.find_agent_by_name(name)

            # Run the agent
            response = agent.run(task, *args, **kwargs)

            return response
        except Exception as e:
            logger.error(f"Error: {e}")
            raise e

    def has_sop(self):
        # We need to check the name of the agents and their description or system prompt
        # TODO: Provide many shot examples of the agents available and even maybe what tools they have access to
        # TODO: Provide better reasoning prompt tiles, such as when do you use a certain agent and specific
        # Things NOT to do.
        return f"""
        
        You're a director boss agent orchestrating worker agents with tasks. Select an agent most relevant to 
        the input task and give them a task. If there is not an agent relevant to the input task then say so and be simple and direct.
        These are the available agents available call them if you need them for a specific 
        task or operation:
        
        Number of agents: {len(self.agents)}
        Agents Available: {
            [
                {"name": agent.name, "description": agent.system_prompt}
                for agent in self.agents
            ]
        }
    
        """
