from typing import List

from pydantic import BaseModel

from swarms.structs.agent import Agent
from swarms.structs.concat import concat_strings
from loguru import logger
from swarms.structs.base_swarm import BaseSwarm
from swarms.structs.conversation import Conversation


class HierarchicalOrderCall(BaseModel):
    agent_name: str
    task: str


class CallTeam(BaseModel):
    calls: List[HierarchicalOrderCall]


class HiearchicalSwarm(BaseSwarm):
    def __init__(
        self,
        agents: List[Agent],
        director: Agent,
        name: str = "HierarchicalSwarm",
        description: str = "A swarm of agents that can be used to distribute tasks to a team of agents.",
        max_loops: int = 3,
        verbose: bool = True,
        create_agents_from_scratch: bool = False,
    ):
        super().__init__()
        self.agents = agents
        self.director = director
        self.max_loops = max_loops
        self.verbose = verbose
        self.name = name
        self.description = description
        self.create_agents_from_scratch = create_agents_from_scratch

        self.agents_check()
        self.director_check()

        # Initialize the conversation
        self.conversation = Conversation(
            time_enabled=True,
        )

        logger.info(f"Initialized {self.name} Hiearchical swarm")

    def agents_check(self):
        if len(self.agents) == 0:
            raise ValueError(
                "No agents found. Please add agents to the swarm."
            )
        return None

    def director_check(self):
        if self.director is None:
            raise ValueError(
                "No director found. Please add a director to the swarm."
            )
        return None

    def run(self, task: str):
        # Plan
        # Plan -> JSON Function call -> workers -> response fetch back to boss -> planner
        responses = []
        responses.append(task)

        for _ in range(self.max_loops):
            # Plan
            plan = self.planner.run(concat_strings(responses))
            logger.info(f"Agent {self.planner.agent_name} planned: {plan}")
            responses.append(plan)

            # Execute json function calls
            calls = self.director.run(plan)
            logger.info(
                f"Agent {self.director.agent_name} called: {calls}"
            )
            responses.append(calls)
            # Parse and send tasks to agents
            output = self.parse_then_send_tasks_to_agents(
                self.agents, calls
            )

            # Fetch back to boss
            responses.append(output)

        return concat_strings(responses)

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

    def agents_list(
        self,
    ) -> str:
        logger.info("Listing agents")

        for agent in self.agents:
            name = agent.agent_name
            description = agent.description or "No description available."
            logger.info(f"Agent: {name}, Description: {description}")
            self.conversation.add(name, description)

        return self.conversation.return_history_as_string()

    def parse_then_send_tasks_to_agents(self, response: dict):
        # Initialize an empty dictionary to store the output of each agent
        output = []

        # Loop over the tasks in the response
        for call in response["calls"]:
            name = call["agent_name"]
            task = call["task"]

            # Loop over the agents
            for agent in self.agents:
                # If the agent's name matches the name in the task, run the task
                if agent.agent_name == name:
                    out = agent.run(task)
                    print(out)

                    output.append(f"{name}: {out}")

                    # Store the output in the dictionary
                    # output[name] = out
                    break

        return output


# # Example usage:
# system_prompt = f"""
# You're a director agent, your responsibility is to serve the user efficiently, effectively and skillfully.You have a swarm of agents available to distribute tasks to, interact with the user and then submit tasks to the worker agents. Provide orders to the worker agents that are direct, explicit, and simple. Ensure that they are given tasks that are understandable, actionable, and simple to execute.


# ######
# Workers available:

# {agents_list(team)}


# """


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
