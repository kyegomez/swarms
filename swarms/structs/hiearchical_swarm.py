from concurrent.futures import ThreadPoolExecutor
from typing import Any, List, Optional, Union

from pydantic import BaseModel, Field

from swarms.structs.agent import Agent
from swarms.structs.base_swarm import BaseSwarm
from swarms.structs.conversation import Conversation
from swarms.utils.formatter import formatter

from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="hierarchical_swarm")


class HierarchicalOrder(BaseModel):
    agent_name: str = Field(
        ...,
        description="Specifies the name of the agent to which the task is assigned. This is a crucial element in the hierarchical structure of the swarm, as it determines the specific agent responsible for the task execution.",
    )
    task: str = Field(
        ...,
        description="Defines the specific task to be executed by the assigned agent. This task is a key component of the swarm's plan and is essential for achieving the swarm's goals.",
    )


class SwarmSpec(BaseModel):
    goals: str = Field(
        ...,
        description="The goal of the swarm. This is the overarching objective that the swarm is designed to achieve. It guides the swarm's plan and the tasks assigned to the agents.",
    )
    plan: str = Field(
        ...,
        description="Outlines the sequence of actions to be taken by the swarm. This plan is a detailed roadmap that guides the swarm's behavior and decision-making.",
    )
    rules: str = Field(
        ...,
        description="Defines the governing principles for swarm behavior and decision-making. These rules are the foundation of the swarm's operations and ensure that the swarm operates in a coordinated and efficient manner.",
    )
    orders: List[HierarchicalOrder] = Field(
        ...,
        description="A collection of task assignments to specific agents within the swarm. These orders are the specific instructions that guide the agents in their task execution and are a key element in the swarm's plan.",
    )


class HierarchicalSwarm(BaseSwarm):
    """
    Represents a hierarchical swarm of agents, with a director that orchestrates tasks among the agents.
    """

    def __init__(
        self,
        name: str = "HierarchicalAgentSwarm",
        description: str = "Distributed task swarm",
        director: Optional[Union[Agent, Any]] = None,
        agents: List[Union[Agent, Any]] = None,
        max_loops: int = 1,
        return_all_history: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initializes the HierarchicalSwarm with the given parameters.

        :param name: The name of the swarm.
        :param description: A description of the swarm.
        :param director: The director agent that orchestrates tasks.
        :param agents: A list of agents within the swarm.
        :param max_loops: The maximum number of feedback loops between the director and agents.
        :param return_all_history: A flag indicating whether to return all conversation history.
        """
        super().__init__(
            name=name,
            description=description,
            agents=agents,
        )
        self.director = director
        self.agents = agents
        self.max_loops = max_loops
        self.return_all_history = return_all_history
        self.conversation = Conversation(time_enabled=True)

        self.add_name_and_description()

        self.check_agents()

        self.list_all_agents()

    def check_agents(self):
        """
        Checks if there are any agents and a director set for the swarm.
        Raises ValueError if either condition is not met.
        """
        if not self.agents:
            raise ValueError(
                "No agents found in the swarm. At least one agent must be provided to create a hierarchical swarm."
            )

        if not self.director:
            raise ValueError(
                "Director not set for the swarm. A director agent is required to coordinate and orchestrate tasks among the agents."
            )

    def run_director(
        self, task: str, img: str = None, *args, **kwargs
    ):
        """
        Runs a task through the director agent.

        :param task: The task to be executed by the director.
        :param img: Optional image to be used with the task.
        :return: The output of the director's task execution.
        """

        function_call = self.director.run(
            task=f"History: {self.conversation.get_str()} Your Task: {task}",
        )

        formatter.print_panel(f"Director Output: {function_call}")

        return function_call

    def run(self, task: str, img: str = None, *args, **kwargs) -> str:
        """
        Runs a task through the swarm, involving the director and agents.

        :param task: The task to be executed by the swarm.
        :param img: Optional image to be used with the task.
        :return: The output of the swarm's task execution.
        """
        self.conversation.add(role="User", content=f"Task: {task}")

        function_call = self.run_director(
            task=self.conversation.get_str()
        )

        self.parse_orders(function_call)

        if self.return_all_history:
            return self.conversation.get_str()
        else:
            return self.conversation.get_str()

    def add_name_and_description(self):
        """
        Adds the swarm's name and description to the conversation.
        """
        self.conversation.add(
            role="User",
            content=f"\n Swarm Name: {self.name} \n Swarm Description: {self.description}",
        )

        formatter.print_panel(
            f"⚡ INITIALIZING HIERARCHICAL SWARM UNIT: {self.name}\n"
            f"🔒 CLASSIFIED DIRECTIVE: {self.description}\n"
            f"📡 STATUS: ACTIVATING SWARM PROTOCOLS\n"
            f"🌐 ESTABLISHING SECURE AGENT MESH NETWORK\n"
            f"⚠️ CYBERSECURITY MEASURES ENGAGED\n",
            title="SWARM CORPORATION - HIERARCHICAL SWARMS ACTIVATING...",
        )

    def list_all_agents(self) -> str:
        """
        Lists all agents available in the swarm.

        :return: A string representation of all agents in the swarm.
        """

        # need to fetch name and description of all agents
        all_agents = "\n".join(
            f"Agent: {agent.agent_name} || Description: {agent.description or agent.system_prompt} \n"
            for agent in self.agents
        )

        self.conversation.add(
            role="User",
            content=f"All Agents Available in the Swarm {self.name}: \n {all_agents}",
        )

        formatter.print_panel(
            all_agents, title="All Agents Available in the Swarm"
        )

    def find_agent(self, name: str) -> Optional[Agent]:
        """
        Finds an agent by its name within the swarm.

        :param name: The name of the agent to find.
        :return: The agent if found, otherwise None.
        """
        for agent in self.agents:
            if agent.agent_name == name:
                return agent
        return None

    def run_agent(self, agent_name: str, task: str, img: str = None):
        """
        Runs a task through a specific agent.

        :param agent_name: The name of the agent to execute the task.
        :param task: The task to be executed by the agent.
        :param img: Optional image to be used with the task.
        :return: The output of the agent's task execution.
        """
        try:
            agent = self.find_agent(agent_name)

            if agent:
                out = agent.run(
                    task=f"History: {self.conversation.get_str()} Your Task: {task}",
                    img=img,
                )

                self.conversation.add(
                    role=agent_name,
                    content=out,
                )

                return out
            else:
                logger.error(
                    f"Agent {agent_name} not found in the swarm {self.name}"
                )
        except Exception as e:
            logger.error(f"Error running agent {agent_name}: {e}")
            return "Error running agent"

    def parse_orders(self, orders: SwarmSpec) -> None:
        """
        Parses the orders from the SwarmSpec and executes them through the agents.

        :param orders: The SwarmSpec containing the orders to be parsed.
        """
        self.add_goal_and_more_in_conversation(orders)

        orders_list = self.parse_swarm_spec(orders)

        try:

            # Example of passing the parsed data to an agent
            for order in orders_list:
                out = self.run_agent(
                    agent_name=order.agent_name,
                    task=order.task,
                )

            return out
        except Exception as e:
            logger.error(f"Error parsing orders: {e}")
            return "Error parsing orders"

    def parse_swarm_spec(self, swarm_spec: SwarmSpec) -> None:
        """
        Parses the SwarmSpec to extract the orders.

        :param swarm_spec: The SwarmSpec to be parsed.
        :return: The list of orders extracted from the SwarmSpec.
        """
        orders_list = swarm_spec.orders

        # return the orders_list
        return orders_list

    def provide_feedback(self, agent_name: str, out: str) -> None:
        """
        Provides feedback to an agent based on its output.

        :param agent_name: The name of the agent to provide feedback to.
        :param out: The output of the agent to base the feedback on.
        """
        orders = self.director.run(
            task=f"Provide feedback to {agent_name} on their output: {out}"
        )

        orders_list = self.parse_swarm_spec(orders)

        for order in orders_list:
            out = self.run_agent(
                agent_name=order.agent_name,
                task=order.task,
            )

        return out

    def add_goal_and_more_in_conversation(
        self, swarm_spec: SwarmSpec
    ) -> None:
        """
        Adds the swarm's goals, plan, and rules to the conversation.

        :param swarm_spec: The SwarmSpec containing the goals, plan, and rules.
        """
        goals = swarm_spec.goals
        plan = swarm_spec.plan
        rules = swarm_spec.rules

        self.conversation.add(
            role="Director",
            content=f"Goals: {goals}\nPlan: {plan}\nRules: {rules}",
        )

    def batch_run(self, tasks: List[str]) -> List[str]:
        """
        Batch run the swarm with the given tasks.
        """
        return [self.run(task) for task in tasks]

    def concurrent_run(self, tasks: List[str]) -> List[str]:
        """
        Concurrent run the swarm with the given tasks.
        """
        with ThreadPoolExecutor(max_workers=len(tasks)) as executor:
            return list(executor.map(self.run, tasks))
