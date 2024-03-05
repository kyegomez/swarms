import yaml
import json
import asyncio
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Sequence
from swarms.utils.loguru_logger import logger
from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation

class AbstractSwarm(ABC):
    """
    Abstract Swarm Class for multi-agent systems

    Attributes:
        agents (List[Agent]): A list of agents
        max_loops (int): The maximum number of loops to run


    Methods:
        communicate: Communicate with the swarm through the orchestrator, protocols, and the universal communication layer
        run: Run the swarm
        step: Step the swarm
        add_agent: Add a agent to the swarm
        remove_agent: Remove a agent from the swarm
        broadcast: Broadcast a message to all agents
        reset: Reset the swarm
        plan: agents must individually plan using a workflow or pipeline
        direct_message: Send a direct message to a agent
        autoscaler: Autoscaler that acts like kubernetes for autonomous agents
        get_agent_by_id: Locate a agent by id
        get_agent_by_name: Locate a agent by name
        assign_task: Assign a task to a agent
        get_all_tasks: Get all tasks
        get_finished_tasks: Get all finished tasks
        get_pending_tasks: Get all pending tasks
        pause_agent: Pause a agent
        resume_agent: Resume a agent
        stop_agent: Stop a agent
        restart_agent: Restart agent
        scale_up: Scale up the number of agents
        scale_down: Scale down the number of agents
        scale_to: Scale to a specific number of agents
        get_all_agents: Get all agents
        get_swarm_size: Get the size of the swarm
        get_swarm_status: Get the status of the swarm
        save_swarm_state: Save the swarm state
        loop: Loop through the swarm
        run_async: Run the swarm asynchronously
        run_batch_async: Run the swarm asynchronously
        run_batch: Run the swarm asynchronously
        batched_run: Run the swarm asynchronously
        abatch_run: Asynchronous batch run with language model
        arun: Asynchronous run

    """

    # @abstractmethod
    def __init__(
        self, 
        agents: List[Agent], 
        max_loops: int = 200,
        callbacks: Optional[Sequence[callable]] = None,
        autosave: bool = False,
        logging: bool = False,
        return_metadata: bool = False,
        metadata_filename: str = "multiagent_structure_metadata.json",
        stopping_function: Optional[Callable] = None,
        stopping_condition: Optional[str] = "stop",
        stopping_condition_args: Optional[Dict] = None,
        *args,
        **kwargs,
    ):
        """Initialize the swarm with agents"""
        self.agents = agents
        self.max_loops = max_loops
        self.callbacks = callbacks
        self.autosave = autosave
        self.logging = logging
        self.return_metadata = return_metadata
        self.metadata_filename = metadata_filename
        self.conversation = Conversation(
            time_enabled=True, *args, **kwargs
        )

        # Handle the case where the agents are not provided
        # Handle agents
        for agent in self.agents:
            if not isinstance(agent, Agent):
                raise TypeError("Agents must be of type Agent.")

        if self.agents is None:
            self.agents = []

        # Handle the case where the callbacks are not provided
        if self.callbacks is None:
            self.callbacks = []

        # Handle the case where the autosave is not provided
        if self.autosave is None:
            self.autosave = False

        # Handle the case where the logging is not provided
        if self.logging is None:
            self.logging = False

        # Handle callbacks
        if callbacks is not None:
            for callback in self.callbacks:
                if not callable(callback):
                    raise TypeError("Callback must be callable.")

        # Handle autosave
        if autosave:
            self.save_to_json(metadata_filename)

    # @abstractmethod
    def communicate(self):
        """Communicate with the swarm through the orchestrator, protocols, and the universal communication layer"""

    # @abstractmethod
    def run(self):
        """Run the swarm"""

    def __call__(
        self,
        task,
        *args,
        **kwargs,
    ):
        """Call self as a function

        Args:
            task (_type_): _description_

        Returns:
            _type_: _description_
        """
        return self.run(task, *args, **kwargs)

    def step(self):
        """Step the swarm"""
        
        

    # @abstractmethod
    def add_agent(self, agent: "Agent"):
        """Add a agent to the swarm"""

    # @abstractmethod
    def remove_agent(self, agent: "Agent"):
        """Remove a agent from the swarm"""

    # @abstractmethod
    def broadcast(
        self, message: str, sender: Optional["Agent"] = None
    ):
        """Broadcast a message to all agents"""

    # @abstractmethod
    def reset(self):
        """Reset the swarm"""

    # @abstractmethod
    def plan(self, task: str):
        """agents must individually plan using a workflow or pipeline"""

    # @abstractmethod
    def direct_message(
        self,
        message: str,
        sender: "Agent",
        recipient: "Agent",
    ):
        """Send a direct message to a agent"""

    # @abstractmethod
    def autoscaler(self, num_agents: int, agent: ["Agent"]):
        """Autoscaler that acts like kubernetes for autonomous agents"""

    # @abstractmethod
    def get_agent_by_id(self, id: str) -> "Agent":
        """Locate a agent by id"""

    # @abstractmethod
    def get_agent_by_name(self, name: str) -> "Agent":
        """Locate a agent by name"""

    # @abstractmethod
    def assign_task(self, agent: "Agent", task: Any) -> Dict:
        """Assign a task to a agent"""

    # @abstractmethod
    def get_all_tasks(self, agent: "Agent", task: Any):
        """Get all tasks"""

    # @abstractmethod
    def get_finished_tasks(self) -> List[Dict]:
        """Get all finished tasks"""

    # @abstractmethod
    def get_pending_tasks(self) -> List[Dict]:
        """Get all pending tasks"""

    # @abstractmethod
    def pause_agent(self, agent: "Agent", agent_id: str):
        """Pause a agent"""

    # @abstractmethod
    def resume_agent(self, agent: "Agent", agent_id: str):
        """Resume a agent"""

    # @abstractmethod
    def stop_agent(self, agent: "Agent", agent_id: str):
        """Stop a agent"""

    # @abstractmethod
    def restart_agent(self, agent: "Agent"):
        """Restart agent"""

    # @abstractmethod
    def scale_up(self, num_agent: int):
        """Scale up the number of agents"""

    # @abstractmethod
    def scale_down(self, num_agent: int):
        """Scale down the number of agents"""

    # @abstractmethod
    def scale_to(self, num_agent: int):
        """Scale to a specific number of agents"""

    # @abstractmethod
    def get_all_agents(self) -> List["Agent"]:
        """Get all agents"""

    # @abstractmethod
    def get_swarm_size(self) -> int:
        """Get the size of the swarm"""

    # #@abstractmethod
    def get_swarm_status(self) -> Dict:
        """Get the status of the swarm"""

    # #@abstractmethod
    def save_swarm_state(self):
        """Save the swarm state"""

    def batched_run(self, tasks: List[Any], *args, **kwargs):
        """_summary_

        Args:
            tasks (List[Any]): _description_
        """
        # Implement batched run
        return [self.run(task, *args, **kwargs) for task in tasks]

    async def abatch_run(self, tasks: List[str], *args, **kwargs):
        """Asynchronous batch run with language model

        Args:
            tasks (List[str]): _description_

        Returns:
            _type_: _description_
        """
        return await asyncio.gather(
            *(self.arun(task, *args, **kwargs) for task in tasks)
        )

    async def arun(self, task: Optional[str] = None, *args, **kwargs):
        """Asynchronous run

        Args:
            task (Optional[str], optional): _description_. Defaults to None.
        """
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self.run, task, *args, **kwargs
        )
        return result

    def loop(
        self,
        task: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """Loop through the swarm

        Args:
            task (Optional[str], optional): _description_. Defaults to None.
        """
        # Loop through the self.max_loops
        for i in range(self.max_loops):
            self.run(task, *args, **kwargs)

    async def aloop(
        self,
        task: Optional[str] = None,
        *args,
        **kwargs,
    ):
        """Asynchronous loop through the swarm

        Args:
            task (Optional[str], optional): _description_. Defaults to None.
        """
        # Async Loop through the self.max_loops
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, self.loop, task, *args, **kwargs
        )
        return result

    def run_async(self, task: Optional[str] = None, *args, **kwargs):
        """Run the swarm asynchronously

        Args:
            task (Optional[str], optional): _description_. Defaults to None.
        """
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            self.arun(task, *args, **kwargs)
        )
        return result

    def run_batch_async(self, tasks: List[str], *args, **kwargs):
        """Run the swarm asynchronously

        Args:
            task (Optional[str], optional): _description_. Defaults to None.
        """
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            self.abatch_run(tasks, *args, **kwargs)
        )
        return result

    def run_batch(self, tasks: List[str], *args, **kwargs):
        """Run the swarm asynchronously

        Args:
            task (Optional[str], optional): _description_. Defaults to None.
        """
        return self.batched_run(tasks, *args, **kwargs)

    def reset_all_agents(self):
        """Reset all agents

        Returns:

        """
        for agent in self.agents:
            agent.reset()

    def select_agent(self, agent_id: str):
        """
        Select an agent through their id
        """
        # Find agent with id
        for agent in self.agents:
            if agent.id == agent_id:
                return agent

    def select_agent_by_name(self, agent_name: str):
        """
        Select an agent through their name
        """
        # Find agent with id
        for agent in self.agents:
            if agent.name == agent_name:
                return agent

    def task_assignment_by_id(
        self, task: str, agent_id: str, *args, **kwargs
    ):
        """
        Assign a task to an agent
        """
        # Assign task to agent by their agent id
        agent = self.select_agent(agent_id)
        return agent.run(task, *args, **kwargs)

    def task_assignment_by_name(
        self, task: str, agent_name: str, *args, **kwargs
    ):
        """
        Assign a task to an agent
        """
        # Assign task to agent by their agent id
        agent = self.select_agent_by_name(agent_name)
        return agent.run(task, *args, **kwargs)

    def concurrent_run(self, task: str) -> List[str]:
        """Synchronously run the task on all llms and collect responses"""
        with ThreadPoolExecutor() as executor:
            future_to_llm = {
                executor.submit(agent, task): agent
                for agent in self.agents
            }
            responses = []
            for future in as_completed(future_to_llm):
                try:
                    responses.append(future.result())
                except Exception as error:
                    print(
                        f"{future_to_llm[future]} generated an"
                        f" exception: {error}"
                    )
        self.last_responses = responses
        self.task_history.append(task)
        return responses

    def add_llm(self, agent: Callable):
        """Add an llm to the god mode"""
        self.agents.append(agent)

    def remove_llm(self, agent: Callable):
        """Remove an llm from the god mode"""
        self.agents.remove(agent)

    # def add_agent(self, agent: Agent = None, *args, **kwargs):
    #     """Add an agent to the swarm

    #     Args:
    #         agent (Agent, optional): _description_. Defaults to None.

    #     Returns:
    #         _type_: _description_
    #     """
    #     self.agents.append(agent)
    #     return agent

    def run_all(self, task: str = None, *args, **kwargs):
        """Run all agents

        Args:
            task (str, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        responses = []
        for agent in self.agents:
            responses.append(agent(task, *args, **kwargs))
        return responses

    def run_on_all_agents(self, task: str = None, *args, **kwargs):
        """Run on all agents

        Args:
            task (str, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        with ThreadPoolExecutor() as executor:
            responses = executor.map(
                lambda agent: agent(task, *args, **kwargs),
                self.agents,
            )
        return list(responses)

    @abstractmethod
    def add_swarm_entry(self, swarm):
        """
        Add the information of a joined Swarm to the registry.

        Args:
            swarm (SwarmManagerBase): Instance of SwarmManagerBase representing the joined Swarm.

        Returns:
            None
        """

    @abstractmethod
    def add_agent_entry(self, agent: Agent):
        """
        Add the information of an Agent to the registry.

        Args:
            agent (Agent): Instance of Agent representing the Agent.

        Returns:
            None
        """

    @abstractmethod
    def retrieve_swarm_information(self, swarm_id: str):
        """
        Retrieve the information of a specific Swarm from the registry.

        Args:
            swarm_id (str): Unique identifier of the Swarm.

        Returns:
            SwarmManagerBase: Instance of SwarmManagerBase representing the retrieved Swarm, or None if not found.
        """

    @abstractmethod
    def retrieve_joined_agents(self, agent_id: str) -> List[Agent]:
        """
        Retrieve the information the Agents which have joined the registry.

        Returns:
            Agent: Instance of Agent representing the retrieved Agent, or None if not found.
        """

    @abstractmethod
    def join_swarm(
        self, from_entity: Agent | Agent, to_entity: Agent
    ):
        """
        Add a relationship between a Swarm and an Agent or other Swarm to the registry.

        Args:
            from (Agent | SwarmManagerBase): Instance of Agent or SwarmManagerBase representing the source of the relationship.
        """

    def metadata(self):
        """
        Get the metadata of the multi-agent structure.

        Returns:
            dict: The metadata of the multi-agent structure.
        """
        return {
            "agents": self.agents,
            "callbacks": self.callbacks,
            "autosave": self.autosave,
            "logging": self.logging,
            "conversation": self.conversation,
        }

    def save_to_json(self, filename: str):
        """
        Save the current state of the multi-agent structure to a JSON file.

        Args:
            filename (str): The name of the file to save the multi-agent structure to.

        Returns:
            None
        """
        try:
            with open(filename, "w") as f:
                json.dump(self.__dict__, f)
        except Exception as e:
            logger.error(e)

    def load_from_json(self, filename: str):
        """
        Load the state of the multi-agent structure from a JSON file.

        Args:
            filename (str): The name of the file to load the multi-agent structure from.

        Returns:
            None
        """
        try:
            with open(filename) as f:
                self.__dict__ = json.load(f)
        except Exception as e:
            logger.error(e)

    def save_to_yaml(self, filename: str):
        """
        Save the current state of the multi-agent structure to a YAML file.

        Args:
            filename (str): The name of the file to save the multi-agent structure to.

        Returns:
            None
        """
        try:
            with open(filename, "w") as f:
                yaml.dump(self.__dict__, f)
        except Exception as e:
            logger.error(e)

    def load_from_yaml(self, filename: str):
        """
        Load the state of the multi-agent structure from a YAML file.

        Args:
            filename (str): The name of the file to load the multi-agent structure from.

        Returns:
            None
        """
        try:
            with open(filename) as f:
                self.__dict__ = yaml.load(f)
        except Exception as e:
            logger.error(e)

    def __repr__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

    def __str__(self):
        return f"{self.__class__.__name__}({self.__dict__})"

    def __len__(self):
        return len(self.agents)

    def __getitem__(self, index):
        return self.agents[index]

    def __setitem__(self, index, value):
        self.agents[index] = value

    def __delitem__(self, index):
        del self.agents[index]

    def __iter__(self):
        return iter(self.agents)

    def __reversed__(self):
        return reversed(self.agents)

    def __contains__(self, value):
        return value in self.agents
