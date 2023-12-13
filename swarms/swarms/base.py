import asyncio
import concurrent.futures
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Callable
from swarms.structs.agent import Agent
from swarms.agents.base import AbstractWorker
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging
from termcolor import colored


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
        add_worker: Add a worker to the swarm
        remove_worker: Remove a worker from the swarm
        broadcast: Broadcast a message to all agents
        reset: Reset the swarm
        plan: agents must individually plan using a workflow or pipeline
        direct_message: Send a direct message to a worker
        autoscaler: Autoscaler that acts like kubernetes for autonomous agents
        get_worker_by_id: Locate a worker by id
        get_worker_by_name: Locate a worker by name
        assign_task: Assign a task to a worker
        get_all_tasks: Get all tasks
        get_finished_tasks: Get all finished tasks
        get_pending_tasks: Get all pending tasks
        pause_worker: Pause a worker
        resume_worker: Resume a worker
        stop_worker: Stop a worker
        restart_worker: Restart worker
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

    # TODO: Pass in abstract LLM class that can utilize Hf or Anthropic models, Move away from OPENAI
    # TODO: ADD Universal Communication Layer, a ocean vectorstore instance
    # TODO: BE MORE EXPLICIT ON TOOL USE, TASK DECOMPOSITION AND TASK COMPLETETION AND ALLOCATION
    # TODO: Add RLHF Data collection, ask user how the swarm is performing
    # TODO: Create an onboarding process if not settings are preconfigured like `from swarms import Swarm, Swarm()` => then initiate onboarding name your swarm + provide purpose + etc

    # @abstractmethod
    def __init__(self, agents: List[Agent], max_loops: int = 200):
        """Initialize the swarm with agents"""
        self.agents = agents
        self.max_loops = max_loops
        pass

    # @abstractmethod
    def communicate(self):
        """Communicate with the swarm through the orchestrator, protocols, and the universal communication layer"""
        pass

    # @abstractmethod
    def run(self):
        """Run the swarm"""
        pass

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
        pass

    # @abstractmethod
    def add_worker(self, worker: "AbstractWorker"):
        """Add a worker to the swarm"""
        pass

    # @abstractmethod
    def remove_worker(self, worker: "AbstractWorker"):
        """Remove a worker from the swarm"""
        pass

    # @abstractmethod
    def broadcast(
        self, message: str, sender: Optional["AbstractWorker"] = None
    ):
        """Broadcast a message to all agents"""
        pass

    # @abstractmethod
    def reset(self):
        """Reset the swarm"""
        pass

    # @abstractmethod
    def plan(self, task: str):
        """agents must individually plan using a workflow or pipeline"""
        pass

    # @abstractmethod
    def direct_message(
        self,
        message: str,
        sender: "AbstractWorker",
        recipient: "AbstractWorker",
    ):
        """Send a direct message to a worker"""
        pass

    # @abstractmethod
    def autoscaler(self, num_agents: int, worker: ["AbstractWorker"]):
        """Autoscaler that acts like kubernetes for autonomous agents"""
        pass

    # @abstractmethod
    def get_worker_by_id(self, id: str) -> "AbstractWorker":
        """Locate a worker by id"""
        pass

    # @abstractmethod
    def get_worker_by_name(self, name: str) -> "AbstractWorker":
        """Locate a worker by name"""
        pass

    # @abstractmethod
    def assign_task(
        self, worker: "AbstractWorker", task: Any
    ) -> Dict:
        """Assign a task to a worker"""
        pass

    # @abstractmethod
    def get_all_tasks(self, worker: "AbstractWorker", task: Any):
        """Get all tasks"""

    # @abstractmethod
    def get_finished_tasks(self) -> List[Dict]:
        """Get all finished tasks"""
        pass

    # @abstractmethod
    def get_pending_tasks(self) -> List[Dict]:
        """Get all pending tasks"""
        pass

    # @abstractmethod
    def pause_worker(self, worker: "AbstractWorker", worker_id: str):
        """Pause a worker"""
        pass

    # @abstractmethod
    def resume_worker(self, worker: "AbstractWorker", worker_id: str):
        """Resume a worker"""
        pass

    # @abstractmethod
    def stop_worker(self, worker: "AbstractWorker", worker_id: str):
        """Stop a worker"""
        pass

    # @abstractmethod
    def restart_worker(self, worker: "AbstractWorker"):
        """Restart worker"""
        pass

    # @abstractmethod
    def scale_up(self, num_worker: int):
        """Scale up the number of agents"""
        pass

    # @abstractmethod
    def scale_down(self, num_worker: int):
        """Scale down the number of agents"""
        pass

    # @abstractmethod
    def scale_to(self, num_worker: int):
        """Scale to a specific number of agents"""
        pass

    # @abstractmethod
    def get_all_agents(self) -> List["AbstractWorker"]:
        """Get all agents"""
        pass

    # @abstractmethod
    def get_swarm_size(self) -> int:
        """Get the size of the swarm"""
        pass

    # #@abstractmethod
    def get_swarm_status(self) -> Dict:
        """Get the status of the swarm"""
        pass

    # #@abstractmethod
    def save_swarm_state(self):
        """Save the swarm state"""
        pass

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

    def add_agent(self, agent: Agent = None, *args, **kwargs):
        """Add an agent to the swarm

        Args:
            agent (Agent, optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        self.agents.append(agent)
        return agent

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
