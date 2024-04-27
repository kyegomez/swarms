from dataclasses import dataclass, field
from typing import List, Optional
from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.loguru_logger import logger
from swarms.utils.try_except_wrapper import try_except_wrapper
from swarms.structs.base_workflow import BaseWorkflow


@dataclass
class SequentialWorkflow(BaseWorkflow):
    name: str = "Sequential Workflow"
    description: str = None
    objective: str = None
    max_loops: int = 1
    autosave: bool = False
    saved_state_filepath: Optional[str] = "sequential_workflow_state.json"
    restore_state_filepath: Optional[str] = None
    dashboard: bool = False
    agent_pool: List[Agent] = field(default_factory=list)
    # task_pool: List[str] = field(
    #     default_factory=list
    # )  # List to store tasks

    def __post_init__(self):
        super().__init__()
        self.conversation = Conversation(
            time_enabled=True,
            autosave=True,
        )

        # If objective exists then set it
        if self.objective is not None:
            self.conversation.system_prompt = self.objective

    def workflow_bootup(self):
        logger.info(f"{self.name} is activating...")

        for agent in self.agent_pool:
            logger.info(f"Agent {agent.agent_name} Activated")

    @try_except_wrapper
    def add(self, task: str, agent: Agent, *args, **kwargs):
        self.agent_pool.append(agent)
        # self.task_pool.append(
        #     task
        # )  # Store tasks corresponding to each agent

        return self.conversation.add(
            role=agent.agent_name, content=task, *args, **kwargs
        )

    def reset_workflow(self) -> None:
        self.conversation = {}

    @try_except_wrapper
    def run(self):
        if not self.agent_pool:
            raise ValueError("No agents have been added to the workflow.")

        self.workflow_bootup()
        loops = 0
        while loops < self.max_loops:
            previous_output = None  # Initialize to None; will hold the output of the previous agent
            for i, agent in enumerate(self.agent_pool):
                # Fetch the last task specific to this agent from the conversation history
                tasks_for_agent = [
                    msg["content"]
                    for msg in self.conversation.conversation_history
                    if msg["role"] == agent.agent_name
                ]
                task = tasks_for_agent[-1] if tasks_for_agent else None

                if task is None and previous_output is not None:
                    # If no specific task for this agent, use the output from the previous agent
                    task = previous_output

                if task is None:
                    # If no initial task is found, and there's no previous output, log error and skip this agent
                    logger.error(
                        f"No initial task found for agent {agent.agent_name}, and no previous output to use."
                    )
                    continue

                logger.info(
                    f" \n Agent {i+1} ({agent.agent_name}) is executing the task: {task} \n"
                )

                # Space the log

                output = agent.run(task)
                if output is None:
                    logger.error(
                        f"Agent {agent.agent_name} returned None for task: {task}"
                    )
                    raise ValueError(
                        f"Agent {agent.agent_name} returned None."
                    )

                # Update the conversation history with the new output using agent's role
                self.conversation.add(
                    role=agent.agent_name, content=output
                )
                previous_output = output  # Update the previous_output to pass to the next agent

            loops += 1
        return self.conversation.return_history_as_string()
