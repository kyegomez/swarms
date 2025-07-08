import concurrent.futures
import os
from typing import Callable, List, Optional, Union

from swarms.structs.agent import Agent
from swarms.structs.base_swarm import BaseSwarm
from swarms.structs.conversation import Conversation
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.formatter import formatter

logger = initialize_logger(log_folder="concurrent_workflow")


class ConcurrentWorkflow(BaseSwarm):
    """
    Represents a concurrent workflow that executes multiple agents concurrently in a production-grade manner.
    Features include:
    - Caching for repeated prompts
    - Enhanced error handling and retries
    - Input validation

    Args:
        name (str): The name of the workflow. Defaults to "ConcurrentWorkflow".
        description (str): The description of the workflow. Defaults to "Execution of multiple agents concurrently".
        agents (List[Agent]): The list of agents to be executed concurrently. Defaults to an empty list.
        metadata_output_path (str): The path to save the metadata output. Defaults to "agent_metadata.json".
        auto_save (bool): Flag indicating whether to automatically save the metadata. Defaults to False.
        output_type (str): The type of output format. Defaults to "dict".
        max_loops (int): The maximum number of loops for each agent. Defaults to 1.
        return_str_on (bool): Flag indicating whether to return the output as a string. Defaults to False.
        auto_generate_prompts (bool): Flag indicating whether to auto-generate prompts for agents. Defaults to False.
        return_entire_history (bool): Flag indicating whether to return the entire conversation history. Defaults to False.
        show_dashboard (bool): Flag indicating whether to show a real-time dashboard. Defaults to True.

    Raises:
        ValueError: If the list of agents is empty or if the description is empty.

    Attributes:
        name (str): The name of the workflow.
        description (str): The description of the workflow.
        agents (List[Agent]): The list of agents to be executed concurrently.
        metadata_output_path (str): The path to save the metadata output.
        auto_save (bool): Flag indicating whether to automatically save the metadata.
        output_type (str): The type of output format.
        max_loops (int): The maximum number of loops for each agent.
        auto_generate_prompts (bool): Flag indicating whether to auto-generate prompts for agents.
        show_dashboard (bool): Flag indicating whether to show a real-time dashboard.
        agent_statuses (dict): Dictionary to track agent statuses.
    """

    def __init__(
        self,
        name: str = "ConcurrentWorkflow",
        description: str = "Execution of multiple agents concurrently",
        agents: List[Union[Agent, Callable]] = [],
        metadata_output_path: str = "agent_metadata.json",
        auto_save: bool = True,
        output_type: str = "dict-all-except-first",
        max_loops: int = 1,
        auto_generate_prompts: bool = False,
        show_dashboard: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(
            name=name,
            description=description,
            agents=agents,
            *args,
            **kwargs,
        )
        self.name = name
        self.description = description
        self.agents = agents
        self.metadata_output_path = metadata_output_path
        self.auto_save = auto_save
        self.max_loops = max_loops
        self.auto_generate_prompts = auto_generate_prompts
        self.output_type = output_type
        self.show_dashboard = show_dashboard
        self.agent_statuses = {
            agent.agent_name: {"status": "pending", "output": ""}
            for agent in agents
        }

        self.reliability_check()
        self.conversation = Conversation()

        if self.show_dashboard is True:
            self.agents = self.fix_agents()

    def fix_agents(self):
        if self.show_dashboard is True:
            for agent in self.agents:
                agent.print_on = False
        return self.agents

    def reliability_check(self):
        try:
            if self.agents is None:
                raise ValueError(
                    "ConcurrentWorkflow: No agents provided"
                )

            if len(self.agents) == 0:
                raise ValueError(
                    "ConcurrentWorkflow: No agents provided"
                )

            if len(self.agents) == 1:
                logger.warning(
                    "ConcurrentWorkflow: Only one agent provided. With ConcurrentWorkflow, you should use at least 2+ agents."
                )
        except Exception as e:
            logger.error(
                f"ConcurrentWorkflow: Reliability check failed: {e}"
            )
            raise

    def activate_auto_prompt_engineering(self):
        """
        Activates the auto-generate prompts feature for all agents in the workflow.

        Example:
            >>> workflow = ConcurrentWorkflow(agents=[Agent()])
            >>> workflow.activate_auto_prompt_engineering()
            >>> # All agents in the workflow will now auto-generate prompts.
        """
        if self.auto_generate_prompts is True:
            for agent in self.agents:
                agent.auto_generate_prompt = True

    def display_agent_dashboard(
        self,
        title: str = "🤖 Agent Dashboard",
        is_final: bool = False,
    ) -> None:
        """
        Displays the current status of all agents in a beautiful dashboard format.

        Args:
            title (str): The title of the dashboard.
            is_final (bool): Flag indicating whether this is the final dashboard.
        """
        agents_data = [
            {
                "name": agent.agent_name,
                "status": self.agent_statuses[agent.agent_name][
                    "status"
                ],
                "output": self.agent_statuses[agent.agent_name][
                    "output"
                ],
            }
            for agent in self.agents
        ]
        formatter.print_agent_dashboard(agents_data, title, is_final)

    def run_with_dashboard(
        self,
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
    ):
        """
        Executes all agents in the workflow concurrently on the given task.
        Now includes real-time dashboard updates.
        """
        try:
            self.conversation.add(role="User", content=task)

            # Reset agent statuses
            for agent in self.agents:
                self.agent_statuses[agent.agent_name] = {
                    "status": "pending",
                    "output": "",
                }

            # Display initial dashboard if enabled
            if self.show_dashboard:
                self.display_agent_dashboard()

            # Use 95% of available CPU cores for optimal performance
            max_workers = int(os.cpu_count() * 0.95)

            # Create a list to store all futures and their results
            futures = []
            results = []

            def run_agent_with_status(agent, task, img, imgs):
                try:
                    # Update status to running
                    self.agent_statuses[agent.agent_name][
                        "status"
                    ] = "running"
                    if self.show_dashboard:
                        self.display_agent_dashboard()

                    # Run the agent
                    output = agent.run(task=task, img=img, imgs=imgs)

                    # Update status to completed
                    self.agent_statuses[agent.agent_name][
                        "status"
                    ] = "completed"
                    self.agent_statuses[agent.agent_name][
                        "output"
                    ] = output
                    if self.show_dashboard:
                        self.display_agent_dashboard()

                    return output
                except Exception as e:
                    # Update status to error
                    self.agent_statuses[agent.agent_name][
                        "status"
                    ] = "error"
                    self.agent_statuses[agent.agent_name][
                        "output"
                    ] = f"Error: {str(e)}"
                    if self.show_dashboard:
                        self.display_agent_dashboard()
                    raise

            # Run agents concurrently using ThreadPoolExecutor
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                # Submit all agent tasks
                futures = [
                    executor.submit(
                        run_agent_with_status, agent, task, img, imgs
                    )
                    for agent in self.agents
                ]

                # Wait for all futures to complete
                concurrent.futures.wait(futures)

                # Process results in order of completion
                for future, agent in zip(futures, self.agents):
                    try:
                        output = future.result()
                        results.append((agent.agent_name, output))
                    except Exception as e:
                        logger.error(
                            f"Agent {agent.agent_name} failed: {str(e)}"
                        )
                        results.append(
                            (agent.agent_name, f"Error: {str(e)}")
                        )

            # Add all results to conversation
            for agent_name, output in results:
                self.conversation.add(role=agent_name, content=output)

            # Display final dashboard if enabled
            if self.show_dashboard:
                self.display_agent_dashboard(
                    "🎉 Final Agent Dashboard", is_final=True
                )

            return history_output_formatter(
                conversation=self.conversation,
                type=self.output_type,
            )
        finally:
            # Always clean up the dashboard display
            if self.show_dashboard:
                formatter.stop_dashboard()

    def _run(
        self,
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
    ):
        """
        Executes all agents in the workflow concurrently on the given task.

        Args:
            task (str): The task to be executed by all agents.
            img (Optional[str]): Optional image path for agents that support image input.
            imgs (Optional[List[str]]): Optional list of image paths for agents that support multiple image inputs.

        Returns:
            The formatted output based on the configured output_type.

        Example:
            >>> workflow = ConcurrentWorkflow(agents=[agent1, agent2])
            >>> result = workflow.run("Analyze this financial data")
            >>> print(result)
        """
        self.conversation.add(role="User", content=task)

        # Use 95% of available CPU cores for optimal performance
        max_workers = int(os.cpu_count() * 0.95)

        # Run agents concurrently using ThreadPoolExecutor
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
            # Submit all agent tasks and store with their index
            future_to_agent = {
                executor.submit(
                    agent.run, task=task, img=img, imgs=imgs
                ): agent
                for agent in self.agents
            }

            # Collect results and add to conversation in completion order
            for future in concurrent.futures.as_completed(
                future_to_agent
            ):
                agent = future_to_agent[future]
                output = future.result()
                self.conversation.add(role=agent.name, content=output)

        return history_output_formatter(
            conversation=self.conversation,
            type=self.output_type,
        )

    def run(
        self,
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
    ):
        """
        Executes all agents in the workflow concurrently on the given task.
        """
        if self.show_dashboard:
            return self.run_with_dashboard(task, img, imgs)
        else:
            return self._run(task, img, imgs)

    def batch_run(
        self,
        tasks: List[str],
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
    ):
        """
        Executes the workflow on multiple tasks sequentially.

        Args:
            tasks (List[str]): List of tasks to be executed by all agents.
            img (Optional[str]): Optional image path for agents that support image input.
            imgs (Optional[List[str]]): Optional list of image paths for agents that support multiple image inputs.

        Returns:
            List of results, one for each task.

        Example:
            >>> workflow = ConcurrentWorkflow(agents=[agent1, agent2])
            >>> tasks = ["Task 1", "Task 2", "Task 3"]
            >>> results = workflow.batch_run(tasks)
            >>> print(len(results))  # 3
        """
        return [
            self.run(task=task, img=img, imgs=imgs) for task in tasks
        ]


# if __name__ == "__main__":
#     # Assuming you've already initialized some agents outside of this class
#     agents = [
#         Agent(
#             agent_name=f"Financial-Analysis-Agent-{i}",
#             system_prompt=FINANCIAL_AGENT_SYS_PROMPT,
#             model_name="gpt-4o",
#             max_loops=1,
#         )
#         for i in range(3)  # Adjust number of agents as needed
#     ]

#     # Initialize the workflow with the list of agents
#     workflow = ConcurrentWorkflow(
#         agents=agents,
#         metadata_output_path="agent_metadata_4.json",
#         return_str_on=True,
#     )

#     # Define the task for all agents
#     task = "How can I establish a ROTH IRA to buy stocks and get a tax break? What are the criteria?"

#     # Run the workflow and save metadata
#     metadata = workflow.run(task)
#     print(metadata)
