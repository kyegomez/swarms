import concurrent.futures
import time
from typing import Callable, List, Optional, Union

from swarms.structs.agent import Agent
from swarms.structs.base_swarm import BaseSwarm
from swarms.structs.conversation import Conversation
from swarms.structs.swarm_id import swarm_id
from swarms.utils.formatter import formatter
from swarms.utils.get_cpu_cores import get_cpu_cores
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.loguru_logger import initialize_logger

logger = initialize_logger(log_folder="concurrent_workflow")


class ConcurrentWorkflow(BaseSwarm):
    """Concurrent workflow for running multiple agents simultaneously."""

    def __init__(
        self,
        id: str = swarm_id(),
        name: str = "ConcurrentWorkflow",
        description: str = "Execution of multiple agents concurrently",
        agents: List[Union[Agent, Callable]] = None,
        auto_save: bool = True,
        output_type: str = "dict-all-except-first",
        max_loops: int = 1,
        auto_generate_prompts: bool = False,
        show_dashboard: bool = False,
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
        self.metadata_output_path = (
            f"concurrent_workflow_name_{name}_id_{id}.json"
        )
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
        """Configure agents for dashboard mode."""
        if self.show_dashboard is True:
            for agent in self.agents:
                agent.print_on = False
        return self.agents

    def reliability_check(self):
        """Validate workflow configuration."""
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
                    "ConcurrentWorkflow: Only one agent provided."
                )
        except Exception as e:
            logger.error(
                f"ConcurrentWorkflow: Reliability check failed: {e}"
            )
            raise

    def activate_auto_prompt_engineering(self):
        """Enable automatic prompt engineering."""
        if self.auto_generate_prompts is True:
            for agent in self.agents:
                agent.auto_generate_prompt = True

    def display_agent_dashboard(
        self,
        title: str = "ConcurrentWorkflow Dashboard",
        is_final: bool = False,
    ):
        """Display real-time dashboard."""
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
        streaming_callback: Optional[
            Callable[[str, str, bool], None]
        ] = None,
    ):
        """Execute agents with dashboard monitoring."""
        try:
            self.conversation.add(role="User", content=task)

            # Reset agent statuses
            for agent in self.agents:
                self.agent_statuses[agent.agent_name] = {
                    "status": "pending",
                    "output": "",
                }

            if self.show_dashboard:
                self.display_agent_dashboard()

            max_workers = int(get_cpu_cores() * 0.95)
            futures = []
            results = []

            def run_agent_with_status(agent, task, img, imgs):
                try:
                    self.agent_statuses[agent.agent_name][
                        "status"
                    ] = "running"
                    if self.show_dashboard:
                        self.display_agent_dashboard()

                    last_update_time = [0]
                    update_interval = 0.1

                    def agent_streaming_callback(chunk: str):
                        try:
                            if self.show_dashboard and chunk:
                                current_output = self.agent_statuses[
                                    agent.agent_name
                                ]["output"]
                                self.agent_statuses[agent.agent_name][
                                    "output"
                                ] = (current_output + chunk)

                                current_time = time.time()
                                if (
                                    current_time - last_update_time[0]
                                    >= update_interval
                                ):
                                    self.display_agent_dashboard()
                                    last_update_time[0] = current_time

                            if (
                                streaming_callback
                                and chunk is not None
                            ):
                                streaming_callback(
                                    agent.agent_name, chunk, False
                                )
                        except Exception as callback_error:
                            logger.warning(
                                f"Dashboard streaming callback failed for {agent.agent_name}: {str(callback_error)}"
                            )

                    output = agent.run(
                        task=task,
                        img=img,
                        imgs=imgs,
                        streaming_callback=agent_streaming_callback,
                    )

                    self.agent_statuses[agent.agent_name][
                        "status"
                    ] = "completed"
                    self.agent_statuses[agent.agent_name][
                        "output"
                    ] = output
                    if self.show_dashboard:
                        self.display_agent_dashboard()

                    if streaming_callback:
                        streaming_callback(agent.agent_name, "", True)

                    return output
                except Exception as e:
                    self.agent_statuses[agent.agent_name][
                        "status"
                    ] = "error"
                    self.agent_statuses[agent.agent_name][
                        "output"
                    ] = f"Error: {str(e)}"
                    if self.show_dashboard:
                        self.display_agent_dashboard()

                    if streaming_callback:
                        streaming_callback(
                            agent.agent_name, f"Error: {str(e)}", True
                        )

                    raise

            with concurrent.futures.ThreadPoolExecutor(
                max_workers=max_workers
            ) as executor:
                futures = [
                    executor.submit(
                        run_agent_with_status, agent, task, img, imgs
                    )
                    for agent in self.agents
                ]
                concurrent.futures.wait(futures)

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

            for agent_name, output in results:
                self.conversation.add(role=agent_name, content=output)

            if self.show_dashboard:
                self.display_agent_dashboard(
                    "Final ConcurrentWorkflow Dashboard",
                    is_final=True,
                )

            return history_output_formatter(
                conversation=self.conversation, type=self.output_type
            )
        finally:
            if self.show_dashboard:
                formatter.stop_dashboard()

    def _run(
        self,
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        streaming_callback: Optional[
            Callable[[str, str, bool], None]
        ] = None,
    ):
        """Execute agents concurrently without dashboard."""
        self.conversation.add(role="User", content=task)

        max_workers = int(get_cpu_cores() * 0.95)

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers
        ) as executor:
            future_to_agent = {
                executor.submit(
                    self._run_agent_with_streaming,
                    agent,
                    task,
                    img,
                    imgs,
                    streaming_callback,
                ): agent
                for agent in self.agents
            }

            for future in concurrent.futures.as_completed(
                future_to_agent
            ):
                agent = future_to_agent[future]
                output = future.result()
                self.conversation.add(
                    role=agent.agent_name, content=output
                )

        return history_output_formatter(
            conversation=self.conversation, type=self.output_type
        )

    def _run_agent_with_streaming(
        self,
        agent: Union[Agent, Callable],
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        streaming_callback: Optional[
            Callable[[str, str, bool], None]
        ] = None,
    ):
        """Run single agent with streaming support."""
        if streaming_callback is None:
            return agent.run(task=task, img=img, imgs=imgs)

        def agent_streaming_callback(chunk: str):
            try:
                # Safely call the streaming callback
                if streaming_callback and chunk is not None:
                    streaming_callback(agent.agent_name, chunk, False)
            except Exception as callback_error:
                logger.warning(
                    f"Streaming callback failed for {agent.agent_name}: {str(callback_error)}"
                )

        try:
            output = agent.run(
                task=task,
                img=img,
                imgs=imgs,
                streaming_callback=agent_streaming_callback,
            )
            # Ensure completion callback is called even if there were issues
            try:
                streaming_callback(agent.agent_name, "", True)
            except Exception as callback_error:
                logger.warning(
                    f"Completion callback failed for {agent.agent_name}: {str(callback_error)}"
                )
            return output
        except Exception as e:
            error_msg = f"Agent {agent.agent_name} failed: {str(e)}"
            logger.error(error_msg)
            # Try to send error through callback
            try:
                streaming_callback(
                    agent.agent_name, f"Error: {str(e)}", True
                )
            except Exception as callback_error:
                logger.warning(
                    f"Error callback failed for {agent.agent_name}: {str(callback_error)}"
                )
            raise

    def cleanup(self):
        """Clean up resources and connections."""
        try:
            # Reset agent statuses
            for agent in self.agents:
                if hasattr(agent, "cleanup"):
                    try:
                        agent.cleanup()
                    except Exception as e:
                        logger.warning(
                            f"Failed to cleanup agent {agent.agent_name}: {str(e)}"
                        )

            # Clear conversation if needed
            if hasattr(self, "conversation"):
                # Keep the conversation for result formatting but reset for next run
                pass

        except Exception as e:
            logger.error(f"Cleanup failed: {str(e)}")

    def run(
        self,
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        streaming_callback: Optional[
            Callable[[str, str, bool], None]
        ] = None,
    ):
        """Execute all agents concurrently."""
        try:
            if self.show_dashboard:
                result = self.run_with_dashboard(
                    task, img, imgs, streaming_callback
                )
            else:
                result = self._run(
                    task, img, imgs, streaming_callback
                )
            return result
        finally:
            # Always cleanup resources
            self.cleanup()

    def batch_run(
        self,
        tasks: List[str],
        imgs: Optional[List[str]] = None,
        streaming_callback: Optional[
            Callable[[str, str, bool], None]
        ] = None,
    ):
        """Execute workflow on multiple tasks sequentially."""
        results = []
        for idx, task in enumerate(tasks):
            img = None
            if imgs is not None and idx < len(imgs):
                img = imgs[idx]
            results.append(
                self.run(
                    task=task,
                    img=img,
                    streaming_callback=streaming_callback,
                )
            )
        return results
