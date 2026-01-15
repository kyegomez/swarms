import concurrent.futures
import os
import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any, List

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.multi_agent_exec import run_agents_concurrently
from swarms.structs.swarm_id import swarm_id
from swarms.utils.formatter import formatter
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.output_types import OutputType
from typing import Callable, Optional

logger = initialize_logger(log_folder="majority_voting")


CONSENSUS_AGENT_PROMPT = """
You are the Consensus Agent, responsible for synthesizing and evaluating the responses from a panel of expert agents. Your task is to deliver a rigorous, insightful, and actionable consensus based on their outputs.

**Instructions:**

1. **Comprehensive Evaluation:**  
   For each agent (referenced by their name), provide a detailed, objective critique of their response. Assess the following dimensions:
   - Accuracy and correctness
   - Depth of analysis and insight
   - Relevance to the original task or question
   - Clarity, structure, and communication quality
   - Unique perspectives or innovative ideas

2. **Comparative Analysis:**  
   Compare and contrast the agents’ responses. Highlight:
   - Overlapping themes or points of agreement
   - Divergent viewpoints or conflicting recommendations
   - Notable strengths and weaknesses of each approach

3. **Consensus Building:**  
   - Identify which response(s) most effectively address the task, providing clear justification for your choices.
   - If appropriate, synthesize the best elements from multiple responses into a unified, superior answer.
   - Clearly explain your reasoning and the criteria used for your judgment.

4. **Ranking and Recommendation:**  
   - Provide a ranked list of agent responses, from most to least effective, with concise rationales for each position.
   - Offer a final, well-justified recommendation or summary that represents the optimal consensus.

5. **Fairness and Rigor:**  
   - Remain impartial, thorough, and evidence-based in your analysis.
   - Avoid bias towards any agent or perspective.
   - Ensure your consensus is actionable, well-supported, and clearly communicated.

**Output Format:**
- For each agent: [Agent Name]: [Evaluation]
- Comparative Analysis: [Summary]
- Ranked List: [1. Agent Name, 2. Agent Name, ...]
- Final Consensus/Recommendation: [Your synthesized answer or recommendation]

Your goal is to deliver a consensus that is not only fair and balanced, but also maximizes the quality, relevance, and utility of the collective agent output.
"""


def default_consensus_agent(
    name: str = "Consensus-Agent",
    system_prompt: str = None,
    description: str = "An agent that uses consensus to generate a final answer.",
    model_name: str = "gpt-4.1",
    streaming_callback: Optional[Callable[[str], None]] = None,
    *args,
    **kwargs,
):
    # If streaming_on is not None, force it to True; else, set to False
    if streaming_callback is not None:
        streaming_on_value = True
    else:
        streaming_on_value = False

    return Agent(
        agent_name=name,
        agent_description=description,
        model_name=model_name,
        max_loops=1,
        system_prompt=system_prompt,
        dynamic_context_window=True,
        dynamic_temperature_enabled=True,
        streaming_on=streaming_on_value,
        *args,
        **kwargs,
    )


class MajorityVoting:
    """
    A multi-loop majority voting system for agents that enables iterative consensus building.

    This system allows agents to run multiple loops where each subsequent loop considers
    the previous consensus, enabling agents to refine their responses and build towards
    a more robust final consensus. The system maintains conversation history across
    all loops and provides methods to analyze the evolution of consensus over time.

    Key Features:
    - Multi-loop consensus building with configurable loop count
    - Agent memory retention across loops
    - Comprehensive consensus history tracking
    - Flexible output formats (string, dict, list)
    - Loop-by-loop analysis capabilities
    """

    def __init__(
        self,
        id: str = swarm_id(),
        name: str = "MajorityVoting",
        description: str = "A multi-loop majority voting system for agents",
        agents: List[Agent] = None,
        autosave: bool = False,
        verbose: bool = False,
        max_loops: int = 1,
        output_type: OutputType = "dict",
        consensus_agent_prompt: str = CONSENSUS_AGENT_PROMPT,
        consensus_agent_name: str = "Consensus-Agent",
        consensus_agent_description: str = "An agent that uses consensus to generate a final answer.",
        consensus_agent_model_name: str = "gpt-4.1",
        additional_consensus_agent_kwargs: dict = {},
        *args,
        **kwargs,
    ):
        self.id = id
        self.name = name
        self.description = description
        self.agents = agents
        self.autosave = autosave
        self.verbose = verbose
        self.max_loops = max_loops
        self.output_type = output_type
        self.consensus_agent_prompt = consensus_agent_prompt

        self.conversation = Conversation(
            time_enabled=False, *args, **kwargs
        )

        self.consensus_agent = default_consensus_agent(
            name=consensus_agent_name,
            system_prompt=consensus_agent_prompt,
            description=consensus_agent_description,
            model_name=consensus_agent_model_name,
            **additional_consensus_agent_kwargs,
        )

        self.reliability_check()

    def reliability_check(self):

        if self.agents is None:
            raise ValueError("Agents list is empty")

        # Log the agents in a more formatted, readable way
        agent_list = "\n".join(
            [f"  - {agent.agent_name}" for agent in self.agents]
        )
        panel_content = (
            f"[bold]Initializing Majority Voting System[/bold]\n"
            f"[bold]Number of agents:[/bold] {len(self.agents)}\n"
            f"[bold]Agents:[/bold]\n{agent_list}"
        )
        formatter.print_panel(
            panel_content,
            title="Majority Voting",
        )

    def run(
        self,
        task: str,
        streaming_callback: Optional[Callable[[str], None]] = None,
        *args,
        **kwargs,
    ) -> List[Any]:
        """
        Runs the majority voting system with multi-loop functionality and returns the majority vote.

        Args:
            task (str): The task to be performed by the agents.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            List[Any]: The majority vote.

        """

        self.conversation.add(
            role="user",
            content=task,
        )

        for i in range(self.max_loops):

            output = run_agents_concurrently(
                agents=self.agents,
                task=self.conversation.get_str(),
                max_workers=os.cpu_count(),
            )

            for agent, output in zip(self.agents, output):
                self.conversation.add(
                    role=agent.agent_name,
                    content=output,
                )

                # If streaming callback provided, emit agent output as a mini-json chunk
                if streaming_callback is not None:
                    try:
                        payload = {"agent": agent.agent_name, "chunk": output}
                        streaming_callback(json.dumps(payload))
                        streaming_callback(json.dumps({"agent": agent.agent_name, "done": True}))
                    except Exception:
                        if self.verbose:
                            logger.exception("streaming callback failed for agent output")

            # Set streaming_on for the consensus agent based on the provided streaming_callback
            self.consensus_agent.streaming_on = (
                streaming_callback is not None
            )

            # Instead of a simple passthrough wrapper, match the callback invocation pattern from the provided reference for the consensus agent:
            consensus_agent_name = self.consensus_agent.agent_name

            if streaming_callback is not None:

                def consensus_streaming_callback(chunk: str, done: bool = False):
                    """Wrapper for consensus agent streaming callback emitting JSON strings."""
                    try:
                        if chunk is not None and chunk.strip():
                            payload = {"agent": consensus_agent_name, "chunk": chunk}
                            streaming_callback(json.dumps(payload))
                        if done:
                            streaming_callback(json.dumps({"agent": consensus_agent_name, "done": True}))
                    except Exception as callback_error:
                        if self.verbose:
                            logger.warning(
                                f"[STREAMING] Callback failed for {consensus_agent_name}: {str(callback_error)}"
                            )

            else:
                consensus_streaming_callback = None

            # Run the consensus agent with the streaming callback, if any
            consensus_output = self.consensus_agent.run(
                task=(f"History: {self.conversation.get_str()}"),
                streaming_callback=consensus_streaming_callback,
            )

            self.conversation.add(
                role=self.consensus_agent.agent_name,
                content=consensus_output,
            )

        return history_output_formatter(
            conversation=self.conversation,
            type=self.output_type,
        )

    def batch_run(
        self, tasks: List[str], *args, **kwargs
    ) -> List[Any]:
        """
        Runs the majority voting system in batch mode.

        Args:
            tasks (List[str]): List of tasks to be performed by the agents.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            List[Any]: List of majority votes for each task.
        """
        return [self.run(task, *args, **kwargs) for task in tasks]

    def run_concurrently(
        self, tasks: List[str], *args, **kwargs
    ) -> List[Any]:
        """
        Runs the majority voting system concurrently.

        Args:
            tasks (List[str]): List of tasks to be performed by the agents.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            List[Any]: List of majority votes for each task.
        """
        with ThreadPoolExecutor(
            max_workers=os.cpu_count()
        ) as executor:
            futures = [
                executor.submit(self.run, task, *args, **kwargs)
                for task in tasks
            ]
            return [
                future.result()
                for future in concurrent.futures.as_completed(futures)
            ]
