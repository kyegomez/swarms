import concurrent.futures
import os
import uuid
from typing import List, Optional

from swarms.prompts.ag_prompt import AGGREGATOR_SYSTEM_PROMPT_MAIN
from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.ma_utils import list_all_agents
from swarms.structs.multi_agent_exec import run_agents_concurrently
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.output_types import OutputType

logger = initialize_logger(log_folder="mixture_of_agents")


class MixtureOfAgents:
    """Run a layered Mixture-of-Agents workflow.

    ``MixtureOfAgents`` runs workers in parallel across multiple layers,
    then synthesises their outputs with an aggregator agent.

    Worker context per layer:
    - Layer 0: each worker receives only the original task.
    - Layer 1+: each worker receives the original task plus the
      concatenated outputs from the previous layer.

    The aggregator always receives the full conversation transcript.

    Args:
        id: Optional identifier accepted for API compatibility.
        name: Human-readable name for this mixture.
        description: Description added to the conversation metadata when
            listing the worker agents.
        agents: Worker agents that run in parallel on each layer.
        aggregator_agent: Optional preconfigured agent used to synthesize
            the final response. If omitted, one is created from
            ``aggregator_system_prompt`` and ``aggregator_model_name``.
        aggregator_system_prompt: System prompt used when creating the
            default aggregator agent.
        layers: Number of worker-agent rounds to run before aggregation.
        max_loops: Stored configuration value for compatibility with other
            swarm classes.
        output_type: Format passed to ``history_output_formatter``.
        aggregator_model_name: Model name for the default aggregator agent.

    Examples:
        >>> from swarms import Agent
        >>> agents = [
        ...     Agent(agent_name="Researcher", model_name="gpt-5.4"),
        ...     Agent(agent_name="Analyst", model_name="gpt-5.4"),
        ... ]
        >>> moa = MixtureOfAgents(agents=agents, layers=2)
        >>> result = moa.run("Explain the trade-offs of multi-agent systems")
    """

    def __init__(
        self,
        id: str = str(uuid.uuid4()),
        name: str = "MixtureOfAgents",
        description: str = "A class to run a mixture of agents and aggregate their responses.",
        agents: List[Agent] = None,
        aggregator_agent: Agent = None,
        aggregator_system_prompt: str = AGGREGATOR_SYSTEM_PROMPT_MAIN,
        layers: int = 3,
        max_loops: int = 1,
        output_type: OutputType = "final",
        aggregator_model_name: str = "claude-sonnet-4-20250514",
    ) -> None:
        """Initialize the mixture with worker and aggregator configuration.

        Args:
            id: Optional identifier accepted for API compatibility.
            name: Human-readable name for this mixture.
            description: Description of this mixture's purpose.
            agents: Worker agents to run concurrently on each layer.
            aggregator_agent: Optional preconfigured aggregator agent.
            aggregator_system_prompt: Prompt used to create the default
                aggregator agent.
            layers: Number of worker-agent rounds before aggregation.
            max_loops: Stored configuration value for compatibility.
            output_type: Desired formatted output type.
            aggregator_model_name: Model used for the default aggregator.

        Raises:
            ValueError: If no agents, aggregator system prompt, or layers
                are provided.
        """
        self.name = name
        self.description = description
        self.agents = agents
        self.aggregator_agent = aggregator_agent
        self.aggregator_system_prompt = aggregator_system_prompt
        self.layers = layers
        self.max_loops = max_loops
        self.output_type = output_type
        self.aggregator_model_name = aggregator_model_name

        self.reliability_check()

        self.conversation = Conversation()

        list_all_agents(
            agents=self.agents,
            conversation=self.conversation,
            description=self.description,
            name=self.name,
            add_to_conversation=True,
        )

        if self.aggregator_agent is None:
            self.aggregator_agent = self.aggregator_agent_setup()

    def reliability_check(self) -> None:
        """Validate required configuration before the workflow starts.

        Raises:
            ValueError: If the worker-agent list is empty, the aggregator
                prompt is missing, or ``layers`` is falsy.
        """
        logger.info(
            "Checking the reliability of the Mixture of Agents class."
        )

        if len(self.agents) == 0:
            raise ValueError("No agents provided.")

        if not self.aggregator_system_prompt:
            raise ValueError("No aggregator system prompt provided.")

        if not self.layers:
            raise ValueError("No layers provided.")

        logger.info("Reliability check passed.")
        logger.info("Mixture of Agents class is ready for use.")

    def aggregator_agent_setup(self):
        """Create the default aggregator agent.

        Returns:
            Agent: An agent configured to synthesize worker responses from
                the shared conversation context.
        """
        return Agent(
            agent_name="Aggregator Agent",
            agent_description="An agent that aggregates the responses of the other agents.",
            system_prompt=self.aggregator_system_prompt,
            model_name=self.aggregator_model_name,
            temperature=0.5,
            max_loops=1,
            output_type="str-all-except-first",
            dynamic_context_window=True,
        )

    def step(
        self,
        task: str,
        img: Optional[str] = None,
    ):
        """Run one worker layer concurrently.

        Args:
            task: On layer 0 this is the raw user task. On later layers it
                is ``"Original task: …\\n\\nPrevious layer synthesis:\\n…"``.
            img: Optional image path, URL, or encoded image payload passed
                through to each worker agent.

        Returns:
            A mapping of agent names to their outputs.
        """
        agent_outputs = run_agents_concurrently(
            agents=self.agents,
            task=task,
            img=img,
            return_agent_output_dict=True,
        )

        return agent_outputs

    def _run(
        self,
        task: str,
        img: Optional[str] = None,
    ):
        """Execute all layers and aggregate the final response.

        Args:
            task: User task for the mixture.
            img: Optional image input forwarded to worker agents.

        Returns:
            The conversation formatted according to ``self.output_type``.
        """

        self.conversation.add(role="User", content=task)

        # Workers receive only the original task on the first layer, and
        # task + previous-layer synthesis on subsequent layers. This avoids
        # re-sending the full growing transcript to every worker on every layer.
        worker_input = task
        prev_layer_output: Optional[str] = None

        for i in range(self.layers):
            if prev_layer_output is not None:
                worker_input = (
                    f"Original task: {task}\n\n"
                    f"Previous layer synthesis:\n{prev_layer_output}"
                )

            step_output = self.step(task=worker_input, img=img)

            for agent_name, agent_output in step_output.items():
                self.conversation.add(
                    role=agent_name, content=agent_output
                )

            # Summarise the layer as the concatenation of worker outputs so
            # the next layer has a compact view of what was produced.
            prev_layer_output = "\n\n".join(
                f"{name}: {out}" for name, out in step_output.items()
            )

        aggregator_output = self.aggregator_agent.run(
            task=self.conversation.get_str()
        )

        self.conversation.add(
            role=self.aggregator_agent.agent_name,
            content=aggregator_output,
        )

        return history_output_formatter(
            conversation=self.conversation, type=self.output_type
        )

    def run(
        self,
        task: str,
        img: Optional[str] = None,
    ):
        """Run the mixture for a single task.

        Args:
            task: User task for the mixture.
            img: Optional image input forwarded to worker agents.

        Returns:
            The formatted mixture output, or an error string if execution
            fails.
        """
        try:
            return self._run(task=task, img=img)
        except Exception as e:
            logger.error(f"Error running Mixture of Agents: {e}")
            return f"Error: {e}"

    def run_batched(self, tasks: List[str]) -> List[str]:
        """Run tasks sequentially through the same mixture instance.

        Args:
            tasks: Tasks to execute in order.

        Returns:
            A list of formatted responses, one per task.
        """
        return [self.run(task) for task in tasks]

    def run_concurrently(self, tasks: List[str]) -> List[str]:
        """Run multiple tasks concurrently through this mixture.

        Args:
            tasks: Tasks to submit to the mixture in parallel.

        Returns:
            A list of formatted responses as each task completes.
        """
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=os.cpu_count()
        ) as executor:
            futures = [
                executor.submit(self.run, task) for task in tasks
            ]
            return [
                future.result()
                for future in concurrent.futures.as_completed(futures)
            ]
