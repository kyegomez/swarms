"""
AdvisorSwarm — Advisor Strategy Pattern

Implements the advisor strategy described in Anthropic's research
(April 2026): pair a cheaper executor model that drives the task
end-to-end with a powerful advisor model consulted on-demand
between executor turns.

Architecture (from Anthropic's diagram):

    Main loop ──> [ Executor (Sonnet) ] ──tool call──> [ Advisor (Opus) ]
                        |       ^                           |       ^
                   read/write   |                      sends advice  |
                        v       |                           v       |
                  [ Shared context: conversation · tools · history ]
                  Advisor reads the same context as Executor

The executor runs every turn. The advisor is on-demand — consulted
between executor turns when budget allows. Both read from and write
to the same shared conversation. The advisor never calls tools or
produces user-facing output.

Reference: "The advisor strategy: Give agents an intelligence
boost" (Anthropic, April 2026)
"""

from typing import Any, Callable, List, Optional

from loguru import logger

from swarms.prompts.advisor_swarm_prompts import (
    ADVISOR_SYSTEM_PROMPT,
    EXECUTOR_SYSTEM_PROMPT,
)
from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.swarm_id import swarm_id
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.loguru_logger import initialize_logger
from swarms.utils.output_types import OutputType

logger = initialize_logger(log_folder="advisor_swarm")


class AdvisorSwarm:
    """Implements the Advisor Strategy: pairs a cheaper executor model
    with a powerful advisor model consulted on-demand between executor
    turns.

    The executor runs in a main loop. Before each executor turn, the
    advisor reads the full shared conversation context and provides
    strategic guidance (if budget allows). Both agents read from and
    write to the same conversation.

    This is provider-agnostic — any model supported by LiteLLM works
    for either role.

    Args:
        id: Unique identifier for this swarm instance.
        name: Human-readable name.
        description: Description of the swarm's purpose.
        executor_model_name: Model for the executor agent.
        advisor_model_name: Model for the advisor agent.
        executor_system_prompt: System prompt for the executor.
        advisor_system_prompt: System prompt for the advisor.
        max_advisor_uses: Max advisor consultations per run().
        max_loops: Number of executor turns.
        output_type: Format for conversation history output.
        verbose: Enable detailed logging.
        executor_agent: Optional pre-configured Agent for execution
            (e.g., with tools or MCP configs).
        advisor_agent: Optional pre-configured Agent for advising.
        tools: Tools available to the executor agent only.

    Examples:
        >>> swarm = AdvisorSwarm(
        ...     executor_model_name="claude-sonnet-4-6",
        ...     advisor_model_name="claude-opus-4-6",
        ... )
        >>> result = swarm.run("Write a Python function to merge two sorted lists")
    """

    def __init__(
        self,
        id: str = None,
        name: str = "AdvisorSwarm",
        description: str = "An executor-advisor swarm implementing the advisor strategy pattern",
        executor_model_name: str = "claude-sonnet-4-6",
        advisor_model_name: str = "claude-opus-4-6",
        executor_system_prompt: str = EXECUTOR_SYSTEM_PROMPT,
        advisor_system_prompt: str = ADVISOR_SYSTEM_PROMPT,
        max_advisor_uses: int = 3,
        max_loops: int = 1,
        output_type: OutputType = "dict-all-except-first",
        verbose: bool = False,
        executor_agent: Optional[Agent] = None,
        advisor_agent: Optional[Agent] = None,
        tools: Optional[List[Callable]] = None,
        *args,
        **kwargs,
    ) -> None:
        self.id = id or swarm_id()
        self.name = name
        self.description = description
        self.executor_model_name = executor_model_name
        self.advisor_model_name = advisor_model_name
        self.executor_system_prompt = executor_system_prompt
        self.advisor_system_prompt = advisor_system_prompt
        self.max_advisor_uses = max_advisor_uses
        self.max_loops = max_loops
        self.output_type = output_type
        self.verbose = verbose
        self.tools = tools

        self.reliability_check()

        self.conversation = Conversation()

        self.executor_agent = (
            executor_agent or self._create_executor()
        )
        self.advisor_agent = advisor_agent or self._create_advisor()

    def reliability_check(self):
        """Validate swarm configuration."""
        if self.max_advisor_uses < 0:
            raise ValueError(
                f"max_advisor_uses must be >= 0, got {self.max_advisor_uses}"
            )
        if self.max_loops < 1:
            raise ValueError(
                f"max_loops must be >= 1, got {self.max_loops}"
            )
        if not self.executor_model_name:
            raise ValueError("executor_model_name must be provided")
        if not self.advisor_model_name:
            raise ValueError("advisor_model_name must be provided")

        if self.verbose:
            logger.info(
                f"AdvisorSwarm initialized: "
                f"executor={self.executor_model_name}, "
                f"advisor={self.advisor_model_name}, "
                f"max_advisor_uses={self.max_advisor_uses}, "
                f"max_loops={self.max_loops}"
            )

    def _create_executor(self) -> Agent:
        """Create the executor agent with tools."""
        return Agent(
            agent_name="Executor",
            agent_description="Executes tasks using advisor strategic guidance",
            system_prompt=self.executor_system_prompt,
            model_name=self.executor_model_name,
            max_loops=1,
            temperature=1.0,
            output_type="final",
            verbose=self.verbose,
            tools=self.tools,
        )

    def _create_advisor(self) -> Agent:
        """Create the advisor agent. No tools — guidance only."""
        return Agent(
            agent_name="Advisor",
            agent_description="Provides strategic guidance to the executor",
            system_prompt=self.advisor_system_prompt,
            model_name=self.advisor_model_name,
            max_loops=1,
            temperature=1.0,
            output_type="final",
            verbose=self.verbose,
        )

    def run(
        self,
        task: Optional[str] = None,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        *args,
        **kwargs,
    ) -> Any:
        """Execute the advisor-executor orchestration flow.

        The executor runs in a main loop for max_loops turns. Before
        each turn, the advisor reads the full shared conversation and
        provides guidance (if budget allows). Both agents read from
        and write to the same conversation.

        Args:
            task: The task to accomplish.
            img: Optional single image input.
            imgs: Optional list of image inputs.

        Returns:
            Formatted conversation history per output_type.
        """
        if not task:
            raise ValueError("A task is required")

        self.conversation.add(role="User", content=task)
        advisor_uses = 0

        for turn in range(self.max_loops):
            if self.verbose:
                logger.info(
                    f"[AdvisorSwarm] Turn {turn + 1}/{self.max_loops}"
                )

            # --- Advisor guidance (if budget allows) ---
            if advisor_uses < self.max_advisor_uses:
                context = self.conversation.get_str()
                advisor_prompt = (
                    f"Read the shared conversation context below and "
                    f"provide strategic guidance for the Executor.\n\n"
                    f"--- SHARED CONTEXT ---\n{context}\n"
                    f"--- END SHARED CONTEXT ---"
                )

                advice = self.advisor_agent.run(task=advisor_prompt)
                advisor_uses += 1
                self.conversation.add(role="Advisor", content=advice)

                if self.verbose:
                    logger.info(
                        f"[AdvisorSwarm] Advisor consulted "
                        f"({advisor_uses}/{self.max_advisor_uses})"
                    )

            # --- Executor turn ---
            context = self.conversation.get_str()
            executor_prompt = (
                f"Read the shared conversation context below — it "
                f"includes the task and any Advisor guidance — then "
                f"produce your output.\n\n"
                f"--- SHARED CONTEXT ---\n{context}\n"
                f"--- END SHARED CONTEXT ---"
            )

            output = self.executor_agent.run(
                task=executor_prompt, img=img, imgs=imgs
            )
            self.conversation.add(role="Executor", content=output)

            if self.verbose:
                logger.info(
                    f"[AdvisorSwarm] Executor completed turn {turn + 1}"
                )

        return history_output_formatter(
            conversation=self.conversation,
            type=self.output_type,
        )

    def batched_run(
        self, tasks: List[str], *args, **kwargs
    ) -> List[Any]:
        """Run multiple tasks sequentially.

        Args:
            tasks: List of task strings.

        Returns:
            List of results, one per task.
        """
        return [self.run(task=t, *args, **kwargs) for t in tasks]

    def __call__(self, task: str, *args, **kwargs) -> Any:
        """Make the swarm callable."""
        return self.run(task=task, *args, **kwargs)
