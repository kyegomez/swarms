"""
LLM Council - A Swarms implementation inspired by Andrej Karpathy's llm-council.

This implementation creates a council of specialized LLM agents that:
1. Each agent responds to the user query independently
2. All agents review and rank each other's (anonymized) responses
3. A Chairman LLM synthesizes all responses and rankings into a final answer

The council demonstrates how different models evaluate and rank each other's work,
often selecting responses from other models as superior to their own.
"""

import random
from typing import List, Optional

from loguru import logger

from swarms.prompts.llm_council_prompts import (
    get_chairman_prompt,
    get_claude_councilor_prompt,
    get_evaluation_prompt,
    get_gemini_councilor_prompt,
    get_gpt_councilor_prompt,
    get_grok_councilor_prompt,
    get_synthesis_prompt,
)
from swarms.structs.agent import Agent, reset_agents
from swarms.structs.context_utils import agent_answer
from swarms.structs.conversation import Conversation
from swarms.structs.execution_utils import batched_run
from swarms.structs.multi_agent_exec import (
    batched_grid_agent_execution,
    run_agents_concurrently,
)
from swarms.telemetry.otel import capture_init, trace_run
from swarms.utils.generate_id import generate_id
from swarms.utils.history_output_formatter import (
    HistoryOutputType,
    history_output_formatter,
)
from swarms.utils.litellm_wrapper import empty_usage

DEFAULT_COUNCIL_MODELS: List[str] = [
    "gpt-5.1",
    "gemini/gemini-3.5-flash-lite",
    "anthropic/claude-sonnet-5-5",
    "xai/grok-4-1-fast-reasoning",
]


def _raise_member_failure(results) -> None:
    """Re-raise the first exception the concurrent runner returned as a result."""
    for result in results:
        if isinstance(result, Exception):
            raise result


class LLMCouncil:
    """
    An LLM Council that orchestrates multiple specialized agents to collaboratively
    answer queries through independent responses, peer review, and synthesis.

    The council follows this workflow:
    1. Dispatch query to all council members in parallel
    2. Collect all responses (anonymized)
    3. Have each member review and rank all responses
    4. Chairman synthesizes everything into final response
    """

    def __init__(
        self,
        id: Optional[str] = None,
        name: str = "LLM Council",
        description: str = "A collaborative council of LLM agents where each member independently answers a query, reviews and ranks anonymized peer responses, and a chairman synthesizes the best elements into a final answer.",
        council_members: Optional[List[Agent]] = None,
        chairman_model: str = "gpt-5.1",
        verbose: bool = True,
        output_type: HistoryOutputType = "dict-all-except-first",
        council_models: Optional[List[str]] = None,
    ):
        """
        Initialize the LLM Council.

        Args:
            council_members: List of Agent instances representing council members.
                           If None, creates a default council of four members:
                           GPT (gpt-5.1), Gemini (gemini/gemini-3.5-flash-lite),
                           Claude (anthropic/claude-sonnet-5-5) and Grok
                           (xai/grok-4-1-fast-reasoning).
            chairman_model: Model name for the Chairman agent that synthesizes responses.
            verbose: Whether to log progress through each stage.
            output_type: Format for the output. Options: "list", "dict", "string", "final", "json", "yaml", etc.
            council_models: Optional list of four model names overriding
                DEFAULT_COUNCIL_MODELS for the preset GPT, Gemini, Claude,
                and Grok councilors when council_members is None.
        """
        self.id = id or generate_id("llm-council")
        self.name = name
        self.description = description
        self.verbose = verbose
        self.output_type = output_type
        self.council_models = council_models
        self.chairman_model = chairman_model

        # Create default council members if none provided
        if council_members is None:
            self.council_members = self._create_default_council(
                council_models
            )
        else:
            self.council_members = council_members

        # Create Chairman agent
        self.chairman = Agent(
            agent_name="Chairman",
            agent_description="Chairman of the LLM Council, responsible for synthesizing all responses and rankings into a final answer",
            system_prompt=get_chairman_prompt(),
            model_name=chairman_model,
            max_loops=1,
            verbose=verbose,
        )

        self.conversation = Conversation(
            name=f"[LLM Council] [Conversation][{name}]"
        )

        if self.verbose:
            members = ", ".join(
                m.agent_name for m in self.council_members
            )
            logger.info(
                f"[{self.name}] Initialized with {len(self.council_members)} members: {members}"
            )

        # Capture the full __init__ configuration if telemetry is enabled.
        capture_init(self)

    def _create_default_council(
        self, model_names: Optional[List[str]] = None
    ) -> List[Agent]:
        """Create default council members with specialized prompts and models.

        Args:
            model_names: Optional list of four model names for GPT, Gemini,
                Claude, and Grok councilors. Defaults to DEFAULT_COUNCIL_MODELS.

        Returns:
            List of Agent instances configured as council members.
        """
        models = (
            list(DEFAULT_COUNCIL_MODELS)
            if model_names is None
            else list(model_names)
        )
        if len(models) != len(DEFAULT_COUNCIL_MODELS):
            raise ValueError(
                f"council_models must contain {len(DEFAULT_COUNCIL_MODELS)} "
                f"model names (GPT, Gemini, Claude, Grok), got {len(models)}"
            )

        gpt_agent = Agent(
            agent_name="GPT-Councilor",
            agent_description="Analytical and comprehensive AI councilor specializing in deep analysis and thorough responses",
            system_prompt=get_gpt_councilor_prompt(),
            model_name=models[0],
            max_loops=1,
            verbose=False,
        )

        gemini_agent = Agent(
            agent_name="Gemini-Councilor",
            agent_description="Concise and well-processed AI councilor specializing in clear, structured responses",
            system_prompt=get_gemini_councilor_prompt(),
            model_name=models[1],
            max_loops=1,
            verbose=False,
        )

        claude_agent = Agent(
            agent_name="Claude-Councilor",
            agent_description="Thoughtful and balanced AI councilor specializing in nuanced and well-reasoned responses",
            system_prompt=get_claude_councilor_prompt(),
            model_name=models[2],
            max_loops=1,
            verbose=False,
            top_p=None,
        )

        grok_agent = Agent(
            agent_name="Grok-Councilor",
            agent_description="Creative and innovative AI councilor specializing in unique perspectives and creative solutions",
            system_prompt=get_grok_councilor_prompt(),
            model_name=models[3],
            max_loops=1,
            verbose=False,
        )

        return [gpt_agent, gemini_agent, claude_agent, grok_agent]

    @trace_run(
        "LLMCouncil.run",
        input_params=("task",),
    )
    def run(self, task: str):
        """
        Execute the full LLM Council workflow.

        Args:
            task: The user's task to process.

        Returns:
            Formatted output based on output_type, containing conversation history
            with all council member responses, evaluations, and final synthesis.

        Raises:
            ValueError: If task is not a non-empty string.
            Exception: The first council member's error, if any member
                fails while answering or ranking. The chairman is not run.
        """
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")

        self.conversation.clear()
        reset_agents([*self.council_members, self.chairman])

        self.conversation.add(role="User", content=task)

        # Step 1: Get responses from all council members in parallel
        if self.verbose:
            logger.info(
                f"[{self.name}] Collecting responses from {len(self.council_members)} members"
            )

        results_dict = run_agents_concurrently(
            self.council_members,
            task=task,
            return_agent_output_dict=True,
        )
        _raise_member_failure(results_dict.values())

        # Map results to member names
        original_responses = {
            member.agent_name: results_dict.get(member.agent_name, "")
            for member in self.council_members
        }

        # Add each council member's response to conversation
        for member_name, response in original_responses.items():
            self.conversation.add(role=member_name, content=response)

        # Anonymise responses as A, B, C... for evaluation
        anonymous_ids = [
            chr(65 + i) for i in range(len(self.council_members))
        ]
        random.shuffle(anonymous_ids)  # Shuffle to ensure anonymity

        anonymous_responses = {
            anonymous_ids[i]: original_responses[member.agent_name]
            for i, member in enumerate(self.council_members)
        }

        # Create mapping from anonymous ID to member name (for later reference)
        id_to_member = {
            anonymous_ids[i]: member.agent_name
            for i, member in enumerate(self.council_members)
        }

        if self.verbose:
            logger.info(
                f"[{self.name}] Members ranking the anonymized responses"
            )

        # Every member ranks all responses, concurrently
        reset_agents(self.council_members)
        evaluation_tasks = [
            get_evaluation_prompt(
                task, anonymous_responses, member.agent_name
            )
            for member in self.council_members
        ]

        # Run evaluations concurrently using batched_grid_agent_execution
        evaluation_results = batched_grid_agent_execution(
            self.council_members, evaluation_tasks
        )
        _raise_member_failure(evaluation_results)

        # Map results to member names
        evaluations = {
            member.agent_name: agent_answer(
                member, fallback=evaluation_results[i]
            )
            for i, member in enumerate(self.council_members)
        }

        # Add each council member's evaluation to conversation
        for member_name, evaluation in evaluations.items():
            self.conversation.add(
                role=f"{member_name}-Evaluation", content=evaluation
            )

        # Step 4: Chairman synthesizes everything
        if self.verbose:
            logger.info(
                f"[{self.name}] Chairman synthesizing the final answer"
            )

        synthesis_prompt = get_synthesis_prompt(
            task, original_responses, evaluations, id_to_member
        )

        final_response = self.chairman.run(task=synthesis_prompt)

        # Add chairman's final response to conversation
        self.conversation.add(role="Chairman", content=final_response)

        if self.verbose:
            logger.info(f"[{self.name}] Session complete")

        # Format and return output using history_output_formatter
        return history_output_formatter(
            conversation=self.conversation, type=self.output_type
        )

    def batched_run(self, tasks: List[str]):
        """
        Run the LLM Council workflow for a batch of tasks.

        Args:
            tasks: List of tasks to process

        Returns:
            List of formatted outputs based on output_type
        """
        return batched_run(self.run, tasks)

    @property
    def usage(self) -> dict:
        """
        Provider token usage summed over every council member and the chairman.

        Returns:
            A new dict with input_tokens, output_tokens, cached_tokens,
            reasoning_tokens and total_tokens, totalled across every run.
        """
        agents = {
            id(agent): agent
            for agent in [*self.council_members, self.chairman]
        }
        total = empty_usage()
        for agent in agents.values():
            for key, value in agent.usage.items():
                if key in total:
                    total[key] += value
        return total
