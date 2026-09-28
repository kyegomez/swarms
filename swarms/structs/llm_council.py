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
from swarms.structs.execution_utils import batched_run
from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.multi_agent_exec import (
    batched_grid_agent_execution,
    run_agents_concurrently,
)
from swarms.utils.history_output_formatter import (
    HistoryOutputType,
    history_output_formatter,
)
from swarms.telemetry.otel import capture_init, trace_run
from swarms.utils.generate_id import generate_id


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
    ):
        """
        Initialize the LLM Council.

        Args:
            council_members: List of Agent instances representing council members.
                           If None, creates default council with GPT-5.1, Gemini 3 Pro,
                           Claude Sonnet 4.5, and Grok-4.
            chairman_model: Model name for the Chairman agent that synthesizes responses.
            verbose: Whether to log progress through each stage.
            output_type: Format for the output. Options: "list", "dict", "string", "final", "json", "yaml", etc.
        """
        self.id = id or generate_id("llm-council")
        self.name = name
        self.description = description
        self.verbose = verbose
        self.output_type = output_type

        # Create default council members if none provided
        if council_members is None:
            self.council_members = self._create_default_council()
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
            temperature=0.7,
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

    def _create_default_council(self) -> List[Agent]:
        """
        Create default council members with specialized prompts and models.

        Returns:
            List of Agent instances configured as council members.
        """

        # GPT-5.1 Agent - Analytical and comprehensive
        gpt_agent = Agent(
            agent_name="GPT-5.1-Councilor",
            agent_description="Analytical and comprehensive AI councilor specializing in deep analysis and thorough responses",
            system_prompt=get_gpt_councilor_prompt(),
            model_name="gpt-5.1",
            max_loops=1,
            verbose=False,
            temperature=0.7,
        )

        # Gemini 3 Pro Agent - Concise and processed
        gemini_agent = Agent(
            agent_name="Gemini-3-Pro-Councilor",
            agent_description="Concise and well-processed AI councilor specializing in clear, structured responses",
            system_prompt=get_gemini_councilor_prompt(),
            model_name="gemini-2.5-flash",  # Using available Gemini model
            max_loops=1,
            verbose=False,
            temperature=0.7,
        )

        # Claude Sonnet 4.5 Agent - Balanced and thoughtful
        claude_agent = Agent(
            agent_name="Claude-Sonnet-4.5-Councilor",
            agent_description="Thoughtful and balanced AI councilor specializing in nuanced and well-reasoned responses",
            system_prompt=get_claude_councilor_prompt(),
            model_name="anthropic/claude-sonnet-4-5",  # Using available Claude model
            max_loops=1,
            verbose=False,
            temperature=0.0,
            top_p=None,
        )

        # Grok-4 Agent - Creative and innovative
        grok_agent = Agent(
            agent_name="Grok-4-Councilor",
            agent_description="Creative and innovative AI councilor specializing in unique perspectives and creative solutions",
            system_prompt=get_grok_councilor_prompt(),
            model_name="xai/grok-4-1-fast-reasoning",  # Using available model as proxy for Grok-4
            max_loops=1,
            verbose=False,
            temperature=0.8,
        )

        members = [gpt_agent, gemini_agent, claude_agent, grok_agent]

        return members

    @trace_run(
        "LLMCouncil.run",
        input_params=("task", "tasks", "img", "imgs"),
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
            ValueError: If ``task`` is not a non-empty string.
        """
        if not isinstance(task, str) or not task.strip():
            raise ValueError("task must be a non-empty string")

        self.conversation.clear()

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

        # Map results to member names
        original_responses = {
            member.agent_name: response
            for member, response in zip(
                self.council_members,
                [
                    results_dict.get(member.agent_name, "")
                    for member in self.council_members
                ],
            )
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

        # Map results to member names
        evaluations = {
            member.agent_name: evaluation_results[i]
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
