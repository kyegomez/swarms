"""
Self-Consistency Agent Implementation

This module implements the SelfConsistencyAgent, a specialized agent that leverages the
self-consistency technique to improve reasoning reliability and accuracy. The agent generates
multiple independent responses to a given task and aggregates them into a single, consistent
final answer using majority voting and sophisticated aggregation techniques.

The self-consistency approach is based on the research paper:
"Self-Consistency Improves Chain of Thought Reasoning in Language Models"
by Wang et al. (2022) - https://arxiv.org/abs/2203.07870

Key Features:
- Concurrent generation of multiple independent responses
- Majority voting aggregation with detailed analysis
- Evaluation mode for answer validation
- Configurable output formats
- Thread-safe execution
"""

import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, Dict, List, Optional, Union

from loguru import logger

from swarms.prompts.self_consistency_prompts import (
    CONSISTENCY_SYSTEM_PROMPT,
    majority_voting_prompt,
)
from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.execution_utils import batched_run
from swarms.utils.any_to_str import any_to_str
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.litellm_wrapper import empty_usage
from swarms.utils.output_types import OutputType


def aggregation_agent(
    responses: List[str],
    prompt: str = majority_voting_prompt,
    model_name: str = "gpt-5.4",
    on_usage: Optional[Callable[[dict], None]] = None,
) -> str:
    """
    Aggregates a list of responses into a single final answer using an AI-powered aggregation agent.

    This function creates a specialized agent that analyzes multiple responses and synthesizes
    them into a coherent final answer. The aggregation process considers consensus, disagreements,
    and minority viewpoints to produce a well-reasoned conclusion.

    Args:
        responses (List[str]): List of responses to be aggregated
        prompt (str, optional): Custom prompt for the aggregation agent.
                               Defaults to the majority_voting_prompt.
        model_name (str, optional): Model to use for aggregation.
                                   Defaults to "gpt-5.4".
        on_usage (Optional[Callable[[dict], None]], optional): Called with
                                   the aggregation agent's usage after it runs.

    Returns:
        str: The aggregated final answer

    Example:
        >>> responses = ["Answer A", "Answer B", "Answer A"]
        >>> final_answer = aggregation_agent(responses)
        >>> print(final_answer)
        "Based on the majority consensus..."
    """
    task = any_to_str(responses)

    agent = Agent(
        agent_name="Aggregation-Agent",
        agent_description="An agent that aggregates a list of responses into a single final answer.",
        model_name=model_name,
        system_prompt=prompt,
        max_loops=1,
    )

    final_answer = agent.run(task)

    if on_usage is not None:
        on_usage(agent.usage)

    return final_answer


class SelfConsistencyAgent:
    """
    A specialized agent that implements self-consistency for improved reasoning reliability.

    The SelfConsistencyAgent generates multiple independent responses to a given task and
    aggregates them into a single, consistent final answer. This approach is based on the
    research paper "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
    by Wang et al. (2022).

    Reference:
        Wang, Y., Dong, W., Han, J., & Wang, W. (2022). Self-Consistency Improves Chain of
        Thought Reasoning in Language Models. arXiv preprint arXiv:2203.07870.
        https://arxiv.org/abs/2203.07870

    Args:
        name (str, optional): Name of the agent. Defaults to "Self-Consistency-Agent".
        description (str, optional): Description of the agent's purpose.
                                    Defaults to "An agent that uses self consistency to generate a final answer.".
        model_name (str, optional): The underlying language model to use.
                                    Defaults to "gpt-5.4".
        system_prompt (str, optional): System prompt for the reasoning agent.
                                        Defaults to CONSISTENCY_SYSTEM_PROMPT.
        num_samples (int, optional): Number of independent responses to generate.
                                    Defaults to 5.
        majority_voting_prompt (Optional[str], optional): Custom prompt for majority voting.
                                                        Defaults to None.
        eval (bool, optional): Enable evaluation mode for answer validation.
                                Defaults to False.
        output_type (OutputType, optional): Format of the output.
                                            Defaults to "dict".
        aggregator_agent (Optional[Agent], optional): Your own agent that votes on the
                                            samples and writes the final answer. Its
                                            conversation is reset before each vote, and
                                            majority_voting_prompt is not used. Needs
                                            preset_agents=False. Defaults to None.
        preset_agents (bool, optional): Use the built-in majority voting agent. Set it to
                                            False to vote with aggregator_agent instead.
                                            Defaults to True.
        **kwargs: Additional keyword arguments passed to the base Agent class.

    Note:
        The num_samples parameter determines how many independent reasoning paths
        will be generated. Higher values generally lead to more reliable results
        but increase computational cost and time.

    Example:
        >>> agent = SelfConsistencyAgent(
        ...     name="Math-Reasoning-Agent",
        ...     model_name="gpt-5.4",
        ...     num_samples=5,
        ... )
        >>> result = agent.run("What is the 40th prime number?")
        >>> print(result)
    """

    def __init__(
        self,
        name: str = "Self-Consistency-Agent",
        description: str = "An agent that uses self consistency to generate a final answer.",
        model_name: str = "gpt-5.4",
        system_prompt: str = CONSISTENCY_SYSTEM_PROMPT,
        num_samples: int = 5,
        majority_voting_prompt: Optional[
            str
        ] = majority_voting_prompt,
        eval: bool = False,
        output_type: OutputType = "dict",
        aggregator_agent: Optional[Agent] = None,
        preset_agents: bool = True,
        *args,
        **kwargs,
    ):
        self.name = name
        self.description = description
        self.model_name = model_name
        self.num_samples = num_samples
        self.majority_voting_prompt = majority_voting_prompt
        self.eval = eval
        self.output_type = output_type
        self.system_prompt = system_prompt
        self.aggregator_agent = aggregator_agent
        self.preset_agents = preset_agents
        self.conversation = Conversation()
        self.args = args
        self.kwargs = kwargs
        self._usage = empty_usage()

        self.reliability_check()

    def reliability_check(self) -> None:
        """Reject a configuration the agent cannot run.

        Raises:
            ValueError: When num_samples is 0, or when preset_agents
                and aggregator_agent do not agree.
        """
        if self.num_samples == 0:
            raise ValueError(
                "SelfConsistencyAgent: num_samples must be greater than 0"
            )

        if self.preset_agents and self.aggregator_agent is not None:
            raise ValueError(
                "SelfConsistencyAgent: set preset_agents=False to "
                "vote with your own aggregator_agent"
            )

        if not self.preset_agents and self.aggregator_agent is None:
            raise ValueError(
                "SelfConsistencyAgent: preset_agents=False needs an "
                "aggregator_agent to vote on the samples"
            )

    @property
    def usage(self) -> dict:
        """Provider token usage summed over every sample and aggregation agent this instance has run."""
        return dict(self._usage)

    def _add_usage(self, call_usage: dict) -> None:
        for key in self._usage:
            self._usage[key] += call_usage.get(key, 0)

    def run(
        self,
        task: str,
        img: Optional[str] = None,
        answer: Optional[str] = None,
        *args,
        **kwargs,
    ) -> Union[str, Dict[str, Any]]:
        """
        Generate multiple responses for the given task and aggregate them concurrently.

        This method implements the core self-consistency algorithm:
        1. Generates multiple independent responses using concurrent execution
        2. Optionally validates responses against a known answer (if eval=True)
        3. Aggregates responses using an AI-powered aggregation agent
        4. Returns the final result in the specified output format

        Args:
            task (str): The input prompt or task to be solved
            answer (Optional[str], optional): Expected answer for validation (if eval=True).
                                            Defaults to None.
            *args: Additional positional arguments passed to the base agent's run method
            **kwargs: Additional keyword arguments passed to the base agent's run method

        Returns:
            Union[str, Dict[str, Any]]: The aggregated final answer in the specified format

        Raises:
            RuntimeError: If evaluation mode is enabled and the expected answer is not found
                         in any of the generated responses

        Example:
            >>> agent = SelfConsistencyAgent(num_samples=3)
            >>> result = agent.run("What is 2 + 2?")
            >>> print(result)

            >>> # With evaluation mode
            >>> result = agent.run("What is 2 + 2?", answer="4", eval=True)
        """
        responses = []

        self.conversation.add(role="User", content=task)

        # Generate multiple independent responses concurrently
        with ThreadPoolExecutor() as executor:
            agents = [
                self._create_reasoning_agent()
                for _ in range(self.num_samples)
            ]
            futures = {
                executor.submit(
                    agent.run,
                    task=task,
                    img=img,
                    *args,
                    **kwargs,
                ): agent
                for agent in agents
            }
            for future in as_completed(futures):
                response = future.result()
                self._add_usage(futures[future].usage)
                responses.append(response)

        self.conversation.add(role=self.name, content=responses)

        # Optional evaluation against known answer
        if self.eval:
            if answer is not None:
                correct = self.check_responses_for_answer(
                    responses, answer
                )

                if not correct:
                    logger.info(
                        "The answer is not correct. Please try again."
                    )
                    return None

        final_answer = self._aggregate(responses)

        self.conversation.add(
            role="Majority Voting Agent", content=final_answer
        )

        return history_output_formatter(
            self.conversation, self.output_type
        )

    def _aggregate(self, responses: List[str]) -> str:
        """Vote on the sample responses and return the final answer.

        Args:
            responses: The answers the sample agents gave.

        Returns:
            The final answer from the preset voting agent or from
            aggregator_agent.
        """
        if self.preset_agents:
            return aggregation_agent(
                responses,
                prompt=self.majority_voting_prompt
                or majority_voting_prompt,
                model_name=self.model_name,
                on_usage=self._add_usage,
            )

        agent = self.aggregator_agent
        before = agent.usage

        # A fresh conversation, so an earlier vote is not in this one
        agent.reset()
        final_answer = agent.run(any_to_str(responses))

        # The caller's agent keeps a lifetime total, so count only this vote
        after = agent.usage
        self._add_usage(
            {key: after[key] - before.get(key, 0) for key in after}
        )

        return final_answer

    def _create_reasoning_agent(self) -> Agent:
        """
        Create a reasoning agent instance for generating individual responses.

        Returns:
            Agent: A configured Agent instance for reasoning tasks
        """
        return Agent(
            agent_name=self.name,
            description=self.description,
            model_name=self.model_name,
            system_prompt=self.system_prompt,
            output_type="str-all-except-first",
            **self.kwargs,
        )

    def check_responses_for_answer(
        self, responses: List[str], answer: str
    ) -> bool:
        """
        Check if the specified answer is present in any of the provided responses.

        The answer must appear on its own, not inside a longer word or number,
        so an answer of 4 matches "The answer is 4." but not "42" or "0.4".
        It's useful for validation and evaluation purposes.

        Args:
            responses (List[str]): List of responses to check
            answer (str): The answer to look for in the responses

        Returns:
            bool: True if the answer is found in any response, False otherwise

        Example:
            >>> agent = SelfConsistencyAgent()
            >>> responses = ["The answer is 42", "I think it's 42", "Not sure"]
            >>> found = agent.check_responses_for_answer(responses, "42")
            >>> print(found)  # True
        """
        pattern = re.compile(
            rf"(?<![\w+-])(?<!\d[.,/]){re.escape(answer)}(?!\w)(?![.,/]\d)"
        )
        if any(pattern.search(response) for response in responses):
            return True

        # If the answer is not found, log the absence for each response
        for response in responses:
            self.conversation.add(
                role="User",
                content=f"The answer '{answer}' is not found in the response: '{response}'",
            )
            logger.info(
                f"The answer '{answer}' is not found in the response: '{response}'"
            )
        return False

    def batched_run(
        self, tasks: List[str], *args, **kwargs
    ) -> List[Union[str, Dict[str, Any]]]:
        """
        Run the agent on multiple tasks in batch.

        This method processes multiple tasks sequentially, applying the self-consistency
        approach to each task independently. It's useful for processing large datasets
        or multiple related problems.

        Args:
            tasks (List[str]): List of tasks to be processed
            *args: Additional positional arguments passed to the run method
            **kwargs: Additional keyword arguments passed to the run method

        Returns:
            List[Union[str, Dict[str, Any]]]: List of results for each task

        Example:
            >>> agent = SelfConsistencyAgent()
            >>> tasks = ["What is 2+2?", "What is 3+3?", "What is 4+4?"]
            >>> results = agent.batched_run(tasks)
            >>> print(len(results))  # 3
        """
        return batched_run(self.run, tasks, *args, **kwargs)
