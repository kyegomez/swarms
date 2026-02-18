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

Author: Swarms Team
License: MIT
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Union, Dict, Any

from loguru import logger

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.utils.output_types import OutputType
from swarms.utils.any_to_str import any_to_str
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)

# System prompt for the reasoning agent that generates individual responses
CONSISTENCY_SYSTEM_PROMPT = """
You are a reasoning agent designed for complex problem-solving and decision-making. Your objective is to provide clear and reliable responses through structured reasoning. Begin by thoroughly understanding the problem, rephrasing it for clarity, and identifying key components. Develop a logical plan that breaks the problem into manageable steps, detailing your approach and any assumptions made. Validate your information with reliable sources and assess the accuracy of your calculations. Explore multiple solutions, weighing their pros and cons, and maintain transparency by documenting your reasoning process, uncertainties, and biases. Summarize your findings in a concise final answer that reflects your thorough analysis, ensuring it is well-organized and accessible. Adapt your reasoning to the context of the problem, integrating new information as needed, and implement error-handling strategies to address any issues that arise. Finally, reflect on your reasoning process to identify areas for improvement and ensure consistency across all reasoning paths.
"""

# Detailed prompt for the majority voting aggregation agent
majority_voting_prompt = """
Engage in a comprehensive and exhaustive majority voting analysis of the following conversation, ensuring a deep and thoughtful examination of the responses provided by each agent. This analysis should not only summarize the responses but also critically engage with the content, context, and implications of each agent's input.

Please adhere to the following detailed guidelines:

1. **Identification of Dominant Responses:**
   - Identify the most prevalent answer or recommendation across all agents. Provide a thorough rationale for its dominance, including an exploration of the factors that may have contributed to its acceptance among the agents. Discuss the context in which this consensus emerged and any relevant historical or theoretical frameworks that support this conclusion.

2. **Exploration of Disparities:**
   - Delve into any significant disparities or contrasting viewpoints between agents. Explore the underlying reasons for these differences, considering aspects such as differing methodologies, assumptions, or interpretations of the task at hand. Analyze how these contrasting perspectives may reflect broader debates within the field and what implications they hold for the overall understanding of the topic.

3. **Consensus and Disagreement Analysis:**
   - Highlight key areas of consensus and disagreement among the agents. Discuss the implications of these findings on the overall argument, including how consensus can strengthen certain claims while disagreement may indicate areas of uncertainty or contention. Provide examples from the conversation to illustrate these points and consider how they might influence future discussions or research directions.

4. **Critical Evaluation of Majority Opinion:**
   - Critically evaluate the strength of the majority opinion, considering factors such as the reasoning behind it and its mathematical validity if applicable. Assess whether the majority opinion is well-supported by evidence and logical reasoning, and discuss any potential weaknesses or oversights that may undermine its credibility. 

5. **Insights from Minority Viewpoints:**
   - Note any unique insights from minority viewpoints, assessing their potential contributions to a more nuanced understanding of the topic. Discuss how these minority perspectives can enrich the conversation and provide alternative angles that may have been overlooked by the majority. Consider the value of dissent in academic discourse and how it can lead to more robust conclusions.

6. **Synthesis of Recommendations:**
   - Provide a final synthesized recommendation based on the majority consensus, ensuring that it reflects a thorough consideration of all perspectives and is grounded in sound reasoning. This recommendation should not only summarize the majority view but also integrate insights from minority opinions, creating a comprehensive and balanced conclusion that acknowledges the complexity of the discussion.

Throughout your analysis, focus on uncovering clear patterns while being attentive to the subtleties and complexities inherent in the responses. Pay particular attention to the nuances of mathematical contexts where algorithmic thinking may be required, ensuring that your examination is both rigorous and accessible to a diverse audience.
"""


def aggregation_agent(
    responses: List[str],
    prompt: str = majority_voting_prompt,
    model_name: str = "gpt-4o-mini",
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
                                   Defaults to "gpt-4o-mini".

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

    return final_answer


class SelfConsistencyAgent:
    """
    A specialized agent that implements self-consistency for improved reasoning reliability.

    The SelfConsistencyAgent generates multiple independent responses to a given task and
    aggregates them into a single, consistent final answer. This approach is based on the
    research paper "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
    by Wang et al. (2022).

    Key Features:
    - Concurrent generation of multiple independent responses
    - Majority voting aggregation with detailed analysis
    - Evaluation mode for answer validation
    - Configurable output formats
    - Thread-safe execution

    The self-consistency technique works by:
    1. Generating multiple independent reasoning paths for the same problem
    2. Analyzing the consistency and agreement among these paths
    3. Aggregating the results using majority voting or consensus building
    4. Producing a final answer that reflects the most reliable consensus

    This approach helps mitigate issues like:
    - Random errors in individual reasoning paths
    - Biases in single reasoning approaches
    - Inconsistencies in complex problem-solving

    Reference:
        Wang, Y., Dong, W., Han, J., & Wang, W. (2022). Self-Consistency Improves Chain of
        Thought Reasoning in Language Models. arXiv preprint arXiv:2203.07870.
        https://arxiv.org/abs/2203.07870

    Example:
        >>> agent = SelfConsistencyAgent(
        ...     name="Math-Reasoning-Agent",
        ...     model_name="gpt-4o-mini",
        ...     num_samples=5,
        ...     max_loops=1
        ... )
        >>> result = agent.run("What is the 40th prime number?")
        >>> print(result)
    """

    def __init__(
        self,
        name: str = "Self-Consistency-Agent",
        description: str = "An agent that uses self consistency to generate a final answer.",
        model_name: str = "gpt-4o-mini",
        system_prompt: str = CONSISTENCY_SYSTEM_PROMPT,
        num_samples: int = 5,
        max_loops: int = 1,
        majority_voting_prompt: Optional[
            str
        ] = majority_voting_prompt,
        eval: bool = False,
        output_type: OutputType = "dict",
        random_models_on: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialize the SelfConsistencyAgent.

        Args:
            name (str, optional): Name of the agent. Defaults to "Self-Consistency-Agent".
            description (str, optional): Description of the agent's purpose.
                                       Defaults to "An agent that uses self consistency to generate a final answer.".
            model_name (str, optional): The underlying language model to use.
                                       Defaults to "gpt-4o-mini".
            system_prompt (str, optional): System prompt for the reasoning agent.
                                         Defaults to CONSISTENCY_SYSTEM_PROMPT.
            num_samples (int, optional): Number of independent responses to generate.
                                       Defaults to 5.
            max_loops (int, optional): Maximum number of reasoning loops per sample.
                                     Defaults to 1.
            majority_voting_prompt (Optional[str], optional): Custom prompt for majority voting.
                                                            Defaults to None.
            eval (bool, optional): Enable evaluation mode for answer validation.
                                 Defaults to False.
            output_type (OutputType, optional): Format of the output.
                                              Defaults to "dict".
            random_models_on (bool, optional): Enable random model selection for diversity.
                                             Defaults to False.
            **kwargs: Additional keyword arguments passed to the base Agent class.

        Note:
            The num_samples parameter determines how many independent reasoning paths
            will be generated. Higher values generally lead to more reliable results
            but increase computational cost and time.
        """
        self.name = name
        self.description = description
        self.model_name = model_name
        self.num_samples = num_samples
        self.max_loops = max_loops
        self.majority_voting_prompt = majority_voting_prompt
        self.eval = eval
        self.output_type = output_type
        self.system_prompt = system_prompt
        self.random_models_on = random_models_on
        self.conversation = Conversation()
        self.args = args
        self.kwargs = kwargs

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
        reasoning_agent = self._create_reasoning_agent()

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    reasoning_agent.run,
                    task=task,
                    img=img,
                    *args,
                    **kwargs,
                ): i
                for i in range(self.num_samples)
            }
            for future in as_completed(futures):
                response = future.result()
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

        # Aggregate responses using AI-powered aggregation
        final_answer = aggregation_agent(responses)

        self.conversation.add(
            role="Majority Voting Agent", content=final_answer
        )

        return history_output_formatter(
            self.conversation, self.output_type
        )

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
            max_loops=self.max_loops,
            random_models_on=self.random_models_on,
            output_type="str-all-except-first",
            **self.kwargs,
        )

    def check_responses_for_answer(
        self, responses: List[str], answer: str
    ) -> bool:
        """
        Check if the specified answer is present in any of the provided responses.

        This method performs a simple string matching to determine if the expected
        answer appears in any of the generated responses. It's useful for validation
        and evaluation purposes.

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
        for response in responses:
            if answer in response:
                return True

        # If the answer is not found, log the absence for each response
        for response in responses:
            if answer not in response:
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
        responses = []
        for task in tasks:
            response = self.run(task, *args, **kwargs)
            responses.append(response)
        return responses
