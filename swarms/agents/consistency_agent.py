from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List

from loguru import logger

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.malt import majority_voting_prompt
from swarms.structs.output_types import OutputType
from swarms.utils.any_to_str import any_to_str
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)

CONSISTENCY_SYSTEM_PROMPT = """
You are a reasoning agent designed for complex problem-solving and decision-making. Your objective is to provide clear and reliable responses through structured reasoning. Begin by thoroughly understanding the problem, rephrasing it for clarity, and identifying key components. Develop a logical plan that breaks the problem into manageable steps, detailing your approach and any assumptions made. Validate your information with reliable sources and assess the accuracy of your calculations. Explore multiple solutions, weighing their pros and cons, and maintain transparency by documenting your reasoning process, uncertainties, and biases. Summarize your findings in a concise final answer that reflects your thorough analysis, ensuring it is well-organized and accessible. Adapt your reasoning to the context of the problem, integrating new information as needed, and implement error-handling strategies to address any issues that arise. Finally, reflect on your reasoning process to identify areas for improvement and ensure consistency across all reasoning paths.
"""


def aggregation_agent(
    responses: List[str],
    prompt: str = majority_voting_prompt,
    model_name: str = "gpt-4o-mini",
) -> str:
    """
    Aggregates a list of responses into a single final answer.
    """
    task = any_to_str(responses)

    agent = Agent(
        agent_name="Aggregation-Agent",
        description="An agent that aggregates a list of responses into a single final answer.",
        model_name=model_name,
        system_prompt=prompt,
        max_loops=1,
    )

    final_answer = agent.run(task)

    return final_answer


class SelfConsistencyAgent(Agent):
    def __init__(
        self,
        name: str = "Self-Consistency-Agent",
        description: str = "An agent that uses self consistency to generate a final answer.",
        system_prompt: str = CONSISTENCY_SYSTEM_PROMPT,
        num_samples: int = 5,
        max_loops: int = 1,
        majority_voting_prompt: str = None,
        eval: bool = False,
        output_type: OutputType = "dict",
        **kwargs,
    ):
        """
        Initializes the SelfConsistencyAgent.

        Args:
            num_samples (int): Number of independent responses to sample.
            **kwargs: Other keyword arguments passed to the base Agent.
        """
        super().__init__(
            name=name,
            description=description,
            **kwargs,
        )
        self.num_samples = num_samples
        self.conversation = Conversation()
        self.max_loops = max_loops
        self.majority_voting_prompt = majority_voting_prompt
        self.eval = eval
        self.output_type = output_type
        self.system_prompt = system_prompt

    def run(
        self, task: str, answer: str = None, *args, **kwargs
    ) -> str:
        """
        Generates multiple responses for the given prompt and aggregates them concurrently.

        Args:
            task (str): The input prompt.

        Returns:
            str: The aggregated final answer.
        """
        responses = []
        logger.info(
            f"Generating {self.num_samples} responses concurrently..."
        )

        self.conversation.add(role="User", content=task)

        with ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(super().run, task, *args, **kwargs): i
                for i in range(self.num_samples)
            }
            for future in as_completed(futures):
                response = future.result()
                responses.append(response)

        self.conversation.add(role=self.agent_name, content=responses)

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

        # Aggregation agent
        # final_answer = self.aggregation_agent(responses)

        final_answer = aggregation_agent(responses)

        self.conversation.add(
            role="Majority Voting Agent", content=final_answer
        )

        return history_output_formatter(
            self.conversation, self.output_type
        )

    def aggregate(self, responses: List[str]) -> str:
        """
        Aggregates a list of responses into a single final answer.

        Here we use a simple majority vote (most common answer) as an example. Depending on
        the task, you might need a more sophisticated aggregation (e.g., weighting, consensus reasoning, etc.).

        Args:
            responses (list of str): The list of responses.

        Returns:
            str: The aggregated answer.
        """
        # Count the frequency of each response.
        counts = Counter(responses)
        most_common, freq = counts.most_common(1)[0]
        logger.info(
            f"Aggregation complete. Most common response (appeared {freq} times):"
        )
        return most_common

    def check_responses_for_answer(
        self, responses: List[str], answer: str
    ) -> bool:
        """
        Checks if the specified answer is present in any of the provided responses.

        Args:
            responses (List[str]): A list of responses to check.
            answer (str): The answer to look for in the responses.

        Returns:
            bool: True if the answer is found in any response, False otherwise.
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
    ) -> List[str]:
        """
        Runs the agent in a batched manner.
        """
        responses = []
        for task in tasks:
            response = self.run(task, *args, **kwargs)
            responses.append(response)
        return responses


# # Example usage:
# if __name__ == "__main__":
#     agent = SelfConsistencyAgent(
#         agent_name="Reasoning-Agent",
#         model_name="gpt-4o-mini",
#         max_loops=1,
#         num_samples=5,  # Number of samples for self consistency
#     )

#     prompt = "What is the 40th prime number?"
#     final_answer = agent.run(prompt)
#     print("\nFinal aggregated answer:")
#     print(final_answer)
