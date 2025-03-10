"""
Iterative Reflective Expansion (IRE) Algorithm

A sophisticated reasoning framework that employs iterative hypothesis generation, simulation, and refinement to solve complex problems. IRE leverages a multi-step approach where an AI agent generates initial solution paths, evaluates their effectiveness through simulation, reflects on errors, and dynamically revises reasoning strategies. Through continuous cycles of hypothesis testing and meta-cognitive reflection, the algorithm progressively converges on optimal solutions by learning from both successful and unsuccessful reasoning attempts.


- IRE is a multi-step approach where an AI agent generates initial solution paths, evaluates their effectiveness through simulation, reflects on errors, and dynamically revises reasoning strategies.
- Through continuous cycles of hypothesis testing and meta-cognitive reflection, the algorithm progressively converges on optimal solutions by learning from both successful and unsuccessful reasoning attempts.


Workflow:
1. Generate initial hypotheses
2. Simulate paths
3. Reflect on errors
4. Revise paths
5. Select promising paths
6. Synthesize solution

"""

from typing import List, Tuple
from loguru import logger
from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation
from swarms.structs.output_types import OutputType
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)

# Define a new system prompt for general problem solving
GENERAL_REASONING_AGENT_SYS_PROMPT = """
You are a highly capable problem-solving agent with a unique ability to reason through complex challenges via iterative reflection and hypothesis testing.
Your role is to assist in generating innovative solutions to a wide array of general problems by engaging in trial and error, reflective evaluation, and dynamic hypothesis expansion.
When presented with a problem statement, generate multiple hypotheses, simulate reasoning paths, reflect on errors, and iteratively refine your approach to produce the best solution.
Do not include any finance-related content.

"""


class IterativeReflectiveExpansion:
    """
    A class implementing the Iterative Reflective Expansion (IRE) reasoning algorithm.

    This algorithm leverages a Swarms agent to iteratively generate, simulate, reflect on, and refine reasoning paths
    in order to solve complex problems through trial and error, reflective evaluation, and dynamic hypothesis expansion.
    """

    def __init__(
        self,
        agent_name: str = "General-Reasoning-Agent",
        description: str = "A reasoning agent that can answer questions and help with tasks.",
        agent: Agent = None,
        max_iterations: int = 5,
        system_prompt: str = GENERAL_REASONING_AGENT_SYS_PROMPT,
        model_name: str = "gpt-4o-mini",
        output_type: OutputType = "dict",
    ) -> None:
        """
        Initialize the Iterative Reflective Expansion engine.

        :param agent: The Swarms agent instance used to perform reasoning tasks.
        :param max_iterations: Maximum number of iterations for the reasoning process.
        """
        self.agent_name = agent_name
        self.description = description
        self.agent = agent
        self.max_iterations = max_iterations
        self.output_type = output_type
        self.system_prompt = system_prompt
        self.conversation = Conversation()

        self.agent = Agent(
            agent_name=self.agent_name,
            system_prompt=self.system_prompt,
            model_name=model_name,
            max_loops=1,
            dynamic_temperature_enabled=True,
        )

    def generate_initial_hypotheses(self, task: str) -> List[str]:
        """
        Generate an initial set of reasoning hypotheses based on the problem input.

        :param task: The problem statement.
        :return: A list of candidate reasoning paths/hypotheses.
        """
        logger.info("Generating initial hypotheses for the problem.")
        prompt = (
            f"Given the following problem:\n\n"
            f"'{task}'\n\n"
            "Generate a list of possible approaches and strategies to solve it. "
            "Present each approach on a new line."
        )
        response = self.agent.run(prompt)
        self.conversation.add(
            role=self.agent.agent_name, content=response
        )
        hypotheses = [
            line.strip()
            for line in response.split("\n")
            if line.strip()
        ]
        logger.debug(f"Initial hypotheses: {hypotheses}")
        return hypotheses

    def simulate_path(self, path: str) -> Tuple[str, float, str]:
        """
        Simulate a given reasoning path and evaluate its effectiveness.

        :param path: A candidate reasoning path.
        :return: A tuple containing the simulated outcome, a numerical score (0.0 to 1.0), and error information.
        """
        logger.info(f"Simulating path: {path}")
        prompt = (
            f"Simulate the following reasoning path step by step and provide:\n"
            f"1. Outcome: A brief summary of the resulting solution.\n"
            f"2. Score: A numerical effectiveness score between 0.0 and 1.0.\n"
            f"3. Errors: Any potential errors or shortcomings identified during the reasoning.\n\n"
            f"Reasoning Path: {path}"
        )
        response = self.agent.run(prompt)
        self.conversation.add(
            role=self.agent.agent_name, content=response
        )
        outcome = ""
        score = 0.0
        error_info = ""
        try:
            # Expecting a response with lines starting with "Outcome:", "Score:", and "Errors:"
            for line in response.splitlines():
                if line.startswith("Outcome:"):
                    outcome = line[len("Outcome:") :].strip()
                elif line.startswith("Score:"):
                    score = float(line[len("Score:") :].strip())
                elif line.startswith("Errors:"):
                    error_info = line[len("Errors:") :].strip()
        except Exception as e:
            logger.error(f"Error parsing simulation response: {e}")
        logger.debug(
            f"Simulated outcome: {outcome}, Score: {score}, Errors: {error_info}"
        )
        return outcome, score, error_info

    def meta_reflect(self, error_info: str) -> str:
        """
        Perform meta-cognitive reflection on the provided error information.

        :param error_info: Information regarding errors in the reasoning path.
        :return: Feedback and suggestions for revising the reasoning path.
        """
        logger.info(
            "Performing meta-reflection on error information."
        )
        prompt = (
            f"Analyze the following error information and suggest modifications to improve the reasoning process:\n"
            f"{error_info}\n"
            "Provide clear and actionable feedback."
        )
        feedback = self.agent.run(prompt)
        self.conversation.add(
            role=self.agent.agent_name, content=feedback
        )
        logger.debug(f"Meta-reflection feedback: {feedback}")
        return feedback

    def revise_path(self, path: str, feedback: str) -> List[str]:
        """
        Revise the reasoning path based on the provided feedback.

        :param path: The original reasoning path.
        :param feedback: Feedback from meta-cognitive reflection.
        :return: A list of revised reasoning paths.
        """
        logger.info("Revising reasoning path based on feedback.")
        prompt = (
            f"Given the reasoning path:\n'{path}'\n\n"
            f"and the following feedback:\n'{feedback}'\n\n"
            "Generate revised reasoning paths that address the issues raised. "
            "Present each revised path on a new line."
        )
        response = self.agent.run(prompt)
        self.conversation.add(
            role=self.agent.agent_name, content=response
        )
        revised_paths = [
            line.strip()
            for line in response.split("\n")
            if line.strip()
        ]
        logger.debug(f"Revised paths: {revised_paths}")
        return revised_paths

    def select_promising_paths(self, paths: List[str]) -> List[str]:
        """
        Select the most promising reasoning paths from a list of candidates.

        :param paths: A list of candidate reasoning paths.
        :return: A pruned list containing the most promising paths.
        """
        logger.info("Selecting promising reasoning paths.")
        prompt = (
            "Evaluate the following reasoning paths and select the ones that appear most promising for further exploration. "
            "List each selected path on a new line:\n"
            + "\n".join(paths)
        )
        response = self.agent.run(prompt)
        self.conversation.add(
            role=self.agent.agent_name, content=response
        )
        selected_paths = [
            line.strip()
            for line in response.split("\n")
            if line.strip()
        ]
        logger.debug(f"Selected paths: {selected_paths}")
        return selected_paths

    def synthesize_solution(
        self, paths: List[str], memory_pool: List[str]
    ) -> str:
        """
        Synthesize a final solution from the promising reasoning paths and historical memory.

        :param paths: The current promising reasoning paths.
        :param memory_pool: A list of all previously generated reasoning paths.
        :return: A coherent final solution.
        """
        logger.info(
            "Synthesizing final solution from promising paths."
        )
        prompt = (
            "Based on the following promising reasoning paths:\n"
            f"{chr(10).join(paths)}\n\n"
            "and the historical reasoning memory:\n"
            f"{chr(10).join(memory_pool)}\n\n"
            "Synthesize a final, coherent solution to the problem."
        )
        solution = self.agent.run(prompt)
        self.conversation.add(
            role=self.agent.agent_name, content=solution
        )
        logger.debug(f"Synthesized solution: {solution}")
        return solution

    def run(self, task: str) -> str:
        """
        Execute the Iterative Reflective Expansion process on the provided problem.

        :param task: The problem statement.
        :return: The final solution generated after iterative reasoning.
        """
        logger.info(
            f"Starting iterative reflective expansion for problem: {task}"
        )
        candidate_paths = self.generate_initial_hypotheses(task)
        memory_pool: List[str] = []

        for iteration in range(self.max_iterations):
            logger.info(
                f"Iteration {iteration + 1}/{self.max_iterations}"
            )
            expanded_paths: List[str] = []

            for path in candidate_paths:
                outcome, score, error_info = self.simulate_path(path)
                # Use a threshold score of 0.7 (this can be adjusted)
                if score < 0.7:
                    feedback = self.meta_reflect(error_info)
                    revised_paths = self.revise_path(path, feedback)
                    expanded_paths.extend(revised_paths)
                else:
                    expanded_paths.append(path)

            memory_pool.extend(candidate_paths)
            candidate_paths = self.select_promising_paths(
                expanded_paths
            )
            logger.info(
                f"Candidate paths for next iteration: {candidate_paths}"
            )

        self.synthesize_solution(candidate_paths, memory_pool)
        logger.info("Final solution generated.")

        return history_output_formatter(
            self.conversation, self.output_type
        )


# def main() -> None:
#     """
#     Main function to execute the Iterative Reflective Expansion algorithm on a sample problem.
#     """
#     problem_statement = "What is the 40th prime number?"
#     reasoning_engine = IterativeReflectiveExpansion(max_iterations=1)
#     final_solution = reasoning_engine.run(problem_statement)
#     print("Final Solution:")
#     print(final_solution)


# if __name__ == "__main__":
#     main()
