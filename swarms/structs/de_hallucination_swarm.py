from typing import List, Dict, Any, Optional
import time
from loguru import logger
from swarms.structs.agent import Agent

# Prompt templates for different agent roles
GENERATOR_PROMPT = """
You are a knowledgeable assistant tasked with providing accurate information on a wide range of topics.

Your responsibilities:
1. Provide accurate information based on your training data
2. Use clear, concise language
3. Acknowledge limitations in your knowledge
4. Abstain from making up information when uncertain

When responding to queries:
- Stick to verified facts
- Cite your sources when possible
- Clearly distinguish between firmly established facts and more tentative claims
- Use phrases like "I'm not certain about..." or "Based on my knowledge up to my training cutoff..." when appropriate
- Avoid overly confident language for uncertain topics

Remember, it's better to acknowledge ignorance than to provide incorrect information.
"""

CRITIC_PROMPT = """
You are a critical reviewer tasked with identifying potential inaccuracies, hallucinations, or unsupported claims in AI-generated text.

Your responsibilities:
1. Carefully analyze the provided text for factual errors
2. Identify claims that lack sufficient evidence
3. Spot logical inconsistencies
4. Flag overly confident language on uncertain topics
5. Detect potentially hallucinated details (names, dates, statistics, etc.)

For each issue detected, you should:
- Quote the specific problematic text
- Explain why it's potentially inaccurate
- Rate the severity of the issue (low/medium/high)
- Suggest a specific correction or improvement

Focus particularly on:
- Unfounded claims presented as facts
- Highly specific details that seem suspicious
- Logical contradictions
- Anachronisms or temporal inconsistencies
- Claims that contradict common knowledge

Be thorough and specific in your critique. Provide actionable feedback for improvement.
"""

REFINER_PROMPT = """
You are a refinement specialist tasked with improving text based on critical feedback.

Your responsibilities:
1. Carefully review the original text and the critical feedback
2. Make precise modifications to address all identified issues
3. Ensure factual accuracy in the refined version
4. Maintain the intended tone and style of the original
5. Add appropriate epistemic status markers (e.g., "likely", "possibly", "according to...")

Guidelines for refinement:
- Remove or qualify unsupported claims
- Replace specific details with more general statements when evidence is lacking
- Add appropriate hedging language where certainty is not warranted
- Maintain the helpful intent of the original response
- Ensure logical consistency throughout the refined text
- Add qualifiers or clarify knowledge limitations where appropriate

The refined text should be helpful and informative while being scrupulously accurate.
"""

VALIDATOR_PROMPT = """
You are a validation expert tasked with ensuring the highest standards of accuracy in refined AI outputs.

Your responsibilities:
1. Verify that all critical issues from previous feedback have been properly addressed
2. Check for any remaining factual inaccuracies or unsupported claims
3. Ensure appropriate epistemic status markers are used
4. Confirm the response maintains a helpful tone while being accurate
5. Provide a final assessment of the response quality

Assessment structure:
- Issue resolution: Have all previously identified issues been addressed? (Yes/No/Partially)
- Remaining concerns: Are there any remaining factual or logical issues? (List if any)
- Epistemics: Does the response appropriately indicate confidence levels? (Yes/No/Needs improvement)
- Helpfulness: Does the response remain helpful despite necessary qualifications? (Yes/No/Partially)
- Overall assessment: Final verdict on whether the response is ready for user consumption (Approved/Needs further refinement)

If approved, explain what makes this response trustworthy. If further refinement is needed, provide specific guidance.
"""


class DeHallucinationSwarm:
    """
    A system of multiple agents that work together to reduce hallucinations in generated content.
    The system works through multiple rounds of generation, criticism, refinement, and validation.
    """

    def __init__(
        self,
        name: str = "DeHallucinationSwarm",
        description: str = "A system of multiple agents that work together to reduce hallucinations in generated content.",
        model_names: List[str] = [
            "gpt-4o-mini",
            "gpt-4o-mini",
            "gpt-4o-mini",
            "gpt-4o-mini",
        ],
        iterations: int = 2,
        system_prompt: str = GENERATOR_PROMPT,
        store_intermediate_results: bool = True,
    ):
        """
        Initialize the DeHallucinationSwarm with configurable agents.

        Args:
            model_names: List of model names for generator, critic, refiner, and validator
            iterations: Number of criticism-refinement cycles to perform
            store_intermediate_results: Whether to store all intermediate outputs
        """
        self.name = name
        self.description = description
        self.iterations = iterations
        self.store_intermediate_results = store_intermediate_results
        self.system_prompt = system_prompt
        self.history = []

        # Initialize all agents
        self.generator = Agent(
            agent_name="Generator",
            description="An agent that generates initial responses to queries",
            system_prompt=GENERATOR_PROMPT,
            model_name=model_names[0],
        )

        self.critic = Agent(
            agent_name="Critic",
            description="An agent that critiques responses for potential inaccuracies",
            system_prompt=CRITIC_PROMPT,
            model_name=model_names[1],
        )

        self.refiner = Agent(
            agent_name="Refiner",
            description="An agent that refines responses based on critique",
            system_prompt=REFINER_PROMPT,
            model_name=model_names[2],
        )

        self.validator = Agent(
            agent_name="Validator",
            description="An agent that performs final validation of refined content",
            system_prompt=VALIDATOR_PROMPT,
            model_name=model_names[3],
        )

    def _log_step(
        self,
        step_name: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Record a step in the swarm's processing history"""
        if self.store_intermediate_results:
            timestamp = time.time()
            step_record = {
                "timestamp": timestamp,
                "step": step_name,
                "content": content,
            }
            if metadata:
                step_record["metadata"] = metadata

            self.history.append(step_record)
            logger.debug(f"Logged step: {step_name}")

    def run(self, query: str) -> Dict[str, Any]:
        """
        Process a query through the swarm's multi-agent refinement cycle.

        Args:
            query: The user's query to process

        Returns:
            Dict containing the final response and processing metadata
        """
        logger.info(f"Processing query: {query}")
        self.history = []  # Reset history for new query

        # Generate initial response
        initial_response = self.generator.run(query)
        self._log_step(
            "initial_generation", initial_response, {"query": query}
        )

        current_response = initial_response

        # Perform multiple iteration cycles
        for i in range(self.iterations):
            logger.info(f"Starting iteration {i+1}/{self.iterations}")

            # Step 1: Critique the current response
            critique = self.critic.run(
                f"Review the following response to the query: '{query}'\n\n{current_response}"
            )
            self._log_step(f"critique_{i+1}", critique)

            # Step 2: Refine based on critique
            refined_response = self.refiner.run(
                f"Refine the following response based on the critique provided.\n\n"
                f"Original query: {query}\n\n"
                f"Original response: {current_response}\n\n"
                f"Critique: {critique}"
            )
            self._log_step(f"refinement_{i+1}", refined_response)

            # Update current response for next iteration
            current_response = refined_response

        # Final validation
        validation = self.validator.run(
            f"Validate the following refined response for accuracy and helpfulness.\n\n"
            f"Original query: {query}\n\n"
            f"Final response: {current_response}"
        )
        self._log_step("final_validation", validation)

        # Prepare results
        result = {
            "query": query,
            "final_response": current_response,
            "validation_result": validation,
            "iteration_count": self.iterations,
        }

        if self.store_intermediate_results:
            result["processing_history"] = self.history

        return result

    def batch_run(self, queries: List[str]) -> List[Dict[str, Any]]:
        """
        Process multiple queries through the swarm.

        Args:
            queries: List of user queries to process

        Returns:
            List of result dictionaries, one per query
        """
        results = []
        for query in queries:
            logger.info(f"Processing batch query: {query}")
            results.append(self.run(query))
        return results


# # Example usage
# if __name__ == "__main__":
#     # Configure logger
#     logger.add("dehallucinationswarm.log", rotation="10 MB")

#     # Create swarm instance
#     swarm = DeHallucinationSwarm(iterations=2)

#     # Example queries that might tempt hallucination
#     test_queries = [
#         "Tell me about the history of quantum computing",
#         "What are the specific details of the Treaty of Utrecht?",
#         "Who won the Nobel Prize in Physics in 2020?",
#         "What are the main causes of the economic recession of 2008?",
#     ]

#     # Process batch of queries
#     results = swarm.batch_run(test_queries)
#     print(results)
