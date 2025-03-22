from typing import List, Dict, Any, Union
import time

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation

from loguru import logger


class KnowledgeGenerator:
    """
    A component that generates relevant knowledge for a given input query.

    The knowledge generator creates detailed contextual information that can be used
    to enhance the reasoning capabilities of the main agent when responding to queries.

    Attributes:
        agent_name (str): Name of the knowledge generator agent
        model_name (str): Model to use for knowledge generation
        num_knowledge_items (int): Number of knowledge items to generate per query
    """

    def __init__(
        self,
        agent_name: str = "knowledge-generator",
        model_name: str = "openai/o1",
        num_knowledge_items: int = 2,
    ) -> None:
        """
        Initialize the knowledge generator component.

        Args:
            agent_name (str): Name identifier for the knowledge generator agent
            model_name (str): LLM model to use for knowledge generation
            num_knowledge_items (int): Number of knowledge snippets to generate for each query
        """
        self.agent_name = agent_name
        self.model_name = model_name
        self.num_knowledge_items = num_knowledge_items

        # Create the knowledge generator agent
        knowledge_system_prompt = (
            self._create_knowledge_system_prompt()
        )
        self.agent = Agent(
            agent_name=agent_name,
            agent_description="Generates factual, relevant knowledge to assist with answering queries",
            system_prompt=knowledge_system_prompt,
            model_name=model_name,
            max_loops=1,
        )

        logger.info(
            f"Initialized {self.agent_name} with model {self.model_name}"
        )

    def _create_knowledge_system_prompt(self) -> str:
        """
        Create the system prompt for the knowledge generator.

        Returns:
            str: System prompt with examples and instructions
        """
        examples_text = ""

        system_prompt = f"""You are a specialized knowledge generator that provides factually accurate, detailed information relevant to a given input query. Your role is to generate precise knowledge that can help answer the query correctly.

        When provided with an input query, generate {self.num_knowledge_items} separate, independent knowledge statements that are directly relevant to the query and provide context that would help answer it accurately.

        Each knowledge statement should be:
        1. Factually accurate and verifiable
        2. Detailed and specific (not general statements)
        3. Directly relevant to addressing the query
        4. Neutral and objective, providing context rather than opinions
        5. Independent from other knowledge statements (provide different perspectives)

        Here are examples of good knowledge generation:

        {examples_text}

        For each input, provide knowledge statements formatted as:
        "Knowledge 1: [factual, detailed information relevant to the query]"
        "Knowledge 2: [alternative factual, detailed information relevant to the query]"
        etc.

        Focus on providing knowledge that would help someone arrive at the correct answer to the query, particularly for questions that require commonsense reasoning or factual information.
        """

        return system_prompt

    def generate_knowledge(self, query: str) -> List[str]:
        """
        Generate relevant knowledge for the input query.

        Args:
            query (str): The input query to generate knowledge for

        Returns:
            List[str]: List of generated knowledge statements
        """
        prompt = f"Input: {query}\nKnowledge:"

        logger.debug(f"Generating knowledge for query: {query}")
        start_time = time.time()

        response = self.agent.run(task=prompt)

        end_time = time.time()
        logger.debug(
            f"Knowledge generation completed in {end_time - start_time:.2f}s"
        )

        # Parse the generated knowledge into separate statements
        knowledge_items = []

        # Handle different response formats
        if "Knowledge 1:" in response:
            # Extract numbered knowledge items
            for i in range(1, self.num_knowledge_items + 1):
                marker = f"Knowledge {i}:"
                next_marker = (
                    f"Knowledge {i+1}:"
                    if i < self.num_knowledge_items
                    else None
                )

                if marker in response:
                    start_idx = response.find(marker) + len(marker)
                    end_idx = (
                        response.find(next_marker)
                        if next_marker and next_marker in response
                        else None
                    )

                    knowledge = (
                        response[start_idx:end_idx].strip()
                        if end_idx
                        else response[start_idx:].strip()
                    )
                    knowledge_items.append(knowledge)
        else:
            # If not properly formatted with numbers, split by paragraphs
            paragraphs = [
                p.strip() for p in response.split("\n\n") if p.strip()
            ]
            for p in paragraphs[: self.num_knowledge_items]:
                if p.startswith("Knowledge:"):
                    p = p[len("Knowledge:") :].strip()
                knowledge_items.append(p)

        # Ensure we have the requested number of knowledge items
        while len(knowledge_items) < self.num_knowledge_items:
            logger.warning(
                f"Only generated {len(knowledge_items)} knowledge items, expected {self.num_knowledge_items}"
            )
            knowledge_items.append(
                ""
            )  # Add empty string as placeholder

        # Truncate if we have too many
        knowledge_items = knowledge_items[: self.num_knowledge_items]

        logger.info(
            f"Generated {len(knowledge_items)} knowledge items"
        )
        return knowledge_items


class Reasoner:
    """
    Component that uses generated knowledge to reason about and answer queries.

    This reasoner takes knowledge generated by the KnowledgeGenerator and uses it
    to make more informed decisions when answering questions.

    Attributes:
        agent_name (str): Name of the reasoner agent
        model_name (str): Model to use for reasoning
    """

    def __init__(
        self,
        agent_name: str = "knowledge-reasoner",
        model_name: str = "openai/o1",
    ) -> None:
        """
        Initialize the reasoner component.

        Args:
            agent_name (str): Name identifier for the reasoner agent
            model_name (str): LLM model to use for reasoning
        """
        self.agent_name = agent_name
        self.model_name = model_name

        # Create the reasoning agent
        reasoning_system_prompt = (
            self._create_reasoning_system_prompt()
        )
        self.agent = Agent(
            agent_name=agent_name,
            agent_description="Reasons about queries using provided knowledge to generate accurate answers",
            system_prompt=reasoning_system_prompt,
            model_name=model_name,
            max_loops=1,
        )

        logger.info(
            f"Initialized {self.agent_name} with model {self.model_name}"
        )

    def _create_reasoning_system_prompt(self) -> str:
        """
        Create the system prompt for the reasoner.

        Returns:
            str: System prompt with instructions
        """
        system_prompt = """
        You are a specialized reasoning agent that answers questions based on provided knowledge. Your role is to carefully analyze the given knowledge and use it to answer the question accurately.

        For each question:
        1. Carefully read the provided knowledge
        2. Analyze how the knowledge relates to the question
        3. Use the knowledge to form a well-reasoned answer
        4. Provide your answer along with an explanation of your reasoning
        5. Include a confidence assessment (very high, high, medium, low, very low)

        Your response should follow this format:
        "Explanation: [Your detailed reasoning based on the knowledge]
        Confidence: [Your confidence level]
        Answer: [Your final answer]"

        Be objective and precise. If the knowledge contradicts itself or is insufficient to answer the question, acknowledge this in your response and provide your best judgment given the available information.

        Focus on using the provided knowledge rather than your pre-existing information, though you may use your general understanding to interpret the knowledge appropriately.
    """

        return system_prompt

    def reason_and_answer(
        self, query: str, knowledge: str
    ) -> Dict[str, str]:
        """
        Reason about the query using the provided knowledge and generate an answer.

        Args:
            query (str): The input query to answer
            knowledge (str): Knowledge to use for reasoning

        Returns:
            Dict[str, str]: Dictionary containing explanation, confidence and answer
        """
        # Format the prompt
        prompt = f"Question: {query}\nKnowledge: {knowledge}\nExplain and Answer:"

        logger.debug(f"Reasoning about query: {query}")
        start_time = time.time()

        response = self.agent.run(task=prompt)

        end_time = time.time()
        logger.debug(
            f"Reasoning completed in {end_time - start_time:.2f}s"
        )

        # Parse the response
        result = {"explanation": "", "confidence": "", "answer": ""}

        if "Explanation:" in response and "Answer:" in response:
            # Get explanation
            explanation_start = response.find("Explanation:") + len(
                "Explanation:"
            )

            # Find the end of explanation (which is either Confidence: or Answer:)
            confidence_pos = response.find("Confidence:")
            answer_pos = response.find("Answer:")

            explanation_end = min(
                pos for pos in [confidence_pos, answer_pos] if pos > 0
            )
            result["explanation"] = response[
                explanation_start:explanation_end
            ].strip()

            # Get confidence if present
            if confidence_pos > 0:
                confidence_start = confidence_pos + len("Confidence:")
                confidence_end = (
                    answer_pos
                    if answer_pos > confidence_pos
                    else len(response)
                )
                result["confidence"] = response[
                    confidence_start:confidence_end
                ].strip()

            # Get answer
            if answer_pos > 0:
                answer_start = answer_pos + len("Answer:")
                result["answer"] = response[answer_start:].strip()
        else:
            # Fallback parsing if not properly formatted
            result["answer"] = response.strip()

        return result


class GKPAgent:
    """
    Generated Knowledge Prompting (GKP) Agent that enhances reasoning by generating
    relevant knowledge before answering queries.

    This agent implements the approach described in Liu et al. 2022, generating knowledge
    to improve performance on tasks requiring commonsense reasoning and factual information.

    Attributes:
        agent_name (str): Name of the GKP agent
        model_name (str): Model to use for all components
        num_knowledge_items (int): Number of knowledge items to generate per query
        knowledge_generator (KnowledgeGenerator): Component for generating knowledge
        reasoner (Reasoner): Component for reasoning using the generated knowledge
        conversation (Conversation): Conversation history manager
    """

    def __init__(
        self,
        agent_name: str = "gkp-agent",
        model_name: str = "openai/o1",
        num_knowledge_items: int = 6,
    ) -> None:
        """
        Initialize the GKP Agent with its components.

        Args:
            agent_name (str): Name identifier for the agent
            model_name (str): LLM model to use for all components
            num_knowledge_items (int): Number of knowledge snippets to generate for each query
        """
        self.agent_name = agent_name
        self.model_name = model_name
        self.num_knowledge_items = num_knowledge_items
        self.conversation = Conversation(time_enabled=True)

        # Initialize components
        self.knowledge_generator = KnowledgeGenerator(
            agent_name=f"{agent_name}-knowledge-generator",
            model_name=model_name,
            num_knowledge_items=num_knowledge_items,
        )

        self.reasoner = Reasoner(
            agent_name=f"{agent_name}-reasoner",
            model_name=model_name,
        )

        # Create the final response coordinator agent
        coordinator_system_prompt = (
            self._create_coordinator_system_prompt()
        )
        self.coordinator = Agent(
            agent_name=f"{agent_name}-coordinator",
            agent_description="Coordinates multiple reasoning paths to provide the best final answer",
            system_prompt=coordinator_system_prompt,
            model_name=model_name,
            max_loops=1,
        )

        logger.info(
            f"Initialized {self.agent_name} with model {self.model_name}"
        )

    def _create_coordinator_system_prompt(self) -> str:
        """
        Create the system prompt for the response coordinator.

        Returns:
            str: System prompt with instructions
        """
        system_prompt = """
        You are a specialized coordination agent that analyzes multiple reasoning paths and answers to determine the most accurate final response.

        For each query, you will receive:
        1. The original question
        2. Multiple reasoning paths, each with:
        - Generated knowledge used for reasoning
        - An explanation of the reasoning process
        - A confidence assessment
        - An answer derived from that reasoning path

        Your task is to:
        1. Analyze all reasoning paths
        2. Determine which path(s) have the most accurate and reliable reasoning
        3. Assess the confidence levels provided
        4. Resolve any contradictions between different answers
        5. Provide a final, definitive answer that represents the most accurate conclusion

        Structure your response as follows:
        "Analysis: [Brief analysis of the different reasoning paths]
        Final Answer: [Clear, definitive answer to the original question]
        Explanation: [Explanation supporting your final answer, drawing from the best elements of the reasoning paths]"

        Be objective and precise. Your goal is to determine the most accurate answer based on the quality of reasoning and knowledge provided in each path.
        """

        return system_prompt

    def process(self, query: str) -> Dict[str, Any]:
        """
        Process a query using the GKP approach.

        Args:
            query (str): The query to process

        Returns:
            Dict[str, Any]: Dictionary containing the full processing results
        """
        start_time = time.time()
        logger.info(f"Processing query: {query}")

        # 1. Generate knowledge
        knowledge_items = self.knowledge_generator.generate_knowledge(
            query
        )

        # 2. Use each knowledge item to reason about the query
        reasoning_results = []
        for i, knowledge in enumerate(knowledge_items):
            logger.debug(f"Reasoning with knowledge item {i+1}")
            reasoning_result = self.reasoner.reason_and_answer(
                query, knowledge
            )
            reasoning_result["knowledge"] = knowledge
            reasoning_results.append(reasoning_result)

        # 3. Coordinate the different reasoning paths to produce final answer
        final_answer = self._coordinate_answers(
            query, reasoning_results
        )

        # 4. Record in conversation history
        self.conversation.add("user", query)
        self.conversation.add("assistant", final_answer["response"])

        end_time = time.time()
        process_time = end_time - start_time
        logger.info(f"Query processed in {process_time:.2f}s")

        # Return complete results
        return {
            "query": query,
            "knowledge_items": knowledge_items,
            "reasoning_results": reasoning_results,
            "final_answer": final_answer,
            "process_time": process_time,
        }

    def _coordinate_answers(
        self, query: str, reasoning_results: List[Dict[str, str]]
    ) -> Dict[str, str]:
        """
        Coordinate multiple reasoning paths to produce the final answer.

        Args:
            query (str): The original query
            reasoning_results (List[Dict[str, str]]): Results from multiple reasoning paths

        Returns:
            Dict[str, str]: The final coordinated answer
        """
        # Format the prompt for the coordinator
        prompt_parts = [f"Question: {query}\n"]

        for i, result in enumerate(reasoning_results):
            prompt_parts.append(f"Reasoning Path {i+1}:")
            prompt_parts.append(f"Knowledge: {result['knowledge']}")
            prompt_parts.append(
                f"Explanation: {result['explanation']}"
            )
            prompt_parts.append(f"Confidence: {result['confidence']}")
            prompt_parts.append(f"Answer: {result['answer']}\n")

        prompt_parts.append(
            "Based on these reasoning paths, provide your final answer."
        )
        prompt = "\n".join(prompt_parts)

        logger.debug("Coordinating multiple reasoning paths")
        response = self.coordinator.run(task=prompt)

        # Parse the coordinated response
        result = {"analysis": "", "response": "", "explanation": ""}

        if "Analysis:" in response and "Final Answer:" in response:
            # Extract analysis
            analysis_start = response.find("Analysis:") + len(
                "Analysis:"
            )
            analysis_end = response.find("Final Answer:")
            result["analysis"] = response[
                analysis_start:analysis_end
            ].strip()

            # Extract final answer
            answer_start = response.find("Final Answer:") + len(
                "Final Answer:"
            )

            if "Explanation:" in response:
                answer_end = response.find("Explanation:")
                explanation_start = answer_end + len("Explanation:")

                result["response"] = response[
                    answer_start:answer_end
                ].strip()
                result["explanation"] = response[
                    explanation_start:
                ].strip()
            else:
                result["response"] = response[answer_start:].strip()
        else:
            # Fallback if not properly formatted
            result["response"] = response.strip()

        return result

    def run(
        self, queries: List[str], detailed_output: bool = False
    ) -> Union[List[str], List[Dict[str, Any]]]:
        """
        Run the GKP agent on a list of queries.

        Args:
            queries (List[str]): List of queries to process
            detailed_output (bool): Whether to return detailed processing results

        Returns:
            Union[List[str], List[Dict[str, Any]]]: List of answers or detailed results
        """
        results = []

        for i, query in enumerate(queries):
            logger.info(f"Processing query {i+1}/{len(queries)}")
            process_result = self.process(query)

            if detailed_output:
                results.append(process_result)
            else:
                results.append(
                    process_result["final_answer"]["response"]
                )

        return results


# # Example usage
# if __name__ == "__main__":
#     # Initialize the GKP Agent
#     agent = GKPAgent(
#         agent_name="gkp-agent",
#         model_name="gpt-4o-mini",  # Using OpenAI's model
#         num_knowledge_items=10,  # Generate 2 knowledge items per query
#     )

#     # Example queries
#     queries = [
#         "Create an entirely new construct of mathematics unifying physics and traditional physics never seen",
#     ]

#     # Run the agent
#     results = agent.run(queries)

#     print(results)

#     # Print results
#     for i, result in enumerate(results):
#         print(f"\n\nQUERY {i+1}:")
#         print(f"{queries[i]}\n")
#         print("FINAL ANSWER:")
#         print(f"{result}")
