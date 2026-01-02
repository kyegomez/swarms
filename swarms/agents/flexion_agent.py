from typing import List, Dict, Any, Tuple
import re
import time
from datetime import datetime

from swarms.structs.agent import Agent
from swarms.structs.conversation import Conversation

from loguru import logger

# Configuration constants for ReflexionAgent
EARLY_TERMINATION_THRESHOLD = 0.8  # Lower than 0.9 for more realistic termination
DEFAULT_SCORE = 0.6  # Higher than 0.5 to increase chance of early termination
SCORE_IMPROVEMENT_THRESHOLD = 0.05  # Minimum improvement to continue iterating

# Define Reflexion prompt with detailed instructions
REFLEXION_PROMPT = """
You are Reflexion, an advanced AI assistant designed to generate high-quality responses and continuously improve through self-reflection.

CAPABILITIES:
- Deep reasoning: Break down complex problems step-by-step
- Self-evaluation: Critically assess your own responses
- Self-reflection: Generate insights about your performance and areas for improvement
- Memory utilization: Learn from past experiences and build upon previous knowledge

PROCESS:
1. UNDERSTAND the user's query thoroughly
2. GENERATE a detailed, thoughtful response
3. EVALUATE your response against these criteria:
   - Accuracy: Is all information factually correct?
   - Completeness: Does it address all aspects of the query?
   - Clarity: Is it well-structured and easy to understand?
   - Relevance: Does it focus on what the user needs?
   - Actionability: Does it provide practical, implementable solutions?
4. REFLECT on your performance and identify improvements
5. REFINE your response based on self-reflection

KEY PRINCIPLES:
- Be thorough but concise
- Prioritize practical, actionable advice
- Maintain awareness of your limitations
- Be transparent about uncertainty
- Learn continuously from each interaction

Always maintain your role as a helpful assistant focused on providing valuable information and solutions.
"""


class ReflexionMemory:
    """
    A memory system for the Reflexion agent to store past experiences, reflections, and feedback.

    Attributes:
        short_term_memory (List[Dict]): Recent interactions and their evaluations
        long_term_memory (List[Dict]): Persistent storage of important reflections and patterns
        memory_capacity (int): Maximum number of entries in long-term memory
    """

    def __init__(self, memory_capacity: int = 100):
        """
        Initialize the memory system.

        Args:
            memory_capacity (int): Maximum number of entries in long-term memory
        """
        self.short_term_memory = []
        self.long_term_memory = []
        self.memory_capacity = memory_capacity

    def add_short_term_memory(self, entry: Dict[str, Any]) -> None:
        """
        Add an entry to short-term memory.

        Args:
            entry (Dict[str, Any]): Memory entry containing task, response, evaluation, etc.
        """
        # Add timestamp to track when memories were created
        entry["timestamp"] = datetime.now().isoformat()
        self.short_term_memory.append(entry)

        # Keep only the most recent 10 entries in short-term memory
        if len(self.short_term_memory) > 10:
            self.short_term_memory.pop(0)

    def add_long_term_memory(self, entry: Dict[str, Any]) -> None:
        """
        Add an important entry to long-term memory.

        Args:
            entry (Dict[str, Any]): Memory entry containing task, response, evaluation, etc.
        """
        entry["timestamp"] = datetime.now().isoformat()

        # Check if similar entry exists to avoid duplication
        for existing in self.long_term_memory:
            if (
                self._similarity(existing, entry) > 0.8
            ):  # Hypothetical similarity threshold
                logger.debug(
                    "Similar entry already exists in long-term memory"
                )
                return

        self.long_term_memory.append(entry)

        # If exceeded capacity, remove oldest or least relevant entry
        if len(self.long_term_memory) > self.memory_capacity:
            self.long_term_memory.pop(0)  # Simple FIFO strategy

    def get_relevant_memories(
        self, task: str, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Retrieve memories relevant to the current task.

        Args:
            task (str): The current task
            limit (int): Maximum number of memories to retrieve

        Returns:
            List[Dict[str, Any]]: Relevant memories
        """
        # In a production implementation, this would use embeddings and vector similarity
        # For now, implement a simple keyword-based relevance scoring
        scored_memories = []

        # Score and combine memories from both short and long-term
        all_memories = self.short_term_memory + self.long_term_memory
        for memory in all_memories:
            relevance = self._calculate_relevance(memory, task)
            scored_memories.append((memory, relevance))

        # Sort by relevance score (descending)
        scored_memories.sort(key=lambda x: x[1], reverse=True)

        # Return the top 'limit' memories
        return [memory for memory, score in scored_memories[:limit]]

    def _calculate_relevance(
        self, memory: Dict[str, Any], task: str
    ) -> float:
        """
        Calculate relevance of a memory to the current task.

        Args:
            memory (Dict[str, Any]): The memory entry
            task (str): The current task

        Returns:
            float: Relevance score between 0 and 1
        """
        # Simple implementation - count shared words between task and memory task
        memory_task = memory.get("task", "")
        memory_reflection = memory.get("reflection", "")

        task_words = set(task.lower().split())
        memory_words = set(
            (memory_task + " " + memory_reflection).lower().split()
        )

        if not task_words or not memory_words:
            return 0.0

        intersection = task_words.intersection(memory_words)
        return len(intersection) / min(
            len(task_words), len(memory_words)
        )

    def _similarity(
        self, entry1: Dict[str, Any], entry2: Dict[str, Any]
    ) -> float:
        """
        Calculate similarity between two memory entries.

        Args:
            entry1 (Dict[str, Any]): First memory entry
            entry2 (Dict[str, Any]): Second memory entry

        Returns:
            float: Similarity score between 0 and 1
        """
        # Simple implementation - compare tasks and reflections
        task1 = entry1.get("task", "")
        task2 = entry2.get("task", "")
        reflection1 = entry1.get("reflection", "")
        reflection2 = entry2.get("reflection", "")

        words1 = set((task1 + " " + reflection1).lower().split())
        words2 = set((task2 + " " + reflection2).lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1.intersection(words2)
        return len(intersection) / (
            len(words1) + len(words2) - len(intersection)
        )


class ReflexionAgent:
    """
    An advanced agent that implements the Reflexion framework to improve through self-reflection.

    The agent follows a process of:
    1. Acting on tasks
    2. Evaluating its performance
    3. Generating self-reflections
    4. Using these reflections to improve future responses

    Attributes:
        agent_name (str): The name of the agent
        system_prompt (str): The system prompt for the agent
        model_name (str): The model name used for generating responses
        conversation (Conversation): Instance to manage conversation history
        max_loops (int): Maximum number of reflection iterations per task
        memory (ReflexionMemory): Memory system to store experiences and reflections
        actor (Agent): The agent that generates initial responses
        evaluator (Agent): The agent that evaluates responses
        reflector (Agent): The agent that generates self-reflections
    """

    def __init__(
        self,
        agent_name: str = "reflexion-agent",
        system_prompt: str = REFLEXION_PROMPT,
        model_name: str = "openai/o1",
        max_loops: int = 3,
        memory_capacity: int = 100,
    ) -> None:
        """
        Initializes the ReflexionAgent with specified parameters.

        Args:
            agent_name (str): The name of the agent
            system_prompt (str): The system prompt for the agent
            model_name (str): The model name used for generating responses
            max_loops (int): Maximum number of reflection iterations per task
            memory_capacity (int): Maximum capacity of long-term memory
        """
        self.agent_name = agent_name
        self.system_prompt = system_prompt
        self.model_name = model_name
        self.conversation = Conversation(time_enabled=True)
        self.max_loops = max_loops
        self.memory = ReflexionMemory(memory_capacity=memory_capacity)

        # Actor agent - generates initial responses
        self.actor = Agent(
            agent_name=f"{agent_name}-actor",
            agent_description="You generate thorough, accurate, and helpful responses to tasks",
            system_prompt=system_prompt,
            model_name=model_name,
            max_loops=1,
        )

        # Evaluator agent - evaluates responses
        self.evaluator = Agent(
            agent_name=f"{agent_name}-evaluator",
            agent_description="You critically evaluate responses against quality criteria",
            system_prompt="""You are an expert evaluator of text quality. 
Your job is to thoroughly assess responses against these criteria:
1. Accuracy: Is all information factually correct?
2. Completeness: Does it address all aspects of the query?
3. Clarity: Is it well-structured and easy to understand?
4. Relevance: Does it focus on what the user needs?
5. Actionability: Does it provide practical, implementable solutions?

For each criterion, provide:
- A score from 1-10
- Specific examples of what was done well or poorly
- Concrete suggestions for improvement

Be precise, objective, and constructive in your criticism. 
Your goal is to help improve responses, not just criticize them.
End with an overall assessment and a final score from 1-10.
""",
            model_name=model_name,
            max_loops=1,
        )

        # Reflector agent - generates self-reflections
        self.reflector = Agent(
            agent_name=f"{agent_name}-reflector",
            agent_description="You generate insightful self-reflections to improve future responses",
            system_prompt="""You are an expert at generating insightful self-reflections.

Given a task, a response to that task, and an evaluation of that response, your job is to create a thoughtful self-reflection that will help improve future responses to similar tasks.

Your reflection should:
1. Identify key strengths and weaknesses in the response
2. Analyze why certain approaches worked or didn't work
3. Extract general principles and lessons learned
4. Provide specific strategies for handling similar tasks better in the future
5. Be concrete and actionable, not vague or general

Focus on extracting lasting insights that will be valuable for improving future performance. Be honest about shortcomings while maintaining a constructive, improvement-oriented tone.
""",
            model_name=model_name,
            max_loops=1,
        )

        logger.info(
            f"Initialized {self.agent_name} with model {self.model_name}"
        )

    def _extract_score_robust(self, evaluation: str) -> float:
        """
        Robustly extract a score from evaluation response using multiple strategies.

        Args:
            evaluation: The evaluation text from the evaluator agent

        Returns:
            float: Extracted score between 0.0 and 1.0, or DEFAULT_SCORE if extraction fails
        """
        # Strategy 1: Look for "final score: X/10" or "overall score: X" (existing pattern)
        # Handle markdown formatting by removing asterisks
        eval_clean = evaluation.replace('*', '').lower()

        score_patterns = [
            r"(?:final|overall)\s+score:?\s*(\d+(?:\.\d+)?)",
            r"score:?\s*(\d+(?:\.\d+)?)\s*/\s*10",
            r"(?:rating|grade):?\s*(\d+(?:\.\d+)?)\s*/\s*10",
            r"(?:rating|grade):?\s*(\d+(?:\.\d+)?)",
        ]

        for pattern in score_patterns:
            matches = re.findall(pattern, eval_clean)
            if matches:
                try:
                    score = float(matches[-1])
                    # Normalize to 0-1 range if needed
                    normalized = score / 10.0 if score > 1.0 else score
                    return max(0.0, min(1.0, normalized))
                except (ValueError, IndexError):
                    continue

        # Strategy 2: Look for any number between 0-10 with context
        context_patterns = [
            r"(\d+(?:\.\d+)?)\s*/\s*10",
            r"(\d+(?:\.\d+)?)\s*out of\s*10",
        ]

        for pattern in context_patterns:
            matches = re.findall(pattern, eval_clean)
            if matches:
                try:
                    score = float(matches[-1])
                    return max(0.0, min(1.0, score / 10.0))
                except (ValueError, IndexError):
                    continue

        # Strategy 3: Sentiment analysis fallback
        positive_keywords = ['excellent', 'great', 'strong', 'good', 'well done', 'impressive', 'thorough', 'comprehensive']
        negative_keywords = ['poor', 'weak', 'lacking', 'insufficient', 'needs improvement', 'unclear', 'incomplete']

        eval_lower = evaluation.lower()
        positive_count = sum(1 for kw in positive_keywords if kw in eval_lower)
        negative_count = sum(1 for kw in negative_keywords if kw in eval_lower)

        if positive_count > negative_count and positive_count > 0:
            logger.debug(f"Score extraction via sentiment: positive ({positive_count} keywords)")
            return 0.75  # Likely good
        elif negative_count > positive_count and negative_count > 0:
            logger.debug(f"Score extraction via sentiment: negative ({negative_count} keywords)")
            return 0.4   # Likely poor

        # Default fallback
        logger.warning(f"Could not extract score from evaluation, using default: {DEFAULT_SCORE}")
        logger.debug(f"Evaluation text (first 200 chars): {evaluation[:200]}")
        return DEFAULT_SCORE

    def act(
        self,
        task: str,
        relevant_memories: List[Dict[str, Any]] = None,
    ) -> str:
        """
        Generate a response to the given task using the actor agent.

        Args:
            task (str): The task to respond to
            relevant_memories (List[Dict[str, Any]]): Relevant past memories to consider

        Returns:
            str: The generated response
        """
        # Construct prompt with relevant memories if available
        prompt = task
        if relevant_memories and len(relevant_memories) > 0:
            memories_text = "\n\n".join(
                [
                    f"PAST REFLECTION: {memory.get('reflection', 'No reflection available')}"
                    for memory in relevant_memories
                ]
            )
            prompt = f"""TASK: {task}

RELEVANT PAST REFLECTIONS:
{memories_text}

Based on the task and relevant past reflections, provide a comprehensive response."""

        logger.debug(f"Actor prompt: {prompt}")

        # Generate response
        start_time = time.time()
        response = self.actor.run(task=prompt)
        end_time = time.time()

        logger.debug(
            f"Actor generated response in {end_time - start_time:.2f}s"
        )

        return response

    def evaluate(self, task: str, response: str) -> Tuple[str, float]:
        """
        Evaluate the quality of a response to a task.

        Args:
            task (str): The original task
            response (str): The response to evaluate

        Returns:
            Tuple[str, float]: Evaluation feedback and numerical score
        """
        prompt = f"""TASK: {task}

RESPONSE:
{response}

Evaluate this response thoroughly according to the criteria in your instructions. Be specific and constructive."""

        logger.debug(f"Evaluating response for task: {task[:100]}...")

        evaluation = self.evaluator.run(task=prompt)

        # Use robust score extraction with multiple fallback strategies
        normalized_score = self._extract_score_robust(evaluation)

        logger.info(
            f"Evaluation complete. Score: {normalized_score:.2f}"
        )

        return evaluation, normalized_score

    def reflect(
        self, task: str, response: str, evaluation: str
    ) -> str:
        """
        Generate a self-reflection based on the task, response, and evaluation.

        Args:
            task (str): The original task
            response (str): The generated response
            evaluation (str): The evaluation feedback

        Returns:
            str: The self-reflection
        """
        prompt = f"""TASK: {task}

RESPONSE:
{response}

EVALUATION:
{evaluation}

Based on this task, response, and evaluation, generate a thoughtful self-reflection that identifies key lessons and strategies for improving future responses to similar tasks."""

        logger.debug(
            f"Generating reflection for task: {task[:100]}..."
        )

        reflection = self.reflector.run(task=prompt)

        logger.debug(f"Reflection generated: {reflection[:100]}...")

        return reflection

    def refine(
        self,
        task: str,
        original_response: str,
        evaluation: str,
        reflection: str,
    ) -> str:
        """
        Refine the original response based on evaluation and reflection.

        Args:
            task (str): The original task
            original_response (str): The original response
            evaluation (str): The evaluation feedback
            reflection (str): The self-reflection

        Returns:
            str: The refined response
        """
        prompt = f"""TASK: {task}

ORIGINAL RESPONSE:
{original_response}

EVALUATION:
{evaluation}

REFLECTION:
{reflection}

Based on the original response, evaluation, and reflection, provide an improved response to the task. Focus on addressing the weaknesses identified while maintaining the strengths."""

        logger.debug(f"Refining response for task: {task[:100]}...")

        refined_response = self.actor.run(task=prompt)

        logger.debug(f"Response refined: {refined_response[:100]}...")

        return refined_response

    def step(
        self,
        task: str,
        iteration: int = 0,
        previous_response: str = None,
    ) -> Dict[str, Any]:
        """
        Process a single task through one iteration of the Reflexion process.

        Args:
            task (str): The task to process
            iteration (int): Current iteration number
            previous_response (str): Response from previous iteration

        Returns:
            Dict[str, Any]: Results of this iteration
        """
        # Retrieve relevant memories if not the first iteration
        relevant_memories = []
        if iteration > 0:
            relevant_memories = self.memory.get_relevant_memories(
                task
            )
            logger.debug(
                f"Retrieved {len(relevant_memories)} relevant memories"
            )

        # Generate response (or use previous response if provided)
        if previous_response is None:
            response = self.act(task, relevant_memories)
        else:
            response = previous_response

        # Evaluate the response
        evaluation, score = self.evaluate(task, response)

        # Generate reflection
        reflection = self.reflect(task, response, evaluation)

        # Store in memory
        memory_entry = {
            "task": task,
            "response": response,
            "evaluation": evaluation,
            "reflection": reflection,
            "score": score,
            "iteration": iteration,
        }

        self.memory.add_short_term_memory(memory_entry)

        # For high-quality reflections or final iterations, add to long-term memory
        if score > 0.8 or iteration == self.max_loops - 1:
            self.memory.add_long_term_memory(memory_entry)

        # Return results of this step
        return {
            "task": task,
            "response": response,
            "evaluation": evaluation,
            "reflection": reflection,
            "score": score,
            "iteration": iteration,
        }

    def run(
        self, tasks: List[str], include_intermediates: bool = False
    ) -> List[Any]:
        """
        Execute the Reflexion process for a list of tasks.

        Args:
            tasks (List[str]): List of tasks to process
            include_intermediates (bool): Whether to include intermediate iterations in results

        Returns:
            List[Any]: Final responses or complete iteration history
        """
        all_results = []

        for task_idx, task in enumerate(tasks):
            logger.info(
                f"\n{'='*60}\nProcessing task {task_idx+1}/{len(tasks)}\n{'='*60}"
            )
            logger.info(f"Task: {task[:100]}...")

            iterations = []
            best_response = None
            best_score = -1
            prev_score = -1

            # Run through multiple iterations of reflection
            for iteration in range(self.max_loops):
                logger.info(
                    f"\n--- Iteration {iteration+1}/{self.max_loops} ---"
                )

                # In first iteration, generate new response
                # In subsequent iterations, refine previous response
                if iteration == 0:
                    step_result = self.step(task, iteration)
                    step_result["response"]
                else:
                    # Refine previous response
                    prev_result = iterations[-1]
                    refined_response = self.refine(
                        task,
                        prev_result["response"],
                        prev_result["evaluation"],
                        prev_result["reflection"],
                    )

                    # Evaluate and reflect on the refined response
                    step_result = self.step(
                        task, iteration, refined_response
                    )

                iterations.append(step_result)

                # Track best response based on evaluation score
                if step_result["score"] > best_score:
                    best_response = step_result["response"]
                    best_score = step_result["score"]

                current_score = step_result["score"]
                logger.info(
                    f"Iteration {iteration+1} complete | Score: {current_score:.2f} | Best: {best_score:.2f}"
                )

                # Early termination condition 1: Score is high enough
                if current_score >= EARLY_TERMINATION_THRESHOLD:
                    logger.info(
                        f"✓ High score achieved ({current_score:.2f} >= {EARLY_TERMINATION_THRESHOLD}). Stopping early."
                    )
                    break

                # Early termination condition 2: Score not improving
                if iteration > 0 and (current_score - prev_score) < SCORE_IMPROVEMENT_THRESHOLD:
                    logger.info(
                        f"✓ Score improvement minimal ({current_score:.2f} vs {prev_score:.2f}). Stopping early."
                    )
                    break

                prev_score = current_score

            # Summary logging
            iterations_used = len(iterations)
            logger.info(
                f"\n{'='*60}\n"
                f"Task complete | Iterations used: {iterations_used}/{self.max_loops} | "
                f"Best score: {best_score:.2f}\n"
                f"{'='*60}"
            )

            # Add to conversation history (simplified)
            self.conversation.add("user", task)
            self.conversation.add("assistant", best_response)

            # Determine what to return
            if include_intermediates:
                all_results.append(iterations)
            else:
                all_results.append(best_response)

        return all_results


# # Example usage
# if __name__ == "__main__":
#     # Initialize the Reflexion Agent
#     agent = ReflexionAgent(
#         agent_name="reflexion-agent",
#         model_name="gpt-4.1",  # Using OpenAI's model
#         max_loops=1,  # Maximum of 3 reflection iterations
#     )

#     # Example tasks
#     tasks = [
#         "Explain QFT to a high school student.",
#     ]

#     # Run the agent
#     results = agent.run(tasks)

#     # Print results
#     for i, result in enumerate(results):
#         print(f"\n\nTASK {i+1}:")
#         print(f"{tasks[i]}\n")
#         print("FINAL RESPONSE:")
#         print(f"{result}")
