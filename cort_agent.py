# AI generate initial response
# AI decides how many "thinking rounds" it needs
# For each round:
# Generates 3 alternative responses
# Evaluates all responses
# Picks the best one
# Final response is the survivor of this AI battle royale
from swarms import Agent


# OpenAI function schema for determining thinking rounds
thinking_rounds_schema = {
    "name": "determine_thinking_rounds",
    "description": "Determines the optimal number of thinking rounds needed for a task",
    "parameters": {
        "type": "object",
        "properties": {
            "num_rounds": {
                "type": "integer",
                "description": "The number of thinking rounds needed (1-5)",
                "minimum": 1,
                "maximum": 5,
            }
        },
        "required": ["num_rounds"],
    },
}

# System prompt for determining thinking rounds
THINKING_ROUNDS_PROMPT = """You are an expert at determining the optimal number of thinking rounds needed for complex tasks. Your role is to analyze the task and determine how many rounds of thinking and evaluation would be most beneficial.

Consider the following factors when determining the number of rounds:
1. Task Complexity: More complex tasks may require more rounds
2. Potential for Multiple Valid Approaches: Tasks with multiple valid solutions need more rounds
3. Risk of Error: Higher-stakes tasks may benefit from more rounds
4. Time Sensitivity: Balance thoroughness with efficiency

Guidelines for number of rounds:
- 1 round: Simple, straightforward tasks with clear solutions
- 2-3 rounds: Moderately complex tasks with some ambiguity
- 4-5 rounds: Highly complex tasks with multiple valid approaches or high-stakes decisions

Your response should be a single number between 1 and 5, representing the optimal number of thinking rounds needed."""

# Schema for generating alternative responses
alternative_responses_schema = {
    "name": "generate_alternatives",
    "description": "Generates multiple alternative responses to a task",
    "parameters": {
        "type": "object",
        "properties": {
            "alternatives": {
                "type": "array",
                "description": "List of alternative responses",
                "items": {
                    "type": "object",
                    "properties": {
                        "response": {
                            "type": "string",
                            "description": "The alternative response",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "Explanation of why this approach was chosen",
                        },
                    },
                    "required": ["response", "reasoning"],
                },
                "minItems": 3,
                "maxItems": 3,
            }
        },
        "required": ["alternatives"],
    },
}

# Schema for evaluating responses
evaluation_schema = {
    "name": "evaluate_responses",
    "description": "Evaluates and ranks alternative responses",
    "parameters": {
        "type": "object",
        "properties": {
            "evaluation": {
                "type": "object",
                "properties": {
                    "best_response": {
                        "type": "string",
                        "description": "The selected best response",
                    },
                    "ranking": {
                        "type": "array",
                        "description": "Ranked list of responses from best to worst",
                        "items": {
                            "type": "object",
                            "properties": {
                                "response": {
                                    "type": "string",
                                    "description": "The response",
                                },
                                "score": {
                                    "type": "number",
                                    "description": "Score from 0-100",
                                },
                                "reasoning": {
                                    "type": "string",
                                    "description": "Explanation of the score",
                                },
                            },
                            "required": [
                                "response",
                                "score",
                                "reasoning",
                            ],
                        },
                    },
                },
                "required": ["best_response", "ranking"],
            }
        },
        "required": ["evaluation"],
    },
}

# System prompt for generating alternatives
ALTERNATIVES_PROMPT = """You are an expert at generating diverse and creative alternative responses to tasks. Your role is to generate 3 distinct approaches to solving the given task.

For each alternative:
1. Consider a different perspective or approach
2. Provide clear reasoning for why this approach might be effective
3. Ensure alternatives are meaningfully different from each other
4. Maintain high quality and relevance to the task

Your response should include 3 alternatives, each with its own reasoning."""

# System prompt for evaluation
EVALUATION_PROMPT = """You are an expert at evaluating and comparing different responses to tasks. Your role is to critically analyze each response and determine which is the most effective.

Consider the following criteria when evaluating:
1. Relevance to the task
2. Completeness of the solution
3. Creativity and innovation
4. Practicality and feasibility
5. Clarity and coherence

Your response should include:
1. The best response selected
2. A ranked list of all responses with scores and reasoning"""


class CortAgent:
    def __init__(
        self,
        alternative_responses: int = 3,
    ):
        self.thinking_rounds = Agent(
            agent_name="CortAgent",
            agent_description="CortAgent is a multi-step agent that uses a battle royale approach to determine the best response to a task.",
            model_name="gpt-4o-mini",
            max_loops=1,
            dynamic_temperature_enabled=True,
            tools_list_dictionary=thinking_rounds_schema,
            system_prompt=THINKING_ROUNDS_PROMPT,
        )

        self.alternatives_agent = Agent(
            agent_name="CortAgentAlternatives",
            agent_description="Generates multiple alternative responses to a task",
            model_name="gpt-4o-mini",
            max_loops=1,
            dynamic_temperature_enabled=True,
            tools_list_dictionary=alternative_responses_schema,
            system_prompt=ALTERNATIVES_PROMPT,
        )

        self.evaluation_agent = Agent(
            agent_name="CortAgentEvaluation",
            agent_description="Evaluates and ranks alternative responses",
            model_name="gpt-4o-mini",
            max_loops=1,
            dynamic_temperature_enabled=True,
            tools_list_dictionary=evaluation_schema,
            system_prompt=EVALUATION_PROMPT,
        )

    def run(self, task: str):
        # First determine number of thinking rounds
        num_rounds = self.thinking_rounds.run(task)

        # Initialize with the task
        current_task = task
        best_response = None

        # Run the battle royale for the determined number of rounds
        for round_num in range(num_rounds):
            # Generate alternatives
            alternatives = self.alternatives_agent.run(current_task)

            # Evaluate alternatives
            evaluation = self.evaluation_agent.run(alternatives)

            # Update best response and current task for next round
            best_response = evaluation["evaluation"]["best_response"]
            current_task = f"Previous best response: {best_response}\nOriginal task: {task}"

        return best_response
