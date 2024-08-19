import os
from swarms import Agent, OpenAIChat
from typing import List, Union, Callable
from collections import Counter

# Aggregation functions


def aggregate_most_common_result(results: List[str]) -> str:
    """
    Aggregate results using the most common result.

    Args:
        results (List[str]): List of results from each iteration.

    Returns:
        str: The most common result.
    """
    result_counter = Counter(results)
    most_common_result = result_counter.most_common(1)[0][0]
    return most_common_result


def aggregate_weighted_vote(results: List[str], weights: List[int]) -> str:
    """
    Aggregate results using a weighted voting system.

    Args:
        results (List[str]): List of results from each iteration.
        weights (List[int]): List of weights corresponding to each result.

    Returns:
        str: The result with the highest weighted vote.
    """
    weighted_results = Counter()
    for result, weight in zip(results, weights):
        weighted_results[result] += weight

    weighted_result = weighted_results.most_common(1)[0][0]
    return weighted_result


def aggregate_average_numerical(results: List[Union[str, float]]) -> float:
    """
    Aggregate results by averaging numerical outputs.

    Args:
        results (List[Union[str, float]]): List of numerical results from each iteration.

    Returns:
        float: The average of the numerical results.
    """
    numerical_results = [
        float(result) for result in results if is_numerical(result)
    ]
    if numerical_results:
        return sum(numerical_results) / len(numerical_results)
    else:
        return float("nan")  # or handle non-numerical case as needed


def aggregate_consensus(results: List[str]) -> Union[str, None]:
    """
    Aggregate results by checking if there's a consensus (all results are the same).

    Args:
        results (List[str]): List of results from each iteration.

    Returns:
        Union[str, None]: The consensus result if there is one, otherwise None.
    """
    if all(result == results[0] for result in results):
        return results[0]
    else:
        return None  # or handle lack of consensus as needed


def is_numerical(value: str) -> bool:
    """
    Check if a string can be interpreted as a numerical value.

    Args:
        value (str): The string to check.

    Returns:
        bool: True if the string is numerical, otherwise False.
    """
    try:
        float(value)
        return True
    except ValueError:
        return False


# MonteCarloSwarm class


class MonteCarloSwarm:
    def __init__(
        self,
        agents: List[Agent],
        iterations: int = 100,
        aggregator: Callable = aggregate_most_common_result,
    ):
        self.agents = agents
        self.iterations = iterations
        self.aggregator = aggregator

    def run(self, task: str) -> Union[str, float, None]:
        """
        Execute the Monte Carlo swarm, passing the output of each agent to the next.
        The final result is aggregated over multiple iterations using the provided aggregator.

        Args:
            task (str): The task for the swarm to execute.

        Returns:
            Union[str, float, None]: The final aggregated result.
        """
        aggregated_results = []

        for i in range(self.iterations):
            result = task
            for agent in self.agents:
                result = agent.run(result)
            aggregated_results.append(result)

        # Apply the selected aggregation function
        final_result = self.aggregator(aggregated_results)
        return final_result


# Example usage:

# Assuming you have the OpenAI API key set up and agents defined
api_key = os.getenv("OPENAI_API_KEY")
model = OpenAIChat(
    api_key=api_key, model_name="gpt-4o-mini", temperature=0.1
)

agent1 = Agent(
    agent_name="Agent1",
    system_prompt="System prompt for agent 1",
    llm=model,
    max_loops=1,
    verbose=True,
)

agent2 = Agent(
    agent_name="Agent2",
    system_prompt="System prompt for agent 2",
    llm=model,
    max_loops=1,
    verbose=True,
)

# Create a MonteCarloSwarm with the agents and a selected aggregation function
swarm = MonteCarloSwarm(
    agents=[agent1, agent2],
    iterations=1,
    aggregator=aggregate_weighted_vote,
)

# Run the swarm on a specific task
final_output = swarm.run(
    "What are the components of a startup's stock incentive plan?"
)
print("Final Output:", final_output)

# You can easily switch the aggregation function by passing a different one to the constructor:
# swarm = MonteCarloSwarm(agents=[agent1, agent2], iterations=100, aggregator=aggregate_weighted_vote)

# If using weighted voting, you'll need to adjust the aggregator call to provide the weights:
# weights = list(range(100, 0, -1))  # Example weights for 100 iterations
# swarm = MonteCarloSwarm(agents=[agent1, agent2], iterations=100, aggregator=lambda results: aggregate_weighted_vote(results, weights))
