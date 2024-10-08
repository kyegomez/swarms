from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Callable, List, Optional, Dict

from swarms import Agent
from swarms.structs.base_swarm import BaseSwarm
from swarms.utils.loguru_logger import logger


class MonteCarloSwarm(BaseSwarm):
    def __init__(
        self,
        agents: List[Agent],
        parallel: bool = False,
        iterations: int = 10,  # Number of Monte Carlo iterations
        result_aggregator: Optional[Callable[[List[Any]], Any]] = None,
        agent_selector: Optional[Callable[[List[Agent], int, Dict], Agent]] = None,
        max_workers: Optional[int] = None,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(agents=agents, *args, **kwargs)

        if not agents:
            raise ValueError("The agents list cannot be empty.")

        self.agents = agents
        self.parallel = parallel
        self.iterations = iterations
        self.result_aggregator = result_aggregator or self.default_aggregator
        self.agent_selector = agent_selector or self.default_agent_selector
        self.max_workers = max_workers or len(agents)
        self.agent_performance: Dict[str, List[float]] = {agent.agent_name: [] for agent in agents}

    def run(self, task: str) -> Any:
        logger.info(f"Starting MonteCarloSwarm with parallel={self.parallel}, iterations={self.iterations}")

        results = []
        for i in range(self.iterations):
            logger.info(f"Starting iteration {i+1}")
            if self.parallel:
                iteration_results = self._run_parallel(task)
            else:
                iteration_results = self._run_sequential(task)

            results.append(self.result_aggregator(iteration_results))

            # Update agent performance metrics (example)
            for j, agent_result in enumerate(iteration_results):
                agent_name = self.agents[j].agent_name
                # Example: Store some performance metric (replace with your actual metric)
                self.agent_performance[agent_name].append(len(str(agent_result)))


        final_output = self.result_aggregator(results) # Aggregate across all iterations
        logger.info(f"MonteCarloSwarm completed. Final output: {final_output}")
        logger.info(f"Agent performance: {self.agent_performance}")
        return final_output

    def _run_sequential(self, task: str) -> List[Any]:
        results = []
        current_input = task
        for i in range(len(self.agents)):
            agent = self.agent_selector(self.agents, i, self.agent_performance)  # Dynamic agent selection
            logger.info(f"Agent {agent.agent_name} processing sequentially...")
            current_output = agent.run(current_input)
            results.append(current_output)
            current_input = current_output # Pass output to the next agent
        return results


    def _run_parallel(self, task: str) -> List[Any]:
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = []
            for i in range(len(self.agents)):
                agent = self.agent_selector(self.agents, i, self.agent_performance)
                logger.info(f"Submitting task to agent {agent.agent_name} in parallel...")
                futures.append(executor.submit(agent.run, task))


            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                    logger.info(f"Agent completed with result: {result}")
                except Exception as e:
                    logger.error(f"Agent encountered an error: {e}")
                    results.append(None)
        return results

    @staticmethod
    def default_aggregator(results: List[Any]) -> Any:
        return results[-1] if results else None  # Return the last result by default

    @staticmethod
    def default_agent_selector(agents: List[Agent], iteration: int, agent_performance: Dict) -> Agent:
        return agents[iteration % len(agents)]  # Round-robin by default

# Example usage with dynamic agent selection and iterative refinement:


def best_performing_agent_selector(agents: List[Agent], iteration: int, agent_performance: Dict) -> Agent:
    """Selects the best performing agent based on average result length."""
    if not all(agent_performance.values()):  # Check if any agent has no performance data yet
        return agents[iteration % len(agents)] # Default to round robin if no performance data

    average_performance = {
        agent_name: sum(scores) / len(scores) if scores else 0
        for agent_name, scores in agent_performance.items()
    }
    best_agent_name = max(average_performance, key=average_performance.get)
    return next((agent for agent in agents if agent.agent_name == best_agent_name), agents[0])
