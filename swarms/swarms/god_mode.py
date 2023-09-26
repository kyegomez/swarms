from concurrent.futures import ThreadPoolExecutor

from tabulate import tabulate
from termcolor import colored

from swarms.workers.worker import Worker

class GodMode:
    def __init__(
        self, 
        num_workers, 
        num_llms, 
        openai_api_key, 
        ai_name
    ):
        self.workers = [
            Worker(
                openai_api_key=openai_api_key, 
                ai_name=ai_name
            ) for _ in range(num_workers)
        ]
        # self.llms = [LLM() for _ in range(num_llms)]
        self.all_agents = self.workers # + self.llms

    def run_all(self, task):
        with ThreadPoolExecutor() as executor:
            responses = executor.map(
                lambda agent: agent.run(task) if hasattr(
                    agent, 'run'
                ) else agent(task), self.all_agents
            )

        return list(responses)

    def print_responses(self, task):
        responses = self.run_all(task)

        table = []

        for i, response in enumerate(responses):
            agent_type = "Worker" if i < len(self.workers) else "LLM"
            table.append([agent_type, response])
        print(
            colored(
                tabulate(
                    table, 
                    headers=["Agent Type", "Response"], 
                    tablefmt="pretty"
                ), "cyan")
            )

# Usage
god_mode = GodMode(num_workers=3, openai_api_key="", ai_name="Optimus Prime")
task = "What were the winning Boston Marathon times for the past 5 years (ending in 2022)? Generate a table of the year, name, country of origin, and times."
god_mode.print_responses(task)