from concurrent.futures import ThreadPoolExecutor
from termcolor import colored
from tabulate import tabulate


class GodMode:
    """
    GodMode
    -----

    Architecture:
    How it works:
    1. GodMode receives a task from the user.
    2. GodMode distributes the task to all LLMs.
    3. GodMode collects the responses from all LLMs.
    4. GodMode prints the responses from all LLMs.

    Parameters:
    llms: list of LLMs

    Methods:
    run(task): distribute task to all LLMs and collect responses
    print_responses(task): print responses from all LLMs

    Usage:
    god_mode = GodMode(llms)
    god_mode.run(task)
    god_mode.print_responses(task)


    """

    def __init__(self, llms):
        self.llms = llms

    def run(self, task):
        with ThreadPoolExecutor() as executor:
            responses = executor.map(lambda llm: llm(task), self.llms)
        return list(responses)

    def print_responses(self, task):
        """Prints the responses in a tabular format"""
        responses = self.run_all(task)
        table = []
        for i, response in enumerate(responses):
            table.append([f"LLM {i+1}", response])
        print(
            colored(
                tabulate(table, headers=["LLM", "Response"], tablefmt="pretty"), "cyan"
            )
        )
