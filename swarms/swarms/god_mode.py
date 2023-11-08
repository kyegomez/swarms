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
        self.last_responses = None
        self.task_history = []

    def run(self, task: str):
        """Run the task string"""
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
                tabulate(table, headers=["LLM", "Response"], tablefmt="pretty"),
                "cyan"))

    def run_all(self, task):
        """Run the task on all LLMs"""
        responses = []
        for llm in self.llms:
            responses.append(llm(task))
        return responses

    def arun_all(self, task):
        """Asynchronous run the task on all LLMs"""
        with ThreadPoolExecutor() as executor:
            responses = executor.map(lambda llm: llm(task), self.llms)
        return list(responses)

    def print_arun_all(self, task):
        """Prints the responses in a tabular format"""
        responses = self.arun_all(task)
        table = []
        for i, response in enumerate(responses):
            table.append([f"LLM {i+1}", response])
        print(
            colored(
                tabulate(table, headers=["LLM", "Response"], tablefmt="pretty"),
                "cyan"))

    # New Features
    def save_responses_to_file(self, filename):
        """Save responses to file"""
        with open(filename, "w") as file:
            table = [[f"LLM {i+1}", response]
                     for i, response in enumerate(self.last_responses)]
            file.write(tabulate(table, headers=["LLM", "Response"]))

    @classmethod
    def load_llms_from_file(cls, filename):
        """Load llms from file"""
        with open(filename, "r") as file:
            llms = [line.strip() for line in file.readlines()]
        return cls(llms)

    def get_task_history(self):
        """Get Task history"""
        return self.task_history

    def summary(self):
        """Summary"""
        print("Tasks History:")
        for i, task in enumerate(self.task_history):
            print(f"{i + 1}. {task}")
        print("\nLast Responses:")
        table = [[f"LLM {i+1}", response]
                 for i, response in enumerate(self.last_responses)]
        print(
            colored(
                tabulate(table, headers=["LLM", "Response"], tablefmt="pretty"),
                "cyan"))
