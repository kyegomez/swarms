import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List

from tabulate import tabulate
from termcolor import colored

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelParallelizer:
    """
    ModelParallelizer
    -----

    Architecture:
    How it works:
    1. ModelParallelizer receives a task from the user.
    2. ModelParallelizer distributes the task to all LLMs.
    3. ModelParallelizer collects the responses from all LLMs.
    4. ModelParallelizer prints the responses from all LLMs.

    Parameters:
    llms: list of LLMs

    Methods:
    run(task): distribute task to all LLMs and collect responses
    print_responses(task): print responses from all LLMs

    Usage:
    parallelizer = ModelParallelizer(llms)
    parallelizer.run(task)
    parallelizer.print_responses(task)


    """

    def __init__(
        self,
        llms: List[Callable] = None,
        retry_attempts: int = 3,
        iters: int = None,
        *args,
        **kwargs,
    ):
        self.llms = llms
        self.retry_attempts = retry_attempts
        self.iters = iters
        self.last_responses = None
        self.task_history = []

    def run(self, task: str):
        """Run the task string"""
        try:
            for i in range(self.iters):
                with ThreadPoolExecutor() as executor:
                    responses = executor.map(
                        lambda llm: llm(task), self.llms
                    )
                return list(responses)
        except Exception as error:
            print(
                f"[ERROR][ModelParallelizer] [ROOT CAUSE] [{error}]"
            )

    def print_responses(self, task):
        """Prints the responses in a tabular format"""
        responses = self.run_all(task)
        table = []
        for i, response in enumerate(responses):
            table.append([f"LLM {i+1}", response])
        print(
            colored(
                tabulate(
                    table,
                    headers=["LLM", "Response"],
                    tablefmt="pretty",
                ),
                "cyan",
            )
        )

    def run_all(self, task):
        """Run the task on all LLMs"""
        responses = []
        for llm in self.llms:
            responses.append(llm(task))
        return responses

    def print_arun_all(self, task):
        """Prints the responses in a tabular format"""
        responses = self.arun_all(task)
        table = []
        for i, response in enumerate(responses):
            table.append([f"LLM {i+1}", response])
        print(
            colored(
                tabulate(
                    table,
                    headers=["LLM", "Response"],
                    tablefmt="pretty",
                ),
                "cyan",
            )
        )

    # New Features
    def save_responses_to_file(self, filename):
        """Save responses to file"""
        with open(filename, "w") as file:
            table = [
                [f"LLM {i+1}", response]
                for i, response in enumerate(self.last_responses)
            ]
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
        table = [
            [f"LLM {i+1}", response]
            for i, response in enumerate(self.last_responses)
        ]
        print(
            colored(
                tabulate(
                    table,
                    headers=["LLM", "Response"],
                    tablefmt="pretty",
                ),
                "cyan",
            )
        )

    async def arun(self, task: str):
        """Asynchronous run the task string"""
        loop = asyncio.get_event_loop()
        futures = [
            loop.run_in_executor(None, lambda llm: llm(task), llm)
            for llm in self.llms
        ]
        for response in await asyncio.gather(*futures):
            print(response)

    def concurrent_run(self, task: str) -> List[str]:
        """Synchronously run the task on all llms and collect responses"""
        try:
            with ThreadPoolExecutor() as executor:
                future_to_llm = {
                    executor.submit(llm, task): llm
                    for llm in self.llms
                }
                responses = []
                for future in as_completed(future_to_llm):
                    try:
                        responses.append(future.result())
                    except Exception as error:
                        print(
                            f"{future_to_llm[future]} generated an"
                            f" exception: {error}"
                        )
            self.last_responses = responses
            self.task_history.append(task)
            return responses
        except Exception as error:
            print(
                f"[ERROR][ModelParallelizer] [ROOT CAUSE] [{error}]"
            )
            raise error

    def add_llm(self, llm: Callable):
        """Add an llm to the god mode"""
        self.llms.append(llm)

    def remove_llm(self, llm: Callable):
        """Remove an llm from the god mode"""
        self.llms.remove(llm)
