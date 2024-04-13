import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, List

from termcolor import colored

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelParallelizer:
    """
    ModelParallelizer, a class that parallelizes the execution of a task
    across multiple language models. It is a wrapper around the
    LanguageModel class.

    Args:
        llms (List[Callable]): A list of language models.
        retry_attempts (int): The number of retry attempts.
        iters (int): The number of iterations to run the task.

    Attributes:
        llms (List[Callable]): A list of language models.
        retry_attempts (int): The number of retry attempts.
        iters (int): The number of iterations to run the task.
        last_responses (List[str]): The last responses from the language
            models.
        task_history (List[str]): The task history.

    Examples:
    >>> from swarms.structs import ModelParallelizer
    >>> from swarms.llms import OpenAIChat
    >>> llms = [
    ...     OpenAIChat(
    ...         temperature=0.5,
    ...         openai_api_key="OPENAI_API_KEY",
    ...     ),
    ...     OpenAIChat(
    ...         temperature=0.5,
    ...         openai_api_key="OPENAI_API_KEY",
    ...     ),
    ... ]
    >>> mp = ModelParallelizer(llms)
    >>> mp.run("Generate a 10,000 word blog on health and wellness.")
    ['Generate a 10,000 word blog on health and wellness.', 'Generate a 10,000 word blog on health and wellness.']

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
            logger.error(
                f"[ERROR][ModelParallelizer] [ROOT CAUSE] [{error}]"
            )

    def run_all(self, task):
        """Run the task on all LLMs"""
        responses = []
        for llm in self.llms:
            responses.append(llm(task))
        return responses

    # New Features
    def save_responses_to_file(self, filename):
        """Save responses to file"""
        with open(filename, "w") as file:
            table = [
                [f"LLM {i + 1}", response]
                for i, response in enumerate(self.last_responses)
            ]
            file.write(table)

    @classmethod
    def load_llms_from_file(cls, filename):
        """Load llms from file"""
        with open(filename) as file:
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
            [f"LLM {i + 1}", response]
            for i, response in enumerate(self.last_responses)
        ]
        print(
            colored(
                table,
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
            logger.error(
                f"[ERROR][ModelParallelizer] [ROOT CAUSE] [{error}]"
            )
            raise error

    def add_llm(self, llm: Callable):
        """Add an llm to the god mode"""
        logger.info(f"[INFO][ModelParallelizer] Adding LLM {llm}")

        try:
            self.llms.append(llm)
        except Exception as error:
            logger.error(
                f"[ERROR][ModelParallelizer] [ROOT CAUSE] [{error}]"
            )
            raise error

    def remove_llm(self, llm: Callable):
        """Remove an llm from the god mode"""
        logger.info(f"[INFO][ModelParallelizer] Removing LLM {llm}")

        try:
            self.llms.remove(llm)
        except Exception as error:
            logger.error(
                f"[ERROR][ModelParallelizer] [ROOT CAUSE] [{error}]"
            )
            raise error
