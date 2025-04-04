from typing import List, Optional, Dict, Any
from loguru import logger

try:
    from vllm import LLM, SamplingParams
except ImportError:
    import subprocess
    import sys

    print("Installing vllm")
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-U", "vllm"]
    )
    print("vllm installed")
    from vllm import LLM, SamplingParams


class VLLMWrapper:
    """
    A wrapper class for vLLM that provides a similar interface to LiteLLM.
    This class handles model initialization and inference using vLLM.
    """

    def __init__(
        self,
        model_name: str = "meta-llama/Llama-2-7b-chat-hf",
        system_prompt: Optional[str] = None,
        stream: bool = False,
        temperature: float = 0.5,
        max_tokens: int = 4000,
        max_completion_tokens: int = 4000,
        tools_list_dictionary: Optional[List[Dict[str, Any]]] = None,
        tool_choice: str = "auto",
        parallel_tool_calls: bool = False,
        *args,
        **kwargs,
    ):
        """
        Initialize the vLLM wrapper with the given parameters.

        Args:
            model_name (str): The name of the model to use. Defaults to "meta-llama/Llama-2-7b-chat-hf".
            system_prompt (str, optional): The system prompt to use. Defaults to None.
            stream (bool): Whether to stream the output. Defaults to False.
            temperature (float): The temperature for sampling. Defaults to 0.5.
            max_tokens (int): The maximum number of tokens to generate. Defaults to 4000.
            max_completion_tokens (int): The maximum number of completion tokens. Defaults to 4000.
            tools_list_dictionary (List[Dict[str, Any]], optional): List of available tools. Defaults to None.
            tool_choice (str): How to choose tools. Defaults to "auto".
            parallel_tool_calls (bool): Whether to allow parallel tool calls. Defaults to False.
        """
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.stream = stream
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_completion_tokens = max_completion_tokens
        self.tools_list_dictionary = tools_list_dictionary
        self.tool_choice = tool_choice
        self.parallel_tool_calls = parallel_tool_calls

        # Initialize vLLM
        self.llm = LLM(model=model_name, **kwargs)
        self.sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def _prepare_prompt(self, task: str) -> str:
        """
        Prepare the prompt for the given task.

        Args:
            task (str): The task to prepare the prompt for.

        Returns:
            str: The prepared prompt.
        """
        if self.system_prompt:
            return f"{self.system_prompt}\n\nUser: {task}\nAssistant:"
        return f"User: {task}\nAssistant:"

    def run(self, task: str, *args, **kwargs) -> str:
        """
        Run the model for the given task.

        Args:
            task (str): The task to run the model for.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The model's response.
        """
        try:
            prompt = self._prepare_prompt(task)

            outputs = self.llm.generate(prompt, self.sampling_params)
            response = outputs[0].outputs[0].text.strip()

            return response

        except Exception as error:
            logger.error(f"Error in VLLMWrapper: {error}")
            raise error

    def __call__(self, task: str, *args, **kwargs) -> str:
        """
        Call the model for the given task.

        Args:
            task (str): The task to run the model for.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            str: The model's response.
        """
        return self.run(task, *args, **kwargs)

    def batched_run(
        self, tasks: List[str], batch_size: int = 10
    ) -> List[str]:
        """
        Run the model for multiple tasks in batches.

        Args:
            tasks (List[str]): List of tasks to run.
            batch_size (int): Size of each batch. Defaults to 10.

        Returns:
            List[str]: List of model responses.
        """
        logger.info(
            f"Running tasks in batches of size {batch_size}. Total tasks: {len(tasks)}"
        )
        results = []

        for i in range(0, len(tasks), batch_size):
            batch = tasks[i : i + batch_size]
            for task in batch:
                logger.info(f"Running task: {task}")
                results.append(self.run(task))

        logger.info("Completed all tasks.")
        return results
