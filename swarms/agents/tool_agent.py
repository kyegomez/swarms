from typing import List, Optional, Dict, Any, Callable
from loguru import logger
from swarms.agents.exceptions import (
    ToolAgentError,
    ToolExecutionError,
    ToolValidationError,
    ToolNotFoundError,
    ToolParameterError
)

class ToolAgent:
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
        retry_attempts: int = 3,
        retry_interval: float = 1.0,
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
            retry_attempts (int): Number of retry attempts for failed operations. Defaults to 3.
            retry_interval (float): Time to wait between retries in seconds. Defaults to 1.0.
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
        self.retry_attempts = retry_attempts
        self.retry_interval = retry_interval

        # Initialize vLLM
        try:
            self.llm = LLM(model=model_name, **kwargs)
            self.sampling_params = SamplingParams(
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as e:
            raise ToolExecutionError(
                "model_initialization",
                e,
                {"model_name": model_name, "kwargs": kwargs}
            )

    def _validate_tool(self, tool_name: str, parameters: Dict[str, Any]) -> None:
        """
        Validate tool parameters before execution.
        Args:
            tool_name (str): Name of the tool to validate
            parameters (Dict[str, Any]): Parameters to validate
        Raises:
            ToolValidationError: If validation fails
        """
        if not self.tools_list_dictionary:
            raise ToolValidationError(
                tool_name,
                "parameters",
                "No tools available for validation"
            )

        tool_spec = next(
            (tool for tool in self.tools_list_dictionary if tool["name"] == tool_name),
            None
        )

        if not tool_spec:
            raise ToolNotFoundError(tool_name)

        required_params = {
            param["name"] for param in tool_spec["parameters"]
            if param.get("required", True)
        }

        missing_params = required_params - set(parameters.keys())
        if missing_params:
            raise ToolParameterError(
                tool_name,
                f"Missing required parameters: {', '.join(missing_params)}"
            )

    def _execute_with_retry(self, func: Callable, *args, **kwargs) -> Any:
        """
        Execute a function with retry logic.
        Args:
            func (Callable): Function to execute
            *args: Positional arguments for the function
            **kwargs: Keyword arguments for the function
        Returns:
            Any: Result of the function execution
        Raises:
            ToolExecutionError: If all retry attempts fail
        """
        last_error = None
        for attempt in range(self.retry_attempts):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                logger.warning(
                    f"Attempt {attempt + 1}/{self.retry_attempts} failed: {str(e)}"
                )
                if attempt < self.retry_attempts - 1:
                    time.sleep(self.retry_interval)

        raise ToolExecutionError(
            func.__name__,
            last_error,
            {"attempts": self.retry_attempts}
        )

    def run(self, task: str, *args, **kwargs) -> str:
        """
        Run the tool agent for the specified task.
        Args:
            task (str): The task to be performed by the tool agent.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.
        Returns:
            The output of the tool agent.
        Raises:
            ToolExecutionError: If an error occurs during execution.
        """
        try:
            if not self.llm:
                raise ToolExecutionError(
                    "run",
                    Exception("LLM not initialized"),
                    {"task": task}
                )

            logger.info(f"Running task: {task}")
            
            # Prepare the prompt
            prompt = self._prepare_prompt(task)
            
            # Execute with retry logic
            outputs = self._execute_with_retry(
                self.llm.generate,
                prompt,
                self.sampling_params
            )
            
            response = outputs[0].outputs[0].text.strip()
            return response

        except Exception as error:
            logger.error(f"Error running task: {error}")
            raise ToolExecutionError(
                "run",
                error,
                {"task": task, "args": args, "kwargs": kwargs}
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

    def batched_run(self, tasks: List[str], batch_size: int = 10) -> List[str]:
        """
        Run the model for multiple tasks in batches.
        Args:
            tasks (List[str]): List of tasks to run.
            batch_size (int): Size of each batch. Defaults to 10.
        Returns:
            List[str]: List of model responses.
        Raises:
            ToolExecutionError: If an error occurs during batch execution.
        """
        logger.info(f"Running tasks in batches of size {batch_size}. Total tasks: {len(tasks)}")
        results = []

        try:
            for i in range(0, len(tasks), batch_size):
                batch = tasks[i:i + batch_size]
                for task in batch:
                    logger.info(f"Running task: {task}")
                    try:
                        result = self.run(task)
                        results.append(result)
                    except ToolExecutionError as e:
                        logger.error(f"Failed to execute task '{task}': {e}")
                        results.append(f"Error: {str(e)}")
                        continue

            logger.info("Completed all tasks.")
            return results

        except Exception as error:
            logger.error(f"Error in batch execution: {error}")
            raise ToolExecutionError(
                "batched_run",
                error,
                {"tasks": tasks, "batch_size": batch_size}
            )
