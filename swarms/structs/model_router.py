import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from typing import Optional
from pydantic import BaseModel, Field

from swarms.utils.function_caller_model import OpenAIFunctionCaller
from swarms.utils.any_to_str import any_to_str
from swarms.utils.formatter import formatter
from swarms.utils.litellm_wrapper import LiteLLM

model_recommendations = {
    "gpt-4o": {
        "description": "Fast and efficient for simple tasks and general queries",
        "best_for": [
            "Simple queries",
            "Basic text generation",
            "Quick responses",
            "Everyday tasks",
        ],
        "provider": "openai",
    },
    "gpt-4-turbo": {
        "description": "Latest GPT-4 model with improved capabilities and knowledge cutoff",
        "best_for": [
            "Complex tasks",
            "Up-to-date knowledge",
            "Long context understanding",
        ],
        "provider": "openai",
    },
    "gpt-3.5-turbo": {
        "description": "Fast and cost-effective model for straightforward tasks",
        "best_for": [
            "Chat applications",
            "Content generation",
            "Basic assistance",
        ],
        "provider": "openai",
    },
    "o3-mini": {
        "description": "Lightweight model good for basic tasks with lower compute requirements",
        "best_for": [
            "Basic text completion",
            "Simple classification",
            "Resource-constrained environments",
        ],
        "provider": "groq",
    },
    "deepseek-reasoner": {
        "description": "Specialized for complex reasoning and analytical tasks",
        "best_for": [
            "Complex problem solving",
            "Logical reasoning",
            "Mathematical analysis",
            "High IQ tasks",
        ],
        "provider": "deepseek",
    },
    "claude-3-5-sonnet": {
        "description": "Well-rounded model with strong reasoning and creativity",
        "best_for": [
            "Creative writing",
            "Detailed analysis",
            "Nuanced responses",
        ],
        "provider": "anthropic",
    },
    "claude-3-opus": {
        "description": "Most capable Claude model with enhanced reasoning and analysis",
        "best_for": [
            "Research",
            "Complex analysis",
            "Technical writing",
            "Code generation",
        ],
        "provider": "anthropic",
    },
    "gemini-pro": {
        "description": "Google's advanced model with strong general capabilities",
        "best_for": [
            "Multimodal tasks",
            "Code generation",
            "Creative content",
        ],
        "provider": "google",
    },
    "mistral-large": {
        "description": "Open source model with strong performance across tasks",
        "best_for": [
            "General purpose tasks",
            "Code assistance",
            "Content generation",
        ],
        "provider": "mistral",
    },
}

providers = {
    "openai": "Primary provider for GPT models",
    "groq": "High-performance inference provider",
    "anthropic": "Provider of Claude models",
    "google": "Provider of PaLM and Gemini models",
    "azure": "Cloud platform for various model deployments",
    "deepseek": "Provider of specialized reasoning models",
    "mistral": "Provider of open source and commercial language models",
}


class ModelOutput(BaseModel):
    rationale: Optional[str]
    model: Optional[str]
    provider: Optional[str]
    task: Optional[str] = Field(
        description="The task to be executed for the model. It should be a clear, concise, and detailed task that the model can execute. It should only include details of the task, not the reasoning or the rationale, model, provider, or anything else. Do not include any other information in the task."
    )
    max_tokens: Optional[int] = Field(
        description="The maximum number of tokens to use for the model"
    )
    temperature: Optional[float] = Field(
        description="The temperature to use for the model"
    )
    system_prompt: Optional[str] = Field(
        description="The system prompt to use for the model. Leverge the best techniques to make the model perform the best. Make sure the prompt is clear, extensive, and detailed."
    )


providers = any_to_str(providers)
model_recommendations = any_to_str(model_recommendations)
data = f"Providers: {providers}\nModel Recommendations: {model_recommendations}"

model_router_system_prompt = f"""
You are an expect model router responsible for recommending the optimal AI model for specific tasks.

Available Models and Their Strengths:
- GPT-4: Best for complex reasoning, coding, and analysis requiring strong logical capabilities
- GPT-3.5-Turbo: Efficient for straightforward tasks, chat, and basic content generation
- Claude-3-Opus: Excels at research, technical writing, and in-depth analysis with strong reasoning
- Claude-3-Sonnet: Well-balanced for creative writing and nuanced responses
- Gemini Pro: Strong at multimodal tasks and code generation
- Mistral Large: Versatile open source model good for general tasks

Provider Considerations:
- OpenAI: Industry standard with consistent performance
- Anthropic: Known for safety and detailed analysis
- Google: Strong technical capabilities and multimodal support
- Groq: Optimized for high-speed inference
- Mistral: Balance of open source and commercial offerings

Data:
{data}

When Making Recommendations Consider:
1. Task requirements (complexity, creativity, technical needs)
2. Performance characteristics needed (speed, accuracy, reliability)
3. Special capabilities required (code generation, analysis, etc)
4. Cost and efficiency tradeoffs
5. Provider-specific strengths and limitations

Provide clear rationale for your model selection based on the specific task requirements.
"""


class ModelRouter:
    """
    A router class that intelligently selects and executes AI models based on task requirements.

    The ModelRouter uses a function calling model to analyze tasks and recommend the optimal
    model and provider combination, then executes the task using the selected model.

    Attributes:
        system_prompt (str): Prompt that guides model selection behavior
        max_tokens (int): Maximum tokens for model outputs
        temperature (float): Temperature parameter for model randomness
        max_workers (int): Maximum concurrent workers for batch processing
        model_output (ModelOutput): Pydantic model for structured outputs
        model_caller (OpenAIFunctionCaller): Function calling interface
    """

    def __init__(
        self,
        system_prompt: str = model_router_system_prompt,
        max_tokens: int = 4000,
        temperature: float = 0.5,
        max_workers: int = 10,
        api_key: str = None,
        max_loops: int = 1,
        *args,
        **kwargs,
    ):
        """
        Initialize the ModelRouter.

        Args:
            system_prompt (str): Prompt for model selection guidance
            max_tokens (int): Maximum output tokens
            temperature (float): Model temperature parameter
            max_workers (int): Max concurrent workers
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
        """
        try:
            self.system_prompt = system_prompt
            self.max_tokens = max_tokens
            self.temperature = temperature
            self.max_workers = max_workers
            self.model_output = ModelOutput
            self.max_loops = max_loops

            if self.max_workers == "auto":
                self.max_workers = os.cpu_count()

            self.model_caller = OpenAIFunctionCaller(
                base_model=ModelOutput,
                temperature=self.temperature,
                system_prompt=self.system_prompt,
                api_key=api_key,
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to initialize ModelRouter: {str(e)}"
            )

    def step(self, task: str):
        """
        Run a single task through the model router.

        Args:
            task (str): The task to be executed

        Returns:
            str: The model's output for the task

        Raises:
            RuntimeError: If model selection or execution fails
        """
        model_router_output = self.model_caller.run(task)

        selected_model = model_router_output.model
        selected_provider = model_router_output.provider
        routed_task = model_router_output.task
        rationale = model_router_output.rationale
        max_tokens = model_router_output.max_tokens
        temperature = model_router_output.temperature
        system_prompt = model_router_output.system_prompt

        formatter.print_panel(
            f"Model: {selected_model}\n\n"
            f"Provider: {selected_provider}\n\n"
            f"Task: {routed_task}\n\n"
            f"Rationale: {rationale}\n\n"
            f"Max Tokens: {max_tokens}\n\n"
            f"Temperature: {temperature}\n\n"
            f"System Prompt: {system_prompt}",
            title="Model Router Output",
        )

        litellm_wrapper = LiteLLM(
            model_name=f"{selected_provider}/{selected_model}",
            max_tokens=max_tokens,
            temperature=temperature,
            system_prompt=system_prompt,
        )

        final_output = litellm_wrapper.run(task=routed_task)

        formatter.print_panel(
            f"Output: {final_output} from {selected_provider}/{selected_model}",
            title=f"Model: {selected_model} Provider: {selected_provider}",
        )

        return final_output

    def run(self, task: str):
        """
        Run a task through the model router with memory.
        """
        task_output = task
        previous_output = None
        for _ in range(self.max_loops):
            if task_output == previous_output:
                break  # Exit if no change in output
            previous_output = task_output
            task_output = self.step(task_output)
        return task_output

    def batch_run(self, tasks: list):
        """
        Run multiple tasks in sequence.

        Args:
            tasks (list): List of tasks to execute

        Returns:
            list: List of outputs for each task

        Raises:
            RuntimeError: If batch execution fails
        """
        try:
            outputs = []
            for task in tasks:
                output = self.run(task)
                outputs.append(output)
            return outputs
        except Exception as e:
            raise RuntimeError(f"Batch execution failed: {str(e)}")

    def __call__(self, task: str, *args, **kwargs):
        """
        Make the class callable to directly execute tasks.

        Args:
            task (str): Task to execute

        Returns:
            str: Model output
        """
        return self.run(task, *args, **kwargs)

    def __batch_call__(self, tasks: list):
        """
        Make the class callable for batch execution.

        Args:
            tasks (list): List of tasks

        Returns:
            list: List of outputs
        """
        return self.batch_run(tasks)

    def __str__(self):
        return f"ModelRouter(max_tokens={self.max_tokens}, temperature={self.temperature})"

    def __repr__(self):
        return f"ModelRouter(max_tokens={self.max_tokens}, temperature={self.temperature})"

    def concurrent_run(self, tasks: list):
        """
        Run multiple tasks concurrently using a thread pool.

        Args:
            tasks (list): List of tasks to execute concurrently

        Returns:
            list: List of outputs from all tasks

        Raises:
            RuntimeError: If concurrent execution fails
        """
        try:
            with ThreadPoolExecutor(
                max_workers=self.max_workers
            ) as executor:
                outputs = list(executor.map(self.run, tasks))
            return outputs
        except Exception as e:
            raise RuntimeError(
                f"Concurrent execution failed: {str(e)}"
            )

    async def async_run(self, task: str, *args, **kwargs):
        """
        Run a task asynchronously.

        Args:
            task (str): Task to execute asynchronously

        Returns:
            asyncio.Task: Async task object

        Raises:
            RuntimeError: If async execution fails
        """
        try:
            return asyncio.create_task(
                self.run(task, *args, **kwargs)
            )
        except Exception as e:
            raise RuntimeError(f"Async execution failed: {str(e)}")
