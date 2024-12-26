import asyncio
import json
import logging
import os
import random
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Tuple,
    Union,
)

import toml
import yaml
from loguru import logger
from pydantic import BaseModel
from swarm_models.tiktoken_wrapper import TikTokenizer

from swarms.agents.ape_agent import auto_generate_prompt
from swarms.artifacts.main_artifact import Artifact
from swarms.prompts.agent_system_prompts import AGENT_SYSTEM_PROMPT_3
from swarms.prompts.multi_modal_autonomous_instruction_prompt import (
    MULTI_MODAL_AUTO_AGENT_SYSTEM_PROMPT_1,
)
from swarms.prompts.tools import tool_sop_prompt
from swarms.schemas.agent_step_schemas import ManySteps, Step
from swarms.schemas.base_schemas import (
    AgentChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessageResponse,
)
from swarms.structs.concat import concat_strings
from swarms.structs.conversation import Conversation
from swarms.structs.safe_loading import (
    SafeLoaderUtils,
    SafeStateManager,
)
from swarms.tools.base_tool import BaseTool
from swarms.tools.tool_parse_exec import parse_and_execute_json
from swarms.utils.data_to_text import data_to_text
from swarms.utils.file_processing import create_file_in_folder
from swarms.utils.formatter import formatter
from swarms.utils.pdf_to_text import pdf_to_text
from swarms.utils.wrapper_clusterop import (
    exec_callable_with_clusterops,
)


# Utils
# Custom stopping condition
def stop_when_repeats(response: str) -> bool:
    # Stop if the word stop appears in the response
    return "stop" in response.lower()


# Parse done token
def parse_done_token(response: str) -> bool:
    """Parse the response to see if the done token is present"""
    return "<DONE>" in response


# Agent ID generator
def agent_id():
    """Generate an agent id"""
    return uuid.uuid4().hex


def exists(val):
    return val is not None


# Agent output types
# agent_output_type = Union[BaseModel, dict, str]
agent_output_type = Literal[
    "string", "str", "list", "json", "dict", "yaml", "json_schema"
]
ToolUsageType = Union[BaseModel, Dict[str, Any]]


# [FEAT][AGENT]
class Agent:
    """
    Agent is the backbone to connect LLMs with tools and long term memory. Agent also provides the ability to
    ingest any type of docs like PDFs, Txts, Markdown, Json, and etc for the agent. Here is a list of features.

    Args:
        llm (Any): The language model to use
        template (str): The template to use
        max_loops (int): The maximum number of loops to run
        stopping_condition (Callable): The stopping condition to use
        loop_interval (int): The loop interval
        retry_attempts (int): The number of retry attempts
        retry_interval (int): The retry interval
        return_history (bool): Return the history
        stopping_token (str): The stopping token
        dynamic_loops (bool): Enable dynamic loops
        interactive (bool): Enable interactive mode
        dashboard (bool): Enable dashboard
        agent_name (str): The name of the agent
        agent_description (str): The description of the agent
        system_prompt (str): The system prompt
        tools (List[BaseTool]): The tools to use
        dynamic_temperature_enabled (bool): Enable dynamic temperature
        sop (str): The standard operating procedure
        sop_list (List[str]): The standard operating procedure list
        saved_state_path (str): The path to the saved state
        autosave (bool): Autosave the state
        context_length (int): The context length
        user_name (str): The user name
        self_healing_enabled (bool): Enable self healing
        code_interpreter (bool): Enable code interpreter
        multi_modal (bool): Enable multimodal
        pdf_path (str): The path to the pdf
        list_of_pdf (str): The list of pdf
        tokenizer (Any): The tokenizer
        long_term_memory (BaseVectorDatabase): The long term memory
        preset_stopping_token (bool): Enable preset stopping token
        traceback (Any): The traceback
        traceback_handlers (Any): The traceback handlers
        streaming_on (bool): Enable streaming
        docs (List[str]): The list of documents
        docs_folder (str): The folder containing the documents
        verbose (bool): Enable verbose mode
        parser (Callable): The parser to use
        best_of_n (int): The number of best responses to return
        callback (Callable): The callback function
        metadata (Dict[str, Any]): The metadata
        callbacks (List[Callable]): The list of callback functions
        search_algorithm (Callable): The search algorithm
        logs_to_filename (str): The filename for the logs
        evaluator (Callable): The evaluator function
        stopping_func (Callable): The stopping function
        custom_loop_condition (Callable): The custom loop condition
        sentiment_threshold (float): The sentiment threshold
        custom_exit_command (str): The custom exit command
        sentiment_analyzer (Callable): The sentiment analyzer
        limit_tokens_from_string (Callable): The function to limit tokens from a string
        custom_tools_prompt (Callable): The custom tools prompt
        tool_schema (ToolUsageType): The tool schema
        output_type (agent_output_type): The output type
        function_calling_type (str): The function calling type
        output_cleaner (Callable): The output cleaner function
        function_calling_format_type (str): The function calling format type
        list_base_models (List[BaseModel]): The list of base models
        metadata_output_type (str): The metadata output type
        state_save_file_type (str): The state save file type
        chain_of_thoughts (bool): Enable chain of thoughts
        algorithm_of_thoughts (bool): Enable algorithm of thoughts
        tree_of_thoughts (bool): Enable tree of thoughts
        tool_choice (str): The tool choice
        execute_tool (bool): Enable tool execution
        rules (str): The rules
        planning (str): The planning
        planning_prompt (str): The planning prompt
        device (str): The device
        custom_planning_prompt (str): The custom planning prompt
        memory_chunk_size (int): The memory chunk size
        agent_ops_on (bool): Enable agent operations
        log_directory (str): The log directory
        tool_system_prompt (str): The tool system prompt
        max_tokens (int): The maximum number of tokens
        frequency_penalty (float): The frequency penalty
        presence_penalty (float): The presence penalty
        temperature (float): The temperature
        workspace_dir (str): The workspace directory
        timeout (int): The timeout
        artifacts_on (bool): Enable artifacts
        artifacts_output_path (str): The artifacts output path
        artifacts_file_extension (str): The artifacts file extension (.pdf, .md, .txt, )
        scheduled_run_date (datetime): The date and time to schedule the task

    Methods:
        run: Run the agent
        run_concurrent: Run the agent concurrently
        bulk_run: Run the agent in bulk
        save: Save the agent
        load: Load the agent
        validate_response: Validate the response
        print_history_and_memory: Print the history and memory
        step: Step through the agent
        graceful_shutdown: Gracefully shutdown the agent
        run_with_timeout: Run the agent with a timeout
        analyze_feedback: Analyze the feedback
        undo_last: Undo the last response
        add_response_filter: Add a response filter
        apply_response_filters: Apply the response filters
        filtered_run: Run the agent with filtered responses
        interactive_run: Run the agent in interactive mode
        streamed_generation: Stream the generation of the response
        save_state: Save the state
        truncate_history: Truncate the history
        add_task_to_memory: Add the task to the memory
        print_dashboard: Print the dashboard
        loop_count_print: Print the loop count
        streaming: Stream the content
        _history: Generate the history
        _dynamic_prompt_setup: Setup the dynamic prompt
        run_async: Run the agent asynchronously
        run_async_concurrent: Run the agent asynchronously and concurrently
        run_async_concurrent: Run the agent asynchronously and concurrently
        construct_dynamic_prompt: Construct the dynamic prompt
        handle_artifacts: Handle artifacts


    Examples:
    >>> from swarm_models import OpenAIChat
    >>> from swarms.structs import Agent
    >>> llm = OpenAIChat()
    >>> agent = Agent(llm=llm, max_loops=1)
    >>> response = agent.run("Generate a report on the financials.")
    >>> print(response)
    >>> # Generate a report on the financials.

    """

    def __init__(
        self,
        agent_id: Optional[str] = agent_id(),
        id: Optional[str] = agent_id(),
        llm: Optional[Any] = None,
        template: Optional[str] = None,
        max_loops: Optional[int] = 1,
        stopping_condition: Optional[Callable[[str], bool]] = None,
        loop_interval: Optional[int] = 0,
        retry_attempts: Optional[int] = 3,
        retry_interval: Optional[int] = 1,
        return_history: Optional[bool] = False,
        stopping_token: Optional[str] = None,
        dynamic_loops: Optional[bool] = False,
        interactive: Optional[bool] = False,
        dashboard: Optional[bool] = False,
        agent_name: Optional[str] = "swarm-worker-01",
        agent_description: Optional[str] = None,
        system_prompt: Optional[str] = AGENT_SYSTEM_PROMPT_3,
        # TODO: Change to callable, then parse the callable to a string
        tools: List[Callable] = None,
        dynamic_temperature_enabled: Optional[bool] = False,
        sop: Optional[str] = None,
        sop_list: Optional[List[str]] = None,
        saved_state_path: Optional[str] = None,
        autosave: Optional[bool] = False,
        context_length: Optional[int] = 8192,
        user_name: Optional[str] = "Human:",
        self_healing_enabled: Optional[bool] = False,
        code_interpreter: Optional[bool] = False,
        multi_modal: Optional[bool] = None,
        pdf_path: Optional[str] = None,
        list_of_pdf: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        long_term_memory: Optional[Any] = None,
        preset_stopping_token: Optional[bool] = False,
        traceback: Optional[Any] = None,
        traceback_handlers: Optional[Any] = None,
        streaming_on: Optional[bool] = False,
        docs: List[str] = None,
        docs_folder: Optional[str] = None,
        verbose: Optional[bool] = False,
        parser: Optional[Callable] = None,
        best_of_n: Optional[int] = None,
        callback: Optional[Callable] = None,
        metadata: Optional[Dict[str, Any]] = None,
        callbacks: Optional[List[Callable]] = None,
        search_algorithm: Optional[Callable] = None,
        logs_to_filename: Optional[str] = None,
        evaluator: Optional[Callable] = None,  # Custom LLM or agent
        stopping_func: Optional[Callable] = None,
        custom_loop_condition: Optional[Callable] = None,
        sentiment_threshold: Optional[
            float
        ] = None,  # Evaluate on output using an external model
        custom_exit_command: Optional[str] = "exit",
        sentiment_analyzer: Optional[Callable] = None,
        limit_tokens_from_string: Optional[Callable] = None,
        # [Tools]
        custom_tools_prompt: Optional[Callable] = None,
        tool_schema: ToolUsageType = None,
        output_type: agent_output_type = "str",
        function_calling_type: str = "json",
        output_cleaner: Optional[Callable] = None,
        function_calling_format_type: Optional[str] = "OpenAI",
        list_base_models: Optional[List[BaseModel]] = None,
        metadata_output_type: str = "json",
        state_save_file_type: str = "json",
        chain_of_thoughts: bool = False,
        algorithm_of_thoughts: bool = False,
        tree_of_thoughts: bool = False,
        tool_choice: str = "auto",
        rules: str = None,  # type: ignore
        planning: Optional[str] = False,
        planning_prompt: Optional[str] = None,
        custom_planning_prompt: str = None,
        memory_chunk_size: int = 2000,
        agent_ops_on: bool = False,
        log_directory: str = None,
        tool_system_prompt: str = tool_sop_prompt(),
        max_tokens: int = 4096,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        temperature: float = 0.1,
        workspace_dir: str = "agent_workspace",
        timeout: Optional[int] = None,
        # short_memory: Optional[str] = None,
        created_at: float = time.time(),
        return_step_meta: Optional[bool] = False,
        tags: Optional[List[str]] = None,
        use_cases: Optional[List[Dict[str, str]]] = None,
        step_pool: List[Step] = [],
        print_every_step: Optional[bool] = False,
        time_created: Optional[str] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime()
        ),
        agent_output: ManySteps = None,
        executor_workers: int = os.cpu_count(),
        data_memory: Optional[Callable] = None,
        load_yaml_path: str = None,
        auto_generate_prompt: bool = False,
        rag_every_loop: bool = False,
        plan_enabled: bool = False,
        artifacts_on: bool = False,
        artifacts_output_path: str = None,
        artifacts_file_extension: str = None,
        device: str = "cpu",
        all_cores: bool = True,
        device_id: int = 0,
        scheduled_run_date: Optional[datetime] = None,
        do_not_use_cluster_ops: bool = True,
        all_gpus: bool = False,
        model_name: str = None,
        llm_args: dict = None,
        load_state_path: str = None,
        *args,
        **kwargs,
    ):
        # super().__init__(*args, **kwargs)
        self.agent_id = agent_id
        self.id = id
        self.llm = llm
        self.template = template
        self.max_loops = max_loops
        self.stopping_condition = stopping_condition
        self.loop_interval = loop_interval
        self.retry_attempts = retry_attempts
        self.retry_interval = retry_interval
        self.task = None
        self.stopping_token = stopping_token
        self.interactive = interactive
        self.dashboard = dashboard
        self.return_history = return_history
        self.dynamic_temperature_enabled = dynamic_temperature_enabled
        self.dynamic_loops = dynamic_loops
        self.user_name = user_name
        self.context_length = context_length
        self.sop = sop
        self.sop_list = sop_list
        self.tools = tools
        self.system_prompt = system_prompt
        self.agent_name = agent_name
        self.agent_description = agent_description
        self.saved_state_path = f"{self.agent_name}_state.json"
        self.autosave = autosave
        self.response_filters = []
        self.self_healing_enabled = self_healing_enabled
        self.code_interpreter = code_interpreter
        self.multi_modal = multi_modal
        self.pdf_path = pdf_path
        self.list_of_pdf = list_of_pdf
        self.tokenizer = tokenizer
        self.long_term_memory = long_term_memory
        self.preset_stopping_token = preset_stopping_token
        self.traceback = traceback
        self.traceback_handlers = traceback_handlers
        self.streaming_on = streaming_on
        self.docs = docs
        self.docs_folder = docs_folder
        self.verbose = verbose
        self.parser = parser
        self.best_of_n = best_of_n
        self.callback = callback
        self.metadata = metadata
        self.callbacks = callbacks
        self.search_algorithm = search_algorithm
        self.logs_to_filename = logs_to_filename
        self.evaluator = evaluator
        self.stopping_func = stopping_func
        self.custom_loop_condition = custom_loop_condition
        self.sentiment_threshold = sentiment_threshold
        self.custom_exit_command = custom_exit_command
        self.sentiment_analyzer = sentiment_analyzer
        self.limit_tokens_from_string = limit_tokens_from_string
        self.tool_schema = tool_schema
        self.output_type = output_type
        self.function_calling_type = function_calling_type
        self.output_cleaner = output_cleaner
        self.function_calling_format_type = (
            function_calling_format_type
        )
        self.list_base_models = list_base_models
        self.metadata_output_type = metadata_output_type
        self.state_save_file_type = state_save_file_type
        self.chain_of_thoughts = chain_of_thoughts
        self.algorithm_of_thoughts = algorithm_of_thoughts
        self.tree_of_thoughts = tree_of_thoughts
        self.tool_choice = tool_choice
        self.planning = planning
        self.planning_prompt = planning_prompt
        self.custom_planning_prompt = custom_planning_prompt
        self.rules = rules
        self.custom_tools_prompt = custom_tools_prompt
        self.memory_chunk_size = memory_chunk_size
        self.agent_ops_on = agent_ops_on
        self.log_directory = log_directory
        self.tool_system_prompt = tool_system_prompt
        self.max_tokens = max_tokens
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.temperature = temperature
        self.workspace_dir = workspace_dir
        self.timeout = timeout
        self.created_at = created_at
        self.return_step_meta = return_step_meta
        self.tags = tags
        self.use_cases = use_cases
        self.name = agent_name
        self.description = agent_description
        self.agent_output = agent_output
        self.step_pool = step_pool
        self.print_every_step = print_every_step
        self.time_created = time_created
        self.data_memory = data_memory
        self.load_yaml_path = load_yaml_path
        self.tokenizer = TikTokenizer()
        self.auto_generate_prompt = auto_generate_prompt
        self.rag_every_loop = rag_every_loop
        self.plan_enabled = plan_enabled
        self.artifacts_on = artifacts_on
        self.artifacts_output_path = artifacts_output_path
        self.artifacts_file_extension = artifacts_file_extension
        self.device = device
        self.all_cores = all_cores
        self.device_id = device_id
        self.scheduled_run_date = scheduled_run_date
        self.do_not_use_cluster_ops = do_not_use_cluster_ops
        self.all_gpus = all_gpus
        self.model_name = model_name
        self.llm_args = llm_args
        self.load_state_path = load_state_path

        # Initialize the short term memory
        self.short_memory = Conversation(
            system_prompt=system_prompt,
            time_enabled=True,
            user=user_name,
            rules=rules,
            *args,
            **kwargs,
        )

        # Initialize the feedback
        self.feedback = []

        # Initialize the executor
        self.executor = ThreadPoolExecutor(
            max_workers=executor_workers
        )

        # Initialize the tool struct
        if (
            exists(tools)
            or exists(list_base_models)
            or exists(tool_schema)
        ):

            self.tool_struct = BaseTool(
                tools=tools,
                base_models=list_base_models,
                tool_system_prompt=tool_system_prompt,
            )

        # The max_loops will be set dynamically if the dynamic_loop
        if self.dynamic_loops is True:
            logger.info("Dynamic loops enabled")
            self.max_loops = "auto"

        # If multimodal = yes then set the sop to the multimodal sop
        if self.multi_modal is True:
            self.sop = MULTI_MODAL_AUTO_AGENT_SYSTEM_PROMPT_1

        # If the preset stopping token is enabled then set the stopping token to the preset stopping token
        if preset_stopping_token is not None:
            self.stopping_token = "<DONE>"

        # If the docs exist then ingest the docs
        if exists(self.docs):
            threading.Thread(
                target=self.ingest_docs, args=(self.docs)
            ).start()

        # If docs folder exists then get the docs from docs folder
        if exists(self.docs_folder):
            threading.Thread(
                target=self.get_docs_from_doc_folders
            ).start()

        if tools is not None:
            logger.info(
                "Tools provided make sure the functions have documentation ++ type hints, otherwise tool execution won't be reliable."
            )
            # Add the tool prompt to the memory
            self.short_memory.add(
                role="system", content=tool_system_prompt
            )

            # Log the tools
            logger.info(
                f"Tools provided: Accessing {len(tools)} tools"
            )

            # Transform the tools into an openai schema
            # self.convert_tool_into_openai_schema()

            # Transform the tools into an openai schema
            tool_dict = (
                self.tool_struct.convert_tool_into_openai_schema()
            )
            self.short_memory.add(role="system", content=tool_dict)

            # Now create a function calling map for every tools
            self.function_map = {
                tool.__name__: tool for tool in tools
            }

        # If the tool schema exists or a list of base models exists then convert the tool schema into an openai schema
        if exists(tool_schema) or exists(list_base_models):
            threading.Thread(
                target=self.handle_tool_schema_ops()
            ).start()

        # If the sop or sop_list exists then handle the sop ops
        if exists(self.sop) or exists(self.sop_list):
            threading.Thread(target=self.handle_sop_ops()).start()

        # If agent_ops is on => activate agentops
        if agent_ops_on is True:
            threading.Thread(target=self.activate_agentops()).start()

        # Many steps
        self.agent_output = ManySteps(
            agent_id=agent_id,
            agent_name=agent_name,
            # run_id=run_id,
            task="",
            max_loops=self.max_loops,
            steps=self.short_memory.to_dict(),
            full_history=self.short_memory.get_str(),
            total_tokens=self.tokenizer.count_tokens(
                self.short_memory.get_str()
            ),
            stopping_token=self.stopping_token,
            interactive=self.interactive,
            dynamic_temperature_enabled=self.dynamic_temperature_enabled,
        )

        # Telemetry Processor to log agent data
        threading.Thread(target=self.log_agent_data).start()

        if self.llm is None and self.model_name is not None:
            self.llm = self.llm_handling()

    def llm_handling(self):
        from swarms.utils.litellm_wrapper import LiteLLM

        if self.llm_args is not None:
            llm = LiteLLM(model_name=self.model_name, **self.llm_args)

        else:
            llm = LiteLLM(
                model_name=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

        return llm

    def check_if_no_prompt_then_autogenerate(self, task: str = None):
        """
        Checks if auto_generate_prompt is enabled and generates a prompt by combining agent name, description and system prompt if available.
        Falls back to task if all other fields are missing.

        Args:
            task (str, optional): The task to use as a fallback if name, description and system prompt are missing. Defaults to None.
        """
        if self.auto_generate_prompt is True:
            # Collect all available prompt components
            components = []

            if self.agent_name:
                components.append(self.agent_name)

            if self.agent_description:
                components.append(self.agent_description)

            if self.system_prompt:
                components.append(self.system_prompt)

            # If no components available, fall back to task
            if not components and task:
                logger.warning(
                    "No agent details found. Using task as fallback for prompt generation."
                )
                self.system_prompt = auto_generate_prompt(
                    task, self.llm
                )
            else:
                # Combine all available components
                combined_prompt = " ".join(components)
                logger.info(
                    f"Auto-generating prompt from: {', '.join(components)}"
                )
                self.system_prompt = auto_generate_prompt(
                    combined_prompt, self.llm
                )
                self.short_memory.add(
                    role="system", content=self.system_prompt
                )

            logger.info("Auto-generated prompt successfully.")

    def set_system_prompt(self, system_prompt: str):
        """Set the system prompt"""
        self.system_prompt = system_prompt

    def provide_feedback(self, feedback: str) -> None:
        """Allow users to provide feedback on the responses."""
        self.feedback.append(feedback)
        logging.info(f"Feedback received: {feedback}")

    def agent_initialization(self):
        try:
            logger.info(
                f"Initializing Autonomous Agent {self.agent_name}..."
            )
            self.check_parameters()
            logger.info(
                f"{self.agent_name} Initialized Successfully."
            )
            logger.info(
                f"Autonomous Agent {self.agent_name} Activated, all systems operational. Executing task..."
            )

            if self.dashboard is True:
                self.print_dashboard()

        except ValueError as e:
            logger.info(f"Error initializing agent: {e}")
            raise e

    def _check_stopping_condition(self, response: str) -> bool:
        """Check if the stopping condition is met."""
        try:
            if self.stopping_condition:
                return self.stopping_condition(response)
            return False
        except Exception as error:
            logger.error(
                f"Error checking stopping condition: {error}"
            )

    def dynamic_temperature(self):
        """
        1. Check the self.llm object for the temperature
        2. If the temperature is not present, then use the default temperature
        3. If the temperature is present, then dynamically change the temperature
        4. for every loop you can randomly change the temperature on a scale from 0.0 to 1.0
        """
        try:
            if hasattr(self.llm, "temperature"):
                # Randomly change the temperature attribute of self.llm object
                self.llm.temperature = random.uniform(0.0, 1.0)
            else:
                # Use a default temperature
                self.llm.temperature = 0.5
        except Exception as error:
            logger.error(
                f"Error dynamically changing temperature: {error}"
            )

    def print_dashboard(self):
        """Print dashboard"""
        formatter.print_panel(
            f"Initializing Agent: {self.agent_name}"
        )

        data = self.to_dict()

        # Beautify the data
        # data = json.dumps(data, indent=4)
        # json_data = json.dumps(data, indent=4)

        formatter.print_panel(
            f"""
            Agent Dashboard
            --------------------------------------------

            Agent {self.agent_name} is initializing for {self.max_loops} with the following configuration:
            ----------------------------------------

            Agent Configuration:
                Configuration: {data}

            ----------------------------------------
        """,
        )

    def loop_count_print(
        self, loop_count: int, max_loops: int
    ) -> None:
        """loop_count_print summary

        Args:
            loop_count (_type_): _description_
            max_loops (_type_): _description_
        """
        logger.info(f"\nLoop {loop_count} of {max_loops}")
        print("\n")

    # Check parameters
    def check_parameters(self):
        if self.llm is None:
            raise ValueError(
                "Language model is not provided. Choose a model from the available models in swarm_models or create a class with a run(task: str) method and or a __call__ method."
            )

        if self.max_loops is None or self.max_loops == 0:
            raise ValueError("Max loops is not provided")

        if self.max_tokens == 0 or self.max_tokens is None:
            raise ValueError("Max tokens is not provided")

        if self.context_length == 0 or self.context_length is None:
            raise ValueError("Context length is not provided")

    # Main function
    def _run(
        self,
        task: Optional[str] = None,
        img: Optional[str] = None,
        speech: Optional[str] = None,
        video: Optional[str] = None,
        is_last: Optional[bool] = False,
        print_task: Optional[bool] = False,
        generate_speech: Optional[bool] = False,
        *args,
        **kwargs,
    ) -> Any:
        """
        run the agent

        Args:
            task (str): The task to be performed.
            img (str): The image to be processed.
            is_last (bool): Indicates if this is the last task.

        Returns:
            Any: The output of the agent.
            (string, list, json, dict, yaml)

        Examples:
            agent(task="What is the capital of France?")
            agent(task="What is the capital of France?", img="path/to/image.jpg")
            agent(task="What is the capital of France?", img="path/to/image.jpg", is_last=True)
        """
        try:
            self.check_if_no_prompt_then_autogenerate(task)

            self.agent_output.task = task

            # Add task to memory
            self.short_memory.add(role=self.user_name, content=task)

            # Plan
            if self.plan_enabled is True:
                self.plan(task)

            # Set the loop count
            loop_count = 0
            # Clear the short memory
            response = None
            all_responses = []

            # Query the long term memory first for the context
            if self.long_term_memory is not None:
                self.memory_query(task)

            # Print the user's request

            if self.autosave:
                self.save()

            # Print the request
            if print_task is True:
                formatter.print_panel(
                    f"\n User: {task}",
                    f"Task Request for {self.agent_name}",
                )

            while (
                self.max_loops == "auto"
                or loop_count < self.max_loops
            ):
                loop_count += 1
                self.loop_count_print(loop_count, self.max_loops)
                print("\n")

                # Dynamic temperature
                if self.dynamic_temperature_enabled is True:
                    self.dynamic_temperature()

                # Task prompt
                task_prompt = (
                    self.short_memory.return_history_as_string()
                )

                # Parameters
                attempt = 0
                success = False
                while attempt < self.retry_attempts and not success:
                    try:
                        if (
                            self.long_term_memory is not None
                            and self.rag_every_loop is True
                        ):
                            logger.info(
                                "Querying RAG database for context..."
                            )
                            self.memory_query(task_prompt)

                        # Generate response using LLM
                        response_args = (
                            (task_prompt, *args)
                            if img is None
                            else (task_prompt, img, *args)
                        )
                        response = self.call_llm(
                            *response_args, **kwargs
                        )

                        # Convert to a str if the response is not a str
                        response = self.llm_output_parser(response)

                        # Print
                        if self.streaming_on is True:
                            # self.stream_response(response)
                            formatter.print_panel_token_by_token(
                                f"{self.agent_name}: {response}",
                                title=f"Agent Name: {self.agent_name} [Max Loops: {loop_count}]",
                            )
                        else:
                            # logger.info(f"Response: {response}")
                            formatter.print_panel(
                                f"{self.agent_name}: {response}",
                                f"Agent Name {self.agent_name} [Max Loops: {loop_count} ]",
                            )

                        # Check if response is a dictionary and has 'choices' key
                        if (
                            isinstance(response, dict)
                            and "choices" in response
                        ):
                            response = response["choices"][0][
                                "message"
                            ]["content"]
                        elif isinstance(response, str):
                            # If response is already a string, use it as is
                            pass
                        else:
                            raise ValueError(
                                f"Unexpected response format: {type(response)}"
                            )

                        # Check and execute tools
                        if self.tools is not None:
                            self.parse_and_execute_tools(response)

                        # Add the response to the memory
                        self.short_memory.add(
                            role=self.agent_name, content=response
                        )

                        # Add to all responses
                        all_responses.append(response)

                        # # TODO: Implement reliability check

                        if self.evaluator:
                            logger.info("Evaluating response...")
                            evaluated_response = self.evaluator(
                                response
                            )
                            print(
                                "Evaluated Response:"
                                f" {evaluated_response}"
                            )
                            self.short_memory.add(
                                role="Evaluator",
                                content=evaluated_response,
                            )

                        # Sentiment analysis
                        if self.sentiment_analyzer:
                            logger.info("Analyzing sentiment...")
                            self.sentiment_analysis_handler(response)

                        success = True  # Mark as successful to exit the retry loop

                    except Exception as e:

                        self.log_agent_data()

                        if self.autosave is True:
                            self.save()

                        logger.error(
                            f"Attempt {attempt+1}: Error generating"
                            f" response: {e}"
                        )
                        attempt += 1

                if not success:

                    self.log_agent_data()

                    if self.autosave is True:
                        self.save()

                    logger.error(
                        "Failed to generate a valid response after"
                        " retry attempts."
                    )
                    break  # Exit the loop if all retry attempts fail

                # Check stopping conditions
                if (
                    self.stopping_condition is not None
                    and self._check_stopping_condition(response)
                ):
                    logger.info("Stopping condition met.")
                    break
                elif (
                    self.stopping_func is not None
                    and self.stopping_func(response)
                ):
                    logger.info("Stopping function met.")
                    break

                if self.interactive:
                    logger.info("Interactive mode enabled.")
                    user_input = input("You: ")

                    # User-defined exit command
                    if (
                        user_input.lower()
                        == self.custom_exit_command.lower()
                    ):
                        print("Exiting as per user request.")
                        break

                    self.short_memory.add(
                        role=self.user_name, content=user_input
                    )

                if self.loop_interval:
                    logger.info(
                        f"Sleeping for {self.loop_interval} seconds"
                    )
                    time.sleep(self.loop_interval)

            if self.autosave is True:
                self.log_agent_data()

                if self.autosave is True:
                    self.save()

            # Apply the cleaner function to the response
            if self.output_cleaner is not None:
                logger.info("Applying output cleaner to response.")
                response = self.output_cleaner(response)
                logger.info(
                    f"Response after output cleaner: {response}"
                )
                self.short_memory.add(
                    role="Output Cleaner",
                    content=response,
                )

            if self.agent_ops_on is True and is_last is True:
                self.check_end_session_agentops()

            # Merge all responses
            all_responses = [
                response
                for response in all_responses
                if response is not None
            ]

            self.agent_output.steps = self.short_memory.to_dict()
            self.agent_output.full_history = (
                self.short_memory.get_str()
            )
            self.agent_output.total_tokens = (
                self.tokenizer.count_tokens(
                    self.short_memory.get_str()
                )
            )

            # Handle artifacts
            if self.artifacts_on is True:
                self.handle_artifacts(
                    concat_strings(all_responses),
                    self.artifacts_output_path,
                    self.artifacts_file_extension,
                )

            self.log_agent_data()
            if self.autosave is True:
                self.save()

            # More flexible output types
            if (
                self.output_type == "string"
                or self.output_type == "str"
            ):
                return concat_strings(all_responses)
            elif self.output_type == "list":
                return all_responses
            elif (
                self.output_type == "json"
                or self.return_step_meta is True
            ):
                return self.agent_output.model_dump_json(indent=4)
            elif self.output_type == "csv":
                return self.dict_to_csv(
                    self.agent_output.model_dump()
                )
            elif self.output_type == "dict":
                return self.agent_output.model_dump()
            elif self.output_type == "yaml":
                return yaml.safe_dump(
                    self.agent_output.model_dump(), sort_keys=False
                )
            elif self.return_history is True:
                history = self.short_memory.get_str()

                formatter.print_panel(
                    history, title=f"{self.agent_name} History"
                )
                return history
            else:
                raise ValueError(
                    f"Invalid output type: {self.output_type}"
                )

        except Exception as error:
            self._handle_run_error(error)

        except KeyboardInterrupt as error:
            self._handle_run_error(error)

    def _handle_run_error(self, error: any):
        self.log_agent_data()

        if self.autosave is True:
            self.save()

        logger.info(
            f"Error detected running your agent {self.agent_name} \n Error {error} \n Optimize your input parameters and or add an issue on the swarms github and contact our team on discord for support ;) "
        )
        raise error

    async def arun(
        self,
        task: Optional[str] = None,
        img: Optional[str] = None,
        is_last: bool = False,
        device: str = "cpu",  # gpu
        device_id: int = 1,
        all_cores: bool = True,
        do_not_use_cluster_ops: bool = True,
        all_gpus: bool = False,
        *args,
        **kwargs,
    ) -> Any:
        """
        Asynchronously runs the agent with the specified parameters.

        Args:
            task (Optional[str]): The task to be performed. Defaults to None.
            img (Optional[str]): The image to be processed. Defaults to None.
            is_last (bool): Indicates if this is the last task. Defaults to False.
            device (str): The device to use for execution. Defaults to "cpu".
            device_id (int): The ID of the GPU to use if device is set to "gpu". Defaults to 1.
            all_cores (bool): If True, uses all available CPU cores. Defaults to True.
            do_not_use_cluster_ops (bool): If True, does not use cluster operations. Defaults to True.
            all_gpus (bool): If True, uses all available GPUs. Defaults to False.
            *args: Additional positional arguments.
            **kwargs: Additional keyword arguments.

        Returns:
            Any: The result of the asynchronous operation.

        Raises:
            Exception: If an error occurs during the asynchronous operation.
        """
        try:
            return await asyncio.to_thread(
                self.run,
                task=task,
                img=img,
                is_last=is_last,
                device=device,
                device_id=device_id,
                all_cores=all_cores,
                do_not_use_cluster_ops=do_not_use_cluster_ops,
                all_gpus=all_gpus,
                *args,
                **kwargs,
            )
        except Exception as error:
            await self._handle_run_error(
                error
            )  # Ensure this is also async if needed

    def __call__(
        self,
        task: Optional[str] = None,
        img: Optional[str] = None,
        is_last: bool = False,
        device: str = "cpu",  # gpu
        device_id: int = 1,
        all_cores: bool = True,
        do_not_use_cluster_ops: bool = True,
        all_gpus: bool = False,
        *args,
        **kwargs,
    ) -> Any:
        """Call the agent

        Args:
            task (Optional[str]): The task to be performed. Defaults to None.
            img (Optional[str]): The image to be processed. Defaults to None.
            is_last (bool): Indicates if this is the last task. Defaults to False.
            device (str): The device to use for execution. Defaults to "cpu".
            device_id (int): The ID of the GPU to use if device is set to "gpu". Defaults to 0.
            all_cores (bool): If True, uses all available CPU cores. Defaults to True.
        """
        try:
            return self.run(
                task=task,
                img=img,
                is_last=is_last,
                device=device,
                device_id=device_id,
                all_cores=all_cores,
                do_not_use_cluster_ops=do_not_use_cluster_ops,
                all_gpus=all_gpus * args,
                **kwargs,
            )
        except Exception as error:
            self._handle_run_error(error)

    def dict_to_csv(self, data: dict) -> str:
        """
        Convert a dictionary to a CSV string.

        Args:
            data (dict): The dictionary to convert.

        Returns:
            str: The CSV string representation of the dictionary.
        """
        import csv
        import io

        output = io.StringIO()
        writer = csv.writer(output)

        # Write header
        writer.writerow(data.keys())

        # Write values
        writer.writerow(data.values())

        return output.getvalue()

    def parse_and_execute_tools(self, response: str, *args, **kwargs):
        try:
            logger.info("Executing tool...")

            # try to Execute the tool and return a string
            out = parse_and_execute_json(
                functions=self.tools,
                json_string=response,
                parse_md=True,
                *args,
                **kwargs,
            )

            out = str(out)

            logger.info(f"Tool Output: {out}")

            # Add the output to the memory
            self.short_memory.add(
                role="Tool Executor",
                content=out,
            )

        except Exception as error:
            logger.error(f"Error executing tool: {error}")
            raise error

    def add_memory(self, message: str):
        """Add a memory to the agent

        Args:
            message (str): _description_

        Returns:
            _type_: _description_
        """
        logger.info(f"Adding memory: {message}")

        return self.short_memory.add(
            role=self.agent_name, content=message
        )

    def plan(self, task: str, *args, **kwargs) -> None:
        """
        Plan the task

        Args:
            task (str): The task to plan
        """
        try:
            if exists(self.planning_prompt):
                # Join the plan and the task
                planning_prompt = f"{self.planning_prompt} {task}"
                plan = self.llm(planning_prompt, *args, **kwargs)
                logger.info(f"Plan: {plan}")

            # Add the plan to the memory
            self.short_memory.add(
                role=self.agent_name, content=str(plan)
            )

            return None
        except Exception as error:
            logger.error(f"Error planning task: {error}")
            raise error

    async def run_concurrent(self, task: str, *args, **kwargs):
        """
        Run a task concurrently.

        Args:
            task (str): The task to run.
        """
        try:
            logger.info(f"Running concurrent task: {task}")
            future = self.executor.submit(
                self.run, task, *args, **kwargs
            )
            result = await asyncio.wrap_future(future)
            logger.info(f"Completed task: {result}")
            return result
        except Exception as error:
            logger.error(
                f"Error running agent: {error} while running concurrently"
            )

    def run_concurrent_tasks(self, tasks: List[str], *args, **kwargs):
        """
        Run multiple tasks concurrently.

        Args:
            tasks (List[str]): A list of tasks to run.
        """
        try:
            logger.info(f"Running concurrent tasks: {tasks}")
            futures = [
                self.executor.submit(
                    self.run, task=task, *args, **kwargs
                )
                for task in tasks
            ]
            results = [future.result() for future in futures]
            logger.info(f"Completed tasks: {results}")
            return results
        except Exception as error:
            logger.error(f"Error running concurrent tasks: {error}")

    def bulk_run(self, inputs: List[Dict[str, Any]]) -> List[str]:
        """
        Generate responses for multiple input sets.

        Args:
            inputs (List[Dict[str, Any]]): A list of input dictionaries containing the necessary data for each run.

        Returns:
            List[str]: A list of response strings generated for each input set.

        Raises:
            Exception: If an error occurs while running the bulk tasks.
        """
        try:
            logger.info(f"Running bulk tasks: {inputs}")
            return [self.run(**input_data) for input_data in inputs]
        except Exception as error:
            logger.info(f"Error running bulk run: {error}", "red")

    async def arun_batched(
        self,
        tasks: List[str],
        *args,
        **kwargs,
    ):
        """Asynchronously runs a batch of tasks."""
        try:
            # Create a list of coroutines for each task
            coroutines = [
                self.arun(task=task, *args, **kwargs)
                for task in tasks
            ]
            # Use asyncio.gather to run them concurrently
            results = await asyncio.gather(*coroutines)
            return results
        except Exception as error:
            logger.error(f"Error running batched tasks: {error}")
            raise

    def save(self, file_path: str = None) -> None:
        """
        Save the agent state to a file using SafeStateManager with atomic writing
        and backup functionality. Automatically handles complex objects and class instances.

        Args:
            file_path (str, optional): Custom path to save the state.
                                    If None, uses configured paths.

        Raises:
            OSError: If there are filesystem-related errors
            Exception: For other unexpected errors
        """
        try:
            # Determine the save path
            resolved_path = (
                file_path
                or self.saved_state_path
                or f"{self.agent_name}_state.json"
            )

            # Ensure path has .json extension
            if not resolved_path.endswith(".json"):
                resolved_path += ".json"

            # Create full path including workspace directory
            full_path = os.path.join(
                self.workspace_dir, resolved_path
            )
            backup_path = full_path + ".backup"
            temp_path = full_path + ".temp"

            # Ensure workspace directory exists
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            # First save to temporary file using SafeStateManager
            SafeStateManager.save_state(self, temp_path)

            # If current file exists, create backup
            if os.path.exists(full_path):
                try:
                    os.replace(full_path, backup_path)
                except Exception as e:
                    logger.warning(f"Could not create backup: {e}")

            # Move temporary file to final location
            os.replace(temp_path, full_path)

            # Clean up old backup if everything succeeded
            if os.path.exists(backup_path):
                try:
                    os.remove(backup_path)
                except Exception as e:
                    logger.warning(
                        f"Could not remove backup file: {e}"
                    )

            # Log saved state information if verbose
            if self.verbose:
                self._log_saved_state_info(full_path)

            logger.info(
                f"Successfully saved agent state to: {full_path}"
            )

            # Handle additional component saves
            self._save_additional_components(full_path)

        except OSError as e:
            logger.error(
                f"Filesystem error while saving agent state: {e}"
            )
            raise
        except Exception as e:
            logger.error(f"Unexpected error saving agent state: {e}")
            raise

    def _save_additional_components(self, base_path: str) -> None:
        """Save additional agent components like memory."""
        try:
            # Save long term memory if it exists
            if (
                hasattr(self, "long_term_memory")
                and self.long_term_memory is not None
            ):
                memory_path = (
                    f"{os.path.splitext(base_path)[0]}_memory.json"
                )
                try:
                    self.long_term_memory.save(memory_path)
                    logger.info(
                        f"Saved long-term memory to: {memory_path}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not save long-term memory: {e}"
                    )

            # Save memory manager if it exists
            if (
                hasattr(self, "memory_manager")
                and self.memory_manager is not None
            ):
                manager_path = f"{os.path.splitext(base_path)[0]}_memory_manager.json"
                try:
                    self.memory_manager.save_memory_snapshot(
                        manager_path
                    )
                    logger.info(
                        f"Saved memory manager state to: {manager_path}"
                    )
                except Exception as e:
                    logger.warning(
                        f"Could not save memory manager: {e}"
                    )

        except Exception as e:
            logger.warning(f"Error saving additional components: {e}")

    def enable_autosave(self, interval: int = 300) -> None:
        """
        Enable automatic saving of agent state using SafeStateManager at specified intervals.

        Args:
            interval (int): Time between saves in seconds. Defaults to 300 (5 minutes).
        """

        def autosave_loop():
            while self.autosave:
                try:
                    self.save()
                    if self.verbose:
                        logger.debug(
                            f"Autosaved agent state (interval: {interval}s)"
                        )
                except Exception as e:
                    logger.error(f"Autosave failed: {e}")
                time.sleep(interval)

        self.autosave = True
        self.autosave_thread = threading.Thread(
            target=autosave_loop,
            daemon=True,
            name=f"{self.agent_name}_autosave",
        )
        self.autosave_thread.start()
        logger.info(f"Enabled autosave with {interval}s interval")

    def disable_autosave(self) -> None:
        """Disable automatic saving of agent state."""
        if hasattr(self, "autosave"):
            self.autosave = False
            if hasattr(self, "autosave_thread"):
                self.autosave_thread.join(timeout=1)
                delattr(self, "autosave_thread")
            logger.info("Disabled autosave")

    def cleanup(self) -> None:
        """Cleanup method to be called on exit. Ensures final state is saved."""
        try:
            if getattr(self, "autosave", False):
                logger.info(
                    "Performing final autosave before exit..."
                )
                self.disable_autosave()
                self.save()
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

    def load(self, file_path: str = None) -> None:
        """
        Load agent state from a file using SafeStateManager.
        Automatically preserves class instances and complex objects.

        Args:
            file_path (str, optional): Path to load state from.
                                    If None, uses default path from agent config.

        Raises:
            FileNotFoundError: If state file doesn't exist
            Exception: If there's an error during loading
        """
        try:
            # Resolve load path conditionally with a check for self.load_state_path
            resolved_path = (
                file_path
                or self.load_state_path
                or (
                    f"{self.saved_state_path}.json"
                    if self.saved_state_path
                    else (
                        f"{self.agent_name}.json"
                        if self.agent_name
                        else (
                            f"{self.workspace_dir}/{self.agent_name}_state.json"
                            if self.workspace_dir and self.agent_name
                            else None
                        )
                    )
                )
            )

            # Load state using SafeStateManager
            SafeStateManager.load_state(self, resolved_path)

            # Reinitialize any necessary runtime components
            self._reinitialize_after_load()

            if self.verbose:
                self._log_loaded_state_info(resolved_path)

        except FileNotFoundError:
            logger.error(f"State file not found: {resolved_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading agent state: {e}")
            raise

    def _reinitialize_after_load(self) -> None:
        """
        Reinitialize necessary components after loading state.
        Called automatically after load() to ensure all components are properly set up.
        """
        try:
            # Reinitialize conversation if needed
            if (
                not hasattr(self, "short_memory")
                or self.short_memory is None
            ):
                self.short_memory = Conversation(
                    system_prompt=self.system_prompt,
                    time_enabled=True,
                    user=self.user_name,
                    rules=self.rules,
                )

            # Reinitialize executor if needed
            if not hasattr(self, "executor") or self.executor is None:
                self.executor = ThreadPoolExecutor(
                    max_workers=os.cpu_count()
                )

            # # Reinitialize tool structure if needed
            # if hasattr(self, 'tools') and (self.tools or getattr(self, 'list_base_models', None)):
            #     self.tool_struct = BaseTool(
            #         tools=self.tools,
            #         base_models=getattr(self, 'list_base_models', None),
            #         tool_system_prompt=self.tool_system_prompt
            #     )

        except Exception as e:
            logger.error(f"Error reinitializing components: {e}")
            raise

    def _log_saved_state_info(self, file_path: str) -> None:
        """Log information about saved state for debugging"""
        try:
            state_dict = SafeLoaderUtils.create_state_dict(self)
            preserved = SafeLoaderUtils.preserve_instances(self)

            logger.info(f"Saved agent state to: {file_path}")
            logger.debug(
                f"Saved {len(state_dict)} configuration values"
            )
            logger.debug(
                f"Preserved {len(preserved)} class instances"
            )

            if self.verbose:
                logger.debug("Preserved instances:")
                for name, instance in preserved.items():
                    logger.debug(
                        f"  - {name}: {type(instance).__name__}"
                    )
        except Exception as e:
            logger.error(f"Error logging state info: {e}")

    def _log_loaded_state_info(self, file_path: str) -> None:
        """Log information about loaded state for debugging"""
        try:
            state_dict = SafeLoaderUtils.create_state_dict(self)
            preserved = SafeLoaderUtils.preserve_instances(self)

            logger.info(f"Loaded agent state from: {file_path}")
            logger.debug(
                f"Loaded {len(state_dict)} configuration values"
            )
            logger.debug(
                f"Preserved {len(preserved)} class instances"
            )

            if self.verbose:
                logger.debug("Current class instances:")
                for name, instance in preserved.items():
                    logger.debug(
                        f"  - {name}: {type(instance).__name__}"
                    )
        except Exception as e:
            logger.error(f"Error logging state info: {e}")

    def get_saveable_state(self) -> Dict[str, Any]:
        """
        Get a dictionary of all saveable state values.
        Useful for debugging or manual state inspection.

        Returns:
            Dict[str, Any]: Dictionary of saveable values
        """
        return SafeLoaderUtils.create_state_dict(self)

    def get_preserved_instances(self) -> Dict[str, Any]:
        """
        Get a dictionary of all preserved class instances.
        Useful for debugging or manual state inspection.

        Returns:
            Dict[str, Any]: Dictionary of preserved instances
        """
        return SafeLoaderUtils.preserve_instances(self)

    def graceful_shutdown(self):
        """Gracefully shutdown the system saving the state"""
        logger.info("Shutting down the system...")
        return self.save()

    def analyze_feedback(self):
        """Analyze the feedback for issues"""
        feedback_counts = {}
        for feedback in self.feedback:
            if feedback in feedback_counts:
                feedback_counts[feedback] += 1
            else:
                feedback_counts[feedback] = 1
        print(f"Feedback counts: {feedback_counts}")

    def undo_last(self) -> Tuple[str, str]:
        """
        Response the last response and return the previous state

        Example:
        # Feature 2: Undo functionality
        response = agent.run("Another task")
        print(f"Response: {response}")
        previous_state, message = agent.undo_last()
        print(message)

        """
        if len(self.short_memory) < 2:
            return None, None

        # Remove the last response but keep the last state, short_memory is a dict
        self.short_memory.delete(-1)

        # Get the previous state
        previous_state = self.short_memory[-1]
        return previous_state, f"Restored to {previous_state}"

    # Response Filtering
    def add_response_filter(self, filter_word: str) -> None:
        """
        Add a response filter to filter out certain words from the response

        Example:
        agent.add_response_filter("Trump")
        agent.run("Generate a report on Trump")


        """
        logger.info(f"Adding response filter: {filter_word}")
        self.reponse_filters.append(filter_word)

    def apply_reponse_filters(self, response: str) -> str:
        """
        Apply the response filters to the response

        """
        logger.info(
            f"Applying response filters to response: {response}"
        )
        for word in self.response_filters:
            response = response.replace(word, "[FILTERED]")
        return response

    def filtered_run(self, task: str) -> str:
        """
        # Feature 3: Response filtering
        agent.add_response_filter("report")
        response = agent.filtered_run("Generate a report on finance")
        print(response)
        """
        logger.info(f"Running filtered task: {task}")
        raw_response = self.run(task)
        return self.apply_response_filters(raw_response)

    def save_to_yaml(self, file_path: str) -> None:
        """
        Save the agent to a YAML file

        Args:
            file_path (str): The path to the YAML file
        """
        try:
            logger.info(f"Saving agent to YAML file: {file_path}")
            with open(file_path, "w") as f:
                yaml.dump(self.to_dict(), f)
        except Exception as error:
            logger.error(f"Error saving agent to YAML: {error}")
            raise error

    def get_llm_parameters(self):
        return str(vars(self.llm))

    def update_system_prompt(self, system_prompt: str):
        """Upddate the system message"""
        self.system_prompt = system_prompt

    def update_max_loops(self, max_loops: int):
        """Update the max loops"""
        self.max_loops = max_loops

    def update_loop_interval(self, loop_interval: int):
        """Update the loop interval"""
        self.loop_interval = loop_interval

    def update_retry_attempts(self, retry_attempts: int):
        """Update the retry attempts"""
        self.retry_attempts = retry_attempts

    def update_retry_interval(self, retry_interval: int):
        """Update the retry interval"""
        self.retry_interval = retry_interval

    def reset(self):
        """Reset the agent"""
        self.short_memory = None

    def ingest_docs(self, docs: List[str], *args, **kwargs):
        """Ingest the docs into the memory

        Args:
            docs (List[str]): Documents of pdfs, text, csvs

        Returns:
            None
        """
        try:
            for doc in docs:
                data = data_to_text(doc)

            return self.short_memory.add(
                role=self.user_name, content=data
            )
        except Exception as error:
            logger.info(f"Error ingesting docs: {error}", "red")

    def ingest_pdf(self, pdf: str):
        """Ingest the pdf into the memory

        Args:
            pdf (str): file path of pdf
        """
        try:
            logger.info(f"Ingesting pdf: {pdf}")
            text = pdf_to_text(pdf)
            return self.short_memory.add(
                role=self.user_name, content=text
            )
        except Exception as error:
            logger.info(f"Error ingesting pdf: {error}", "red")

    def receieve_message(self, name: str, message: str):
        """Receieve a message"""
        try:
            message = f"{name}: {message}"
            return self.short_memory.add(role=name, content=message)
        except Exception as error:
            logger.info(f"Error receiving message: {error}")
            raise error

    def send_agent_message(
        self, agent_name: str, message: str, *args, **kwargs
    ):
        """Send a message to the agent"""
        try:
            logger.info(f"Sending agent message: {message}")
            message = f"{agent_name}: {message}"
            return self.run(message, *args, **kwargs)
        except Exception as error:
            logger.info(f"Error sending agent message: {error}")
            raise error

    def add_tool(self, tool: Callable):
        """Add a single tool to the agent's tools list.

        Args:
            tool (Callable): The tool function to add

        Returns:
            The result of appending the tool to the tools list
        """
        logger.info(f"Adding tool: {tool.__name__}")
        return self.tools.append(tool)

    def add_tools(self, tools: List[Callable]):
        """Add multiple tools to the agent's tools list.

        Args:
            tools (List[Callable]): List of tool functions to add

        Returns:
            The result of extending the tools list
        """
        logger.info(f"Adding tools: {[t.__name__ for t in tools]}")
        return self.tools.extend(tools)

    def remove_tool(self, tool: Callable):
        """Remove a single tool from the agent's tools list.

        Args:
            tool (Callable): The tool function to remove

        Returns:
            The result of removing the tool from the tools list
        """
        logger.info(f"Removing tool: {tool.__name__}")
        return self.tools.remove(tool)

    def remove_tools(self, tools: List[Callable]):
        """Remove multiple tools from the agent's tools list.

        Args:
            tools (List[Callable]): List of tool functions to remove
        """
        logger.info(f"Removing tools: {[t.__name__ for t in tools]}")
        for tool in tools:
            self.tools.remove(tool)

    def get_docs_from_doc_folders(self):
        """Get the docs from the files"""
        try:
            logger.info("Getting docs from doc folders")
            # Get the list of files then extract them and add them to the memory
            files = os.listdir(self.docs_folder)

            # Extract the text from the files
            # Process each file and combine their contents
            all_text = ""
            for file in files:
                file_path = os.path.join(self.docs_folder, file)
                text = data_to_text(file_path)
                all_text += f"\nContent from {file}:\n{text}\n"

            # Add the combined content to memory
            return self.short_memory.add(
                role=self.user_name, content=all_text
            )
        except Exception as error:
            logger.error(
                f"Error getting docs from doc folders: {error}"
            )
            raise error

    def check_end_session_agentops(self):
        if self.agent_ops_on is True:
            try:
                from swarms.utils.agent_ops_check import (
                    end_session_agentops,
                )

                # Try ending the session
                return end_session_agentops()
            except ImportError:
                logger.error(
                    "Could not import agentops, try installing agentops: $ pip3 install agentops"
                )

    def memory_query(self, task: str = None, *args, **kwargs) -> None:
        try:
            # Query the long term memory
            if self.long_term_memory is not None:
                formatter.print_panel(f"Querying RAG for: {task}")

                memory_retrieval = self.long_term_memory.query(
                    task, *args, **kwargs
                )

                memory_retrieval = (
                    f"Documents Available: {str(memory_retrieval)}"
                )

                # # Count the tokens
                # memory_token_count = self.tokenizer.count_tokens(
                #     memory_retrieval
                # )
                # if memory_token_count > self.memory_chunk_size:
                #     # Truncate the memory by the memory chunk size
                #     memory_retrieval = self.truncate_string_by_tokens(
                #         memory_retrieval, self.memory_chunk_size
                #     )

                self.short_memory.add(
                    role="Database",
                    content=memory_retrieval,
                )

                return None
        except Exception as e:
            logger.error(f"An error occurred: {e}")
            raise e

    def sentiment_analysis_handler(self, response: str = None):
        """
        Performs sentiment analysis on the given response and stores the result in the short-term memory.

        Args:
            response (str): The response to analyze sentiment for.

        Returns:
            None
        """
        try:
            # Sentiment analysis
            if self.sentiment_analyzer:
                sentiment = self.sentiment_analyzer(response)
                print(f"Sentiment: {sentiment}")

                if sentiment > self.sentiment_threshold:
                    print(
                        f"Sentiment: {sentiment} is above"
                        " threshold:"
                        f" {self.sentiment_threshold}"
                    )
                elif sentiment < self.sentiment_threshold:
                    print(
                        f"Sentiment: {sentiment} is below"
                        " threshold:"
                        f" {self.sentiment_threshold}"
                    )

                self.short_memory.add(
                    role=self.agent_name,
                    content=sentiment,
                )
        except Exception as e:
            print(f"Error occurred during sentiment analysis: {e}")

    def stream_response(
        self, response: str, delay: float = 0.001
    ) -> None:
        """
        Streams the response token by token.

        Args:
            response (str): The response text to be streamed.
            delay (float, optional): Delay in seconds between printing each token. Default is 0.1 seconds.

        Raises:
            ValueError: If the response is not provided.
            Exception: For any errors encountered during the streaming process.

        Example:
            response = "This is a sample response from the API."
            stream_response(response)
        """
        # Check for required inputs
        if not response:
            raise ValueError("Response is required.")

        try:
            # Stream and print the response token by token
            for token in response.split():
                print(token, end=" ", flush=True)
                time.sleep(delay)
            print()  # Ensure a newline after streaming
        except Exception as e:
            print(f"An error occurred during streaming: {e}")

    def check_available_tokens(self):
        # Log the amount of tokens left in the memory and in the task
        if self.tokenizer is not None:
            tokens_used = self.tokenizer.count_tokens(
                self.short_memory.return_history_as_string()
            )
            logger.info(
                f"Tokens available: {self.context_length - tokens_used}"
            )

        return tokens_used

    def tokens_checks(self):
        # Check the tokens available
        tokens_used = self.tokenizer.count_tokens(
            self.short_memory.return_history_as_string()
        )
        out = self.check_available_tokens()

        logger.info(
            f"Tokens available: {out} Context Length: {self.context_length} Tokens in memory: {tokens_used}"
        )

        return out

    def parse_function_call_and_execute(self, response: str):
        """
        Parses a function call from the given response and executes it.

        Args:
            response (str): The response containing the function call.

        Returns:
            None

        Raises:
            Exception: If there is an error parsing and executing the function call.
        """
        try:
            if self.tools is not None:
                tool_call_output = parse_and_execute_json(
                    self.tools, response, parse_md=True
                )

                if tool_call_output is not str:
                    tool_call_output = str(tool_call_output)

                logger.info(f"Tool Call Output: {tool_call_output}")
                self.short_memory.add(
                    role=self.agent_name,
                    content=tool_call_output,
                )

                return tool_call_output
        except Exception as error:
            logger.error(
                f"Error parsing and executing function call: {error}"
            )

            # Raise a custom exception with the error message
            raise Exception(
                "Error parsing and executing function call"
            ) from error

    def activate_agentops(self):
        if self.agent_ops_on is True:
            try:
                from swarms.utils.agent_ops_check import (
                    try_import_agentops,
                )

                # Try importing agent ops
                logger.info(
                    "Agent Ops Initializing, ensure that you have the agentops API key and the pip package installed."
                )
                try_import_agentops()
                self.agent_ops_agent_name = self.agent_name

                logger.info("Agentops successfully activated!")
            except ImportError:
                logger.error(
                    "Could not import agentops, try installing agentops: $ pip3 install agentops"
                )

    def llm_output_parser(self, response: Any) -> str:
        """Parse the output from the LLM"""
        try:
            if isinstance(response, dict):
                if "choices" in response:
                    return response["choices"][0]["message"][
                        "content"
                    ]
                else:
                    return json.dumps(
                        response
                    )  # Convert dict to string
            elif isinstance(response, str):
                return response
            else:
                return str(
                    response
                )  # Convert any other type to string
        except Exception as e:
            logger.error(f"Error parsing LLM output: {e}")
            return str(
                response
            )  # Return string representation as fallback

    def log_step_metadata(
        self, loop: int, task: str, response: str
    ) -> Step:
        """Log metadata for each step of agent execution."""
        # Generate unique step ID
        step_id = f"step_{loop}_{uuid.uuid4().hex}"

        # Calculate token usage
        # full_memory = self.short_memory.return_history_as_string()
        # prompt_tokens = self.tokenizer.count_tokens(full_memory)
        # completion_tokens = self.tokenizer.count_tokens(response)
        # total_tokens = prompt_tokens + completion_tokens
        total_tokens = (
            self.tokenizer.count_tokens(task)
            + self.tokenizer.count_tokens(response),
        )

        # # Get memory responses
        # memory_responses = {
        #     "short_term": (
        #         self.short_memory.return_history_as_string()
        #         if self.short_memory
        #         else None
        #     ),
        #     "long_term": (
        #         self.long_term_memory.query(task)
        #         if self.long_term_memory
        #         else None
        #     ),
        # }

        # # Get tool responses if tool was used
        # if self.tools:
        #     try:
        #         tool_call_output = parse_and_execute_json(
        #             self.tools, response, parse_md=True
        #         )
        #         if tool_call_output:
        #             {
        #                 "tool_name": tool_call_output.get(
        #                     "tool_name", "unknown"
        #                 ),
        #                 "tool_args": tool_call_output.get("args", {}),
        #                 "tool_output": str(
        #                     tool_call_output.get("output", "")
        #                 ),
        #             }
        #     except Exception as e:
        #         logger.debug(
        #             f"No tool call detected in response: {e}"
        #         )

        # Create memory usage tracking
        # memory_usage = {
        #     "short_term": (
        #         len(self.short_memory.messages)
        #         if self.short_memory
        #         else 0
        #     ),
        #     "long_term": (
        #         self.long_term_memory.count
        #         if self.long_term_memory
        #         else 0
        #     ),
        #     "responses": memory_responses,
        # }

        step_log = Step(
            step_id=step_id,
            time=time.time(),
            tokens=total_tokens,
            response=AgentChatCompletionResponse(
                id=self.agent_id,
                agent_name=self.agent_name,
                object="chat.completion",
                choices=ChatCompletionResponseChoice(
                    index=loop,
                    input=task,
                    message=ChatMessageResponse(
                        role=self.agent_name,
                        content=response,
                    ),
                ),
                # usage=UsageInfo(
                #     prompt_tokens=prompt_tokens,
                #     completion_tokens=completion_tokens,
                #     total_tokens=total_tokens,
                # ),
                # tool_calls=(
                #     [] if tool_response is None else [tool_response]
                # ),
                # memory_usage=None,
            ),
        )

        # Update total tokens if agent_output exists
        # if hasattr(self, "agent_output"):
        #     self.agent_output.total_tokens += (
        #         self.response.total_tokens
        #     )

        # Add step to agent output tracking
        self.step_pool.append(step_log)

    def update_tool_usage(
        self,
        step_id: str,
        tool_name: str,
        tool_args: dict,
        tool_response: Any,
    ):
        """Update tool usage information for a specific step."""
        for step in self.agent_output.steps:
            if step.step_id == step_id:
                step.response.tool_calls.append(
                    {
                        "tool": tool_name,
                        "arguments": tool_args,
                        "response": str(tool_response),
                    }
                )
                break

    def _serialize_callable(
        self, attr_value: Callable
    ) -> Dict[str, Any]:
        """
        Serializes callable attributes by extracting their name and docstring.

        Args:
            attr_value (Callable): The callable to serialize.

        Returns:
            Dict[str, Any]: Dictionary with name and docstring of the callable.
        """
        return {
            "name": getattr(
                attr_value, "__name__", type(attr_value).__name__
            ),
            "doc": getattr(attr_value, "__doc__", None),
        }

    def _serialize_attr(self, attr_name: str, attr_value: Any) -> Any:
        """
        Serializes an individual attribute, handling non-serializable objects.

        Args:
            attr_name (str): The name of the attribute.
            attr_value (Any): The value of the attribute.

        Returns:
            Any: The serialized value of the attribute.
        """
        try:
            if callable(attr_value):
                return self._serialize_callable(attr_value)
            elif hasattr(attr_value, "to_dict"):
                return (
                    attr_value.to_dict()
                )  # Recursive serialization for nested objects
            else:
                json.dumps(
                    attr_value
                )  # Attempt to serialize to catch non-serializable objects
                return attr_value
        except (TypeError, ValueError):
            return f"<Non-serializable: {type(attr_value).__name__}>"

    def to_dict(self) -> Dict[str, Any]:
        """
        Converts all attributes of the class, including callables, into a dictionary.
        Handles non-serializable attributes by converting them or skipping them.

        Returns:
            Dict[str, Any]: A dictionary representation of the class attributes.
        """
        return {
            attr_name: self._serialize_attr(attr_name, attr_value)
            for attr_name, attr_value in self.__dict__.items()
        }

    def to_json(self, indent: int = 4, *args, **kwargs):
        return json.dumps(
            self.to_dict(), indent=indent, *args, **kwargs
        )

    def to_yaml(self, indent: int = 4, *args, **kwargs):
        return yaml.dump(
            self.to_dict(), indent=indent, *args, **kwargs
        )

    def to_toml(self, *args, **kwargs):
        return toml.dumps(self.to_dict(), *args, **kwargs)

    def model_dump_json(self):
        logger.info(
            f"Saving {self.agent_name} model to JSON in the {self.workspace_dir} directory"
        )

        create_file_in_folder(
            self.workspace_dir,
            f"{self.agent_name}.json",
            str(self.to_json()),
        )

        return f"Model saved to {self.workspace_dir}/{self.agent_name}.json"

    def model_dump_yaml(self):
        logger.info(
            f"Saving {self.agent_name} model to YAML in the {self.workspace_dir} directory"
        )

        create_file_in_folder(
            self.workspace_dir,
            f"{self.agent_name}.yaml",
            str(self.to_yaml()),
        )

        return f"Model saved to {self.workspace_dir}/{self.agent_name}.yaml"

    def log_agent_data(self):
        import requests

        data_dict = {"data": self.to_dict()}
        logger.info(f"was going to send to swarms { data_dict}")
        #        url = "https://swarms.world/api/get-agents/log-agents"
        #        headers = {
        #            "Content-Type": "application/json",
        #            "Authorization": "Bearer sk-f24a13ed139f757d99cdd9cdcae710fccead92681606a97086d9711f69d44869",
        #        }
        #        response = requests.post(url, json=data_dict, headers=headers)
        #        return response.json()
        return {}

    def handle_tool_schema_ops(self):
        if exists(self.tool_schema):
            logger.info(f"Tool schema provided: {self.tool_schema}")

            output = self.tool_struct.base_model_to_dict(
                self.tool_schema, output_str=True
            )

            # Add the tool schema to the short memory
            self.short_memory.add(
                role=self.agent_name, content=output
            )

        # If multiple base models, then conver them.
        if exists(self.list_base_models):
            logger.info(
                "Multiple base models provided, Automatically converting to OpenAI function"
            )

            schemas = self.tool_struct.multi_base_models_to_dict(
                output_str=True
            )

            # If the output is a string then add it to the memory
            self.short_memory.add(
                role=self.agent_name, content=schemas
            )

        return None

    def call_llm(self, task: str, *args, **kwargs) -> str:
        """
        Calls the appropriate method on the `llm` object based on the given task.

        Args:
            task (str): The task to be performed by the `llm` object.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            str: The result of the method call on the `llm` object.

        Raises:
            AttributeError: If no suitable method is found in the llm object.
            TypeError: If task is not a string or llm object is None.
            ValueError: If task is empty.
        """
        if not isinstance(task, str):
            raise TypeError("Task must be a string")

        if not task.strip():
            raise ValueError("Task cannot be empty")

        if self.llm is None:
            raise TypeError("LLM object cannot be None")

        try:
            out = self.llm.run(task, *args, **kwargs)

            return out
        except AttributeError as e:
            logger.error(
                f"Error calling LLM: {e} You need a class with a run(task: str) method"
            )
            raise e

    def handle_sop_ops(self):
        # If the user inputs a list of strings for the sop then join them and set the sop
        if exists(self.sop_list):
            self.sop = "\n".join(self.sop_list)
            self.short_memory.add(
                role=self.user_name, content=self.sop
            )

        if exists(self.sop):
            self.short_memory.add(
                role=self.user_name, content=self.sop
            )

        logger.info("SOP Uploaded into the memory")

    def run(
        self,
        task: Optional[str] = None,
        img: Optional[str] = None,
        device: Optional[str] = "cpu",  # gpu
        device_id: Optional[int] = 0,
        all_cores: Optional[bool] = True,
        scheduled_run_date: Optional[datetime] = None,
        do_not_use_cluster_ops: Optional[bool] = True,
        all_gpus: Optional[bool] = False,
        *args,
        **kwargs,
    ) -> Any:
        """
        Executes the agent's run method on a specified device, with optional scheduling.

        This method attempts to execute the agent's run method on a specified device, either CPU or GPU. It logs the device selection and the number of cores or GPU ID used. If the device is set to CPU, it can use all available cores or a specific core specified by `device_id`. If the device is set to GPU, it uses the GPU specified by `device_id`.

        If a `scheduled_date` is provided, the method will wait until that date and time before executing the task.

        Args:
            task (Optional[str], optional): The task to be executed. Defaults to None.
            img (Optional[str], optional): The image to be processed. Defaults to None.
            device (str, optional): The device to use for execution. Defaults to "cpu".
            device_id (int, optional): The ID of the GPU to use if device is set to "gpu". Defaults to 0.
            all_cores (bool, optional): If True, uses all available CPU cores. Defaults to True.
            scheduled_run_date (Optional[datetime], optional): The date and time to schedule the task. Defaults to None.
            do_not_use_cluster_ops (bool, optional): If True, does not use cluster ops. Defaults to False.
            *args: Additional positional arguments to be passed to the execution method.
            **kwargs: Additional keyword arguments to be passed to the execution method.

        Returns:
            Any: The result of the execution.

        Raises:
            ValueError: If an invalid device is specified.
            Exception: If any other error occurs during execution.
        """
        device = device or self.device
        device_id = device_id or self.device_id
        all_cores = all_cores or self.all_cores
        all_gpus = all_gpus or self.all_gpus

        do_not_use_cluster_ops = (
            do_not_use_cluster_ops or self.do_not_use_cluster_ops
        )

        if scheduled_run_date:
            while datetime.now() < scheduled_run_date:
                time.sleep(
                    1
                )  # Sleep for a short period to avoid busy waiting

        try:
            # If cluster ops disabled, run directly
            if do_not_use_cluster_ops is True:
                logger.info("Running without cluster operations")
                return self._run(
                    task=task,
                    img=img,
                    *args,
                    **kwargs,
                )

            else:
                return exec_callable_with_clusterops(
                    device=device,
                    device_id=device_id,
                    all_cores=all_cores,
                    all_gpus=all_gpus,
                    func=self._run,
                    task=task,
                    img=img,
                    *args,
                    **kwargs,
                )

        except ValueError as e:
            self._handle_run_error(e)

        except Exception as e:
            self._handle_run_error(e)

    def handle_artifacts(
        self, text: str, file_output_path: str, file_extension: str
    ) -> None:
        """Handle creating and saving artifacts with error handling."""
        try:
            # Ensure file_extension starts with a dot
            if not file_extension.startswith("."):
                file_extension = "." + file_extension

            # If file_output_path doesn't have an extension, treat it as a directory
            # and create a default filename based on timestamp
            if not os.path.splitext(file_output_path)[1]:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"artifact_{timestamp}{file_extension}"
                full_path = os.path.join(file_output_path, filename)
            else:
                full_path = file_output_path

            # Create the directory if it doesn't exist
            os.makedirs(os.path.dirname(full_path), exist_ok=True)

            logger.info(f"Creating artifact for file: {full_path}")
            artifact = Artifact(
                file_path=full_path,
                file_type=file_extension,
                contents=text,
                edit_count=0,
            )

            logger.info(
                f"Saving artifact with extension: {file_extension}"
            )
            artifact.save_as(file_extension)
            logger.success(
                f"Successfully saved artifact to {full_path}"
            )

        except ValueError as e:
            logger.error(
                f"Invalid input values for artifact: {str(e)}"
            )
            raise
        except IOError as e:
            logger.error(f"Error saving artifact to file: {str(e)}")
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error handling artifact: {str(e)}"
            )
            raise

    def showcase_config(self):

        # Convert all values in config_dict to concise string representations
        config_dict = self.to_dict()
        for key, value in config_dict.items():
            if isinstance(value, list):
                # Format list as a comma-separated string
                config_dict[key] = ", ".join(
                    str(item) for item in value
                )
            elif isinstance(value, dict):
                # Format dict as key-value pairs in a single string
                config_dict[key] = ", ".join(
                    f"{k}: {v}" for k, v in value.items()
                )
            else:
                # Ensure any non-iterable value is a string
                config_dict[key] = str(value)

        return formatter.print_table(
            f"Agent: {self.agent_name} Configuration", config_dict
        )
