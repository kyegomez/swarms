import asyncio
import json
import logging
import os
import random
import threading
import time
import traceback
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
    Sequence,
    Tuple,
    Union,
)

import toml
import yaml
from litellm import model_list
from litellm.exceptions import (
    AuthenticationError,
    BadRequestError,
    InternalServerError,
)
from litellm.utils import (
    get_max_tokens,
    supports_function_calling,
    supports_parallel_function_calling,
    supports_vision,
)
from loguru import logger
from pydantic import BaseModel

from swarms.agents.ape_agent import auto_generate_prompt
from swarms.artifacts.main_artifact import Artifact
from swarms.prompts.agent_system_prompts import AGENT_SYSTEM_PROMPT_3
from swarms.prompts.max_loop_prompt import generate_reasoning_prompt
from swarms.prompts.multi_modal_autonomous_instruction_prompt import (
    MULTI_MODAL_AUTO_AGENT_SYSTEM_PROMPT_1,
)
from swarms.prompts.react_base_prompt import REACT_SYS_PROMPT
from swarms.prompts.safety_prompt import SAFETY_PROMPT
from swarms.prompts.tools import tool_sop_prompt
from swarms.schemas.agent_mcp_errors import (
    AgentMCPConnectionError,
    AgentMCPToolError,
)
from swarms.schemas.agent_step_schemas import ManySteps, Step
from swarms.schemas.base_schemas import (
    AgentChatCompletionResponse,
)
from swarms.schemas.conversation_schema import ConversationSchema
from swarms.schemas.mcp_schemas import (
    MCPConnection,
    MultipleMCPConnections,
)
from swarms.structs.agent_roles import agent_roles
from swarms.structs.conversation import Conversation
from swarms.structs.ma_utils import set_random_models_for_agents
from swarms.structs.multi_agent_router import MultiAgentRouter
from swarms.structs.safe_loading import (
    SafeLoaderUtils,
    SafeStateManager,
)
from swarms.structs.transforms import (
    MessageTransforms,
    TransformConfig,
    handle_transforms,
)
from swarms.telemetry.main import log_agent_data
from swarms.tools.base_tool import BaseTool
from swarms.tools.mcp_client_tools import (
    execute_multiple_tools_on_multiple_mcp_servers_sync,
    execute_tool_call_simple,
    get_mcp_tools_sync,
    get_tools_for_multiple_mcp_servers,
)
from swarms.tools.py_func_to_openai_func_str import (
    convert_multiple_functions_to_openai_function_schema,
)
from swarms.utils.data_to_text import data_to_text
from swarms.utils.dynamic_context_window import dynamic_auto_chunking
from swarms.utils.file_processing import create_file_in_folder
from swarms.utils.formatter import formatter
from swarms.utils.generate_keys import generate_api_key
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.index import (
    exists,
    format_data_structure,
)
from swarms.utils.litellm_tokenizer import count_tokens
from swarms.utils.litellm_wrapper import LiteLLM
from swarms.utils.output_types import OutputType
from swarms.utils.pdf_to_text import pdf_to_text
from swarms.utils.swarms_marketplace_utils import (
    add_prompt_to_marketplace,
)


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
    return f"agent-{uuid.uuid4().hex}"


# Agent output types
ToolUsageType = Union[BaseModel, Dict[str, Any]]


# Agent Exceptions
class AgentError(Exception):
    """Base class for all agent-related exceptions."""

    pass


class AgentInitializationError(AgentError):
    """Exception raised when the agent fails to initialize properly. Please check the configuration and parameters."""

    pass


class AgentRunError(AgentError):
    """Exception raised when the agent encounters an error during execution. Ensure that the task and environment are set up correctly."""

    pass


class AgentLLMError(AgentError):
    """Exception raised when there is an issue with the language model (LLM). Verify the model's availability and compatibility."""

    pass


class AgentToolError(AgentError):
    """Exception raised when the agent fails to utilize a tool. Check the tool's configuration and availability."""

    pass


class AgentMemoryError(AgentError):
    """Exception raised when the agent encounters a memory-related issue. Ensure that memory resources are properly allocated and accessible."""

    pass


class AgentLLMInitializationError(AgentError):
    """Exception raised when the LLM fails to initialize properly. Please check the configuration and parameters."""

    pass


class AgentToolExecutionError(AgentError):
    """Exception raised when the agent fails to execute a tool. Check the tool's configuration and availability."""

    pass


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
        transforms (Optional[Union[TransformConfig, dict]]): Message transformation configuration for handling context limits
        user_name (str): The user name
        self_healing_enabled (bool): Enable self healing
        code_interpreter (bool): Enable code interpreter
        multi_modal (bool): Enable multimodal
        pdf_path (str): The path to the pdf
        list_of_pdf (str): The list of pdf
        tokenizer (Any): The tokenizer
        long_term_memory (BaseVectorDatabase): The long term memory
        fallback_model_name (str): The fallback model name to use if primary model fails
        fallback_models (List[str]): List of model names to try in order. First model is primary, rest are fallbacks
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
        output_type (agent_output_type): The output type. Supported: 'str', 'string', 'list', 'json', 'dict', 'yaml', 'xml'.
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
    >>> from swarms import Agent
    >>> agent = Agent(model_name="gpt-4.1", max_loops=1)
    >>> response = agent.run("Generate a report on the financials.")
    >>> print(response)
    >>> # Generate a report on the financials.

    >>> # Real-time streaming example
    >>> agent = Agent(model_name="gpt-4.1", max_loops=1, streaming_on=True)
    >>> response = agent.run("Tell me a long story.")  # Will stream in real-time
    >>> print(response)  # Final complete response

    >>> # Fallback model example
    >>> agent = Agent(
    ...     fallback_models=["gpt-4.1", "gpt-4o-mini", "gpt-3.5-turbo"],
    ...     max_loops=1
    ... )
    >>> response = agent.run("Generate a report on the financials.")
    >>> # Will try gpt-4o first, then gpt-4o-mini, then gpt-3.5-turbo if each fails

    """

    def __init__(
        self,
        id: Optional[str] = agent_id(),
        llm: Optional[Any] = None,
        template: Optional[str] = None,
        max_loops: Optional[Union[int, str]] = 1,
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
        transforms: Optional[Union[TransformConfig, dict]] = None,
        user_name: Optional[str] = "Human",
        self_healing_enabled: Optional[bool] = False,
        code_interpreter: Optional[bool] = False,
        multi_modal: Optional[bool] = None,
        pdf_path: Optional[str] = None,
        list_of_pdf: Optional[str] = None,
        tokenizer: Optional[Any] = None,
        long_term_memory: Optional[Union[Callable, Any]] = None,
        fallback_model_name: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
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
        output_type: OutputType = "str-all-except-first",
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
        frequency_penalty: float = 0.8,
        presence_penalty: float = 0.6,
        temperature: float = 0.5,
        workspace_dir: str = "agent_workspace",
        timeout: Optional[int] = None,
        # short_memory: Optional[str] = None,
        created_at: float = time.time(),
        return_step_meta: Optional[bool] = False,
        tags: Optional[List[str]] = None,
        step_pool: List[Step] = [],
        print_every_step: Optional[bool] = False,
        time_created: Optional[str] = time.strftime(
            "%Y-%m-%d %H:%M:%S", time.localtime()
        ),
        agent_output: ManySteps = None,
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
        role: agent_roles = "worker",
        print_on: bool = True,
        tools_list_dictionary: Optional[List[Dict[str, Any]]] = None,
        mcp_url: Optional[Union[str, MCPConnection]] = None,
        mcp_urls: List[str] = None,
        react_on: bool = False,
        safety_prompt_on: bool = False,
        random_models_on: bool = False,
        mcp_config: Optional[MCPConnection] = None,
        mcp_configs: Optional[MultipleMCPConnections] = None,
        top_p: Optional[float] = 0.90,
        conversation_schema: Optional[ConversationSchema] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        tool_call_summary: bool = True,
        output_raw_json_from_tool_call: bool = False,
        summarize_multiple_images: bool = False,
        tool_retry_attempts: int = 3,
        reasoning_prompt_on: bool = True,
        dynamic_context_window: bool = True,
        show_tool_execution_output: bool = True,
        reasoning_effort: str = None,
        drop_params: bool = True,
        thinking_tokens: int = None,
        reasoning_enabled: bool = False,
        handoffs: Optional[Union[Sequence[Callable], Any]] = None,
        capabilities: Optional[List[str]] = None,
        mode: Literal["interactive", "fast", "standard"] = "standard",
        publish_to_marketplace: bool = False,
        use_cases: Optional[List[Dict[str, Any]]] = None,
        *args,
        **kwargs,
    ):
        # super().__init__(*args, **kwargs)
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
        self.saved_state_path = saved_state_path
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
        # self.saved_state_path = f"{self.agent_name}_{generate_api_key(prefix='agent-')}_state.json"
        self.saved_state_path = (
            f"{generate_api_key(prefix='agent-')}_state.json"
        )
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
        self.tokenizer = tokenizer
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
        self.role = role
        self.print_on = print_on
        self.tools_list_dictionary = tools_list_dictionary
        self.mcp_url = mcp_url
        self.mcp_urls = mcp_urls
        self.react_on = react_on
        self.safety_prompt_on = safety_prompt_on
        self.random_models_on = random_models_on
        self.mcp_config = mcp_config
        self.top_p = top_p
        self.conversation_schema = conversation_schema
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.tool_call_summary = tool_call_summary
        self.output_raw_json_from_tool_call = (
            output_raw_json_from_tool_call
        )
        self.summarize_multiple_images = summarize_multiple_images
        self.tool_retry_attempts = tool_retry_attempts
        self.reasoning_prompt_on = reasoning_prompt_on
        self.dynamic_context_window = dynamic_context_window
        self.show_tool_execution_output = show_tool_execution_output
        self.mcp_configs = mcp_configs
        self.reasoning_effort = reasoning_effort
        self.drop_params = drop_params
        self.thinking_tokens = thinking_tokens
        self.reasoning_enabled = reasoning_enabled
        self.fallback_model_name = fallback_model_name
        self.handoffs = handoffs
        self.capabilities = capabilities
        self.mode = mode
        self.publish_to_marketplace = publish_to_marketplace

        # Initialize transforms
        if transforms is None:
            self.transforms = None
        elif isinstance(transforms, TransformConfig):
            self.transforms = MessageTransforms(transforms)
        elif isinstance(transforms, dict):
            config = TransformConfig(**transforms)
            self.transforms = MessageTransforms(config)
        else:
            pass

        self.fallback_models = fallback_models or []
        self.current_model_index = 0
        self.model_attempts = {}

        # If fallback_models is provided, use the first model as the primary model
        if self.fallback_models and not self.model_name:
            self.model_name = self.fallback_models[0]

        # self.init_handling()
        self.setup_config()

        # Initialize the short memory
        self.short_memory = self.short_memory_init()

        # Initialize the tools
        self.tool_struct = self.setup_tools()

        if exists(self.docs_folder):
            self.get_docs_from_doc_folders()

        if exists(self.tool_schema) or exists(self.list_base_models):
            self.handle_tool_schema_ops()

        if exists(self.sop) or exists(self.sop_list):
            self.handle_sop_ops()

        if self.interactive is True:
            self.reasoning_prompt_on = False

        if self.reasoning_prompt_on is True and (
            (isinstance(self.max_loops, int) and self.max_loops >= 2)
            or self.max_loops == "auto"
        ):
            self.system_prompt += generate_reasoning_prompt(
                self.max_loops
            )

        if self.react_on is True:
            self.system_prompt += REACT_SYS_PROMPT

        if self.autosave is True:
            log_agent_data(self.to_dict())

        if exists(self.tools):
            self.tool_handling()

        if self.llm is None:
            self.llm = self.llm_handling()

        if self.random_models_on is True:
            self.model_name = set_random_models_for_agents()

        if self.dashboard is True:
            self.print_dashboard()

        self.reliability_check()

        if self.mode == "fast":
            self.print_on = False
            self.verbose = False

        if self.publish_to_marketplace is True:
            # Join tags and capabilities into a single string
            tags_and_capabilities = ", ".join(
                self.tags + self.capabilities
                if self.tags and self.capabilities
                else None
            )

            if self.use_cases is None:
                raise AgentInitializationError(
                    "Use cases are required when publishing to the marketplace. The schema is a list of dictionaries with 'title' and 'description' keys."
                )

            add_prompt_to_marketplace(
                name=self.agent_name,
                prompt=self.short_memory.get_str(),
                description=self.agent_description,
                tags=tags_and_capabilities,
                category="research",
                use_cases=(
                    self.use_cases if self.use_cases else None
                ),
            )

    def handle_handoffs(self, task: Optional[str] = None):
        router = MultiAgentRouter(
            name=self.agent_name,
            description=self.agent_description,
            agents=self.handoffs,
            model=self.model_name,
            temperature=self.temperature,
            system_prompt=self.system_prompt,
            output_type=self.output_type,
        )

        return router.run(task=task)

    def setup_tools(self):
        return BaseTool(
            tools=self.tools,
            verbose=self.verbose,
        )

    def tool_handling(self):

        # Convert all the tools into a list of dictionaries
        self.tools_list_dictionary = (
            convert_multiple_functions_to_openai_function_schema(
                self.tools
            )
        )

        self.short_memory.add(
            role=self.agent_name,
            content=self.tools_list_dictionary,
        )

    def short_memory_init(self):
        # Assemble initial prompt context as a dict, update conditionally, then compose as string
        prompt_dict = {}

        if self.agent_name is not None:
            prompt_dict["name"] = f"Your Name: {self.agent_name}"
        if self.agent_description is not None:
            prompt_dict["description"] = (
                f"Your Description: {self.agent_description}"
            )
        if self.system_prompt is not None:
            prompt_dict["instructions"] = (
                f"Your Instructions: {self.system_prompt}"
            )

        # Compose prompt, prioritizing adding everything present in the dict
        # (entries are newline separated, order: name → description → instructions)
        prompt = ""
        for key in ["name", "description", "instructions"]:
            if key in prompt_dict:
                prompt += f"\n {prompt_dict[key]} \n"

        if self.safety_prompt_on is True:
            prompt += SAFETY_PROMPT

        # Initialize the short term memory
        memory = Conversation(
            name=f"{self.agent_name}_id_{self.id}_conversation",
            system_prompt=prompt,
            user=self.user_name,
            rules=self.rules,
            token_count=False,
            message_id_on=False,
            time_enabled=True,
            dynamic_context_window=self.dynamic_context_window,
            tokenizer_model_name=self.model_name,
            context_length=self.context_length,
        )

        return memory

    def llm_handling(self, *args, **kwargs):
        """Initialize the LiteLLM instance with combined configuration from all sources.

        This method combines llm_args, tools_list_dictionary, MCP tools, and any additional
        arguments passed to this method into a single unified configuration.

        Args:
            *args: Positional arguments that can be used for additional configuration.
                  If a single dictionary is passed, it will be merged into the configuration.
                  Other types of args will be stored under 'additional_args' key.
            **kwargs: Keyword arguments that will be merged into the LiteLLM configuration.
                     These take precedence over existing configuration.

        Returns:
            LiteLLM: The initialized LiteLLM instance
        """

        if self.model_name is None:
            self.model_name = "gpt-4o-mini"

        # Use current model (which may be a fallback) only if fallbacks are configured
        if self.fallback_models:
            current_model = self.get_current_model()
        else:
            current_model = self.model_name

        # Determine if parallel tool calls should be enabled
        if exists(self.tools) and len(self.tools) >= 2:
            parallel_tool_calls = True
        elif exists(self.mcp_url) or exists(self.mcp_urls):
            parallel_tool_calls = True
        elif exists(self.mcp_config):
            parallel_tool_calls = True
        else:
            parallel_tool_calls = False

        try:
            # Base configuration that's always included
            common_args = {
                "model_name": current_model,
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "system_prompt": self.system_prompt,
                "stream": self.streaming_on,
                "top_p": self.top_p,
                "retries": self.retry_attempts,
                "reasoning_effort": self.reasoning_effort,
                "thinking_tokens": self.thinking_tokens,
                "reasoning_enabled": self.reasoning_enabled,
            }

            # Initialize tools_list_dictionary, if applicable
            tools_list = []

            # Append tools from different sources
            if self.tools_list_dictionary is not None:
                tools_list.extend(self.tools_list_dictionary)

            if exists(self.mcp_url) or exists(self.mcp_urls):
                if self.verbose:
                    logger.info(
                        f"Adding MCP tools to memory for {self.agent_name}"
                    )
                # tools_list.extend(self.add_mcp_tools_to_memory())
                mcp_tools = self.add_mcp_tools_to_memory()

                if self.verbose:
                    logger.info(f"MCP tools: {mcp_tools}")

                tools_list.extend(mcp_tools)

            # Additional arguments for LiteLLM initialization
            additional_args = {}

            if self.llm_args is not None:
                additional_args.update(self.llm_args)

            if tools_list:
                additional_args.update(
                    {
                        "tools_list_dictionary": tools_list,
                        "tool_choice": "auto",
                        "parallel_tool_calls": parallel_tool_calls,
                    }
                )

            if exists(self.mcp_url) or exists(self.mcp_urls):
                additional_args.update({"mcp_call": True})

            # if args or kwargs are provided, then update the additional_args
            if args or kwargs:
                # Handle keyword arguments first
                if kwargs:
                    additional_args.update(kwargs)

                # Handle positional arguments (args)
                # These could be additional configuration parameters
                # that should be merged into the additional_args
                if args:
                    # If args contains a dictionary, merge it directly
                    if len(args) == 1 and isinstance(args[0], dict):
                        additional_args.update(args[0])
                    else:
                        # For other types of args, log them for debugging
                        # and potentially handle them based on their type
                        logger.debug(
                            f"Received positional args in llm_handling: {args}"
                        )
                        # You can add specific handling for different arg types here
                        # For now, we'll add them as additional configuration
                        additional_args.update(
                            {"additional_args": args}
                        )

            # Final LiteLLM initialization with combined arguments
            self.llm = LiteLLM(**{**common_args, **additional_args})

            return self.llm
        except AgentLLMInitializationError as e:
            logger.error(
                f"AgentLLMInitializationError: Agent Name: {self.agent_name} Error in llm_handling: {e} Your current configuration is not supported. Please check the configuration and parameters. Traceback: {traceback.format_exc()}"
            )
            return None

    def add_mcp_tools_to_memory(self):
        """
        Adds MCP tools to the agent's short-term memory.

        This function checks for either a single MCP URL or multiple MCP URLs and adds the available tools
        to the agent's memory. The tools are listed in JSON format.

        Raises:
            Exception: If there's an error accessing the MCP tools
        """
        try:
            # Determine which MCP configuration to use
            if exists(self.mcp_url):
                tools = get_mcp_tools_sync(server_path=self.mcp_url)
            elif exists(self.mcp_config):
                tools = get_mcp_tools_sync(connection=self.mcp_config)
            elif exists(self.mcp_urls):
                logger.info(
                    f"Getting MCP tools for multiple MCP servers for {self.agent_name}"
                )
                tools = get_tools_for_multiple_mcp_servers(
                    urls=self.mcp_urls,
                )

                if self.verbose:
                    logger.info(f"MCP tools: {tools}")
            else:
                raise AgentMCPConnectionError(
                    "mcp_url must be either a string URL or MCPConnection object"
                )

            # Print success message if any MCP configuration exists
            if self.print_on:
                self.pretty_print(
                    f"✨ [SYSTEM] Successfully integrated {len(tools)} MCP tools into agent: {self.agent_name} | Status: ONLINE | Time: {time.strftime('%H:%M:%S')} ✨",
                    loop_count=0,
                )

            return tools
        except AgentMCPConnectionError as e:
            logger.error(
                f"Error Adding MCP Tools to Agent: {self.agent_name} Error: {e} Traceback: {traceback.format_exc()}"
            )
            raise e

    def setup_config(self):
        # The max_loops will be set dynamically if the dynamic_loop
        if self.dynamic_loops is True:
            logger.info("Dynamic loops enabled")
            self.max_loops = "auto"

        # If multimodal = yes then set the sop to the multimodal sop
        if self.multi_modal is True:
            self.sop = MULTI_MODAL_AUTO_AGENT_SYSTEM_PROMPT_1

        # If the preset stopping token is enabled then set the stopping token to the preset stopping token
        if self.preset_stopping_token is not None:
            self.stopping_token = "<DONE>"

        # Initialize the feedback
        self.feedback = []

    def check_model_supports_utilities(
        self, img: Optional[str] = None
    ) -> bool:
        """
        Check if the current model supports vision capabilities.

        Args:
            img (str, optional): Image input to check vision support for. Defaults to None.

        Returns:
            bool: True if model supports vision and image is provided, False otherwise.
        """

        # Only check vision support if an image is provided
        if img is not None:
            out = supports_vision(self.model_name)
            if out is False:
                logger.error(
                    f"[Agent: {self.agent_name}] Model '{self.model_name}' does not support vision capabilities. "
                    f"Image input was provided: {img[:100]}{'...' if len(img) > 100 else ''}. "
                    f"Please use a vision-enabled model."
                )

        if self.tools_list_dictionary is not None:
            out = supports_function_calling(self.model_name)
            if out is False:
                logger.error(
                    f"[Agent: {self.agent_name}] Model '{self.model_name}' does not support function calling capabilities. "
                    f"tools_list_dictionary is set: {self.tools_list_dictionary}. "
                    f"Please use a function calling-enabled model."
                )

        if self.tools is not None:
            if len(self.tools) > 2:
                out = supports_parallel_function_calling(
                    self.model_name
                )
                if out is False:
                    logger.error(
                        f"[Agent: {self.agent_name}] Model '{self.model_name}' does not support parallel function calling capabilities. "
                        f"Please use a parallel function calling-enabled model."
                    )

        return None

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
                    task=task, model=self.llm
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
        """
        Print a dashboard displaying the agent's current status and configuration.
        Uses square brackets instead of emojis for section headers and bullet points.
        """
        tools_activated = True if self.tools is not None else False
        mcp_activated = True if self.mcp_url is not None else False
        formatter.print_panel(
            f"""
            
            [Agent {self.agent_name} Dashboard]
            ===========================================================
            
            [Agent {self.agent_name} Status]: ONLINE & OPERATIONAL
            -----------------------------------------------------------
            
            [Agent Identity]
            - [Name]: {self.agent_name}
            - [Description]: {self.agent_description}
            
            [Technical Specifications]
            - [Model]: {self.model_name}
            - [Internal Loops]: {self.max_loops}
            - [Max Tokens]: {self.max_tokens}
            - [Dynamic Temperature]: {self.dynamic_temperature_enabled}
            
            [System Modules]
            - [Tools Activated]: {tools_activated}
            - [MCP Activated]: {mcp_activated}
            
            ===========================================================
            [Ready for Tasks]
                              
            """,
            title=f"Agent {self.agent_name} Dashboard",
        )

    def handle_rag_query(self, query: str):
        """
        Handle RAG query
        """
        try:
            logger.info(
                f"Agent: {self.agent_name} Querying RAG memory for: {query}"
            )
            output = self.long_term_memory.query(
                query,
            )

            output = dynamic_auto_chunking(
                content=output,
                context_length=self.max_tokens,
                tokenizer_model_name=self.model_name,
            )

            self.short_memory.add(
                role="system",
                content=(
                    "[RAG Query Initiated]\n"
                    "----------------------------------\n"
                    f"Query:\n{query}\n\n"
                    f"Retrieved Knowledge (RAG Output):\n{output}\n"
                    "----------------------------------\n"
                    "The above information was retrieved from the agent's long-term memory using Retrieval-Augmented Generation (RAG). "
                    "Use this context to inform your next response or reasoning step."
                ),
            )
        except AgentMemoryError as e:
            logger.error(
                f"Agent: {self.agent_name} Error handling RAG query: {e} Traceback: {traceback.format_exc()}"
            )
            raise e

    # Main function
    def _run(
        self,
        task: Optional[Union[str, Any]] = None,
        img: Optional[str] = None,
        streaming_callback: Optional[Callable[[str], None]] = None,
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute the agent's main loop for a given task.

        This function manages the agent's reasoning and action loop, including:
        - Initializing the task and context
        - Handling Retrieval-Augmented Generation (RAG) queries if enabled
        - Planning (if enabled)
        - Managing internal reasoning loops and memory
        - Optionally processing images and streaming output
        - Autosaving agent state if configured

        Args:
            task (Optional[Union[str, Any]]): The task or prompt for the agent to process.
            img (Optional[str]): Optional image path or data to be processed by the agent.
            streaming_callback (Optional[Callable[[str], None]]): Optional callback for streaming output.
            *args: Additional positional arguments for extensibility.
            **kwargs: Additional keyword arguments for extensibility.

        Returns:
            Any: The agent's output, which may be a string, list, JSON, dict, YAML, XML, or other type depending on the agent's configuration and the task.

        Examples:
            agent(task="What is the capital of France?")
            agent(task="Summarize this document.", img="path/to/image.jpg")
            agent(task="Analyze this image.", img="path/to/image.jpg", is_last=True)
        """
        try:

            self.check_if_no_prompt_then_autogenerate(task)

            self.check_model_supports_utilities(img=img)

            self.short_memory.add(role=self.user_name, content=task)

            # Handle RAG query only once
            if (
                self.long_term_memory is not None
                and self.rag_every_loop is False
            ):
                self.handle_rag_query(task)

            if self.plan_enabled is True:
                self.plan(task)

            # Set the loop count
            loop_count = 0

            # Clear the short memory
            response = None

            # Autosave
            if self.autosave:
                log_agent_data(self.to_dict())
                self.save()

            while (
                self.max_loops == "auto"
                or loop_count < self.max_loops
            ):
                loop_count += 1

                # Handle RAG query every loop
                if (
                    self.long_term_memory is not None
                    and self.rag_every_loop is True
                ):
                    self.handle_rag_query(task)

                if (
                    isinstance(self.max_loops, int)
                    and self.max_loops >= 2
                ):
                    if self.reasoning_prompt_on is True:
                        self.short_memory.add(
                            role=self.agent_name,
                            content=f"Current Internal Reasoning Loop: {loop_count}/{self.max_loops}",
                        )

                # If it is the final loop, then add the final loop message
                if (
                    loop_count >= 2
                    and isinstance(self.max_loops, int)
                    and loop_count == self.max_loops
                ):
                    if self.reasoning_prompt_on is True:
                        self.short_memory.add(
                            role=self.agent_name,
                            content=f"🎉 Final Internal Reasoning Loop: {loop_count}/{self.max_loops} Prepare your comprehensive response.",
                        )

                # Dynamic temperature
                if self.dynamic_temperature_enabled is True:
                    self.dynamic_temperature()

                # Task prompt with optional transforms
                if self.transforms is not None:
                    task_prompt = handle_transforms(
                        transforms=self.transforms,
                        short_memory=self.short_memory,
                        model_name=self.model_name,
                    )

                else:
                    # Use original method if no transforms
                    task_prompt = (
                        self.short_memory.return_history_as_string()
                    )

                # Parameters
                attempt = 0
                success = False
                while attempt < self.retry_attempts and not success:
                    try:

                        if img is not None:
                            response = self.call_llm(
                                task=task_prompt,
                                img=img,
                                current_loop=loop_count,
                                streaming_callback=streaming_callback,
                                *args,
                                **kwargs,
                            )
                        else:
                            response = self.call_llm(
                                task=task_prompt,
                                current_loop=loop_count,
                                streaming_callback=streaming_callback,
                                *args,
                                **kwargs,
                            )

                        # If streaming is enabled, then don't print the response

                        # Parse the response from the agent with the output type
                        if exists(self.tools_list_dictionary):
                            if isinstance(response, BaseModel):
                                response = response.model_dump()

                        # Parse the response from the agent with the output type
                        response = self.parse_llm_output(response)

                        self.short_memory.add(
                            role=self.agent_name,
                            content=response,
                        )

                        # Print
                        if self.print_on is True:
                            if isinstance(response, list):
                                self.pretty_print(
                                    # f"Structured Output - Attempting Function Call Execution [{time.strftime('%H:%M:%S')}] \n\n Output: {format_data_structure(response)} ",
                                    f"[Structured Output] [Time: {time.strftime('%H:%M:%S')}] \n\n {json.dumps(response, indent=4)}",
                                    loop_count,
                                )
                            elif self.streaming_on:
                                pass
                            else:
                                self.pretty_print(
                                    response, loop_count
                                )

                        # Check and execute callable tools
                        if exists(self.tools):
                            self.tool_execution_retry(
                                response, loop_count
                            )

                        # Handle MCP tools
                        if (
                            exists(self.mcp_url)
                            or exists(self.mcp_config)
                            or exists(self.mcp_urls)
                        ):
                            # Only handle MCP tools if response is not None
                            if response is not None:
                                self.mcp_tool_handling(
                                    response=response,
                                    current_loop=loop_count,
                                )
                            else:
                                logger.warning(
                                    f"LLM returned None response in loop {loop_count}, skipping MCP tool handling"
                                )

                        # self.sentiment_and_evaluator(response)

                        success = True  # Mark as successful to exit the retry loop

                    except (
                        BadRequestError,
                        InternalServerError,
                        AuthenticationError,
                        Exception,
                    ) as e:

                        if self.autosave is True:
                            log_agent_data(self.to_dict())
                            self.save()

                        logger.error(
                            f"Attempt {attempt+1}/{self.retry_attempts}: Error generating response in loop {loop_count} for agent '{self.agent_name}': {str(e)} | Traceback: {traceback.format_exc()}"
                        )
                        attempt += 1

                if not success:

                    if self.autosave is True:
                        log_agent_data(self.to_dict())
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
                    logger.info(
                        f"Agent '{self.agent_name}' stopping condition met. "
                        f"Loop: {loop_count}, Response length: {len(str(response)) if response else 0}"
                    )
                    break
                elif (
                    self.stopping_func is not None
                    and self.stopping_func(response)
                ):
                    logger.info(
                        f"Agent '{self.agent_name}' stopping function condition met. "
                        f"Loop: {loop_count}, Response length: {len(str(response)) if response else 0}"
                    )
                    break

                if self.interactive:

                    # logger.info("Interactive mode enabled.")
                    user_input = input("You: ")

                    # User-defined exit command
                    if (
                        user_input.lower()
                        == self.custom_exit_command.lower()
                    ):
                        self.pretty_print(
                            "Exiting as per user request.",
                            loop_count=loop_count,
                        )
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
                log_agent_data(self.to_dict())

                self.save()

            # Output formatting based on output_type
            return history_output_formatter(
                self.short_memory, type=self.output_type
            )

        except Exception as error:
            self._handle_run_error(error)

        except KeyboardInterrupt as error:
            self._handle_run_error(error)

    def _handle_run_error(self, error: any):
        if self.autosave is True:
            self.save()
            log_agent_data(self.to_dict())

        # Get detailed error information
        error_type = type(error).__name__
        error_message = str(error)
        traceback_info = traceback.format_exc()

        logger.error(
            f"Agent: {self.agent_name} An error occurred while running your agent.\n"
            f"Error Type: {error_type}\n"
            f"Error Message: {error_message}\n"
            f"Traceback:\n{traceback_info}\n"
            f"Agent State: {self.to_dict()}\n"
            f"Please optimize your input parameters, or create an issue on the Swarms GitHub and contact our team on Discord for support. "
            f"For technical support, refer to this document: https://docs.swarms.world/en/latest/swarms/support/"
        )

        raise error

    async def arun(
        self,
        task: Optional[str] = None,
        img: Optional[str] = None,
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
        *args,
        **kwargs,
    ) -> Any:
        """Call the agent

        Args:
            task (Optional[str]): The task to be performed. Defaults to None.
            img (Optional[str]): The image to be processed. Defaults to None.
        """
        try:
            return self.run(
                task=task,
                img=img,
                *args,
                **kwargs,
            )
        except Exception as error:
            self._handle_run_error(error)

    def receive_message(
        self, agent_name: str, task: str, *args, **kwargs
    ):
        improved_prompt = (
            f"You have received a message from agent '{agent_name}':\n\n"
            f'"{task}"\n\n'
            "Please process this message and respond appropriately."
        )
        return self.run(task=improved_prompt, *args, **kwargs)

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
        Create a strategic plan for executing the given task.

        This method generates a step-by-step plan by combining the conversation
        history, planning prompt, and current task. The plan is then added to
        the agent's short-term memory for reference during execution.

        Args:
            task (str): The task to create a plan for
            *args: Additional positional arguments passed to the LLM
            **kwargs: Additional keyword arguments passed to the LLM

        Returns:
            None: The plan is stored in memory rather than returned

        Raises:
            Exception: If planning fails, the original exception is re-raised
        """
        try:
            # Get the current conversation history
            history = self.short_memory.get_str()

            plan_prompt = f"Create a comprehensive step-by-step plan to complete the following task: \n\n {task}"

            # Construct the planning prompt by combining history, planning prompt, and task
            if exists(self.planning_prompt):
                planning_prompt = f"{history}\n\n{self.planning_prompt}\n\nTask: {task}"
            else:
                planning_prompt = (
                    f"{history}\n\n{plan_prompt}\n\nTask: {task}"
                )

            # Generate the plan using the LLM
            plan = self.llm.run(task=planning_prompt, *args, **kwargs)

            # Store the generated plan in short-term memory
            self.short_memory.add(role=self.agent_name, content=plan)

            return None

        except Exception as error:
            logger.error(
                f"Failed to create plan for task '{task}': {error}"
            )
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

    def reliability_check(self):

        if self.system_prompt is None:
            logger.warning(
                "The system prompt is not set. Please set a system prompt for the agent to improve reliability."
            )

        if self.agent_name is None:
            logger.warning(
                "The agent name is not set. Please set an agent name to improve reliability."
            )

        if self.max_loops is None or self.max_loops == 0:
            raise AgentInitializationError(
                "Max loops is not provided or is set to 0. Please set max loops to 1 or more."
            )

        # Ensure max_tokens is set to a valid value based on the model, with a robust fallback.
        if self.max_tokens is None or self.max_tokens <= 0:
            suggested_tokens = get_max_tokens(self.model_name)
            if suggested_tokens is not None and suggested_tokens > 0:
                self.max_tokens = suggested_tokens
            else:
                logger.warning(
                    f"Could not determine max_tokens for model '{self.model_name}'. Falling back to default value of 8192."
                )
                self.max_tokens = 8192

        if self.context_length is None or self.context_length == 0:
            raise AgentInitializationError(
                "Context length is not provided. Please set a valid context length."
            )

        if self.tools_list_dictionary is not None:
            if not supports_function_calling(self.model_name):
                logger.warning(
                    f"The model '{self.model_name}' does not support function calling. Please use a model that supports function calling."
                )

        try:
            if self.max_tokens > get_max_tokens(self.model_name):
                logger.warning(
                    f"Max tokens is set to {self.max_tokens}, but the model '{self.model_name}' may or may not support {get_max_tokens(self.model_name)} tokens. Please set max tokens to {get_max_tokens(self.model_name)} or less."
                )

        except Exception:
            pass

        if self.model_name not in model_list:
            logger.warning(
                f"The model '{self.model_name}' may not be supported. Please use a supported model, or override the model name with the 'llm' parameter, which should be a class with a 'run(task: str)' method or a '__call__' method."
            )

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
                    time_enabled=False,
                    user=self.user_name,
                    rules=self.rules,
                )

            # Reinitialize executor if needed
            # if not hasattr(self, "executor") or self.executor is None:
            with ThreadPoolExecutor(
                max_workers=os.cpu_count()
            ) as executor:
                self.executor = executor

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
        self.response_filters.append(filter_word)

    def apply_response_filters(self, response: str) -> str:
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

    def update_max_loops(self, max_loops: Union[int, str]):
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
            # Process all documents and combine their content
            all_data = []
            for doc in docs:
                data = data_to_text(doc)
                all_data.append(f"Document: {doc}\n{data}")

            # Combine all document content
            combined_data = "\n\n".join(all_data)

            return self.short_memory.add(
                role=self.user_name, content=combined_data
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
            message = f"To: {agent_name}: {message}"
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

                if sentiment > self.sentiment_threshold:
                    pass
                elif sentiment < self.sentiment_threshold:
                    pass

                self.short_memory.add(
                    role=self.agent_name,
                    content=sentiment,
                )
        except Exception:
            pass

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
                time.sleep(delay)
        except Exception:
            pass

    def check_available_tokens(self):
        # Log the amount of tokens left in the memory and in the task
        if self.tokenizer is not None:
            tokens_used = count_tokens(
                self.short_memory.return_history_as_string()
            )
            logger.info(
                f"Tokens available: {self.context_length - tokens_used}"
            )

        return tokens_used

    def tokens_checks(self):
        # Check the tokens available
        tokens_used = count_tokens(
            self.short_memory.return_history_as_string()
        )
        out = self.check_available_tokens()

        logger.info(
            f"Tokens available: {out} Context Length: {self.context_length} Tokens in memory: {tokens_used}"
        )

        return out

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

        # Create a copy of the dict to avoid mutating the original object
        # Remove the llm object from the copy since it's not serializable
        dict_copy = self.__dict__.copy()
        dict_copy.pop("llm", None)

        return {
            attr_name: self._serialize_attr(attr_name, attr_value)
            for attr_name, attr_value in dict_copy.items()
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

    def call_llm(
        self,
        task: str,
        img: Optional[str] = None,
        current_loop: int = 0,
        streaming_callback: Optional[Callable[[str], None]] = None,
        *args,
        **kwargs,
    ) -> str:
        """
        Calls the appropriate method on the `llm` object based on the given task.

        Args:
            task (str): The task to be performed by the `llm` object.
            img (str, optional): Path or URL to an image file.
            audio (str, optional): Path or URL to an audio file.
            streaming_callback (Optional[Callable[[str], None]]): Callback function to receive streaming tokens in real-time.
            *args: Variable length argument list.
            **kwargs: Arbitrary keyword arguments.

        Returns:
            str: The result of the method call on the `llm` object.

        Raises:
            AttributeError: If no suitable method is found in the llm object.
            TypeError: If task is not a string or llm object is None.
            ValueError: If task is empty.
        """

        # Filter out is_last from kwargs if present
        if "is_last" in kwargs:
            del kwargs["is_last"]

        try:
            # Set streaming parameter in LLM if streaming is enabled
            if self.streaming_on and hasattr(self.llm, "stream"):
                original_stream = self.llm.stream
                self.llm.stream = True

                if img is not None:
                    streaming_response = self.llm.run(
                        task=task, img=img, *args, **kwargs
                    )
                else:
                    streaming_response = self.llm.run(
                        task=task, *args, **kwargs
                    )

                # If we get a streaming response, handle it with the new streaming panel
                if hasattr(
                    streaming_response, "__iter__"
                ) and not isinstance(streaming_response, str):
                    # Check if streaming_callback is provided (for ConcurrentWorkflow dashboard integration)
                    if streaming_callback is not None:
                        # Real-time callback streaming for dashboard integration
                        chunks = []
                        for chunk in streaming_response:
                            if (
                                hasattr(chunk, "choices")
                                and chunk.choices[0].delta.content
                            ):
                                content = chunk.choices[
                                    0
                                ].delta.content
                                chunks.append(content)
                                # Call the streaming callback with the new chunk
                                streaming_callback(content)
                        complete_response = "".join(chunks)
                    # Check print_on parameter for different streaming behaviors
                    elif self.print_on is False:
                        # Silent streaming - no printing, just collect chunks
                        chunks = []
                        for chunk in streaming_response:
                            if (
                                hasattr(chunk, "choices")
                                and chunk.choices[0].delta.content
                            ):
                                content = chunk.choices[
                                    0
                                ].delta.content
                                chunks.append(content)
                        complete_response = "".join(chunks)
                    else:
                        # Collect chunks for conversation saving
                        collected_chunks = []

                        def on_chunk_received(chunk: str):
                            """Callback to collect chunks as they arrive"""
                            collected_chunks.append(chunk)
                            # Optional: Save each chunk to conversation in real-time
                            # This creates a more detailed conversation history
                            if self.verbose:
                                logger.debug(
                                    f"Streaming chunk received: {chunk[:50]}..."
                                )

                        # Use the streaming panel to display and collect the response
                        complete_response = formatter.print_streaming_panel(
                            streaming_response,
                            title=f"🤖 Agent: {self.agent_name} Loops: {current_loop}",
                            style=None,  # Use random color like non-streaming approach
                            collect_chunks=True,
                            on_chunk_callback=on_chunk_received,
                        )

                    # Restore original stream setting
                    self.llm.stream = original_stream

                    # Return the complete response for further processing
                    return complete_response
                else:
                    # Restore original stream setting
                    self.llm.stream = original_stream
                    return streaming_response
            else:
                args = {
                    "task": task,
                }

                if img is not None:
                    args["img"] = img

                out = self.llm.run(**args, **kwargs)

                return out

        except (
            AgentLLMError,
            BadRequestError,
            InternalServerError,
            AuthenticationError,
            Exception,
        ) as e:
            logger.error(
                f"Error calling LLM with model '{self.get_current_model()}': {e}. "
                f"Task: {task}, Args: {args}, Kwargs: {kwargs} Traceback: {traceback.format_exc()}"
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
        task: Optional[Union[str, Any]] = None,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        correct_answer: Optional[str] = None,
        streaming_callback: Optional[Callable[[str], None]] = None,
        n: int = 1,
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
            imgs (Optional[List[str]], optional): The list of images to be processed. Defaults to None.
            streaming_callback (Optional[Callable[[str], None]], optional): Callback function to receive streaming tokens in real-time. Defaults to None.
            *args: Additional positional arguments to be passed to the execution method.
            **kwargs: Additional keyword arguments to be passed to the execution method.

        Returns:
            Any: The result of the execution.

        Raises:
            ValueError: If an invalid device is specified.
            Exception: If any other error occurs during execution.
        """

        if not isinstance(task, str):
            task = format_data_structure(task)

        try:
            if exists(imgs):
                output = self.run_multiple_images(
                    task=task, imgs=imgs, *args, **kwargs
                )
            elif exists(correct_answer):
                output = self.continuous_run_with_answer(
                    task=task,
                    img=img,
                    correct_answer=correct_answer,
                    *args,
                    **kwargs,
                )
            elif exists(self.handoffs):
                output = self.handle_handoffs(task=task)
            elif n > 1:
                output = [self.run(task=task) for _ in range(n)]
            else:
                output = self._run(
                    task=task,
                    img=img,
                    streaming_callback=streaming_callback,
                    *args,
                    **kwargs,
                )

            return output

        except (
            AgentRunError,
            AgentLLMError,
            BadRequestError,
            InternalServerError,
            AuthenticationError,
            Exception,
        ) as e:
            # Try fallback models if available
            if self.is_fallback_available():
                return self._handle_fallback_execution(
                    task=task,
                    img=img,
                    imgs=imgs,
                    correct_answer=correct_answer,
                    streaming_callback=streaming_callback,
                    original_error=e,
                    *args,
                    **kwargs,
                )
            else:
                if self.verbose:
                    # No fallback available
                    logger.error(
                        f"Agent Name: {self.agent_name} [NO FALLBACK] failed with model '{self.get_current_model()}' "
                        f"and no fallback models are configured. Error: {str(e)[:100]}{'...' if len(str(e)) > 100 else ''}"
                    )

                self._handle_run_error(e)

        except KeyboardInterrupt:
            logger.warning(
                f"Agent Name: {self.agent_name} Keyboard interrupt detected. "
                "If autosave is enabled, the agent's state will be saved to the workspace directory. "
                "To enable autosave, please initialize the agent with Agent(autosave=True)."
                "For technical support, refer to this document: https://docs.swarms.world/en/latest/swarms/support/"
            )
            raise KeyboardInterrupt

    def _handle_fallback_execution(
        self,
        task: Optional[Union[str, Any]] = None,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        correct_answer: Optional[str] = None,
        streaming_callback: Optional[Callable[[str], None]] = None,
        original_error: Exception = None,
        *args,
        **kwargs,
    ) -> Any:
        """
        Handles fallback execution when the primary model fails.

        This method attempts to execute the task using fallback models when the primary
        model encounters an error. It will try each available fallback model in sequence
        until either the task succeeds or all fallback models are exhausted.

        Args:
            task (Optional[Union[str, Any]], optional): The task to be executed. Defaults to None.
            img (Optional[str], optional): The image to be processed. Defaults to None.
            imgs (Optional[List[str]], optional): The list of images to be processed. Defaults to None.
            correct_answer (Optional[str], optional): The correct answer for continuous run mode. Defaults to None.
            streaming_callback (Optional[Callable[[str], None]], optional): Callback function to receive streaming tokens in real-time. Defaults to None.
            original_error (Exception): The original error that triggered the fallback. Defaults to None.
            *args: Additional positional arguments to be passed to the execution method.
            **kwargs: Additional keyword arguments to be passed to the execution method.

        Returns:
            Any: The result of the execution if successful.

        Raises:
            Exception: If all fallback models fail or no fallback models are available.
        """
        # Check if fallback models are available
        if not self.is_fallback_available():
            if self.verbose:
                logger.error(
                    f"Agent Name: {self.agent_name} [NO FALLBACK] failed with model '{self.get_current_model()}' "
                    f"and no fallback models are configured. Error: {str(original_error)[:100]}{'...' if len(str(original_error)) > 100 else ''}"
                )
            self._handle_run_error(original_error)
            return None

        # Try to switch to the next fallback model
        if not self.switch_to_next_model():
            if self.verbose:
                logger.error(
                    f"Agent Name: {self.agent_name} [FALLBACK EXHAUSTED] has exhausted all available models. "
                    f"Tried {len(self.get_available_models())} models: {self.get_available_models()}"
                )
            self._handle_run_error(original_error)
            return None

        # Log fallback attempt
        if self.verbose:
            logger.warning(
                f"Agent Name: {self.agent_name} [FALLBACK] failed with model '{self.get_current_model()}'. "
                f"Switching to fallback model '{self.get_current_model()}' (attempt {self.current_model_index + 1}/{len(self.get_available_models())})"
            )

        try:
            # Recursive call to run() with the new model
            result = self.run(
                task=task,
                img=img,
                imgs=imgs,
                correct_answer=correct_answer,
                streaming_callback=streaming_callback,
                *args,
                **kwargs,
            )

            if self.verbose:
                # Log successful completion with fallback model
                logger.info(
                    f"Agent Name: {self.agent_name} [FALLBACK SUCCESS] successfully completed task "
                    f"using fallback model '{self.get_current_model()}'"
                )
            return result

        except Exception as fallback_error:
            logger.error(
                f"Agent Name: {self.agent_name} Fallback model '{self.get_current_model()}' also failed: {fallback_error}"
            )

            # Try the next fallback model recursively
            return self._handle_fallback_execution(
                task=task,
                img=img,
                imgs=imgs,
                correct_answer=correct_answer,
                streaming_callback=streaming_callback,
                original_error=original_error,
                *args,
                **kwargs,
            )

    def run_batched(
        self,
        tasks: List[str],
        imgs: List[str] = None,
        *args,
        **kwargs,
    ):
        """
        Run a batch of tasks concurrently.

        Args:
            tasks (List[str]): List of tasks to run.
            imgs (List[str], optional): List of images to run. Defaults to None.
            *args: Additional positional arguments to be passed to the execution method.
            **kwargs: Additional keyword arguments to be passed to the execution method.

        Returns:
            List[Any]: List of results from each task execution.
        """
        return [
            self.run(task=task, imgs=imgs, *args, **kwargs)
            for task, imgs in zip(tasks, imgs)
        ]

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

    def talk_to(
        self, agent: Any, task: str, img: str = None, *args, **kwargs
    ) -> Any:
        """
        Talk to another agent.
        """
        # return agent.run(f"{agent.agent_name}: {task}", img, *args, **kwargs)
        output = self.run(
            f"{self.agent_name}: {task}", img, *args, **kwargs
        )

        return agent.run(
            task=f"From {self.agent_name}: Message: {output}",
            img=img,
            *args,
            **kwargs,
        )

    def talk_to_multiple_agents(
        self,
        agents: List[Union[Any, Callable]],
        task: str,
        *args,
        **kwargs,
    ) -> Any:
        """
        Talk to multiple agents.
        """
        # o# Use the existing executor from self.executor or create a new one if needed
        with ThreadPoolExecutor() as executor:
            # Create futures for each agent conversation
            futures = [
                executor.submit(
                    self.talk_to, agent, task, *args, **kwargs
                )
                for agent in agents
            ]

            # Wait for all futures to complete and collect results
            outputs = []
            for future in futures:
                try:
                    result = future.result()
                    outputs.append(result)
                except Exception as e:
                    logger.error(f"Error in agent communication: {e}")
                    outputs.append(
                        None
                    )  # or handle error case as needed

        return outputs

    def get_agent_role(self) -> str:
        """
        Get the role of the agent.
        """
        return self.role

    def pretty_print(self, response: str, loop_count: int):
        """Print the response in a formatted panel"""
        # Handle None response
        if response is None:
            response = "No response generated"

        if self.streaming_on:
            pass

        if self.print_on:
            formatter.print_panel(
                response,
                f"Agent Name {self.agent_name} [Max Loops: {loop_count} ]",
            )

    def parse_llm_output(self, response: Any):
        """Parse and standardize the output from the LLM.

        Args:
            response (Any): The response from the LLM in any format

        Returns:
            str: Standardized string output

        Raises:
            ValueError: If the response format is unexpected and can't be handled
        """
        try:

            if isinstance(response, dict):
                if "choices" in response:
                    return response["choices"][0]["message"][
                        "content"
                    ]
                return json.dumps(
                    response
                )  # Convert other dicts to string

            elif isinstance(response, BaseModel):
                response = response.model_dump()

            # Handle List[BaseModel] responses
            elif (
                isinstance(response, list)
                and response
                and isinstance(response[0], BaseModel)
            ):
                return [item.model_dump() for item in response]

            return response

        except AgentChatCompletionResponse as e:
            logger.error(f"Error parsing LLM output: {e}")
            raise ValueError(
                f"Failed to parse LLM output: {type(response)}"
            )

    def sentiment_and_evaluator(self, response: str):
        if self.evaluator:
            logger.info("Evaluating response...")

            evaluated_response = self.evaluator(response)
            self.short_memory.add(
                role="Evaluator",
                content=evaluated_response,
            )

        # Sentiment analysis
        if self.sentiment_analyzer:
            logger.info("Analyzing sentiment...")
            self.sentiment_analysis_handler(response)

    def output_cleaner_op(self, response: str):
        # Apply the cleaner function to the response
        if self.output_cleaner is not None:
            logger.info("Applying output cleaner to response.")

            response = self.output_cleaner(response)

            logger.info(f"Response after output cleaner: {response}")

            self.short_memory.add(
                role="Output Cleaner",
                content=response,
            )

    def mcp_tool_handling(
        self, response: any, current_loop: Optional[int] = 0
    ):
        try:

            if exists(self.mcp_url):
                # Execute the tool call
                tool_response = asyncio.run(
                    execute_tool_call_simple(
                        response=response,
                        server_path=self.mcp_url,
                    )
                )
            elif exists(self.mcp_config):
                # Execute the tool call
                tool_response = asyncio.run(
                    execute_tool_call_simple(
                        response=response,
                        connection=self.mcp_config,
                    )
                )
            elif exists(self.mcp_urls):
                tool_response = execute_multiple_tools_on_multiple_mcp_servers_sync(
                    responses=response,
                    urls=self.mcp_urls,
                    output_type="json",
                )
                # tool_response = format_data_structure(tool_response)

                # print(f"Multiple MCP Tool Response: {tool_response}")
            else:
                raise AgentMCPConnectionError(
                    "mcp_url must be either a string URL or MCPConnection object"
                )

            # Get the text content from the tool response
            # execute_tool_call_simple returns a string directly, not an object with content attribute
            text_content = f"MCP Tool Response: \n\n {json.dumps(tool_response, indent=2)}"

            if self.print_on is True:
                formatter.print_panel(
                    content=text_content,
                    title="MCP Tool Response: 🛠️",
                    style="green",
                )

            # Add to the memory
            self.short_memory.add(
                role="Tool Executor",
                content=text_content,
            )

            # Create a temporary LLM instance without tools for the follow-up call
            try:
                temp_llm = self.temp_llm_instance_for_tool_summary()

                summary = temp_llm.run(
                    task=self.short_memory.get_str()
                )
            except Exception as e:
                logger.error(
                    f"Error calling LLM after MCP tool execution: {e}"
                )
                # Fallback: provide a default summary
                summary = "I successfully executed the MCP tool and retrieved the information above."

            if self.print_on is True:
                self.pretty_print(summary, loop_count=current_loop)

            # Add to the memory
            self.short_memory.add(
                role=self.agent_name, content=summary
            )
        except AgentMCPToolError as e:
            logger.error(f"Error in MCP tool: {e}")
            raise e

    def temp_llm_instance_for_tool_summary(self):
        return LiteLLM(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            system_prompt=self.system_prompt,
            stream=False,  # Always disable streaming for tool summaries
            tools_list_dictionary=None,
            parallel_tool_calls=False,
            base_url=self.llm_base_url,
            api_key=self.llm_api_key,
        )

    def get_available_models(self) -> List[str]:
        """
        Get the list of available models including primary and fallback models.

        Returns:
            List[str]: List of model names in order of preference
        """
        models = []

        # If fallback_models is specified, use it as the primary list
        if self.fallback_models:
            models = self.fallback_models.copy()
        else:
            # Otherwise, build the list from individual parameters
            if self.model_name:
                models.append(self.model_name)
            if (
                self.fallback_model_name
                and self.fallback_model_name not in models
            ):
                models.append(self.fallback_model_name)

        return models

    def get_current_model(self) -> str:
        """
        Get the current model being used.

        Returns:
            str: Current model name
        """
        available_models = self.get_available_models()
        if self.current_model_index < len(available_models):
            return available_models[self.current_model_index]
        return (
            available_models[0] if available_models else "gpt-4o-mini"
        )

    def switch_to_next_model(self) -> bool:
        """
        Switch to the next available model in the fallback list.

        Returns:
            bool: True if successfully switched to next model, False if no more models available
        """
        available_models = self.get_available_models()

        if self.current_model_index + 1 < len(available_models):
            previous_model = (
                available_models[self.current_model_index]
                if self.current_model_index < len(available_models)
                else "Unknown"
            )
            self.current_model_index += 1
            new_model = available_models[self.current_model_index]

            # Always log model switches
            logger.warning(
                f"🔄 [MODEL SWITCH] Agent '{self.agent_name}' switching from '{previous_model}' to fallback model: '{new_model}' "
                f"(attempt {self.current_model_index + 1}/{len(available_models)})"
            )

            # Update the model name and reinitialize LLM
            self.model_name = new_model
            self.llm = self.llm_handling()

            return True
        else:
            logger.error(
                f"❌ [NO MORE MODELS] Agent '{self.agent_name}' has exhausted all available models. "
                f"Tried {len(available_models)} models: {available_models}"
            )
            return False

    def reset_model_index(self) -> None:
        """Reset the model index to use the primary model."""
        self.current_model_index = 0
        available_models = self.get_available_models()
        if available_models:
            self.model_name = available_models[0]
            self.llm = self.llm_handling()

    def is_fallback_available(self) -> bool:
        """
        Check if fallback models are available.

        Returns:
            bool: True if fallback models are configured
        """
        available_models = self.get_available_models()
        return len(available_models) > 1

    def execute_tools(self, response: any, loop_count: int):
        # Handle None response gracefully
        if response is None:
            logger.warning(
                f"Cannot execute tools with None response in loop {loop_count}. "
                "This may indicate the LLM did not return a valid response."
            )
            return

        try:
            output = self.tool_struct.execute_function_calls_from_api_response(
                response
            )
        except Exception as e:
            # Retry the tool call
            output = self.tool_struct.execute_function_calls_from_api_response(
                response
            )

            if output is None:
                logger.error(f"Error executing tools: {e}")
                raise e

        self.short_memory.add(
            role="Tool Executor",
            content=format_data_structure(output),
        )

        if self.print_on is True:
            if self.show_tool_execution_output is True:

                self.pretty_print(
                    f"Tool Executed Successfully [{time.strftime('%H:%M:%S')}] \n\nTool Output: {format_data_structure(output)}",
                    loop_count,
                )
            else:
                self.pretty_print(
                    f"Tool Executed Successfully [{time.strftime('%H:%M:%S')}]",
                    loop_count,
                )

        # Now run the LLM again without tools - create a temporary LLM instance
        # instead of modifying the cached one
        # Create a temporary LLM instance without tools for the follow-up call
        if self.tool_call_summary is True:
            temp_llm = self.temp_llm_instance_for_tool_summary()

            tool_response = temp_llm.run(
                f"""
                Please analyze and summarize the following tool execution output in a clear and concise way. 
                Focus on the key information and insights that would be most relevant to the user's original request.
                If there are any errors or issues, highlight them prominently.
                
                Tool Output:
                {output}
                """
            )

            self.short_memory.add(
                role=self.agent_name,
                content=tool_response,
            )

            if self.print_on is True:
                self.pretty_print(
                    tool_response,
                    loop_count,
                )

    def list_output_types(self):
        return OutputType

    def run_multiple_images(
        self, task: str, imgs: List[str], *args, **kwargs
    ):
        """
        Run the agent with multiple images using concurrent processing.

        Args:
            task (str): The task to be performed on each image.
            imgs (List[str]): List of image paths or URLs to process.
            *args: Additional positional arguments to pass to the agent's run method.
            **kwargs: Additional keyword arguments to pass to the agent's run method.

        Returns:
            List[Any]: A list of outputs generated for each image in the same order as the input images.

        Examples:
            >>> agent = Agent()
            >>> outputs = agent.run_multiple_images(
            ...     task="Describe what you see in this image",
            ...     imgs=["image1.jpg", "image2.png", "image3.jpeg"]
            ... )
            >>> print(f"Processed {len(outputs)} images")
            Processed 3 images

        Raises:
            Exception: If an error occurs while processing any of the images.
        """
        # Calculate number of workers as 95% of available CPU cores
        cpu_count = os.cpu_count()
        max_workers = max(1, int(cpu_count * 0.95))

        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all image processing tasks
            future_to_img = {
                executor.submit(
                    self.run, task=task, img=img, *args, **kwargs
                ): img
                for img in imgs
            }

            # Collect results in order
            outputs = []
            for future in future_to_img:
                try:
                    output = future.result()
                    outputs.append(output)
                except Exception as e:
                    logger.error(f"Error processing image: {e}")
                    outputs.append(
                        None
                    )  # or raise the exception based on your preference

        # Combine the outputs into a single string if summarization is enabled
        if self.summarize_multiple_images is True:
            output = "\n".join(outputs)

            prompt = f"""
            You have already analyzed {len(outputs)} images and provided detailed descriptions for each one. 
            Now, based on your previous analysis of these images, create a comprehensive report that:

            1. Synthesizes the key findings across all images
            2. Identifies common themes, patterns, or relationships between the images
            3. Provides an overall summary that captures the most important insights
            4. Highlights any notable differences or contrasts between the images

            Here are your previous analyses of the images:
            {output}

            Please create a well-structured report that brings together your insights from all {len(outputs)} images.
            """

            outputs = self.run(task=prompt, *args, **kwargs)

        return outputs

    def continuous_run_with_answer(
        self,
        task: str,
        img: Optional[str] = None,
        correct_answer: str = None,
        max_attempts: int = 10,
    ):
        """
        Run the agent with the task until the correct answer is provided.

        Args:
            task (str): The task to be performed
            correct_answer (str): The correct answer that must be found in the response
            max_attempts (int): Maximum number of attempts before giving up (default: 10)

        Returns:
            str: The response containing the correct answer

        Raises:
            Exception: If max_attempts is reached without finding the correct answer
        """
        attempts = 0

        while attempts < max_attempts:
            attempts += 1

            if self.verbose:
                logger.info(
                    f"Attempt {attempts}/{max_attempts} to find correct answer"
                )

            response = self._run(task=task, img=img)

            # Check if the correct answer is in the response (case-insensitive)
            if correct_answer.lower() in response.lower():
                if self.verbose:
                    logger.info(
                        f"Correct answer found on attempt {attempts}"
                    )
                return response
            else:
                # Add feedback to help guide the agent
                feedback = "Your previous response was incorrect. Think carefully about the question and ensure your response directly addresses what was asked."
                self.short_memory.add(role="User", content=feedback)

                if self.verbose:
                    logger.info(
                        f"Correct answer not found. Expected: '{correct_answer}'"
                    )

        # If we reach here, we've exceeded max_attempts
        raise Exception(
            f"Failed to find correct answer '{correct_answer}' after {max_attempts} attempts"
        )

    def tool_execution_retry(self, response: any, loop_count: int):
        """
        Execute tools with retry logic for handling failures.

        This method attempts to execute tools based on the LLM response. If the response
        is None, it logs a warning and skips execution. If an exception occurs during
        tool execution, it logs the error with full traceback and retries the operation
        using the configured retry attempts.

        Args:
            response (any): The response from the LLM that may contain tool calls to execute.
                          Can be None if the LLM failed to provide a valid response.
            loop_count (int): The current iteration loop number for logging and debugging purposes.

        Returns:
            None

        Raises:
            Exception: Re-raises any exception that occurs during tool execution after
                      all retry attempts have been exhausted.

        Note:
            - Uses self.tool_retry_attempts for the maximum number of retry attempts
            - Logs detailed error information including agent name and loop count
            - Skips execution gracefully if response is None
        """
        try:
            if response is not None:
                self.execute_tools(
                    response=response,
                    loop_count=loop_count,
                )
            else:
                logger.warning(
                    f"Agent '{self.agent_name}' received None response from LLM in loop {loop_count}. "
                    f"This may indicate an issue with the model or prompt. Skipping tool execution."
                )
        except AgentToolExecutionError as e:
            logger.error(
                f"Agent '{self.agent_name}' encountered error during tool execution in loop {loop_count}: {str(e)}. "
                f"Full traceback: {traceback.format_exc()}. "
                f"Attempting to retry tool execution with 3 attempts"
            )
