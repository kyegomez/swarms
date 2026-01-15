import asyncio
import json
import os
import random
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
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
from swarms.prompts.autonomous_agent_prompt import (
    AUTONOMOUS_AGENT_SYSTEM_PROMPT,
)
from swarms.prompts.handoffs_prompt import get_handoffs_prompt
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
from swarms.schemas.base_schemas import (
    AgentChatCompletionResponse,
)
from swarms.schemas.mcp_schemas import (
    MCPConnection,
    MultipleMCPConnections,
)
from swarms.structs.agent_roles import agent_roles
from swarms.structs.autonomous_loop_utils import (
    MAX_PLANNING_ATTEMPTS,
    MAX_SUBTASK_ITERATIONS,
    MAX_SUBTASK_LOOPS,
    create_file_tool,
    delete_file_tool,
    get_autonomous_planning_tools,
    get_execution_prompt,
    get_planning_prompt,
    get_summary_prompt,
    list_directory_tool,
    read_file_tool,
    respond_to_user_tool,
    update_file_tool,
)
from swarms.structs.conversation import Conversation
from swarms.structs.ma_utils import set_random_models_for_agents
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
from swarms.tools.handoffs_tool import handoff_task
from swarms.tools.handoffs_tool_schema import get_handoff_tool_schema
from swarms.tools.mcp_client_tools import (
    execute_multiple_tools_on_multiple_mcp_servers_sync,
    execute_tool_call_simple,
    get_mcp_tools_sync,
    get_tools_for_multiple_mcp_servers,
)
from swarms.tools.py_func_to_openai_func_str import (
    convert_multiple_functions_to_openai_function_schema,
)
from swarms.utils.dynamic_context_window import dynamic_auto_chunking
from swarms.utils.fetch_prompts_marketplace import (
    fetch_prompts_from_marketplace,
)
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
from swarms.utils.swarms_marketplace_utils import (
    add_prompt_to_marketplace,
)
from swarms.utils.workspace_utils import get_workspace_dir
from swarms.utils.check_all_model_max_tokens import (
    get_single_model_max_tokens,
)
from swarms.structs.dynamic_skills_loader import DynamicSkillsLoader


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
        multi_modal (bool): Enable multimodal
        tokenizer (Any): The tokenizer
        long_term_memory (BaseVectorDatabase): The long term memory
        fallback_model_name (str): The fallback model name to use if primary model fails
        fallback_models (List[str]): List of model names to try in order. First model is primary, rest are fallbacks
        preset_stopping_token (bool): Enable preset stopping token
        streaming_on (bool): Enable basic streaming with formatted panels
        stream (bool): Enable detailed token-by-token streaming with metadata (citations, tokens used, etc.)
        streaming_callback (Optional[Callable[[str], None]]): Callback function to receive streaming tokens in real-time. Defaults to None.
        verbose (bool): Enable verbose mode
        stopping_func (Callable): The stopping function
        custom_exit_command (str): The custom exit command
        custom_tools_prompt (Callable): The custom tools prompt
        tool_schema (ToolUsageType): The tool schema
        output_type (agent_output_type): The output type. Supported: 'str', 'string', 'list', 'json', 'dict', 'yaml', 'xml'.
        function_calling_type (str): The function calling type
        output_cleaner (Callable): The output cleaner function
        function_calling_format_type (str): The function calling format type
        list_base_models (List[BaseModel]): The list of base models
        metadata_output_type (str): The metadata output type
        state_save_file_type (str): The state save file type
        tool_choice (str): The tool choice
        rules (str): The rules
        planning_prompt (str): The planning prompt
        custom_planning_prompt (str): The custom planning prompt
        memory_chunk_size (int): The memory chunk size
        tool_system_prompt (str): The tool system prompt
        max_tokens (int): The maximum number of tokens
        temperature (float): The temperature
        workspace_dir (str, optional): Ignored - workspace directory is always read from
            the 'workspace_dir' environment variable. Defaults to 'agent_workspace' if
            the environment variable is not set.
        timeout (int): The timeout
        artifacts_on (bool): Enable artifacts
        artifacts_output_path (str): The artifacts output path
        artifacts_file_extension (str): The artifacts file extension (.pdf, .md, .txt, )
        marketplace_prompt_id (str): The unique UUID identifier of a prompt from the Swarms marketplace.
            When provided, the agent will automatically fetch and load the prompt from the marketplace
            as the system prompt. This enables one-line prompt loading from the Swarms marketplace.
            Requires the SWARMS_API_KEY environment variable to be set.
        skills_dir (str): Path to directory containing Agent Skills in SKILL.md format.
            Implements Anthropic's Agent Skills framework for modular, composable capabilities.
            Each subdirectory should contain a SKILL.md file with YAML frontmatter (name, description)
            and markdown instructions. Skills are auto-loaded into system prompt for context-aware activation.
            Example: skills_dir="./skills" loads from ./skills/*/SKILL.md

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
        load_skills_metadata: Load Agent Skills metadata from directory
        load_full_skill: Load complete skill content (Tier 2 loading)
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

    >>> # Detailed token streaming example
    >>> agent = Agent(model_name="gpt-4.1", max_loops=1, stream=True)
    >>> response = agent.run("Tell me a story.")  # Will stream each token with detailed metadata
    >>> print(response)  # Final complete response

    >>> # Fallback model example
    >>> agent = Agent(
    ...     fallback_models=["gpt-4.1", "gpt-4o-mini", "gpt-3.5-turbo"],
    ...     max_loops=1
    ... )
    >>> response = agent.run("Generate a report on the financials.")
    >>> # Will try gpt-4o first, then gpt-4o-mini, then gpt-3.5-turbo if each fails

    >>> # Marketplace prompt example - load a prompt in one line
    >>> agent = Agent(
    ...     model_name="gpt-4.1",
    ...     marketplace_prompt_id="550e8400-e29b-41d4-a716-446655440000",
    ...     max_loops=1
    ... )
    >>> response = agent.run("Execute the marketplace prompt task")
    >>> # The agent automatically loads the system prompt from the Swarms marketplace

    """

    def __init__(
        self,
        id: Optional[str] = agent_id(),
        agent_name: Optional[str] = "swarm-worker-01",
        agent_description: Optional[
            str
        ] = "An autonomous agent that can perform tasks and learn from experience powered by Swarms",
        system_prompt: Optional[str] = AGENT_SYSTEM_PROMPT_3,
        llm: Optional[Any] = None,
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
        # TODO: Change to callable, then parse the callable to a string
        tools: List[Callable] = None,
        dynamic_temperature_enabled: Optional[bool] = False,
        sop: Optional[str] = None,
        sop_list: Optional[List[str]] = None,
        saved_state_path: Optional[str] = None,
        autosave: Optional[bool] = False,
        context_length: Optional[int] = None,
        transforms: Optional[Union[TransformConfig, dict]] = None,
        user_name: Optional[str] = "Human",
        multi_modal: Optional[bool] = None,
        tokenizer: Optional[Any] = None,
        long_term_memory: Optional[Union[Callable, Any]] = None,
        fallback_model_name: Optional[str] = None,
        fallback_models: Optional[List[str]] = None,
        preset_stopping_token: Optional[bool] = False,
        streaming_on: Optional[bool] = False,
        stream: Optional[bool] = False,
        streaming_callback: Optional[Callable[[str], None]] = None,
        verbose: Optional[bool] = False,
        stopping_func: Optional[Callable] = None,
        custom_exit_command: Optional[str] = "exit",
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
        tool_choice: str = "auto",
        rules: str = None,  # type: ignore
        planning_prompt: Optional[str] = None,
        custom_planning_prompt: str = None,
        memory_chunk_size: int = 2000,
        tool_system_prompt: str = tool_sop_prompt(),
        max_tokens: int = 4096,
        temperature: float = 0.5,
        timeout: Optional[int] = None,
        tags: Optional[List[str]] = None,
        auto_generate_prompt: bool = False,
        rag_every_loop: bool = False,
        plan_enabled: bool = False,
        artifacts_on: bool = False,
        artifacts_output_path: str = None,
        artifacts_file_extension: str = None,
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
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        tool_call_summary: bool = True,
        summarize_multiple_images: bool = False,
        tool_retry_attempts: int = 3,
        reasoning_prompt_on: bool = True,
        dynamic_context_window: bool = True,
        show_tool_execution_output: bool = True,
        reasoning_effort: str = None,
        thinking_tokens: int = None,
        reasoning_enabled: bool = False,
        handoffs: Optional[Union[Sequence[Callable], Any]] = None,
        capabilities: Optional[List[str]] = None,
        mode: Literal["interactive", "fast", "standard"] = "standard",
        publish_to_marketplace: bool = False,
        use_cases: Optional[List[Dict[str, Any]]] = None,
        marketplace_prompt_id: Optional[str] = None,
        skills_dir: Optional[str] = None,
        *args,
        **kwargs,
    ):
        # super().__init__(*args, **kwargs)
        self.id = id
        self.skills_dir = skills_dir
        self.skills_metadata = []
        self.llm = llm
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
        self.multi_modal = multi_modal
        self.tokenizer = tokenizer
        self.long_term_memory = long_term_memory
        self.preset_stopping_token = preset_stopping_token
        self.streaming_on = streaming_on
        self.stream = stream
        self.streaming_callback = streaming_callback
        self.verbose = verbose
        self.stopping_func = stopping_func
        self.custom_exit_command = custom_exit_command
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
        self.tool_choice = tool_choice
        self.planning_prompt = planning_prompt
        self.custom_planning_prompt = custom_planning_prompt
        self.rules = rules
        self.custom_tools_prompt = custom_tools_prompt
        self.memory_chunk_size = memory_chunk_size
        self.tool_system_prompt = tool_system_prompt
        self.max_tokens = max_tokens
        self.temperature = temperature
        # Always use environment variable for workspace_dir, ignore user input
        # Fallback to default if environment variable is not set
        self.workspace_dir = get_workspace_dir()
        self.timeout = timeout
        self.tags = tags
        self.use_cases = use_cases
        self.name = agent_name
        self.description = agent_description
        self.auto_generate_prompt = auto_generate_prompt
        self.rag_every_loop = rag_every_loop
        self.plan_enabled = plan_enabled
        self.artifacts_on = artifacts_on
        self.artifacts_output_path = artifacts_output_path
        self.artifacts_file_extension = artifacts_file_extension
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
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.tool_call_summary = tool_call_summary
        self.summarize_multiple_images = summarize_multiple_images
        self.tool_retry_attempts = tool_retry_attempts
        self.reasoning_prompt_on = reasoning_prompt_on
        self.dynamic_context_window = dynamic_context_window
        self.show_tool_execution_output = show_tool_execution_output
        self.mcp_configs = mcp_configs
        self.reasoning_effort = reasoning_effort
        self.thinking_tokens = thinking_tokens
        self.reasoning_enabled = reasoning_enabled
        self.fallback_model_name = fallback_model_name
        self.handoffs = handoffs
        self.capabilities = capabilities
        self.mode = mode
        self.publish_to_marketplace = publish_to_marketplace
        self.marketplace_prompt_id = marketplace_prompt_id

        # Yes, this works: it sets context_length based on the model_name, defaulting to 16000 if not set.
        self.context_length = (
            get_single_model_max_tokens(model_name)
            if model_name
            else 16000
        )

        if self.max_loops == "auto":
            self.system_prompt = None

            self.system_prompt = AUTONOMOUS_AGENT_SYSTEM_PROMPT

        else:
            self.system_prompt = AGENT_SYSTEM_PROMPT_3

        # Initialize autonomous loop tracking structures
        self.autonomous_subtasks = []  # List of subtasks from plan
        self.current_subtask_index = (
            0  # Current subtask being executed
        )
        self.subtask_status = {}  # Track status of each subtask
        self.plan_created = False  # Whether a plan has been created
        self.think_call_count = (
            0  # Track consecutive think calls to prevent loops
        )
        self.max_consecutive_thinks = (
            2  # Maximum consecutive think calls
        )

        # Load prompt from marketplace if marketplace_prompt_id is provided
        if self.marketplace_prompt_id:
            self._load_prompt_from_marketplace()

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

        # Add handoff tool if handoffs are configured
        if exists(self.handoffs):
            handoff_tool_schema = get_handoff_tool_schema()
            if self.tools_list_dictionary is None:
                self.tools_list_dictionary = []
            self.tools_list_dictionary.extend(handoff_tool_schema)

            # Add handoff prompt to system prompt
            agent_registry = self._get_agent_registry()
            if agent_registry:
                handoff_prompt = get_handoffs_prompt(
                    list(agent_registry.values())
                )
                self.system_prompt += "\n\n" + handoff_prompt

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

    def handle_skills(self, task: Optional[str] = None):
        """
        Handle skills loading based on task similarity.

        Args:
            task: Optional task description. If provided, loads skills dynamically
                  based on similarity to the task. If not provided, loads all skills statically.
        """
        if task is not None:
            # Dynamic skills loading based on task
            self._load_dynamic_skills_for_task(task)
        else:
            # Static skills loading (original behavior)
            self._load_static_skills()

    def _load_static_skills(self):
        """Load all skills statically (original behavior)."""
        skills_prompt = (
            "\n\n# Available Skills\n\n"
            "You have access to the following specialized skills. "
            "Follow their instructions when relevant:\n\n"
        )

        self.system_prompt += skills_prompt

        self.skills_metadata = self.load_skills_metadata(
            self.skills_dir
        )

        if self.skills_metadata:
            self.system_prompt += self._build_skills_prompt(
                self.skills_metadata
            )
            logger.info(
                f"Loaded {len(self.skills_metadata)} skills from {self.skills_dir}"
            )

    def _load_dynamic_skills_for_task(self, task: str):
        """
        Load skills dynamically based on task similarity.

        Args:
            task: The task description to match skills against
        """
        # Initialize the dynamic skills loader if not already done
        if (
            not hasattr(self, "dynamic_skills_loader")
            or self.dynamic_skills_loader is None
        ):
            self.dynamic_skills_loader = DynamicSkillsLoader(
                self.skills_dir
            )

        logger.info(
            f"Loading dynamic skills for task: {task[:100]}..."
        )

        relevant_skills = (
            self.dynamic_skills_loader.load_relevant_skills(task)
        )

        if relevant_skills:
            skills_prompt = (
                "\n\n# Available Skills\n\n"
                "You have access to the following specialized skills. "
                "Follow their instructions when relevant:\n\n"
            )
            self.system_prompt += skills_prompt
            self.system_prompt += self._build_skills_prompt(
                relevant_skills
            )
            logger.info(
                f"Dynamically loaded {len(relevant_skills)} relevant skills for task: {task[:100]}..."
            )
        else:
            logger.info(
                f"No relevant skills found for task: {task[:100]}..."
            )

    def _get_agent_workspace_dir(self) -> str:
        """
        Get the agent-specific workspace directory path.

        Creates a unique subdirectory for each agent instance in the format:
        workspace_dir/agents/{name-of-agent}-{uuid}/

        Returns:
            str: The full path to the agent-specific workspace directory.
        """
        # Generate a sanitized agent name in "name-of-agent" format (lowercase with hyphens)
        if self.agent_name:
            # Convert to lowercase and replace spaces/special chars with hyphens
            safe_agent_name = (
                self.agent_name.lower()
                .replace(" ", "-")
                .replace("_", "-")
                .replace("/", "-")
                .replace("\\", "-")
                .replace(":", "-")
                .replace("*", "-")
                .replace("?", "-")
                .replace('"', "-")
                .replace("<", "-")
                .replace(">", "-")
                .replace("|", "-")
                # Remove multiple consecutive hyphens
                .replace("--", "-")
                .replace("--", "-")
                .strip("-")
            )
        else:
            safe_agent_name = "agent"

        # Extract UUID from agent ID
        if self.id.startswith("agent-"):
            agent_uuid = self.id.replace("agent-", "")
        else:
            agent_uuid = self.id

        # Limit UUID length for directory name (use last 12 chars for brevity)
        agent_uuid_short = (
            agent_uuid[-12:] if len(agent_uuid) > 12 else agent_uuid
        )

        # Create directory name: {name-of-agent}-{uuid} (no "agent-" prefix)
        dir_name = f"{safe_agent_name}-{agent_uuid_short}"

        # Create full path: workspace_dir/agents/{name-of-agent}-{uuid}/
        agents_dir = os.path.join(self.workspace_dir, "agents")
        agent_workspace = os.path.join(agents_dir, dir_name)

        # Ensure directory exists
        os.makedirs(agent_workspace, exist_ok=True)

        return agent_workspace

    def _get_agent_registry(self) -> Dict[str, Any]:
        """
        Get the agent registry from handoffs configuration.

        Returns:
            Dict mapping agent names to agent instances.
        """
        agent_registry = {}
        if self.handoffs:
            if isinstance(self.handoffs, (list, tuple)):
                for agent in self.handoffs:
                    agent_name = getattr(
                        agent, "agent_name", str(agent)
                    )
                    agent_registry[agent_name] = agent
            elif isinstance(self.handoffs, dict):
                agent_registry = self.handoffs
        return agent_registry

    def _handoff_task_tool(
        self, handoffs: List[Dict[str, str]]
    ) -> str:
        """
        Tool handler for handoff_task function calls.

        This method processes handoff requests from the LLM and delegates tasks
        to other agents in the handoffs registry. It supports delegating to
        multiple agents concurrently and aggregates their responses.

        **Handoff Process:**
        1. Retrieves agent registry from handoffs configuration
        2. Validates that requested agents exist in the registry
        3. Delegates tasks to specified agents using handoff_task function
        4. Returns aggregated responses from all delegated agents

        **Handoff Request Format:**
        Each handoff request must contain:
        - agent_name (str): The name of the agent to delegate to (must exist in registry)
        - task (str): The specific task to be delegated to that agent
        - reasoning (str): Explanation of why this agent was selected for the task

        **Agent Registry:**
        The agent registry is built from:
        - List of Agent instances: Uses agent_name attribute
        - Dictionary: Uses keys as agent names
        - Empty if handoffs is not configured

        Args:
            handoffs (List[Dict[str, str]]): List of handoff requests. Each request
                is a dictionary containing:
                - agent_name (str): The name of the agent to delegate to.
                    Must match an agent in the handoffs registry.
                - task (str): The task to be delegated to that agent.
                - reasoning (str): Explanation of why this agent was selected.

        Returns:
            str: Aggregated response from all delegated agents. The format depends
                on the handoff_task implementation, typically a concatenated string
                of responses from each agent.

        Raises:
            KeyError: If an agent_name in handoffs doesn't exist in the registry.
            Exception: If handoff_task execution fails for any agent.

        Note:
            - Requires handoffs to be configured during agent initialization
            - Agent names must match exactly (case-sensitive)
            - Multiple agents can be delegated to concurrently
            - Handoff results are automatically added to conversation memory

        Examples:
            >>> # Configure handoffs
            >>> agent1 = Agent(agent_name="researcher")
            >>> agent2 = Agent(agent_name="writer")
            >>> main_agent = Agent(handoffs=[agent1, agent2])
            >>>
            >>> # LLM can now call handoff_task
            >>> handoffs = [
            ...     {
            ...         "agent_name": "researcher",
            ...         "task": "Research the topic",
            ...         "reasoning": "This agent specializes in research"
            ...     }
            ... ]
            >>> result = main_agent._handoff_task_tool(handoffs)
        """
        agent_registry = self._get_agent_registry()
        return handoff_task(
            handoffs=handoffs,
            agent_registry=agent_registry,
        )

    def setup_tools(self):
        """
        Initialize the BaseTool structure for tool execution.

        This method creates a BaseTool instance that handles tool execution,
        validation, and management. The BaseTool structure is used throughout
        the agent's lifecycle for executing function calls from LLM responses.

        **BaseTool Functionality:**
        - Converts tool functions to executable format
        - Validates tool calls from LLM responses
        - Executes tools with proper error handling
        - Formats tool execution results
        - Supports parallel tool execution

        Args:
            None: Uses self.tools and self.verbose from instance.

        Returns:
            BaseTool: An initialized BaseTool instance configured with:
                - tools: List of user-provided tool functions
                - verbose: Verbosity setting for tool execution logging

        Note:
            - This method is called automatically during agent initialization
            - The BaseTool instance is stored in self.tool_struct
            - Tools must be callable Python functions
            - Tool functions should have proper type hints for schema generation

        Examples:
            >>> agent = Agent(tools=[my_function])
            >>> # setup_tools() is called automatically
            >>> # agent.tool_struct is now ready to execute tools
        """
        return BaseTool(
            tools=self.tools,
            verbose=self.verbose,
        )

    def tool_handling(self):
        """
        Process and integrate user-defined tools into the agent's tool system.

        This method converts user-provided tools (callable functions) into OpenAI
        function schema format and adds them to the agent's tools_list_dictionary.
        It preserves existing tools (e.g., handoff tools) and avoids duplicates.

        **Process:**
        1. Converts user tools to OpenAI function schema format
        2. Initializes tools_list_dictionary if None
        3. Tracks existing tool names to prevent duplicates
        4. Adds new tools that don't already exist
        5. Adds tools to conversation memory for LLM context

        **Tool Schema Format:**
        Tools are converted to OpenAI function calling format:
        {
            "type": "function",
            "function": {
                "name": "function_name",
                "description": "Function description",
                "parameters": {
                    "type": "object",
                    "properties": {...},
                    "required": [...]
                }
            }
        }

        **Duplicate Prevention:**
        The method checks tool names before adding to prevent duplicate tools.
        This is important when handoff tools or other system tools are already
        present in tools_list_dictionary.

        Args:
            None: Uses self.tools and self.tools_list_dictionary from instance.

        Returns:
            None: Modifies self.tools_list_dictionary and self.short_memory.

        Note:
            - This method is called automatically during agent initialization if tools are provided
            - Tools are added to conversation memory so the LLM knows what tools are available
            - The method preserves existing tools in tools_list_dictionary (e.g., handoff tools)
            - Tool names are case-sensitive for duplicate detection

        Raises:
            Exception: If tool conversion fails or tools cannot be added to memory.

        Examples:
            >>> def my_tool(query: str) -> str:
            ...     return f"Searching for {query}"
            >>> agent = Agent(tools=[my_tool])
            >>> # tool_handling() is called automatically during initialization
            >>> # The tool is now available for the LLM to use
        """
        # Convert all the tools into a list of dictionaries
        user_tools = (
            convert_multiple_functions_to_openai_function_schema(
                self.tools
            )
        )

        # Preserve existing tools in tools_list_dictionary (e.g., handoff tools)
        if self.tools_list_dictionary is None:
            self.tools_list_dictionary = []

        # Get existing tool names to avoid duplicates
        existing_tool_names = set()
        for tool in self.tools_list_dictionary:
            if isinstance(tool, dict) and "function" in tool:
                existing_tool_names.add(
                    tool["function"].get("name", "")
                )

        # Add user tools, avoiding duplicates
        for tool in user_tools:
            tool_name = tool.get("function", {}).get("name", "")
            if tool_name not in existing_tool_names:
                self.tools_list_dictionary.append(tool)
                existing_tool_names.add(tool_name)

        self.short_memory.add(
            role=self.agent_name,
            content=self.tools_list_dictionary,
        )

    def short_memory_init(self):
        # Compactly assemble initial prompt as a string with available fields
        prompt = ""

        if self.safety_prompt_on is True:
            prompt += SAFETY_PROMPT

        # Initialize the short term memory
        memory = Conversation(
            name=f"{self.agent_name}_id_{self.id}_conversation",
            system_prompt=prompt,
            user=self.user_name,
            rules=self.rules,
            token_count=False,
            message_id_on=True,
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

    def _load_prompt_from_marketplace(self) -> None:
        """
        Load a prompt from the Swarms marketplace using the marketplace_prompt_id.

        This method fetches the prompt content from the Swarms marketplace API
        and sets it as the agent's system prompt. If the agent_name and agent_description
        are not already set, they will be populated from the marketplace prompt data.

        The method uses the fetch_prompts_from_marketplace utility function to retrieve
        the prompt data, which includes the prompt name, description, and content.

        Raises:
            ValueError: If the prompt cannot be found in the marketplace.
            Exception: If there's an error fetching the prompt from the API.

        Note:
            Requires the SWARMS_API_KEY environment variable to be set for
            authenticated API access.
        """
        try:
            logger.info(
                f"Loading prompt from marketplace with ID: {self.marketplace_prompt_id}"
            )

            result = fetch_prompts_from_marketplace(
                prompt_id=self.marketplace_prompt_id,
                return_params_on=True,
            )

            if result is None:
                raise ValueError(
                    f"Prompt with ID '{self.marketplace_prompt_id}' not found in the marketplace. "
                    "Please verify the prompt ID is correct."
                )

            name, description, prompt = result

            # Set the system prompt from the marketplace
            if prompt:
                self.system_prompt += prompt
                logger.info(
                    f"Successfully loaded prompt '{name}' from marketplace"
                )

            # Optionally set agent name and description if not already set
            if name and self.agent_name == "swarm-worker-01":
                self.agent_name = name
                self.name = name

            if description and self.agent_description is None:
                self.agent_description = description
                self.description = description

            if self.print_on:
                self.pretty_print(
                    f"[Marketplace] Loaded prompt '{name}' from Swarms Marketplace",
                    loop_count=0,
                )

        except Exception as e:
            logger.error(
                f"Error loading prompt from marketplace: {e} Traceback: {traceback.format_exc()}"
            )
            raise

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

        This is the core execution method that manages the agent's reasoning and action loop.
        It handles the complete lifecycle of task execution, from initialization to completion.

        **Execution Flow:**

        1. **Initialization:**
           - Auto-generates prompt if enabled
           - Validates model supports required utilities (vision, function calling)
           - Adds task to conversation memory
           - Handles RAG query if long_term_memory is configured (once or every loop)

        2. **Planning (if enabled):**
           - Creates strategic plan using plan() method
           - Breaks down task into manageable steps

        3. **Main Loop:**
           - Runs for max_loops iterations (or until stopping condition)
           - Each iteration:
             * Handles RAG query if rag_every_loop=True
             * Applies dynamic temperature if enabled
             * Applies message transforms if configured
             * Calls LLM with task prompt
             * Parses and validates LLM response
             * Executes tools if tool calls are present
             * Handles MCP tools if configured
             * Handles handoff tool calls if configured
             * Checks stopping conditions
             * Handles interactive mode if enabled
             * Autosaves state if configured

        4. **Output Formatting:**
           - Formats output based on output_type configuration
           - Returns formatted result (string, list, JSON, dict, YAML, XML, etc.)

        **Stopping Conditions:**
        The loop stops when:
        - Maximum loops reached (if max_loops is an integer)
        - Stopping condition function returns True
        - Stopping function returns True
        - Interactive mode exit command entered
        - Error occurs after retry attempts

        **Error Handling:**
        - Retries LLM calls up to retry_attempts times
        - Autosaves state on errors if enabled
        - Logs detailed error information
        - Falls back to fallback models if configured

        **Memory Management:**
        - Adds task to conversation memory
        - Adds LLM responses to memory
        - Adds tool execution results to memory
        - Handles RAG queries and adds results to memory

        Args:
            task (Optional[Union[str, Any]]): The task or prompt for the agent to process.
                Can be a string or any format that can be converted to string. This is
                the main input that drives the agent's execution.
            img (Optional[str]): Optional image path or data to be processed by the agent.
                Used for vision-enabled models. Can be a file path or image data string.
            streaming_callback (Optional[Callable[[str], None]]): Optional callback function
                to receive streaming tokens in real-time. Useful for dashboard integration
                or real-time UI updates. Defaults to None.
            *args: Additional positional arguments passed to LLM calls. Used for extensibility.
            **kwargs: Additional keyword arguments passed to LLM calls. Used for extensibility.

        Returns:
            Any: The agent's output, formatted according to output_type configuration:
                - "str" or "string": String representation
                - "list": List format
                - "json": JSON string
                - "dict": Dictionary
                - "yaml": YAML string
                - "xml": XML string
                - "final": Comprehensive final summary (for autonomous loop)
                - Other types: As configured

        Raises:
            AgentRunError: If execution fails after all retry attempts.
            AgentLLMError: If LLM calls fail and no fallback models are available.
            KeyboardInterrupt: If interrupted by user (handles gracefully with autosave).

        Note:
            - This method is called by run() which handles autonomous loop routing
            - Autosave is performed at start, each loop, and on errors if enabled
            - Tool execution is handled automatically when tool calls are detected
            - MCP tools are handled automatically if MCP is configured
            - Handoff tools are handled automatically if handoffs are configured
            - Interactive mode allows user input between loops

        Examples:
            >>> # Simple text task
            >>> response = agent._run("What is the capital of France?")
            >>> print(response)

            >>> # Multimodal task
            >>> response = agent._run(
            ...     "Describe this image",
            ...     img="path/to/image.jpg"
            ... )

            >>> # With streaming callback
            >>> def on_token(token):
            ...     print(f"Token: {token}")
            >>> response = agent._run(
            ...     "Tell me a story",
            ...     streaming_callback=on_token
            ... )
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
                self._autosave_config_step(loop_count=0)

            while (
                self.max_loops == "auto"
                or loop_count < self.max_loops
            ):
                loop_count += 1

                # Autosave config at the start of each loop step
                if self.autosave:
                    self._autosave_config_step(loop_count=loop_count)

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
                            # Skip printing structured output (list of tool calls) here
                            # Function call visualization is handled in execute_tools
                            if isinstance(response, list):
                                # Tool calls will be visualized in execute_tools, skip here
                                pass
                            elif self.streaming_on:
                                pass
                            elif self.stream:
                                pass
                            else:
                                self.pretty_print(
                                    response, loop_count
                                )

                        # Handle handoff tool calls
                        if isinstance(response, list):
                            for tool_call in response:
                                if (
                                    isinstance(tool_call, dict)
                                    and tool_call.get(
                                        "function", {}
                                    ).get("name")
                                    == "handoff_task"
                                ):
                                    arguments = json.loads(
                                        tool_call["function"][
                                            "arguments"
                                        ]
                                    )
                                    handoffs_list = arguments.get(
                                        "handoffs", []
                                    )

                                    # Visualize handoff tool call
                                    if self.print_on:
                                        self._visualize_handoff_call(
                                            handoffs_list, tool_call
                                        )

                                    result = self._handoff_task_tool(
                                        handoffs=handoffs_list
                                    )
                                    # Add result to memory
                                    self.short_memory.add(
                                        role="Tool Executor",
                                        content=f"Handoff Result:\n{result}",
                                    )
                                    if self.print_on:
                                        delegated_agents = ", ".join(
                                            agent.get(
                                                "agent_name",
                                                "<unknown>",
                                            )
                                            for agent in handoffs_list
                                        )
                                        self.pretty_print(
                                            f"[Handoff] Delegated tasks to {len(handoffs_list)} agent(s): {delegated_agents}\nSuccessfully executed handoff_task function.",
                                            loop_count,
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

                        success = True  # Mark as successful to exit the retry loop

                        # Autosave config after successful step
                        if self.autosave:
                            self._autosave_config_step(
                                loop_count=loop_count
                            )

                    except (
                        BadRequestError,
                        InternalServerError,
                        AuthenticationError,
                        Exception,
                    ) as e:

                        if self.autosave is True:
                            log_agent_data(self.to_dict())
                            self.save()
                            self._autosave_config_step(
                                loop_count=loop_count
                            )

                        logger.error(
                            f"Attempt {attempt+1}/{self.retry_attempts}: Error generating response in loop {loop_count} for agent '{self.agent_name}': {str(e)} | Traceback: {traceback.format_exc()}"
                        )
                        attempt += 1

                if not success:

                    if self.autosave is True:
                        log_agent_data(self.to_dict())
                        self.save()
                        self._autosave_config_step(
                            loop_count=loop_count
                        )

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
                self._autosave_config_step(loop_count=loop_count)

            # Output formatting based on output_type
            return history_output_formatter(
                self.short_memory, type=self.output_type
            )

        except Exception as error:
            self._handle_run_error(error)

        except KeyboardInterrupt as error:
            # Save config on interrupt
            if self.autosave:
                try:
                    self._autosave_config_step(loop_count=None)
                except Exception:
                    pass  # Don't let autosave errors mask the interrupt
            self._handle_run_error(error)

    def _autosave_config_step(
        self, loop_count: Optional[int] = None
    ) -> None:
        """
        Save the agent's configuration dictionary to a JSON file in the agent-specific workspace directory.
        This method is called at each step of the agent's run to maintain an up-to-date
        configuration snapshot. It only runs when autosave is enabled.

        Args:
            loop_count (Optional[int]): The current loop count for logging purposes. Defaults to None.

        Note:
            This method handles errors gracefully to ensure it doesn't interrupt the main execution.
            The saved file will be named `config.json` in the agent-specific workspace directory:
            workspace_dir/agents/{name-of-agent}-{uuid}/config.json
        """
        if not self.autosave:
            return

        try:
            # Get agent-specific workspace directory
            agent_workspace = self._get_agent_workspace_dir()

            # Save as config.json in the agent-specific directory
            file_path = os.path.join(agent_workspace, "config.json")

            # Get the current configuration dictionary
            config_dict = self.to_dict()

            # Add metadata about when this was saved
            config_dict["_autosave_metadata"] = {
                "timestamp": time.strftime(
                    "%Y-%m-%d %H:%M:%S", time.localtime()
                ),
                "loop_count": loop_count,
                "agent_id": self.id,
                "agent_name": self.agent_name,
            }

            # Write to JSON file atomically (write to temp file first, then rename)
            temp_path = file_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(
                    config_dict, f, indent=2, ensure_ascii=False
                )

            # Atomic rename
            os.replace(temp_path, file_path)

            if self.verbose and loop_count is not None:
                logger.debug(
                    f"Autosaved config at loop {loop_count} to {file_path}"
                )

        except Exception as e:
            # Log error but don't raise - we don't want autosave to break execution
            logger.warning(
                f"Failed to autosave config step for agent '{self.agent_name}': {e}"
            )

    def _handle_run_error(self, error: any):
        if self.autosave is True:
            # Save full state
            self.save()
            log_agent_data(self.to_dict())
            # Also save config step on error
            self._autosave_config_step(loop_count=None)

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

    def _visualize_function_call(
        self,
        function_name: str,
        arguments: Dict[str, Any],
        result: str = None,
    ) -> None:
        """
        Visualize a function call using formatter.

        Args:
            function_name: Name of the function being called
            arguments: Arguments passed to the function
            result: Optional result of the function call
        """
        if not self.print_on:
            return

        # Format function call visualization
        call_content = f"Function: {function_name}\n\n"
        call_content += "Arguments:\n"
        for key, value in arguments.items():
            # Truncate long values for readability
            value_str = str(value)
            if len(value_str) > 200:
                value_str = value_str[:200] + "..."
            call_content += f"  {key}: {value_str}\n"

        if result:
            result_str = str(result)
            if len(result_str) > 500:
                result_str = result_str[:500] + "..."
            call_content += f"\nResult:\n{result_str}"

        formatter.print_panel(
            call_content,
            title=f"Agent: {self.agent_name} Function Call: {function_name}",
        )

    def _visualize_handoff_call(
        self,
        handoffs: List[Dict[str, str]],
        tool_call: Dict[str, Any] = None,
    ) -> None:
        """
        Visualize a handoff tool call with detailed information about all delegations.

        Args:
            handoffs: List of handoff requests, each containing agent_name, task, and reasoning
            tool_call: Optional tool call dictionary for additional metadata
        """
        if not self.print_on:
            return

        # Build visualization content
        call_content = "Function: handoff_task\n"
        call_content += f"Delegating to {len(handoffs)} agent(s)\n\n"

        if tool_call and tool_call.get("id"):
            call_content += f"Call ID: {tool_call.get('id')}\n\n"

        call_content += "Handoff Details:\n"
        call_content += "=" * 80 + "\n"

        for i, handoff in enumerate(handoffs, 1):
            agent_name = handoff.get("agent_name", "<unknown>")
            task = handoff.get("task", "")
            reasoning = handoff.get("reasoning", "")

            call_content += f"\n[{i}] Agent: {agent_name}\n"
            call_content += f"    Task: {task[:150]}{'...' if len(task) > 150 else ''}\n"
            call_content += f"    Reasoning: {reasoning[:150]}{'...' if len(reasoning) > 150 else ''}\n"
            if i < len(handoffs):
                call_content += "\n" + "-" * 80 + "\n"

        formatter.print_panel(
            call_content,
            title=f"Agent: {self.agent_name} Handoff Tool Call",
        )

    def _run_autonomous_loop(
        self,
        task: Optional[Union[str, Any]] = None,
        img: Optional[str] = None,
        streaming_callback: Optional[Callable[[str], None]] = None,
        *args,
        **kwargs,
    ) -> Any:
        """
        Execute the autonomous loop structure: plan -> execute subtasks -> summary.

        This method implements the optimized autonomous looping when max_loops="auto"
        and interactive=False. It follows a three-phase structure:

        **Phase 1: Planning**
        - Creates a detailed plan using the `create_plan` tool
        - Breaks down the task into subtasks with dependencies, priorities, and step IDs
        - Supports handoff delegation during planning if handoffs are configured
        - Maximum planning attempts are controlled by MAX_PLANNING_ATTEMPTS

        **Phase 2: Execution**
        - Executes each subtask in dependency order
        - For each subtask, runs a thinking -> tool actions -> observation loop
        - Supports both planning tools (think, subtask_done, complete_task) and user-defined tools
        - Prevents infinite thinking loops with max_consecutive_thinks limit
        - Each subtask has a maximum iteration limit (MAX_SUBTASK_LOOPS)
        - Overall execution has a maximum iteration limit (MAX_SUBTASK_ITERATIONS)

        **Phase 3: Summary**
        - Generates a comprehensive final summary when all subtasks are complete
        - Uses the `complete_task` tool or generates summary manually
        - Returns formatted output based on output_type configuration

        The method automatically integrates:
        - Planning tools (create_plan, think, subtask_done, complete_task, file operations)
        - Handoff tools (if handoffs are configured)
        - User-defined tools (added after planning phase)
        - MCP tools (if configured)

        Args:
            task (Optional[Union[str, Any]]): The task or prompt for the agent to process.
                This is the main objective that will be broken down into subtasks.
            img (Optional[str]): Optional image path or data to be processed during execution.
            streaming_callback (Optional[Callable[[str], None]]): Optional callback function
                to receive streaming tokens in real-time. Useful for dashboard integration.
            *args: Additional positional arguments passed to LLM calls.
            **kwargs: Additional keyword arguments passed to LLM calls.

        Returns:
            Any: The agent's output with comprehensive summary. Format depends on output_type:
                - "final": Returns comprehensive task completion summary
                - Other types: Returns formatted conversation history based on output_type

        Raises:
            Exception: If planning phase fails after maximum attempts.
            Exception: If execution exceeds maximum iteration limits.

        Note:
            - This method is automatically called when max_loops="auto" and interactive=False
            - The method resets autonomous loop state at the start of each execution
            - Tool execution results are automatically added to conversation memory
            - Progress visualization is shown if print_on=True

        Examples:
            >>> agent = Agent(max_loops="auto", interactive=False)
            >>> result = agent.run("Build a web application with authentication")
            >>> # The agent will:
            >>> # 1. Create a plan with subtasks
            >>> # 2. Execute each subtask with tool calls
            >>> # 3. Generate a comprehensive summary
        """
        try:

            self.short_memory.add(role=self.user_name, content=task)

            # Reset autonomous loop state
            self.autonomous_subtasks = []
            self.current_subtask_index = 0
            self.subtask_status = {}
            self.plan_created = False
            self.think_call_count = 0

            # Add planning tools to tools_list_dictionary
            planning_tools = get_autonomous_planning_tools()
            if self.tools_list_dictionary is None:
                self.tools_list_dictionary = []

            # Get existing tool names to avoid duplicates
            existing_tool_names = set()
            if self.tools_list_dictionary:
                for tool in self.tools_list_dictionary:
                    if isinstance(tool, dict) and "function" in tool:
                        existing_tool_names.add(
                            tool["function"].get("name", "")
                        )

            # Add planning tools (avoid duplicates)
            for tool in planning_tools:
                tool_name = tool.get("function", {}).get("name", "")
                if tool_name not in existing_tool_names:
                    self.tools_list_dictionary.append(tool)
                    existing_tool_names.add(tool_name)

            # Add handoff tool if handoffs are configured (avoid duplicates)
            if exists(self.handoffs):
                handoff_tool_schema = get_handoff_tool_schema()
                for tool in handoff_tool_schema:
                    tool_name = tool.get("function", {}).get(
                        "name", ""
                    )
                    if tool_name not in existing_tool_names:
                        self.tools_list_dictionary.append(tool)
                        existing_tool_names.add(tool_name)

                # Add handoff prompt to system prompt
                agent_registry = self._get_agent_registry()
                if agent_registry:
                    handoff_prompt = get_handoffs_prompt(
                        list(agent_registry.values())
                    )
                    self.system_prompt += "\n\n" + handoff_prompt

            # Reinitialize LLM with planning tools (and handoff tool if configured)
            if self.llm is not None:
                self.llm = self.llm_handling()

            # Register planning tool handlers
            planning_tool_handlers = {
                "create_plan": self._create_plan_tool,
                "think": self._think_tool,
                "subtask_done": self._subtask_done_tool,
                "complete_task": self._complete_task_tool,
                "respond_to_user": lambda **kwargs: respond_to_user_tool(
                    self, **kwargs
                ),
                "create_file": lambda **kwargs: create_file_tool(
                    self, **kwargs
                ),
                "update_file": lambda **kwargs: update_file_tool(
                    self, **kwargs
                ),
                "read_file": lambda **kwargs: read_file_tool(
                    self, **kwargs
                ),
                "list_directory": lambda **kwargs: list_directory_tool(
                    self, **kwargs
                ),
                "delete_file": lambda **kwargs: delete_file_tool(
                    self, **kwargs
                ),
            }

            # Add handoff tool handler if handoffs are configured
            if exists(self.handoffs):
                planning_tool_handlers["handoff_task"] = (
                    self._handoff_task_tool
                )

            # Phase 1: Planning
            if self.print_on:
                formatter.print_panel(
                    f"Starting planning phase for task:\n\n{task}",
                    title="Autonomous Loop: Planning Phase",
                )

            planning_prompt = get_planning_prompt(task)
            self.short_memory.add(
                role=self.user_name, content=planning_prompt
            )

            plan_created = False
            planning_attempts = 0
            max_planning_attempts = MAX_PLANNING_ATTEMPTS

            while (
                not plan_created
                and planning_attempts < max_planning_attempts
            ):
                planning_attempts += 1
                try:
                    task_prompt = (
                        self.short_memory.return_history_as_string()
                    )
                    response = self.call_llm(
                        task=task_prompt,
                        img=img,
                        current_loop=0,
                        streaming_callback=streaming_callback,
                        *args,
                        **kwargs,
                    )

                    response = self.parse_llm_output(response)
                    self.short_memory.add(
                        role=self.agent_name, content=response
                    )

                    # Check if response contains create_plan or handoff_task tool call
                    if isinstance(response, list):
                        for tool_call in response:
                            if isinstance(tool_call, dict):
                                function_name = tool_call.get(
                                    "function", {}
                                ).get("name")

                                if function_name == "create_plan":
                                    # Execute create_plan tool
                                    arguments = json.loads(
                                        tool_call["function"][
                                            "arguments"
                                        ]
                                    )

                                    # Visualize function call
                                    self._visualize_function_call(
                                        "create_plan", arguments
                                    )

                                    result = planning_tool_handlers[
                                        "create_plan"
                                    ](**arguments)

                                    # Add result to memory
                                    self.short_memory.add(
                                        role="Tool Executor",
                                        content=f"create_plan result: {result}",
                                    )

                                elif (
                                    function_name == "handoff_task"
                                    and exists(self.handoffs)
                                ):
                                    # Handle handoff tool call in planning phase
                                    arguments = json.loads(
                                        tool_call["function"][
                                            "arguments"
                                        ]
                                    )
                                    handoffs_list = arguments.get(
                                        "handoffs", []
                                    )

                                    # Visualize handoff tool call
                                    if self.print_on:
                                        self._visualize_handoff_call(
                                            handoffs_list, tool_call
                                        )

                                    result = self._handoff_task_tool(
                                        handoffs=handoffs_list
                                    )

                                    # Add result to memory
                                    self.short_memory.add(
                                        role="Tool Executor",
                                        content=f"handoff_task result: {result}",
                                    )

                                # Show plan creation result
                                if self.print_on:
                                    plan_summary = f"Plan created with {len(self.autonomous_subtasks)} subtasks:\n\n"
                                    for i, subtask in enumerate(
                                        self.autonomous_subtasks, 1
                                    ):
                                        plan_summary += f"{i}. {subtask['step_id']}: {subtask['description']}\n"
                                        plan_summary += f"   Priority: {subtask['priority']}\n"
                                        if subtask.get(
                                            "dependencies"
                                        ):
                                            plan_summary += f"   Dependencies: {', '.join(subtask['dependencies'])}\n"

                                    formatter.print_panel(
                                        plan_summary,
                                        title="Plan Created",
                                    )

                                plan_created = True
                                break

                    # Also check if plan was created via tool execution
                    if self.plan_created:
                        plan_created = True
                        break

                except Exception as e:
                    if self.verbose:
                        logger.error(
                            f"Error in planning phase (attempt {planning_attempts}): {e}"
                        )
                    if planning_attempts >= max_planning_attempts:
                        raise

            if not plan_created:
                raise Exception(
                    "Failed to create plan after maximum attempts"
                )

            # Integrate user tools after planning phase
            if exists(self.tools):
                # Convert user tools to function schema
                user_tools = convert_multiple_functions_to_openai_function_schema(
                    self.tools
                )

                # Get existing tool names to avoid duplicates
                existing_tool_names = set()
                if self.tools_list_dictionary:
                    for tool in self.tools_list_dictionary:
                        if (
                            isinstance(tool, dict)
                            and "function" in tool
                        ):
                            existing_tool_names.add(
                                tool["function"].get("name", "")
                            )

                # Add user tools to tools_list_dictionary (avoid duplicates)
                if self.tools_list_dictionary is None:
                    self.tools_list_dictionary = []

                tools_added = 0
                for tool in user_tools:
                    tool_name = tool.get("function", {}).get(
                        "name", ""
                    )
                    if tool_name not in existing_tool_names:
                        self.tools_list_dictionary.append(tool)
                        existing_tool_names.add(tool_name)
                        tools_added += 1

                # Reinitialize LLM with both planning tools and user tools
                if self.llm is not None:
                    self.llm = self.llm_handling()

                if self.print_on and tools_added > 0:
                    formatter.print_panel(
                        f"Integrated {tools_added} user tools into autonomous loop",
                        title="Tools Integration",
                    )

            # Phase 2: Execution - For each subtask
            if self.print_on:
                formatter.print_panel(
                    f"Starting execution phase with {len(self.autonomous_subtasks)} subtasks",
                    title="Autonomous Loop: Execution Phase",
                )

            max_subtask_iterations = MAX_SUBTASK_ITERATIONS
            total_iterations = 0

            while not self._all_subtasks_complete():
                total_iterations += 1
                if total_iterations > max_subtask_iterations:
                    if self.print_on:
                        formatter.print_panel(
                            f"Maximum iterations ({max_subtask_iterations}) reached. Stopping execution.",
                            title="Execution Limit Reached",
                        )
                    if self.verbose:
                        logger.warning(
                            f"Maximum iterations ({max_subtask_iterations}) reached. Stopping execution."
                        )
                    break

                # Get next executable subtask
                current_subtask = self._get_next_executable_subtask()
                if current_subtask is None:
                    # All subtasks are done or blocked
                    if self._all_subtasks_complete():
                        break
                    else:
                        if self.verbose:
                            logger.warning(
                                "No executable subtasks found, but not all are complete"
                            )
                        break

                subtask_id = current_subtask["step_id"]
                subtask_desc = current_subtask["description"]
                subtask_priority = current_subtask.get(
                    "priority", "medium"
                )

                # Show subtask start
                if self.print_on:
                    progress = f"{sum(1 for s in self.autonomous_subtasks if s['status'] in ['completed', 'failed'])}/{len(self.autonomous_subtasks)}"
                    formatter.print_panel(
                        f"Subtask: {subtask_id}\nDescription: {subtask_desc}\nPriority: {subtask_priority}\nProgress: {progress} subtasks completed",
                        title=f"Executing Subtask: {subtask_id}",
                    )

                # Subtask execution loop: thinking -> tool actions -> observation
                subtask_iterations = 0
                max_subtask_loops = MAX_SUBTASK_LOOPS
                subtask_done = False

                while (
                    not subtask_done
                    and subtask_iterations < max_subtask_loops
                ):
                    subtask_iterations += 1
                    self.think_call_count = (
                        0  # Reset for each subtask
                    )

                    try:
                        # Create execution prompt for current subtask
                        execution_prompt = get_execution_prompt(
                            subtask_id,
                            subtask_desc,
                            self.autonomous_subtasks,
                        )
                        self.short_memory.add(
                            role=self.user_name,
                            content=execution_prompt,
                        )

                        task_prompt = (
                            self.short_memory.return_history_as_string()
                        )
                        response = self.call_llm(
                            task=task_prompt,
                            img=img,
                            current_loop=subtask_iterations,
                            streaming_callback=streaming_callback,
                            *args,
                            **kwargs,
                        )

                        response = self.parse_llm_output(response)
                        self.short_memory.add(
                            role=self.agent_name, content=response
                        )

                        # Handle tool calls
                        if isinstance(response, list):
                            regular_tool_calls = []

                            for tool_call in response:
                                if isinstance(
                                    tool_call, dict
                                ) and tool_call.get(
                                    "function", {}
                                ).get(
                                    "name"
                                ):
                                    function_name = tool_call[
                                        "function"
                                    ]["name"]
                                    arguments = json.loads(
                                        tool_call["function"][
                                            "arguments"
                                        ]
                                    )

                                    # Handle planning tools and handoff tool
                                    if (
                                        function_name
                                        in planning_tool_handlers
                                    ):
                                        # Special handling for handoff_task tool
                                        if (
                                            function_name
                                            == "handoff_task"
                                        ):
                                            # Visualize handoff tool call
                                            handoffs_list = (
                                                arguments.get(
                                                    "handoffs", []
                                                )
                                            )
                                            if self.print_on:
                                                self._visualize_handoff_call(
                                                    handoffs_list,
                                                    tool_call,
                                                )

                                            result = self._handoff_task_tool(
                                                handoffs=handoffs_list
                                            )
                                        else:
                                            # Visualize function call for other tools
                                            self._visualize_function_call(
                                                function_name,
                                                arguments,
                                            )

                                            result = planning_tool_handlers[
                                                function_name
                                            ](
                                                **arguments
                                            )

                                        # Add result to memory
                                        self.short_memory.add(
                                            role="Tool Executor",
                                            content=f"{function_name} result: {result}",
                                        )

                                        # Visualize result for important tools
                                        if function_name in [
                                            "subtask_done",
                                            "complete_task",
                                        ]:
                                            self._visualize_function_call(
                                                function_name,
                                                arguments,
                                                result,
                                            )

                                        # Check if subtask is done
                                        if (
                                            function_name
                                            == "subtask_done"
                                        ):
                                            if (
                                                arguments.get(
                                                    "task_id"
                                                )
                                                == subtask_id
                                            ):
                                                subtask_done = True
                                                # Show subtask completion
                                                if self.print_on:
                                                    status = (
                                                        "completed"
                                                        if arguments.get(
                                                            "success"
                                                        )
                                                        else "failed"
                                                    )
                                                    formatter.print_panel(
                                                        f"Subtask {subtask_id} marked as {status}\n\nSummary: {arguments.get('summary', 'N/A')}",
                                                        title=f"Subtask {status.title()}: {subtask_id}",
                                                    )
                                                break

                                        # Check if main task is complete
                                        if (
                                            function_name
                                            == "complete_task"
                                        ):
                                            # Task is complete, exit all loops
                                            return (
                                                self._generate_final_summary()
                                            )
                                    else:
                                        # Collect regular tool calls for batch visualization and execution
                                        regular_tool_calls.append(
                                            tool_call
                                        )

                            # Handle all regular tools together
                            if regular_tool_calls and exists(
                                self.tools
                            ):
                                # Visualize all regular tool calls first
                                if self.print_on:
                                    for (
                                        tool_call
                                    ) in regular_tool_calls:
                                        func_name = tool_call.get(
                                            "function", {}
                                        ).get("name", "Unknown")
                                        func_args = {}
                                        try:
                                            func_args = json.loads(
                                                tool_call.get(
                                                    "function", {}
                                                ).get(
                                                    "arguments", "{}"
                                                )
                                            )
                                        except (
                                            json.JSONDecodeError,
                                            AttributeError,
                                        ):
                                            pass
                                        self._visualize_function_call(
                                            func_name, func_args
                                        )

                                # Execute all regular tools together
                                try:
                                    tool_output = self.tool_struct.execute_function_calls_from_api_response(
                                        regular_tool_calls
                                    )

                                    # Add to memory
                                    self.short_memory.add(
                                        role="Tool Executor",
                                        content=format_data_structure(
                                            tool_output
                                        ),
                                    )

                                    # Display tool execution results using formatter
                                    if self.print_on:
                                        tool_names = [
                                            tc.get(
                                                "function", {}
                                            ).get("name", "Unknown")
                                            for tc in regular_tool_calls
                                        ]
                                        tool_display = f"Tools Executed: {', '.join(tool_names)}\n\n"
                                        tool_display += f"Output:\n{format_data_structure(tool_output)}"

                                        formatter.print_panel(
                                            tool_display,
                                            title="Tool Execution Results",
                                        )

                                    # Handle tool call summary if enabled
                                    if self.tool_call_summary is True:
                                        temp_llm = (
                                            self.temp_llm_instance_for_tool_summary()
                                        )
                                        tool_response = temp_llm.run(
                                            f"""
                                            Please analyze and summarize the following tool execution output in a clear and concise way. 
                                            Focus on the key information and insights that would be most relevant to the user's original request.
                                            If there are any errors or issues, highlight them prominently.
                                            
                                            Tool Output:
                                            {tool_output}
                                            """
                                        )
                                        self.short_memory.add(
                                            role=self.agent_name,
                                            content=tool_response,
                                        )

                                except Exception as e:
                                    # Fallback to tool_execution_retry if direct execution fails
                                    if self.verbose:
                                        logger.warning(
                                            f"Direct tool execution failed, using retry mechanism: {e}"
                                        )
                                    self.tool_execution_retry(
                                        regular_tool_calls,
                                        subtask_iterations,
                                    )
                        else:
                            # Handle regular tool execution
                            if exists(self.tools):
                                # Visualize tool calls before execution
                                if (
                                    isinstance(response, list)
                                    and self.print_on
                                ):
                                    for tool_call in response:
                                        if isinstance(
                                            tool_call, dict
                                        ):
                                            func_name = tool_call.get(
                                                "function", {}
                                            ).get("name", "Unknown")
                                            func_args = {}
                                            try:
                                                func_args = json.loads(
                                                    tool_call.get(
                                                        "function", {}
                                                    ).get(
                                                        "arguments",
                                                        "{}",
                                                    )
                                                )
                                            except (
                                                json.JSONDecodeError,
                                                AttributeError,
                                            ):
                                                pass

                                            # Only visualize if it's not a planning tool
                                            if (
                                                func_name
                                                not in planning_tool_handlers
                                            ):
                                                self._visualize_function_call(
                                                    func_name,
                                                    func_args,
                                                )

                                # Execute tools and capture output for display
                                try:
                                    tool_output = self.tool_struct.execute_function_calls_from_api_response(
                                        response
                                    )

                                    # Add to memory
                                    self.short_memory.add(
                                        role="Tool Executor",
                                        content=format_data_structure(
                                            tool_output
                                        ),
                                    )

                                    # Display tool execution results using formatter
                                    if self.print_on:
                                        tool_display = f"Tool Output:\n{format_data_structure(tool_output)}"
                                        formatter.print_panel(
                                            tool_display,
                                            title="Tool Execution Results",
                                        )

                                    # Handle tool call summary if enabled
                                    if self.tool_call_summary is True:
                                        temp_llm = (
                                            self.temp_llm_instance_for_tool_summary()
                                        )
                                        tool_response = temp_llm.run(
                                            f"""
                                            Please analyze and summarize the following tool execution output in a clear and concise way. 
                                            Focus on the key information and insights that would be most relevant to the user's original request.
                                            If there are any errors or issues, highlight them prominently.
                                            
                                            Tool Output:
                                            {tool_output}
                                            """
                                        )
                                        self.short_memory.add(
                                            role=self.agent_name,
                                            content=tool_response,
                                        )

                                except Exception as e:
                                    # Fallback to tool_execution_retry if direct execution fails
                                    if self.verbose:
                                        logger.warning(
                                            f"Direct tool execution failed, using retry mechanism: {e}"
                                        )
                                    self.tool_execution_retry(
                                        response, subtask_iterations
                                    )

                        # Check if subtask status changed
                        if (
                            subtask_id in self.subtask_status
                            and self.subtask_status[subtask_id]
                            in ["completed", "failed"]
                        ):
                            subtask_done = True

                        # Prevent infinite thinking loops
                        if (
                            self.think_call_count
                            >= self.max_consecutive_thinks
                        ):
                            if self.print_on:
                                formatter.print_panel(
                                    f"Too many consecutive think calls ({self.think_call_count}). Forcing action.",
                                    title="Loop Prevention",
                                )
                            if self.verbose:
                                logger.warning(
                                    f"Too many consecutive think calls ({self.think_call_count}). Forcing action."
                                )
                            # Force action by adding a prompt
                            self.short_memory.add(
                                role="system",
                                content="You have been thinking too much. Take action now using available tools.",
                            )

                    except Exception as e:
                        if self.verbose:
                            logger.error(
                                f"Error in subtask execution loop: {e}"
                            )
                        # Continue to next iteration

                if not subtask_done:
                    if self.print_on:
                        formatter.print_panel(
                            f"Subtask {subtask_id} not completed after {max_subtask_loops} iterations",
                            title="Subtask Timeout",
                        )
                    if self.verbose:
                        logger.warning(
                            f"Subtask {subtask_id} not completed after {max_subtask_loops} iterations"
                        )

            # Phase 3: Final Summary
            if self.print_on:
                formatter.print_panel(
                    "All subtasks completed. Generating final summary...",
                    title="Autonomous Loop: Summary Phase",
                )

            if self.output_type == "final":
                return self._generate_final_summary()
            else:
                return history_output_formatter(
                    conversation=self.short_memory,
                    type=self.output_type,
                )

        except Exception as error:
            self._handle_run_error(error)

    def _generate_final_summary(self) -> str:
        """
        Generate a comprehensive final summary of the autonomous task execution.

        Returns:
            str: Comprehensive summary
        """
        summary_prompt = get_summary_prompt()
        self.short_memory.add(
            role=self.user_name, content=summary_prompt
        )

        try:
            task_prompt = self.short_memory.return_history_as_string()
            response = self.call_llm(
                task=task_prompt,
                current_loop=0,
            )

            response = self.parse_llm_output(response)

            # Add LLM response to memory
            self.short_memory.add(
                role=self.agent_name, content=str(response)
            )

            # Check if complete_task was called
            if isinstance(response, list):
                for tool_call in response:
                    if (
                        isinstance(tool_call, dict)
                        and tool_call.get("function", {}).get("name")
                        == "complete_task"
                    ):
                        arguments = json.loads(
                            tool_call["function"]["arguments"]
                        )

                        # Visualize final task completion
                        self._visualize_function_call(
                            "complete_task", arguments
                        )

                        result = self._complete_task_tool(**arguments)

                        # Add result to memory
                        self.short_memory.add(
                            role="Tool Executor",
                            content=f"complete_task result: {result}",
                        )

                        # Show comprehensive summary
                        if self.print_on:
                            formatter.print_panel(
                                result,
                                title="Task Completion Summary",
                            )

                        return result

            # If complete_task wasn't called, generate summary manually
            comprehensive_summary = f"""Task Execution Summary

Original Task: {self.short_memory.messages[0].get('content', 'N/A') if self.short_memory.messages else 'N/A'}

Subtask Breakdown:
"""
            for subtask in self.autonomous_subtasks:
                comprehensive_summary += (
                    f"\n{subtask['step_id']}: {subtask['status']}\n"
                )
                comprehensive_summary += (
                    f"  Description: {subtask['description']}\n"
                )
                if "summary" in subtask:
                    comprehensive_summary += (
                        f"  Summary: {subtask['summary']}\n"
                    )

            comprehensive_summary += f"\nFinal Response:\n{response}"

            self.short_memory.add(
                role=self.agent_name, content=comprehensive_summary
            )

            if self.print_on:
                formatter.print_panel(
                    comprehensive_summary,
                    title="Task Execution Summary",
                )

            return history_output_formatter(
                self.short_memory, type=self.output_type
            )

        except Exception as e:
            if self.verbose:
                logger.error(f"Error generating final summary: {e}")
            # Return basic summary
            return history_output_formatter(
                self.short_memory, type=self.output_type
            )

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
        Files are saved in the agent-specific workspace directory: workspace_dir/agent-{agent_name}-{uuid}/

        Args:
            file_path (str, optional): Custom path to save the state. If relative, will be saved in
                                    the agent-specific workspace directory. If None, uses configured paths.

        Raises:
            OSError: If there are filesystem-related errors
            Exception: For other unexpected errors
        """
        try:
            # Get agent-specific workspace directory
            agent_workspace = self._get_agent_workspace_dir()

            # Determine the save path
            resolved_path = (
                file_path
                or self.saved_state_path
                or f"{self.agent_name}_state.json"
            )

            # Ensure path has .json extension
            if not resolved_path.endswith(".json"):
                resolved_path += ".json"

            # If file_path is absolute, use it as-is; otherwise, use agent workspace
            if file_path and os.path.isabs(file_path):
                full_path = file_path
            else:
                # Create full path in agent-specific workspace directory
                full_path = os.path.join(
                    agent_workspace, resolved_path
                )

            backup_path = full_path + ".backup"
            temp_path = full_path + ".temp"

            # Ensure directory exists
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
        """
        Save the agent model configuration to JSON in the agent-specific workspace directory.

        Returns:
            str: Message indicating where the file was saved.
        """
        agent_workspace = self._get_agent_workspace_dir()
        logger.info(
            f"Saving {self.agent_name} model to JSON in the {agent_workspace} directory"
        )

        create_file_in_folder(
            agent_workspace,
            f"{self.agent_name}.json",
            str(self.to_json()),
        )

        return (
            f"Model saved to {agent_workspace}/{self.agent_name}.json"
        )

    def model_dump_yaml(self):
        """
        Save the agent model configuration to YAML in the agent-specific workspace directory.

        Returns:
            str: Message indicating where the file was saved.
        """
        agent_workspace = self._get_agent_workspace_dir()
        logger.info(
            f"Saving {self.agent_name} model to YAML in the {agent_workspace} directory"
        )

        create_file_in_folder(
            agent_workspace,
            f"{self.agent_name}.yaml",
            str(self.to_yaml()),
        )

        return (
            f"Model saved to {agent_workspace}/{self.agent_name}.yaml"
        )

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
        Calls the LLM with the given task, handling streaming and multimodal inputs.

        This method is the primary interface for calling the language model. It handles
        three modes of operation: detailed streaming, basic streaming, and non-streaming.
        It also supports multimodal inputs (images) and provides comprehensive error handling.

        **Streaming Modes:**

        1. **Detailed Streaming (stream=True):**
           - Streams tokens with full metadata (citations, tokens used, logprobs, etc.)
           - Each token includes: token_index, model, id, created, object, token,
             system_fingerprint, finish_reason, citations, provider_specific_fields,
             service_tier, obfuscation, usage, logprobs, timestamp
           - Calls streaming_callback with token_info dictionary for each token
           - Prints final ModelResponseStream with usage statistics

        2. **Basic Streaming (streaming_on=True):**
           - Streams tokens with formatted panels
           - Supports three behaviors:
             - With streaming_callback: Real-time callback for dashboard integration
             - With print_on=False: Silent streaming (collects chunks only)
             - With print_on=True: Displays streaming panel with collected chunks
           - Uses formatter.print_streaming_panel for visual display

        3. **Non-Streaming:**
           - Direct call to llm.run() with task and optional image
           - Returns complete response string

        **Multimodal Support:**
        - If img is provided, passes it to llm.run() for vision-enabled models
        - Automatically handles image path or URL strings

        **Error Handling:**
        - Catches AgentLLMError, BadRequestError, InternalServerError, AuthenticationError
        - Logs errors with model name, task, and full traceback
        - Re-raises exceptions for upstream handling

        Args:
            task (str): The task or prompt to send to the LLM. This is the main
                input that the model will process.
            img (Optional[str]): Optional image path or URL for multimodal processing.
                Only used with vision-enabled models. Defaults to None.
            current_loop (int): The current loop iteration number. Used for:
                - Streaming panel titles
                - Progress tracking
                - Error logging context
                Defaults to 0.
            streaming_callback (Optional[Callable[[str], None]]): Optional callback
                function to receive streaming tokens in real-time. For detailed streaming
                (stream=True), receives token_info dictionaries. For basic streaming
                (streaming_on=True), receives token strings. Defaults to None.
            *args: Additional positional arguments passed directly to llm.run().
            **kwargs: Additional keyword arguments passed directly to llm.run().
                Note: 'is_last' is automatically filtered out if present.

        Returns:
            str: The complete response from the LLM. For streaming modes, this is
                the concatenated result of all streamed tokens. For non-streaming,
                this is the direct response from llm.run().

        Raises:
            AgentLLMError: If there's an issue with the language model.
            BadRequestError: If the request is malformed or invalid.
            InternalServerError: If the LLM service encounters an internal error.
            AuthenticationError: If authentication fails with the LLM service.
            Exception: For other unexpected errors during LLM calls.

        Note:
            - The method automatically restores original stream setting after execution
            - Detailed streaming mode requires stream=True and llm.stream attribute
            - Basic streaming mode requires streaming_on=True and llm.stream attribute
            - Image processing requires a vision-enabled model
            - The method uses self.get_current_model() for error logging

        Examples:
            >>> # Non-streaming call
            >>> response = agent.call_llm("What is Python?", current_loop=1)
            >>> print(response)

            >>> # Streaming with callback
            >>> def on_token(token):
            ...     print(f"Received: {token}")
            >>> response = agent.call_llm(
            ...     "Tell me a story",
            ...     streaming_callback=on_token,
            ...     current_loop=1
            ... )

            >>> # Multimodal call
            >>> response = agent.call_llm(
            ...     "Describe this image",
            ...     img="path/to/image.jpg",
            ...     current_loop=1
            ... )
        """

        # Filter out is_last from kwargs if present
        if "is_last" in kwargs:
            del kwargs["is_last"]

        try:
            if self.stream and hasattr(self.llm, "stream"):
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

                if hasattr(
                    streaming_response, "__iter__"
                ) and not isinstance(streaming_response, str):
                    complete_response = ""
                    token_count = 0
                    final_chunk = None
                    first_chunk = None

                    for chunk in streaming_response:
                        if first_chunk is None:
                            first_chunk = chunk

                        if (
                            hasattr(chunk, "choices")
                            and chunk.choices[0].delta.content
                        ):
                            content = chunk.choices[0].delta.content
                            complete_response += content
                            token_count += 1

                            # Schema per token outputted
                            token_info = {
                                "token_index": token_count,
                                "model": getattr(
                                    chunk,
                                    "model",
                                    self.get_current_model(),
                                ),
                                "id": getattr(chunk, "id", ""),
                                "created": getattr(
                                    chunk, "created", int(time.time())
                                ),
                                "object": getattr(
                                    chunk,
                                    "object",
                                    "chat.completion.chunk",
                                ),
                                "token": content,
                                "system_fingerprint": getattr(
                                    chunk, "system_fingerprint", ""
                                ),
                                "finish_reason": chunk.choices[
                                    0
                                ].finish_reason,
                                "citations": getattr(
                                    chunk, "citations", None
                                ),
                                "provider_specific_fields": getattr(
                                    chunk,
                                    "provider_specific_fields",
                                    None,
                                ),
                                "service_tier": getattr(
                                    chunk, "service_tier", "default"
                                ),
                                "obfuscation": getattr(
                                    chunk, "obfuscation", None
                                ),
                                "usage": getattr(
                                    chunk, "usage", None
                                ),
                                "logprobs": chunk.choices[0].logprobs,
                                "timestamp": time.time(),
                            }

                            print(f"ResponseStream {token_info}")

                            if streaming_callback is not None:
                                streaming_callback(token_info)

                        final_chunk = chunk

                    # Final ModelResponse to stream
                    if (
                        final_chunk
                        and hasattr(final_chunk, "usage")
                        and final_chunk.usage
                    ):
                        usage = final_chunk.usage
                        print(
                            f"ModelResponseStream(id='{getattr(final_chunk, 'id', 'N/A')}', "
                            f"created={getattr(final_chunk, 'created', 'N/A')}, "
                            f"model='{getattr(final_chunk, 'model', self.get_current_model())}', "
                            f"object='{getattr(final_chunk, 'object', 'chat.completion.chunk')}', "
                            f"system_fingerprint='{getattr(final_chunk, 'system_fingerprint', 'N/A')}', "
                            f"choices=[StreamingChoices(finish_reason='{final_chunk.choices[0].finish_reason}', "
                            f"index=0, delta=Delta(provider_specific_fields=None, content=None, role=None, "
                            f"function_call=None, tool_calls=None, audio=None), logprobs=None)], "
                            f"provider_specific_fields=None, "
                            f"usage=Usage(completion_tokens={usage.completion_tokens}, "
                            f"prompt_tokens={usage.prompt_tokens}, "
                            f"total_tokens={usage.total_tokens}, "
                            f"completion_tokens_details=CompletionTokensDetailsWrapper("
                            f"accepted_prediction_tokens={usage.completion_tokens_details.accepted_prediction_tokens}, "
                            f"audio_tokens={usage.completion_tokens_details.audio_tokens}, "
                            f"reasoning_tokens={usage.completion_tokens_details.reasoning_tokens}, "
                            f"rejected_prediction_tokens={usage.completion_tokens_details.rejected_prediction_tokens}, "
                            f"text_tokens={usage.completion_tokens_details.text_tokens}), "
                            f"prompt_tokens_details=PromptTokensDetailsWrapper("
                            f"audio_tokens={usage.prompt_tokens_details.audio_tokens}, "
                            f"cached_tokens={usage.prompt_tokens_details.cached_tokens}, "
                            f"text_tokens={usage.prompt_tokens_details.text_tokens}, "
                            f"image_tokens={usage.prompt_tokens_details.image_tokens})))"
                        )
                    else:
                        print(
                            f"ModelResponseStream(id='{getattr(final_chunk, 'id', 'N/A')}', "
                            f"created={getattr(final_chunk, 'created', 'N/A')}, "
                            f"model='{getattr(final_chunk, 'model', self.get_current_model())}', "
                            f"object='{getattr(final_chunk, 'object', 'chat.completion.chunk')}', "
                            f"system_fingerprint='{getattr(final_chunk, 'system_fingerprint', 'N/A')}', "
                            f"choices=[StreamingChoices(finish_reason='{final_chunk.choices[0].finish_reason}', "
                            f"index=0, delta=Delta(provider_specific_fields=None, content=None, role=None, "
                            f"function_call=None, tool_calls=None, audio=None), logprobs=None)], "
                            f"provider_specific_fields=None)"
                        )

                    self.llm.stream = original_stream
                    return complete_response
                else:
                    self.llm.stream = original_stream
                    return streaming_response

            elif self.streaming_on and hasattr(self.llm, "stream"):
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

    def load_skills_metadata(
        self, skills_dir: str
    ) -> List[Dict[str, str]]:
        """
        Load skill metadata from SKILL.md files in the skills directory.

        Implements Tier 1 loading from Anthropic's Agent Skills framework:
        loads skill name and description into memory for context-aware activation.

        Args:
            skills_dir: Path to directory containing skill folders.
                Each folder should contain a SKILL.md file with YAML frontmatter.

        Returns:
            List of skill metadata dicts with 'name', 'description', 'path', 'content'

        Example:
            >>> agent = Agent(skills_dir="./skills")
            >>> # Loads all skills from ./skills/*/SKILL.md
        """
        skills = []

        if not os.path.exists(skills_dir):
            logger.warning(
                f"Skills directory not found: {skills_dir}"
            )
            return skills

        for skill_folder in os.listdir(skills_dir):
            skill_path = os.path.join(skills_dir, skill_folder)

            if not os.path.isdir(skill_path):
                continue

            skill_file = os.path.join(skill_path, "SKILL.md")

            if not os.path.exists(skill_file):
                continue

            try:
                with open(skill_file, "r", encoding="utf-8") as f:
                    content = f.read()

                # Parse YAML frontmatter
                if content.startswith("---"):
                    parts = content.split("---", 2)
                    if len(parts) >= 3:
                        frontmatter = yaml.safe_load(parts[1])
                        skills.append(
                            {
                                "name": frontmatter.get(
                                    "name", skill_folder
                                ),
                                "description": frontmatter.get(
                                    "description", ""
                                ),
                                "path": skill_file,
                                "content": (
                                    parts[2].strip()
                                    if len(parts) > 2
                                    else ""
                                ),
                            }
                        )
                        logger.info(
                            f"Loaded skill: {frontmatter.get('name')}"
                        )
            except Exception as e:
                logger.warning(
                    f"Failed to load skill from {skill_file}: {e}"
                )
                continue

        return skills

    def _build_skills_prompt(
        self, skills: List[Dict[str, str]]
    ) -> str:
        """
        Build the skills section to append to system prompt.

        Loads full skill content (YAML frontmatter + markdown instructions)
        into the system prompt for immediate availability.

        Args:
            skills: List of skill metadata dicts from load_skills_metadata()

        Returns:
            Formatted skills prompt section to append to system_prompt

        Example:
            >>> skills = [{"name": "financial-analysis", "description": "DCF modeling", "content": "..."}]
            >>> prompt = agent._build_skills_prompt(skills)
            >>> # Returns formatted skills section with full instructions
        """
        if not skills:
            return ""

        prompt = "\n\n# Available Skills\n\n"
        prompt += (
            "You have access to the following specialized skills. "
        )
        prompt += "Follow their instructions when relevant:\n\n"

        for skill in skills:
            prompt += f"## {skill['name']}\n\n"
            prompt += f"**Description**: {skill['description']}\n\n"
            prompt += skill["content"]
            prompt += "\n\n---\n\n"

        return prompt

    def load_full_skill(self, skill_name: str) -> Optional[str]:
        """
        Load the full content of a specific skill (Tier 2 loading).

        This implements Tier 2 progressive disclosure: loads the complete
        SKILL.md content when the skill is actively needed, rather than
        loading everything upfront.

        Args:
            skill_name: Name of the skill to load (from metadata)

        Returns:
            Full skill content (markdown below frontmatter) or None if not found

        Example:
            >>> agent = Agent(skills_dir="./skills")
            >>> content = agent.load_full_skill("financial-analysis")
            >>> # Returns full markdown instructions for the skill
        """
        for skill in self.skills_metadata:
            if skill["name"] == skill_name:
                try:
                    with open(
                        skill["path"], "r", encoding="utf-8"
                    ) as f:
                        content = f.read()
                    # Return everything after frontmatter
                    if content.startswith("---"):
                        parts = content.split("---", 2)
                        if len(parts) >= 3:
                            return parts[2].strip()
                except Exception as e:
                    logger.error(
                        f"Failed to load full skill {skill_name}: {e}"
                    )
        return None

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
        Execute the agent's main reasoning/thinking flow (single or multi-step).

        This is the primary entrypoint for running an agent on a given task, optionally with one or more images and with support for both interactive and autonomous flows.

        Core Features:
            - Handles both interactive (asks user) and autonomous (auto-plan/execute) operation modes.
            - Supports passing a single image or batch of images.
            - Supports streaming outputs via a callback for real-time token generation.
            - Runs multiple outputs (n > 1), single or batched.
            - Accepts an optional ground truth (correct_answer) for evals.
            - Merges configuration and per-call streaming callback.
            - Handles errors and device selection internally (but device_id is not used directly by this method).

        Args:
            task (Optional[str|Any]): Task for the agent to process. If not a string, will be formatted. Defaults to None.
            img (Optional[str]): Path or reference to a single image input. Defaults to None.
            imgs (Optional[List[str]]): List of multiple images if processing a batch. Defaults to None.
            correct_answer (Optional[str]): Ground truth answer for evaluation comparisons. Defaults to None.
            streaming_callback (Optional[Callable[[str], None]]): Function to receive streamed tokens as output is generated (real-time). If not given, uses self.streaming_callback if available. Defaults to None.
            n (int): How many outputs to generate (number of runs). Defaults to 1.
            *args: Additional positional arguments for extensibility.
            **kwargs: Additional keyword arguments passed to LLM/tool execution.

        Returns:
            Any: The agent's output. This can be:
                - A string or structured dict (for single response).
                - A list (if running multiple outputs/images).
                - The final agent answer, streaming response, or summary (for autonomous "auto" mode).

        Raises:
            ValueError: If required arguments are invalid or missing (e.g. image input without actual image).
            Exception: For any error that occurs during agent execution, LLM/tool call, or planning.

        Examples:
            >>> agent.run("Write a poem about the ocean")
            >>> agent.run("Describe this image", img="cat.png")
            >>> agent.run("Summarize", imgs=["a.png", "b.png"])
            >>> agent.run(task="Who won the World Cup?", streaming_callback=print)
        """

        # # If interactive mode is enabled and no task is provided, prompt the user
        # if self.interactive and (
        #     task is None
        #     or (isinstance(task, str) and task.strip() == "")
        if (
            task is None
            or isinstance(task, str)
            and task.strip() == ""
        ):
            # Always show prompt when asking for initial task, even if print_on is False
            self.pretty_print(
                "Interactive mode enabled. Please enter your initial task:",
                loop_count=0,
            )
            task = input("You: ").strip()

            if not task:
                raise ValueError(
                    "No task provided. Exiting interactive mode."
                )

        if exists(self.skills_dir):
            self.handle_skills(task=task)

        if not isinstance(task, str):
            task = format_data_structure(task)

        # Use instance streaming_callback if not provided in method call
        # Priority: local callback (method parameter) > instance callback (__init__)
        # Check both: use local if provided, otherwise fall back to instance callback
        # If both are None, streaming_callback remains None
        if streaming_callback is None:
            if self.streaming_callback is not None:
                streaming_callback = self.streaming_callback
            # else: both are None, streaming_callback stays None

        try:
            if self.max_loops == "auto":
                # Use autonomous loop structure: plan -> execute subtasks -> summary
                output = self._run_autonomous_loop(
                    task=task,
                    img=img,
                    streaming_callback=streaming_callback,
                    *args,
                    **kwargs,
                )
            elif imgs is not None:
                output = self.run_multiple_images(
                    task=task, imgs=imgs, *args, **kwargs
                )
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
            # Save config on interrupt
            if self.autosave:
                try:
                    self._autosave_config_step(loop_count=None)
                except Exception:
                    pass  # Don't let autosave errors mask the interrupt
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
        """
        Handle creating and saving artifacts with error handling.
        Artifacts are saved in the agent-specific workspace directory if the path is relative.

        Args:
            text (str): The content to save as an artifact.
            file_output_path (str): The path where the artifact should be saved. If relative,
                                  will be saved in the agent-specific workspace directory.
            file_extension (str): The file extension for the artifact (e.g., '.md', '.txt', '.pdf').
        """
        try:
            # Ensure file_extension starts with a dot
            if not file_extension.startswith("."):
                file_extension = "." + file_extension

            # Get agent-specific workspace directory
            agent_workspace = self._get_agent_workspace_dir()

            # If file_output_path doesn't have an extension, treat it as a directory
            # and create a default filename based on timestamp
            if not os.path.splitext(file_output_path)[1]:
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"artifact_{timestamp}{file_extension}"
                # If path is relative, use agent workspace; otherwise use as-is
                if os.path.isabs(file_output_path):
                    full_path = os.path.join(
                        file_output_path, filename
                    )
                else:
                    full_path = os.path.join(
                        agent_workspace, file_output_path, filename
                    )
            else:
                # If path is absolute, use as-is; otherwise use agent workspace
                if os.path.isabs(file_output_path):
                    full_path = file_output_path
                else:
                    full_path = os.path.join(
                        agent_workspace, file_output_path
                    )

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
        elif self.stream:
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

    def _create_plan_tool(
        self, task_description: str, steps: List[Dict], **kwargs
    ) -> str:
        """
        Create a detailed plan for task execution.

        This tool is used by the autonomous loop to break down a complex task into
        manageable subtasks with dependencies, priorities, and execution order.

        **Plan Structure:**
        Each step in the plan must contain:
        - step_id (str): Unique identifier for the subtask
        - description (str): Detailed description of what needs to be done
        - priority (str, optional): Priority level (e.g., "high", "medium", "low")
        - dependencies (List[str], optional): List of step_ids that must complete first

        **Plan Storage:**
        The plan is stored in:
        - self.autonomous_subtasks: List of all subtasks with their details
        - self.subtask_status: Dictionary mapping step_id to status ("pending", "completed", "failed")
        - self.plan_created: Boolean flag indicating plan creation

        **Execution Order:**
        Subtasks are executed based on:
        1. Dependencies: Tasks with unmet dependencies are blocked
        2. Priority: Higher priority tasks are preferred when multiple are available
        3. Creation order: Used as tiebreaker

        Args:
            task_description (str): High-level description of the overall task to be completed.
                This provides context for the subtask planning.
            steps (List[Dict]): List of step dictionaries, each containing:
                - step_id (str): Unique identifier for the subtask (required)
                - description (str): What needs to be accomplished (required)
                - priority (str): Priority level, e.g., "high", "medium", "low" (optional)
                - dependencies (List[str]): List of step_ids that must complete first (optional)
            **kwargs: Additional arguments (currently unused, reserved for future use).

        Returns:
            str: Confirmation message indicating successful plan creation with the number
                of subtasks created. Format: "Plan created successfully with {n} subtasks"

        Note:
            - This method is called automatically by the autonomous loop during planning phase
            - The plan replaces any existing autonomous_subtasks
            - All subtasks start with status "pending"
            - current_subtask_index is reset to 0
            - If verbose=True, plan creation is logged with step details

        Examples:
            >>> steps = [
            ...     {
            ...         "step_id": "step1",
            ...         "description": "Set up project structure",
            ...         "priority": "high",
            ...         "dependencies": []
            ...     },
            ...     {
            ...         "step_id": "step2",
            ...         "description": "Implement authentication",
            ...         "priority": "high",
            ...         "dependencies": ["step1"]
            ...     }
            ... ]
            >>> result = agent._create_plan_tool(
            ...     "Build a web application",
            ...     steps
            ... )
            >>> # Returns: "Plan created successfully with 2 subtasks"
        """
        if self.verbose:
            logger.info(f"Creating plan for task: {task_description}")

        # Store the plan
        self.autonomous_subtasks = []
        for step in steps:
            subtask = {
                "step_id": step.get("step_id", ""),
                "description": step.get("description", ""),
                "priority": step.get("priority", "medium"),
                "dependencies": step.get("dependencies", []),
                "status": "pending",
            }
            self.autonomous_subtasks.append(subtask)
            self.subtask_status[subtask["step_id"]] = "pending"

        self.plan_created = True
        self.current_subtask_index = 0

        if self.verbose:
            logger.info(
                f"Plan created with {len(steps)} steps: {[s['step_id'] for s in self.autonomous_subtasks]}"
            )
        return f"Plan created successfully with {len(steps)} subtasks"

    def _think_tool(
        self,
        current_state: str,
        analysis: str,
        next_actions: List[str],
        confidence: float,
        **kwargs,
    ) -> str:
        """
        Analyze current situation and plan next actions.

        This tool allows the agent to pause and think about the current state of
        task execution, analyze the situation, and plan the next steps. It's used
        in the autonomous loop to enable reflective reasoning before taking action.

        **Thinking Process:**
        1. Agent analyzes the current state of execution
        2. Provides reasoning about the situation
        3. Lists potential next actions
        4. Assigns a confidence level to the analysis

        **Loop Prevention:**
        The method tracks consecutive think calls using self.think_call_count.
        If too many consecutive think calls occur (exceeds max_consecutive_thinks),
        the autonomous loop will force action to prevent infinite thinking loops.

        **Memory Integration:**
        The thinking result is added to conversation memory with format:
        "[THINKING] {analysis}\nNext actions: {actions}\nConfidence: {confidence}"

        Args:
            current_state (str): Description of the current state of task execution.
                This should include what has been accomplished and what remains.
            analysis (str): The agent's analysis of the current situation. Should include
                observations, insights, and reasoning about the current state.
            next_actions (List[str]): List of potential next actions to take. Each action
                should be a clear, actionable step the agent can take.
            confidence (float): Confidence level in the analysis, ranging from 0.0 to 1.0.
                Higher values indicate greater confidence in the analysis and planned actions.
            **kwargs: Additional arguments (currently unused, reserved for future use).

        Returns:
            str: Formatted analysis result string containing:
                - Analysis confirmation
                - Confidence level
                - List of next actions

        Note:
            - This method increments self.think_call_count to track consecutive calls
            - Thinking results are automatically added to conversation memory
            - If verbose=True, thinking details are logged
            - Excessive thinking is prevented by max_consecutive_thinks limit

        Examples:
            >>> result = agent._think_tool(
            ...     current_state="Completed step 1, working on step 2",
            ...     analysis="Step 2 requires additional data from step 1",
            ...     next_actions=["Retrieve data from step 1", "Process the data"],
            ...     confidence=0.85
            ... )
            >>> # Returns formatted analysis with confidence and actions
        """
        # Increment think call count
        self.think_call_count += 1

        if self.verbose:
            logger.info(f"Thinking: {analysis}")
            logger.info(f"Next actions: {next_actions}")
            logger.info(f"Confidence: {confidence}")

        result = f"Analysis complete. Confidence: {confidence}. Next actions: {', '.join(next_actions)}"

        # Add to memory
        self.short_memory.add(
            role=self.agent_name,
            content=f"[THINKING] {analysis}\nNext actions: {', '.join(next_actions)}\nConfidence: {confidence}",
        )

        return result

    def _subtask_done_tool(
        self, task_id: str, summary: str, success: bool, **kwargs
    ) -> str:
        """
        Mark a subtask as completed and move to the next task in the plan.

        This tool is used in the autonomous loop to signal that a subtask has been
        completed (either successfully or with failure). It updates the subtask
        status, stores a summary, and allows the loop to proceed to the next subtask.

        **Status Updates:**
        - Updates self.subtask_status[task_id] to "completed" or "failed"
        - Updates the corresponding subtask in self.autonomous_subtasks
        - Stores the summary in the subtask dictionary
        - Resets think_call_count to allow fresh thinking for next subtask

        **Progress Tracking:**
        - Increments current_subtask_index to move to next subtask
        - The autonomous loop uses this to determine when all subtasks are done

        **Memory Integration:**
        The completion is added to conversation memory with format:
        "[SUBTASK DONE] {task_id}: {summary} (Success: {success})"

        Args:
            task_id (str): The unique identifier (step_id) of the subtask being completed.
                Must match a step_id from the plan created by _create_plan_tool.
            summary (str): A summary of what was accomplished in this subtask. Should
                include key results, findings, or outcomes.
            success (bool): Whether the subtask was completed successfully.
                - True: Subtask completed as intended
                - False: Subtask failed but execution continues
            **kwargs: Additional arguments (currently unused, reserved for future use).

        Returns:
            str: Confirmation message indicating the subtask status. Format:
                "Subtask {task_id} marked as {completed/failed}"

        Note:
            - This method is called automatically by the autonomous loop when a subtask finishes
            - The task_id must exist in autonomous_subtasks
            - Failed subtasks don't block execution but are tracked for final summary
            - Think call count is reset to prevent carryover thinking loops
            - If verbose=True, subtask completion is logged

        Examples:
            >>> result = agent._subtask_done_tool(
            ...     task_id="step1",
            ...     summary="Created project structure with 5 directories",
            ...     success=True
            ... )
            >>> # Returns: "Subtask step1 marked as completed"
            >>> # Updates status and allows loop to proceed to next subtask
        """
        if self.verbose:
            logger.info(f"Completing subtask {task_id}: {summary}")

        # Update subtask status
        if task_id in self.subtask_status:
            self.subtask_status[task_id] = (
                "completed" if success else "failed"
            )

        # Update subtask in list
        for subtask in self.autonomous_subtasks:
            if subtask["step_id"] == task_id:
                subtask["status"] = (
                    "completed" if success else "failed"
                )
                subtask["summary"] = summary
                break

        # Reset think call count when subtask is done
        self.think_call_count = 0

        # Move to next subtask
        self.current_subtask_index += 1

        if self.verbose:
            logger.info(
                f"Subtask {task_id} marked as {'completed' if success else 'failed'}. Moving to next subtask."
            )

        # Add to memory
        self.short_memory.add(
            role=self.agent_name,
            content=f"[SUBTASK DONE] {task_id}: {summary} (Success: {success})",
        )

        return f"Subtask {task_id} marked as {'completed' if success else 'failed'}"

    def _complete_task_tool(
        self,
        task_id: str,
        summary: str,
        success: bool,
        results: Optional[str] = None,
        lessons_learned: Optional[str] = None,
        **kwargs,
    ) -> str:
        """
        Mark the main task as complete and provide comprehensive summary.

        This tool signals that the entire task has been completed and generates
        a comprehensive summary of the entire execution. It's typically called
        at the end of the autonomous loop to provide a final report.

        **Summary Generation:**
        Creates a comprehensive summary including:
        - Task ID and overall status (Success/Failed)
        - High-level summary of the entire task
        - Detailed results (if provided)
        - Lessons learned (if provided)
        - Breakdown of all subtasks with their individual statuses

        **Task Verification:**
        Before completing, the method checks if all subtasks are done. If incomplete
        subtasks exist, a warning is logged but the task can still be marked complete.

        **Memory Integration:**
        The comprehensive summary is added to conversation memory and can be
        retrieved for final output formatting.

        Args:
            task_id (str): The unique identifier of the main task. This should match
                the original task or be a descriptive identifier.
            summary (str): Comprehensive summary of the entire task completion.
                Should cover what was accomplished, key outcomes, and overall status.
            success (bool): Whether the main task was completed successfully.
                - True: Task completed as intended
                - False: Task failed or partially completed
            results (Optional[str]): Detailed results from task execution. Can include
                specific outputs, data, or findings. Defaults to None.
            lessons_learned (Optional[str]): Key insights, patterns, or learnings
                from the task execution. Useful for future reference. Defaults to None.
            **kwargs: Additional arguments (currently unused, reserved for future use).

        Returns:
            str: Comprehensive task completion summary. The summary includes:
                - Task ID and status
                - Summary text
                - Results (if provided)
                - Lessons learned (if provided)
                - Subtask breakdown with individual statuses

        Note:
            - This method is called automatically by the autonomous loop when task is complete
            - The summary replaces the need for a separate summary phase
            - Incomplete subtasks are logged as warnings but don't block completion
            - The comprehensive summary is stored in memory for final output
            - If verbose=True, task completion is logged

        Examples:
            >>> result = agent._complete_task_tool(
            ...     task_id="build_web_app",
            ...     summary="Successfully built web application with authentication",
            ...     success=True,
            ...     results="Created 10 files, implemented 5 features",
            ...     lessons_learned="Authentication should be implemented early"
            ... )
            >>> # Returns comprehensive summary with all details
        """
        if self.verbose:
            logger.info(f"Completing main task {task_id}: {summary}")

        # Verify all subtasks are complete
        incomplete = [
            s["step_id"]
            for s in self.autonomous_subtasks
            if s["status"] not in ["completed", "failed"]
        ]
        if incomplete:
            if self.verbose:
                logger.warning(
                    f"Attempting to complete task but {len(incomplete)} subtasks are not done: {incomplete}"
                )

        # Create comprehensive summary
        comprehensive_summary = f"""Task Completion Summary

Task ID: {task_id}
Status: {'Success' if success else 'Failed'}
Summary: {summary}
"""
        if results:
            comprehensive_summary += f"\nResults:\n{results}\n"
        if lessons_learned:
            comprehensive_summary += (
                f"\nLessons Learned:\n{lessons_learned}\n"
            )

        comprehensive_summary += "\nSubtask Breakdown:\n"
        for subtask in self.autonomous_subtasks:
            comprehensive_summary += f"- {subtask['step_id']}: {subtask.get('status', 'unknown')} - {subtask.get('description', '')}\n"
            if "summary" in subtask:
                comprehensive_summary += (
                    f"  Summary: {subtask['summary']}\n"
                )

        # Add to memory
        self.short_memory.add(
            role=self.agent_name, content=comprehensive_summary
        )

        if self.verbose:
            logger.info(
                "Main task marked as completed with comprehensive summary"
            )
        return comprehensive_summary

    def _get_next_executable_subtask(
        self,
    ) -> Optional[Dict[str, Any]]:
        """
        Get the next executable subtask based on dependencies and status.

        Returns:
            Dictionary of the next subtask or None if all are done
        """
        if not self.autonomous_subtasks:
            return None

        # Find subtasks that are pending and have all dependencies completed
        for subtask in self.autonomous_subtasks:
            if subtask["status"] == "pending":
                # Check if all dependencies are completed
                dependencies = subtask.get("dependencies", [])
                if not dependencies or all(
                    self.subtask_status.get(dep, "completed")
                    in ["completed", "failed"]
                    for dep in dependencies
                ):
                    return subtask

        return None

    def _all_subtasks_complete(self) -> bool:
        """
        Check if all subtasks are completed.

        Returns:
            bool: True if all subtasks are completed or failed
        """
        if not self.autonomous_subtasks:
            return False

        return all(
            subtask["status"] in ["completed", "failed"]
            for subtask in self.autonomous_subtasks
        )

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
        """
        Handle execution of MCP (Model Context Protocol) tools.

        This method processes tool calls from the LLM response and executes them
        using MCP servers. It supports single MCP server (mcp_url or mcp_config)
        and multiple MCP servers (mcp_urls or mcp_configs).

        **MCP Tool Execution:**
        - Single server: Uses execute_tool_call_simple with server_path or connection
        - Multiple servers: Uses execute_multiple_tools_on_multiple_mcp_servers_sync
        - Tool responses are formatted as JSON and added to conversation memory
        - Results are displayed in formatted panels if print_on=True

        **Post-Execution Processing:**
        After tool execution, the method:
        1. Formats tool response as JSON
        2. Adds response to conversation memory
        3. Optionally generates a summary using a temporary LLM instance
        4. Displays summary if print_on=True

        **Error Handling:**
        - AgentMCPConnectionError: Raised if MCP configuration is invalid
        - AgentMCPToolError: Raised if tool execution fails
        - Other exceptions: Logged with full traceback

        Args:
            response (any): The LLM response containing MCP tool calls. Can be:
                - List of tool call dictionaries
                - Single tool call dictionary
                - Tool call in OpenAI function calling format
            current_loop (Optional[int]): The current loop iteration number.
                Used for logging and progress display. Defaults to 0.

        Returns:
            None: This method modifies internal state (adds to memory, displays output)
                but does not return a value.

        Raises:
            AgentMCPConnectionError: If MCP configuration is missing or invalid.
                Raised when neither mcp_url, mcp_config, mcp_urls, nor mcp_configs are set.
            AgentMCPToolError: If MCP tool execution fails.
            Exception: For other unexpected errors during tool execution.

        Note:
            - Requires MCP configuration (mcp_url, mcp_config, mcp_urls, or mcp_configs)
            - Tool responses are automatically formatted as JSON
            - Summary generation uses a temporary LLM instance without tools
            - The method handles both single and multiple MCP server scenarios

        Examples:
            >>> # Single MCP server
            >>> agent = Agent(mcp_url="path/to/mcp/server")
            >>> response = [{"function": {"name": "mcp_tool", "arguments": "{}"}}]
            >>> agent.mcp_tool_handling(response, current_loop=1)

            >>> # Multiple MCP servers
            >>> agent = Agent(mcp_urls=["server1", "server2"])
            >>> agent.mcp_tool_handling(response, current_loop=2)
        """
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
            text_content = f"MCP Tool Response: \n\n {json.dumps(tool_response, indent=2, sort_keys=True)}"

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

            # always log model switches
            logger.warning(
                f"[Model Switch] agent '{self.agent_name}' switching from '{previous_model}' to fallback model: '{new_model}' "
                f"(attempt {self.current_model_index + 1}/{len(available_models)})"
            )

            # Update the model name and reinitialize LLM
            self.model_name = new_model
            self.llm = self.llm_handling()

            return True
        else:
            logger.error(
                f"No more models: agent '{self.agent_name}' has exhausted all available models. "
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
        """
        Execute tools based on LLM response containing function calls.

        This method processes tool calls from the LLM response, executes them,
        and handles the results. It supports both single and multiple tool calls,
        visualizes function calls before execution, and optionally summarizes
        tool execution results.

        **Process Flow:**
        1. Validates response is not None
        2. Visualizes function calls if print_on=True
        3. Executes tools using tool_struct
        4. Adds tool output to conversation memory
        5. Displays execution results (detailed or brief based on show_tool_execution_output)
        6. Optionally generates tool execution summary using LLM

        **Tool Call Format:**
        The method accepts tool calls in two formats:
        - List of tool calls: [{"function": {"name": "...", "arguments": "..."}, "id": "..."}, ...]
        - Single tool call dict: {"function": {"name": "...", "arguments": "..."}, "id": "..."}

        **Visualization:**
        If print_on=True, function calls are visualized with:
        - Function name
        - Call ID (if available)
        - Arguments (truncated if >200 chars)

        **Tool Execution Summary:**
        If tool_call_summary=True, a temporary LLM instance is created to summarize
        tool execution results. This helps the agent understand tool outputs better.

        Args:
            response (any): The LLM response containing tool calls. Can be:
                - List of tool call dictionaries
                - Single tool call dictionary
                - None (will log warning and return early)
            loop_count (int): The current loop iteration number. Used for logging
                and progress tracking.

        Returns:
            None: This method modifies internal state (adds to memory, displays output)
                but does not return a value.

        Raises:
            Exception: If tool execution fails after retry attempts. The error is
                logged with full traceback before raising.

        Note:
            - Tool execution results are automatically formatted and added to memory
            - If show_tool_execution_output=False, only brief confirmation is shown
            - Tool execution summary uses a temporary LLM instance without tools
            - The method handles both JSON string and dict format for arguments

        Examples:
            >>> # Single tool call
            >>> response = [{
            ...     "function": {"name": "search_web", "arguments": '{"query": "Python"}'},
            ...     "id": "call_123"
            ... }]
            >>> agent.execute_tools(response, loop_count=1)

            >>> # Multiple tool calls
            >>> response = [
            ...     {"function": {"name": "tool1", "arguments": "{}"}, "id": "call_1"},
            ...     {"function": {"name": "tool2", "arguments": "{}"}, "id": "call_2"}
            ... ]
            >>> agent.execute_tools(response, loop_count=2)
        """
        # Handle None response gracefully
        if response is None:
            logger.warning(
                f"Cannot execute tools with None response in loop {loop_count}. "
                "This may indicate the LLM did not return a valid response."
            )
            return

        # Visualize function calls before execution
        if self.print_on:
            # Handle both list and single dict responses
            tool_calls_to_visualize = []
            if isinstance(response, list):
                tool_calls_to_visualize = response
            elif isinstance(response, dict):
                # Single tool call as dict
                tool_calls_to_visualize = [response]

            for tool_call in tool_calls_to_visualize:
                if isinstance(tool_call, dict):
                    func_name = tool_call.get("function", {}).get(
                        "name", "Unknown"
                    )
                    func_args = {}
                    tool_call_id = tool_call.get("id", "N/A")

                    try:
                        func_args = json.loads(
                            tool_call.get("function", {}).get(
                                "arguments", "{}"
                            )
                        )
                    except (
                        json.JSONDecodeError,
                        AttributeError,
                        TypeError,
                    ):
                        # If arguments is already a dict, use it directly
                        func_args = tool_call.get("function", {}).get(
                            "arguments", {}
                        )
                        if not isinstance(func_args, dict):
                            func_args = {}

                    # Visualize the function call with enhanced details
                    call_content = f"Function: {func_name}\n"
                    if tool_call_id != "N/A":
                        call_content += f"Call ID: {tool_call_id}\n"
                    call_content += "\nArguments:\n"
                    for key, value in func_args.items():
                        # Truncate long values for readability
                        value_str = str(value)
                        if len(value_str) > 200:
                            value_str = value_str[:200] + "..."
                        call_content += f"  {key}: {value_str}\n"

                    formatter.print_panel(
                        call_content,
                        title=f"Agent: {self.agent_name} Function Call",
                    )

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
            # Extract tool names and details from response for better display
            tool_names = []
            tool_details = []

            # Handle both list and single dict responses
            tool_calls_to_process = []
            if isinstance(response, list):
                tool_calls_to_process = response
            elif isinstance(response, dict):
                tool_calls_to_process = [response]

            for tool_call in tool_calls_to_process:
                if isinstance(tool_call, dict):
                    func_name = tool_call.get("function", {}).get(
                        "name", "Unknown"
                    )
                    tool_names.append(func_name)
                    tool_details.append(
                        {
                            "name": func_name,
                            "id": tool_call.get("id", "N/A"),
                            "type": tool_call.get("type", "function"),
                        }
                    )

            if self.show_tool_execution_output is True:
                # Create detailed output display with enhanced information
                tool_display = (
                    f"Execution Time: {time.strftime('%H:%M:%S')}\n\n"
                )

                if tool_details:
                    tool_display += "Tools Executed:\n"
                    for detail in tool_details:
                        tool_display += f"  - {detail['name']}"
                        if detail["id"] != "N/A":
                            tool_display += f" (ID: {detail['id']})"
                        tool_display += f" [{detail['type']}]\n"
                    tool_display += "\n"

                # Format output for better readability
                output_str = format_data_structure(output)
                tool_display += f"Output:\n{output_str}"

                # Show results in a panel
                formatter.print_panel(
                    tool_display,
                    title="Tool Execution Results",
                )
            else:
                # Show brief execution confirmation with tool names
                if tool_names:
                    brief_display = (
                        f"Tools Executed: {', '.join(tool_names)}\n"
                    )
                    brief_display += (
                        f"Time: {time.strftime('%H:%M:%S')}"
                    )
                    formatter.print_panel(
                        brief_display,
                        title="Tool Execution",
                    )
                else:
                    formatter.print_panel(
                        f"Tool Executed Successfully [{time.strftime('%H:%M:%S')}]",
                        title="Tool Execution",
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

    def tool_execution_retry(self, response: any, loop_count: int):
        """
        Execute tools with retry logic for handling failures.

        This method provides a robust wrapper around tool execution with automatic
        retry on failure. It handles None responses gracefully and implements
        retry logic using the configured tool_retry_attempts.

        **Retry Strategy:**
        - If tool execution fails, the method automatically retries
        - Maximum retry attempts are controlled by self.tool_retry_attempts (default: 3)
        - Each retry is logged with detailed error information
        - After all retries are exhausted, the exception is re-raised

        **Error Handling:**
        - None responses: Logs warning and skips execution (does not raise)
        - AgentToolExecutionError: Logs error with full traceback and retries
        - Other exceptions: Logs error and retries

        **Logging:**
        All errors are logged with:
        - Agent name for identification
        - Loop count for context
        - Full traceback for debugging
        - Retry attempt number

        Args:
            response (any): The response from the LLM that may contain tool calls to execute.
                Can be:
                - List of tool call dictionaries
                - Single tool call dictionary
                - None (will log warning and return without raising)
            loop_count (int): The current iteration loop number. Used for:
                - Logging context
                - Error reporting
                - Debugging tool execution issues

        Returns:
            None: This method modifies internal state but does not return a value.

        Raises:
            AgentToolExecutionError: If tool execution fails after all retry attempts.
            Exception: Any other exception that occurs during tool execution after
                retries are exhausted.

        Note:
            - Uses self.tool_retry_attempts (default: 3) for maximum retry attempts
            - None responses are handled gracefully without raising exceptions
            - Detailed error logging helps with debugging tool execution issues
            - The method delegates actual tool execution to execute_tools()

        Examples:
            >>> # Normal execution
            >>> response = [{"function": {"name": "my_tool", "arguments": "{}"}}]
            >>> agent.tool_execution_retry(response, loop_count=1)

            >>> # Handles None response gracefully
            >>> agent.tool_execution_retry(None, loop_count=2)
            >>> # Logs warning but does not raise exception
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
