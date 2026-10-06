import asyncio
import json
import os
import threading
import time
import traceback
from contextlib import nullcontext
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Literal,
    Optional,
    Sequence,
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
    get_model_info,
    supports_function_calling,
)
from loguru import logger
from pydantic import BaseModel

from swarms.agents.agent_marketplace_handler import (
    AgentMarketplaceHandler,
)
from swarms.agents.ape_agent import auto_generate_prompt
from swarms.agents.autonomous_loop import AutonomousAgentLoop
from swarms.agents.context_compressor import ContextCompressor
from swarms.agents.llm_manager import LLMManager
from swarms.agents.skills_manager import SkillsManager
from swarms.agents.tool_manager import ToolManager
from swarms.prompts.agent_system_prompts import (
    build_agent_system_prompt,
)
from swarms.prompts.autonomous_agent_prompt import (
    get_autonomous_agent_prompt,
)
from swarms.prompts.max_loop_prompt import generate_reasoning_prompt
from swarms.prompts.multi_modal_autonomous_instruction_prompt import (
    MULTI_MODAL_AUTO_AGENT_SYSTEM_PROMPT_1,
)
from swarms.prompts.react_base_prompt import REACT_SYS_PROMPT
from swarms.prompts.safety_prompt import SAFETY_PROMPT
from swarms.schemas.agent_errors import (
    AgentInitializationError,
    AgentLLMError,
    AgentRunError,
    AgentToolExecutionError,
)
from swarms.schemas.mcp_schemas import (
    MCPConnection,
    MCPOAuthConfig,
)
from swarms.structs.agent_roles import agent_roles
from swarms.structs.autonomous_loop_utils import (
    MAX_PLANNING_ATTEMPTS,
    MAX_SUBTASK_ITERATIONS,
    MAX_SUBTASK_LOOPS,
    get_summary_prompt,
)
from swarms.structs.conversation import Conversation
from swarms.structs.ma_utils import set_random_models_for_agents
from swarms.structs.safe_loading import (
    SafeLoaderUtils,
    SafeStateManager,
)
from swarms.structs.transcript import Transcript
from swarms.structs.transforms import (
    MessageTransforms,
    TransformConfig,
    handle_transforms,
)
from swarms.telemetry.otel import (
    ContextThreadPoolExecutor,
    capture_error,
    capture_init,
    log_agent_data,
    trace_run,
)
from swarms.tools.dynamic_tool_loader import (
    DynamicToolLoader,
)
from swarms.tools.mcp_manager import MCPManager
from swarms.utils.file_processing import create_file_in_folder
from swarms.utils.formatter import formatter
from swarms.utils.generate_id import generate_id
from swarms.utils.generate_keys import generate_api_key
from swarms.utils.get_reasoning_efforts import ReasoningEffort
from swarms.utils.history_output_formatter import (
    history_output_formatter,
)
from swarms.utils.index import (
    exists,
    format_data_structure,
)
from swarms.utils.litellm_tokenizer import count_tokens
from swarms.utils.litellm_wrapper import empty_usage
from swarms.utils.output_types import OutputType
from swarms.utils.workspace_manager import WorkspaceManager
from swarms.utils.workspace_utils import get_workspace_dir


def stop_when_repeats(response: str) -> bool:
    # Stop if the word stop appears in the response
    return "stop" in response.lower()


# Agent ID generator
def agent_id() -> str:
    """Deprecated: use ``generate_id("agent")``."""
    return generate_id("agent")


# Agent output types
ToolUsageType = Union[BaseModel, Dict[str, Any]]


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
        tool_schema (ToolUsageType): The tool schema
        output_type (agent_output_type): The output type. Supported: 'str', 'string', 'list', 'json', 'dict', 'yaml'.
        output_cleaner (Callable): The output cleaner function
        list_base_models (List[BaseModel]): The list of base models
        rules (str): The rules
        planning_prompt (str): The planning prompt
        max_tokens (int): The maximum number of tokens
        temperature (float): The temperature
        workspace_dir (str, optional): Ignored - workspace directory is always read from
            the 'workspace_dir' environment variable. Defaults to 'agent_workspace' if
            the environment variable is not set.
        marketplace_prompt_id (str): The unique UUID identifier of a prompt from the Swarms marketplace.
            When provided, the agent will automatically fetch and load the prompt from the marketplace
            as the system prompt. This enables one-line prompt loading from the Swarms marketplace.
            Requires the SWARMS_API_KEY environment variable to be set.
        skills_dir (str): Path to directory containing Agent Skills in SKILL.md format.
            Implements Anthropic's Agent Skills framework for modular, composable capabilities.
            Each subdirectory should contain a SKILL.md file with YAML frontmatter (name, description)
            and markdown instructions. Skills are auto-loaded into system prompt for context-aware activation.
            Example: skills_dir="./skills" loads from ./skills/*/SKILL.md
        think_tool (bool): Whether the autonomous looper (max_loops="auto") offers the
            `think` tool. Defaults to False. A `think` call spends a full round-trip to
            produce reasoning the model could emit inline alongside its actions, so it is
            off unless asked for. Enable it for models that do not reason natively, or
            when an explicit analysis step is worth the extra turn. When False, the system
            prompt is adjusted to match so the model is not told to call a tool it lacks.
        max_planning_attempts (int): Autonomous loop (max_loops="auto") only. How many
            times to ask the model for a plan before giving up. Defaults to 5.
        max_subtask_iterations (int): Autonomous loop only. Ceiling on execution
            iterations across the whole run, which is also its worst-case number of
            LLM calls. Defaults to 100.
        max_subtask_loops (int): Autonomous loop only. Ceiling on iterations spent
            inside any one subtask before moving on. Defaults to 20.
        selected_tools (Union[str, List[str]]): Tools to enable for the autonomous looper when max_loops="auto".
            Available tools: "create_plan", "think", "subtask_done", "complete_task", "respond_to_user",
            "create_file", "update_file", "read_file", "list_directory", "delete_file", "run_bash",
            "create_sub_agent", "assign_task".
            Defaults to "all" (all tools enabled). Pass a list of tool names to restrict tools, or "all"
            for unrestricted access. Use this to control which tools the agent can use during autonomous execution.
        prompt_caching (bool): Enable provider-side prompt caching. When True, ephemeral
            cache_control breakpoints are added to the stable prefix of each request (system
            prompt, tools, and the last message) so it is cached and re-billed at a discount.
            Applies to the Anthropic model family (Claude on Anthropic / Bedrock / Vertex);
            providers that cache automatically (e.g. OpenAI) are left untouched. Defaults to False.
        cache_config (dict): Fine-grained prompt-caching options; only consulted when
            prompt_caching=True. All keys optional:
                "ttl" (str): "5m" (default) or "1h" for Anthropic's extended cache.
                "cache_system_prompt" (bool): cache the system prefix (default True).
                "cache_messages" (bool): cache through the last message (default True).
                "cache_tools" (bool): cache the tool-definitions block (default True).
                "override" (bool): force cache_control injection on/off regardless of the
                    detected provider (e.g. opt Gemini/Vertex in, or a custom alias out).
                    Default None (auto-detect: Anthropic only).
                "prompt_cache_key" (str): OpenAI routing hint for higher cache hit rates.
                "prompt_cache_retention" (str): OpenAI cache TTL ("in_memory" | "24h").
            Defaults to None.
        mcp_url (Union[str, MCPConnection, dict]): A single MCP server. Pass a URL string for
            an unauthenticated server, or an MCPConnection/dict to configure auth, transport,
            headers and timeouts.
        mcp_urls (List[Union[str, MCPConnection, dict]]): Several MCP servers. Tools from every
            server are merged and each tool call is routed back to the server that owns it.
        mcp_config (Union[MCPConnection, dict]): A single MCP server given as a connection object.
        mcp_configs (List[Union[MCPConnection, dict]]): Several MCP servers given as connection objects.
        mcp_api_key (str): API key applied to every MCP server that does not define its own.
            Sent as "Authorization: Bearer <key>" by default; override the header or prefix
            per-server with MCPConnection(api_key_header=..., api_key_prefix=...). Supports
            "env:MY_VAR" / "${MY_VAR}" indirection so secrets stay out of code.
        mcp_authorization_token (str): Bearer token applied to every MCP server that does not
            define its own. Equivalent to mcp_api_key with the default header/prefix.
        mcp_oauth (Union[MCPOAuthConfig, dict]): OAuth 2.1 settings applied to every MCP server
            without its own. Supports the interactive authorization-code flow (PKCE + dynamic
            client registration, tokens cached on disk), the headless client_credentials grant,
            and pre-issued access tokens.
        mcp_headers (Dict[str, str]): Extra headers merged into every MCP request.
        mcp_transport (str): Force a transport for every MCP server: "streamable_http", "sse",
            "stdio", or "auto". Defaults to auto-detection from the URL.
        mcp_timeout (int): Request timeout in seconds for every MCP server. Defaults to 30.

    Methods:
        run: Run the agent
        run_concurrent: Run the agent concurrently
        bulk_run: Run the agent in bulk
        save: Save the agent
        load: Load the agent
        validate_response: Validate the response
        print_history_and_memory: Print the history and memory
        step: Step through the agent
        run_with_timeout: Run the agent with a timeout
        load_skills_metadata: Load Agent Skills metadata from directory
        load_full_skill: Load complete skill content (Tier 2 loading)
        analyze_feedback: Analyze the feedback
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


    Examples:
    >>> from swarms import Agent
    >>> agent = Agent(model_name="gpt-5.4", max_loops=1)
    >>> response = agent.run("Generate a report on the financials.")
    >>> print(response)
    >>> # Generate a report on the financials.

    >>> # Detailed token streaming example
    >>> agent = Agent(model_name="gpt-5.4", max_loops=1, stream=True)
    >>> response = agent.run("Tell me a story.")  # Will stream each token with detailed metadata
    >>> print(response)  # Final complete response

    >>> # Fallback model example
    >>> agent = Agent(
    ...     fallback_models=["gpt-5.4", "gpt-5.4", "gpt-3.5-turbo"],
    ...     max_loops=1
    ... )
    >>> response = agent.run("Generate a report on the financials.")
    >>> # Will try gpt-4o first, then gpt-4o-mini, then gpt-3.5-turbo if each fails

    >>> # Marketplace prompt example - load a prompt in one line
    >>> agent = Agent(
    ...     model_name="gpt-5.4",
    ...     marketplace_prompt_id="550e8400-e29b-41d4-a716-446655440000",
    ...     max_loops=1
    ... )
    >>> response = agent.run("Execute the marketplace prompt task")
    >>> # The agent automatically loads the system prompt from the Swarms marketplace

    """

    def __init__(
        self,
        id: Optional[str] = None,
        agent_name: Optional[str] = "swarm-worker-01",
        agent_description: Optional[
            str
        ] = "An autonomous agent that can perform tasks and learn from experience powered by Swarms",
        system_prompt: Optional[str] = None,
        llm: Optional[Any] = None,
        max_loops: Optional[Union[int, str]] = 1,
        stopping_condition: Optional[Callable[[str], bool]] = None,
        loop_interval: Optional[int] = 0,
        retry_attempts: Optional[int] = 3,
        stopping_token: Optional[str] = None,
        dynamic_loops: Optional[bool] = False,
        interactive: Optional[bool] = False,
        dashboard: Optional[bool] = False,
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
        tool_schema: ToolUsageType = None,
        output_type: OutputType = "str-all-except-first",
        output_cleaner: Optional[Callable] = None,
        list_base_models: Optional[List[BaseModel]] = None,
        rules: str = None,  # type: ignore
        planning_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tags: Optional[List[str]] = None,
        auto_generate_prompt: bool = False,
        plan_enabled: bool = False,
        model_name: str = "gpt-5.4",
        llm_args: dict = None,
        prompt_caching: bool = False,
        cache_config: dict = None,
        load_state_path: str = None,
        role: agent_roles = "worker",
        print_on: bool = True,
        tools_list_dictionary: Optional[List[Dict[str, Any]]] = None,
        mcp_url: Optional[Union[str, MCPConnection, Dict]] = None,
        mcp_urls: Optional[
            List[Union[str, MCPConnection, Dict]]
        ] = None,
        react_on: bool = False,
        safety_prompt_on: bool = False,
        random_models_on: bool = False,
        mcp_config: Optional[Union[MCPConnection, Dict]] = None,
        mcp_configs: Optional[
            List[Union[MCPConnection, Dict]]
        ] = None,
        mcp_api_key: Optional[str] = None,
        mcp_authorization_token: Optional[str] = None,
        mcp_oauth: Optional[Union[MCPOAuthConfig, Dict]] = None,
        mcp_headers: Optional[Dict[str, str]] = None,
        mcp_transport: Optional[
            Literal["streamable_http", "sse", "stdio", "auto"]
        ] = None,
        mcp_timeout: Optional[int] = None,
        top_p: Optional[float] = None,
        llm_base_url: Optional[str] = None,
        llm_api_key: Optional[str] = None,
        tool_call_summary: bool = False,
        tool_retry_attempts: int = 3,
        reasoning_prompt_on: bool = True,
        dynamic_context_window: bool = True,
        show_tool_execution_output: bool = True,
        reasoning_effort: Optional[ReasoningEffort] = None,
        thinking_tokens: int = 1024,
        think_tool: bool = False,
        max_planning_attempts: int = MAX_PLANNING_ATTEMPTS,
        max_subtask_iterations: int = MAX_SUBTASK_ITERATIONS,
        max_subtask_loops: int = MAX_SUBTASK_LOOPS,
        dynamic_tools: bool = False,
        reasoning_enabled: bool = False,
        handoffs: Optional[Union[Sequence[Callable], Any]] = None,
        capabilities: Optional[List[str]] = None,
        mode: Literal["interactive", "fast", "standard"] = "standard",
        publish_to_marketplace: bool = False,
        use_cases: Optional[List[Dict[str, Any]]] = None,
        marketplace_prompt_id: Optional[str] = None,
        skills_dir: Optional[str] = None,
        selected_tools: Optional[Union[str, List[str]]] = "all",
        context_compression: bool = True,
        persistent_memory: bool = False,
        messages: Optional[List[Dict[str, Any]]] = None,
        *args,
        **kwargs,
    ):
        # super().__init__(*args, **kwargs)
        self.id = id or generate_id("agent")
        self.skills = SkillsManager(skills_dir=skills_dir)
        self._skills_prompt = ""
        self.selected_tools = selected_tools
        self.llm = llm
        self.max_loops = max_loops
        self.stopping_condition = stopping_condition
        self.loop_interval = loop_interval
        self.retry_attempts = retry_attempts
        self.task = None
        self.stopping_token = stopping_token
        self.interactive = interactive
        self.dashboard = dashboard
        self.dynamic_temperature_enabled = dynamic_temperature_enabled
        self.dynamic_loops = dynamic_loops
        self.user_name = user_name
        self.context_length = context_length
        self.sop = sop
        self.sop_list = sop_list
        self.tools = tools
        if system_prompt is None:
            system_prompt = build_agent_system_prompt()
        self.system_prompt = system_prompt or ""
        self.agent_name = agent_name
        self.agent_description = agent_description
        # Fallback: this once overwrote the caller's own path.
        self.saved_state_path = saved_state_path or (
            f"{generate_api_key(prefix='agent-')}_state.json"
        )
        self.autosave = autosave
        self.multi_modal = multi_modal
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
        self.output_cleaner = output_cleaner
        self.list_base_models = list_base_models
        self.planning_prompt = planning_prompt
        self.rules = rules
        self.max_tokens = max_tokens
        self.temperature = temperature
        # The environment wins over the argument, with a default when unset
        self.workspace_dir = get_workspace_dir()
        # Built on first use, constructing an agent must not create a directory
        self._workspace = None
        self.tags = tags
        self.use_cases = use_cases
        self.name = agent_name
        self.description = agent_description
        self.auto_generate_prompt = auto_generate_prompt
        self.plan_enabled = plan_enabled
        self.model_name = model_name
        self.llm_args = llm_args
        self.prompt_caching = prompt_caching
        self.cache_config = cache_config
        self.load_state_path = load_state_path
        self.role = role
        self.print_on = print_on
        # Own list per agent: a literal [] default made every agent share one object.
        self.tools_list_dictionary = (
            tools_list_dictionary
            if tools_list_dictionary is not None
            else []
        )
        self.mcp_url = mcp_url
        self.mcp_urls = mcp_urls
        self.react_on = react_on
        self.safety_prompt_on = safety_prompt_on
        self.random_models_on = random_models_on
        self.mcp_config = mcp_config
        self.mcp_configs = mcp_configs
        self.mcp_api_key = mcp_api_key
        self.mcp_authorization_token = mcp_authorization_token
        self.mcp_oauth = mcp_oauth
        self.mcp_headers = mcp_headers
        self.mcp_transport = mcp_transport
        self.mcp_timeout = mcp_timeout
        self.top_p = top_p
        self.llm_base_url = llm_base_url
        self.llm_api_key = llm_api_key
        self.tool_call_summary = tool_call_summary
        self.tool_retry_attempts = tool_retry_attempts
        self.reasoning_prompt_on = reasoning_prompt_on
        self.dynamic_context_window = dynamic_context_window
        self.show_tool_execution_output = show_tool_execution_output
        self.reasoning_effort = reasoning_effort
        self.thinking_tokens = thinking_tokens

        self.dynamic_tools = dynamic_tools
        self.tool_loader: Optional[DynamicToolLoader] = None
        self._mcp_tools_deferred = False
        self._usage = empty_usage()
        self._mcp_schemas_cache: Optional[List[dict]] = None

        self.think_tool = think_tool
        # A budget below 1 would silently make the phase it bounds do nothing.
        for name, value in (
            ("max_planning_attempts", max_planning_attempts),
            ("max_subtask_iterations", max_subtask_iterations),
            ("max_subtask_loops", max_subtask_loops),
        ):
            if value < 1:
                raise ValueError(
                    f"{name} must be at least 1, got {value}"
                )
        self.max_planning_attempts = max_planning_attempts
        self.max_subtask_iterations = max_subtask_iterations
        self.max_subtask_loops = max_subtask_loops
        self.reasoning_enabled = reasoning_enabled
        self.fallback_model_name = fallback_model_name
        self.handoffs = handoffs
        self.capabilities = capabilities
        self.mode = mode
        self.publish_to_marketplace = publish_to_marketplace
        self.marketplace_prompt_id = marketplace_prompt_id

        self.mcp_manager = MCPManager(
            mcp_url=self.mcp_url,
            mcp_urls=self.mcp_urls,
            mcp_config=self.mcp_config,
            mcp_configs=self.mcp_configs,
            api_key=self.mcp_api_key,
            authorization_token=self.mcp_authorization_token,
            oauth=self.mcp_oauth,
            headers=self.mcp_headers,
            transport=self.mcp_transport,
            timeout=self.mcp_timeout,
            agent_name=self.agent_name,
            verbose=self.verbose,
            retry_attempts=self.tool_retry_attempts,
        )

        if self.context_length is None:
            self.context_length = self._default_context_length()

        if self.max_tokens is None or self.max_tokens <= 0:
            self.max_tokens = self._default_max_tokens() or 16000

        if self.max_loops == "auto":
            # Without this the prompt tells the model to call a think tool it lacks
            self.system_prompt += (
                "\n\n"
                + get_autonomous_agent_prompt(
                    include_think_tool=self.think_tool
                )
            )

        # When False the agent does not read or write MEMORY.md across sessions.
        self.persistent_memory = persistent_memory

        # Prior turns, seeded into short_memory and re-sent as context on every run.
        self.messages = messages

        # Applies to auto and integer max_loops alike
        self.context_compression = context_compression
        if self.context_compression:
            self._context_compressor = ContextCompressor(
                threshold=0.9
            )
        else:
            self._context_compressor = None

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

        # Async subagent support
        self._subagent_registry = None

        # Owns fetching prompts from, and publishing prompts to, the Swarms Marketplace
        self.marketplace = AgentMarketplaceHandler(agent=self)

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

        # If fallback_models is provided, use the first model as the primary model
        if self.fallback_models and not self.model_name:
            self.model_name = self.fallback_models[0]

        # Reads config off this agent, so it must come after the config is set
        self.llm_manager = LLMManager(agent=self)
        self.tool_manager = ToolManager(agent=self)
        self.autonomous_loop = AutonomousAgentLoop(agent=self)

        # self.init_handling()
        self.setup_config()

        # Conversation owns MEMORY.md under $WORKSPACE_DIR/agents/{agent_name}-{id}/.
        self.short_memory = self.short_memory_init()

        # Initialize the tools
        self.tool_manager.setup_tools()

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

        self.tool_manager.load_tools()

        if self.llm is None:
            self.llm = self.llm_handling()
        elif getattr(
            self.llm, "usage_hook", None
        ) is None and hasattr(self.llm, "usage_hook"):
            # A caller-supplied LiteLLM reports into this agent too.
            self.llm.usage_hook = self._add_usage

        if self.random_models_on is True:
            self.model_name = set_random_models_for_agents()

        if self.dashboard is True:
            self.print_dashboard()

        self.reliability_check()

        if self.mode == "fast":
            self.print_on = False
            self.verbose = False

        if self.publish_to_marketplace is True:
            self.handle_publish_to_marketplace()

        # Capture the full __init__ configuration if telemetry is enabled.
        capture_init(self)

    def handle_publish_to_marketplace(self):
        """
        Publish this agent's prompt and metadata to the Swarms Marketplace.

        Requires `use_cases` to be set and SWARMS_API_KEY to be present.
        """
        return self.marketplace.publish()

    @property
    def skills_dir(self) -> Optional[str]:
        """Directory the agent loads Agent Skills from."""
        return self.skills.skills_dir

    @skills_dir.setter
    def skills_dir(self, skills_dir: Optional[str]) -> None:
        self.skills.set_skills_dir(skills_dir)

    @property
    def skills_metadata(self) -> List[Dict[str, str]]:
        """Metadata for the skills loaded so far."""
        return self.skills.metadata

    @skills_metadata.setter
    def skills_metadata(self, metadata: List[Dict[str, str]]) -> None:
        self.skills.metadata = metadata

    def handle_skills(self, task: Optional[str] = None):
        """
        Select the Agent Skills for this run.

        The rendered section is sent with every LLM call by :meth:`call_llm`.
        ``system_prompt`` is left unchanged: the LLM copies it once at
        construction, so text appended to it here never reached the model,
        and each run appended another copy.

        Args:
            task: Optional task description. If provided, loads skills dynamically
                  based on similarity to the task. If not provided, loads all skills statically.
        """
        self._skills_prompt = self.skills.prompt_for_task(task)

    @property
    def workspace(self) -> "WorkspaceManager":
        """
        This agent's workspace manager, created on first access.

        Returns:
            WorkspaceManager: Rooted at ``{workspace}/agents/{name}-{uuid}``.
        """
        if self._workspace is None:
            self._workspace = WorkspaceManager.for_agent(
                self, verbose=self.verbose
            )
        return self._workspace

    def _get_agent_workspace_dir(self) -> str:
        """
        Get the agent-specific workspace directory path.

        Creates a unique subdirectory for each agent instance in the format:
        workspace_dir/agents/{name-of-agent}-{uuid}/

        Returns:
            str: The full path to the agent-specific workspace directory.
        """
        return self.workspace.dir

    def short_memory_init(self):
        # Compactly assemble initial prompt as a string with available fields
        prompt = self.system_prompt

        if self.safety_prompt_on is True:
            prompt += "\n\n"
            prompt += SAFETY_PROMPT

        # Keyed on agent_name, not self.id, which is a fresh uuid per run and would orphan MEMORY.md.
        memory_md_path = None
        if self.persistent_memory:
            try:
                base = get_workspace_dir() or os.path.join(
                    os.getcwd(), "agent_workspace"
                )
                memory_md_path = os.path.join(
                    base, "agents", self.agent_name, "MEMORY.md"
                )
            except Exception as e:
                logger.error(f"Failed to resolve MEMORY.md path: {e}")

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
            memory_md_path=memory_md_path,
            messages=self.messages,
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
        self.llm = self.llm_manager.build(*args, **kwargs)
        return self.llm

    @property
    def mcp_enabled(self) -> bool:
        """
        Whether this agent has at least one MCP server configured.

        Backed by the agent's :class:`MCPManager`, which normalizes
        ``mcp_url``, ``mcp_urls``, ``mcp_config`` and ``mcp_configs`` into a
        single list of connections.
        """
        manager = getattr(self, "mcp_manager", None)
        return manager is not None and manager.enabled

    def _load_prompt_from_marketplace(self) -> None:
        """
        Load a prompt from the Swarms marketplace using the marketplace_prompt_id.

        Appends the marketplace prompt to this agent's system prompt and
        back-fills `agent_name` / `agent_description` when they are still at
        their defaults.

        Raises:
            ValueError: If the prompt cannot be found in the marketplace.
            Exception: If there's an error fetching the prompt from the API.

        Note:
            Requires the SWARMS_API_KEY environment variable to be set for
            authenticated API access.
        """
        self.marketplace.load_prompt()

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
        return self.llm_manager.check_model_supports_utilities(
            img=img
        )

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
        Randomly reset the LLM's temperature on a 0.0-1.0 scale between loops.
        Falls back to 0.5 when the LLM exposes no temperature attribute.
        """
        self.llm_manager.randomize_temperature()

    def print_dashboard(self):
        """
        Print a dashboard displaying the agent's current status and configuration.
        Uses square brackets instead of emojis for section headers and bullet points.
        """
        tools_activated = True if self.tools is not None else False
        mcp_activated = self.mcp_enabled
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

    # Main function
    def _run(
        self,
        task: Optional[Union[str, Any]] = None,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        streaming_callback: Optional[Callable[[str], None]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
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
            history_start = len(
                self.short_memory.conversation_history
            )

            self.check_if_no_prompt_then_autogenerate(task)

            self.check_model_supports_utilities(img=img)

            self.short_memory.add(role=self.user_name, content=task)

            if self.plan_enabled is True:
                self.plan(task)

            # Set the loop count
            loop_count = 0

            # Built lazily so the transforms path can keep its flattened prompt
            transcript: Optional[Transcript] = None

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

                if self._context_compressor is not None:
                    self._context_compressor.maybe_compress(self)

                # Autosave config at the start of each loop step
                if self.autosave:
                    self._autosave_config_step(loop_count=loop_count)

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

                # Task prompt with optional transforms.
                task_prompt = None
                use_transcript = self.transforms is None

                if self.transforms is not None:
                    task_prompt = handle_transforms(
                        transforms=self.transforms,
                        short_memory=self.short_memory,
                        model_name=self.model_name,
                    )
                elif transcript is None:
                    transcript = self._transcript_from_messages(
                        messages, task
                    )

                # Parameters
                attempt = 0
                success = False
                last_error: Optional[Exception] = None
                while attempt < self.retry_attempts and not success:
                    # Outside the try: except must answer tool calls.
                    turn_calls = []
                    turn_results = {}
                    try:

                        show_loading = (
                            self.interactive
                            and not self.streaming_on
                            and not self.stream
                        )
                        loading_ctx = (
                            formatter.loading_status(
                                f"👾 Agent: {self.agent_name} is thinking..."
                            )
                            if show_loading
                            else nullcontext()
                        )

                        with loading_ctx:
                            llm_kwargs = dict(kwargs)
                            if use_transcript:
                                llm_kwargs["messages"] = (
                                    transcript.messages
                                )

                            response = self.call_llm(
                                task=task_prompt,
                                img=img,
                                imgs=imgs,
                                current_loop=loop_count,
                                streaming_callback=streaming_callback,
                                *args,
                                **llm_kwargs,
                            )

                        response = self.tool_manager.parse_response(
                            response
                        )

                        self.short_memory.add(
                            role=self.agent_name,
                            content=response,
                        )

                        # Every tool call in this turn needs a matching result before the next request.
                        if use_transcript:
                            turn_calls = transcript.record_assistant(
                                response
                            )

                        # Print
                        if self.print_on is True:
                            # Tool calls are visualised in execute_tools
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

                        self.tool_manager.handle_tool_calls(
                            response,
                            loop_count,
                            transcript=(
                                transcript if use_transcript else None
                            ),
                            turn_calls=turn_calls,
                            turn_results=turn_results,
                        )

                        success = True  # Mark as successful to exit the retry loop

                        # Autosave config after successful step
                        if self.autosave:
                            self._autosave_config_step(
                                loop_count=loop_count
                            )

                    except AgentToolExecutionError as e:
                        # A tool failure is not a provider failure, re-running the model cannot fix it
                        if use_transcript and turn_calls:
                            transcript.flush_tool_results(
                                turn_calls, turn_results
                            )

                        capture_error(
                            e,
                            self,
                            name="Agent.tool_error",
                            loop=loop_count,
                        )

                        self.short_memory.add(
                            role="Tool Executor",
                            content=(
                                f"Tool execution failed after "
                                f"{self.tool_retry_attempts} attempts: {e}"
                            ),
                        )

                        # Exit the retry loop, not the run, so the model can read the failure
                        success = True

                    except (
                        BadRequestError,
                        InternalServerError,
                        AuthenticationError,
                        Exception,
                    ) as e:
                        last_error = e

                        # Answer the recorded tool calls so the retried request is well formed
                        if use_transcript and turn_calls:
                            transcript.flush_tool_results(
                                turn_calls, turn_results
                            )

                        # The retry loop swallows this, so capture_run never sees it
                        capture_error(
                            e,
                            self,
                            name="Agent.llm_error",
                            loop=loop_count,
                        )

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
                    # Drop this run's turns so a fallback model starts from clean history.
                    while (
                        len(self.short_memory.conversation_history)
                        > history_start
                    ):
                        self.short_memory.delete(
                            len(
                                self.short_memory.conversation_history
                            )
                            - 1
                        )

                    raise AgentLLMError(
                        f"Agent '{self.agent_name}' got no response from "
                        f"'{self.model_name}' after {self.retry_attempts} "
                        f"attempt(s): {last_error}"
                    ) from last_error

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
                    formatter.console.print()
                    try:
                        user_input = formatter.console.input(
                            "[bold cyan]You[/bold cyan] [bold green]❯[/bold green] "
                        )
                    except (KeyboardInterrupt, EOFError):
                        # Ctrl+C / Ctrl+D during input exits without a traceback
                        formatter.console.print()
                        self.pretty_print(
                            "Session ended by user. Goodbye.",
                            loop_count=loop_count,
                        )
                        break

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
                    if transcript is not None:
                        transcript.append_user(user_input)

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
        Write a config snapshot to the agent workspace, once per step.

        Args:
            loop_count (Optional[int]): Current loop, recorded in the
                saved metadata and used only for logging. Defaults to None.

        Note:
            Writes ``config.json`` under
            workspace_dir/agents/{name-of-agent}-{uuid}/. Never raises -
            autosave must not interrupt a run.
        """
        if not self.autosave:
            return

        path = self.workspace.save_config(
            additional_metadata={"loop_count": loop_count}
        )

        if path and self.verbose and loop_count is not None:
            logger.debug(
                f"Autosaved config at loop {loop_count} to {path}"
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
            f"For technical support, refer to this document: https://docs.swarms.world/community/technical-support"
        )

        raise error

    def _run_autonomous_loop(
        self,
        task: str,
        img: Optional[str] = None,
        streaming_callback: Optional[Callable[[str], None]] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        *args,
        **kwargs,
    ):
        """
        Run the plan-execute-summarize loop used when ``max_loops="auto"``.

        Delegates to :class:`~swarms.agents.autonomous_loop.AutonomousAgentLoop`,
        which owns the loop's planning, execution, and tool-dispatch logic.

        Args:
            task (str): The task for the agent to work through autonomously.
            img (Optional[str]): Optional image input for multimodal models.
            streaming_callback (Optional[Callable[[str], None]]): Callback
                receiving streaming tokens in real time.
            messages (Optional[List[Dict[str, Any]]]): Prior turns in chat
                format that the loop's transcript starts from.
            *args: Passed through to the loop.
            **kwargs: Passed through to the loop.

        Returns:
            The agent's final answer once it determines the task is complete.
        """
        return self.autonomous_loop._run_autonomous_loop(
            task=task,
            img=img,
            streaming_callback=streaming_callback,
            messages=messages,
            *args,
            **kwargs,
        )

    def _transcript_from_memory(self) -> Transcript:
        """
        Seed a structured transcript from ``short_memory``.

        Conversation roles are free-form strings ("User", the agent name,
        "Tool Executor", ...), so they are mapped onto chat roles here. The
        system prompt is skipped because the LLM wrapper supplies it. Turns
        added *during* a run are appended structurally on top of this prefix,
        which is what preserves tool-call fidelity where it matters most.
        """
        transcript = Transcript()
        for message in self.short_memory.conversation_history:
            if not isinstance(message, dict):
                continue
            role = message.get("role")
            content = message.get("content")
            if content is None or str(role).lower() == "system":
                continue
            if role == self.agent_name:
                transcript.append_assistant_text(content)
            else:
                transcript.append_user(content)
        return transcript

    def _memory_and_transcript(
        self, role: str, content: Any, transcript: Transcript
    ) -> None:
        """Record a turn in both ``short_memory`` and the live transcript."""
        self.short_memory.add(role=role, content=content)
        if role == self.agent_name:
            transcript.append_assistant_text(content)
        else:
            transcript.append_user(content)

    def _generate_final_summary(
        self,
        streaming_callback: Optional[Callable[[str], None]] = None,
        messages: Optional[List[dict]] = None,
    ) -> Any:
        """
        Generate a comprehensive final summary of the autonomous task execution.

        Args:
            streaming_callback: Optional callback receiving streaming tokens.
            messages: The autonomous loop's structured transcript. When given,
                the summary is requested against the real conversation - with
                its tool calls and tool results intact - rather than against a
                flattened string rendering of it.

        Returns:
            Any: The conversation shaped by ``output_type``, on every path.
        """
        summary_prompt = get_summary_prompt()
        self.short_memory.add(
            role=self.user_name, content=summary_prompt
        )

        try:
            if messages is not None:
                call_kwargs = {
                    "task": None,
                    "messages": messages
                    + [{"role": "user", "content": summary_prompt}],
                }
            else:
                call_kwargs = {
                    "task": self.short_memory.return_history_as_string()
                }

            response = self.call_llm(
                current_loop=0,
                streaming_callback=streaming_callback,
                **call_kwargs,
            )

            response = self.tool_manager.parse_llm_output(response)

            # Add LLM response to memory
            self.short_memory.add(
                role=self.agent_name, content=str(response)
            )

            if (
                self.tool_manager.handle_complete_task(response)
                is not None
            ):
                return history_output_formatter(
                    self.short_memory, type=self.output_type
                )

            # If complete_task wasn't called, generate summary manually
            comprehensive_summary = f"""Task Execution Summary

Original Task: {self.short_memory.conversation_history[0].get('content', 'N/A') if self.short_memory.conversation_history else 'N/A'}

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
            # Positional, in run()'s order: keywords plus *args made every extra positional collide with task.
            return await asyncio.to_thread(
                self.run,
                task,
                img,
                *args,
                **kwargs,
            )
        except Exception as error:
            # Not awaited: _handle_run_error is sync and always raises, as the other seven call sites assume.
            self._handle_run_error(error)

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

    def run_concurrent_tasks(self, tasks: List[str], *args, **kwargs):
        """
        Run multiple tasks concurrently.

        Args:
            tasks (List[str]): A list of tasks to run.

        Returns:
            List[Any]: One result per task, in the order the tasks were given.

        Raises:
            Exception: Whatever the underlying runs raise. Failures are logged
                and re-raised rather than swallowed, so a caller never receives
                None in place of results.
        """
        try:
            logger.info(f"Running concurrent tasks: {tasks}")
            # Call-scoped pool, as in heavy_swarm: no idle threads per agent.
            with ContextThreadPoolExecutor(
                max_workers=os.cpu_count()
            ) as executor:
                futures = [
                    executor.submit(
                        self.run, *args, task=task, **kwargs
                    )
                    for task in tasks
                ]
                results = [future.result() for future in futures]
            logger.info(f"Completed tasks: {results}")
            return results
        except Exception as error:
            logger.error(f"Error running concurrent tasks: {error}")
            raise

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

    def _default_context_length(self) -> int:
        """
        Returns the maximum input token window for the agent's underlying model.

        Attempts to determine the input (context) window based on the current model name by checking
        for the "max_input_tokens" property. If the value can't be determined (e.g., unknown model or
        missing field), defaults to 16000.

        Returns:
            int: The maximum number of input tokens for the model. Returns 16000 if undetermined.

        Notes:
            - Do NOT use ``get_max_tokens`` for context (input window). That reports max *output* tokens,
              which may not correspond to available context for input (e.g., 32768 for gpt-4.1 output, but
              over a million for certain input windows).
        """
        try:
            return (
                get_model_info(self.model_name).get(
                    "max_input_tokens"
                )
                or 16000
            )
        except Exception:
            return 16000

    def _default_max_tokens(self) -> int:
        """
        Returns the maximum output token count for the agent's underlying model.

        Determines the model's output window (number of output tokens) by checking for
        the "max_output_tokens" property in the model info. Returns 16000 as default if the
        property does not exist or can't be determined.

        Returns:
            int: The maximum number of output tokens for the model. Returns 16000 if undetermined.
        """
        # get_model_info raises for unmapped ids, which would otherwise take down __init__ for custom models.
        try:
            return (
                get_model_info(self.model_name).get(
                    "max_output_tokens"
                )
                or 16000
            )
        except Exception:
            return 16000

    def reliability_check(self):

        if self.system_prompt is None:
            logger.warning(
                "The system prompt is not set. Please set a system prompt for the agent to improve reliability."
            )

        if self.agent_name is None:
            logger.warning(
                "The agent name is not set. Please set an agent name to improve reliability."
            )

        if self.max_loops != "auto" and (
            not isinstance(self.max_loops, int) or self.max_loops <= 0
        ):
            raise AgentInitializationError(
                "max_loops must be a positive integer or 'auto', "
                f"got {self.max_loops!r}."
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

        # Truthiness, not "is not None": tools is normalised to [], so the None check never matched.
        if self.tools_list_dictionary:
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
                self._log_state_info(full_path, saved=True)

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
                self._log_state_info(resolved_path, saved=False)

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

            # Nothing to restore: concurrent work builds its own call-scoped pool.

        except Exception as e:
            logger.error(f"Error reinitializing components: {e}")
            raise

    def _log_state_info(self, file_path: str, *, saved: bool) -> None:
        """Log information about saved or loaded state for debugging."""
        try:
            state_dict = SafeLoaderUtils.create_state_dict(self)
            preserved = SafeLoaderUtils.preserve_instances(self)

            verb = "Saved" if saved else "Loaded"
            logger.info(
                f"{verb} agent state {'to' if saved else 'from'}: {file_path}"
            )
            logger.debug(
                f"{verb} {len(state_dict)} configuration values"
            )
            logger.debug(
                f"Preserved {len(preserved)} class instances"
            )

            if self.verbose:
                logger.debug(
                    "Preserved instances:"
                    if saved
                    else "Current class instances:"
                )
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
        return self.llm_manager.get_parameters()

    def update_system_prompt(self, system_prompt: str):
        """Upddate the system message"""
        self.system_prompt = system_prompt

    def update_max_loops(self, max_loops: Union[int, str]):
        """Update the max loops"""
        self.max_loops = max_loops

    def update_loop_interval(self, loop_interval: int):
        """Update the loop interval"""
        self.loop_interval = loop_interval

    def reset(self):
        """Reset the agent"""
        self.short_memory = None

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

    def list_tools(self) -> List[str]:
        """
        Names of every tool this agent can call.

        See :meth:`swarms.agents.tool_manager.ToolManager.list_tools`.

        Returns:
            List[str]: Tool names, without duplicates, in a stable order.
        """
        return self.tool_manager.list_tools()

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
        tokens_used = count_tokens(
            self.short_memory.return_history_as_string(),
            model=self.model_name,
        )

        limit = self.context_length - tokens_used

        if self.verbose:
            logger.info(
                f"Tokens available: {limit} You have {tokens_used} tokens used"
            )
        return limit

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

        # The llm object is not serializable
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

    def _stream_with_tool_collection(
        self, stream, tool_calls_out: list
    ):
        """Yield every chunk unchanged while assembling delta.tool_calls fragments.

        See :meth:`swarms.agents.llm_manager.LLMManager.stream_with_tool_collection`.
        """
        return self.llm_manager.stream_with_tool_collection(
            stream, tool_calls_out
        )

    def _extract_thinking_from_stream(self, stream):
        """Yield content chunks, flushing any reasoning chunks to a panel first.

        See :meth:`swarms.agents.llm_manager.LLMManager.extract_thinking_from_stream`.
        """
        return self.llm_manager.extract_thinking_from_stream(stream)

    def call_llm(
        self,
        task: str,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        current_loop: int = 0,
        streaming_callback: Optional[Callable[[str], None]] = None,
        *args,
        **kwargs,
    ) -> str:
        """
        Calls the LLM with the given task, handling streaming and multimodal inputs.

        Delegates to :meth:`swarms.agents.llm_manager.LLMManager.call`, which
        handles detailed streaming, panel streaming, silent streaming, and
        non-streaming calls, plus image input and tool-call collection.

        Args:
            task (str): The task or prompt to send to the LLM.
            img (Optional[str]): Optional image input for multimodal processing. Can be a
                file path, URL, data URI, or raw base64-encoded string.
            current_loop (int): The current loop iteration number, used for streaming
                panel titles and error logging context. Defaults to 0.
            streaming_callback (Optional[Callable[[str], None]]): Optional callback
                receiving streaming tokens in real time.
            *args: Additional positional arguments passed directly to llm.run().
            **kwargs: Additional keyword arguments passed directly to llm.run().

        Returns:
            str: The complete response from the LLM, or the assembled tool-call
                list when the model made tool calls mid-stream.

        Raises:
            AgentLLMError: If there's an issue with the language model.
            BadRequestError: If the request is malformed or invalid.
            InternalServerError: If the LLM service encounters an internal error.
            AuthenticationError: If authentication fails with the LLM service.

        Examples:
            >>> response = agent.call_llm("What is Python?", current_loop=1)
            >>> response = agent.call_llm("Describe this image", img="chart.png")
        """
        skills = self._skills_prompt.strip()
        if skills:
            if kwargs.get("messages") is not None:
                kwargs["messages"] = [
                    {"role": "system", "content": skills},
                    *kwargs["messages"],
                ]
            elif isinstance(task, str):
                # Not converted to messages: that path drops img and imgs.
                task = f"{skills}\n\n{task}"

        return self.llm_manager.call(
            task=task,
            img=img,
            imgs=imgs,
            current_loop=current_loop,
            streaming_callback=streaming_callback,
            *args,
            **kwargs,
        )

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
        self, skills_dir: str = None
    ) -> List[Dict[str, str]]:
        """
        Load skill metadata from SKILL.md files in the skills directory.

        Implements Tier 1 loading from Anthropic's Agent Skills framework:
        loads skill name and description into memory for context-aware activation.

        Args:
            skills_dir: Path to directory containing skill folders. Defaults to
                the agent's configured `skills_dir`.

        Returns:
            List of skill metadata dicts with 'name', 'description', 'path', 'content'

        Example:
            >>> agent = Agent(skills_dir="./skills")
            >>> # Loads all skills from ./skills/*/SKILL.md
        """
        return self.skills.load_metadata(skills_dir)

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
        return self.skills.load_full_skill(skill_name)

    @trace_run("Agent.run", input_params=("task", "img", "imgs"))
    def run(
        self,
        task: Optional[Union[str, Any]] = None,
        img: Optional[str] = None,
        imgs: Optional[List[str]] = None,
        correct_answer: Optional[str] = None,
        streaming_callback: Optional[Callable[[str], None]] = None,
        n: int = 1,
        messages: Optional[List[Dict[str, Any]]] = None,
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
            img (Optional[str]): Path, URL, data URI, or raw base64-encoded string for a single image input.
                Supported formats: file paths (e.g., "image.jpg"), URLs (e.g., "https://example.com/image.png"),
                data URIs (e.g., "data:image/jpeg;base64,..."), or raw base64 strings. Defaults to None.
            imgs (Optional[List[str]]): List of multiple images if processing a batch. Each image can be a path,
                URL, data URI, or raw base64 string. Defaults to None.
            correct_answer (Optional[str]): Ground truth answer for evaluation comparisons. Defaults to None.
            streaming_callback (Optional[Callable[[str], None]]): Function to receive streamed tokens as output is generated (real-time). If not given, uses self.streaming_callback if available. Defaults to None.
            n (int): How many outputs to generate (number of runs). Defaults to 1.
            messages (Optional[List[Dict[str, Any]]]): Prior turns in chat format,
                e.g. ``[{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]``.
                They are added to the agent's conversation and sent as the context
                this task continues from. Defaults to the agent's own ``messages``.
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
            >>> # Using base64-encoded image
            >>> import base64
            >>> with open("image.jpg", "rb") as f:
            ...     img_base64 = base64.b64encode(f.read()).decode("utf-8")
            >>> agent.run("Describe this image", img=img_base64)
            >>> agent.run(
            ...     "And what did I just ask you?",
            ...     messages=[{"role": "user", "content": "Name three primes."}],
            ... )
        """

        # Outside interactive mode, fail fast instead of blocking on stdin
        if task is None or (
            isinstance(task, str) and task.strip() == ""
        ):
            if not self.interactive:
                raise ValueError(
                    "No task provided. Pass a non-empty `task`, or set "
                    "interactive=True to be prompted for one."
                )
            # Always show prompt when asking for initial task, even if print_on is False
            self.pretty_print(
                "Interactive mode enabled. Please enter your initial task:",
                loop_count=0,
            )
            formatter.console.print()
            try:
                task = formatter.console.input(
                    "[bold cyan]You[/bold cyan] [bold green]❯[/bold green] "
                ).strip()
            except (KeyboardInterrupt, EOFError):
                # Ctrl+C / Ctrl+D before the first task exits without a traceback
                formatter.console.print()
                self.pretty_print(
                    "Session ended by user. Goodbye.",
                    loop_count=0,
                )
                return None

            if not task:
                raise ValueError(
                    "No task provided. Exiting interactive mode."
                )

        if exists(self.skills_dir):
            self.handle_skills(task=task)

        if not isinstance(task, str):
            task = format_data_structure(task)

        if streaming_callback is None:
            if self.streaming_callback is not None:
                streaming_callback = self.streaming_callback

        # Constructor messages are already in short_memory; per-call ones are not.
        if messages:
            self.short_memory.add_messages(messages)

        self.mcp_manager.begin_run()
        try:
            if self.max_loops == "auto":
                # Use autonomous loop structure: plan -> execute subtasks -> summary
                output = self._run_autonomous_loop(
                    task=task,
                    img=img,
                    streaming_callback=streaming_callback,
                    messages=messages,
                    *args,
                    **kwargs,
                )
            elif n > 1:
                output = [
                    self._run(
                        task=task,
                        img=img,
                        imgs=imgs,
                        streaming_callback=streaming_callback,
                        messages=messages,
                        *args,
                        **kwargs,
                    )
                    for _ in range(n)
                ]
            else:
                output = self._run(
                    task=task,
                    img=img,
                    imgs=imgs,
                    streaming_callback=streaming_callback,
                    messages=messages,
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
                "For technical support, refer to this document: https://docs.swarms.world/community/technical-support"
            )
            raise KeyboardInterrupt

        finally:
            self.mcp_manager.end_run()

    def run_stream(
        self,
        task: str,
        img: Optional[str] = None,
        **kwargs,
    ):
        """Run the agent and yield response tokens one-by-one as they are generated.

        The full auto-loop (multi-step reasoning, tool calls, MCP, etc.) runs in a
        background daemon thread.  Each token the model emits is put onto a queue and
        yielded to the caller immediately, so the first characters appear on screen
        before the model has finished generating.

        Tool-call results are fed back into the loop automatically (same as a normal
        run); the tokens from each subsequent LLM turn are also streamed through.

        Args:
            task: The prompt / task string.
            img:  Optional image path or base64 string for vision models.
            **kwargs: Any extra kwargs forwarded to run().

        Yields:
            str: Individual token strings in generation order.

        Example::

            for token in agent.run_stream("Analyse NVDA"):
                print(token, end="", flush=True)
        """
        import queue

        token_queue: queue.Queue = queue.Queue()
        _DONE = object()
        _exc: list = [None]

        def _on_token(token):
            if isinstance(token, str) and token:
                token_queue.put(token)
            elif isinstance(token, dict):
                t = token.get("token", "")
                if t:
                    token_queue.put(t)

        original_streaming_on = self.streaming_on
        self.streaming_on = True

        def _run_thread():
            try:
                self.run(
                    task=task,
                    img=img,
                    streaming_callback=_on_token,
                    **kwargs,
                )
            except Exception as exc:
                _exc[0] = exc
            finally:
                self.streaming_on = original_streaming_on
                token_queue.put(_DONE)

        thread = threading.Thread(target=_run_thread, daemon=True)
        thread.start()

        while True:
            item = token_queue.get()
            if item is _DONE:
                break
            yield item

        thread.join()

        if _exc[0] is not None:
            raise _exc[0]

    async def arun_stream(
        self,
        task: str,
        img: Optional[str] = None,
        **kwargs,
    ):
        """Async generator version of run_stream — yields tokens as they arrive.

        The agent loop runs in a thread-pool executor so it does not block the
        event loop.  Each token is forwarded to an asyncio.Queue and yielded to
        the async caller immediately.

        Args:
            task: The prompt / task string.
            img:  Optional image path or base64 string for vision models.
            **kwargs: Extra kwargs forwarded to run().

        Yields:
            str: Individual token strings in generation order.

        Example::

            async for token in agent.arun_stream("Analyse NVDA"):
                print(token, end="", flush=True)
        """
        import asyncio

        loop = asyncio.get_running_loop()
        token_queue: asyncio.Queue = asyncio.Queue()
        _DONE = object()
        _exc: list = [None]

        def _on_token(token):
            if isinstance(token, str) and token:
                loop.call_soon_threadsafe(
                    token_queue.put_nowait, token
                )
            elif isinstance(token, dict):
                t = token.get("token", "")
                if t:
                    loop.call_soon_threadsafe(
                        token_queue.put_nowait, t
                    )

        original_streaming_on = self.streaming_on
        self.streaming_on = True

        def _run_sync():
            try:
                self.run(
                    task=task,
                    img=img,
                    streaming_callback=_on_token,
                    **kwargs,
                )
            except Exception as exc:
                _exc[0] = exc
            finally:
                self.streaming_on = original_streaming_on
                loop.call_soon_threadsafe(
                    token_queue.put_nowait, _DONE
                )

        thread = threading.Thread(target=_run_sync, daemon=True)
        thread.start()

        while True:
            item = await token_queue.get()
            if item is _DONE:
                break
            yield item

        if _exc[0] is not None:
            raise _exc[0]

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

        Delegates to :meth:`swarms.agents.llm_manager.LLMManager.handle_fallback_execution`,
        which walks the fallback chain until the task succeeds or every model is
        exhausted.

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
        """
        return self.llm_manager.handle_fallback_execution(
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
        Run a batch of tasks, one after another.

        Args:
            tasks (List[str]): List of tasks to run.
            imgs (List[str], optional): One image per task, paired by position.
                Omit to run the tasks without images. Defaults to None.
            *args: Additional positional arguments to be passed to the execution method.
            **kwargs: Additional keyword arguments to be passed to the execution method.

        Returns:
            List[Any]: List of results from each task execution.
        """
        # Index imgs rather than zip: zip rebound imgs and raised when it was None.
        if imgs is None:
            return [
                self.run(task=task, *args, **kwargs) for task in tasks
            ]

        if len(imgs) != len(tasks):
            raise ValueError(
                f"run_batched got {len(tasks)} tasks and {len(imgs)} images; "
                "pass one image per task, or omit imgs entirely. Zipping them "
                "would silently drop the extras."
            )

        return [
            self.run(task=task, img=img, *args, **kwargs)
            for task, img in zip(tasks, imgs)
        ]

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

        Args:
            agents (List[Union[Any, Callable]]): The agents to talk to.
            task (str): The message to send to each agent.

        Returns:
            List[Any]: One entry per agent, in the order the agents were given.
                An agent whose conversation raised contributes None.
        """
        # Scoped to the call, see run_concurrent_tasks for why
        with ContextThreadPoolExecutor(
            max_workers=os.cpu_count()
        ) as executor:
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
                f"Agent Name {self.agent_name} [Loop: {loop_count}/{self.max_loops}]",
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

    @property
    def input_tokens(self) -> int:
        """Tokens the agent's next request would carry, counted with its model's tokenizer.

        Covers everything the agent sends as input: the system prompt, the
        whole conversation in ``short_memory``, and the tool schemas. Use it
        to see how full the context window is before a run. It is an
        estimate — the conversation is counted as rendered text, role labels
        included — so it runs a little above what the provider bills. For the
        billed figure, summed over past calls, see :attr:`usage`.
        """
        parts = [self.short_memory.return_history_as_string()]
        if self.tools_list_dictionary:
            parts.append(json.dumps(self.tools_list_dictionary))
        return count_tokens(
            "\n".join(part for part in parts if part),
            model=self.model_name,
        )

    @property
    def usage(self) -> dict:
        """Token usage reported by the provider, summed over every LLM call this agent has made.

        Keys: ``input_tokens``, ``output_tokens``, ``cached_tokens`` (the
        part of ``input_tokens`` served from the provider's prompt cache),
        ``reasoning_tokens`` (the part of ``output_tokens`` the model spent
        thinking, 0 when the provider does not report it), ``total_tokens``.
        Streaming calls count once their stream has been consumed, since the
        provider reports usage in the final chunk.
        """
        return dict(self._usage)

    def _add_usage(self, call_usage: dict) -> None:
        """Fold one completion's usage into the running total."""
        for key in self._usage:
            self._usage[key] += call_usage.get(key, 0)

    def get_available_models(self) -> List[str]:
        """
        Get the list of available models including primary and fallback models.

        Returns:
            List[str]: List of model names in order of preference
        """
        return self.llm_manager.get_available_models()

    def get_current_model(self) -> str:
        """
        Get the current model being used.

        Returns:
            str: Current model name
        """
        return self.llm_manager.get_current_model()

    def switch_to_next_model(self) -> bool:
        """
        Switch to the next available model in the fallback list.

        Returns:
            bool: True if successfully switched to next model, False if no more models available
        """
        return self.llm_manager.switch_to_next_model()

    def reset_model_index(self) -> None:
        """Reset the model index to use the primary model."""
        self.llm_manager.reset_model_index()

    def is_fallback_available(self) -> bool:
        """
        Check if fallback models are available.

        Returns:
            bool: True if fallback models are configured
        """
        return self.llm_manager.is_fallback_available()

    def list_output_types(self):
        return OutputType

    def _transcript_from_messages(
        self,
        messages: Optional[List[Dict[str, Any]]],
        task: Optional[Any],
    ) -> Transcript:
        """
        Build this run's transcript, preferring caller-supplied turns.

        Args:
            messages: Prior conversation as typed chat messages. When given,
                these replace the memory-derived prefix and ``task`` is
                appended as the new user turn. ``None`` falls back to
                :meth:`_transcript_from_memory`.
            task: The instruction for this turn.

        Returns:
            The transcript to send with the next request.
        """
        if messages is None:
            return self._transcript_from_memory()

        transcript = Transcript(list(messages))
        if task is not None:
            transcript.append_user(task)
        return transcript
