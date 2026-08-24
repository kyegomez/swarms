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
from swarms.agents.context_compressor import ContextCompressor
from swarms.agents.autonomous_loop import AutonomousAgentLoop
from swarms.agents.llm_manager import LLMManager
from swarms.agents.skills_manager import SkillsManager
from swarms.prompts.agent_system_prompts import AGENT_SYSTEM_PROMPT_3
from swarms.prompts.autonomous_agent_prompt import (
    get_autonomous_agent_prompt,
)
from swarms.prompts.handoffs_prompt import get_handoffs_prompt
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
    get_autonomous_loop_tool_names,
    get_summary_prompt,
)
from swarms.structs.conversation import Conversation
from swarms.structs.ma_utils import set_random_models_for_agents
from swarms.structs.transcript import Transcript
from swarms.tools.dynamic_tool_loader import (
    DYNAMIC_TOOLS_NOTICE,
    SEARCH_TOOL_NAME,
    DynamicToolLoader,
)
from swarms.structs.safe_loading import (
    SafeLoaderUtils,
    SafeStateManager,
)
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
from swarms.tools.base_tool import BaseTool
from swarms.tools.handoffs_tool import handoff_task
from swarms.tools.handoffs_tool_schema import get_handoff_tool_schema
from swarms.tools.mcp_manager import MCPManager
from swarms.tools.py_func_to_openai_func_str import (
    convert_multiple_functions_to_openai_function_schema,
)
from swarms.utils.file_processing import create_file_in_folder
from swarms.utils.formatter import formatter
from swarms.utils.generate_id import generate_id
from swarms.utils.generate_keys import generate_api_key
from swarms.utils.get_reasoning_efforts import get_reasoning_efforts
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
from swarms.utils.workspace_manager import WorkspaceManager
from swarms.utils.workspace_utils import get_workspace_dir


def stop_when_repeats(response: str) -> bool:
    # Stop if the word stop appears in the response
    return "stop" in response.lower()


# Parse done token
def parse_done_token(response: str) -> bool:
    """Parse the response to see if the done token is present"""
    return "<DONE>" in response


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
        output_type (agent_output_type): The output type. Supported: 'str', 'string', 'list', 'json', 'dict', 'yaml', 'xml'.
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
        system_prompt: Optional[str] = AGENT_SYSTEM_PROMPT_3,
        llm: Optional[Any] = None,
        max_loops: Optional[Union[int, str]] = 1,
        stopping_condition: Optional[Callable[[str], bool]] = None,
        loop_interval: Optional[int] = 0,
        retry_attempts: Optional[int] = 3,
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
        tool_schema: ToolUsageType = None,
        output_type: OutputType = "str-all-except-first",
        output_cleaner: Optional[Callable] = None,
        list_base_models: Optional[List[BaseModel]] = None,
        rules: str = None,  # type: ignore
        planning_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: float = 0.5,
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
        tool_call_summary: bool = True,
        tool_retry_attempts: int = 3,
        reasoning_prompt_on: bool = True,
        dynamic_context_window: bool = True,
        show_tool_execution_output: bool = True,
        reasoning_effort: Literal[get_reasoning_efforts()] = "medium",
        thinking_tokens: int = 1024,
        think_tool: bool = False,
        dynamic_tools: bool = True,
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
        *args,
        **kwargs,
    ):
        # super().__init__(*args, **kwargs)
        self.id = id or generate_id("agent")
        self.skills = SkillsManager(skills_dir=skills_dir)
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
        self.saved_state_path = saved_state_path
        self.dynamic_temperature_enabled = dynamic_temperature_enabled
        self.dynamic_loops = dynamic_loops
        self.user_name = user_name
        self.context_length = context_length
        self.sop = sop
        self.sop_list = sop_list
        self.tools = tools
        self.system_prompt = system_prompt or ""
        self.agent_name = agent_name
        self.agent_description = agent_description
        # self.saved_state_path = f"{self.agent_name}_{generate_api_key(prefix='agent-')}_state.json"
        self.saved_state_path = (
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
        # Always use environment variable for workspace_dir, ignore user input
        # Fallback to default if environment variable is not set
        self.workspace_dir = get_workspace_dir()
        # Built on first use: file tools need the dir even without
        # autosave, but constructing every agent must not create one.
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
        # Own list per agent. The default used to be a literal [], so every
        # agent that skipped this argument shared one object and inherited
        # the tool schemas any other agent appended to it.
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

        # Defer tool schemas and let the agent search for what it needs.
        self.dynamic_tools = dynamic_tools
        self.tool_loader: Optional[DynamicToolLoader] = None
        # MCP schemas are fetched over the network, so they are folded into
        # the catalog once rather than on every LLM rebuild.
        self._mcp_tools_deferred = False
        # Fetched MCP schemas, kept so a rebuilt loader can be repopulated
        # without another network call. The autonomous loop constructs a fresh
        # loader per run, so a boolean "already deferred" flag is not enough.
        self._mcp_schemas_cache: Optional[List[dict]] = None

        # Whether the autonomous loop offers the `think` tool. Off by default:
        # a think call costs a full round-trip to produce reasoning the model
        # could have emitted alongside its actions in the same response. Turn
        # it on for models that do not reason natively, or when an explicit
        # analysis step is worth the extra turn.
        self.think_tool = think_tool
        self.reasoning_enabled = reasoning_enabled
        self.fallback_model_name = fallback_model_name
        self.handoffs = handoffs
        self.capabilities = capabilities
        self.mode = mode
        self.publish_to_marketplace = publish_to_marketplace
        self.marketplace_prompt_id = marketplace_prompt_id

        # All MCP (Model Context Protocol) behaviour — connection handling,
        # API key / bearer token / OAuth authentication, transport selection,
        # tool discovery and tool execution — lives in MCPManager.
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
            # The prompt must agree with the tool list: without this the model
            # is instructed to call a `think` tool it was never given.
            self.system_prompt += (
                "\n\n"
                + get_autonomous_agent_prompt(
                    include_think_tool=self.think_tool
                )
            )

        # When False the agent does not read or write MEMORY.md across sessions.
        self.persistent_memory = persistent_memory

        # Context compression is available for both max_loops="auto" and
        # integer max_loops runs. Gated purely on the user-facing boolean.
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

        # Owns model rotation, LiteLLM construction, and LLM invocation.
        # Reads config off this agent, so it must be built after config is set.
        self.llm_manager = LLMManager(agent=self)
        self.autonomous_loop = AutonomousAgentLoop(agent=self)

        # self.init_handling()
        self.setup_config()

        # Initialize the short memory. The Conversation manages the
        # persistent MEMORY.md file at
        # $WORKSPACE_DIR/agents/{agent_name}-{id}/MEMORY.md.
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

        # One condition so the notice cannot diverge from the loader.
        defers_tools = self.dynamic_tools and (
            exists(self.tools)
            or self.mcp_enabled
            or self.max_loops == "auto"
        )

        # Appended once here, not per run.
        if defers_tools:
            self.system_prompt += DYNAMIC_TOOLS_NOTICE
            self.setup_dynamic_tools()
        elif exists(self.tools):
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
        Load Agent Skills into the system prompt.

        Args:
            task: Optional task description. If provided, loads skills dynamically
                  based on similarity to the task. If not provided, loads all skills statically.
        """
        self.system_prompt += self.skills.prompt_for_task(task)

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
        prompt = self.system_prompt

        if self.safety_prompt_on is True:
            prompt += "\n\n"
            prompt += SAFETY_PROMPT

        # Compute the persistent MEMORY.md path under the workspace dir.
        # Only resolved when persistent_memory=True (the default). When False
        # the Conversation receives no path and operates as a pure in-process
        # store with no on-disk read or write across sessions.
        # Key on agent_name only (not self.id) so memory is stable across
        # process restarts — self.id defaults to a fresh uuid every run,
        # which would otherwise create a new empty MEMORY.md each time.
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

    def add_mcp_tools_to_memory(self) -> List[Dict[str, Any]]:
        """
        Fetch the tool schemas exposed by the configured MCP servers.

        Connection handling, authentication (API key, bearer token, or OAuth)
        and transport selection are all delegated to :class:`MCPManager`. The
        returned schemas are OpenAI function-calling definitions, ready to be
        passed straight to the LLM.

        Returns:
            List[Dict[str, Any]]: OpenAI tool schemas from every MCP server.

        Raises:
            AgentMCPConnectionError: If no server could be reached.
        """
        try:
            tools = self.mcp_manager.get_tools()

            if self.print_on:
                self.pretty_print(
                    f"✨ [SYSTEM] Successfully integrated {len(tools)} MCP tools into agent: {self.agent_name} | Status: ONLINE | Time: {time.strftime('%H:%M:%S')} ✨",
                    loop_count=0,
                )

            return tools
        except Exception as e:
            logger.error(
                f"Error Adding MCP Tools to Agent: {self.agent_name} Error: {e} Traceback: {traceback.format_exc()}"
            )
            raise e

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

            if self.plan_enabled is True:
                self.plan(task)

            # Set the loop count
            loop_count = 0

            # Structured conversation for this run. Built lazily below so the
            # transforms path can keep its flattened prompt.
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

                # Compress short-term memory if an auto-loop run has
                # crossed the configured fraction of the context window.
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
                #
                # `transforms` rewrites the whole history into a single string
                # by design, so that path keeps the legacy flattened prompt.
                # Everything else sends a real message list, which preserves
                # assistant `tool_calls` turns and `tool` results instead of
                # collapsing them into prose.
                task_prompt = None
                use_transcript = self.transforms is None

                if self.transforms is not None:
                    task_prompt = handle_transforms(
                        transforms=self.transforms,
                        short_memory=self.short_memory,
                        model_name=self.model_name,
                    )
                elif transcript is None:
                    transcript = self._transcript_from_memory()

                # Parameters
                attempt = 0
                success = False
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

                        # Record the model's turn. Any tool calls it made must
                        # each receive a matching tool result before the next
                        # request, which `flush_tool_results` guarantees below.
                        if use_transcript:
                            turn_calls = transcript.record_assistant(
                                response
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

                        # Handle tool_search calls. It is dispatched here
                        # rather than through tool_struct because it is an
                        # agent method, not one of the user's callables.
                        if (
                            isinstance(response, list)
                            and self.tool_loader
                        ):
                            remaining = []
                            for tool_call in response:
                                name = (
                                    tool_call.get("function", {}).get(
                                        "name"
                                    )
                                    if isinstance(tool_call, dict)
                                    else None
                                )
                                if name != SEARCH_TOOL_NAME:
                                    remaining.append(tool_call)
                                    continue

                                try:
                                    arguments = json.loads(
                                        tool_call["function"][
                                            "arguments"
                                        ]
                                    )
                                except (
                                    json.JSONDecodeError,
                                    TypeError,
                                ):
                                    arguments = {}

                                result = self._tool_search_tool(
                                    **arguments
                                )
                                self.short_memory.add(
                                    role="Tool Executor",
                                    content=f"tool_search result: {result}",
                                )
                                turn_results[
                                    tool_call.get("id", "")
                                ] = result
                                if self.print_on:
                                    formatter.print_panel(
                                        result, title="Tool Search"
                                    )

                            # Anything left goes on to normal execution. If
                            # tool_search was the whole response there is
                            # nothing to execute, and falling through would log
                            # misleading "no function calls found" warnings.
                            response = remaining
                            if not remaining:
                                if use_transcript and turn_calls:
                                    transcript.flush_tool_results(
                                        turn_calls, turn_results
                                    )
                                success = True
                                continue

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
                                    turn_results[
                                        tool_call.get("id", "")
                                    ] = result
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
                            tool_output = self.tool_execution_retry(
                                response, loop_count
                            )
                            if use_transcript and turn_calls:
                                transcript.map_batch_results(
                                    [
                                        {"id": c["id"]}
                                        for c in turn_calls
                                    ],
                                    tool_output,
                                    turn_results,
                                    formatter=format_data_structure,
                                )

                        # Handle MCP tools
                        if self.mcp_enabled:
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

                        # Answer every tool call in the assistant turn just
                        # recorded. A gap here makes the *next* request invalid,
                        # so this runs on every path that reached this point.
                        if use_transcript and turn_calls:
                            transcript.flush_tool_results(
                                turn_calls, turn_results
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

                        # Close out any tool calls recorded before the failure,
                        # so the retried request is still well formed.
                        if use_transcript and turn_calls:
                            transcript.flush_tool_results(
                                turn_calls, turn_results
                            )

                        # Track the LLM/generation error via telemetry — the
                        # retry loop swallows it, so capture_run never sees it.
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
                    formatter.console.print()
                    try:
                        user_input = formatter.console.input(
                            "[bold cyan]You[/bold cyan] [bold green]❯[/bold green] "
                        )
                    except (KeyboardInterrupt, EOFError):
                        # Graceful exit on Ctrl+C / Ctrl+D during
                        # interactive input. No traceback, no error.
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

    def get_all_selected_tools(self) -> List[str]:
        """
        Return a list of all autonomous loop tool names.

        Returns:
            List of tool name strings (e.g. ["create_plan", "think", "subtask_done", ...])
        """
        return get_autonomous_loop_tool_names()

    def _run_autonomous_loop(
        self,
        task: str,
        img: Optional[str] = None,
        streaming_callback: Optional[Callable[[str], None]] = None,
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
            *args: Passed through to the loop.
            **kwargs: Passed through to the loop.

        Returns:
            The agent's final answer once it determines the task is complete.
        """
        return self.autonomous_loop._run_autonomous_loop(
            task=task,
            img=img,
            streaming_callback=streaming_callback,
            *args,
            **kwargs,
        )

    def setup_dynamic_tools(
        self, always_loaded: Optional[List[dict]] = None
    ) -> DynamicToolLoader:
        """
        Defer this agent's tool schemas behind a ``tool_search`` tool.

        Tool definitions are re-sent on every request and sit in the cached
        prefix, so a large tool set is paid for continuously. Deferring sends
        only ``tool_search`` up front; the agent loads what it needs, and the
        loaded schemas are included from the next request onwards.

        Args:
            always_loaded: Schemas that must never be deferred. Control-flow
                tools belong here - an agent that has to search for its own
                ``complete_task`` cannot finish.

        Returns:
            The loader, also stored on ``self.tool_loader``.
        """
        # Anything already registered - handoff tools, MCP tools - was
        # configured explicitly and must survive. Assigning schemas() straight
        # over tools_list_dictionary would silently drop it, which is how
        # handoffs stopped working when dynamic_tools was first added.
        user_tool_names = {
            schema.get("function", {}).get("name")
            for schema in convert_multiple_functions_to_openai_function_schema(
                list(self.tools or [])
            )
        }
        # The search tool is excluded: schemas() re-adds it, and setup runs
        # twice for an autonomous agent (once in __init__, once in the loop),
        # so keeping it here would list it twice in the tool array.
        preserved = [
            schema
            for schema in (self.tools_list_dictionary or [])
            if isinstance(schema, dict)
            and schema.get("function", {}).get("name")
            not in user_tool_names | {SEARCH_TOOL_NAME}
        ]

        keep: List[dict] = []
        seen: set = set()
        for schema in list(always_loaded or []) + preserved:
            name = schema.get("function", {}).get("name")
            if name and name not in seen:
                seen.add(name)
                keep.append(schema)

        self.tool_loader = DynamicToolLoader(
            tools=self.tools or [],
            always_loaded=keep,
        )

        # A rebuilt loader is empty; the fetch guard will not refetch.
        for schema in self._mcp_schemas_cache or []:
            self.tool_loader.register_schema(schema)

        self.tools_list_dictionary = self.tool_loader.schemas()
        return self.tool_loader

    def defer_tool_schemas(self, schemas: List[dict]) -> None:
        """Add pre-built schemas to the deferred catalog (MCP, loop tools)."""
        if self.tool_loader is None:
            return
        for schema in schemas:
            self.tool_loader.register_schema(schema)
        self.tools_list_dictionary = self.tool_loader.schemas()

    def defer_mcp_tools(self) -> int:
        """
        Move this agent's MCP tool schemas into the deferred catalog.

        MCP is the case dynamic tools exist for: a single server can expose
        dozens of tools, and every one of them is otherwise re-sent with every
        request. Deferring them makes them searchable instead, so the request
        carries only what the agent actually loaded.

        The fetch is a network call, and rebuilding the LLM after each
        ``tool_search`` would repeat it, so it runs once per agent.

        Returns:
            int: How many schemas were added to the catalog. 0 if MCP is not
            configured, dynamic tools are off, or this already ran.
        """
        if self.tool_loader is None or not self.mcp_enabled:
            return 0

        # Already registered in the *current* loader - nothing to do. Keyed on
        # the loader's contents rather than a boolean, because the autonomous
        # loop builds a fresh loader per run and a flag would stop the rebuilt
        # one from ever being repopulated.
        cached = self._mcp_schemas_cache
        if cached is not None and all(
            schema.get("function", {}).get("name") in self.tool_loader
            for schema in cached
        ):
            return 0

        if cached is None:
            try:
                cached = self.add_mcp_tools_to_memory()
            except Exception as error:
                # A server being unreachable must not take down agent setup;
                # the agent simply runs without those tools.
                logger.error(
                    f"Could not fetch MCP tools to defer: {error}"
                )
                self._mcp_schemas_cache = []
                self._mcp_tools_deferred = True
                return 0
            self._mcp_schemas_cache = cached

        schemas = cached
        self._mcp_tools_deferred = True
        self.defer_tool_schemas(schemas)

        if self.verbose:
            logger.info(
                f"Deferred {len(schemas)} MCP tool(s) into the catalog: "
                f"{[s.get('function', {}).get('name') for s in schemas]}"
            )
        return len(schemas)

    def _tool_search_tool(
        self,
        query: str,
        max_results: int = 5,
        min_score_ratio: float = 0.0,
        **kwargs,
    ) -> str:
        """
        Handler for the ``tool_search`` tool.

        Loading changes the tool list, so the LLM is rebuilt here - otherwise
        the newly loaded schemas would not be sent and the model could not
        call what it just found.
        """
        if self.tool_loader is None:
            return (
                "Tool search is unavailable: this agent was not built with "
                "dynamic_tools=True."
            )

        result = self.tool_loader.run_search(
            query=query,
            max_results=max_results,
            min_score_ratio=min_score_ratio,
        )
        self.tools_list_dictionary = self.tool_loader.schemas()
        if self.llm is not None:
            self.llm = self.llm_handling()

        if self.verbose:
            logger.info(
                f"tool_search({query!r}) -> loaded "
                f"{self.tool_loader.loaded_names}"
            )
        return result

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
    ) -> str:
        """
        Generate a comprehensive final summary of the autonomous task execution.

        Args:
            streaming_callback: Optional callback receiving streaming tokens.
            messages: The autonomous loop's structured transcript. When given,
                the summary is requested against the real conversation - with
                its tool calls and tool results intact - rather than against a
                flattened string rendering of it.

        Returns:
            str: Comprehensive summary
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
            # Forward positionally, in run()'s own parameter order. Passing
            # task/img as keywords while also splatting *args made every
            # extra positional collide with `task`:
            #   TypeError: run() got multiple values for argument 'task'
            # so arun(task, img, extra) raised instead of running.
            return await asyncio.to_thread(
                self.run,
                task,
                img,
                *args,
                **kwargs,
            )
        except Exception as error:
            # _handle_run_error is sync and re-raises; the other seven call
            # sites do not await it. The await was only harmless because the
            # method always raises before returning — the day it stops, this
            # becomes `await None`.
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
        # get_model_info raises for an unmapped model id rather than
        # returning an empty mapping, so an unknown, custom or self-hosted
        # model would otherwise take down Agent.__init__. The sibling
        # _default_context_length guards the same call the same way.
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

        # Truthiness, not "is not None": the attribute is normalised to an
        # empty list for every agent built without tools, so the None check
        # warned about function calling for agents that never use it.
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
        """

        # If no task is provided, prompt for one only in interactive mode.
        # Outside interactive mode, fail fast instead of blocking on stdin.
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
                # Graceful exit on Ctrl+C / Ctrl+D before the first task
                # has even been entered. No traceback, no error.
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
            elif n > 1:
                output = [self.run(task=task) for _ in range(n)]
            else:
                output = self._run(
                    task=task,
                    img=img,
                    imgs=imgs,
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
                "For technical support, refer to this document: https://docs.swarms.world/community/technical-support"
            )
            raise KeyboardInterrupt

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
            **kwargs: Any extra kwargs forwarded to _run().

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
                self._run(
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
            **kwargs: Extra kwargs forwarded to _run().

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
                self._run(
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
        # Pool is scoped to the call — see run_concurrent_tasks for why this is
        # not an Agent-level executor.
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

                # A lone tool call. With MCP enabled the wrapper returns a
                # bare dict for one call and a list for several, so callers
                # that test `isinstance(response, list)` silently ignored
                # single calls - which broke planning outright whenever MCP
                # was configured. Normalise to the list form.
                if "function" in response:
                    return [response]

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

        except Exception as e:
            logger.error(f"Error parsing LLM output: {e}")
            raise ValueError(
                f"Failed to parse LLM output: {type(response)}"
            ) from e

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
        Execute the MCP tool calls contained in an LLM response.

        All of the MCP mechanics — routing each tool call to the server that
        owns it, authenticating (API key, bearer token, or OAuth), opening the
        session and shaping the result — are handled by
        :class:`swarms.tools.mcp_manager.MCPManager`. This method only wires
        the result back into the agent's conversation.

        **Post-Execution Processing:**
        1. Formats the tool results as JSON
        2. Adds them to conversation memory under the "Tool Executor" role
        3. Generates a natural-language summary with a tool-free LLM instance
        4. Displays the summary if ``print_on=True``

        Args:
            response (any): The LLM response containing MCP tool calls. Can be
                a list of tool calls, a single tool call, a full assistant
                message, or a JSON string of any of those.
            current_loop (Optional[int]): The current loop iteration number,
                used for logging and progress display. Defaults to 0.

        Returns:
            None: Modifies internal state (memory, printed output) only.

        Raises:
            AgentMCPConnectionError: If no MCP server could be reached.
            AgentMCPToolError: If tool execution fails outright.

        Examples:
            >>> # Single MCP server secured with an API key
            >>> agent = Agent(mcp_url="https://api.example.com/mcp", mcp_api_key="sk-...")
            >>> response = [{"function": {"name": "mcp_tool", "arguments": "{}"}}]
            >>> agent.mcp_tool_handling(response, current_loop=1)

            >>> # Multiple MCP servers
            >>> agent = Agent(mcp_urls=["https://a/mcp", "https://b/mcp"])
            >>> agent.mcp_tool_handling(response, current_loop=2)
        """
        try:
            tool_response = self.mcp_manager.execute_tool_calls(
                response, output_type="dict"
            )

            if not tool_response:
                if self.verbose:
                    logger.info(
                        f"No MCP tool calls found in the response for {self.agent_name}"
                    )
                return

            text_content = f"MCP Tool Response: \n\n {json.dumps(tool_response, indent=2, default=str)}"

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
        except Exception as e:
            logger.error(
                f"Error in MCP tool handling for {self.agent_name}: {e} Traceback: {traceback.format_exc()}"
            )
            raise e

    def temp_llm_instance_for_tool_summary(self):
        return LiteLLM(
            model_name=self.model_name,
            temperature=self.temperature,
            top_p=self.top_p,  # Anthropic rejects requests with both temperature and top_p
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

        # Stored so a transcript builder can map it to tool_call ids.
        self._last_tool_output = output

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
        - After all retries are exhausted, AgentToolExecutionError is raised,
          chained from the last underlying error via `raise ... from`

        **Error Handling:**
        - None responses: Logs warning and skips execution (does not raise)
        - Any exception from execute_tools: Logs error with full traceback and
          retries, since execute_tools re-raises the tool's own exception type

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
        if response is None:
            logger.warning(
                f"Agent '{self.agent_name}' received None response from LLM in loop {loop_count}. "
                f"This may indicate an issue with the model or prompt. Skipping tool execution."
            )
            return

        # Catch broadly: nothing raises AgentToolExecutionError, so that caught nothing.
        attempts = max(1, int(self.tool_retry_attempts or 1))
        last_error: Optional[Exception] = None

        for attempt in range(1, attempts + 1):
            try:
                self.execute_tools(
                    response=response,
                    loop_count=loop_count,
                )
                return getattr(self, "_last_tool_output", None)
            except Exception as e:
                last_error = e
                logger.error(
                    f"Agent '{self.agent_name}' tool execution failed on attempt "
                    f"{attempt}/{attempts} in loop {loop_count}: {str(e)}. "
                    f"Full traceback: {traceback.format_exc()}"
                )

        # Attempts exhausted: raise, or the model reads a silent no-op as success.
        raise AgentToolExecutionError(
            f"Agent '{self.agent_name}' failed to execute tools in loop "
            f"{loop_count} after {attempts} attempt(s): {last_error}"
        ) from last_error
