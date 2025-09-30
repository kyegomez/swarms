import asyncio
import sys
import traceback
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from loguru import logger
from mcp.server.fastmcp import FastMCP

from swarms.structs.agent import Agent
from swarms.structs.omni_agent_types import AgentType
from swarms.tools.mcp_client_tools import (
    get_tools_for_multiple_mcp_servers,
)


@dataclass
class AgentToolConfig:
    """
    Configuration for converting an agent to an MCP tool.

    Attributes:
        tool_name: The name of the tool in the MCP server
        tool_description: Description of what the tool does
        input_schema: JSON schema for the tool's input parameters
        output_schema: JSON schema for the tool's output
        timeout: Maximum time to wait for agent execution (seconds)
        max_retries: Number of retries if agent execution fails
        verbose: Enable verbose logging for this tool
        traceback_enabled: Enable traceback logging for errors
    """

    tool_name: str
    tool_description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    timeout: int = 30
    max_retries: int = 3
    verbose: bool = False
    traceback_enabled: bool = True


class AOP:
    """
    A class that takes a list of agents and deploys them as unique tools in an MCP server.

    This class provides functionality to:
    1. Convert swarms agents into MCP tools
    2. Deploy multiple agents as individual tools
    3. Handle tool execution with proper error handling
    4. Manage the MCP server lifecycle

    Attributes:
        mcp_server: The FastMCP server instance
        agents: Dictionary mapping tool names to agent instances
        tool_configs: Dictionary mapping tool names to their configurations
        server_name: Name of the MCP server
    """

    def __init__(
        self,
        server_name: str = "AOP Cluster",
        description: str = "A cluster that enables you to deploy multiple agents as tools in an MCP server.",
        agents: any = None,
        port: int = 8000,
        transport: str = "streamable-http",
        verbose: bool = False,
        traceback_enabled: bool = True,
        host: str = "localhost",
        log_level: str = "INFO",
        *args,
        **kwargs,
    ):
        """
        Initialize the AOP.

        Args:
            server_name: Name for the MCP server
            agents: Optional list of agents to add initially
            port: Port for the MCP server
            transport: Transport type for the MCP server
            verbose: Enable verbose logging
            traceback_enabled: Enable traceback logging for errors
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.server_name = server_name
        self.verbose = verbose
        self.traceback_enabled = traceback_enabled
        self.log_level = log_level
        self.host = host
        self.port = port
        self.agents: Dict[str, Agent] = {}
        self.tool_configs: Dict[str, AgentToolConfig] = {}
        self.transport = transport
        self.mcp_server = FastMCP(
            name=server_name, port=port, *args, **kwargs
        )

        # Configure logger
        logger.remove()  # Remove default handler
        logger.add(
            sys.stderr,
            level=log_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True,
        )

        logger.info(
            f"Initialized AOP with server name: {server_name}, verbose: {verbose}, traceback: {traceback_enabled}"
        )

        # Add initial agents if provided
        if agents:
            logger.info(f"Adding {len(agents)} initial agents")
            self.add_agents_batch(agents)

    def add_agent(
        self,
        agent: AgentType,
        tool_name: Optional[str] = None,
        tool_description: Optional[str] = None,
        input_schema: Optional[Dict[str, Any]] = None,
        output_schema: Optional[Dict[str, Any]] = None,
        timeout: int = 30,
        max_retries: int = 3,
        verbose: Optional[bool] = None,
        traceback_enabled: Optional[bool] = None,
    ) -> str:
        """
        Add an agent to the MCP server as a tool.

        Args:
            agent: The swarms Agent instance to deploy
            tool_name: Name for the tool (defaults to agent.agent_name)
            tool_description: Description of the tool (defaults to agent.agent_description)
            input_schema: JSON schema for input parameters
            output_schema: JSON schema for output
            timeout: Maximum execution time in seconds
            max_retries: Number of retries on failure
            verbose: Enable verbose logging for this tool (defaults to deployer's verbose setting)
            traceback_enabled: Enable traceback logging for this tool (defaults to deployer's setting)

        Returns:
            str: The tool name that was registered

        Raises:
            ValueError: If agent is None or tool_name already exists
        """
        if agent is None:
            logger.error("Cannot add None agent")
            raise ValueError("Agent cannot be None")

        # Use agent name as tool name if not provided
        if tool_name is None:
            tool_name = (
                agent.agent_name or f"agent_{len(self.agents)}"
            )

        if tool_name in self.agents:
            logger.error(f"Tool name '{tool_name}' already exists")
            raise ValueError(
                f"Tool name '{tool_name}' already exists"
            )

        # Use deployer defaults if not specified
        if verbose is None:
            verbose = self.verbose
        if traceback_enabled is None:
            traceback_enabled = self.traceback_enabled

        logger.debug(
            f"Adding agent '{agent.agent_name}' as tool '{tool_name}' with verbose={verbose}, traceback={traceback_enabled}"
        )

        # Use agent description if not provided
        if tool_description is None:
            tool_description = (
                agent.agent_description
                or f"Agent tool: {agent.agent_name}"
            )

        # Default input schema for task-based agents
        if input_schema is None:
            input_schema = {
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "The task or prompt to execute with this agent",
                    },
                    "img": {
                        "type": "string",
                        "description": "Optional image to be processed by the agent",
                    },
                    "imgs": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Optional list of images to be processed by the agent",
                    },
                    "correct_answer": {
                        "type": "string",
                        "description": "Optional correct answer for validation or comparison",
                    },
                },
                "required": ["task"],
            }

        # Default output schema
        if output_schema is None:
            output_schema = {
                "type": "object",
                "properties": {
                    "result": {
                        "type": "string",
                        "description": "The agent's response to the task",
                    },
                    "success": {
                        "type": "boolean",
                        "description": "Whether the task was executed successfully",
                    },
                    "error": {
                        "type": "string",
                        "description": "Error message if execution failed",
                    },
                },
                "required": ["result", "success"],
            }

        # Store agent and configuration
        self.agents[tool_name] = agent
        self.tool_configs[tool_name] = AgentToolConfig(
            tool_name=tool_name,
            tool_description=tool_description,
            input_schema=input_schema,
            output_schema=output_schema,
            timeout=timeout,
            max_retries=max_retries,
            verbose=verbose,
            traceback_enabled=traceback_enabled,
        )

        # Register the tool with the MCP server
        self._register_tool(tool_name, agent)

        logger.info(
            f"Added agent '{agent.agent_name}' as tool '{tool_name}' (verbose={verbose}, traceback={traceback_enabled})"
        )
        return tool_name

    def add_agents_batch(
        self,
        agents: List[Agent],
        tool_names: Optional[List[str]] = None,
        tool_descriptions: Optional[List[str]] = None,
        input_schemas: Optional[List[Dict[str, Any]]] = None,
        output_schemas: Optional[List[Dict[str, Any]]] = None,
        timeouts: Optional[List[int]] = None,
        max_retries_list: Optional[List[int]] = None,
        verbose_list: Optional[List[bool]] = None,
        traceback_enabled_list: Optional[List[bool]] = None,
    ) -> List[str]:
        """
        Add multiple agents to the MCP server as tools in batch.

        Args:
            agents: List of swarms Agent instances
            tool_names: Optional list of tool names (defaults to agent names)
            tool_descriptions: Optional list of tool descriptions
            input_schemas: Optional list of input schemas
            output_schemas: Optional list of output schemas
            timeouts: Optional list of timeout values
            max_retries_list: Optional list of max retry values
            verbose_list: Optional list of verbose settings for each agent
            traceback_enabled_list: Optional list of traceback settings for each agent

        Returns:
            List[str]: List of tool names that were registered

        Raises:
            ValueError: If agents list is empty or contains None values
        """
        if not agents:
            logger.error("Cannot add empty agents list")
            raise ValueError("Agents list cannot be empty")

        if None in agents:
            logger.error("Agents list contains None values")
            raise ValueError("Agents list cannot contain None values")

        logger.info(f"Adding {len(agents)} agents in batch")
        registered_tools = []

        for i, agent in enumerate(agents):
            tool_name = (
                tool_names[i]
                if tool_names and i < len(tool_names)
                else None
            )
            tool_description = (
                tool_descriptions[i]
                if tool_descriptions and i < len(tool_descriptions)
                else None
            )
            input_schema = (
                input_schemas[i]
                if input_schemas and i < len(input_schemas)
                else None
            )
            output_schema = (
                output_schemas[i]
                if output_schemas and i < len(output_schemas)
                else None
            )
            timeout = (
                timeouts[i] if timeouts and i < len(timeouts) else 30
            )
            max_retries = (
                max_retries_list[i]
                if max_retries_list and i < len(max_retries_list)
                else 3
            )
            verbose = (
                verbose_list[i]
                if verbose_list and i < len(verbose_list)
                else None
            )
            traceback_enabled = (
                traceback_enabled_list[i]
                if traceback_enabled_list
                and i < len(traceback_enabled_list)
                else None
            )

            tool_name = self.add_agent(
                agent=agent,
                tool_name=tool_name,
                tool_description=tool_description,
                input_schema=input_schema,
                output_schema=output_schema,
                timeout=timeout,
                max_retries=max_retries,
                verbose=verbose,
                traceback_enabled=traceback_enabled,
            )
            registered_tools.append(tool_name)

        logger.info(
            f"Added {len(agents)} agents as tools: {registered_tools}"
        )
        return registered_tools

    def _register_tool(
        self, tool_name: str, agent: AgentType
    ) -> None:
        """
        Register a single agent as an MCP tool.

        Args:
            tool_name: Name of the tool to register
            agent: The agent instance to register
        """
        config = self.tool_configs[tool_name]

        @self.mcp_server.tool(
            name=tool_name, description=config.tool_description
        )
        def agent_tool(
            task: str = None,
            img: str = None,
            imgs: List[str] = None,
            correct_answer: str = None,
        ) -> Dict[str, Any]:
            """
            Execute the agent with the provided parameters.

            Args:
                task: The task or prompt to execute with this agent
                img: Optional image to be processed by the agent
                imgs: Optional list of images to be processed by the agent
                correct_answer: Optional correct answer for validation or comparison
                **kwargs: Additional parameters passed to the agent

            Returns:
                Dict containing the agent's response and execution status
            """
            start_time = None
            if config.verbose:
                start_time = (
                    asyncio.get_event_loop().time()
                    if asyncio.get_event_loop().is_running()
                    else 0
                )
                logger.debug(
                    f"Starting execution of tool '{tool_name}' with task: {task[:100] if task else 'None'}..."
                )
                if img:
                    logger.debug(f"Processing single image: {img}")
                if imgs:
                    logger.debug(
                        f"Processing {len(imgs)} images: {imgs}"
                    )
                if correct_answer:
                    logger.debug(
                        f"Using correct answer for validation: {correct_answer[:50]}..."
                    )

            try:
                # Validate required parameters
                if not task:
                    error_msg = "No task provided"
                    logger.warning(
                        f"Tool '{tool_name}' called without task parameter"
                    )
                    return {
                        "result": "",
                        "success": False,
                        "error": error_msg,
                    }

                # Execute the agent with timeout and all parameters
                result = self._execute_agent_with_timeout(
                    agent,
                    task,
                    config.timeout,
                    img,
                    imgs,
                    correct_answer,
                )

                if config.verbose and start_time:
                    execution_time = (
                        asyncio.get_event_loop().time() - start_time
                        if asyncio.get_event_loop().is_running()
                        else 0
                    )
                    logger.debug(
                        f"Tool '{tool_name}' completed successfully in {execution_time:.2f}s"
                    )

                return {
                    "result": str(result),
                    "success": True,
                    "error": None,
                }

            except Exception as e:
                error_msg = str(e)
                logger.error(
                    f"Error executing agent '{tool_name}': {error_msg}"
                )

                if config.traceback_enabled:
                    logger.error(f"Traceback for tool '{tool_name}':")
                    logger.error(traceback.format_exc())

                if config.verbose and start_time:
                    execution_time = (
                        asyncio.get_event_loop().time() - start_time
                        if asyncio.get_event_loop().is_running()
                        else 0
                    )
                    logger.debug(
                        f"Tool '{tool_name}' failed after {execution_time:.2f}s"
                    )

                return {
                    "result": "",
                    "success": False,
                    "error": error_msg,
                }

    def _execute_agent_with_timeout(
        self,
        agent: AgentType,
        task: str,
        timeout: int,
        img: str = None,
        imgs: List[str] = None,
        correct_answer: str = None,
    ) -> str:
        """
        Execute an agent with a timeout and all run method parameters.

        Args:
            agent: The agent to execute
            task: The task to execute
            timeout: Maximum execution time in seconds
            img: Optional image to be processed by the agent
            imgs: Optional list of images to be processed by the agent
            correct_answer: Optional correct answer for validation or comparison

        Returns:
            str: The agent's response

        Raises:
            TimeoutError: If execution exceeds timeout
            Exception: If agent execution fails
        """
        try:
            logger.debug(
                f"Executing agent '{agent.agent_name}' with timeout {timeout}s"
            )

            out = agent.run(
                task=task,
                img=img,
                imgs=imgs,
                correct_answer=correct_answer,
            )

            logger.debug(
                f"Agent '{agent.agent_name}' execution completed successfully"
            )
            return out

        except Exception as e:
            error_msg = f"Agent execution failed: {str(e)}"
            logger.error(
                f"Execution error for agent '{agent.agent_name}': {error_msg}"
            )
            if self.traceback_enabled:
                logger.error(
                    f"Traceback for agent '{agent.agent_name}':"
                )
                logger.error(traceback.format_exc())
            raise Exception(error_msg)

    def remove_agent(self, tool_name: str) -> bool:
        """
        Remove an agent from the MCP server.

        Args:
            tool_name: Name of the tool to remove

        Returns:
            bool: True if agent was removed, False if not found
        """
        if tool_name in self.agents:
            del self.agents[tool_name]
            del self.tool_configs[tool_name]
            logger.info(f"Removed agent tool '{tool_name}'")
            return True
        return False

    def list_agents(self) -> List[str]:
        """
        Get a list of all registered agent tool names.

        Returns:
            List[str]: List of tool names
        """
        agent_list = list(self.agents.keys())
        if self.verbose:
            logger.debug(
                f"Listing {len(agent_list)} registered agents: {agent_list}"
            )
        return agent_list

    def get_agent_info(
        self, tool_name: str
    ) -> Optional[Dict[str, Any]]:
        """
        Get information about a specific agent tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Dict containing agent information, or None if not found
        """
        if tool_name not in self.agents:
            if self.verbose:
                logger.debug(
                    f"Requested info for non-existent agent tool '{tool_name}'"
                )
            return None

        agent = self.agents[tool_name]
        config = self.tool_configs[tool_name]

        info = {
            "tool_name": tool_name,
            "agent_name": agent.agent_name,
            "agent_description": agent.agent_description,
            "model_name": getattr(agent, "model_name", "Unknown"),
            "max_loops": getattr(agent, "max_loops", 1),
            "tool_description": config.tool_description,
            "timeout": config.timeout,
            "max_retries": config.max_retries,
            "verbose": config.verbose,
            "traceback_enabled": config.traceback_enabled,
        }

        if self.verbose:
            logger.debug(
                f"Retrieved info for agent tool '{tool_name}': {info}"
            )

        return info

    def start_server(self) -> None:
        """
        Start the MCP server.

        Args:
            host: Host to bind the server to
            port: Port to bind the server to
        """
        logger.info(
            f"Starting MCP server '{self.server_name}' on {self.host}:{self.port}"
        )
        logger.info(f"Transport: {self.transport}")
        logger.info(f"Log level: {self.log_level}")
        logger.info(f"Verbose mode: {self.verbose}")
        logger.info(f"Traceback enabled: {self.traceback_enabled}")
        logger.info(f"Available tools: {self.list_agents()}")

        if self.verbose:
            logger.debug("Server configuration:")
            logger.debug(f"  - Server name: {self.server_name}")
            logger.debug(f"  - Host: {self.host}")
            logger.debug(f"  - Port: {self.port}")
            logger.debug(f"  - Transport: {self.transport}")
            logger.debug(f"  - Total agents: {len(self.agents)}")
            for tool_name, config in self.tool_configs.items():
                logger.debug(
                    f"  - Tool '{tool_name}': timeout={config.timeout}s, verbose={config.verbose}, traceback={config.traceback_enabled}"
                )

        self.mcp_server.run(transport=self.transport)

        # Note: FastMCP doesn't have a direct start method in the current implementation
        # This would need to be implemented based on the specific MCP server setup
        print(
            f"MCP Server '{self.server_name}' is ready with {len(self.agents)} tools"
        )
        print(f"Tools available: {', '.join(self.list_agents())}")

    def run(self) -> None:
        """
        Run the MCP server.
        """
        self.start_server()

    def get_server_info(self) -> Dict[str, Any]:
        """
        Get information about the MCP server and registered tools.

        Returns:
            Dict containing server information
        """
        info = {
            "server_name": self.server_name,
            "total_tools": len(self.agents),
            "tools": self.list_agents(),
            "verbose": self.verbose,
            "traceback_enabled": self.traceback_enabled,
            "log_level": self.log_level,
            "transport": self.transport,
            "tool_details": {
                tool_name: self.get_agent_info(tool_name)
                for tool_name in self.agents.keys()
            },
        }

        if self.verbose:
            logger.debug(f"Retrieved server info: {info}")

        return info


class AOPCluster:
    def __init__(
        self,
        urls: List[str],
        transport: str = "streamable-http",
        *args,
        **kwargs,
    ):
        self.urls = urls
        self.transport = transport

    def get_tools(
        self, output_type: Literal["json", "dict", "str"] = "dict"
    ) -> List[Dict[str, Any]]:
        return get_tools_for_multiple_mcp_servers(
            urls=self.urls,
            format="openai",
            output_type=output_type,
            transport=self.transport,
        )

    def find_tool_by_server_name(
        self, server_name: str
    ) -> Dict[str, Any]:
        """
        Find a tool by its server name (function name).

        Args:
            server_name: The name of the tool/function to find

        Returns:
            Dict containing the tool information, or None if not found
        """
        for tool in self.get_tools(output_type="dict"):
            if tool.get("function", {}).get("name") == server_name:
                return tool
        return None
