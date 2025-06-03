import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from typing import Any, Callable, Literal, Optional

from mcp.server.fastmcp import FastMCP
from mcp.client import Client

from loguru import logger
from swarms.utils.any_to_str import any_to_str


class AOP:
    """
    Agent-Orchestration Protocol (AOP) class for managing tools, agents, and swarms.

    This class provides decorators and methods for registering and running various components
    in a Swarms environment. It handles logging, metadata management, and execution control.

    Attributes:
        name (str): The name of the AOP instance
        description (str): A description of the AOP instance
        mcp (FastMCP): The underlying FastMCP instance for managing components
    """

    def __init__(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
        url: Optional[str] = "http://localhost:8000/sse",
        urls: Optional[list[str]] = None,
        *args,
        **kwargs,
    ):
        """
        Initialize the AOP instance.

        Args:
            name (str): The name of the AOP instance
            description (str): A description of the AOP instance
            url (str): The URL of the MCP instance
            *args: Additional positional arguments passed to FastMCP
            **kwargs: Additional keyword arguments passed to FastMCP
        """
        logger.info(f"[AOP] Initializing AOP instance: {name}")
        self.name = name
        self.description = description
        self.url = url
        self.urls = urls
        self.tools = {}
        self.swarms = {}

        self.mcp = FastMCP(name=name, *args, **kwargs)

        logger.success(
            f"[AOP] Successfully initialized AOP instance: {name}"
        )

    def tool(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """
        Decorator to register an MCP tool with optional metadata.

        This decorator registers a function as a tool in the MCP system. It handles
        logging, metadata management, and execution tracking.

        Args:
            name (Optional[str]): Custom name for the tool. If None, uses function name
            description (Optional[str]): Custom description. If None, uses function docstring

        Returns:
            Callable: A decorator function that registers the tool
        """
        logger.debug(
            f"[AOP] Creating tool decorator with name={name}, description={description}"
        )

        def decorator(func: Callable):
            tool_name = name or func.__name__
            tool_description = description or (
                inspect.getdoc(func) or ""
            )

            logger.debug(
                f"[AOP] Registering tool: {tool_name} - {tool_description}"
            )

            self.tools[tool_name] = {
                "name": tool_name,
                "description": tool_description,
                "function": func,
            }

            @self.mcp.tool(
                name=f"tool_{tool_name}", description=tool_description
            )
            @wraps(func)
            async def wrapper(*args, **kwargs) -> Any:
                logger.info(
                    f"[TOOL:{tool_name}] ➤ called with args={args}, kwargs={kwargs}"
                )
                try:
                    result = await func(*args, **kwargs)
                    logger.success(f"[TOOL:{tool_name}] ✅ completed")
                    return result
                except Exception as e:
                    logger.error(
                        f"[TOOL:{tool_name}] ❌ failed with error: {str(e)}"
                    )
                    raise

            return wrapper

        return decorator

    def agent(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """
        Decorator to define an agent entry point.

        This decorator registers a function as an agent in the MCP system. It handles
        logging, metadata management, and execution tracking for agent operations.

        Args:
            name (Optional[str]): Custom name for the agent. If None, uses 'agent_' + function name
            description (Optional[str]): Custom description. If None, uses function docstring

        Returns:
            Callable: A decorator function that registers the agent
        """
        logger.debug(
            f"[AOP] Creating agent decorator with name={name}, description={description}"
        )

        def decorator(func: Callable):
            agent_name = name or f"agent_{func.__name__}"
            agent_description = description or (
                inspect.getdoc(func) or ""
            )

            @self.mcp.tool(
                name=agent_name, description=agent_description
            )
            @wraps(func)
            async def wrapper(*args, **kwargs):
                logger.info(f"[AGENT:{agent_name}] 👤 Starting")
                try:
                    result = await func(*args, **kwargs)
                    logger.success(
                        f"[AGENT:{agent_name}] ✅ Finished"
                    )
                    return result
                except Exception as e:
                    logger.error(
                        f"[AGENT:{agent_name}] ❌ failed with error: {str(e)}"
                    )
                    raise

            wrapper._is_agent = True
            wrapper._agent_name = agent_name
            wrapper._agent_description = agent_description
            return wrapper

        return decorator

    def swarm(
        self,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ):
        """
        Decorator to define a swarm controller.

        This decorator registers a function as a swarm controller in the MCP system.
        It handles logging, metadata management, and execution tracking for swarm operations.

        Args:
            name (Optional[str]): Custom name for the swarm. If None, uses 'swarm_' + function name
            description (Optional[str]): Custom description. If None, uses function docstring

        Returns:
            Callable: A decorator function that registers the swarm
        """
        logger.debug(
            f"[AOP] Creating swarm decorator with name={name}, description={description}"
        )

        def decorator(func: Callable):
            swarm_name = name or f"swarm_{func.__name__}"
            swarm_description = description or (
                inspect.getdoc(func) or ""
            )

            @self.mcp.tool(
                name=swarm_name, description=swarm_description
            )
            @wraps(func)
            async def wrapper(*args, **kwargs):
                logger.info(
                    f"[SWARM:{swarm_name}] 🐝 Spawning swarm..."
                )
                try:
                    result = await func(*args, **kwargs)
                    logger.success(
                        f"[SWARM:{swarm_name}] 🐝 Completed"
                    )
                    return result
                except Exception as e:
                    logger.error(
                        f"[SWARM:{swarm_name}] ❌ failed with error: {str(e)}"
                    )
                    raise

            wrapper._is_swarm = True
            wrapper._swarm_name = swarm_name
            wrapper._swarm_description = swarm_description
            return wrapper

        return decorator

    def run(self, method: Literal["stdio", "sse"], *args, **kwargs):
        """
        Run the MCP with the specified method.

        Args:
            method (Literal['stdio', 'sse']): The execution method to use
            *args: Additional positional arguments for the run method
            **kwargs: Additional keyword arguments for the run method

        Returns:
            Any: The result of the MCP run operation
        """
        logger.info(f"[AOP] Running MCP with method: {method}")
        try:
            result = self.mcp.run(method, *args, **kwargs)
            logger.success(
                f"[AOP] Successfully ran MCP with method: {method}"
            )
            return result
        except Exception as e:
            logger.error(
                f"[AOP] Failed to run MCP with method {method}: {str(e)}"
            )
            raise

    def run_stdio(self, *args, **kwargs):
        """
        Run the MCP using standard I/O method.

        Args:
            *args: Additional positional arguments for the run method
            **kwargs: Additional keyword arguments for the run method

        Returns:
            Any: The result of the MCP run operation
        """
        logger.info("[AOP] Running MCP with stdio method")
        return self.run("stdio", *args, **kwargs)

    def run_sse(self, *args, **kwargs):
        """
        Run the MCP using Server-Sent Events method.

        Args:
            *args: Additional positional arguments for the run method
            **kwargs: Additional keyword arguments for the run method

        Returns:
            Any: The result of the MCP run operation
        """
        logger.info("[AOP] Running MCP with SSE method")
        return self.run("sse", *args, **kwargs)

    def list_available(
        self, output_type: Literal["str", "list"] = "str"
    ):
        """
        List all available tools in the MCP.

        Returns:
            list: A list of all registered tools
        """
        if output_type == "str":
            return any_to_str(self.mcp.list_tools())
        elif output_type == "list":
            return self.mcp.list_tools()
        else:
            raise ValueError(f"Invalid output type: {output_type}")

    async def check_utility_exists(
        self, url: str, name: str, *args, **kwargs
    ):
        async with Client(url, *args, **kwargs) as client:
            if any(tool.name == name for tool in client.list_tools()):
                return True
            else:
                return False

    async def _call_tool(
        self, url: str, name: str, arguments: dict, *args, **kwargs
    ):
        try:
            async with Client(url, *args, **kwargs) as client:
                result = await client.call_tool(name, arguments)
                logger.info(
                    f"Client connected: {client.is_connected()}"
                )
                return result
        except Exception as e:
            logger.error(f"Error calling tool: {e}")
            return None

    def call_tool(
        self,
        url: str,
        name: str,
        arguments: dict,
        *args,
        **kwargs,
    ):
        return asyncio.run(
            self._call_tool(url, name, arguments, *args, **kwargs)
        )

    def call_tool_or_agent(
        self,
        url: str,
        name: str,
        arguments: dict,
        output_type: Literal["str", "list"] = "str",
        *args,
        **kwargs,
    ):
        """
        Execute a tool or agent by name.

        Args:
            name (str): The name of the tool or agent to execute
            arguments (dict): The arguments to pass to the tool or agent
        """
        if output_type == "str":
            return any_to_str(
                self.call_tool(
                    url=url, name=name, arguments=arguments
                )
            )
        elif output_type == "list":
            return self.call_tool(
                url=url, name=name, arguments=arguments
            )
        else:
            raise ValueError(f"Invalid output type: {output_type}")

    def call_tool_or_agent_batched(
        self,
        url: str,
        names: list[str],
        arguments: list[dict],
        output_type: Literal["str", "list"] = "str",
        *args,
        **kwargs,
    ):
        """
        Execute a list of tools or agents by name.

        Args:
            names (list[str]): The names of the tools or agents to execute
        """
        if output_type == "str":
            return [
                any_to_str(
                    self.call_tool_or_agent(
                        url=url,
                        name=name,
                        arguments=argument,
                        *args,
                        **kwargs,
                    )
                )
                for name, argument in zip(names, arguments)
            ]
        elif output_type == "list":
            return [
                self.call_tool_or_agent(
                    url=url,
                    name=name,
                    arguments=argument,
                    *args,
                    **kwargs,
                )
                for name, argument in zip(names, arguments)
            ]
        else:
            raise ValueError(f"Invalid output type: {output_type}")

    def call_tool_or_agent_concurrently(
        self,
        url: str,
        names: list[str],
        arguments: list[dict],
        output_type: Literal["str", "list"] = "str",
        *args,
        **kwargs,
    ):
        """
        Execute a list of tools or agents by name concurrently.

        Args:
            names (list[str]): The names of the tools or agents to execute
            arguments (list[dict]): The arguments to pass to the tools or agents
        """
        outputs = []
        with ThreadPoolExecutor(max_workers=len(names)) as executor:
            futures = [
                executor.submit(
                    self.call_tool_or_agent,
                    url=url,
                    name=name,
                    arguments=argument,
                    *args,
                    **kwargs,
                )
                for name, argument in zip(names, arguments)
            ]
            for future in as_completed(futures):
                outputs.append(future.result())

        if output_type == "str":
            return any_to_str(outputs)
        elif output_type == "list":
            return outputs
        else:
            raise ValueError(f"Invalid output type: {output_type}")

    def call_swarm(
        self,
        url: str,
        name: str,
        arguments: dict,
        output_type: Literal["str", "list"] = "str",
        *args,
        **kwargs,
    ):
        """
        Execute a swarm by name.

        Args:
            name (str): The name of the swarm to execute
        """
        if output_type == "str":
            return any_to_str(
                asyncio.run(
                    self._call_tool(
                        url=url,
                        name=name,
                        arguments=arguments,
                    )
                )
            )
        elif output_type == "list":
            return asyncio.run(
                self._call_tool(
                    url=url,
                    name=name,
                    arguments=arguments,
                )
            )
        else:
            raise ValueError(f"Invalid output type: {output_type}")

    def list_agents(
        self, output_type: Literal["str", "list"] = "str"
    ):
        """
        List all available agents in the MCP.

        Returns:
            list: A list of all registered agents
        """

        out = self.list_all()
        agents = []
        for item in out:
            if "agent" in item["name"]:
                agents.append(item)
        return agents

    def list_swarms(
        self, output_type: Literal["str", "list"] = "str"
    ):
        """
        List all available swarms in the MCP.

        Returns:
            list: A list of all registered swarms
        """
        out = self.list_all()
        agents = []
        for item in out:
            if "swarm" in item["name"]:
                agents.append(item)
        return agents

    async def _list_all(self):
        async with Client(self.url) as client:
            return await client.list_tools()

    def list_all(self):
        out = asyncio.run(self._list_all())

        outputs = []
        for tool in out:
            outputs.append(tool.model_dump())

        return outputs

    def list_tool_parameters(self, name: str):
        out = self.list_all()

        # Find the tool by name
        for tool in out:
            if tool["name"] == name:
                return tool
        return None

    def list_tools_for_multiple_urls(self):
        out = []
        for url in self.urls:
            out.append(self.list_all(url))
        return out

    def search_if_tool_exists(self, name: str):
        out = self.list_all()
        for tool in out:
            if tool["name"] == name:
                return True
        return False

    def search(
        self,
        type: Literal["tool", "agent", "swarm"],
        name: str,
        output_type: Literal["str", "list"] = "str",
    ):
        """
        Search for a tool, agent, or swarm by name.

        Args:
            type (Literal["tool", "agent", "swarm"]): The type of the item to search for
            name (str): The name of the item to search for

        Returns:
            dict: The item if found, otherwise None
        """
        all_items = self.list_all()
        for item in all_items:
            if item["name"] == name:
                return item
        return None
