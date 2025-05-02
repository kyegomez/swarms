import asyncio
import json
from typing import List, Literal, Dict, Any, Union
from fastmcp import Client
from swarms.utils.str_to_dict import str_to_dict
from loguru import logger


def parse_agent_output(
    dictionary: Union[str, Dict[Any, Any]]
) -> tuple[str, Dict[Any, Any]]:
    """
    Parse agent output into tool name and parameters.

    Args:
        dictionary: Either a string or dictionary containing tool information.
                   If string, it will be converted to a dictionary.
                   Must contain a 'name' key for the tool name.

    Returns:
        tuple[str, Dict[Any, Any]]: A tuple containing the tool name and its parameters.

    Raises:
        ValueError: If the input is invalid or missing required 'name' key.
    """
    try:
        if isinstance(dictionary, str):
            dictionary = str_to_dict(dictionary)

        elif not isinstance(dictionary, dict):
            raise ValueError("Invalid dictionary")

        # Handle regular dictionary format
        if "name" in dictionary:
            name = dictionary["name"]
            # Remove the name key and use remaining key-value pairs as parameters
            params = dict(dictionary)
            params.pop("name")
            return name, params

        raise ValueError("Invalid function call format")
    except Exception as e:
        raise ValueError(f"Error parsing agent output: {str(e)}")


async def _list_all(url: str):
    """
    Asynchronously list all tools available on a given MCP server.

    Args:
        url: The URL of the MCP server to query.

    Returns:
        List of available tools.

    Raises:
        ValueError: If there's an error connecting to or querying the server.
    """
    try:
        async with Client(url) as client:
            return await client.list_tools()
    except Exception as e:
        raise ValueError(f"Error listing tools: {str(e)}")


def list_all(url: str, output_type: Literal["str", "json"] = "json"):
    """
    Synchronously list all tools available on a given MCP server.

    Args:
        url: The URL of the MCP server to query.

    Returns:
        List of dictionaries containing tool information.

    Raises:
        ValueError: If there's an error connecting to or querying the server.
    """
    try:
        out = asyncio.run(_list_all(url))

        outputs = []
        for tool in out:
            outputs.append(tool.model_dump())

        if output_type == "json":
            return json.dumps(outputs, indent=4)
        else:
            return outputs
    except Exception as e:
        raise ValueError(f"Error in list_all: {str(e)}")


def list_tools_for_multiple_urls(
    urls: List[str], output_type: Literal["str", "json"] = "json"
):
    """
    List tools available across multiple MCP servers.

    Args:
        urls: List of MCP server URLs to query.
        output_type: Format of the output, either "json" (string) or "str" (list).

    Returns:
        If output_type is "json": JSON string containing all tools with server URLs.
        If output_type is "str": List of tools with server URLs.

    Raises:
        ValueError: If there's an error querying any of the servers.
    """
    try:
        out = []
        for url in urls:
            tools = list_all(url)
            # Add server URL to each tool's data
            for tool in tools:
                tool["server_url"] = url
            out.append(tools)

        if output_type == "json":
            return json.dumps(out, indent=4)
        else:
            return out
    except Exception as e:
        raise ValueError(
            f"Error listing tools for multiple URLs: {str(e)}"
        )


async def _execute_mcp_tool(
    url: str,
    parameters: Dict[Any, Any] = None,
    *args,
    **kwargs,
) -> Dict[Any, Any]:
    """
    Asynchronously execute a tool on an MCP server.

    Args:
        url: The URL of the MCP server.
        parameters: Dictionary containing tool name and parameters.
        *args: Additional positional arguments for the Client.
        **kwargs: Additional keyword arguments for the Client.

    Returns:
        Dictionary containing the tool execution results.

    Raises:
        ValueError: If the URL is invalid or tool execution fails.
    """
    try:

        name, params = parse_agent_output(parameters)

        outputs = []

        async with Client(url, *args, **kwargs) as client:
            out = await client.call_tool(
                name=name,
                arguments=params,
            )

            for output in out:
                outputs.append(output.model_dump())

        # convert outputs to string
        return json.dumps(outputs, indent=4)
    except Exception as e:
        raise ValueError(f"Error executing MCP tool: {str(e)}")


def execute_mcp_tool(
    url: str,
    parameters: Dict[Any, Any] = None,
) -> Dict[Any, Any]:
    """
    Synchronously execute a tool on an MCP server.

    Args:
        url: The URL of the MCP server.
        parameters: Dictionary containing tool name and parameters.

    Returns:
        Dictionary containing the tool execution results.

    Raises:
        ValueError: If tool execution fails.
    """
    try:
        logger.info(f"Executing MCP tool with URL: {url}")
        logger.debug(f"Tool parameters: {parameters}")

        result = asyncio.run(
            _execute_mcp_tool(
                url=url,
                parameters=parameters,
            )
        )

        logger.info("MCP tool execution completed successfully")
        logger.debug(f"Tool execution result: {result}")
        return result
    except Exception as e:
        logger.error(f"Error in execute_mcp_tool: {str(e)}")
        raise ValueError(f"Error in execute_mcp_tool: {str(e)}")


def find_and_execute_tool(
    urls: List[str], tool_name: str, parameters: Dict[Any, Any]
) -> Dict[Any, Any]:
    """
    Find a tool across multiple servers and execute it with the given parameters.

    Args:
        urls: List of server URLs to search through.
        tool_name: Name of the tool to find and execute.
        parameters: Parameters to pass to the tool.

    Returns:
        Dict containing the tool execution results.

    Raises:
        ValueError: If tool is not found on any server or execution fails.
    """
    try:
        # Search for tool across all servers
        for url in urls:
            try:
                tools = list_all(url)
                # Check if tool exists on this server
                if any(tool["name"] == tool_name for tool in tools):
                    # Prepare parameters in correct format
                    tool_params = {"name": tool_name, **parameters}
                    # Execute tool on this server
                    return execute_mcp_tool(
                        url=url, parameters=tool_params
                    )
            except Exception:
                # Skip servers that fail and continue searching
                continue

        raise ValueError(
            f"Tool '{tool_name}' not found on any provided servers"
        )
    except Exception as e:
        raise ValueError(f"Error in find_and_execute_tool: {str(e)}")
