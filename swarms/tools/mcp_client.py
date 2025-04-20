import asyncio
from typing import Literal, Dict, Any, Union
from fastmcp import Client
from swarms.utils.any_to_str import any_to_str
from swarms.utils.str_to_dict import str_to_dict


def parse_agent_output(
    dictionary: Union[str, Dict[Any, Any]]
) -> tuple[str, Dict[Any, Any]]:
    if isinstance(dictionary, str):
        dictionary = str_to_dict(dictionary)

    elif not isinstance(dictionary, dict):
        raise ValueError("Invalid dictionary")

    # Handle OpenAI function call format
    if "function_call" in dictionary:
        name = dictionary["function_call"]["name"]
        # arguments is a JSON string, so we need to parse it
        params = str_to_dict(dictionary["function_call"]["arguments"])
        return name, params

    # Handle OpenAI tool calls format
    if "tool_calls" in dictionary:
        # Get the first tool call (or you could handle multiple if needed)
        tool_call = dictionary["tool_calls"][0]
        name = tool_call["function"]["name"]
        params = str_to_dict(tool_call["function"]["arguments"])
        return name, params

    # Handle regular dictionary format
    if "name" in dictionary:
        name = dictionary["name"]
        params = dictionary.get("arguments", {})
        return name, params

    raise ValueError("Invalid function call format")


async def _execute_mcp_tool(
    url: str,
    method: Literal["stdio", "sse"] = "sse",
    parameters: Dict[Any, Any] = None,
    output_type: Literal["str", "dict"] = "str",
    *args,
    **kwargs,
) -> Dict[Any, Any]:

    if "sse" or "stdio" not in url:
        raise ValueError("Invalid URL")

    url = f"{url}/{method}"

    name, params = parse_agent_output(parameters)

    if output_type == "str":
        async with Client(url, *args, **kwargs) as client:
            out = await client.call_tool(
                name=name,
                arguments=params,
            )
            return any_to_str(out)
    elif output_type == "dict":
        async with Client(url, *args, **kwargs) as client:
            out = await client.call_tool(
                name=name,
                arguments=params,
            )
            return out
    else:
        raise ValueError(f"Invalid output type: {output_type}")


def execute_mcp_tool(
    url: str,
    tool_name: str = None,
    method: Literal["stdio", "sse"] = "sse",
    parameters: Dict[Any, Any] = None,
    output_type: Literal["str", "dict"] = "str",
) -> Dict[Any, Any]:
    return asyncio.run(
        _execute_mcp_tool(
            url=url,
            tool_name=tool_name,
            method=method,
            parameters=parameters,
            output_type=output_type,
        )
    )
