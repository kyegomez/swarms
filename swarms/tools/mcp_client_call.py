import litellm
import asyncio
import contextlib
import random
from functools import wraps
from typing import Any, Dict, List

from litellm.experimental_mcp_client import (
    call_openai_tool,
    load_mcp_tools,
)
from loguru import logger
from mcp import ClientSession
from mcp.client.sse import sse_client

import os


def retry_with_backoff(retries=3, backoff_in_seconds=1):
    """Decorator for retrying functions with exponential backoff."""

    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            x = 0
            while True:
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if x == retries:
                        logger.error(
                            f"Failed after {retries} retries: {str(e)}"
                        )
                        raise
                    sleep_time = (
                        backoff_in_seconds * 2**x
                        + random.uniform(0, 1)
                    )
                    logger.warning(
                        f"Attempt {x + 1} failed, retrying in {sleep_time:.2f}s"
                    )
                    await asyncio.sleep(sleep_time)
                    x += 1

        return wrapper

    return decorator


@contextlib.contextmanager
def get_or_create_event_loop():
    """Context manager to handle event loop creation and cleanup."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    try:
        yield loop
    finally:
        if loop.is_running():
            loop.stop()
        if not loop.is_closed():
            loop.close()


@retry_with_backoff(retries=3)
async def aget_mcp_tools(
    server_path: str, format: str = "openai", *args, **kwargs
) -> List[Dict[str, Any]]:
    """
    Fetch available MCP tools from the server with retry logic.

    Args:
        server_path (str): Path to the MCP server script

    Returns:
        List[Dict[str, Any]]: List of available MCP tools in OpenAI format

    Raises:
        ValueError: If server_path is invalid
        ConnectionError: If connection to server fails
    """
    if not server_path or not isinstance(server_path, str):
        raise ValueError("Invalid server path provided")

    logger.info(f"Fetching MCP tools from server: {server_path}")

    try:
        async with sse_client(server_path, *args, **kwargs) as (
            read,
            write,
        ):
            async with ClientSession(read, write) as session:
                await session.initialize()
                tools = await load_mcp_tools(
                    session=session, format=format
                )
                logger.info(
                    f"Successfully fetched {len(tools)} tools"
                )
                return tools
    except Exception as e:
        logger.error(f"Error fetching MCP tools: {str(e)}")
        raise


async def get_mcp_tools(
    server_path: str, *args, **kwargs
) -> List[Dict[str, Any]]:
    return await aget_mcp_tools(server_path, *args, **kwargs)


def get_mcp_tools_sync(
    server_path: str, format: str = "openai", *args, **kwargs
) -> List[Dict[str, Any]]:
    """
    Synchronous version of get_mcp_tools that handles event loop management.

    Args:
        server_path (str): Path to the MCP server script

    Returns:
        List[Dict[str, Any]]: List of available MCP tools in OpenAI format

    Raises:
        ValueError: If server_path is invalid
        ConnectionError: If connection to server fails
        RuntimeError: If event loop management fails
    """
    with get_or_create_event_loop() as loop:
        try:
            return loop.run_until_complete(
                aget_mcp_tools(server_path, format, *args, **kwargs)
            )
        except Exception as e:
            logger.error(f"Error in get_mcp_tools_sync: {str(e)}")
            raise


async def execute_tool_call(
    server_path: str,
    messages: List[Dict[str, Any]],
    model: str = "o3-mini",
    *args,
    **kwargs,
) -> Dict[str, Any]:
    """
    Execute a tool call using the MCP client with retry logic.

    Args:
        server_path (str): Path to the MCP server script
        messages (List[Dict[str, Any]]): Current conversation messages
        model (str): The model to use for completion (default: "gpt-4")

    Returns:
        Dict[str, Any]: Final LLM response after tool execution

    Raises:
        ValueError: If inputs are invalid
        ConnectionError: If connection to server fails
        RuntimeError: If tool execution fails
    """

    async with sse_client(server_path, *args, **kwargs) as (
        read,
        write,
    ):
        async with ClientSession(read, write) as session:
            try:
                # Initialize the connection
                await session.initialize()

                # Get tools
                tools = await load_mcp_tools(
                    session=session, format="openai"
                )
                logger.info(f"Tools: {tools}")

                # First LLM call to get tool call
                llm_response = await litellm.acompletion(
                    model=model,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    # parallel_tool_calls=True,
                )
                logger.info(f"Initial LLM Response: {llm_response}")

                message = llm_response["choices"][0]["message"]
                if not message.get("tool_calls"):
                    logger.warning("No tool calls in LLM response")
                    return llm_response

                # Call the tool using MCP client
                openai_tool = message["tool_calls"][0]
                call_result = await call_openai_tool(
                    session=session,
                    openai_tool=openai_tool,
                )
                logger.info(f"Tool call completed: {call_result}")

                # Update messages with tool result
                messages.append(message)
                messages.append(
                    {
                        "role": "tool",
                        "content": str(call_result.content[0].text),
                        "tool_call_id": openai_tool["id"],
                    }
                )
                logger.debug(
                    "Updated messages with tool result",
                    extra={"messages": messages},
                )

                # Second LLM call with tool result
                final_response = await litellm.acompletion(
                    model=model,
                    api_key=os.getenv("OPENAI_API_KEY"),
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                    # parallel_tool_calls=True,
                )

                logger.info(f"Final LLM Response: {final_response}")
                return final_response

            except Exception as e:
                logger.error(f"Error in execute_tool_call: {str(e)}")
                raise RuntimeError(f"Tool execution failed: {str(e)}")


# def execute_tool_call_sync(
#     server_path: str,
#     tool_call: Dict[str, Any],
#     task: str,
#     *args,
#     **kwargs,
# ) -> Dict[str, Any]:
#     """
#     Synchronous version of execute_tool_call that handles event loop management.

#     Args:
#         server_path (str): Path to the MCP server script
#         tool_call (Dict[str, Any]): The OpenAI tool call to execute
#         messages (List[Dict[str, Any]]): Current conversation messages

#     Returns:
#         Dict[str, Any]: Final LLM response after tool execution

#     Raises:
#         ValueError: If inputs are invalid
#         ConnectionError: If connection to server fails
#         RuntimeError: If event loop management fails
#     """
#     with get_or_create_event_loop() as loop:
#         try:
#             return loop.run_until_complete(
#                 execute_tool_call(
#                     server_path, tool_call, task, *args, **kwargs
#                 )
#             )
#         except Exception as e:
#             logger.error(f"Error in execute_tool_call_sync: {str(e)}")
#             raise
