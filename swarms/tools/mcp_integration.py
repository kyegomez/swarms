from contextlib import AsyncExitStack
from types import TracebackType
from typing import (
    Any,
    Callable,
    Coroutine,
    List,
    Literal,
    Optional,
    TypedDict,
    cast,
)

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.types import (
    CallToolResult,
    EmbeddedResource,
    ImageContent,
    PromptMessage,
    TextContent,
)
from mcp.types import (
    Tool as MCPTool,
)


def convert_mcp_prompt_message_to_message(
    message: PromptMessage,
) -> str:
    """Convert an MCP prompt message to a string message.

    Args:
        message: MCP prompt message to convert

    Returns:
        a string message
    """
    if message.content.type == "text":
        if message.role == "user":
            return str(message.content.text)
        elif message.role == "assistant":
            return str(
                message.content.text
            )  # Fixed attribute name from str to text
        else:
            raise ValueError(
                f"Unsupported prompt message role: {message.role}"
            )

    raise ValueError(
        f"Unsupported prompt message content type: {message.content.type}"
    )


async def load_mcp_prompt(
    session: ClientSession,
    name: str,
    arguments: Optional[dict[str, Any]] = None,
) -> List[str]:
    """Load MCP prompt and convert to messages."""
    response = await session.get_prompt(name, arguments)

    return [
        convert_mcp_prompt_message_to_message(message)
        for message in response.messages
    ]


DEFAULT_ENCODING = "utf-8"
DEFAULT_ENCODING_ERROR_HANDLER = "strict"

DEFAULT_HTTP_TIMEOUT = 5
DEFAULT_SSE_READ_TIMEOUT = 60 * 5


class StdioConnection(TypedDict):
    transport: Literal["stdio"]

    command: str
    """The executable to run to start the server."""

    args: list[str]
    """Command line arguments to pass to the executable."""

    env: dict[str, str] | None
    """The environment to use when spawning the process."""

    encoding: str
    """The text encoding used when sending/receiving messages to the server."""

    encoding_error_handler: Literal["strict", "ignore", "replace"]
    """
    The text encoding error handler.

    See https://docs.python.org/3/library/codecs.html#codec-base-classes for
    explanations of possible values
    """


class SSEConnection(TypedDict):
    transport: Literal["sse"]

    url: str
    """The URL of the SSE endpoint to connect to."""

    headers: dict[str, Any] | None
    """HTTP headers to send to the SSE endpoint"""

    timeout: float
    """HTTP timeout"""

    sse_read_timeout: float
    """SSE read timeout"""


NonTextContent = ImageContent | EmbeddedResource


def _convert_call_tool_result(
    call_tool_result: CallToolResult,
) -> tuple[str | list[str], list[NonTextContent] | None]:
    text_contents: list[TextContent] = []
    non_text_contents = []
    for content in call_tool_result.content:
        if isinstance(content, TextContent):
            text_contents.append(content)
        else:
            non_text_contents.append(content)

    tool_content: str | list[str] = [
        content.text for content in text_contents
    ]
    if len(text_contents) == 1:
        tool_content = tool_content[0]

    if call_tool_result.isError:
        raise ValueError("Error calling tool")

    return tool_content, non_text_contents or None


def convert_mcp_tool_to_function(
    session: ClientSession,
    tool: MCPTool,
) -> Callable[
    ...,
    Coroutine[
        Any, Any, tuple[str | list[str], list[NonTextContent] | None]
    ],
]:
    """Convert an MCP tool to a callable function.

    NOTE: this tool can be executed only in a context of an active MCP client session.

    Args:
        session: MCP client session
        tool: MCP tool to convert

    Returns:
        a callable function
    """

    async def call_tool(
        **arguments: dict[str, Any],
    ) -> tuple[str | list[str], list[NonTextContent] | None]:
        """Execute the tool with the given arguments."""
        call_tool_result = await session.call_tool(
            tool.name, arguments
        )
        return _convert_call_tool_result(call_tool_result)

    # Add metadata as attributes to the function
    call_tool.__name__ = tool.name
    call_tool.__doc__ = tool.description or ""
    call_tool.schema = tool.inputSchema

    return call_tool


async def load_mcp_tools(session: ClientSession) -> list[Callable]:
    """Load all available MCP tools and convert them to callable functions."""
    tools = await session.list_tools()
    return [
        convert_mcp_tool_to_function(session, tool)
        for tool in tools.tools
    ]


class MultiServerMCPClient:
    """Client for connecting to multiple MCP servers and loading tools from them."""

    def __init__(
        self,
        connections: dict[
            str, StdioConnection | SSEConnection
        ] = None,
    ) -> None:
        """Initialize a MultiServerMCPClient with MCP servers connections.

        Args:
            connections: A dictionary mapping server names to connection configurations.
                Each configuration can be either a StdioConnection or SSEConnection.
                If None, no initial connections are established.

        Example:

            ```python
            async with MultiServerMCPClient(
                {
                    "math": {
                        "command": "python",
                        # Make sure to update to the full absolute path to your math_server.py file
                        "args": ["/path/to/math_server.py"],
                        "transport": "stdio",
                    },
                    "weather": {
                        # make sure you start your weather server on port 8000
                        "url": "http://localhost:8000/sse",
                        "transport": "sse",
                    }
                }
            ) as client:
                all_tools = client.get_tools()
                ...
            ```
        """
        self.connections = connections
        self.exit_stack = AsyncExitStack()
        self.sessions: dict[str, ClientSession] = {}
        self.server_name_to_tools: dict[str, list[Callable]] = {}

    async def _initialize_session_and_load_tools(
        self, server_name: str, session: ClientSession
    ) -> None:
        """Initialize a session and load tools from it.

        Args:
            server_name: Name to identify this server connection
            session: The ClientSession to initialize
        """
        # Initialize the session
        await session.initialize()
        self.sessions[server_name] = session

        # Load tools from this server
        server_tools = await load_mcp_tools(session)
        self.server_name_to_tools[server_name] = server_tools

    async def connect_to_server(
        self,
        server_name: str,
        *,
        transport: Literal["stdio", "sse"] = "stdio",
        **kwargs,
    ) -> None:
        """Connect to an MCP server using either stdio or SSE.

        This is a generic method that calls either connect_to_server_via_stdio or connect_to_server_via_sse
        based on the provided transport parameter.

        Args:
            server_name: Name to identify this server connection
            transport: Type of transport to use ("stdio" or "sse"), defaults to "stdio"
            **kwargs: Additional arguments to pass to the specific connection method

        Raises:
            ValueError: If transport is not recognized
            ValueError: If required parameters for the specified transport are missing
        """
        if transport == "sse":
            if "url" not in kwargs:
                raise ValueError(
                    "'url' parameter is required for SSE connection"
                )
            await self.connect_to_server_via_sse(
                server_name,
                url=kwargs["url"],
                headers=kwargs.get("headers"),
                timeout=kwargs.get("timeout", DEFAULT_HTTP_TIMEOUT),
                sse_read_timeout=kwargs.get(
                    "sse_read_timeout", DEFAULT_SSE_READ_TIMEOUT
                ),
            )
        elif transport == "stdio":
            if "command" not in kwargs:
                raise ValueError(
                    "'command' parameter is required for stdio connection"
                )
            if "args" not in kwargs:
                raise ValueError(
                    "'args' parameter is required for stdio connection"
                )
            await self.connect_to_server_via_stdio(
                server_name,
                command=kwargs["command"],
                args=kwargs["args"],
                env=kwargs.get("env"),
                encoding=kwargs.get("encoding", DEFAULT_ENCODING),
                encoding_error_handler=kwargs.get(
                    "encoding_error_handler",
                    DEFAULT_ENCODING_ERROR_HANDLER,
                ),
            )
        else:
            raise ValueError(
                f"Unsupported transport: {transport}. Must be 'stdio' or 'sse'"
            )

    async def connect_to_server_via_stdio(
        self,
        server_name: str,
        *,
        command: str,
        args: list[str],
        env: dict[str, str] | None = None,
        encoding: str = DEFAULT_ENCODING,
        encoding_error_handler: Literal[
            "strict", "ignore", "replace"
        ] = DEFAULT_ENCODING_ERROR_HANDLER,
    ) -> None:
        """Connect to a specific MCP server using stdio

        Args:
            server_name: Name to identify this server connection
            command: Command to execute
            args: Arguments for the command
            env: Environment variables for the command
            encoding: Character encoding
            encoding_error_handler: How to handle encoding errors
        """
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env,
            encoding=encoding,
            encoding_error_handler=encoding_error_handler,
        )

        # Create and store the connection
        stdio_transport = await self.exit_stack.enter_async_context(
            stdio_client(server_params)
        )
        read, write = stdio_transport
        session = cast(
            ClientSession,
            await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            ),
        )

        await self._initialize_session_and_load_tools(
            server_name, session
        )

    async def connect_to_server_via_sse(
        self,
        server_name: str,
        *,
        url: str,
        headers: dict[str, Any] | None = None,
        timeout: float = DEFAULT_HTTP_TIMEOUT,
        sse_read_timeout: float = DEFAULT_SSE_READ_TIMEOUT,
    ) -> None:
        """Connect to a specific MCP server using SSE

        Args:
            server_name: Name to identify this server connection
            url: URL of the SSE server
            headers: HTTP headers to send to the SSE endpoint
            timeout: HTTP timeout
            sse_read_timeout: SSE read timeout
        """
        # Create and store the connection
        sse_transport = await self.exit_stack.enter_async_context(
            sse_client(url, headers, timeout, sse_read_timeout)
        )
        read, write = sse_transport
        session = cast(
            ClientSession,
            await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            ),
        )

        await self._initialize_session_and_load_tools(
            server_name, session
        )

    def get_tools(self) -> list[Callable]:
        """Get a list of all tools from all connected servers."""
        all_tools: list[Callable] = []
        for server_tools in self.server_name_to_tools.values():
            all_tools.extend(server_tools)
        return all_tools

    async def get_prompt(
        self,
        server_name: str,
        prompt_name: str,
        arguments: Optional[dict[str, Any]] = None,
    ) -> List[str]:
        """Get a prompt from a given MCP server."""
        session = self.sessions[server_name]
        return await load_mcp_prompt(session, prompt_name, arguments)

    async def __aenter__(self) -> "MultiServerMCPClient":
        try:
            connections = self.connections or {}
            for server_name, connection in connections.items():
                connection_dict = connection.copy()
                transport = connection_dict.pop("transport")
                if transport == "stdio":
                    await self.connect_to_server_via_stdio(
                        server_name, **connection_dict
                    )
                elif transport == "sse":
                    await self.connect_to_server_via_sse(
                        server_name, **connection_dict
                    )
                else:
                    raise ValueError(
                        f"Unsupported transport: {transport}. Must be 'stdio' or 'sse'"
                    )
            return self
        except Exception:
            await self.exit_stack.aclose()
            raise

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        await self.exit_stack.aclose()


# #!/usr/bin/env python3
# import asyncio
# import os
# import json
# from typing import List, Any, Callable

# # # Import our MCP client module
# # from mcp_client import MultiServerMCPClient


# async def main():
#     """Test script for demonstrating MCP client usage."""
#     print("Starting MCP Client test...")

#     # Create a connection to multiple MCP servers
#     # You'll need to update these paths to match your setup
#     async with MultiServerMCPClient(
#         {
#             "math": {
#                 "transport": "stdio",
#                 "command": "python",
#                 "args": ["/path/to/math_server.py"],
#                 "env": {"DEBUG": "1"},
#             },
#             "search": {
#                 "transport": "sse",
#                 "url": "http://localhost:8000/sse",
#                 "headers": {
#                     "Authorization": f"Bearer {os.environ.get('API_KEY', '')}"
#                 },
#             },
#         }
#     ) as client:
#         # Get all available tools
#         tools = client.get_tools()
#         print(f"Found {len(tools)} tools across all servers")

#         # Print tool information
#         for i, tool in enumerate(tools):
#             print(f"\nTool {i+1}: {tool.__name__}")
#             print(f"  Description: {tool.__doc__}")
#             if hasattr(tool, "schema") and tool.schema:
#                 print(
#                     f"  Schema: {json.dumps(tool.schema, indent=2)[:100]}..."
#                 )

#         # Example: Use a specific tool if available
#         calculator_tool = next(
#             (t for t in tools if t.__name__ == "calculator"), None
#         )
#         if calculator_tool:
#             print("\n\nTesting calculator tool:")
#             try:
#                 # Call the tool as an async function
#                 result, artifacts = await calculator_tool(
#                     expression="2 + 2 * 3"
#                 )
#                 print(f"  Calculator result: {result}")
#                 if artifacts:
#                     print(
#                         f"  With {len(artifacts)} additional artifacts"
#                     )
#             except Exception as e:
#                 print(f"  Error using calculator: {e}")

#         # Example: Load a prompt from a server
#         try:
#             print("\n\nTesting prompt loading:")
#             prompt_messages = await client.get_prompt(
#                 "math",
#                 "calculation_introduction",
#                 {"user_name": "Test User"},
#             )
#             print(
#                 f"  Loaded prompt with {len(prompt_messages)} messages:"
#             )
#             for i, msg in enumerate(prompt_messages):
#                 print(f"  Message {i+1}: {msg[:50]}...")
#         except Exception as e:
#             print(f"  Error loading prompt: {e}")


# async def create_custom_tool():
#     """Example of creating a custom tool function."""

#     # Define a tool function with metadata
#     async def add_numbers(a: float, b: float) -> tuple[str, None]:
#         """Add two numbers together."""
#         result = a + b
#         return f"The sum of {a} and {b} is {result}", None

#     # Add metadata to the function
#     add_numbers.__name__ = "add_numbers"
#     add_numbers.__doc__ = (
#         "Add two numbers together and return the result."
#     )
#     add_numbers.schema = {
#         "type": "object",
#         "properties": {
#             "a": {"type": "number", "description": "First number"},
#             "b": {"type": "number", "description": "Second number"},
#         },
#         "required": ["a", "b"],
#     }

#     # Use the tool
#     result, _ = await add_numbers(a=5, b=7)
#     print(f"\nCustom tool result: {result}")


# if __name__ == "__main__":
#     # Run both examples
#     loop = asyncio.get_event_loop()
#     loop.run_until_complete(main())
#     loop.run_until_complete(create_custom_tool())
