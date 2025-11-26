"""
TOON SDK Client for Token-Optimized Serialization

This module provides a client for interacting with TOON (Token-Oriented
Object Notation) SDK services, enabling 30-60% token reduction for LLM prompts.

Key Features:
    - Automatic JSON to TOON encoding/decoding
    - Schema-aware compression for optimal results
    - Retry logic with exponential backoff
    - Async and sync execution modes
    - OpenAI-compatible tool conversion
    - Batch processing support

References:
    - TOON Spec: https://github.com/toon-format
    - Integration Pattern: Similar to swarms/tools/mcp_client_tools.py
"""

import asyncio
import contextlib
import random
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps
from typing import Any, Dict, List, Literal, Optional, Union

import httpx
from loguru import logger
from openai.types.chat import ChatCompletionToolParam
from openai.types.shared_params.function_definition import (
    FunctionDefinition,
)

from swarms.schemas.toon_schemas import (
    TOONConnection,
    TOONRequest,
    TOONResponse,
    TOONSerializationOptions,
    TOONToolDefinition,
)
from swarms.utils.index import exists


# Custom Exceptions
class TOONError(Exception):
    """Base exception for TOON-related errors."""

    pass


class TOONConnectionError(TOONError):
    """Raised when there are issues connecting to TOON SDK."""

    pass


class TOONSerializationError(TOONError):
    """Raised when serialization/deserialization fails."""

    pass


class TOONValidationError(TOONError):
    """Raised when validation issues occur."""

    pass


class TOONExecutionError(TOONError):
    """Raised when execution issues occur."""

    pass


########################################################
# TOON Tool Transformation
########################################################


def transform_toon_tool_to_openai_tool(
    toon_tool: TOONToolDefinition,
    verbose: bool = False,
) -> ChatCompletionToolParam:
    """
    Convert a TOON tool definition to OpenAI tool format.

    Args:
        toon_tool: TOON tool definition object
        verbose: Enable verbose logging

    Returns:
        OpenAI-compatible ChatCompletionToolParam

    Examples:
        >>> tool_def = TOONToolDefinition(
        ...     name="get_weather",
        ...     description="Get weather data",
        ...     input_schema={"type": "object", "properties": {...}}
        ... )
        >>> openai_tool = transform_toon_tool_to_openai_tool(tool_def)
    """
    if verbose:
        logger.info(
            f"Transforming TOON tool '{toon_tool.name}' to OpenAI format"
        )

    return ChatCompletionToolParam(
        type="function",
        function=FunctionDefinition(
            name=toon_tool.name,
            description=toon_tool.description or "",
            parameters=toon_tool.input_schema or {},
            strict=False,
        ),
    )


########################################################
# TOON SDK Client
########################################################


class TOONSDKClient:
    """
    Client for interacting with TOON SDK services.

    This client handles encoding, decoding, validation, and tool
    management for TOON format, providing seamless integration
    with the Swarms framework.

    Attributes:
        connection: TOON connection configuration
        client: HTTP client for API requests
        tools: Registry of TOON tool definitions
        verbose: Enable verbose logging

    Examples:
        >>> connection = TOONConnection(
        ...     url="https://api.toon-format.com/v1",
        ...     api_key="toon_key_xxx"
        ... )
        >>> client = TOONSDKClient(connection=connection)
        >>> encoded = await client.encode({"user": "Alice", "age": 30})
    """

    def __init__(
        self,
        connection: TOONConnection,
        verbose: bool = True,
    ):
        """
        Initialize TOON SDK client.

        Args:
            connection: TOONConnection configuration
            verbose: Enable verbose logging
        """
        self.connection = connection
        self.verbose = verbose
        self.tools: Dict[str, TOONToolDefinition] = {}

        # Initialize HTTP client
        headers = connection.headers or {}
        if connection.api_key:
            headers["Authorization"] = f"Bearer {connection.api_key}"
        headers["Content-Type"] = "application/json"

        self.client = httpx.AsyncClient(
            base_url=connection.url,
            headers=headers,
            timeout=connection.timeout or 30,
        )

        if self.verbose:
            logger.info(
                f"Initialized TOON SDK client for {connection.url}"
            )

    async def __aenter__(self):
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.close()

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
        if self.verbose:
            logger.info("Closed TOON SDK client")

    async def encode(
        self,
        data: Union[Dict[str, Any], List[Any]],
        schema: Optional[Dict[str, Any]] = None,
        options: Optional[TOONSerializationOptions] = None,
    ) -> str:
        """
        Encode JSON data to TOON format.

        Args:
            data: JSON data to encode
            schema: Optional JSON Schema for optimization
            options: Serialization options

        Returns:
            TOON-formatted string

        Raises:
            TOONSerializationError: If encoding fails

        Examples:
            >>> data = {"user": "Alice", "age": 30, "city": "NYC"}
            >>> toon_str = await client.encode(data)
            >>> print(toon_str)  # "usr:Alice age:30 city:NYC"
        """
        try:
            request = TOONRequest(
                operation="encode",
                data=data,
                schema=schema,
                options=options,
                format=self.connection.serialization_format,
            )

            response = await self._make_request("/encode", request)

            if response.status != "success":
                raise TOONSerializationError(
                    f"Encoding failed: {response.errors}"
                )

            if self.verbose:
                logger.info(
                    f"Encoded data: {response.original_tokens} → {response.compressed_tokens} tokens "
                    f"({response.compression_ratio:.1%} compression)"
                )

            return response.result

        except Exception as e:
            logger.error(f"TOON encoding error: {e}")
            raise TOONSerializationError(
                f"Failed to encode data: {e}"
            ) from e

    async def decode(
        self,
        toon_data: str,
        schema: Optional[Dict[str, Any]] = None,
    ) -> Union[Dict[str, Any], List[Any]]:
        """
        Decode TOON format back to JSON.

        Args:
            toon_data: TOON-formatted string
            schema: Optional JSON Schema for validation

        Returns:
            Decoded JSON data

        Raises:
            TOONSerializationError: If decoding fails

        Examples:
            >>> toon_str = "usr:Alice age:30 city:NYC"
            >>> data = await client.decode(toon_str)
            >>> print(data)  # {"user": "Alice", "age": 30, "city": "NYC"}
        """
        try:
            request = TOONRequest(
                operation="decode",
                data=toon_data,
                schema=schema,
                format="json",
            )

            response = await self._make_request("/decode", request)

            if response.status != "success":
                raise TOONSerializationError(
                    f"Decoding failed: {response.errors}"
                )

            if self.verbose:
                logger.info("Successfully decoded TOON data")

            return response.result

        except Exception as e:
            logger.error(f"TOON decoding error: {e}")
            raise TOONSerializationError(
                f"Failed to decode data: {e}"
            ) from e

    async def validate(
        self,
        data: Union[Dict[str, Any], str],
        schema: Dict[str, Any],
    ) -> bool:
        """
        Validate data against a JSON Schema.

        Args:
            data: Data to validate (JSON or TOON format)
            schema: JSON Schema for validation

        Returns:
            True if valid, False otherwise

        Examples:
            >>> schema = {"type": "object", "properties": {...}}
            >>> is_valid = await client.validate(data, schema)
        """
        try:
            request = TOONRequest(
                operation="validate",
                data=data,
                schema=schema,
            )

            response = await self._make_request("/validate", request)

            if response.status == "success":
                if self.verbose:
                    logger.info("Validation passed")
                return True
            else:
                if self.verbose:
                    logger.warning(
                        f"Validation failed: {response.errors}"
                    )
                return False

        except Exception as e:
            logger.error(f"TOON validation error: {e}")
            return False

    async def batch_encode(
        self,
        data_list: List[Union[Dict[str, Any], List[Any]]],
        schema: Optional[Dict[str, Any]] = None,
        options: Optional[TOONSerializationOptions] = None,
    ) -> List[str]:
        """
        Encode multiple JSON objects to TOON format in batch.

        Args:
            data_list: List of JSON data objects
            schema: Optional JSON Schema for optimization
            options: Serialization options

        Returns:
            List of TOON-formatted strings

        Examples:
            >>> data_list = [
            ...     {"user": "Alice", "age": 30},
            ...     {"user": "Bob", "age": 25}
            ... ]
            >>> toon_list = await client.batch_encode(data_list)
        """
        tasks = [
            self.encode(data, schema, options) for data in data_list
        ]
        return await asyncio.gather(*tasks)

    async def batch_decode(
        self,
        toon_list: List[str],
        schema: Optional[Dict[str, Any]] = None,
    ) -> List[Union[Dict[str, Any], List[Any]]]:
        """
        Decode multiple TOON strings to JSON in batch.

        Args:
            toon_list: List of TOON-formatted strings
            schema: Optional JSON Schema for validation

        Returns:
            List of decoded JSON objects

        Examples:
            >>> toon_list = ["usr:Alice age:30", "usr:Bob age:25"]
            >>> data_list = await client.batch_decode(toon_list)
        """
        tasks = [self.decode(toon, schema) for toon in toon_list]
        return await asyncio.gather(*tasks)

    async def list_tools(self) -> List[TOONToolDefinition]:
        """
        List all available TOON tools.

        Returns:
            List of TOON tool definitions

        Examples:
            >>> tools = await client.list_tools()
            >>> for tool in tools:
            ...     print(tool.name, tool.description)
        """
        try:
            response = await self.client.get("/tools")
            response.raise_for_status()

            tools_data = response.json()
            self.tools = {
                tool["name"]: TOONToolDefinition(**tool)
                for tool in tools_data.get("tools", [])
            }

            if self.verbose:
                logger.info(
                    f"Found {len(self.tools)} TOON tools"
                )

            return list(self.tools.values())

        except Exception as e:
            logger.error(f"Failed to list TOON tools: {e}")
            raise TOONExecutionError(
                f"Failed to list tools: {e}"
            ) from e

    def get_tools_as_openai_format(
        self,
    ) -> List[ChatCompletionToolParam]:
        """
        Get all tools in OpenAI-compatible format.

        Returns:
            List of OpenAI ChatCompletionToolParam

        Examples:
            >>> openai_tools = client.get_tools_as_openai_format()
            >>> # Use with OpenAI API or Agent
        """
        return [
            transform_toon_tool_to_openai_tool(tool, self.verbose)
            for tool in self.tools.values()
        ]

    async def _make_request(
        self,
        endpoint: str,
        request: TOONRequest,
    ) -> TOONResponse:
        """
        Make an HTTP request to TOON SDK API.

        Args:
            endpoint: API endpoint path
            request: TOON request payload

        Returns:
            TOONResponse object

        Raises:
            TOONConnectionError: If request fails
        """
        max_retries = self.connection.max_retries or 3
        backoff = self.connection.retry_backoff or 2.0

        for attempt in range(max_retries):
            try:
                response = await self.client.post(
                    endpoint,
                    json=request.model_dump(exclude_none=True),
                )
                response.raise_for_status()

                response_data = response.json()
                return TOONResponse(**response_data)

            except httpx.HTTPStatusError as e:
                if attempt < max_retries - 1:
                    wait_time = backoff**attempt + random.uniform(
                        0, 1
                    )
                    if self.verbose:
                        logger.warning(
                            f"Request failed (attempt {attempt + 1}/{max_retries}), "
                            f"retrying in {wait_time:.2f}s: {e}"
                        )
                    await asyncio.sleep(wait_time)
                else:
                    raise TOONConnectionError(
                        f"Request failed after {max_retries} attempts: {e}"
                    ) from e

            except Exception as e:
                raise TOONConnectionError(
                    f"Request error: {e}"
                ) from e


########################################################
# Synchronous Wrapper Functions
########################################################


@contextlib.contextmanager
def get_or_create_event_loop():
    """
    Context manager to handle event loop creation and cleanup.

    Yields:
        Event loop to use
    """
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    try:
        yield loop
    finally:
        if loop != asyncio.get_event_loop() and not loop.is_running():
            if not loop.is_closed():
                loop.close()


def retry_with_backoff(retries=3, backoff_in_seconds=1):
    """
    Decorator for retrying async functions with exponential backoff.

    Args:
        retries: Number of retry attempts
        backoff_in_seconds: Initial backoff time

    Returns:
        Decorated async function with retry logic
    """

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
                            f"Failed after {retries} retries: {str(e)}\n{traceback.format_exc()}"
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


@retry_with_backoff(retries=3)
async def encode_with_toon(
    data: Union[Dict[str, Any], List[Any]],
    connection: Optional[TOONConnection] = None,
    schema: Optional[Dict[str, Any]] = None,
    options: Optional[TOONSerializationOptions] = None,
    verbose: bool = True,
) -> str:
    """
    Async function to encode JSON data to TOON format.

    Args:
        data: JSON data to encode
        connection: TOON connection configuration
        schema: Optional JSON Schema for optimization
        options: Serialization options
        verbose: Enable verbose logging

    Returns:
        TOON-formatted string

    Examples:
        >>> data = {"user": "Alice", "age": 30}
        >>> toon_str = await encode_with_toon(data, connection)
    """
    if verbose:
        logger.info("Encoding data with TOON SDK")

    async with TOONSDKClient(
        connection=connection, verbose=verbose
    ) as client:
        return await client.encode(data, schema, options)


def encode_with_toon_sync(
    data: Union[Dict[str, Any], List[Any]],
    connection: Optional[TOONConnection] = None,
    schema: Optional[Dict[str, Any]] = None,
    options: Optional[TOONSerializationOptions] = None,
    verbose: bool = True,
) -> str:
    """
    Synchronous wrapper for encode_with_toon.

    Args:
        data: JSON data to encode
        connection: TOON connection configuration
        schema: Optional JSON Schema for optimization
        options: Serialization options
        verbose: Enable verbose logging

    Returns:
        TOON-formatted string

    Examples:
        >>> data = {"user": "Alice", "age": 30}
        >>> toon_str = encode_with_toon_sync(data, connection)
    """
    with get_or_create_event_loop() as loop:
        try:
            return loop.run_until_complete(
                encode_with_toon(
                    data, connection, schema, options, verbose
                )
            )
        except Exception as e:
            logger.error(f"Sync encoding error: {e}")
            raise TOONExecutionError(
                f"Failed to encode data: {e}"
            ) from e


@retry_with_backoff(retries=3)
async def decode_with_toon(
    toon_data: str,
    connection: Optional[TOONConnection] = None,
    schema: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Union[Dict[str, Any], List[Any]]:
    """
    Async function to decode TOON format to JSON.

    Args:
        toon_data: TOON-formatted string
        connection: TOON connection configuration
        schema: Optional JSON Schema for validation
        verbose: Enable verbose logging

    Returns:
        Decoded JSON data

    Examples:
        >>> toon_str = "usr:Alice age:30"
        >>> data = await decode_with_toon(toon_str, connection)
    """
    if verbose:
        logger.info("Decoding TOON data")

    async with TOONSDKClient(
        connection=connection, verbose=verbose
    ) as client:
        return await client.decode(toon_data, schema)


def decode_with_toon_sync(
    toon_data: str,
    connection: Optional[TOONConnection] = None,
    schema: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> Union[Dict[str, Any], List[Any]]:
    """
    Synchronous wrapper for decode_with_toon.

    Args:
        toon_data: TOON-formatted string
        connection: TOON connection configuration
        schema: Optional JSON Schema for validation
        verbose: Enable verbose logging

    Returns:
        Decoded JSON data

    Examples:
        >>> toon_str = "usr:Alice age:30"
        >>> data = decode_with_toon_sync(toon_str, connection)
    """
    with get_or_create_event_loop() as loop:
        try:
            return loop.run_until_complete(
                decode_with_toon(toon_data, connection, schema, verbose)
            )
        except Exception as e:
            logger.error(f"Sync decoding error: {e}")
            raise TOONExecutionError(
                f"Failed to decode data: {e}"
            ) from e


async def get_toon_tools(
    connection: Optional[TOONConnection] = None,
    format: Literal["toon", "openai"] = "openai",
    verbose: bool = True,
) -> List[Union[TOONToolDefinition, ChatCompletionToolParam]]:
    """
    Fetch available TOON tools from the SDK.

    Args:
        connection: TOON connection configuration
        format: Output format ('toon' or 'openai')
        verbose: Enable verbose logging

    Returns:
        List of tools in specified format

    Examples:
        >>> tools = await get_toon_tools(connection, format="openai")
        >>> # Use with Agent
    """
    if verbose:
        logger.info(f"Fetching TOON tools in '{format}' format")

    async with TOONSDKClient(
        connection=connection, verbose=verbose
    ) as client:
        await client.list_tools()

        if format == "openai":
            return client.get_tools_as_openai_format()
        else:
            return list(client.tools.values())


def get_toon_tools_sync(
    connection: Optional[TOONConnection] = None,
    format: Literal["toon", "openai"] = "openai",
    verbose: bool = True,
) -> List[Union[TOONToolDefinition, ChatCompletionToolParam]]:
    """
    Synchronous wrapper for get_toon_tools.

    Args:
        connection: TOON connection configuration
        format: Output format ('toon' or 'openai')
        verbose: Enable verbose logging

    Returns:
        List of tools in specified format

    Examples:
        >>> tools = get_toon_tools_sync(connection, format="openai")
    """
    with get_or_create_event_loop() as loop:
        try:
            return loop.run_until_complete(
                get_toon_tools(connection, format, verbose)
            )
        except Exception as e:
            logger.error(f"Failed to fetch TOON tools: {e}")
            raise TOONExecutionError(
                f"Failed to fetch tools: {e}"
            ) from e


########################################################
# Batch Processing with ThreadPoolExecutor
########################################################


def batch_encode_parallel(
    data_list: List[Union[Dict[str, Any], List[Any]]],
    connection: Optional[TOONConnection] = None,
    schema: Optional[Dict[str, Any]] = None,
    options: Optional[TOONSerializationOptions] = None,
    max_workers: Optional[int] = None,
    verbose: bool = True,
) -> List[str]:
    """
    Encode multiple JSON objects in parallel.

    Args:
        data_list: List of JSON data objects
        connection: TOON connection configuration
        schema: Optional JSON Schema
        options: Serialization options
        max_workers: Max worker threads
        verbose: Enable verbose logging

    Returns:
        List of TOON-formatted strings

    Examples:
        >>> data_list = [{"user": "Alice"}, {"user": "Bob"}]
        >>> toon_list = batch_encode_parallel(data_list, connection)
    """
    if verbose:
        logger.info(f"Batch encoding {len(data_list)} items")

    max_workers = max_workers or min(
        32, len(data_list), (os.cpu_count() or 1) + 4
    )

    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                encode_with_toon_sync,
                data,
                connection,
                schema,
                options,
                verbose,
            ): i
            for i, data in enumerate(data_list)
        }

        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                logger.error(f"Batch encoding error: {e}")
                raise TOONExecutionError(
                    f"Batch encoding failed: {e}"
                ) from e

    return results
