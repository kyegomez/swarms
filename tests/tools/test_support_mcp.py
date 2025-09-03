import asyncio
import traceback
from datetime import datetime

from loguru import logger

# Import the functions to test (assuming they're in a module called mcp_client)
# from mcp_client import *  # Replace with actual import
from swarms.tools.mcp_client_tools import (
    MCPConnectionError,
    MCPValidationError,
    _create_server_tool_mapping_async,
    _fetch_tools_for_server,
    _get_function_arguments,
    aget_mcp_tools,
    auto_detect_transport,
    connect_to_mcp_server,
    execute_multiple_tools_on_multiple_mcp_servers,
    execute_multiple_tools_on_multiple_mcp_servers_sync,
    execute_tool_call_simple,
    get_mcp_tools_sync,
    get_tools_for_multiple_mcp_servers,
    transform_mcp_tool_to_openai_tool,
    transform_openai_tool_call_request_to_mcp_tool_call_request,
)

# Configure logging
logger.add("test_results.log", rotation="10 MB", level="DEBUG")

# Test configuration
TEST_CONFIG = {
    "server_url": "http://localhost:8080/mcp",
    "transport": "streamable_http",
    "timeout": 10,
}

# Test results storage
test_results = []


def log_test_result(
    test_name: str, status: str, message: str = "", error: str = ""
):
    """Log test result and add to results list"""
    result = {
        "test_name": test_name,
        "status": status,
        "message": message,
        "error": error,
        "timestamp": datetime.now().isoformat(),
    }
    test_results.append(result)

    if status == "PASS":
        logger.success(f"✓ {test_name}: {message}")
    elif status == "FAIL":
        logger.error(f"✗ {test_name}: {error}")
    else:
        logger.info(f"~ {test_name}: {message}")


def test_transform_mcp_tool_to_openai_tool():
    """Test MCP tool to OpenAI tool transformation"""
    test_name = "test_transform_mcp_tool_to_openai_tool"

    try:
        # Create mock MCP tool
        class MockMCPTool:
            def __init__(self, name, description, input_schema):
                self.name = name
                self.description = description
                self.inputSchema = input_schema

        mock_tool = MockMCPTool(
            name="test_function",
            description="Test function description",
            input_schema={
                "type": "object",
                "properties": {"param1": {"type": "string"}},
            },
        )

        result = transform_mcp_tool_to_openai_tool(mock_tool)

        # Validate result structure
        assert result["type"] == "function"
        assert result["function"]["name"] == "test_function"
        assert (
            result["function"]["description"]
            == "Test function description"
        )
        assert result["function"]["parameters"]["type"] == "object"

        log_test_result(
            test_name,
            "PASS",
            "Successfully transformed MCP tool to OpenAI format",
        )

    except Exception as e:
        log_test_result(
            test_name,
            "FAIL",
            error=f"Failed to transform tool: {str(e)}",
        )


def test_get_function_arguments():
    """Test function argument extraction"""
    test_name = "test_get_function_arguments"

    try:
        # Test with dict arguments
        function_def = {
            "arguments": {"param1": "value1", "param2": "value2"}
        }
        result = _get_function_arguments(function_def)
        assert isinstance(result, dict)
        assert result["param1"] == "value1"

        # Test with string arguments
        function_def_str = {
            "arguments": '{"param1": "value1", "param2": "value2"}'
        }
        result_str = _get_function_arguments(function_def_str)
        assert isinstance(result_str, dict)
        assert result_str["param1"] == "value1"

        # Test with empty arguments
        function_def_empty = {}
        result_empty = _get_function_arguments(function_def_empty)
        assert result_empty == {}

        log_test_result(
            test_name,
            "PASS",
            "Successfully extracted function arguments in all formats",
        )

    except Exception as e:
        log_test_result(
            test_name,
            "FAIL",
            error=f"Failed to extract arguments: {str(e)}",
        )


def test_transform_openai_tool_call_request_to_mcp_tool_call_request():
    """Test OpenAI tool call to MCP tool call transformation"""
    test_name = "test_transform_openai_tool_call_request_to_mcp_tool_call_request"

    try:
        openai_tool = {
            "function": {
                "name": "test_function",
                "arguments": {"param1": "value1", "param2": "value2"},
            }
        }

        result = transform_openai_tool_call_request_to_mcp_tool_call_request(
            openai_tool
        )

        assert result.name == "test_function"
        assert result.arguments["param1"] == "value1"
        assert result.arguments["param2"] == "value2"

        log_test_result(
            test_name,
            "PASS",
            "Successfully transformed OpenAI tool call to MCP format",
        )

    except Exception as e:
        log_test_result(
            test_name,
            "FAIL",
            error=f"Failed to transform tool call: {str(e)}",
        )


def test_auto_detect_transport():
    """Test transport auto-detection"""
    test_name = "test_auto_detect_transport"

    try:
        # Test HTTP URL
        http_url = "http://localhost:8080/mcp"
        transport = auto_detect_transport(http_url)
        assert transport == "streamable_http"

        # Test HTTPS URL
        https_url = "https://example.com/mcp"
        transport = auto_detect_transport(https_url)
        assert transport == "streamable_http"

        # Test WebSocket URL
        ws_url = "ws://localhost:8080/mcp"
        transport = auto_detect_transport(ws_url)
        assert transport == "sse"

        # Test stdio
        stdio_url = "stdio://local"
        transport = auto_detect_transport(stdio_url)
        assert transport == "stdio"

        # Test unknown scheme
        unknown_url = "unknown://test"
        transport = auto_detect_transport(unknown_url)
        assert transport == "sse"  # Default

        log_test_result(
            test_name,
            "PASS",
            "Successfully auto-detected all transport types",
        )

    except Exception as e:
        log_test_result(
            test_name,
            "FAIL",
            error=f"Failed to auto-detect transport: {str(e)}",
        )


def test_connect_to_mcp_server():
    """Test MCP server connection configuration"""
    test_name = "test_connect_to_mcp_server"

    try:
        from swarms.schemas.mcp_schemas import MCPConnection

        # Create connection object
        connection = MCPConnection(
            url="http://localhost:8080/mcp",
            transport="streamable_http",
            timeout=10,
            headers={"Content-Type": "application/json"},
            authorization_token="test_token",
        )

        headers, timeout, transport, url = connect_to_mcp_server(
            connection
        )

        assert url == "http://localhost:8080/mcp"
        assert transport == "streamable_http"
        assert timeout == 10
        assert "Authorization" in headers
        assert headers["Authorization"] == "Bearer test_token"

        log_test_result(
            test_name,
            "PASS",
            "Successfully configured MCP server connection",
        )

    except Exception as e:
        log_test_result(
            test_name,
            "FAIL",
            error=f"Failed to configure connection: {str(e)}",
        )


async def test_aget_mcp_tools():
    """Test async MCP tools fetching"""
    test_name = "test_aget_mcp_tools"

    try:
        # This will attempt to connect to the actual server
        tools = await aget_mcp_tools(
            server_path=TEST_CONFIG["server_url"],
            format="openai",
            transport=TEST_CONFIG["transport"],
        )

        assert isinstance(tools, list)
        log_test_result(
            test_name,
            "PASS",
            f"Successfully fetched {len(tools)} tools from server",
        )

    except MCPConnectionError as e:
        log_test_result(
            test_name, "SKIP", f"Server not available: {str(e)}"
        )
    except Exception as e:
        log_test_result(
            test_name,
            "FAIL",
            error=f"Failed to fetch tools: {str(e)}",
        )


def test_get_mcp_tools_sync():
    """Test synchronous MCP tools fetching"""
    test_name = "test_get_mcp_tools_sync"

    try:
        tools = get_mcp_tools_sync(
            server_path=TEST_CONFIG["server_url"],
            format="openai",
            transport=TEST_CONFIG["transport"],
        )

        assert isinstance(tools, list)
        log_test_result(
            test_name,
            "PASS",
            f"Successfully fetched {len(tools)} tools synchronously",
        )

    except MCPConnectionError as e:
        log_test_result(
            test_name, "SKIP", f"Server not available: {str(e)}"
        )
    except Exception as e:
        log_test_result(
            test_name,
            "FAIL",
            error=f"Failed to fetch tools sync: {str(e)}",
        )


def test_fetch_tools_for_server():
    """Test fetching tools for a single server"""
    test_name = "test_fetch_tools_for_server"

    try:
        tools = _fetch_tools_for_server(
            url=TEST_CONFIG["server_url"],
            format="openai",
            transport=TEST_CONFIG["transport"],
        )

        assert isinstance(tools, list)
        log_test_result(
            test_name,
            "PASS",
            f"Successfully fetched tools for single server: {len(tools)} tools",
        )

    except MCPConnectionError as e:
        log_test_result(
            test_name, "SKIP", f"Server not available: {str(e)}"
        )
    except Exception as e:
        log_test_result(
            test_name,
            "FAIL",
            error=f"Failed to fetch tools for server: {str(e)}",
        )


def test_get_tools_for_multiple_mcp_servers():
    """Test fetching tools from multiple servers"""
    test_name = "test_get_tools_for_multiple_mcp_servers"

    try:
        urls = [
            TEST_CONFIG["server_url"]
        ]  # Using single server for testing

        tools = get_tools_for_multiple_mcp_servers(
            urls=urls,
            format="openai",
            transport=TEST_CONFIG["transport"],
            max_workers=2,
        )

        assert isinstance(tools, list)
        log_test_result(
            test_name,
            "PASS",
            f"Successfully fetched tools from multiple servers: {len(tools)} tools",
        )

    except MCPConnectionError as e:
        log_test_result(
            test_name, "SKIP", f"Server not available: {str(e)}"
        )
    except Exception as e:
        log_test_result(
            test_name,
            "FAIL",
            error=f"Failed to fetch tools from multiple servers: {str(e)}",
        )


async def test_execute_tool_call_simple():
    """Test simple tool execution"""
    test_name = "test_execute_tool_call_simple"

    try:
        # First try to get available tools
        try:
            tools = await aget_mcp_tools(
                server_path=TEST_CONFIG["server_url"],
                format="openai",
                transport=TEST_CONFIG["transport"],
            )

            if not tools:
                log_test_result(
                    test_name,
                    "SKIP",
                    "No tools available for testing",
                )
                return

            # Use the first available tool for testing
            first_tool = tools[0]
            tool_name = first_tool["function"]["name"]

            # Create a basic tool call request
            tool_call_request = {
                "function": {
                    "name": tool_name,
                    "arguments": {},  # Basic empty arguments
                }
            }

            result = await execute_tool_call_simple(
                response=tool_call_request,
                server_path=TEST_CONFIG["server_url"],
                transport=TEST_CONFIG["transport"],
                output_type="str",
            )

            assert result is not None
            log_test_result(
                test_name,
                "PASS",
                f"Successfully executed tool call for {tool_name}",
            )

        except MCPConnectionError:
            log_test_result(
                test_name,
                "SKIP",
                "Server not available for tool execution test",
            )

    except Exception as e:
        log_test_result(
            test_name,
            "FAIL",
            error=f"Failed to execute tool call: {str(e)}",
        )


async def test_create_server_tool_mapping():
    """Test server tool mapping creation"""
    test_name = "test_create_server_tool_mapping"

    try:
        urls = [TEST_CONFIG["server_url"]]

        mapping = await _create_server_tool_mapping_async(
            urls=urls,
            format="openai",
            transport=TEST_CONFIG["transport"],
        )

        assert isinstance(mapping, dict)
        log_test_result(
            test_name,
            "PASS",
            f"Successfully created server tool mapping with {len(mapping)} functions",
        )

    except MCPConnectionError as e:
        log_test_result(
            test_name, "SKIP", f"Server not available: {str(e)}"
        )
    except Exception as e:
        log_test_result(
            test_name,
            "FAIL",
            error=f"Failed to create server tool mapping: {str(e)}",
        )


async def test_execute_multiple_tools_on_multiple_servers():
    """Test executing multiple tools across servers"""
    test_name = "test_execute_multiple_tools_on_multiple_servers"

    try:
        urls = [TEST_CONFIG["server_url"]]

        # First get available tools
        try:
            tools = await aget_mcp_tools(
                server_path=TEST_CONFIG["server_url"],
                format="openai",
                transport=TEST_CONFIG["transport"],
            )

            if not tools:
                log_test_result(
                    test_name,
                    "SKIP",
                    "No tools available for testing",
                )
                return

            # Create test requests using available tools
            responses = []
            for tool in tools[:2]:  # Test with first 2 tools
                tool_call = {
                    "function": {
                        "name": tool["function"]["name"],
                        "arguments": {},
                    }
                }
                responses.append(tool_call)

            if not responses:
                log_test_result(
                    test_name,
                    "SKIP",
                    "No suitable tools found for testing",
                )
                return

            results = (
                await execute_multiple_tools_on_multiple_mcp_servers(
                    responses=responses,
                    urls=urls,
                    transport=TEST_CONFIG["transport"],
                    max_concurrent=2,
                )
            )

            assert isinstance(results, list)
            log_test_result(
                test_name,
                "PASS",
                f"Successfully executed {len(results)} tool calls",
            )

        except MCPConnectionError:
            log_test_result(
                test_name,
                "SKIP",
                "Server not available for multiple tool execution test",
            )

    except Exception as e:
        log_test_result(
            test_name,
            "FAIL",
            error=f"Failed to execute multiple tools: {str(e)}",
        )


def test_execute_multiple_tools_sync():
    """Test synchronous multiple tool execution"""
    test_name = "test_execute_multiple_tools_sync"

    try:
        urls = [TEST_CONFIG["server_url"]]

        # Create minimal test requests
        responses = [
            {
                "function": {
                    "name": "test_function",  # This will likely fail but tests the sync wrapper
                    "arguments": {},
                }
            }
        ]

        results = execute_multiple_tools_on_multiple_mcp_servers_sync(
            responses=responses,
            urls=urls,
            transport=TEST_CONFIG["transport"],
            max_concurrent=1,
        )

        assert isinstance(results, list)
        log_test_result(
            test_name,
            "PASS",
            f"Successfully ran sync multiple tools execution (got {len(results)} results)",
        )

    except MCPConnectionError as e:
        log_test_result(
            test_name, "SKIP", f"Server not available: {str(e)}"
        )
    except Exception as e:
        log_test_result(
            test_name,
            "FAIL",
            error=f"Failed sync multiple tools execution: {str(e)}",
        )


def test_error_handling():
    """Test error handling for various scenarios"""
    test_name = "test_error_handling"

    try:
        # Test invalid server URL
        try:
            get_mcp_tools_sync(
                server_path="http://invalid-url:99999/mcp",
                transport="streamable_http",
            )
            assert False, "Should have raised an exception"
        except MCPConnectionError:
            pass  # Expected

        # Test invalid connection object
        try:
            connect_to_mcp_server("invalid_connection")
            assert False, "Should have raised an exception"
        except MCPValidationError:
            pass  # Expected

        # Test invalid transport detection
        transport = auto_detect_transport("")
        assert transport == "sse"  # Should default to sse

        log_test_result(
            test_name, "PASS", "All error handling tests passed"
        )

    except Exception as e:
        log_test_result(
            test_name,
            "FAIL",
            error=f"Error handling test failed: {str(e)}",
        )


async def run_all_tests():
    """Run all test functions"""
    logger.info("Starting MCP unit tests...")

    # Synchronous tests
    test_transform_mcp_tool_to_openai_tool()
    test_get_function_arguments()
    test_transform_openai_tool_call_request_to_mcp_tool_call_request()
    test_auto_detect_transport()
    test_connect_to_mcp_server()
    test_get_mcp_tools_sync()
    test_fetch_tools_for_server()
    test_get_tools_for_multiple_mcp_servers()
    test_execute_multiple_tools_sync()
    test_error_handling()

    # Asynchronous tests
    await test_aget_mcp_tools()
    await test_execute_tool_call_simple()
    await test_create_server_tool_mapping()
    await test_execute_multiple_tools_on_multiple_servers()

    logger.info(
        f"Completed all tests. Total tests run: {len(test_results)}"
    )


def generate_markdown_report():
    """Generate markdown report of test results"""

    passed_tests = [r for r in test_results if r["status"] == "PASS"]
    failed_tests = [r for r in test_results if r["status"] == "FAIL"]
    skipped_tests = [r for r in test_results if r["status"] == "SKIP"]

    markdown_content = f"""# MCP Unit Test Results

## Summary
- **Total Tests**: {len(test_results)}
- **Passed**: {len(passed_tests)}
- **Failed**: {len(failed_tests)}
- **Skipped**: {len(skipped_tests)}
- **Success Rate**: {(len(passed_tests)/len(test_results)*100):.1f}%

## Test Configuration
- **Server URL**: {TEST_CONFIG["server_url"]}
- **Transport**: {TEST_CONFIG["transport"]}
- **Timeout**: {TEST_CONFIG["timeout"]}s

## Test Results

### ✅ Passed Tests ({len(passed_tests)})
"""

    for test in passed_tests:
        markdown_content += (
            f"- **{test['test_name']}**: {test['message']}\n"
        )

    if failed_tests:
        markdown_content += (
            f"\n### ❌ Failed Tests ({len(failed_tests)})\n"
        )
        for test in failed_tests:
            markdown_content += (
                f"- **{test['test_name']}**: {test['error']}\n"
            )

    if skipped_tests:
        markdown_content += (
            f"\n### ⏭️ Skipped Tests ({len(skipped_tests)})\n"
        )
        for test in skipped_tests:
            markdown_content += (
                f"- **{test['test_name']}**: {test['message']}\n"
            )

    markdown_content += """
## Detailed Results

| Test Name | Status | Message/Error | Timestamp |
|-----------|---------|---------------|-----------|
"""

    for test in test_results:
        status_emoji = {"PASS": "✅", "FAIL": "❌", "SKIP": "⏭️"}.get(
            test["status"], "❓"
        )
        message = test.get("message") or test.get("error", "")
        markdown_content += f"| {test['test_name']} | {status_emoji} {test['status']} | {message} | {test['timestamp']} |\n"

    markdown_content += f"""
## Notes
- Tests marked as SKIP typically indicate the MCP server was not available at {TEST_CONFIG["server_url"]}
- Connection tests may fail if the server is not running or configured differently
- Tool execution tests depend on the specific tools available on the server

Generated at: {datetime.now().isoformat()}
"""

    return markdown_content


async def main():
    """Main test runner"""
    try:
        await run_all_tests()

        # Generate and save markdown report
        markdown_report = generate_markdown_report()

        with open("mcp_test_results.md", "w") as f:
            f.write(markdown_report)

        logger.info("Test results saved to mcp_test_results.md")

        # Print summary
        passed = len(
            [r for r in test_results if r["status"] == "PASS"]
        )
        failed = len(
            [r for r in test_results if r["status"] == "FAIL"]
        )
        skipped = len(
            [r for r in test_results if r["status"] == "SKIP"]
        )

        print(f"\n{'='*50}")
        print("TEST SUMMARY")
        print(f"{'='*50}")
        print(f"Total Tests: {len(test_results)}")
        print(f"Passed: {passed}")
        print(f"Failed: {failed}")
        print(f"Skipped: {skipped}")
        print(f"Success Rate: {(passed/len(test_results)*100):.1f}%")
        print(f"{'='*50}")

    except Exception as e:
        logger.error(f"Error running tests: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    asyncio.run(main())
