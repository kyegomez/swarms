from swarms.utils.formatter import Formatter


def test_formatter():
    """Test the formatter with various markdown content."""
    formatter = Formatter(md=True)

    # Test 1: Basic markdown with headers
    content1 = """# Main Title

This is a paragraph with **bold** text and *italic* text.

## Section 1
- Item 1
- Item 2
- Item 3

### Subsection
This is another paragraph with `inline code`.
"""

    formatter.print_panel(
        content1, title="Test 1: Basic Markdown", style="bold blue"
    )

    # Test 2: Code blocks with syntax highlighting
    content2 = """## Code Examples

Here's a Python example:

```python
def hello_world():
    '''A simple hello world function.'''
    print("Hello, World!")
    return True
```

And here's some JavaScript:

```javascript
function greet(name) {
    console.log(`Hello, ${name}!`);
}
```

Plain text code block:

```
This is just plain text
without any syntax highlighting
```
"""

    formatter.print_panel(
        content2, title="Test 2: Code Blocks", style="bold green"
    )

    # Test 3: Mixed content
    content3 = """## Mixed Content Test

This paragraph includes **various** formatting options:
- Lists with `code`
- Links [like this](https://example.com)
- And more...

```python
# Python code with comments
class Example:
    def __init__(self, name):
        self.name = name
    
    def greet(self):
        return f"Hello, {self.name}!"
```

### Table Example

| Column 1 | Column 2 | Column 3 |
|----------|----------|----------|
| Data 1   | Data 2   | Data 3   |
| Data 4   | Data 5   | Data 6   |
"""

    formatter.print_panel(
        content3, title="Test 3: Mixed Content", style="bold magenta"
    )

    # Test 4: Edge cases
    content4 = """This content starts without a header

It should still be formatted correctly.

```
No language specified
```

Single line content."""

    formatter.print_panel(
        content4, title="Test 4: Edge Cases", style="bold yellow"
    )

    # Test 5: Empty content
    formatter.print_panel(
        "", title="Test 5: Empty Content", style="bold red"
    )

    # Test 6: Using print_markdown method
    content6 = """# Direct Markdown Rendering

This uses the `print_markdown` method directly.

```python
# Syntax highlighted code
result = 42 * 2
print(f"The answer is {result}")
```
"""

    formatter.print_markdown(
        content6, title="Test 6: Direct Markdown", border_style="cyan"
    )


def test_tool_call_visualization():
    """Test tool call visualization methods."""
    formatter = Formatter(md=False)

    # Test print_tool_call for regular function
    print("\n--- Test: Tool Call (function) ---")
    formatter.print_tool_call(
        tool_name="get_weather",
        tool_args={"location": "New York", "units": "celsius"},
        tool_type="function",
    )

    # Test print_tool_call for MCP tool
    print("\n--- Test: Tool Call (MCP) ---")
    formatter.print_tool_call(
        tool_name="search_documents",
        tool_args={"query": "quarterly report", "limit": 10},
        tool_type="mcp",
    )

    # Test print_tool_call with no arguments
    print("\n--- Test: Tool Call (no args) ---")
    formatter.print_tool_call(
        tool_name="get_current_time",
        tool_args={},
        tool_type="function",
    )


def test_tool_result_visualization():
    """Test tool result visualization methods."""
    formatter = Formatter(md=False)

    # Test successful tool result
    print("\n--- Test: Tool Result (success) ---")
    formatter.print_tool_result(
        tool_name="get_weather",
        status="success",
        output={"temperature": 22, "conditions": "sunny", "humidity": 45},
        duration=0.35,
        show_output=True,
    )

    # Test error tool result
    print("\n--- Test: Tool Result (error) ---")
    formatter.print_tool_result(
        tool_name="database_query",
        status="error",
        output="Connection timeout: unable to reach database server",
        duration=5.02,
        show_output=True,
    )

    # Test tool result without output
    print("\n--- Test: Tool Result (no output) ---")
    formatter.print_tool_result(
        tool_name="send_notification",
        status="success",
        output="Notification sent successfully",
        duration=0.12,
        show_output=False,
    )


def test_mcp_tool_result():
    """Test MCP tool result visualization."""
    formatter = Formatter(md=False)

    print("\n--- Test: MCP Tool Result ---")
    formatter.print_mcp_tool_result(
        output={
            "results": [
                {"id": 1, "title": "Q1 Report", "score": 0.95},
                {"id": 2, "title": "Q2 Report", "score": 0.87},
            ],
            "total": 2,
        },
        duration=1.23,
    )


def test_tool_execution_summary():
    """Test tool execution summary table."""
    formatter = Formatter(md=False)

    executions = [
        {
            "tool_name": "get_weather",
            "status": "success",
            "duration": 0.35,
            "tokens": 150,
            "result_preview": "Temperature: 22C, sunny",
        },
        {
            "tool_name": "search_documents",
            "status": "success",
            "duration": 1.23,
            "tokens": 320,
            "result_preview": "Found 5 matching documents",
        },
        {
            "tool_name": "database_query",
            "status": "error",
            "duration": 5.02,
            "tokens": 0,
            "result_preview": "Connection timeout",
        },
        {
            "tool_name": "MCP:file_reader",
            "status": "success",
            "duration": 0.08,
            "tokens": 45,
            "result_preview": "Read 1024 bytes from config.json",
        },
    ]

    print("\n--- Test: Tool Execution Summary ---")
    formatter.print_tool_execution_summary(
        executions, title="Test Run Summary"
    )


def test_token_usage():
    """Test token usage display."""
    formatter = Formatter(md=False)

    print("\n--- Test: Token Usage ---")
    formatter.print_token_usage(
        prompt_tokens=1250,
        completion_tokens=580,
        total_tokens=1830,
        model="gpt-4",
    )

    # Test without model
    print("\n--- Test: Token Usage (no model) ---")
    formatter.print_token_usage(
        prompt_tokens=500,
        completion_tokens=200,
        total_tokens=700,
    )


def test_progress_creation():
    """Test progress indicator creation."""
    formatter = Formatter(md=False)

    # Test creating progress for function tool
    print("\n--- Test: Progress Creation (function) ---")
    progress = formatter.create_tool_progress(
        tool_name="analyze_data", tool_type="function"
    )
    assert progress is not None

    # Test creating progress for MCP tool
    print("\n--- Test: Progress Creation (MCP) ---")
    progress = formatter.create_tool_progress(
        tool_name="fetch_remote", tool_type="mcp"
    )
    assert progress is not None

    print("[OK] Progress creation tests passed")


def test_ascii_only_output():
    """Test that output uses ASCII only (no emojis)."""
    from io import StringIO
    from rich.console import Console

    # Create formatter with a string buffer to capture output
    buffer = StringIO()
    console = Console(file=buffer, force_terminal=True)
    formatter = Formatter(md=False)
    formatter.console = console

    # Test tool result
    formatter.print_tool_result(
        tool_name="test_tool",
        status="success",
        output="test output",
        duration=1.0,
        show_output=True,
    )

    output = buffer.getvalue()

    # Check that common emojis are not present
    emoji_patterns = [
        "\U0001f600",  # grinning face
        "\u2713",  # check mark
        "\u2717",  # cross mark
        "\U0001f4cb",  # clipboard
        "\U0001f534",  # red circle
        "\U0001f7e0",  # orange circle
        "\U0001f7e1",  # yellow circle
        "\U0001f7e2",  # green circle
        "\U0001f3af",  # target
        "\U0001f916",  # robot
        "\u2705",  # white check mark
        "\u274c",  # cross mark
    ]

    for emoji in emoji_patterns:
        assert (
            emoji not in output
        ), f"Found emoji {repr(emoji)} in output"

    print("[OK] ASCII-only output test passed")


if __name__ == "__main__":
    test_formatter()
    test_tool_call_visualization()
    test_tool_result_visualization()
    test_mcp_tool_result()
    test_tool_execution_summary()
    test_token_usage()
    test_progress_creation()
    test_ascii_only_output()
