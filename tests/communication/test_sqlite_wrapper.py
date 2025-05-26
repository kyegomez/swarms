import json
import datetime
import os
from typing import Dict, List, Any, Tuple
from loguru import logger
from swarms.communication.sqlite_wrap import (
    SQLiteConversation,
    Message,
    MessageType,
)
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


def print_test_header(test_name: str) -> None:
    """Print a formatted test header."""
    console.print(
        Panel(
            f"[bold blue]Running Test: {test_name}[/bold blue]",
            expand=False,
        )
    )


def print_test_result(
    test_name: str, success: bool, message: str, execution_time: float
) -> None:
    """Print a formatted test result."""
    status = (
        "[bold green]PASSED[/bold green]"
        if success
        else "[bold red]FAILED[/bold red]"
    )
    console.print(f"\n{status} - {test_name}")
    console.print(f"Message: {message}")
    console.print(f"Execution time: {execution_time:.3f} seconds\n")


def print_messages(
    messages: List[Dict], title: str = "Messages"
) -> None:
    """Print messages in a formatted table."""
    table = Table(title=title)
    table.add_column("Role", style="cyan")
    table.add_column("Content", style="green")
    table.add_column("Type", style="yellow")
    table.add_column("Timestamp", style="magenta")

    for msg in messages:
        content = str(msg.get("content", ""))
        if isinstance(content, (dict, list)):
            content = json.dumps(content)
        table.add_row(
            msg.get("role", ""),
            content,
            str(msg.get("message_type", "")),
            str(msg.get("timestamp", "")),
        )

    console.print(table)


def run_test(
    test_func: callable, *args, **kwargs
) -> Tuple[bool, str, float]:
    """
    Run a test function and return its results.

    Args:
        test_func: The test function to run
        *args: Arguments for the test function
        **kwargs: Keyword arguments for the test function

    Returns:
        Tuple[bool, str, float]: (success, message, execution_time)
    """
    start_time = datetime.datetime.now()
    try:
        result = test_func(*args, **kwargs)
        end_time = datetime.datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        return True, str(result), execution_time
    except Exception as e:
        end_time = datetime.datetime.now()
        execution_time = (end_time - start_time).total_seconds()
        return False, str(e), execution_time


def test_basic_conversation() -> bool:
    """Test basic conversation operations."""
    print_test_header("Basic Conversation Test")

    db_path = "test_conversations.db"
    conversation = SQLiteConversation(db_path=db_path)

    # Test adding messages
    console.print("\n[bold]Adding messages...[/bold]")
    conversation.add("user", "Hello")
    conversation.add("assistant", "Hi there!")

    # Test getting messages
    console.print("\n[bold]Retrieved messages:[/bold]")
    messages = conversation.get_messages()
    print_messages(messages)

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"

    # Cleanup
    os.remove(db_path)
    return True


def test_message_types() -> bool:
    """Test different message types and content formats."""
    print_test_header("Message Types Test")

    db_path = "test_conversations.db"
    conversation = SQLiteConversation(db_path=db_path)

    # Test different content types
    console.print("\n[bold]Adding different message types...[/bold]")
    conversation.add("user", "Simple text")
    conversation.add(
        "assistant", {"type": "json", "content": "Complex data"}
    )
    conversation.add("system", ["list", "of", "items"])
    conversation.add(
        "function",
        "Function result",
        message_type=MessageType.FUNCTION,
    )

    console.print("\n[bold]Retrieved messages:[/bold]")
    messages = conversation.get_messages()
    print_messages(messages)

    assert len(messages) == 4

    # Cleanup
    os.remove(db_path)
    return True


def test_conversation_operations() -> bool:
    """Test various conversation operations."""
    print_test_header("Conversation Operations Test")

    db_path = "test_conversations.db"
    conversation = SQLiteConversation(db_path=db_path)

    # Test batch operations
    console.print("\n[bold]Adding batch messages...[/bold]")
    messages = [
        Message(role="user", content="Message 1"),
        Message(role="assistant", content="Message 2"),
        Message(role="user", content="Message 3"),
    ]
    conversation.batch_add(messages)

    console.print("\n[bold]Retrieved messages:[/bold]")
    all_messages = conversation.get_messages()
    print_messages(all_messages)

    # Test statistics
    console.print("\n[bold]Conversation Statistics:[/bold]")
    stats = conversation.get_statistics()
    console.print(json.dumps(stats, indent=2))

    # Test role counting
    console.print("\n[bold]Role Counts:[/bold]")
    role_counts = conversation.count_messages_by_role()
    console.print(json.dumps(role_counts, indent=2))

    assert stats["total_messages"] == 3
    assert role_counts["user"] == 2
    assert role_counts["assistant"] == 1

    # Cleanup
    os.remove(db_path)
    return True


def test_file_operations() -> bool:
    """Test file operations (JSON/YAML)."""
    print_test_header("File Operations Test")

    db_path = "test_conversations.db"
    json_path = "test_conversation.json"
    yaml_path = "test_conversation.yaml"

    conversation = SQLiteConversation(db_path=db_path)
    conversation.add("user", "Test message")

    # Test JSON operations
    console.print("\n[bold]Testing JSON operations...[/bold]")
    assert conversation.save_as_json(json_path)
    console.print(f"Saved to JSON: {json_path}")

    conversation.start_new_conversation()
    assert conversation.load_from_json(json_path)
    console.print("Loaded from JSON")

    # Test YAML operations
    console.print("\n[bold]Testing YAML operations...[/bold]")
    assert conversation.save_as_yaml(yaml_path)
    console.print(f"Saved to YAML: {yaml_path}")

    conversation.start_new_conversation()
    assert conversation.load_from_yaml(yaml_path)
    console.print("Loaded from YAML")

    # Cleanup
    os.remove(db_path)
    os.remove(json_path)
    os.remove(yaml_path)
    return True


def test_search_and_filter() -> bool:
    """Test search and filter operations."""
    print_test_header("Search and Filter Test")

    db_path = "test_conversations.db"
    conversation = SQLiteConversation(db_path=db_path)

    # Add test messages
    console.print("\n[bold]Adding test messages...[/bold]")
    conversation.add("user", "Hello world")
    conversation.add("assistant", "Hello there")
    conversation.add("user", "Goodbye world")

    # Test search
    console.print("\n[bold]Searching for 'world'...[/bold]")
    results = conversation.search_messages("world")
    print_messages(results, "Search Results")

    # Test role filtering
    console.print("\n[bold]Filtering user messages...[/bold]")
    user_messages = conversation.get_messages_by_role("user")
    print_messages(user_messages, "User Messages")

    assert len(results) == 2
    assert len(user_messages) == 2

    # Cleanup
    os.remove(db_path)
    return True


def test_conversation_management() -> bool:
    """Test conversation management features."""
    print_test_header("Conversation Management Test")

    db_path = "test_conversations.db"
    conversation = SQLiteConversation(db_path=db_path)

    # Test conversation ID generation
    console.print("\n[bold]Testing conversation IDs...[/bold]")
    conv_id1 = conversation.get_conversation_id()
    console.print(f"First conversation ID: {conv_id1}")

    conversation.start_new_conversation()
    conv_id2 = conversation.get_conversation_id()
    console.print(f"Second conversation ID: {conv_id2}")

    assert conv_id1 != conv_id2

    # Test conversation deletion
    console.print("\n[bold]Testing conversation deletion...[/bold]")
    conversation.add("user", "Test message")
    assert conversation.delete_current_conversation()
    console.print("Conversation deleted successfully")

    # Cleanup
    os.remove(db_path)
    return True


def generate_test_report(
    test_results: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Generate a test report in JSON format.

    Args:
        test_results: List of test results

    Returns:
        Dict containing the test report
    """
    total_tests = len(test_results)
    passed_tests = sum(
        1 for result in test_results if result["success"]
    )
    failed_tests = total_tests - passed_tests
    total_time = sum(
        result["execution_time"] for result in test_results
    )

    report = {
        "timestamp": datetime.datetime.now().isoformat(),
        "summary": {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "total_execution_time": total_time,
            "average_execution_time": (
                total_time / total_tests if total_tests > 0 else 0
            ),
        },
        "test_results": test_results,
    }

    return report


def run_all_tests() -> None:
    """Run all tests and generate a report."""
    console.print(
        Panel(
            "[bold blue]Starting Test Suite[/bold blue]", expand=False
        )
    )

    tests = [
        ("Basic Conversation", test_basic_conversation),
        ("Message Types", test_message_types),
        ("Conversation Operations", test_conversation_operations),
        ("File Operations", test_file_operations),
        ("Search and Filter", test_search_and_filter),
        ("Conversation Management", test_conversation_management),
    ]

    test_results = []

    for test_name, test_func in tests:
        logger.info(f"Running test: {test_name}")
        success, message, execution_time = run_test(test_func)

        print_test_result(test_name, success, message, execution_time)

        result = {
            "test_name": test_name,
            "success": success,
            "message": message,
            "execution_time": execution_time,
            "timestamp": datetime.datetime.now().isoformat(),
        }

        if success:
            logger.success(f"Test passed: {test_name}")
        else:
            logger.error(f"Test failed: {test_name} - {message}")

        test_results.append(result)

    # Generate and save report
    report = generate_test_report(test_results)
    report_path = "test_report.json"

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print final summary
    console.print("\n[bold blue]Test Suite Summary[/bold blue]")
    console.print(
        Panel(
            f"Total tests: {report['summary']['total_tests']}\n"
            f"Passed tests: {report['summary']['passed_tests']}\n"
            f"Failed tests: {report['summary']['failed_tests']}\n"
            f"Total execution time: {report['summary']['total_execution_time']:.2f} seconds",
            title="Summary",
            expand=False,
        )
    )

    logger.info(f"Test report saved to {report_path}")


if __name__ == "__main__":
    run_all_tests()
