import time
import json
from datetime import datetime
from loguru import logger

from swarms.communication.redis_wrap import (
    RedisConversation,
    REDIS_AVAILABLE,
)


class TestResults:
    def __init__(self):
        self.results = []
        self.start_time = datetime.now()
        self.end_time = None
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0

    def add_result(
        self, test_name: str, passed: bool, error: str = None
    ):
        self.total_tests += 1
        if passed:
            self.passed_tests += 1
            status = "✅ PASSED"
        else:
            self.failed_tests += 1
            status = "❌ FAILED"

        self.results.append(
            {
                "test_name": test_name,
                "status": status,
                "error": error if error else "None",
            }
        )

    def generate_markdown(self) -> str:
        self.end_time = datetime.now()
        duration = (self.end_time - self.start_time).total_seconds()

        md = [
            "# Redis Conversation Test Results",
            "",
            f"Test Run: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"Duration: {duration:.2f} seconds",
            "",
            "## Summary",
            f"- Total Tests: {self.total_tests}",
            f"- Passed: {self.passed_tests}",
            f"- Failed: {self.failed_tests}",
            f"- Success Rate: {(self.passed_tests/self.total_tests*100):.1f}%",
            "",
            "## Detailed Results",
            "",
            "| Test Name | Status | Error |",
            "|-----------|--------|-------|",
        ]

        for result in self.results:
            md.append(
                f"| {result['test_name']} | {result['status']} | {result['error']} |"
            )

        return "\n".join(md)


class RedisConversationTester:
    def __init__(self):
        self.results = TestResults()
        self.conversation = None
        self.redis_server = None

    def run_test(self, test_func: callable, test_name: str):
        """Run a single test and record its result."""
        try:
            test_func()
            self.results.add_result(test_name, True)
        except Exception as e:
            self.results.add_result(test_name, False, str(e))
            logger.error(f"Test '{test_name}' failed: {str(e)}")

    def setup(self):
        """Initialize Redis server and conversation for testing."""
        try:
            # # Start embedded Redis server
            # self.redis_server = EmbeddedRedis(port=6379)
            # if not self.redis_server.start():
            #     logger.error("Failed to start embedded Redis server")
            #     return False

            # Initialize Redis conversation
            self.conversation = RedisConversation(
                system_prompt="Test System Prompt",
                redis_host="localhost",
                redis_port=6379,
                redis_retry_attempts=3,
                use_embedded_redis=True,
            )
            return True
        except Exception as e:
            logger.error(
                f"Failed to initialize Redis conversation: {str(e)}"
            )
            return False

    def cleanup(self):
        """Cleanup resources after tests."""
        if self.redis_server:
            self.redis_server.stop()

    def test_initialization(self):
        """Test basic initialization."""
        assert (
            self.conversation is not None
        ), "Failed to initialize RedisConversation"
        assert (
            self.conversation.system_prompt == "Test System Prompt"
        ), "System prompt not set correctly"

    def test_add_message(self):
        """Test adding messages."""
        self.conversation.add("user", "Hello")
        self.conversation.add("assistant", "Hi there!")
        messages = self.conversation.return_messages_as_list()
        assert len(messages) >= 2, "Failed to add messages"

    def test_json_message(self):
        """Test adding JSON messages."""
        json_content = {"key": "value", "nested": {"data": 123}}
        self.conversation.add("system", json_content)
        last_message = self.conversation.get_final_message_content()
        assert isinstance(
            json.loads(last_message), dict
        ), "Failed to handle JSON message"

    def test_search(self):
        """Test search functionality."""
        self.conversation.add("user", "searchable message")
        results = self.conversation.search("searchable")
        assert len(results) > 0, "Search failed to find message"

    def test_delete(self):
        """Test message deletion."""
        initial_count = len(
            self.conversation.return_messages_as_list()
        )
        self.conversation.delete(0)
        new_count = len(self.conversation.return_messages_as_list())
        assert (
            new_count == initial_count - 1
        ), "Failed to delete message"

    def test_update(self):
        """Test message update."""
        # Add initial message
        self.conversation.add("user", "original message")

        # Update the message
        self.conversation.update(0, "user", "updated message")

        # Get the message directly using query
        updated_message = self.conversation.query(0)

        # Verify the update
        assert (
            updated_message["content"] == "updated message"
        ), "Message content should be updated"

    def test_clear(self):
        """Test clearing conversation."""
        self.conversation.add("user", "test message")
        self.conversation.clear()
        messages = self.conversation.return_messages_as_list()
        assert len(messages) == 0, "Failed to clear conversation"

    def test_export_import(self):
        """Test export and import functionality."""
        self.conversation.add("user", "export test")
        self.conversation.export_conversation("test_export.txt")
        self.conversation.clear()
        self.conversation.import_conversation("test_export.txt")
        messages = self.conversation.return_messages_as_list()
        assert (
            len(messages) > 0
        ), "Failed to export/import conversation"

    def test_json_operations(self):
        """Test JSON operations."""
        self.conversation.add("user", "json test")
        json_data = self.conversation.to_json()
        assert isinstance(
            json.loads(json_data), list
        ), "Failed to convert to JSON"

    def test_yaml_operations(self):
        """Test YAML operations."""
        self.conversation.add("user", "yaml test")
        yaml_data = self.conversation.to_yaml()
        assert isinstance(yaml_data, str), "Failed to convert to YAML"

    def test_token_counting(self):
        """Test token counting functionality."""
        self.conversation.add("user", "token test message")
        time.sleep(1)  # Wait for async token counting
        messages = self.conversation.to_dict()
        assert any(
            "token_count" in msg for msg in messages
        ), "Failed to count tokens"

    def test_cache_operations(self):
        """Test cache operations."""
        self.conversation.add("user", "cache test")
        stats = self.conversation.get_cache_stats()
        assert isinstance(stats, dict), "Failed to get cache stats"

    def test_conversation_stats(self):
        """Test conversation statistics."""
        self.conversation.add("user", "stats test")
        counts = self.conversation.count_messages_by_role()
        assert isinstance(
            counts, dict
        ), "Failed to get message counts"

    def run_all_tests(self):
        """Run all tests and generate report."""
        if not REDIS_AVAILABLE:
            logger.error(
                "Redis is not available. Please install redis package."
            )
            return "# Redis Tests Failed\n\nRedis package is not installed."

        try:
            if not self.setup():
                logger.error("Failed to setup Redis connection.")
                return "# Redis Tests Failed\n\nFailed to connect to Redis server."

            tests = [
                (self.test_initialization, "Initialization Test"),
                (self.test_add_message, "Add Message Test"),
                (self.test_json_message, "JSON Message Test"),
                (self.test_search, "Search Test"),
                (self.test_delete, "Delete Test"),
                (self.test_update, "Update Test"),
                (self.test_clear, "Clear Test"),
                (self.test_export_import, "Export/Import Test"),
                (self.test_json_operations, "JSON Operations Test"),
                (self.test_yaml_operations, "YAML Operations Test"),
                (self.test_token_counting, "Token Counting Test"),
                (self.test_cache_operations, "Cache Operations Test"),
                (
                    self.test_conversation_stats,
                    "Conversation Stats Test",
                ),
            ]

            for test_func, test_name in tests:
                self.run_test(test_func, test_name)

            return self.results.generate_markdown()
        finally:
            self.cleanup()


def main():
    """Main function to run tests and save results."""
    tester = RedisConversationTester()
    markdown_results = tester.run_all_tests()

    # Save results to file
    with open("redis_test_results.md", "w") as f:
        f.write(markdown_results)

    logger.info(
        "Test results have been saved to redis_test_results.md"
    )


if __name__ == "__main__":
    main()
