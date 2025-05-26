import json
import time
import os
import sys
import socket
import subprocess
from datetime import datetime
from typing import Dict, Callable, Tuple
from loguru import logger
from swarms.communication.pulsar_struct import (
    PulsarConversation,
    Message,
)


def check_pulsar_client_installed() -> bool:
    """Check if pulsar-client package is installed."""
    try:
        import pulsar

        return True
    except ImportError:
        return False


def install_pulsar_client() -> bool:
    """Install pulsar-client package using pip."""
    try:
        logger.info("Installing pulsar-client package...")
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "pulsar-client"],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0:
            logger.info("Successfully installed pulsar-client")
            return True
        else:
            logger.error(
                f"Failed to install pulsar-client: {result.stderr}"
            )
            return False
    except Exception as e:
        logger.error(f"Error installing pulsar-client: {str(e)}")
        return False


def check_port_available(
    host: str = "localhost", port: int = 6650
) -> bool:
    """Check if a port is open on the given host."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.settimeout(2)  # 2 second timeout
        result = sock.connect_ex((host, port))
        return result == 0
    except Exception:
        return False
    finally:
        sock.close()


def setup_test_broker() -> Tuple[bool, str]:
    """
    Set up a test broker for running tests.
    Returns (success, message).
    """
    try:
        from pulsar import Client

        # Create a memory-based standalone broker for testing
        client = Client("pulsar://localhost:6650")
        producer = client.create_producer("test-topic")
        producer.close()
        client.close()
        return True, "Test broker setup successful"
    except Exception as e:
        return False, f"Failed to set up test broker: {str(e)}"


class PulsarTestSuite:
    """Custom test suite for PulsarConversation class."""

    def __init__(self, pulsar_host: str = "pulsar://localhost:6650"):
        self.pulsar_host = pulsar_host
        self.host = pulsar_host.split("://")[1].split(":")[0]
        self.port = int(pulsar_host.split(":")[-1])
        self.test_results = {
            "test_suite": "PulsarConversation Tests",
            "timestamp": datetime.now().isoformat(),
            "total_tests": 0,
            "passed_tests": 0,
            "failed_tests": 0,
            "skipped_tests": 0,
            "results": [],
        }

    def check_pulsar_setup(self) -> bool:
        """
        Check if Pulsar is properly set up and provide guidance if it's not.
        """
        # First check if pulsar-client is installed
        if not check_pulsar_client_installed():
            logger.error(
                "\nPulsar client library is not installed. Installing now..."
            )
            if not install_pulsar_client():
                logger.error(
                    "\nFailed to install pulsar-client. Please install it manually:\n"
                    "   $ pip install pulsar-client\n"
                )
                return False

            # Import the newly installed package
            try:
                from swarms.communication.pulsar_struct import (
                    PulsarConversation,
                    Message,
                )
            except ImportError as e:
                logger.error(
                    f"Failed to import PulsarConversation after installation: {str(e)}"
                )
                return False

        # Try to set up test broker
        success, message = setup_test_broker()
        if not success:
            logger.error(
                f"\nFailed to set up test environment: {message}"
            )
            return False

        logger.info("Pulsar setup check passed successfully")
        return True

    def run_test(self, test_func: Callable) -> Dict:
        """Run a single test and return its result."""
        start_time = time.time()
        test_name = test_func.__name__

        try:
            logger.info(f"Running test: {test_name}")
            test_func()
            success = True
            error = None
            status = "PASSED"
        except Exception as e:
            success = False
            error = str(e)
            status = "FAILED"
            logger.error(f"Test {test_name} failed: {error}")

        end_time = time.time()
        duration = round(end_time - start_time, 3)

        result = {
            "test_name": test_name,
            "success": success,
            "duration": duration,
            "error": error,
            "timestamp": datetime.now().isoformat(),
            "status": status,
        }

        self.test_results["total_tests"] += 1
        if success:
            self.test_results["passed_tests"] += 1
        else:
            self.test_results["failed_tests"] += 1

        self.test_results["results"].append(result)
        return result

    def test_initialization(self):
        """Test PulsarConversation initialization."""
        conversation = PulsarConversation(
            pulsar_host=self.pulsar_host,
            system_prompt="Test system prompt",
        )
        assert conversation.conversation_id is not None
        assert conversation.health_check()["client_connected"] is True
        conversation.__del__()

    def test_add_message(self):
        """Test adding a message."""
        conversation = PulsarConversation(
            pulsar_host=self.pulsar_host
        )
        msg_id = conversation.add("user", "Test message")
        assert msg_id is not None

        # Verify message was added
        messages = conversation.get_messages()
        assert len(messages) > 0
        assert messages[0]["content"] == "Test message"
        conversation.__del__()

    def test_batch_add_messages(self):
        """Test adding multiple messages."""
        conversation = PulsarConversation(
            pulsar_host=self.pulsar_host
        )
        messages = [
            Message(role="user", content="Message 1"),
            Message(role="assistant", content="Message 2"),
        ]
        msg_ids = conversation.batch_add(messages)
        assert len(msg_ids) == 2

        # Verify messages were added
        stored_messages = conversation.get_messages()
        assert len(stored_messages) == 2
        assert stored_messages[0]["content"] == "Message 1"
        assert stored_messages[1]["content"] == "Message 2"
        conversation.__del__()

    def test_get_messages(self):
        """Test retrieving messages."""
        conversation = PulsarConversation(
            pulsar_host=self.pulsar_host
        )
        conversation.add("user", "Test message")
        messages = conversation.get_messages()
        assert len(messages) > 0
        conversation.__del__()

    def test_search_messages(self):
        """Test searching messages."""
        conversation = PulsarConversation(
            pulsar_host=self.pulsar_host
        )
        conversation.add("user", "Unique test message")
        results = conversation.search("unique")
        assert len(results) > 0
        conversation.__del__()

    def test_conversation_clear(self):
        """Test clearing conversation."""
        conversation = PulsarConversation(
            pulsar_host=self.pulsar_host
        )
        conversation.add("user", "Test message")
        conversation.clear()
        messages = conversation.get_messages()
        assert len(messages) == 0
        conversation.__del__()

    def test_conversation_export_import(self):
        """Test exporting and importing conversation."""
        conversation = PulsarConversation(
            pulsar_host=self.pulsar_host
        )
        conversation.add("user", "Test message")
        conversation.export_conversation("test_export.json")

        new_conversation = PulsarConversation(
            pulsar_host=self.pulsar_host
        )
        new_conversation.import_conversation("test_export.json")
        messages = new_conversation.get_messages()
        assert len(messages) > 0
        conversation.__del__()
        new_conversation.__del__()

    def test_message_count(self):
        """Test message counting."""
        conversation = PulsarConversation(
            pulsar_host=self.pulsar_host
        )
        conversation.add("user", "Message 1")
        conversation.add("assistant", "Message 2")
        counts = conversation.count_messages_by_role()
        assert counts["user"] == 1
        assert counts["assistant"] == 1
        conversation.__del__()

    def test_conversation_string(self):
        """Test string representation."""
        conversation = PulsarConversation(
            pulsar_host=self.pulsar_host
        )
        conversation.add("user", "Test message")
        string_rep = conversation.get_str()
        assert "Test message" in string_rep
        conversation.__del__()

    def test_conversation_json(self):
        """Test JSON conversion."""
        conversation = PulsarConversation(
            pulsar_host=self.pulsar_host
        )
        conversation.add("user", "Test message")
        json_data = conversation.to_json()
        assert isinstance(json_data, str)
        assert "Test message" in json_data
        conversation.__del__()

    def test_conversation_yaml(self):
        """Test YAML conversion."""
        conversation = PulsarConversation(
            pulsar_host=self.pulsar_host
        )
        conversation.add("user", "Test message")
        yaml_data = conversation.to_yaml()
        assert isinstance(yaml_data, str)
        assert "Test message" in yaml_data
        conversation.__del__()

    def test_last_message(self):
        """Test getting last message."""
        conversation = PulsarConversation(
            pulsar_host=self.pulsar_host
        )
        conversation.add("user", "Test message")
        last_msg = conversation.get_last_message()
        assert last_msg["content"] == "Test message"
        conversation.__del__()

    def test_messages_by_role(self):
        """Test getting messages by role."""
        conversation = PulsarConversation(
            pulsar_host=self.pulsar_host
        )
        conversation.add("user", "User message")
        conversation.add("assistant", "Assistant message")
        user_messages = conversation.get_messages_by_role("user")
        assert len(user_messages) == 1
        conversation.__del__()

    def test_conversation_summary(self):
        """Test getting conversation summary."""
        conversation = PulsarConversation(
            pulsar_host=self.pulsar_host
        )
        conversation.add("user", "Test message")
        summary = conversation.get_conversation_summary()
        assert summary["message_count"] == 1
        conversation.__del__()

    def test_conversation_statistics(self):
        """Test getting conversation statistics."""
        conversation = PulsarConversation(
            pulsar_host=self.pulsar_host
        )
        conversation.add("user", "Test message")
        stats = conversation.get_statistics()
        assert stats["total_messages"] == 1
        conversation.__del__()

    def test_health_check(self):
        """Test health check functionality."""
        conversation = PulsarConversation(
            pulsar_host=self.pulsar_host
        )
        health = conversation.health_check()
        assert health["client_connected"] is True
        conversation.__del__()

    def test_cache_stats(self):
        """Test cache statistics."""
        conversation = PulsarConversation(
            pulsar_host=self.pulsar_host
        )
        stats = conversation.get_cache_stats()
        assert "hits" in stats
        assert "misses" in stats
        conversation.__del__()

    def run_all_tests(self):
        """Run all test cases."""
        if not self.check_pulsar_setup():
            logger.error(
                "Pulsar setup check failed. Please check the error messages above."
            )
            return

        test_methods = [
            method
            for method in dir(self)
            if method.startswith("test_")
            and callable(getattr(self, method))
        ]

        logger.info(f"Running {len(test_methods)} tests...")

        for method_name in test_methods:
            test_method = getattr(self, method_name)
            self.run_test(test_method)

        self.save_results()

    def save_results(self):
        """Save test results to JSON file."""
        total_tests = (
            self.test_results["passed_tests"]
            + self.test_results["failed_tests"]
        )

        if total_tests > 0:
            self.test_results["success_rate"] = round(
                (self.test_results["passed_tests"] / total_tests)
                * 100,
                2,
            )
        else:
            self.test_results["success_rate"] = 0

        # Add test environment info
        self.test_results["environment"] = {
            "pulsar_host": self.pulsar_host,
            "pulsar_port": self.port,
            "pulsar_client_installed": check_pulsar_client_installed(),
            "os": os.uname().sysname,
            "python_version": subprocess.check_output(
                ["python", "--version"]
            )
            .decode()
            .strip(),
        }

        with open("pulsar_test_results.json", "w") as f:
            json.dump(self.test_results, f, indent=2)

        logger.info(
            f"\nTest Results Summary:\n"
            f"Total tests: {self.test_results['total_tests']}\n"
            f"Passed: {self.test_results['passed_tests']}\n"
            f"Failed: {self.test_results['failed_tests']}\n"
            f"Skipped: {self.test_results['skipped_tests']}\n"
            f"Success rate: {self.test_results['success_rate']}%\n"
            f"Results saved to: pulsar_test_results.json"
        )


if __name__ == "__main__":
    try:
        test_suite = PulsarTestSuite()
        test_suite.run_all_tests()
    except KeyboardInterrupt:
        logger.warning("Tests interrupted by user")
        exit(1)
    except Exception as e:
        logger.error(f"Test suite failed: {str(e)}")
        exit(1)
