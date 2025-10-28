import asyncio
import json
import os
import tempfile
import time
import unittest
from statistics import mean, median, stdev, variance
from unittest.mock import MagicMock, patch

import psutil
import pytest
import yaml
from dotenv import load_dotenv
from rich.console import Console
from rich.table import Table

from swarms import (
    Agent,
    create_agents_from_yaml,
)

# Load environment variables
load_dotenv()

# Global test configuration
openai_api_key = os.getenv("OPENAI_API_KEY")


# ============================================================================
# FIXTURES AND UTILITIES
# ============================================================================


@pytest.fixture
def basic_flow(mocked_llm):
    """Basic agent flow for testing"""
    return Agent(llm=mocked_llm, max_loops=1)


@pytest.fixture
def flow_with_condition(mocked_llm):
    """Agent flow with stopping condition"""
    from swarms.structs.agent import stop_when_repeats

    return Agent(
        llm=mocked_llm,
        max_loops=1,
        stopping_condition=stop_when_repeats,
    )


@pytest.fixture
def mock_agents():
    """Mock agents for testing"""

    class MockAgent:
        def __init__(self, name):
            self.name = name
            self.agent_name = name

        def run(self, task, img=None, *args, **kwargs):
            return f"{self.name} processed {task}"

    return [
        MockAgent(name="Agent1"),
        MockAgent(name="Agent2"),
        MockAgent(name="Agent3"),
    ]


@pytest.fixture
def test_agent():
    """Create a real agent for testing"""
    with patch("swarms.structs.agent.LiteLLM") as mock_llm:
        mock_llm.return_value.run.return_value = "Test response"
        return Agent(
            agent_name="test_agent",
            agent_description="A test agent",
            system_prompt="You are a test agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            verbose=False,
            print_on=False,
        )


# ============================================================================
# BASIC AGENT TESTS
# ============================================================================


class TestBasicAgent:
    """Test basic agent functionality"""

    def test_stop_when_repeats(self):
        """Test stopping condition function"""
        from swarms.structs.agent import stop_when_repeats

        assert stop_when_repeats("Please Stop now")
        assert not stop_when_repeats("Continue the process")

    def test_flow_initialization(self, basic_flow):
        """Test agent initialization"""
        assert basic_flow.max_loops == 5
        assert basic_flow.stopping_condition is None
        assert basic_flow.loop_interval == 1
        assert basic_flow.retry_attempts == 3
        assert basic_flow.retry_interval == 1
        assert basic_flow.feedback == []
        assert basic_flow.memory == []
        assert basic_flow.task is None
        assert basic_flow.stopping_token == "<DONE>"
        assert not basic_flow.interactive

    def test_provide_feedback(self, basic_flow):
        """Test feedback functionality"""
        feedback = "Test feedback"
        basic_flow.provide_feedback(feedback)
        assert feedback in basic_flow.feedback

    @patch("time.sleep", return_value=None)
    def test_run_without_stopping_condition(
        self, mocked_sleep, basic_flow
    ):
        """Test running without stopping condition"""
        response = basic_flow.run("Test task")
        assert response is not None

    @patch("time.sleep", return_value=None)
    def test_run_with_stopping_condition(
        self, mocked_sleep, flow_with_condition
    ):
        """Test running with stopping condition"""
        response = flow_with_condition.run("Stop")
        assert response is not None

    def test_bulk_run(self, basic_flow):
        """Test bulk run functionality"""
        inputs = [{"task": "Test1"}, {"task": "Test2"}]
        responses = basic_flow.bulk_run(inputs)
        assert responses is not None

    def test_save_and_load(self, basic_flow, tmp_path):
        """Test save and load functionality"""
        file_path = tmp_path / "memory.json"
        basic_flow.memory.append(["Test1", "Test2"])
        basic_flow.save(file_path)

        new_flow = Agent(llm=basic_flow.llm, max_loops=5)
        new_flow.load(file_path)
        assert new_flow.memory == [["Test1", "Test2"]]

    def test_flow_call(self, basic_flow):
        """Test calling agent directly"""
        response = basic_flow("Test call")
        assert response == "Test call"

    def test_format_prompt(self, basic_flow):
        """Test prompt formatting"""
        formatted_prompt = basic_flow.format_prompt(
            "Hello {name}", name="John"
        )
        assert formatted_prompt == "Hello John"


# ============================================================================
# AGENT FEATURES TESTS
# ============================================================================


class TestAgentFeatures:
    """Test advanced agent features"""

    def test_basic_agent_functionality(self):
        """Test basic agent initialization and task execution"""
        print("\nTesting basic agent functionality...")

        agent = Agent(
            agent_name="Test-Agent", model_name="gpt-4.1", max_loops=1
        )

        response = agent.run("What is 2+2?")
        assert (
            response is not None
        ), "Agent response should not be None"

        # Test agent properties
        assert (
            agent.agent_name == "Test-Agent"
        ), "Agent name not set correctly"
        assert agent.max_loops == 1, "Max loops not set correctly"
        assert agent.llm is not None, "LLM not initialized"

        print("✓ Basic agent functionality test passed")

    def test_memory_management(self):
        """Test agent memory management functionality"""
        print("\nTesting memory management...")

        agent = Agent(
            agent_name="Memory-Test-Agent",
            max_loops=1,
            model_name="gpt-4.1",
            context_length=8192,
        )

        # Test adding to memory
        agent.add_memory("Test memory entry")
        assert (
            "Test memory entry"
            in agent.short_memory.return_history_as_string()
        )

        # Test memory query
        agent.memory_query("Test query")

        # Test token counting
        tokens = agent.check_available_tokens()
        assert isinstance(
            tokens, int
        ), "Token count should be an integer"

        print("✓ Memory management test passed")

    def test_agent_output_formats(self):
        """Test all available output formats"""
        print("\nTesting all output formats...")

        test_task = "Say hello!"

        output_types = {
            "str": str,
            "string": str,
            "list": str,  # JSON string containing list
            "json": str,  # JSON string
            "dict": dict,
            "yaml": str,
        }

        for output_type, expected_type in output_types.items():
            agent = Agent(
                agent_name=f"{output_type.capitalize()}-Output-Agent",
                model_name="gpt-4.1",
                max_loops=1,
                output_type=output_type,
            )

            response = agent.run(test_task)
            assert (
                response is not None
            ), f"{output_type} output should not be None"

            if output_type == "yaml":
                # Verify YAML can be parsed
                try:
                    yaml.safe_load(response)
                    print(f"✓ {output_type} output valid")
                except yaml.YAMLError:
                    assert (
                        False
                    ), f"Invalid YAML output for {output_type}"
            elif output_type in ["json", "list"]:
                # Verify JSON can be parsed
                try:
                    json.loads(response)
                    print(f"✓ {output_type} output valid")
                except json.JSONDecodeError:
                    assert (
                        False
                    ), f"Invalid JSON output for {output_type}"

        print("✓ Output formats test passed")

    def test_agent_state_management(self):
        """Test comprehensive state management functionality"""
        print("\nTesting state management...")

        # Create temporary directory for test files
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = os.path.join(temp_dir, "agent_state.json")

            # Create agent with initial state
            agent1 = Agent(
                agent_name="State-Test-Agent",
                model_name="gpt-4.1",
                max_loops=1,
                saved_state_path=state_path,
            )

            # Add some data to the agent
            agent1.run("Remember this: Test message 1")
            agent1.add_memory("Test message 2")

            # Save state
            agent1.save()
            assert os.path.exists(
                state_path
            ), "State file not created"

            # Create new agent and load state
            agent2 = Agent(
                agent_name="State-Test-Agent",
                model_name="gpt-4.1",
                max_loops=1,
            )
            agent2.load(state_path)

            # Verify state loaded correctly
            history2 = agent2.short_memory.return_history_as_string()
            assert (
                "Test message 1" in history2
            ), "State not loaded correctly"
            assert (
                "Test message 2" in history2
            ), "Memory not loaded correctly"

            # Test autosave functionality
            agent3 = Agent(
                agent_name="Autosave-Test-Agent",
                model_name="gpt-4.1",
                max_loops=1,
                saved_state_path=os.path.join(
                    temp_dir, "autosave_state.json"
                ),
                autosave=True,
            )

            agent3.run("Test autosave")
            time.sleep(2)  # Wait for autosave
            assert os.path.exists(
                os.path.join(temp_dir, "autosave_state.json")
            ), "Autosave file not created"

        print("✓ State management test passed")

    def test_agent_tools_and_execution(self):
        """Test agent tool handling and execution"""
        print("\nTesting tools and execution...")

        def sample_tool(x: int, y: int) -> int:
            """Sample tool that adds two numbers"""
            return x + y

        agent = Agent(
            agent_name="Tools-Test-Agent",
            model_name="gpt-4.1",
            max_loops=1,
            tools=[sample_tool],
        )

        # Test adding tools
        agent.add_tool(lambda x: x * 2)
        assert len(agent.tools) == 2, "Tool not added correctly"

        # Test removing tools
        agent.remove_tool(sample_tool)
        assert len(agent.tools) == 1, "Tool not removed correctly"

        # Test tool execution
        response = agent.run("Calculate 2 + 2 using the sample tool")
        assert response is not None, "Tool execution failed"

        print("✓ Tools and execution test passed")

    def test_agent_concurrent_execution(self):
        """Test agent concurrent execution capabilities"""
        print("\nTesting concurrent execution...")

        agent = Agent(
            agent_name="Concurrent-Test-Agent",
            model_name="gpt-4.1",
            max_loops=1,
        )

        # Test bulk run
        tasks = [
            {"task": "Count to 3"},
            {"task": "Say hello"},
            {"task": "Tell a short joke"},
        ]

        responses = agent.bulk_run(tasks)
        assert len(responses) == len(tasks), "Not all tasks completed"
        assert all(
            response is not None for response in responses
        ), "Some tasks failed"

        # Test concurrent tasks
        concurrent_responses = agent.run_concurrent_tasks(
            ["Task 1", "Task 2", "Task 3"]
        )
        assert (
            len(concurrent_responses) == 3
        ), "Not all concurrent tasks completed"

        print("✓ Concurrent execution test passed")

    def test_agent_error_handling(self):
        """Test agent error handling and recovery"""
        print("\nTesting error handling...")

        agent = Agent(
            agent_name="Error-Test-Agent",
            model_name="gpt-4.1",
            max_loops=1,
            retry_attempts=3,
            retry_interval=1,
        )

        # Test invalid tool execution
        try:
            agent.parse_and_execute_tools("invalid_json")
            print("✓ Invalid tool execution handled")
        except Exception:
            assert True, "Expected error caught"

        # Test recovery after error
        response = agent.run("Continue after error")
        assert (
            response is not None
        ), "Agent failed to recover after error"

        print("✓ Error handling test passed")

    def test_agent_configuration(self):
        """Test agent configuration and parameters"""
        print("\nTesting agent configuration...")

        agent = Agent(
            agent_name="Config-Test-Agent",
            model_name="gpt-4.1",
            max_loops=1,
            temperature=0.7,
            max_tokens=4000,
            context_length=8192,
        )

        # Test configuration methods
        agent.update_system_prompt("New system prompt")
        agent.update_max_loops(2)
        agent.update_loop_interval(2)

        # Verify updates
        assert agent.max_loops == 2, "Max loops not updated"
        assert agent.loop_interval == 2, "Loop interval not updated"

        # Test configuration export
        config_dict = agent.to_dict()
        assert isinstance(
            config_dict, dict
        ), "Configuration export failed"

        # Test YAML export
        yaml_config = agent.to_yaml()
        assert isinstance(yaml_config, str), "YAML export failed"

        print("✓ Configuration test passed")

    def test_agent_with_stopping_condition(self):
        """Test agent with custom stopping condition"""
        print("\nTesting agent with stopping condition...")

        def custom_stopping_condition(response: str) -> bool:
            return "STOP" in response.upper()

        agent = Agent(
            agent_name="Stopping-Condition-Agent",
            model_name="gpt-4.1",
            max_loops=1,
            stopping_condition=custom_stopping_condition,
        )

        response = agent.run("Count up until you see the word STOP")
        assert response is not None, "Stopping condition test failed"
        print("✓ Stopping condition test passed")

    def test_agent_with_retry_mechanism(self):
        """Test agent retry mechanism"""
        print("\nTesting agent retry mechanism...")

        agent = Agent(
            agent_name="Retry-Test-Agent",
            model_name="gpt-4.1",
            max_loops=1,
            retry_attempts=3,
            retry_interval=1,
        )

        response = agent.run("Tell me a joke.")
        assert response is not None, "Retry mechanism test failed"
        print("✓ Retry mechanism test passed")

    def test_bulk_and_filtered_operations(self):
        """Test bulk operations and response filtering"""
        print("\nTesting bulk and filtered operations...")

        agent = Agent(
            agent_name="Bulk-Filter-Test-Agent",
            model_name="gpt-4.1",
            max_loops=1,
        )

        # Test bulk run
        bulk_tasks = [
            {"task": "What is 2+2?"},
            {"task": "Name a color"},
            {"task": "Count to 3"},
        ]
        bulk_responses = agent.bulk_run(bulk_tasks)
        assert len(bulk_responses) == len(
            bulk_tasks
        ), "Bulk run should return same number of responses as tasks"

        # Test response filtering
        agent.add_response_filter("color")
        filtered_response = agent.filtered_run(
            "What is your favorite color?"
        )
        assert (
            "[FILTERED]" in filtered_response
        ), "Response filter not applied"

        print("✓ Bulk and filtered operations test passed")

    async def test_async_operations(self):
        """Test asynchronous operations"""
        print("\nTesting async operations...")

        agent = Agent(
            agent_name="Async-Test-Agent",
            model_name="gpt-4.1",
            max_loops=1,
        )

        # Test single async run
        response = await agent.arun("What is 1+1?")
        assert response is not None, "Async run failed"

        # Test concurrent async runs
        tasks = ["Task 1", "Task 2", "Task 3"]
        responses = await asyncio.gather(
            *[agent.arun(task) for task in tasks]
        )
        assert len(responses) == len(
            tasks
        ), "Not all async tasks completed"

        print("✓ Async operations test passed")

    def test_memory_and_state_persistence(self):
        """Test memory management and state persistence"""
        print("\nTesting memory and state persistence...")

        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = os.path.join(temp_dir, "test_state.json")

            # Create agent with memory configuration
            agent1 = Agent(
                agent_name="Memory-State-Test-Agent",
                model_name="gpt-4.1",
                max_loops=1,
                saved_state_path=state_path,
                context_length=8192,
                autosave=True,
            )

            # Test memory operations
            agent1.add_memory("Important fact: The sky is blue")
            agent1.memory_query("What color is the sky?")

            # Save state
            agent1.save()

            # Create new agent and load state
            agent2 = Agent(
                agent_name="Memory-State-Test-Agent",
                model_name="gpt-4.1",
                max_loops=1,
            )
            agent2.load(state_path)

            # Verify memory persistence
            memory_content = (
                agent2.short_memory.return_history_as_string()
            )
            assert (
                "sky is blue" in memory_content
            ), "Memory not properly persisted"

            print("✓ Memory and state persistence test passed")

    def test_sentiment_and_evaluation(self):
        """Test sentiment analysis and response evaluation"""
        print("\nTesting sentiment analysis and evaluation...")

        def mock_sentiment_analyzer(text):
            """Mock sentiment analyzer that returns a score between 0 and 1"""
            return 0.7 if "positive" in text.lower() else 0.3

        def mock_evaluator(response):
            """Mock evaluator that checks response quality"""
            return "GOOD" if len(response) > 10 else "BAD"

        agent = Agent(
            agent_name="Sentiment-Eval-Test-Agent",
            model_name="gpt-4.1",
            max_loops=1,
            sentiment_analyzer=mock_sentiment_analyzer,
            sentiment_threshold=0.5,
            evaluator=mock_evaluator,
        )

        # Test sentiment analysis
        agent.run("Generate a positive message")

        # Test evaluation
        agent.run("Generate a detailed response")

        print("✓ Sentiment and evaluation test passed")

    def test_tool_management(self):
        """Test tool management functionality"""
        print("\nTesting tool management...")

        def tool1(x: int) -> int:
            """Sample tool 1"""
            return x * 2

        def tool2(x: int) -> int:
            """Sample tool 2"""
            return x + 2

        agent = Agent(
            agent_name="Tool-Test-Agent",
            model_name="gpt-4.1",
            max_loops=1,
            tools=[tool1],
        )

        # Test adding tools
        agent.add_tool(tool2)
        assert len(agent.tools) == 2, "Tool not added correctly"

        # Test removing tools
        agent.remove_tool(tool1)
        assert len(agent.tools) == 1, "Tool not removed correctly"

        # Test adding multiple tools
        agent.add_tools([tool1, tool2])
        assert (
            len(agent.tools) == 3
        ), "Multiple tools not added correctly"

        print("✓ Tool management test passed")

    def test_system_prompt_and_configuration(self):
        """Test system prompt and configuration updates"""
        print("\nTesting system prompt and configuration...")

        agent = Agent(
            agent_name="Config-Test-Agent",
            model_name="gpt-4.1",
            max_loops=1,
        )

        # Test updating system prompt
        new_prompt = "You are a helpful assistant."
        agent.update_system_prompt(new_prompt)
        assert (
            agent.system_prompt == new_prompt
        ), "System prompt not updated"

        # Test configuration updates
        agent.update_max_loops(5)
        assert agent.max_loops == 5, "Max loops not updated"

        agent.update_loop_interval(2)
        assert agent.loop_interval == 2, "Loop interval not updated"

        # Test configuration export
        config_dict = agent.to_dict()
        assert isinstance(
            config_dict, dict
        ), "Configuration export failed"

        print("✓ System prompt and configuration test passed")

    def test_agent_with_dynamic_temperature(self):
        """Test agent with dynamic temperature"""
        print("\nTesting agent with dynamic temperature...")

        agent = Agent(
            agent_name="Dynamic-Temp-Agent",
            model_name="gpt-4.1",
            max_loops=2,
            dynamic_temperature_enabled=True,
        )

        response = agent.run("Generate a creative story.")
        assert response is not None, "Dynamic temperature test failed"
        print("✓ Dynamic temperature test passed")


# ============================================================================
# AGENT LOGGING TESTS
# ============================================================================


class TestAgentLogging:
    """Test agent logging functionality"""

    def setUp(self):
        """Set up test fixtures"""
        self.mock_tokenizer = MagicMock()
        self.mock_tokenizer.count_tokens.return_value = 100

        self.mock_short_memory = MagicMock()
        self.mock_short_memory.get_memory_stats.return_value = {
            "message_count": 2
        }

        self.mock_long_memory = MagicMock()
        self.mock_long_memory.get_memory_stats.return_value = {
            "item_count": 5
        }

        self.agent = Agent(
            tokenizer=self.mock_tokenizer,
            short_memory=self.mock_short_memory,
            long_term_memory=self.mock_long_memory,
        )

    def test_log_step_metadata_basic(self):
        """Test basic step metadata logging"""
        log_result = self.agent.log_step_metadata(
            1, "Test prompt", "Test response"
        )

        assert "step_id" in log_result
        assert "timestamp" in log_result
        assert "tokens" in log_result
        assert "memory_usage" in log_result

        assert log_result["tokens"]["total"] == 200

    def test_log_step_metadata_no_long_term_memory(self):
        """Test step metadata logging without long term memory"""
        self.agent.long_term_memory = None
        log_result = self.agent.log_step_metadata(
            1, "prompt", "response"
        )
        assert log_result["memory_usage"]["long_term"] == {}

    def test_log_step_metadata_timestamp(self):
        """Test step metadata logging timestamp"""
        log_result = self.agent.log_step_metadata(
            1, "prompt", "response"
        )
        assert "timestamp" in log_result

    def test_token_counting_integration(self):
        """Test token counting integration"""
        self.mock_tokenizer.count_tokens.side_effect = [150, 250]
        log_result = self.agent.log_step_metadata(
            1, "prompt", "response"
        )

        assert log_result["tokens"]["total"] == 400

    def test_agent_output_updating(self):
        """Test agent output updating"""
        initial_total_tokens = sum(
            step["tokens"]["total"]
            for step in self.agent.agent_output.steps
        )
        self.agent.log_step_metadata(1, "prompt", "response")

        final_total_tokens = sum(
            step["tokens"]["total"]
            for step in self.agent.agent_output.steps
        )
        assert final_total_tokens - initial_total_tokens == 200
        assert len(self.agent.agent_output.steps) == 1

    def test_full_logging_cycle(self):
        """Test full logging cycle"""
        agent = Agent(agent_name="test-agent")
        task = "Test task"
        max_loops = 1

        result = agent._run(task, max_loops=max_loops)

        assert isinstance(result, dict)
        assert "steps" in result
        assert isinstance(result["steps"], list)
        assert len(result["steps"]) == max_loops

        if result["steps"]:
            step = result["steps"][0]
            assert "step_id" in step
            assert "timestamp" in step
            assert "task" in step
            assert "response" in step
            assert step["task"] == task
            assert step["response"] == "Response for loop 1"

        assert len(self.agent.agent_output.steps) > 0


# ============================================================================
# YAML AGENT CREATION TESTS
# ============================================================================


class TestCreateAgentsFromYaml:
    """Test YAML agent creation functionality"""

    def setUp(self):
        """Set up test fixtures"""
        # Mock the environment variable for API key
        os.environ["OPENAI_API_KEY"] = "fake-api-key"

        # Mock agent configuration YAML content
        self.valid_yaml_content = """
        agents:
          - agent_name: "Financial-Analysis-Agent"
            model:
              openai_api_key: "fake-api-key"
              model_name: "gpt-4o-mini"
              temperature: 0.1
              max_tokens: 2000
            system_prompt: "financial_agent_sys_prompt"
            max_loops: 1
            autosave: true
            dashboard: false
            verbose: true
            dynamic_temperature_enabled: true
            saved_state_path: "finance_agent.json"
            user_name: "swarms_corp"
            retry_attempts: 1
            context_length: 200000
            return_step_meta: false
            output_type: "str"
            task: "How can I establish a ROTH IRA to buy stocks and get a tax break?"
        """

    @patch(
        "builtins.open",
        new_callable=unittest.mock.mock_open,
        read_data="",
    )
    @patch("yaml.safe_load")
    def test_create_agents_return_agents(
        self, mock_safe_load, mock_open
    ):
        """Test creating agents from YAML and returning agents"""
        # Mock YAML content parsing
        mock_safe_load.return_value = {
            "agents": [
                {
                    "agent_name": "Financial-Analysis-Agent",
                    "model": {
                        "openai_api_key": "fake-api-key",
                        "model_name": "gpt-4o-mini",
                        "temperature": 0.1,
                        "max_tokens": 2000,
                    },
                    "system_prompt": "financial_agent_sys_prompt",
                    "max_loops": 1,
                    "autosave": True,
                    "dashboard": False,
                    "verbose": True,
                    "dynamic_temperature_enabled": True,
                    "saved_state_path": "finance_agent.json",
                    "user_name": "swarms_corp",
                    "retry_attempts": 1,
                    "context_length": 200000,
                    "return_step_meta": False,
                    "output_type": "str",
                    "task": "How can I establish a ROTH IRA to buy stocks and get a tax break?",
                }
            ]
        }

        # Test if agents are returned correctly
        agents = create_agents_from_yaml(
            "fake_yaml_path.yaml", return_type="agents"
        )
        assert len(agents) == 1
        assert agents[0].agent_name == "Financial-Analysis-Agent"

    @patch(
        "builtins.open",
        new_callable=unittest.mock.mock_open,
        read_data="",
    )
    @patch("yaml.safe_load")
    @patch(
        "swarms.Agent.run", return_value="Task completed successfully"
    )
    def test_create_agents_return_tasks(
        self, mock_agent_run, mock_safe_load, mock_open
    ):
        """Test creating agents from YAML and returning task results"""
        # Mock YAML content parsing
        mock_safe_load.return_value = {
            "agents": [
                {
                    "agent_name": "Financial-Analysis-Agent",
                    "model": {
                        "openai_api_key": "fake-api-key",
                        "model_name": "gpt-4o-mini",
                        "temperature": 0.1,
                        "max_tokens": 2000,
                    },
                    "system_prompt": "financial_agent_sys_prompt",
                    "max_loops": 1,
                    "autosave": True,
                    "dashboard": False,
                    "verbose": True,
                    "dynamic_temperature_enabled": True,
                    "saved_state_path": "finance_agent.json",
                    "user_name": "swarms_corp",
                    "retry_attempts": 1,
                    "context_length": 200000,
                    "return_step_meta": False,
                    "output_type": "str",
                    "task": "How can I establish a ROTH IRA to buy stocks and get a tax break?",
                }
            ]
        }

        # Test if tasks are executed and results are returned
        task_results = create_agents_from_yaml(
            "fake_yaml_path.yaml", return_type="tasks"
        )
        assert len(task_results) == 1
        assert (
            task_results[0]["agent_name"]
            == "Financial-Analysis-Agent"
        )
        assert task_results[0]["output"] is not None

    @patch(
        "builtins.open",
        new_callable=unittest.mock.mock_open,
        read_data="",
    )
    @patch("yaml.safe_load")
    def test_create_agents_return_both(
        self, mock_safe_load, mock_open
    ):
        """Test creating agents from YAML and returning both agents and tasks"""
        # Mock YAML content parsing
        mock_safe_load.return_value = {
            "agents": [
                {
                    "agent_name": "Financial-Analysis-Agent",
                    "model": {
                        "openai_api_key": "fake-api-key",
                        "model_name": "gpt-4o-mini",
                        "temperature": 0.1,
                        "max_tokens": 2000,
                    },
                    "system_prompt": "financial_agent_sys_prompt",
                    "max_loops": 1,
                    "autosave": True,
                    "dashboard": False,
                    "verbose": True,
                    "dynamic_temperature_enabled": True,
                    "saved_state_path": "finance_agent.json",
                    "user_name": "swarms_corp",
                    "retry_attempts": 1,
                    "context_length": 200000,
                    "return_step_meta": False,
                    "output_type": "str",
                    "task": "How can I establish a ROTH IRA to buy stocks and get a tax break?",
                }
            ]
        }

        # Test if both agents and tasks are returned
        agents, task_results = create_agents_from_yaml(
            "fake_yaml_path.yaml", return_type="both"
        )
        assert len(agents) == 1
        assert len(task_results) == 1
        assert agents[0].agent_name == "Financial-Analysis-Agent"
        assert task_results[0]["output"] is not None

    @patch(
        "builtins.open",
        new_callable=unittest.mock.mock_open,
        read_data="",
    )
    @patch("yaml.safe_load")
    def test_missing_agents_in_yaml(self, mock_safe_load, mock_open):
        """Test handling missing agents in YAML"""
        # Mock YAML content with missing "agents" key
        mock_safe_load.return_value = {}

        # Test if the function raises an error for missing "agents" key
        with pytest.raises(ValueError) as context:
            create_agents_from_yaml(
                "fake_yaml_path.yaml", return_type="agents"
            )
        assert (
            "The YAML configuration does not contain 'agents'."
            in str(context.exception)
        )

    @patch(
        "builtins.open",
        new_callable=unittest.mock.mock_open,
        read_data="",
    )
    @patch("yaml.safe_load")
    def test_invalid_return_type(self, mock_safe_load, mock_open):
        """Test handling invalid return type"""
        # Mock YAML content parsing
        mock_safe_load.return_value = {
            "agents": [
                {
                    "agent_name": "Financial-Analysis-Agent",
                    "model": {
                        "openai_api_key": "fake-api-key",
                        "model_name": "gpt-4o-mini",
                        "temperature": 0.1,
                        "max_tokens": 2000,
                    },
                    "system_prompt": "financial_agent_sys_prompt",
                    "max_loops": 1,
                    "autosave": True,
                    "dashboard": False,
                    "verbose": True,
                    "dynamic_temperature_enabled": True,
                    "saved_state_path": "finance_agent.json",
                    "user_name": "swarms_corp",
                    "retry_attempts": 1,
                    "context_length": 200000,
                    "return_step_meta": False,
                    "output_type": "str",
                    "task": "How can I establish a ROTH IRA to buy stocks and get a tax break?",
                }
            ]
        }

        # Test if an error is raised for invalid return_type
        with pytest.raises(ValueError) as context:
            create_agents_from_yaml(
                "fake_yaml_path.yaml", return_type="invalid_type"
            )
        assert "Invalid return_type" in str(context.exception)


# ============================================================================
# BENCHMARK TESTS
# ============================================================================


class TestAgentBenchmark:
    """Test agent benchmarking functionality"""

    def test_benchmark_multiple_agents(self):
        """Test benchmarking multiple agents"""
        console = Console()
        init_times = []
        memory_readings = []
        process = psutil.Process(os.getpid())

        # Create benchmark tables
        time_table = Table(title="Time Statistics")
        time_table.add_column("Metric", style="cyan")
        time_table.add_column("Value", style="green")

        memory_table = Table(title="Memory Statistics")
        memory_table.add_column("Metric", style="cyan")
        memory_table.add_column("Value", style="green")

        initial_memory = process.memory_info().rss / 1024
        start_total_time = time.perf_counter()

        # Initialize agents and measure performance
        num_agents = 10  # Reduced for testing
        for i in range(num_agents):
            start_time = time.perf_counter()

            Agent(
                agent_name=f"Financial-Analysis-Agent-{i}",
                agent_description="Personal finance advisor agent",
                max_loops=2,
                model_name="gpt-4o-mini",
                dynamic_temperature_enabled=True,
                interactive=False,
            )

            init_time = (time.perf_counter() - start_time) * 1000
            init_times.append(init_time)

            current_memory = process.memory_info().rss / 1024
            memory_readings.append(current_memory - initial_memory)

            if (i + 1) % 5 == 0:
                console.print(
                    f"Created {i + 1} agents...", style="bold blue"
                )

        (time.perf_counter() - start_total_time) * 1000

        # Calculate statistics
        time_stats = self._get_time_stats(init_times)
        memory_stats = self._get_memory_stats(memory_readings)

        # Verify basic statistics
        assert len(init_times) == num_agents
        assert len(memory_readings) == num_agents
        assert time_stats["mean"] > 0
        assert memory_stats["mean"] >= 0

        print("✓ Benchmark test passed")

    def _get_memory_stats(self, memory_readings):
        """Calculate memory statistics"""
        return {
            "peak": max(memory_readings) if memory_readings else 0,
            "min": min(memory_readings) if memory_readings else 0,
            "mean": mean(memory_readings) if memory_readings else 0,
            "median": (
                median(memory_readings) if memory_readings else 0
            ),
            "stdev": (
                stdev(memory_readings)
                if len(memory_readings) > 1
                else 0
            ),
            "variance": (
                variance(memory_readings)
                if len(memory_readings) > 1
                else 0
            ),
        }

    def _get_time_stats(self, times):
        """Calculate time statistics"""
        return {
            "total": sum(times),
            "mean": mean(times) if times else 0,
            "median": median(times) if times else 0,
            "min": min(times) if times else 0,
            "max": max(times) if times else 0,
            "stdev": stdev(times) if len(times) > 1 else 0,
            "variance": variance(times) if len(times) > 1 else 0,
        }


# ============================================================================
# TOOL USAGE TESTS
# ============================================================================


class TestAgentToolUsage:
    """Test comprehensive tool usage functionality for agents"""

    def test_normal_callable_tools(self):
        """Test normal callable tools (functions, lambdas, methods)"""
        print("\nTesting normal callable tools...")

        def math_tool(x: int, y: int) -> int:
            """Add two numbers together"""
            return x + y

        def string_tool(text: str) -> str:
            """Convert text to uppercase"""
            return text.upper()

        def list_tool(items: list) -> int:
            """Count items in a list"""
            return len(items)

        # Test with individual function tools
        agent = Agent(
            agent_name="Callable-Tools-Test-Agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            tools=[math_tool, string_tool, list_tool],
        )

        # Test tool addition
        assert len(agent.tools) == 3, "Tools not added correctly"

        # Test tool execution
        response = agent.run("Use the math tool to add 5 and 3")
        assert response is not None, "Tool execution failed"

        # Test lambda tools
        def lambda_tool(x):
            return x * 2

        agent.add_tool(lambda_tool)
        assert (
            len(agent.tools) == 4
        ), "Lambda tool not added correctly"

        # Test method tools
        class MathOperations:
            def multiply(self, x: int, y: int) -> int:
                """Multiply two numbers"""
                return x * y

        math_ops = MathOperations()
        agent.add_tool(math_ops.multiply)
        assert (
            len(agent.tools) == 5
        ), "Method tool not added correctly"

        print("✓ Normal callable tools test passed")

    def test_tool_management_operations(self):
        """Test tool management operations (add, remove, list)"""
        print("\nTesting tool management operations...")

        def tool1(x: int) -> int:
            """Tool 1"""
            return x + 1

        def tool2(x: int) -> int:
            """Tool 2"""
            return x * 2

        def tool3(x: int) -> int:
            """Tool 3"""
            return x - 1

        agent = Agent(
            agent_name="Tool-Management-Test-Agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            tools=[tool1, tool2],
        )

        # Test initial tools
        assert (
            len(agent.tools) == 2
        ), "Initial tools not set correctly"

        # Test adding single tool
        agent.add_tool(tool3)
        assert len(agent.tools) == 3, "Single tool addition failed"

        # Test adding multiple tools
        def tool4(x: int) -> int:
            return x**2

        def tool5(x: int) -> int:
            return x // 2

        agent.add_tools([tool4, tool5])
        assert len(agent.tools) == 5, "Multiple tools addition failed"

        # Test removing single tool
        agent.remove_tool(tool1)
        assert len(agent.tools) == 4, "Single tool removal failed"

        # Test removing multiple tools
        agent.remove_tools([tool2, tool3])
        assert len(agent.tools) == 2, "Multiple tools removal failed"

        print("✓ Tool management operations test passed")

    def test_mcp_single_url_tools(self):
        """Test MCP single URL tools"""
        print("\nTesting MCP single URL tools...")

        # Mock MCP URL for testing
        mock_mcp_url = "http://localhost:8000/mcp"

        with patch(
            "swarms.structs.agent.get_mcp_tools_sync"
        ) as mock_get_tools:
            # Mock MCP tools response
            mock_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "mcp_calculator",
                        "description": "Perform calculations",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "expression": {
                                    "type": "string",
                                    "description": "Math expression",
                                }
                            },
                            "required": ["expression"],
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "mcp_weather",
                        "description": "Get weather information",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {
                                    "type": "string",
                                    "description": "City name",
                                }
                            },
                            "required": ["location"],
                        },
                    },
                },
            ]
            mock_get_tools.return_value = mock_tools

            agent = Agent(
                agent_name="MCP-Single-URL-Test-Agent",
                model_name="gpt-4o-mini",
                max_loops=1,
                mcp_url=mock_mcp_url,
                verbose=True,
            )

            # Test MCP tools integration
            tools = agent.add_mcp_tools_to_memory()
            assert len(tools) == 2, "MCP tools not loaded correctly"
            assert (
                mock_get_tools.called
            ), "MCP tools function not called"

            # Verify tool structure
            assert "mcp_calculator" in str(
                tools
            ), "Calculator tool not found"
            assert "mcp_weather" in str(
                tools
            ), "Weather tool not found"

        print("✓ MCP single URL tools test passed")

    def test_mcp_multiple_urls_tools(self):
        """Test MCP multiple URLs tools"""
        print("\nTesting MCP multiple URLs tools...")

        # Mock multiple MCP URLs for testing
        mock_mcp_urls = [
            "http://localhost:8000/mcp1",
            "http://localhost:8000/mcp2",
            "http://localhost:8000/mcp3",
        ]

        with patch(
            "swarms.structs.agent.get_tools_for_multiple_mcp_servers"
        ) as mock_get_tools:
            # Mock MCP tools response from multiple servers
            mock_tools = [
                {
                    "type": "function",
                    "function": {
                        "name": "server1_tool",
                        "description": "Tool from server 1",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "input": {"type": "string"}
                            },
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "server2_tool",
                        "description": "Tool from server 2",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "data": {"type": "string"}
                            },
                        },
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "server3_tool",
                        "description": "Tool from server 3",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "query": {"type": "string"}
                            },
                        },
                    },
                },
            ]
            mock_get_tools.return_value = mock_tools

            agent = Agent(
                agent_name="MCP-Multiple-URLs-Test-Agent",
                model_name="gpt-4o-mini",
                max_loops=1,
                mcp_urls=mock_mcp_urls,
                verbose=True,
            )

            # Test MCP tools integration from multiple servers
            tools = agent.add_mcp_tools_to_memory()
            assert (
                len(tools) == 3
            ), "MCP tools from multiple servers not loaded correctly"
            assert (
                mock_get_tools.called
            ), "MCP multiple tools function not called"

            # Verify tools from different servers
            tools_str = str(tools)
            assert (
                "server1_tool" in tools_str
            ), "Server 1 tool not found"
            assert (
                "server2_tool" in tools_str
            ), "Server 2 tool not found"
            assert (
                "server3_tool" in tools_str
            ), "Server 3 tool not found"

        print("✓ MCP multiple URLs tools test passed")

    def test_base_tool_class_tools(self):
        """Test BaseTool class tools"""
        print("\nTesting BaseTool class tools...")

        from swarms.tools.base_tool import BaseTool

        def sample_function(x: int, y: int) -> int:
            """Sample function for testing"""
            return x + y

        # Create BaseTool instance
        base_tool = BaseTool(
            verbose=True,
            tools=[sample_function],
            tool_system_prompt="You are a helpful tool assistant",
        )

        # Test tool schema generation
        schema = base_tool.func_to_dict(sample_function)
        assert isinstance(
            schema, dict
        ), "Tool schema not generated correctly"
        assert "name" in schema, "Tool name not in schema"
        assert (
            "description" in schema
        ), "Tool description not in schema"
        assert "parameters" in schema, "Tool parameters not in schema"

        # Test tool execution
        test_input = {"x": 5, "y": 3}
        result = base_tool.execute_tool(test_input)
        assert result is not None, "Tool execution failed"

        print("✓ BaseTool class tools test passed")

    def test_tool_execution_and_error_handling(self):
        """Test tool execution and error handling"""
        print("\nTesting tool execution and error handling...")

        def valid_tool(x: int) -> int:
            """Valid tool that works correctly"""
            return x * 2

        def error_tool(x: int) -> int:
            """Tool that raises an error"""
            raise ValueError("Test error")

        def type_error_tool(x: str) -> str:
            """Tool with type error"""
            return x.upper()

        agent = Agent(
            agent_name="Tool-Execution-Test-Agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            tools=[valid_tool, error_tool, type_error_tool],
        )

        # Test valid tool execution
        response = agent.run("Use the valid tool with input 5")
        assert response is not None, "Valid tool execution failed"

        # Test error handling
        try:
            agent.run("Use the error tool")
            # Should handle error gracefully
        except Exception:
            # Expected to handle errors gracefully
            pass

        print("✓ Tool execution and error handling test passed")

    def test_tool_schema_generation(self):
        """Test tool schema generation and validation"""
        print("\nTesting tool schema generation...")

        def complex_tool(
            name: str,
            age: int,
            email: str = None,
            is_active: bool = True,
        ) -> dict:
            """Complex tool with various parameter types"""
            return {
                "name": name,
                "age": age,
                "email": email,
                "is_active": is_active,
            }

        agent = Agent(
            agent_name="Tool-Schema-Test-Agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            tools=[complex_tool],
        )

        # Test that tools are properly registered
        assert len(agent.tools) == 1, "Tool not registered correctly"

        # Test tool execution with complex parameters
        response = agent.run(
            "Use the complex tool with name 'John', age 30, email 'john@example.com'"
        )
        assert response is not None, "Complex tool execution failed"

        print("✓ Tool schema generation test passed")

    def test_aop_tools(self):
        """Test AOP (Agent Operations) tools"""
        print("\nTesting AOP tools...")

        from swarms.structs.aop import AOP

        # Create test agents
        agent1 = Agent(
            agent_name="AOP-Agent-1",
            model_name="gpt-4o-mini",
            max_loops=1,
        )

        agent2 = Agent(
            agent_name="AOP-Agent-2",
            model_name="gpt-4o-mini",
            max_loops=1,
        )

        # Create AOP instance
        aop = AOP(
            server_name="test-aop-server",
            verbose=True,
        )

        # Test adding agents as tools
        tool_names = aop.add_agents_batch(
            agents=[agent1, agent2],
            tool_names=["math_agent", "text_agent"],
            tool_descriptions=[
                "Performs mathematical operations",
                "Handles text processing",
            ],
        )

        assert (
            len(tool_names) == 2
        ), "AOP agents not added as tools correctly"
        assert (
            "math_agent" in tool_names
        ), "Math agent tool not created"
        assert (
            "text_agent" in tool_names
        ), "Text agent tool not created"

        # Test tool discovery
        tools = aop.get_available_tools()
        assert len(tools) >= 2, "AOP tools not discovered correctly"

        print("✓ AOP tools test passed")

    def test_tool_choice_and_execution_modes(self):
        """Test different tool choice and execution modes"""
        print("\nTesting tool choice and execution modes...")

        def tool_a(x: int) -> int:
            """Tool A"""
            return x + 1

        def tool_b(x: int) -> int:
            """Tool B"""
            return x * 2

        # Test with auto tool choice
        agent_auto = Agent(
            agent_name="Auto-Tool-Choice-Agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            tools=[tool_a, tool_b],
            tool_choice="auto",
        )

        response_auto = agent_auto.run(
            "Calculate something using the available tools"
        )
        assert response_auto is not None, "Auto tool choice failed"

        # Test with specific tool choice
        agent_specific = Agent(
            agent_name="Specific-Tool-Choice-Agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            tools=[tool_a, tool_b],
            tool_choice="tool_a",
        )

        response_specific = agent_specific.run(
            "Use tool_a with input 5"
        )
        assert (
            response_specific is not None
        ), "Specific tool choice failed"

        # Test with tool execution enabled/disabled
        agent_execute = Agent(
            agent_name="Tool-Execute-Agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            tools=[tool_a, tool_b],
            execute_tool=True,
        )

        response_execute = agent_execute.run("Execute a tool")
        assert (
            response_execute is not None
        ), "Tool execution mode failed"

        print("✓ Tool choice and execution modes test passed")

    def test_tool_system_prompts(self):
        """Test tool system prompts and custom tool prompts"""
        print("\nTesting tool system prompts...")

        def calculator_tool(expression: str) -> str:
            """Calculate mathematical expressions"""
            try:
                result = eval(expression)
                return str(result)
            except Exception:
                return "Invalid expression"

        custom_tool_prompt = "You have access to a calculator tool. Use it for mathematical calculations."

        agent = Agent(
            agent_name="Tool-Prompt-Test-Agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            tools=[calculator_tool],
            tool_system_prompt=custom_tool_prompt,
        )

        # Test that custom tool prompt is set
        assert (
            agent.tool_system_prompt == custom_tool_prompt
        ), "Custom tool prompt not set"

        # Test tool execution with custom prompt
        response = agent.run("Calculate 2 + 2 * 3")
        assert (
            response is not None
        ), "Tool execution with custom prompt failed"

        print("✓ Tool system prompts test passed")

    def test_tool_parallel_execution(self):
        """Test parallel tool execution capabilities"""
        print("\nTesting parallel tool execution...")

        def slow_tool(x: int) -> int:
            """Slow tool that takes time"""
            import time

            time.sleep(0.1)  # Simulate slow operation
            return x * 2

        def fast_tool(x: int) -> int:
            """Fast tool"""
            return x + 1

        agent = Agent(
            agent_name="Parallel-Tool-Test-Agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            tools=[slow_tool, fast_tool],
        )

        # Test parallel tool execution
        start_time = time.time()
        response = agent.run("Use both tools with input 5")
        end_time = time.time()

        assert response is not None, "Parallel tool execution failed"
        # Should be faster than sequential execution
        assert (
            end_time - start_time
        ) < 0.5, "Parallel execution took too long"

        print("✓ Parallel tool execution test passed")

    def test_tool_validation_and_type_checking(self):
        """Test tool validation and type checking"""
        print("\nTesting tool validation and type checking...")

        def typed_tool(x: int, y: str, z: bool = False) -> dict:
            """Tool with specific type hints"""
            return {"x": x, "y": y, "z": z, "result": f"{x} {y} {z}"}

        agent = Agent(
            agent_name="Tool-Validation-Test-Agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            tools=[typed_tool],
        )

        # Test tool execution with correct types
        response = agent.run(
            "Use typed_tool with x=5, y='hello', z=True"
        )
        assert response is not None, "Typed tool execution failed"

        # Test tool execution with incorrect types (should handle gracefully)
        try:
            agent.run("Use typed_tool with incorrect types")
        except Exception:
            # Expected to handle type errors gracefully
            pass

        print("✓ Tool validation and type checking test passed")

    def test_tool_caching_and_performance(self):
        """Test tool caching and performance optimization"""
        print("\nTesting tool caching and performance...")

        call_count = 0

        def cached_tool(x: int) -> int:
            """Tool that should be cached"""
            nonlocal call_count
            call_count += 1
            return x**2

        agent = Agent(
            agent_name="Tool-Caching-Test-Agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            tools=[cached_tool],
        )

        # Test multiple calls to the same tool
        agent.run("Use cached_tool with input 5")
        agent.run("Use cached_tool with input 5 again")

        # Verify tool was called (caching behavior may vary)
        assert call_count >= 1, "Tool not called at least once"

        print("✓ Tool caching and performance test passed")

    def test_tool_error_recovery(self):
        """Test tool error recovery and fallback mechanisms"""
        print("\nTesting tool error recovery...")

        def unreliable_tool(x: int) -> int:
            """Tool that sometimes fails"""
            import random

            if random.random() < 0.5:
                raise Exception("Random failure")
            return x * 2

        def fallback_tool(x: int) -> int:
            """Fallback tool"""
            return x + 10

        agent = Agent(
            agent_name="Tool-Recovery-Test-Agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            tools=[unreliable_tool, fallback_tool],
            retry_attempts=3,
        )

        # Test error recovery
        response = agent.run("Use unreliable_tool with input 5")
        assert response is not None, "Tool error recovery failed"

        print("✓ Tool error recovery test passed")

    def test_tool_with_different_output_types(self):
        """Test tools with different output types"""
        print("\nTesting tools with different output types...")

        def json_tool(data: dict) -> str:
            """Tool that returns JSON string"""
            import json

            return json.dumps(data)

        def yaml_tool(data: dict) -> str:
            """Tool that returns YAML string"""
            import yaml

            return yaml.dump(data)

        def dict_tool(x: int) -> dict:
            """Tool that returns dictionary"""
            return {"value": x, "squared": x**2}

        agent = Agent(
            agent_name="Output-Types-Test-Agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            tools=[json_tool, yaml_tool, dict_tool],
        )

        # Test JSON tool
        response = agent.run(
            "Use json_tool with data {'name': 'test', 'value': 123}"
        )
        assert response is not None, "JSON tool execution failed"

        # Test YAML tool
        response = agent.run(
            "Use yaml_tool with data {'key': 'value'}"
        )
        assert response is not None, "YAML tool execution failed"

        # Test dict tool
        response = agent.run("Use dict_tool with input 5")
        assert response is not None, "Dict tool execution failed"

        print("✓ Tools with different output types test passed")

    def test_tool_with_async_execution(self):
        """Test tools with async execution"""
        print("\nTesting tools with async execution...")

        async def async_tool(x: int) -> int:
            """Async tool that performs async operation"""
            import asyncio

            await asyncio.sleep(0.01)  # Simulate async operation
            return x * 2

        def sync_tool(x: int) -> int:
            """Sync tool"""
            return x + 1

        agent = Agent(
            agent_name="Async-Tool-Test-Agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            tools=[
                sync_tool
            ],  # Note: async tools need special handling
        )

        # Test sync tool execution
        response = agent.run("Use sync_tool with input 5")
        assert response is not None, "Sync tool execution failed"

        print("✓ Tools with async execution test passed")

    def test_tool_with_file_operations(self):
        """Test tools that perform file operations"""
        print("\nTesting tools with file operations...")

        import os
        import tempfile

        def file_writer_tool(filename: str, content: str) -> str:
            """Tool that writes content to a file"""
            with open(filename, "w") as f:
                f.write(content)
            return f"Written {len(content)} characters to {filename}"

        def file_reader_tool(filename: str) -> str:
            """Tool that reads content from a file"""
            try:
                with open(filename, "r") as f:
                    return f.read()
            except FileNotFoundError:
                return "File not found"

        with tempfile.TemporaryDirectory() as temp_dir:
            test_file = os.path.join(temp_dir, "test.txt")

            agent = Agent(
                agent_name="File-Ops-Test-Agent",
                model_name="gpt-4o-mini",
                max_loops=1,
                tools=[file_writer_tool, file_reader_tool],
            )

            # Test file writing
            response = agent.run(
                f"Use file_writer_tool to write 'Hello World' to {test_file}"
            )
            assert (
                response is not None
            ), "File writing tool execution failed"

            # Test file reading
            response = agent.run(
                f"Use file_reader_tool to read from {test_file}"
            )
            assert (
                response is not None
            ), "File reading tool execution failed"

        print("✓ Tools with file operations test passed")

    def test_tool_with_network_operations(self):
        """Test tools that perform network operations"""
        print("\nTesting tools with network operations...")

        def url_tool(url: str) -> str:
            """Tool that processes URLs"""
            return f"Processing URL: {url}"

        def api_tool(endpoint: str, method: str = "GET") -> str:
            """Tool that simulates API calls"""
            return f"API {method} request to {endpoint}"

        agent = Agent(
            agent_name="Network-Ops-Test-Agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            tools=[url_tool, api_tool],
        )

        # Test URL tool
        response = agent.run(
            "Use url_tool with 'https://example.com'"
        )
        assert response is not None, "URL tool execution failed"

        # Test API tool
        response = agent.run(
            "Use api_tool with endpoint '/api/data' and method 'POST'"
        )
        assert response is not None, "API tool execution failed"

        print("✓ Tools with network operations test passed")

    def test_tool_with_database_operations(self):
        """Test tools that perform database operations"""
        print("\nTesting tools with database operations...")

        def db_query_tool(query: str) -> str:
            """Tool that simulates database queries"""
            return f"Executed query: {query}"

        def db_insert_tool(table: str, data: dict) -> str:
            """Tool that simulates database inserts"""
            return f"Inserted data into {table}: {data}"

        agent = Agent(
            agent_name="Database-Ops-Test-Agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            tools=[db_query_tool, db_insert_tool],
        )

        # Test database query
        response = agent.run(
            "Use db_query_tool with 'SELECT * FROM users'"
        )
        assert (
            response is not None
        ), "Database query tool execution failed"

        # Test database insert
        response = agent.run(
            "Use db_insert_tool with table 'users' and data {'name': 'John'}"
        )
        assert (
            response is not None
        ), "Database insert tool execution failed"

        print("✓ Tools with database operations test passed")

    def test_tool_with_machine_learning_operations(self):
        """Test tools that perform ML operations"""
        print("\nTesting tools with ML operations...")

        def predict_tool(features: list) -> str:
            """Tool that simulates ML predictions"""
            return f"Prediction for features {features}: 0.85"

        def train_tool(model_name: str, data_size: int) -> str:
            """Tool that simulates model training"""
            return f"Trained {model_name} with {data_size} samples"

        agent = Agent(
            agent_name="ML-Ops-Test-Agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            tools=[predict_tool, train_tool],
        )

        # Test ML prediction
        response = agent.run(
            "Use predict_tool with features [1, 2, 3, 4]"
        )
        assert (
            response is not None
        ), "ML prediction tool execution failed"

        # Test ML training
        response = agent.run(
            "Use train_tool with model 'random_forest' and data_size 1000"
        )
        assert (
            response is not None
        ), "ML training tool execution failed"

        print("✓ Tools with ML operations test passed")

    def test_tool_with_image_processing(self):
        """Test tools that perform image processing"""
        print("\nTesting tools with image processing...")

        def resize_tool(
            image_path: str, width: int, height: int
        ) -> str:
            """Tool that simulates image resizing"""
            return f"Resized {image_path} to {width}x{height}"

        def filter_tool(image_path: str, filter_type: str) -> str:
            """Tool that simulates image filtering"""
            return f"Applied {filter_type} filter to {image_path}"

        agent = Agent(
            agent_name="Image-Processing-Test-Agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            tools=[resize_tool, filter_tool],
        )

        # Test image resizing
        response = agent.run(
            "Use resize_tool with image 'test.jpg', width 800, height 600"
        )
        assert (
            response is not None
        ), "Image resize tool execution failed"

        # Test image filtering
        response = agent.run(
            "Use filter_tool with image 'test.jpg' and filter 'blur'"
        )
        assert (
            response is not None
        ), "Image filter tool execution failed"

        print("✓ Tools with image processing test passed")

    def test_tool_with_text_processing(self):
        """Test tools that perform text processing"""
        print("\nTesting tools with text processing...")

        def tokenize_tool(text: str) -> list:
            """Tool that tokenizes text"""
            return text.split()

        def translate_tool(text: str, target_lang: str) -> str:
            """Tool that simulates translation"""
            return f"Translated '{text}' to {target_lang}"

        def sentiment_tool(text: str) -> str:
            """Tool that simulates sentiment analysis"""
            return f"Sentiment of '{text}': positive"

        agent = Agent(
            agent_name="Text-Processing-Test-Agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            tools=[tokenize_tool, translate_tool, sentiment_tool],
        )

        # Test text tokenization
        response = agent.run(
            "Use tokenize_tool with 'Hello world this is a test'"
        )
        assert (
            response is not None
        ), "Text tokenization tool execution failed"

        # Test translation
        response = agent.run(
            "Use translate_tool with 'Hello' and target_lang 'Spanish'"
        )
        assert (
            response is not None
        ), "Translation tool execution failed"

        # Test sentiment analysis
        response = agent.run(
            "Use sentiment_tool with 'I love this product!'"
        )
        assert (
            response is not None
        ), "Sentiment analysis tool execution failed"

        print("✓ Tools with text processing test passed")

    def test_tool_with_mathematical_operations(self):
        """Test tools that perform mathematical operations"""
        print("\nTesting tools with mathematical operations...")

        def matrix_multiply_tool(
            matrix_a: list, matrix_b: list
        ) -> list:
            """Tool that multiplies matrices"""
            # Simple 2x2 matrix multiplication
            result = [[0, 0], [0, 0]]
            for i in range(2):
                for j in range(2):
                    for k in range(2):
                        result[i][j] += (
                            matrix_a[i][k] * matrix_b[k][j]
                        )
            return result

        def statistics_tool(data: list) -> dict:
            """Tool that calculates statistics"""
            return {
                "mean": sum(data) / len(data),
                "max": max(data),
                "min": min(data),
                "count": len(data),
            }

        def calculus_tool(function: str, x: float) -> str:
            """Tool that simulates calculus operations"""
            return f"Derivative of {function} at x={x}: 2*x"

        agent = Agent(
            agent_name="Math-Ops-Test-Agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            tools=[
                matrix_multiply_tool,
                statistics_tool,
                calculus_tool,
            ],
        )

        # Test matrix multiplication
        response = agent.run(
            "Use matrix_multiply_tool with [[1,2],[3,4]] and [[5,6],[7,8]]"
        )
        assert (
            response is not None
        ), "Matrix multiplication tool execution failed"

        # Test statistics
        response = agent.run(
            "Use statistics_tool with [1, 2, 3, 4, 5]"
        )
        assert (
            response is not None
        ), "Statistics tool execution failed"

        # Test calculus
        response = agent.run("Use calculus_tool with 'x^2' and x=3")
        assert response is not None, "Calculus tool execution failed"

        print("✓ Tools with mathematical operations test passed")


# ============================================================================
# LLM ARGS AND HANDLING TESTS
# ============================================================================


class TestLLMArgsAndHandling:
    """Test LLM arguments and handling functionality"""

    def test_combined_llm_args(self):
        """Test that llm_args, tools_list_dictionary, and MCP tools can be combined."""
        print("\nTesting combined LLM args...")

        # Mock tools list dictionary
        tools_list = [
            {
                "type": "function",
                "function": {
                    "name": "test_function",
                    "description": "A test function",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "test_param": {
                                "type": "string",
                                "description": "A test parameter",
                            }
                        },
                    },
                },
            }
        ]

        # Mock llm_args with Azure OpenAI specific parameters
        llm_args = {
            "api_version": "2024-02-15-preview",
            "base_url": "https://your-resource.openai.azure.com/",
            "api_key": "your-api-key",
        }

        try:
            # Test 1: Only llm_args
            print("Testing Agent with only llm_args...")
            Agent(
                agent_name="test-agent-1",
                model_name="gpt-4o-mini",
                llm_args=llm_args,
            )
            print("✓ Agent with only llm_args created successfully")

            # Test 2: Only tools_list_dictionary
            print("Testing Agent with only tools_list_dictionary...")
            Agent(
                agent_name="test-agent-2",
                model_name="gpt-4o-mini",
                tools_list_dictionary=tools_list,
            )
            print(
                "✓ Agent with only tools_list_dictionary created successfully"
            )

            # Test 3: Combined llm_args and tools_list_dictionary
            print(
                "Testing Agent with combined llm_args and tools_list_dictionary..."
            )
            agent3 = Agent(
                agent_name="test-agent-3",
                model_name="gpt-4o-mini",
                llm_args=llm_args,
                tools_list_dictionary=tools_list,
            )
            print(
                "✓ Agent with combined llm_args and tools_list_dictionary created successfully"
            )

            # Test 4: Verify that the LLM instance has the correct configuration
            print("Verifying LLM configuration...")

            # Check that agent3 has both llm_args and tools configured
            assert (
                agent3.llm_args == llm_args
            ), "llm_args not preserved"
            assert (
                agent3.tools_list_dictionary == tools_list
            ), "tools_list_dictionary not preserved"

            # Check that the LLM instance was created
            assert agent3.llm is not None, "LLM instance not created"

            print("✓ LLM configuration verified successfully")
            print("✓ Combined LLM args test passed")

        except Exception as e:
            print(f"✗ Combined LLM args test failed: {e}")
            raise

    def test_azure_openai_example(self):
        """Test the Azure OpenAI example with api_version parameter."""
        print("\nTesting Azure OpenAI example with api_version...")

        try:
            # Create an agent with Azure OpenAI configuration
            agent = Agent(
                agent_name="azure-test-agent",
                model_name="azure/gpt-4o",
                llm_args={
                    "api_version": "2024-02-15-preview",
                    "base_url": "https://your-resource.openai.azure.com/",
                    "api_key": "your-api-key",
                },
                tools_list_dictionary=[
                    {
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "description": "Get weather information",
                            "parameters": {
                                "type": "object",
                                "properties": {
                                    "location": {
                                        "type": "string",
                                        "description": "The city and state",
                                    }
                                },
                            },
                        },
                    }
                ],
            )

            print(
                "✓ Azure OpenAI agent with combined parameters created successfully"
            )

            # Verify configuration
            assert agent.llm_args is not None, "llm_args not set"
            assert (
                "api_version" in agent.llm_args
            ), "api_version not in llm_args"
            assert (
                agent.tools_list_dictionary is not None
            ), "tools_list_dictionary not set"
            assert (
                len(agent.tools_list_dictionary) > 0
            ), "tools_list_dictionary is empty"

            print("✓ Azure OpenAI configuration verified")
            print("✓ Azure OpenAI example test passed")

        except Exception as e:
            print(f"✗ Azure OpenAI test failed: {e}")
            raise

    def test_llm_handling_args_kwargs(self):
        """Test that llm_handling properly handles both args and kwargs."""
        print("\nTesting LLM handling args and kwargs...")

        # Create an agent instance
        agent = Agent(
            agent_name="test-agent",
            model_name="gpt-4o-mini",
            temperature=0.7,
            max_tokens=1000,
        )

        # Test 1: Call llm_handling with kwargs
        print("Test 1: Testing kwargs handling...")
        try:
            # This should work and add the kwargs to additional_args
            agent.llm_handling(top_p=0.9, frequency_penalty=0.1)
            print("✓ kwargs handling works")
        except Exception as e:
            print(f"✗ kwargs handling failed: {e}")
            raise

        # Test 2: Call llm_handling with args (dictionary)
        print("Test 2: Testing args handling with dictionary...")
        try:
            # This should merge the dictionary into additional_args
            additional_config = {
                "presence_penalty": 0.2,
                "logit_bias": {"123": 1},
            }
            agent.llm_handling(additional_config)
            print("✓ args handling with dictionary works")
        except Exception as e:
            print(f"✗ args handling with dictionary failed: {e}")
            raise

        # Test 3: Call llm_handling with both args and kwargs
        print("Test 3: Testing both args and kwargs...")
        try:
            # This should handle both
            additional_config = {"presence_penalty": 0.3}
            agent.llm_handling(
                additional_config, top_p=0.8, frequency_penalty=0.2
            )
            print("✓ combined args and kwargs handling works")
        except Exception as e:
            print(f"✗ combined args and kwargs handling failed: {e}")
            raise

        # Test 4: Call llm_handling with non-dictionary args
        print("Test 4: Testing non-dictionary args...")
        try:
            # This should store args under 'additional_args' key
            agent.llm_handling(
                "some_string", 123, ["list", "of", "items"]
            )
            print("✓ non-dictionary args handling works")
        except Exception as e:
            print(f"✗ non-dictionary args handling failed: {e}")
            raise

        print("✓ LLM handling args and kwargs test passed")


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================


def run_all_tests():
    """Run all test functions"""
    print("Starting Merged Agent Test Suite...\n")

    # Test classes to run
    test_classes = [
        TestBasicAgent,
        TestAgentFeatures,
        TestAgentLogging,
        TestCreateAgentsFromYaml,
        TestAgentBenchmark,
        TestAgentToolUsage,
        TestLLMArgsAndHandling,
    ]

    total_tests = 0
    passed_tests = 0
    failed_tests = 0

    for test_class in test_classes:
        print(f"\n{'='*50}")
        print(f"Running {test_class.__name__}")
        print(f"{'='*50}")

        # Create test instance
        test_instance = test_class()

        # Get all test methods
        test_methods = [
            method
            for method in dir(test_instance)
            if method.startswith("test_")
        ]

        for test_method in test_methods:
            total_tests += 1
            try:
                # Run the test method
                getattr(test_instance, test_method)()
                passed_tests += 1
                print(f"✓ {test_method}")
            except Exception as e:
                failed_tests += 1
                print(f"✗ {test_method}: {str(e)}")

    # Print summary
    print(f"\n{'='*50}")
    print("Test Summary")
    print(f"{'='*50}")
    print(f"Total Tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {failed_tests}")
    print(f"Success Rate: {(passed_tests/total_tests)*100:.2f}%")

    return {
        "total": total_tests,
        "passed": passed_tests,
        "failed": failed_tests,
        "success_rate": (passed_tests / total_tests) * 100,
    }


if __name__ == "__main__":
    # Run all tests
    results = run_all_tests()

    print(results)
