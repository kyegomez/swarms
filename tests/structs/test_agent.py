import asyncio
import json
import os
import tempfile
import threading
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml
from dotenv import load_dotenv

import swarms.utils.litellm_wrapper as litellm_wrapper
from swarms import Agent
from swarms.agents.autonomous_loop import AutonomousAgentLoop
from swarms.schemas.agent_errors import AgentToolExecutionError

load_dotenv()


@pytest.fixture
def mocked_llm():
    """Stand-in for the LLM, so these tests need no provider.

    `6f4803ef` (2025-10-21) deleted this but left the fixtures requesting it,
    so the tests below errored at setup and have not run since. Echoing the
    task back keeps every assertion about plumbing, not model output.
    """

    class MockedLLM:
        def run(self, task=None, *args, **kwargs):
            return task

        async def arun(self, task=None, *args, **kwargs):
            return task

    return MockedLLM()


def _patched_agent(name, **kwargs):
    """An Agent with the model client stubbed, so __init__ makes no calls."""
    # setdefault, not a keyword: callers override max_loops (notably "auto").
    kwargs.setdefault("max_loops", 1)
    with patch("swarms.structs.agent.LiteLLM"):
        return Agent(
            agent_name=name,
            print_on=False,
            verbose=False,
            persistent_memory=False,
            **kwargs,
        )


def _flow(name, llm, tmp_path, **kwargs):
    return Agent(
        agent_name=name,
        llm=llm,
        max_loops=1,
        print_on=False,
        verbose=False,
        persistent_memory=False,
        autosave=False,
        workspace_dir=str(tmp_path),
        **kwargs,
    )


@pytest.fixture
def basic_flow(mocked_llm, tmp_path):
    return _flow("basic-flow", mocked_llm, tmp_path)


@pytest.fixture
def flow_with_condition(mocked_llm, tmp_path):
    from swarms.structs.agent import stop_when_repeats

    return _flow(
        "flow-with-condition",
        mocked_llm,
        tmp_path,
        stopping_condition=stop_when_repeats,
    )


class TestBasicAgent:
    """Constructor, run plumbing and the save/load round trip."""

    def test_stop_when_repeats(self):
        from swarms.structs.agent import stop_when_repeats

        assert stop_when_repeats("Please Stop now")
        assert not stop_when_repeats("Continue the process")

    def test_flow_initialization(self, basic_flow):
        """The constructor arguments survive __init__.

        This asserted `max_loops == 5` against a fixture passing 1, plus
        `.feedback` and `.memory`, which the Agent has long since dropped.
        """
        assert basic_flow.max_loops == 1
        assert basic_flow.stopping_condition is None
        assert basic_flow.retry_attempts == 3
        assert basic_flow.task is None
        assert basic_flow.stopping_token == "<DONE>"
        assert not basic_flow.interactive

    @patch("time.sleep", return_value=None)
    def test_run_without_stopping_condition(
        self, mocked_sleep, basic_flow
    ):
        assert basic_flow.run("Test task") is not None

    @patch("time.sleep", return_value=None)
    def test_run_with_stopping_condition(
        self, mocked_sleep, flow_with_condition
    ):
        assert flow_with_condition.run("Stop") is not None

    def test_bulk_run(self, basic_flow):
        inputs = [{"task": "Test1"}, {"task": "Test2"}]
        assert basic_flow.bulk_run(inputs) is not None

    def test_save_and_load(self, basic_flow, mocked_llm, tmp_path):
        """State written by save() comes back through load().

        The only save()/load() round trip at the Agent level, and it has not
        executed since 2025-10-21 — which is how a load() crash shipped
        unnoticed. load() restores scalar configuration and deliberately
        preserves live instances, so the conversation is not round-tripped.
        """
        file_path = str(tmp_path / "agent_state.json")
        basic_flow.max_loops = 3
        basic_flow.save(file_path)

        assert os.path.exists(file_path)

        restored = _flow("basic-flow-restored", mocked_llm, tmp_path)
        restored.load(file_path)

        assert restored.max_loops == 3
        assert restored.agent_name == "basic-flow"

    def test_flow_call(self, basic_flow):
        """__call__ forwards to run() rather than doing its own thing.

        Comparing two live calls does not work — the conversation grows
        between them — so the delegation is what is worth pinning.
        """
        with patch.object(
            basic_flow, "run", return_value="routed"
        ) as run:
            assert basic_flow("Test call") == "routed"

        run.assert_called_once()
        assert "Test call" in run.call_args.args or (
            run.call_args.kwargs.get("task") == "Test call"
        )


class TestAgentFeatures:
    """End-to-end agent behaviour against a live model."""

    def test_basic_agent_functionality(self):
        agent = Agent(
            agent_name="Test-Agent",
            model_name="gpt-5.4",
            max_loops=1,
        )

        assert agent.run("What is 2+2?") is not None
        assert agent.llm is not None

    @pytest.mark.parametrize(
        "output_type",
        ["str", "string", "list", "json", "dict", "yaml"],
    )
    def test_agent_output_formats(self, output_type):
        agent = Agent(
            agent_name=f"{output_type.capitalize()}-Output-Agent",
            model_name="gpt-5.4",
            max_loops=1,
            output_type=output_type,
        )

        response = agent.run("Say hello!")
        assert response is not None

        if output_type == "yaml":
            yaml.safe_load(response)
        elif output_type == "json":
            json.loads(response)
        elif output_type == "list":
            assert isinstance(response, list)

    def test_agent_state_management(self):
        """save() writes to the caller's saved_state_path, load() restores
        scalar config, and autosave writes with no explicit call.

        The conversation is deliberately not round-tripped: preserve_instances
        keeps the target's live Conversation rather than overwriting it, so
        asserting restored history here would pin behaviour load() disclaims.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            state_path = os.path.join(temp_dir, "agent_state.json")

            def build(name, **kwargs):
                return Agent(
                    agent_name=name,
                    model_name="gpt-5.4",
                    max_loops=1,
                    **kwargs,
                )

            agent1 = build("State", saved_state_path=state_path)
            agent1.run("Remember this: Test message 1")
            agent1.max_loops = 7
            agent1.save()

            assert os.path.exists(state_path)

            agent2 = build("State")
            agent2.load(state_path)

            assert agent2.max_loops == 7

            autosave_path = os.path.join(temp_dir, "autosave.json")
            build(
                "Autosave",
                saved_state_path=autosave_path,
                autosave=True,
            ).run("Test autosave")
            time.sleep(2)

            assert os.path.exists(autosave_path)

    def test_agent_concurrent_execution(self):
        agent = Agent(
            agent_name="Concurrent-Test-Agent",
            model_name="gpt-5.4",
            max_loops=1,
        )

        tasks = [
            {"task": "Count to 3"},
            {"task": "Say hello"},
            {"task": "Tell a short joke"},
        ]
        responses = agent.bulk_run(tasks)
        assert len(responses) == len(tasks)
        assert all(r is not None for r in responses)

        concurrent = agent.run_concurrent_tasks(
            ["Task 1", "Task 2", "Task 3"]
        )
        assert len(concurrent) == 3

    def test_agent_error_handling(self):
        """A malformed tool call must not stop the next run from working."""
        agent = Agent(
            agent_name="Error-Test-Agent",
            model_name="gpt-5.4",
            max_loops=1,
            retry_attempts=3,
        )

        try:
            agent.parse_and_execute_tools("invalid_json")
        except Exception:
            pass

        assert agent.run("Continue after error") is not None

    def test_agent_configuration(self):
        agent = Agent(
            agent_name="Config-Test-Agent",
            model_name="gpt-5.4",
            max_loops=1,
            temperature=0.7,
            max_tokens=4000,
            context_length=8192,
        )

        agent.update_system_prompt("New system prompt")
        agent.update_max_loops(2)
        agent.update_loop_interval(2)

        assert agent.system_prompt == "New system prompt"
        assert agent.max_loops == 2
        assert agent.loop_interval == 2
        assert isinstance(agent.to_dict(), dict)

    def test_agent_with_stopping_condition(self):
        def custom_stopping_condition(response: str) -> bool:
            return "STOP" in response.upper()

        agent = Agent(
            agent_name="Stopping-Condition-Agent",
            model_name="gpt-5.4",
            max_loops=1,
            stopping_condition=custom_stopping_condition,
        )

        assert (
            agent.run("Count up until you see the word STOP")
            is not None
        )

    async def test_async_operations(self):
        agent = Agent(
            agent_name="Async-Test-Agent",
            model_name="gpt-5.4",
            max_loops=1,
        )

        assert await agent.arun("What is 1+1?") is not None

        tasks = ["Task 1", "Task 2", "Task 3"]
        responses = await asyncio.gather(
            *[agent.arun(task) for task in tasks]
        )
        assert len(responses) == len(tasks)

    def test_sentiment_and_evaluation(self):
        """Both hooks must run without breaking the loop."""
        agent = Agent(
            agent_name="Sentiment-Eval-Test-Agent",
            model_name="gpt-5.4",
            max_loops=1,
            sentiment_analyzer=lambda text: 0.7,
            sentiment_threshold=0.5,
            evaluator=lambda response: "GOOD",
        )

        assert agent.run("Generate a positive message") is not None

    def test_agent_with_dynamic_temperature(self):
        agent = Agent(
            agent_name="Dynamic-Temp-Agent",
            model_name="gpt-5.4",
            max_loops=2,
            dynamic_temperature_enabled=True,
        )

        assert agent.run("Generate a creative story.") is not None


def _file_tool(filename: str, content: str) -> str:
    """Write content to a file."""
    return f"Written {len(content)} characters to {filename}"


def _url_tool(url: str) -> str:
    """Process a URL."""
    return f"Processing URL: {url}"


def _query_tool(table: str, limit: int) -> str:
    """Query a database table."""
    return f"Queried {table}, limit {limit}"


def _predict_tool(features: list) -> str:
    """Run an ML prediction."""
    return f"Prediction for features {features}: 0.85"


def _resize_tool(image_path: str, width: int, height: int) -> str:
    """Resize an image."""
    return f"Resized {image_path} to {width}x{height}"


def _summarize_tool(text: str) -> str:
    """Summarize text."""
    return f"Summary of {len(text)} characters"


def _math_tool(expression: str) -> str:
    """Evaluate a mathematical expression."""
    return f"Result of {expression}"


def _json_tool(data: dict) -> str:
    """Return the given data as a JSON string."""
    return json.dumps(data)


def _dict_tool(x: int) -> dict:
    """Return a dictionary describing x."""
    return {"value": x, "squared": x**2}


DOMAIN_TOOLS = [
    pytest.param(_file_tool, "write 'hi' to notes.txt", id="file"),
    pytest.param(_url_tool, "process https://example.com", id="net"),
    pytest.param(_query_tool, "query users with limit 10", id="db"),
    pytest.param(_predict_tool, "predict for [1, 2, 3]", id="ml"),
    pytest.param(_resize_tool, "resize a.png to 64x64", id="image"),
    pytest.param(_summarize_tool, "summarize 'hello'", id="text"),
    pytest.param(_math_tool, "evaluate 2 + 2 * 3", id="math"),
    pytest.param(_json_tool, "serialize {'a': 1}", id="json-out"),
    pytest.param(_dict_tool, "describe 5", id="dict-out"),
]


class TestAgentToolUsage:
    """Tool registration, schema generation and execution."""

    def test_callable_tools_register_and_unregister(self):
        """Functions, lambdas and bound methods all register, and add/remove
        in both singular and plural forms keep the list consistent."""

        def tool1(x: int) -> int:
            """Tool 1"""
            return x + 1

        def tool2(x: int) -> int:
            """Tool 2"""
            return x * 2

        agent = Agent(
            agent_name="Callable-Tools-Test-Agent",
            model_name="gpt-5.4",
            max_loops=1,
            tools=[tool1, tool2],
        )
        assert len(agent.tools) == 2
        assert agent.run("Use tool1 to add 1 to 5") is not None

        class MathOperations:
            def multiply(self, x: int, y: int) -> int:
                """Multiply two numbers"""
                return x * y

        agent.add_tool(lambda x: x**2)
        agent.add_tool(MathOperations().multiply)
        assert len(agent.tools) == 4

        agent.add_tools([lambda x: x // 2, lambda x: x - 1])
        assert len(agent.tools) == 6

        agent.remove_tool(tool1)
        assert len(agent.tools) == 5

        agent.remove_tools([tool2])
        assert len(agent.tools) == 4

    @staticmethod
    def _mcp_schema(name, prop):
        return {
            "type": "function",
            "function": {
                "name": name,
                "description": f"{name} description",
                "parameters": {
                    "type": "object",
                    "properties": {prop: {"type": "string"}},
                },
            },
        }

    @pytest.mark.parametrize(
        "kwargs, names",
        [
            (
                {"mcp_url": "http://localhost:8000/mcp"},
                ["mcp_calculator", "mcp_weather"],
            ),
            (
                {
                    "mcp_urls": [
                        "http://localhost:8000/mcp1",
                        "http://localhost:8000/mcp2",
                    ]
                },
                ["server1_tool", "server2_tool"],
            ),
        ],
        ids=["single-url", "multiple-urls"],
    )
    def test_mcp_tools_reach_the_agent(self, kwargs, names):
        schemas = [self._mcp_schema(n, "input") for n in names]
        with patch(
            "swarms.tools.mcp_manager.MCPManager.get_tools",
            return_value=schemas,
        ) as get_tools:
            agent = Agent(
                agent_name="MCP-Test-Agent",
                model_name="gpt-5.4",
                max_loops=1,
                **kwargs,
            )
            tools = agent.add_mcp_tools_to_memory()

        assert len(tools) == len(names)
        assert get_tools.called
        for name in names:
            assert name in str(tools)

    def test_tool_execution_and_error_recovery(self):
        """A raising tool must not take the agent down, and retry_attempts
        must carry the run past it."""

        def valid_tool(x: int) -> int:
            """Valid tool that works correctly"""
            return x * 2

        def error_tool(x: int) -> int:
            """Tool that always raises"""
            raise ValueError("Test error")

        agent = Agent(
            agent_name="Tool-Execution-Test-Agent",
            model_name="gpt-5.4",
            max_loops=1,
            tools=[valid_tool, error_tool],
            retry_attempts=3,
        )

        assert (
            agent.run("Use the valid tool with input 5") is not None
        )

        try:
            agent.run("Use the error tool")
        except Exception:
            pass

    def test_tool_schema_generation_and_typed_parameters(self):
        """Mixed required and optional parameter types survive both schema
        conversion and a typed call."""

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
            model_name="gpt-5.4",
            max_loops=1,
            tools=[complex_tool],
        )

        assert len(agent.tools) == 1
        assert (
            agent.run("Use complex_tool with name 'John', age 30")
            is not None
        )

    @pytest.mark.parametrize("tool_choice", ["auto", "tool_a"])
    def test_tool_choice_modes(self, tool_choice):
        def tool_a(x: int) -> int:
            """Tool A"""
            return x + 1

        def tool_b(x: int) -> int:
            """Tool B"""
            return x * 2

        agent = Agent(
            agent_name=f"Tool-Choice-{tool_choice}-Agent",
            model_name="gpt-5.4",
            max_loops=1,
            tools=[tool_a, tool_b],
            tool_choice=tool_choice,
        )

        assert agent.run("Use tool_a with input 5") is not None

    @pytest.mark.parametrize("tool, prompt", DOMAIN_TOOLS)
    def test_domain_tools_execute(self, tool, prompt):
        """One live run per tool domain the suite used to cover with seven
        near-identical tests. The domain is incidental — what is pinned is
        that an arbitrary typed callable survives registration and a run.
        """
        agent = Agent(
            agent_name=f"Domain-{tool.__name__}-Agent",
            model_name="gpt-5.4",
            max_loops=1,
            tools=[tool],
        )

        assert (
            agent.run(f"Use {tool.__name__} to {prompt}") is not None
        )


class TestLLMArgsAndHandling:
    """llm_args, credentials and the kwargs path into litellm."""

    TOOLS_LIST = [
        {
            "type": "function",
            "function": {
                "name": "test_function",
                "description": "A test function",
                "parameters": {
                    "type": "object",
                    "properties": {"test_param": {"type": "string"}},
                },
            },
        }
    ]

    LLM_ARGS = {
        "api_version": "2024-02-15-preview",
        "base_url": "https://your-resource.openai.azure.com/",
        "api_key": "your-api-key",
    }

    @pytest.mark.parametrize(
        "model_name", ["gpt-5.4", "azure/gpt-4o"]
    )
    def test_llm_args_and_tools_survive_together(self, model_name):
        """llm_args — including the api_version an Azure deployment needs —
        and tools_list_dictionary must both be preserved."""
        agent = Agent(
            agent_name="llm-args-agent",
            model_name=model_name,
            llm_args=self.LLM_ARGS,
            tools_list_dictionary=self.TOOLS_LIST,
        )

        assert agent.llm_args == self.LLM_ARGS
        assert "api_version" in agent.llm_args
        assert agent.tools_list_dictionary == self.TOOLS_LIST
        assert agent.llm is not None

    def test_llm_handling_args_kwargs(self):
        """llm_handling accepts kwargs, dict-args, both, and loose args."""
        agent = Agent(
            agent_name="test-agent",
            model_name="gpt-5.4",
            temperature=0.7,
            max_tokens=1000,
        )

        agent.llm_handling(top_p=0.9, frequency_penalty=0.1)
        agent.llm_handling({"presence_penalty": 0.2})
        agent.llm_handling({"presence_penalty": 0.3}, top_p=0.8)
        agent.llm_handling("some_string", 123, ["list"])

    def _capture_completion_params(self, llm):
        """Run llm with completion stubbed, returning the kwargs it got."""
        captured = {}
        message = SimpleNamespace(content="ok", tool_calls=None)
        response = SimpleNamespace(
            choices=[SimpleNamespace(message=message)]
        )

        def fake_completion(**kwargs):
            captured.update(kwargs)
            return response

        with patch.object(
            litellm_wrapper, "completion", side_effect=fake_completion
        ):
            llm.run("hello")

        return captured

    def test_reasoning_effort_is_absent_by_default(self):
        """It defaulted to "medium" and rode along on every request. OpenAI
        rejects reasoning_effort alongside function tools on
        /v1/chat/completions, so `Agent(model_name="gpt-5.4-mini",
        tools=[...])` raised BadRequestError out of the box.
        """
        agent = Agent(
            agent_name="reasoning-default-probe",
            model_name="gpt-5.4-mini",
            max_loops=1,
            persistent_memory=False,
            print_on=False,
        )

        assert agent.reasoning_effort is None
        params = self._capture_completion_params(agent.llm)
        assert "reasoning_effort" not in params

    def test_explicit_reasoning_effort_still_reaches_the_provider(
        self,
    ):
        """Defaulting to absent must not make the option unusable."""
        agent = Agent(
            agent_name="reasoning-explicit-probe",
            model_name="gpt-5.4-mini",
            reasoning_effort="high",
            max_loops=1,
            persistent_memory=False,
            print_on=False,
        )

        params = self._capture_completion_params(agent.llm)
        assert params.get("reasoning_effort") == "high"

    def test_llm_base_url_and_api_key_reach_the_provider_call(self):
        """A custom endpoint and key must survive Agent -> completion().

        Both values cross two layers and each dropped them independently, so
        requests silently went to the default provider instead.
        """
        base_url = "https://api.together.xyz/v1"
        api_key = "sk-test-not-a-real-key"

        agent = Agent(
            agent_name="credential-forwarding-probe",
            model_name="gpt-4o-mini",
            llm_base_url=base_url,
            llm_api_key=api_key,
            max_loops=1,
            persistent_memory=False,
            print_on=False,
        )

        assert agent.llm.base_url == base_url
        assert agent.llm.api_key == api_key

        params = self._capture_completion_params(agent.llm)
        assert params.get("base_url") == base_url
        assert params.get("api_key") == api_key

    def test_unset_credentials_are_omitted_from_the_provider_call(
        self,
    ):
        """An unset key must be absent, not passed as None.

        litellm falls back to the provider env var only when the kwarg is
        absent; an explicit None would override that and break every caller
        not using a custom endpoint.
        """
        agent = Agent(
            agent_name="credential-default-probe",
            model_name="gpt-4o-mini",
            max_loops=1,
            persistent_memory=False,
            print_on=False,
        )

        params = self._capture_completion_params(agent.llm)
        assert "api_key" not in params
        assert "base_url" not in params


class TestConstructorWindows:
    """context_length and max_tokens must survive __init__ rather than being
    overwritten with a flat default."""

    @staticmethod
    def _agent(**kwargs):
        return _patched_agent("window_agent", **kwargs)

    def test_context_length_explicit_value_is_honoured(self):
        """The constructor argument used to be overwritten with 16000."""
        agent = self._agent(
            model_name="gpt-4.1", context_length=200000
        )

        assert agent.context_length == 200000
        assert agent.short_memory.context_length == 200000

    def test_context_length_defaults_to_the_model_input_window(self):
        """Omitting it should size to the model, not a flat 16000."""
        # The relation, not the literal, so a litellm model-table refresh
        # does not break this.
        assert self._agent(model_name="gpt-4o").context_length > 16000

    def test_max_tokens_explicit_value_is_honoured(self):
        """A caller capping output was overwritten with the model's limit."""
        agent = self._agent(model_name="gpt-4o-mini", max_tokens=500)

        assert agent.max_tokens == 500

    def test_max_tokens_defaults_to_the_model_output_window(self):
        assert self._agent(model_name="gpt-4o-mini").max_tokens > 500

    def test_non_positive_max_tokens_falls_back_to_the_model(self):
        for bad in (0, -1, None):
            agent = self._agent(
                model_name="gpt-4o-mini", max_tokens=bad
            )
            assert agent.max_tokens > 0

    def test_saved_state_path_is_honoured(self):
        """It was assigned, then overwritten with a generated name twelve
        lines later, so the caller's path never reached save()."""
        agent = self._agent(saved_state_path="/tmp/chosen-state.json")

        assert agent.saved_state_path == "/tmp/chosen-state.json"

    def test_saved_state_path_defaults_when_unset(self):
        assert self._agent().saved_state_path.endswith("_state.json")

    def test_unknown_model_falls_back_for_both(self):
        """An unrecognised model must not raise out of __init__."""
        agent = self._agent(model_name="totally-made-up-model-xyz")

        assert agent.context_length == 16000
        assert agent.max_tokens == 16000


class TestFunctionCallingWarning:
    """The function-calling warning is for agents that actually have tools."""

    @staticmethod
    def _warnings_for(**kwargs):
        with patch(
            "swarms.structs.agent.supports_function_calling",
            return_value=False,
        ), patch("swarms.structs.agent.logger") as log:
            _patched_agent(
                "warn_agent", model_name="gpt-5.4", **kwargs
            )
        return " ".join(str(c) for c in log.warning.call_args_list)

    def test_no_tools_no_function_calling_warning(self):
        assert (
            "does not support function calling"
            not in self._warnings_for()
        )

    def test_with_tools_the_warning_still_fires(self):
        warnings = self._warnings_for(
            tools_list_dictionary=[{"function": {"name": "x"}}]
        )
        assert "does not support function calling" in warnings


class TestToolsListIsolation:
    """tools_list_dictionary must not be shared between Agent instances."""

    @staticmethod
    def _agent(name, **kwargs):
        return _patched_agent(name, model_name="gpt-5.4", **kwargs)

    def test_default_is_per_instance(self):
        """One agent's tools must not reach another, or the next one."""
        first = self._agent("First")
        second = self._agent("Second")
        first.tools_list_dictionary.append(
            {"function": {"name": "x"}}
        )

        assert second.tools_list_dictionary == []
        # a fresh agent too: the old pollution outlived agents on __defaults__
        assert self._agent("Third").tools_list_dictionary == []
        given = [{"function": {"name": "given"}}]
        assert (
            self._agent(
                "Fourth", tools_list_dictionary=given
            ).tools_list_dictionary
            == given
        )


class TestAutonomousAgentLoop:
    """The ``max_loops="auto"`` loop lives in ``AutonomousAgentLoop``, not on
    ``Agent``. These pin the seam: the loop is wired up, it reads and writes
    agent state through its back-reference, and ``Agent.run`` routes to it.
    """

    @staticmethod
    def _agent(max_loops="auto", **kwargs):
        return _patched_agent(
            "AutoLoopAgent",
            model_name="gpt-5.4",
            max_loops=max_loops,
            **kwargs,
        )

    def test_loop_is_constructed_and_back_references_agent(self):
        """Every agent owns a loop, and the loop can reach its agent."""
        agent = self._agent()
        assert isinstance(agent.autonomous_loop, AutonomousAgentLoop)
        assert agent.autonomous_loop.agent is agent

    def test_loop_owns_the_tool_methods_not_the_agent(self):
        """The moved methods live on the loop; Agent must not regrow them."""
        agent = self._agent()
        for name in (
            "_create_plan_tool",
            "_think_tool",
            "_subtask_done_tool",
            "_get_next_executable_subtask",
            "_all_subtasks_complete",
        ):
            assert hasattr(agent.autonomous_loop, name)
            assert not hasattr(agent, name)

    def test_agent_entry_point_delegates_to_the_loop(self):
        """Agent._run_autonomous_loop is a passthrough, not a rewrite."""
        agent = self._agent()
        with patch.object(
            agent.autonomous_loop,
            "_run_autonomous_loop",
            return_value="delegated",
        ) as loop_run:
            result = agent._run_autonomous_loop(task="do it")

        assert result == "delegated"
        loop_run.assert_called_once()
        assert loop_run.call_args.kwargs["task"] == "do it"

    def test_run_routes_auto_mode_to_the_loop(self):
        """max_loops="auto" reaches the loop; a fixed count does not."""
        agent = self._agent()
        with patch.object(
            agent.autonomous_loop,
            "_run_autonomous_loop",
            return_value="auto-path",
        ) as loop_run:
            assert agent.run("go") == "auto-path"
        loop_run.assert_called_once()

        fixed = self._agent(max_loops=1)
        with patch.object(
            fixed.autonomous_loop, "_run_autonomous_loop"
        ) as loop_run, patch.object(
            fixed, "_run", return_value="fixed-path"
        ):
            assert fixed.run("go") == "fixed-path"
        loop_run.assert_not_called()

    def test_all_subtasks_complete_reads_agent_state(self):
        """The loop's view of completion comes from the agent, not itself."""
        agent = self._agent()
        loop = agent.autonomous_loop

        agent.autonomous_subtasks = []
        assert loop._all_subtasks_complete() is False

        agent.autonomous_subtasks = [
            {"id": 1, "status": "completed"},
            {"id": 2, "status": "pending"},
        ]
        assert loop._all_subtasks_complete() is False

        # "failed" counts as finished -- the loop must not spin on it
        agent.autonomous_subtasks = [
            {"id": 1, "status": "completed"},
            {"id": 2, "status": "failed"},
        ]
        assert loop._all_subtasks_complete() is True

    def test_next_executable_subtask_respects_dependencies(self):
        """A pending subtask is only returned once its dependencies finish."""
        agent = self._agent()
        loop = agent.autonomous_loop

        agent.autonomous_subtasks = []
        agent.subtask_status = {}
        assert loop._get_next_executable_subtask() is None

        agent.autonomous_subtasks = [
            {"id": 1, "status": "pending", "dependencies": []},
            {"id": 2, "status": "pending", "dependencies": [1]},
        ]
        agent.subtask_status = {1: "pending"}
        assert loop._get_next_executable_subtask()["id"] == 1

        agent.autonomous_subtasks[0]["status"] = "completed"
        agent.subtask_status = {1: "completed"}
        assert loop._get_next_executable_subtask()["id"] == 2

        agent.autonomous_subtasks[1]["status"] = "completed"
        assert loop._get_next_executable_subtask() is None

    def test_loop_drives_plan_then_execution_against_a_stub_llm(self):
        """A canned LLM carries the loop from planning through completion."""
        prompts = []

        step = {
            "step_id": "s1",
            "description": "step one",
            "priority": "high",
            "dependencies": [],
        }
        plan = {"task_description": "the loop", "steps": [step]}
        done = {
            "task_id": "s1",
            "summary": "done",
            "success": True,
        }

        def call(name, arguments):
            fn = {"name": name, "arguments": json.dumps(arguments)}
            return [{"function": fn}]

        def fake_call_llm(self, task=None, *args, **kwargs):
            prompts.append(task)
            if len(prompts) == 1:
                return call("create_plan", plan)
            if len(prompts) <= 3:
                return call("subtask_done", done)
            return "FINAL"

        agent = self._agent()
        with patch.object(Agent, "call_llm", fake_call_llm):
            output = agent.run("exercise the loop")

        assert prompts, "the loop never called the LLM"
        assert output is not None
        # the plan the loop built lands on the agent, reached via self.agent
        assert agent.plan_created is True
        assert [s["step_id"] for s in agent.autonomous_subtasks] == [
            "s1"
        ]
        assert agent.subtask_status["s1"] == "completed"


class TestArunForwarding:
    """arun() must forward its arguments to run(), not await a sync method."""

    def _bare_agent(self):
        # No LLM/model setup — these tests only exercise arun's plumbing.
        agent = Agent.__new__(Agent)
        agent.autosave = False
        agent.agent_name = "arun-test"
        agent.to_dict = lambda: {}
        return agent

    def test_extra_positional_args_reach_run(self):
        """`task=`/`img=` as keywords alongside *args made every extra
        positional collide with `task`: arun(task, img, extra) raised
        `TypeError: got multiple values for argument 'task'`.
        """
        agent = self._bare_agent()
        seen = {}

        def fake_run(*args, **kwargs):
            seen["args"] = args
            seen["kwargs"] = kwargs
            return "ok"

        agent.run = fake_run

        result = asyncio.run(Agent.arun(agent, "T", "I", "EXTRA"))

        assert result == "ok"
        assert seen["args"] == ("T", "I", "EXTRA")

    def test_error_path_does_not_await_a_sync_handler(self):
        """`_handle_run_error` is sync and re-raises; the await was harmless
        only because it always raises before returning.
        """
        import inspect

        assert not inspect.iscoroutinefunction(
            Agent._handle_run_error
        )
        assert (
            "await self._handle_run_error"
            not in inspect.getsource(Agent.arun)
        )

        agent = self._bare_agent()

        def boom(*args, **kwargs):
            raise ValueError("boom")

        agent.run = boom

        # The original error surfaces — not a TypeError from awaiting None.
        with pytest.raises(ValueError, match="boom"):
            asyncio.run(Agent.arun(agent, "T"))


class TestEmptyTaskGuard:
    """run() must not read stdin for an empty task when interactive=False."""

    @pytest.mark.parametrize("empty_task", ["", "   ", "\n\t ", None])
    def test_non_interactive_empty_task_raises(self, empty_task):
        """A commented-out `self.interactive and ...` left the guard
        unconditional, so an empty task blocked on console input. Raising
        proves the prompt, which follows this branch, is never reached.
        """
        agent = Agent.__new__(Agent)
        agent.interactive = False

        with pytest.raises(ValueError, match="No task provided"):
            Agent.run(agent, empty_task)


class TestToolExecutionRetry:
    """#1794: it called execute_tools once regardless of tool_retry_attempts,
    caught only AgentToolExecutionError — which nothing raises — and returned,
    so a failed tool run left no Tool Executor entry and the model carried on
    as though the call had succeeded.
    """

    @staticmethod
    def _agent(attempts=3, name="A"):
        """A bare Agent carrying only what tool_execution_retry reads."""
        agent = Agent.__new__(Agent)
        agent.agent_name = name
        agent.tool_retry_attempts = attempts
        return agent

    @pytest.mark.parametrize("attempts", [3, 5])
    def test_retries_up_to_the_configured_attempts(self, attempts):
        agent = self._agent(attempts=attempts)
        calls = []

        def failing(response, loop_count):
            calls.append(loop_count)
            raise RuntimeError("simulated tool failure")

        agent.execute_tools = failing
        with pytest.raises(AgentToolExecutionError):
            Agent.tool_execution_retry(agent, [{"function": {}}], 1)

        # The regression: this was 1 regardless of tool_retry_attempts.
        assert len(calls) == attempts

    def test_failure_is_raised_not_swallowed(self):
        """Returning None hid tool failures: no error, and no Tool Executor
        entry in short_memory."""
        agent = self._agent(attempts=1)

        def failing(response, loop_count):
            raise RuntimeError("simulated tool failure")

        agent.execute_tools = failing
        with pytest.raises(AgentToolExecutionError) as excinfo:
            Agent.tool_execution_retry(agent, [{"function": {}}], 1)

        # The underlying error is chained, not discarded.
        assert isinstance(excinfo.value.__cause__, RuntimeError)
        assert "simulated tool failure" in str(
            excinfo.value.__cause__
        )

    def test_catches_the_exception_types_actually_raised(self):
        """Nothing raises AgentToolExecutionError, so catching only that type
        caught nothing. A ValueError must be retried like any other.
        """
        agent = self._agent(attempts=2)
        calls = []

        def failing(response, loop_count):
            calls.append(1)
            raise ValueError("a tool's own error type")

        agent.execute_tools = failing
        with pytest.raises(AgentToolExecutionError):
            Agent.tool_execution_retry(agent, [{"function": {}}], 1)
        assert len(calls) == 2

    def test_stops_retrying_once_a_attempt_succeeds(self):
        agent = self._agent(attempts=4)
        calls = []

        def flaky(response, loop_count):
            calls.append(1)
            if len(calls) < 3:
                raise RuntimeError("transient")

        agent.execute_tools = flaky
        Agent.tool_execution_retry(agent, [{"function": {}}], 1)

        # Third attempt succeeded, so the fourth must not run.
        assert len(calls) == 3

    def test_none_response_does_not_execute_or_raise(self):
        agent = self._agent()
        called = []
        agent.execute_tools = lambda **kw: called.append(1)

        Agent.tool_execution_retry(agent, None, 1)
        assert called == []

    def test_a_zero_or_none_attempt_count_still_runs_once(self):
        """0 must not mean 'never run the tools', which would silently
        disable tool execution entirely.
        """
        for attempts in (0, None):
            agent = self._agent(attempts=attempts)
            calls = []
            agent.execute_tools = (
                lambda response, loop_count: calls.append(1)
            )
            Agent.tool_execution_retry(agent, [{"function": {}}], 1)
            assert (
                len(calls) == 1
            ), f"attempts={attempts!r} should still run once"


class TestToolFailureIsNotAnLLMError:
    """#1924: a tool that exhausted its own retries raised into the generation
    handler, which logged it as Agent.llm_error and re-ran the model, so one
    broken tool cost up to retry_attempts extra completions.
    """

    @staticmethod
    def _broken_tool(query: str) -> str:
        """Always fails.

        Args:
            query: anything at all.
        """
        raise RuntimeError("tool is broken")

    def test_a_failing_tool_does_not_re_run_the_model(self):
        agent = _patched_agent(
            "ToolFailAgent",
            model_name="gpt-5.4",
            dynamic_tools=False,
            tools=[self._broken_tool],
            retry_attempts=3,
        )

        llm_calls = []

        def fake_call_llm(task=None, *args, **kwargs):
            llm_calls.append(task)
            return [
                {
                    "type": "function",
                    "id": "call-1",
                    "function": {
                        "name": "_broken_tool",
                        "arguments": '{"query": "x"}',
                    },
                }
            ]

        agent.call_llm = fake_call_llm
        agent.run("use the tool")

        assert (
            len(llm_calls) == 1
        ), f"a tool failure re-ran the model {len(llm_calls)} times"


class TestConcurrentExecutionPool:
    """#1793: both concurrent entry points referenced self.executor, which
    __init__ never assigned — run_concurrent_tasks swallowed the AttributeError
    and returned None, talk_to_multiple_agents raised it. Both now build a
    call-scoped pool.
    """

    @staticmethod
    def _agent(name="A"):
        """A bare Agent — no __init__, so no client or provider calls."""
        agent = Agent.__new__(Agent)
        agent.agent_name = name
        return agent

    def test_no_executor_attribute_is_required(self):
        """The concurrent paths must not depend on an attribute __init__
        does not set."""
        agent = self._agent()
        assert not hasattr(agent, "executor")

    def test_run_concurrent_tasks_returns_one_result_per_task_in_order(
        self,
    ):
        agent = self._agent()
        with patch.object(
            Agent,
            "run",
            side_effect=lambda task, *a, **kw: f"ran:{task}",
        ):
            results = Agent.run_concurrent_tasks(
                agent, ["t1", "t2", "t3"]
            )

        # Order matters: results are zipped against the caller's task list.
        assert results == ["ran:t1", "ran:t2", "ran:t3"]

    def test_run_concurrent_tasks_propagates_failure(self):
        """The except branch logged and fell through, so a failed batch
        returned None and the caller saw no error."""
        agent = self._agent()
        with patch.object(
            Agent, "run", side_effect=RuntimeError("boom")
        ):
            with pytest.raises(RuntimeError, match="boom"):
                Agent.run_concurrent_tasks(agent, ["t1"])

    def test_talk_to_multiple_agents_returns_one_entry_per_agent(
        self,
    ):
        agent = self._agent()
        others = [self._agent("B"), self._agent("C")]
        with patch.object(
            Agent,
            "talk_to",
            side_effect=lambda other, task, *a, **kw: f"to:{other.agent_name}",
        ):
            outputs = Agent.talk_to_multiple_agents(
                agent, others, "hi"
            )

        assert outputs == ["to:B", "to:C"]

    def test_talk_to_multiple_agents_isolates_a_failing_agent(self):
        """One bad conversation contributes None; the others still return."""
        agent = self._agent()
        others = [self._agent("B"), self._agent("C")]

        def talk(other, task, *a, **kw):
            if other.agent_name == "B":
                raise RuntimeError("dead")
            return f"to:{other.agent_name}"

        with patch.object(Agent, "talk_to", side_effect=talk):
            outputs = Agent.talk_to_multiple_agents(
                agent, others, "hi"
            )

        assert outputs == [None, "to:C"]

    def test_pool_is_shut_down_after_each_call(self):
        """Call-scoped, so a reused agent must not leak a pool per call."""
        agent = self._agent()
        before = threading.active_count()
        with patch.object(
            Agent, "run", side_effect=lambda task, *a, **kw: task
        ):
            for _ in range(3):
                Agent.run_concurrent_tasks(agent, ["t1", "t2"])

        # Worker threads are joined on __exit__; allow a moment for teardown.
        for _ in range(50):
            if threading.active_count() <= before:
                break
            time.sleep(0.02)
        assert threading.active_count() <= before


class TestRunBatchedImagePairing:
    """`for task, imgs in zip(tasks, imgs)` rebound the parameter, so `imgs`
    received a bare str — and with the documented imgs=None default the zip
    raised before any task ran.
    """

    @staticmethod
    def _agent():
        agent = Agent.__new__(Agent)
        agent.agent_name = "batched"
        return agent

    def test_tasks_without_images_run(self):
        """The documented default: imgs is optional."""
        agent = self._agent()
        with patch.object(Agent, "run", side_effect=lambda **kw: kw):
            assert Agent.run_batched(agent, ["t1", "t2"]) == [
                {"task": "t1"},
                {"task": "t2"},
            ]

    def test_each_task_gets_its_own_image_as_a_single_image(self):
        agent = self._agent()
        with patch.object(Agent, "run", side_effect=lambda **kw: kw):
            assert Agent.run_batched(
                agent, ["t1", "t2"], imgs=["a.png", "b.png"]
            ) == [
                {"task": "t1", "img": "a.png"},
                {"task": "t2", "img": "b.png"},
            ]

    def test_mismatched_lengths_raise_instead_of_dropping_tasks(self):
        """zip() would have run one task and discarded the rest in silence."""
        agent = self._agent()
        with patch.object(Agent, "run", side_effect=lambda **kw: kw):
            with pytest.raises(
                ValueError, match="one image per task"
            ):
                Agent.run_batched(
                    agent, ["t1", "t2", "t3"], imgs=["a.png"]
                )


class TestEmptyToolsList:
    """`tools=[]` must mean the same as `tools=None`.

    `exists()` is `is not None`, so an empty list counted as having tools and
    bought the agent a `tool_search` schema with nothing to search.
    """

    @staticmethod
    def _agent(**kwargs):
        return Agent(
            agent_name="empty_tools_agent",
            model_name="gpt-4o-mini",
            max_loops=1,
            **kwargs,
        )

    def test_empty_list_advertises_no_tools(self):
        assert self._agent(tools=[]).tools_list_dictionary == []

    def test_empty_list_matches_none(self):
        empty = self._agent(tools=[])
        none = self._agent(tools=None)
        assert (
            empty.tools_list_dictionary == none.tools_list_dictionary
        )
        assert ("tool_search" in empty.system_prompt) == (
            "tool_search" in none.system_prompt
        )

    def test_a_real_tool_still_defers(self):
        def sample(x: str) -> str:
            """Return x.

            Args:
                x: anything
            """
            return x

        agent = self._agent(tools=[sample])
        assert agent.tools_list_dictionary
        assert "tool_search" in agent.system_prompt


class TestAgentUsage:
    """``agent.usage`` is the provider's token count, summed over the agent's calls."""

    @staticmethod
    def _response(prompt, completion, cached=0):
        from types import SimpleNamespace

        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content="answer", tool_calls=None
                    )
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=prompt,
                completion_tokens=completion,
                total_tokens=prompt + completion,
                prompt_tokens_details=(
                    SimpleNamespace(cached_tokens=cached)
                    if cached
                    else None
                ),
            ),
        )

    @staticmethod
    def _agent(**kwargs):
        return Agent(
            agent_name="UsageAgent",
            model_name="gpt-5.4",
            max_loops=1,
            print_on=False,
            verbose=False,
            persistent_memory=False,
            autosave=False,
            **kwargs,
        )

    def test_starts_at_zero_and_returns_a_copy(self):
        agent = _patched_agent("UsageAgent")
        usage = agent.usage
        assert usage == {
            "input_tokens": 0,
            "output_tokens": 0,
            "cached_tokens": 0,
            "total_tokens": 0,
        }
        usage["input_tokens"] = 999
        assert agent.usage["input_tokens"] == 0

    def test_a_run_reports_what_the_provider_returned(self):
        agent = self._agent()
        with patch(
            "swarms.utils.litellm_wrapper.completion",
            return_value=self._response(120, 30, cached=100),
        ):
            agent.run("Say hi.")

        assert agent.usage == {
            "input_tokens": 120,
            "output_tokens": 30,
            "cached_tokens": 100,
            "total_tokens": 150,
        }

    def test_totals_survive_an_llm_rebuild(self):
        agent = self._agent()
        with patch(
            "swarms.utils.litellm_wrapper.completion",
            return_value=self._response(10, 5),
        ):
            agent.run("one")
            # tool_search and fallback models rebuild the client mid-run.
            agent.llm = agent.llm_handling()
            agent.run("two")

        assert agent.usage["total_tokens"] == 30

    def test_a_caller_supplied_llm_reports_into_the_agent(self):
        from swarms.utils.litellm_wrapper import LiteLLM

        agent = self._agent(llm=LiteLLM(model_name="gpt-5.4"))
        with patch(
            "swarms.utils.litellm_wrapper.completion",
            return_value=self._response(4, 4),
        ):
            agent.run("x")

        assert agent.usage["total_tokens"] == 8

    def test_every_call_in_a_multi_loop_tool_run_is_counted(self):
        """Each loop makes two provider calls — the tool call and the tool
        summary — and all of them land in the total."""
        import json

        def add(a: int, b: int) -> int:
            """Add two numbers.

            Args:
                a: first
                b: second
            """
            return a + b

        calls = []

        def fake_completion(**kwargs):
            calls.append(kwargs)
            n = len(calls)
            if kwargs.get("tools"):
                tool_call = {
                    "id": f"call_{n}",
                    "type": "function",
                    "function": {
                        "name": "add",
                        "arguments": json.dumps({"a": 1, "b": 2}),
                    },
                }
                response = self._response(100 * n, n)
                response.choices[0].message.content = None
                response.choices[0].message.tool_calls = [tool_call]
                return response
            return self._response(100 * n, n)

        agent = self._agent(tools=[add], dynamic_tools=False)
        agent.max_loops = 3
        with patch(
            "swarms.utils.litellm_wrapper.completion",
            side_effect=fake_completion,
        ):
            agent.run("Add 1 and 2.")

        # 3 loops x (tool call + summary call), with no call skipped.
        assert len(calls) == 6
        assert agent.usage == {
            "input_tokens": sum(100 * i for i in range(1, 7)),
            "output_tokens": sum(range(1, 7)),
            "cached_tokens": 0,
            "total_tokens": sum(100 * i + i for i in range(1, 7)),
        }
