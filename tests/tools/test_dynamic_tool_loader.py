"""
Tests for :class:`swarms.tools.dynamic_tool_loader.DynamicToolLoader`.

Fully offline - the loader does no I/O and calls no model. Matching is pure
token overlap by design, so it is deterministic and can be asserted on
directly.

Run:
    cd /Users/swarms_wd/Desktop/research/swarms
    PYTHONPATH=. python3 -m pytest tests/tools/test_dynamic_tool_loader.py -q -p no:randomly
"""

import json

import pytest

from swarms import Agent
from swarms.tools.dynamic_tool_loader import (
    SEARCH_TOOL_NAME,
    DynamicToolLoader,
)


def get_weather(city: str) -> str:
    """Get the current weather for a city.

    Args:
        city: The city name.
    """
    return "sunny"


def send_email(to: str, subject: str, body: str) -> str:
    """Send an email message to a recipient.

    Args:
        to: Recipient address.
        subject: Subject line.
        body: Message body.
    """
    return "sent"


def read_csv(path: str) -> str:
    """Read a CSV file and return its rows.

    Args:
        path: Path to the CSV file.
    """
    return ""


@pytest.fixture
def loader():
    return DynamicToolLoader(
        tools=[get_weather, send_email, read_csv]
    )


def names(loader):
    return [s["function"]["name"] for s in loader.schemas()]


class TestDeferral:
    def test_nothing_is_exposed_except_the_search_tool(self, loader):
        assert names(loader) == [SEARCH_TOOL_NAME]
        assert len(loader) == 3
        assert loader.loaded_names == []

    def test_always_loaded_tools_are_never_deferred(self):
        control = {
            "type": "function",
            "function": {"name": "complete_task", "description": "d"},
        }
        loader = DynamicToolLoader(
            tools=[get_weather], always_loaded=[control]
        )
        assert names(loader) == ["complete_task", SEARCH_TOOL_NAME]

    def test_loading_exposes_the_tool(self, loader):
        loader.run_search("weather")
        assert "get_weather" in names(loader)
        assert loader.loaded_names == ["get_weather"]
        assert "get_weather" not in loader.deferred_names


class TestSearch:
    def test_matches_on_description(self, loader):
        assert [t.name for t in loader.search("weather")] == [
            "get_weather"
        ]

    def test_matches_on_parameter_names(self, loader):
        assert "send_email" in [
            t.name for t in loader.search("recipient subject")
        ]

    def test_stopwords_do_not_match_everything(self, loader):
        """
        Without stopword filtering, "weather in a city" matches every tool
        whose description contains "a", loading the whole catalog and
        defeating the point of deferring.
        """
        assert [
            t.name for t in loader.search("weather in a city")
        ] == ["get_weather"]
        assert loader.search("please can you help with the") == []

    def test_name_matches_outrank_description_matches(self):
        loader = DynamicToolLoader()
        loader.register_schema(
            {
                "type": "function",
                "function": {
                    "name": "unrelated",
                    "description": "Mentions weather in passing.",
                    "parameters": {},
                },
            }
        )
        loader.register_schema(
            {
                "type": "function",
                "function": {
                    "name": "weather",
                    "description": "Nothing relevant here.",
                    "parameters": {},
                },
            }
        )
        assert [t.name for t in loader.search("weather")][
            0
        ] == "weather"

    def test_results_are_stable(self, loader):
        """A stable order keeps the cached prompt prefix stable across runs."""
        first = [t.name for t in loader.search("a message file city")]
        second = [
            t.name for t in loader.search("a message file city")
        ]
        assert first == second

    def test_limit_is_respected(self, loader):
        assert len(loader.search("city recipient file", limit=1)) == 1


class TestSelectByName:
    def test_exact_names_load_without_matching(self, loader):
        loader.run_search("select:send_email,read_csv")
        assert loader.loaded_names == ["read_csv", "send_email"]

    def test_unknown_names_are_ignored(self, loader):
        loader.run_search("select:send_email,does_not_exist")
        assert loader.loaded_names == ["send_email"]


class TestSearchResultText:
    def test_reports_what_it_loaded(self, loader):
        result = loader.run_search("weather")
        assert "get_weather" in result
        assert "callable from your next turn" in result

    def test_repeat_search_does_not_reload(self, loader):
        loader.run_search("weather")
        result = loader.run_search("weather")
        assert "already loaded" in result

    def test_a_miss_lists_what_is_available(self, loader):
        result = loader.run_search("quantum tunnelling")
        assert "No tools matched" in result
        for name in ("get_weather", "send_email", "read_csv"):
            assert name in result
        assert "select:" in result

    def test_a_miss_caps_the_listing(self):
        loader = DynamicToolLoader()
        for i in range(50):
            loader.register_schema(
                {
                    "type": "function",
                    "function": {
                        "name": f"tool_{i:02d}",
                        "description": f"Does thing {i}.",
                        "parameters": {},
                    },
                }
            )
        result = loader.run_search("nonexistent")
        assert "more)" in result, "the full catalog was dumped"


class TestHandlers:
    def test_only_loaded_tools_expose_handlers(self, loader):
        assert loader.handlers() == {}
        loader.run_search("weather")
        assert set(loader.handlers()) == {"get_weather"}
        assert loader.handlers()["get_weather"]("Paris") == "sunny"

    def test_schema_only_tools_have_no_handler(self):
        """MCP-style entries are dispatched elsewhere."""
        loader = DynamicToolLoader(
            schemas=[
                {
                    "type": "function",
                    "function": {
                        "name": "remote",
                        "description": "A remote tool.",
                        "parameters": {},
                    },
                }
            ]
        )
        loader.run_search("remote")
        assert loader.handlers() == {}
        assert "remote" in loader.loaded_names


class TestTokenCost:
    def test_deferring_is_cheaper_than_exposing(self, loader):
        import json

        deferred = len(json.dumps(loader.schemas()))
        loader.run_search("select:get_weather,send_email,read_csv")
        loaded = len(json.dumps(loader.schemas()))
        assert (
            deferred < loaded
        ), "deferring should send strictly less than exposing everything"


# --------------------------------------------------------------------------
# Integration with Agent
# --------------------------------------------------------------------------


def tool_call(name, call_id="c1", **arguments):
    return {
        "id": call_id,
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def exposed(agent):
    """Tool names the model would actually be sent."""
    return [
        t["function"]["name"]
        for t in (agent.tools_list_dictionary or [])
        if isinstance(t, dict) and "function" in t
    ]


def script(agent, monkeypatch, responses):
    """Drive the agent with a fixed sequence of model responses."""
    queue = list(responses)
    seen = []

    def fake(task=None, *args, **kwargs):
        seen.append(exposed(agent))
        return queue.pop(0) if queue else "done"

    monkeypatch.setattr(agent, "call_llm", fake)
    return seen


def build(**kwargs):
    kwargs.setdefault("agent_name", "DynIntegration")
    kwargs.setdefault("model_name", "gpt-4o-mini")
    kwargs.setdefault("persistent_memory", False)
    kwargs.setdefault("context_compression", False)
    kwargs.setdefault("print_on", False)
    kwargs.setdefault("verbose", False)
    kwargs.setdefault("autosave", False)
    kwargs.setdefault("tool_call_summary", False)
    return Agent(**kwargs)


class TestAgentIntegration:
    """`Agent(dynamic_tools=True)` defers instead of registering eagerly."""

    def test_on_by_default(self):
        """Deferring is the default; every tool goes into the catalog."""
        agent = build(max_loops=1, tools=[get_weather])
        assert agent.dynamic_tools is True
        assert exposed(agent) == [SEARCH_TOOL_NAME]

    def test_can_be_turned_off(self):
        """dynamic_tools=False keeps the classic eager registration."""
        agent = build(
            max_loops=1, tools=[get_weather], dynamic_tools=False
        )
        assert agent.tool_loader is None
        assert "get_weather" in exposed(agent)

    def test_on_defers_user_tools(self):
        agent = build(
            max_loops=1, tools=[get_weather], dynamic_tools=True
        )
        assert exposed(agent) == [SEARCH_TOOL_NAME]
        assert "get_weather" in agent.tool_loader.deferred_names

    def test_search_loads_and_exposes(self):
        agent = build(
            max_loops=1, tools=[get_weather], dynamic_tools=True
        )
        agent._tool_search_tool(query="weather")
        assert "get_weather" in exposed(agent)

    def test_the_prompt_says_tools_need_loading(self):
        """
        Deferring without saying so makes the prompt lie: the model believes
        the tools it can see are everything and gives up instead of searching.
        """
        agent = build(
            max_loops=1, tools=[get_weather], dynamic_tools=True
        )
        assert "MOST TOOLS ARE NOT LOADED" in agent.system_prompt
        assert SEARCH_TOOL_NAME in agent.system_prompt

    def test_the_notice_is_appended_once_not_per_run(
        self, monkeypatch
    ):
        """Per-run prompt mutation is how a system prompt grows unbounded."""
        agent = build(
            max_loops=1, tools=[get_weather], dynamic_tools=True
        )
        script(agent, monkeypatch, ["done", "done"])
        agent.run("one")
        agent.run("two")
        assert (
            agent.system_prompt.count("MOST TOOLS ARE NOT LOADED")
            == 1
        )

    def test_no_notice_when_there_is_nothing_to_defer(self):
        agent = build(max_loops=1, dynamic_tools=True)
        assert "MOST TOOLS ARE NOT LOADED" not in agent.system_prompt

    def test_search_without_a_loader_explains_itself(self):
        agent = build(
            max_loops=1, tools=[get_weather], dynamic_tools=False
        )
        result = agent._tool_search_tool(query="weather")
        assert "dynamic_tools=True" in result

    def test_fixed_loop_dispatches_tool_search(self, monkeypatch):
        """
        tool_search is an agent method, not one of the user's callables, so
        _run has to intercept it rather than routing it through tool_struct.
        """
        agent = build(
            max_loops=2, tools=[get_weather], dynamic_tools=True
        )
        seen = script(
            agent,
            monkeypatch,
            [[tool_call(SEARCH_TOOL_NAME, query="weather")], "done"],
        )
        agent.run("weather?")

        assert seen[0] == [SEARCH_TOOL_NAME]
        assert (
            "get_weather" in seen[1]
        ), "the loaded tool was not sent on the following turn"


class TestAutonomousLoopIntegration:
    """max_loops="auto" defers its own tools behind tool_search."""

    def _auto(self, **kwargs):
        return build(max_loops="auto", dynamic_tools=True, **kwargs)

    def test_control_tools_are_never_deferred(self, monkeypatch):
        """An agent that must search for its own subtask_done cannot finish."""
        agent = self._auto()
        seen = script(
            agent,
            monkeypatch,
            [
                [
                    tool_call(
                        "complete_task",
                        task_id="m",
                        summary="d",
                        success=True,
                    )
                ]
            ],
        )
        agent.run("demo")

        first_turn = seen[0]
        for control in (
            "create_plan",
            "subtask_done",
            "complete_task",
            "respond_to_user",
        ):
            assert control in first_turn, f"{control} was deferred"
        assert SEARCH_TOOL_NAME in first_turn

    def test_worker_tools_are_deferred(self, monkeypatch):
        agent = self._auto()
        script(
            agent,
            monkeypatch,
            [
                [
                    tool_call(
                        "complete_task",
                        task_id="m",
                        summary="d",
                        success=True,
                    )
                ]
            ],
        )
        agent.run("demo")

        deferred = agent.tool_loader.deferred_names
        for worker in (
            "create_file",
            "read_file",
            "run_bash",
            "grep",
        ):
            assert worker in deferred, f"{worker} was exposed eagerly"

    def test_user_tools_are_deferred_too(self, monkeypatch):
        agent = self._auto(tools=[get_weather])
        seen = script(
            agent,
            monkeypatch,
            [
                [
                    tool_call(
                        "complete_task",
                        task_id="m",
                        summary="d",
                        success=True,
                    )
                ]
            ],
        )
        agent.run("demo")
        assert "get_weather" not in seen[0]
        assert "get_weather" in agent.tool_loader.deferred_names

    def test_a_deferred_tool_becomes_usable_after_search(
        self, monkeypatch, tmp_path
    ):
        """The whole point, end to end: search, then use what was loaded."""
        agent = self._auto()
        target = tmp_path / "written.txt"

        seen = script(
            agent,
            monkeypatch,
            [
                # plan
                [
                    tool_call(
                        "create_plan",
                        task_description="t",
                        steps=[
                            {
                                "step_id": "s1",
                                "description": "proceed as needed",
                                "priority": "high",
                                "dependencies": [],
                            }
                        ],
                    )
                ],
                # the file tools are not loaded yet, so search for them
                [
                    tool_call(
                        SEARCH_TOOL_NAME, query="create write file"
                    )
                ],
                # now callable
                [
                    tool_call(
                        "create_file",
                        file_path=str(target),
                        content="payload",
                    )
                ],
                [
                    tool_call(
                        "subtask_done",
                        task_id="s1",
                        summary="done",
                        success=True,
                    )
                ],
                [
                    tool_call(
                        "complete_task",
                        task_id="m",
                        summary="d",
                        success=True,
                    )
                ],
            ],
        )
        agent.run("proceed")

        assert (
            "create_file" not in seen[1]
        ), "precondition: it was deferred, not pre-warmed"
        assert (
            "create_file" in seen[2]
        ), "the search did not expose it"
        assert target.exists(), "the loaded tool never ran"
        assert target.read_text() == "payload"

    def test_the_search_result_reaches_the_model(
        self, monkeypatch, tmp_path
    ):
        """It must land in the transcript as a tool result, not just memory."""
        agent = self._auto()
        script(
            agent,
            monkeypatch,
            [
                [
                    tool_call(
                        "create_plan",
                        task_description="t",
                        steps=[
                            {
                                "step_id": "s1",
                                "description": "d",
                                "priority": "high",
                                "dependencies": [],
                            }
                        ],
                    )
                ],
                [tool_call(SEARCH_TOOL_NAME, query="file")],
                [
                    tool_call(
                        "subtask_done",
                        task_id="s1",
                        summary="done",
                        success=True,
                    )
                ],
                [
                    tool_call(
                        "complete_task",
                        task_id="m",
                        summary="d",
                        success=True,
                    )
                ],
            ],
        )
        agent.run("demo")

        transcript = agent.autonomous_loop._transcript.messages
        tool_results = [
            m["content"] for m in transcript if m["role"] == "tool"
        ]
        assert any(
            "Loaded" in str(c) or "already loaded" in str(c)
            for c in tool_results
        ), "the tool_search result never reached the transcript"


class TestRelevanceThreshold:
    """Speculative loading has to demand stronger relevance than a search."""

    def test_ratio_drops_weak_matches(self, loader):
        """
        A long query contains enough common words to give unrelated tools a
        nonzero score. Without a threshold, pre-warming from a whole plan
        loads most of the catalog and undoes the saving.
        """
        query = (
            "get the current weather for a city and send a message"
        )
        loose = loader.search(query, limit=10)
        strict = loader.search(query, limit=10, min_score_ratio=0.6)

        assert len(strict) < len(
            loose
        ), "the threshold changed nothing"
        assert strict, "the threshold dropped everything"

    def test_zero_ratio_keeps_every_match(self, loader):
        query = "weather city recipient file"
        assert loader.search(query, limit=10) == loader.search(
            query, limit=10, min_score_ratio=0.0
        )


class TestPlanPrewarming:
    """The plan pre-loads the tools it implies, at no extra turn."""

    def _auto(self, **kwargs):
        return build(max_loops="auto", dynamic_tools=True, **kwargs)

    def _plan_call(self, *descriptions):
        return tool_call(
            "create_plan",
            task_description=" ".join(descriptions),
            steps=[
                {
                    "step_id": f"s{i}",
                    "description": d,
                    "priority": "high",
                    "dependencies": [],
                }
                for i, d in enumerate(descriptions)
            ],
        )

    def test_planning_loads_the_tools_the_plan_implies(
        self, monkeypatch
    ):
        agent = self._auto(tools=[get_weather])
        seen = script(
            agent,
            monkeypatch,
            [
                [
                    self._plan_call(
                        "Get the current weather for Tokyo"
                    )
                ],
                [
                    tool_call(
                        "complete_task",
                        task_id="m",
                        summary="d",
                        success=True,
                    )
                ],
            ],
        )
        agent.run("weather?")

        assert (
            "get_weather" not in seen[0]
        ), "precondition: it was deferred"
        assert (
            "get_weather" in agent.tool_loader.loaded_names
        ), "planning did not pre-load the tool the plan implies"
        assert (
            "get_weather" in seen[1]
        ), "the pre-loaded tool was not sent on the next turn"

    def test_prewarming_does_not_load_the_whole_catalog(
        self, monkeypatch
    ):
        """The saving is the point; a greedy pre-warm would undo it."""
        agent = self._auto(tools=[get_weather])
        script(
            agent,
            monkeypatch,
            [
                [
                    self._plan_call(
                        "Get the current weather for Tokyo"
                    )
                ],
                [
                    tool_call(
                        "complete_task",
                        task_id="m",
                        summary="d",
                        success=True,
                    )
                ],
            ],
        )
        agent.run("weather?")

        loaded = agent.tool_loader.loaded_names
        assert len(loaded) <= 3, f"pre-warm loaded too much: {loaded}"
        assert len(agent.tool_loader.deferred_names) > 5

    def test_the_model_is_told_what_was_preloaded(self, monkeypatch):
        """Otherwise it searches again for tools it already has."""
        agent = self._auto(tools=[get_weather])
        script(
            agent,
            monkeypatch,
            [
                [
                    self._plan_call(
                        "Get the current weather for Tokyo"
                    )
                ],
                [
                    tool_call(
                        "complete_task",
                        task_id="m",
                        summary="d",
                        success=True,
                    )
                ],
            ],
        )
        agent.run("weather?")

        results = [
            str(m["content"])
            for m in agent.autonomous_loop._transcript.messages
            if m["role"] == "tool"
        ]
        assert any(
            "Pre-loaded" in r for r in results
        ), "the plan result did not report what it pre-loaded"

    def test_prewarming_is_skipped_without_dynamic_tools(
        self, monkeypatch
    ):
        agent = build(
            max_loops="auto", tools=[get_weather], dynamic_tools=False
        )
        script(
            agent,
            monkeypatch,
            [
                [
                    self._plan_call(
                        "Get the current weather for Tokyo"
                    )
                ],
                [
                    tool_call(
                        "complete_task",
                        task_id="m",
                        summary="d",
                        success=True,
                    )
                ],
            ],
        )
        agent.run("weather?")
        assert agent.tool_loader is None


class TestDoesNotClobberOtherTools:
    """Deferral must not silently drop tools registered by other features."""

    def test_handoff_tools_survive(self):
        """
        Regression: setup_dynamic_tools assigned schemas() straight over
        tools_list_dictionary, discarding the handoff tool registered just
        above it - so delegation stopped working entirely, silently.
        """
        worker = build(agent_name="Worker", max_loops=1)
        agent = build(
            max_loops=1,
            tools=[get_weather],
            handoffs=[worker],
            dynamic_tools=True,
        )
        assert "handoff_task" in exposed(agent)

    def test_handoff_tools_survive_the_autonomous_loop(
        self, monkeypatch
    ):
        worker = build(agent_name="Worker", max_loops=1)
        agent = build(
            max_loops="auto", handoffs=[worker], dynamic_tools=True
        )
        script(
            agent,
            monkeypatch,
            [
                [
                    tool_call(
                        "complete_task",
                        task_id="m",
                        summary="d",
                        success=True,
                    )
                ]
            ],
        )
        agent.run("demo")
        assert "handoff_task" in exposed(agent)

    def test_user_tools_are_still_deferred_alongside(self):
        worker = build(agent_name="Worker", max_loops=1)
        agent = build(
            max_loops=1,
            tools=[get_weather],
            handoffs=[worker],
            dynamic_tools=True,
        )
        assert "get_weather" not in exposed(agent)
        assert "get_weather" in agent.tool_loader.deferred_names


class TestNoDuplicateSchemas:
    """The tool array must never list the same name twice."""

    def test_search_tool_appears_once_after_two_setups(
        self, monkeypatch
    ):
        """
        setup_dynamic_tools runs twice for an autonomous agent - once in
        __init__ and once when the loop starts - and the second pass preserves
        what the first registered. Without excluding the search tool from that
        carry-over it ends up in the array twice.
        """
        agent = build(
            max_loops="auto", tools=[get_weather], dynamic_tools=True
        )
        script(
            agent,
            monkeypatch,
            [
                [
                    tool_call(
                        "complete_task",
                        task_id="m",
                        summary="d",
                        success=True,
                    )
                ]
            ],
        )
        agent.run("demo")

        listed = exposed(agent)
        assert len(listed) == len(
            set(listed)
        ), f"duplicate tool names in the request: {listed}"
        assert listed.count(SEARCH_TOOL_NAME) == 1


class TestNameCollision:
    """A catalog entry may not shadow the search tool."""

    def test_a_tool_named_tool_search_is_rejected(self):
        def tool_search(query: str) -> str:
            """A user tool colliding with the search tool name.

            Args:
                query: anything.
            """
            return "user version"

        agent = build(
            max_loops=1, tools=[tool_search], dynamic_tools=True
        )
        # Exactly one tool_search, and it is the real one.
        assert exposed(agent).count(SEARCH_TOOL_NAME) == 1
        assert agent.tool_loader.deferred_names == []

    def test_the_loader_rejects_it_directly(self):
        loader = DynamicToolLoader()
        loader.register_schema(
            {
                "type": "function",
                "function": {
                    "name": SEARCH_TOOL_NAME,
                    "description": "shadow",
                    "parameters": {},
                },
            }
        )
        assert SEARCH_TOOL_NAME not in loader


class TestSelectedToolsInteraction:
    """selected_tools filters first; whatever survives is then deferred."""

    def test_filtered_tools_are_not_in_the_catalog(self, monkeypatch):
        agent = build(
            max_loops="auto",
            dynamic_tools=True,
            selected_tools=[
                "create_plan",
                "complete_task",
                "read_file",
            ],
        )
        script(
            agent,
            monkeypatch,
            [
                [
                    tool_call(
                        "complete_task",
                        task_id="m",
                        summary="d",
                        success=True,
                    )
                ]
            ],
        )
        agent.run("demo")

        assert "read_file" in agent.tool_loader.deferred_names
        assert "run_bash" not in agent.tool_loader.deferred_names
        assert "run_bash" not in exposed(agent)


MCP_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "slack_post_message",
            "description": "Post a message to a Slack channel.",
            "parameters": {},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "jira_search",
            "description": "Search Jira issues with JQL.",
            "parameters": {},
        },
    },
]


@pytest.fixture
def mcp_agent(request):
    """
    An agent with a stubbed MCP surface.

    `mcp_enabled` is a property, so it has to be patched on the class; it is
    restored afterwards to keep the pollution out of other tests. The agent is
    built with MCP off so construction does not try to reach a server.
    """
    original = Agent.mcp_enabled
    dynamic = getattr(request, "param", True)

    agent = build(max_loops=1, tools=[], dynamic_tools=dynamic)
    Agent.mcp_enabled = property(lambda self: True)
    agent.add_mcp_tools_to_memory = lambda: list(MCP_SCHEMAS)
    if dynamic:
        agent.setup_dynamic_tools()

    yield agent
    Agent.mcp_enabled = original


class TestMCPDeferral:
    """MCP is the case this exists for: one server can expose dozens of tools."""

    def test_mcp_tools_go_to_the_catalog_not_the_request(
        self, mcp_agent
    ):
        mcp_agent.llm = mcp_agent.llm_handling()

        sent = [
            t["function"]["name"]
            for t in (mcp_agent.llm.tools_list_dictionary or [])
        ]
        assert sent == [SEARCH_TOOL_NAME]
        assert (
            "slack_post_message"
            in mcp_agent.tool_loader.deferred_names
        )
        assert "jira_search" in mcp_agent.tool_loader.deferred_names

    def test_a_deferred_mcp_tool_can_be_found_and_loaded(
        self, mcp_agent
    ):
        mcp_agent.llm = mcp_agent.llm_handling()
        mcp_agent._tool_search_tool(query="post a slack message")

        sent = [
            t["function"]["name"]
            for t in (mcp_agent.llm.tools_list_dictionary or [])
        ]
        assert "slack_post_message" in sent
        assert (
            "jira_search" not in sent
        ), "loaded more than was asked for"

    def test_the_fetch_happens_once_not_per_rebuild(self, mcp_agent):
        """
        Fetching is a network call, and every tool_search rebuilds the LLM, so
        an uncached fetch would hit the server on every search.
        """
        calls = {"n": 0}

        def counted():
            calls["n"] += 1
            return list(MCP_SCHEMAS)

        mcp_agent.add_mcp_tools_to_memory = counted
        for _ in range(3):
            mcp_agent.llm = mcp_agent.llm_handling()

        assert (
            calls["n"] == 1
        ), f"fetched {calls['n']} times, expected 1"

    def test_an_unreachable_server_does_not_break_setup(
        self, mcp_agent
    ):
        def explode():
            raise ConnectionError("server unreachable")

        mcp_agent.add_mcp_tools_to_memory = explode
        assert mcp_agent.defer_mcp_tools() == 0
        mcp_agent.llm = mcp_agent.llm_handling()  # must not raise

    @pytest.mark.parametrize("mcp_agent", [False], indirect=True)
    def test_without_dynamic_tools_mcp_is_still_eager(
        self, mcp_agent
    ):
        """The existing behaviour must be untouched when the flag is off."""
        mcp_agent.llm = mcp_agent.llm_handling()

        sent = [
            t["function"]["name"]
            for t in (mcp_agent.llm.tools_list_dictionary or [])
        ]
        assert "slack_post_message" in sent
        assert "jira_search" in sent


class FakeMCPManager:
    """Stands in for a real MCP server; records what it was asked to run."""

    def __init__(self, schemas):
        self.schemas = schemas
        self.executed = []

    def get_tools(self):
        return self.schemas

    def execute_tool_calls(self, response, output_type="dict"):
        self.executed.append(response)
        return [{"ok": True}]


@pytest.fixture
def mcp_auto_agent():
    """An autonomous agent with a stubbed MCP surface and dynamic tools."""
    original = Agent.mcp_enabled
    agent = build(max_loops="auto", tools=[], dynamic_tools=True)
    manager = FakeMCPManager(list(MCP_SCHEMAS))

    Agent.mcp_enabled = property(lambda self: True)
    agent.mcp_manager = manager
    agent.add_mcp_tools_to_memory = lambda: list(MCP_SCHEMAS)
    agent.setup_dynamic_tools()
    # Deferral normally happens inside llm_handling(); prime it here so the
    # catalog is populated before the run, as it would be in real use.
    agent.defer_mcp_tools()

    yield agent, manager
    Agent.mcp_enabled = original


def mcp_plan_call():
    """A one-step plan. Returned as a list: the loop only treats a response as
    tool calls when it is a list."""
    return [
        tool_call(
            "create_plan",
            task_description="post a message to slack",
            steps=[
                {
                    "step_id": "s1",
                    "description": "post a slack message",
                    "priority": "high",
                    "dependencies": [],
                }
            ],
        )
    ]


class TestMCPOnlyAgent:
    """An agent whose only tools come from MCP still needs a loader."""

    def test_a_loader_is_created_without_local_tools(self):
        """
        Regression: the loader was only built when local tools existed, so an
        MCP-only agent - the main reason to defer at all - never got one. MCP
        deferral silently did not happen, and a server failure propagated raw
        out of llm_handling() instead of being caught.
        """
        original = Agent.mcp_enabled
        try:
            agent = build(max_loops=1, dynamic_tools=True)
            Agent.mcp_enabled = property(lambda self: True)
            agent.add_mcp_tools_to_memory = lambda: list(MCP_SCHEMAS)
            agent.setup_dynamic_tools()
            agent.defer_mcp_tools()

            assert agent.tool_loader is not None
            assert (
                "slack_post_message"
                in agent.tool_loader.deferred_names
            )
        finally:
            Agent.mcp_enabled = original

    def test_an_unreachable_server_leaves_a_usable_agent(self):
        """The agent should run without those tools, not fail to build."""
        original = Agent.mcp_enabled
        try:
            agent = build(max_loops=1, dynamic_tools=True)
            Agent.mcp_enabled = property(lambda self: True)

            def explode():
                raise ConnectionError("server unreachable")

            agent.add_mcp_tools_to_memory = explode
            agent.setup_dynamic_tools()

            assert agent.defer_mcp_tools() == 0
            agent.llm = agent.llm_handling()  # must not raise
            assert exposed(agent) == [SEARCH_TOOL_NAME]
        finally:
            Agent.mcp_enabled = original


class TestMCPInAutonomousLoop:
    """MCP has to survive the loop's own setup, and be executable there."""

    def test_mcp_tools_survive_the_loops_loader_rebuild(
        self, mcp_auto_agent, monkeypatch
    ):
        """
        Regression: the loop calls setup_dynamic_tools() to install its control
        tools, which built a fresh loader and dropped every deferred MCP tool -
        and the once-per-agent fetch guard stopped them being re-registered.
        """
        agent, _ = mcp_auto_agent
        assert (
            "slack_post_message" in agent.tool_loader.deferred_names
        )

        script(
            agent,
            monkeypatch,
            [
                mcp_plan_call(),
                [
                    tool_call(
                        "complete_task",
                        task_id="m",
                        summary="d",
                        success=True,
                    )
                ],
            ],
        )
        agent.run("post to slack")

        known = (
            agent.tool_loader.loaded_names
            + agent.tool_loader.deferred_names
        )
        assert (
            "slack_post_message" in known
        ), "the loop's loader rebuild discarded the MCP tools"

    def test_an_mcp_tool_call_is_executed_not_rejected(
        self, mcp_auto_agent, monkeypatch
    ):
        """
        Regression: MCP calls fell through to tool_struct, which resolves
        against self.tools only, so every one raised ToolNotFoundError.
        """
        agent, manager = mcp_auto_agent
        script(
            agent,
            monkeypatch,
            [
                mcp_plan_call(),
                [
                    tool_call(
                        "slack_post_message",
                        call_id="mcp1",
                        text="hello",
                    )
                ],
                [
                    tool_call(
                        "subtask_done",
                        task_id="s1",
                        summary="d",
                        success=True,
                    )
                ],
                [
                    tool_call(
                        "complete_task",
                        task_id="m",
                        summary="d",
                        success=True,
                    )
                ],
            ],
        )
        agent.run("post to slack")

        assert manager.executed, "the MCP manager was never invoked"
        history = (
            agent.short_memory.return_history_as_string().lower()
        )
        assert "not found" not in history

    def test_the_mcp_result_is_paired_in_the_transcript(
        self, mcp_auto_agent, monkeypatch
    ):
        """A tool call without a matching result invalidates the next request."""
        agent, _ = mcp_auto_agent
        script(
            agent,
            monkeypatch,
            [
                mcp_plan_call(),
                [
                    tool_call(
                        "slack_post_message",
                        call_id="mcp1",
                        text="hello",
                    )
                ],
                [
                    tool_call(
                        "subtask_done",
                        task_id="s1",
                        summary="d",
                        success=True,
                    )
                ],
                [
                    tool_call(
                        "complete_task",
                        task_id="m",
                        summary="d",
                        success=True,
                    )
                ],
            ],
        )
        agent.run("post to slack")

        transcript = agent.autonomous_loop._transcript.messages
        assert any(
            m["role"] == "tool" and m["tool_call_id"] == "mcp1"
            for m in transcript
        ), "the MCP call has no matching tool result"

    def test_a_failing_mcp_call_becomes_a_tool_error(
        self, mcp_auto_agent, monkeypatch
    ):
        """The model must see the failure, not have it raised past it."""
        agent, manager = mcp_auto_agent

        def explode(response, output_type="dict"):
            raise ConnectionError("mcp server unreachable")

        manager.execute_tool_calls = explode
        script(
            agent,
            monkeypatch,
            [
                mcp_plan_call(),
                [
                    tool_call(
                        "slack_post_message",
                        call_id="mcp1",
                        text="hello",
                    )
                ],
                [
                    tool_call(
                        "subtask_done",
                        task_id="s1",
                        summary="d",
                        success=False,
                    )
                ],
                [
                    tool_call(
                        "complete_task",
                        task_id="m",
                        summary="d",
                        success=False,
                    )
                ],
            ],
        )
        agent.run("post to slack")

        results = [
            str(m["content"])
            for m in agent.autonomous_loop._transcript.messages
            if m["role"] == "tool"
        ]
        assert any("unreachable" in r for r in results)


if __name__ == "__main__":
    pytest.main([__file__, "-q", "-p", "no:randomly"])
