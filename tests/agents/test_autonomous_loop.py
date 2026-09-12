"""
Regression tests for the ``max_loops="auto"`` autonomous loop.

Approach
--------
Fully offline. Constructing a real ``swarms.Agent`` performs no network I/O
(the LLM is only contacted on ``.run()``), so every test below builds a real
agent and a real :class:`AutonomousAgentLoop`. The single seam that would talk
to a provider — ``Agent.call_llm`` — is monkeypatched per-test with a scripted
sequence of canned responses, so the loop's real control flow, real tool
dispatch, and real state transitions are exercised end to end.

Covers four fixed defects:

* #1963 — tool errors were logged and discarded, so the model never learned a
  call had failed and re-emitted it until the iteration budget was gone.
* #1964 — ``subtask_done`` broke out of the tool-call loop, silently dropping
  every call the model batched after it, while still marking the subtask done.
* #1966 — a *failed* dependency satisfied its dependents, and an unknown
  ``step_id`` defaulted to satisfied.
* #1962 — ``ContextCompressor.maybe_compress`` was never called on the
  ``max_loops="auto"`` path, so history grew unbounded in exactly the mode
  that needs compression most.

Run:
    cd /Users/swarms_wd/Desktop/research/swarms
    PYTHONPATH=. python3 -m pytest tests/agents/test_autonomous_loop.py -q -p no:randomly
"""

import json
import os
import subprocess

import pytest

from swarms import Agent
from swarms.agents.autonomous_loop import AutonomousAgentLoop
from swarms.structs.autonomous_loop_utils import (
    MAX_SUBTASK_LOOPS,
    get_autonomous_planning_tools,
    glob_tool,
    TOOL_OUTPUT_CONTEXT_SHARE,
    read_file_tool,
)
from swarms.utils.litellm_tokenizer import count_tokens


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def build_agent(**overrides):
    """Build a real, offline agent configured for the autonomous loop."""
    kwargs = dict(
        agent_name="AutoLoopTestAgent",
        model_name="gpt-4o-mini",
        max_loops="auto",
        persistent_memory=False,
        context_compression=False,
        print_on=False,
        verbose=False,
        autosave=False,
    )
    kwargs.update(overrides)
    return Agent(**kwargs)


def tool_call(name, **arguments):
    """Build a tool call in the shape ``parse_llm_output`` yields."""
    return {
        "type": "function",
        "function": {
            "name": name,
            "arguments": json.dumps(arguments),
        },
    }


def plan(*steps):
    """Build a ``create_plan`` response from ``(step_id, deps)`` pairs."""
    return [
        tool_call(
            "create_plan",
            task_description="test task",
            steps=[
                {
                    "step_id": step_id,
                    "description": f"do {step_id}",
                    "priority": "high",
                    "dependencies": list(deps),
                }
                for step_id, deps in steps
            ],
        )
    ]


def script_llm(agent, monkeypatch, responses):
    """
    Replace ``agent.call_llm`` with a scripted sequence.

    Each call pops the next canned response. Running past the end returns a
    plain string, which the loop treats as a no-op turn — that keeps a test
    from hanging if the loop iterates more than expected.
    """
    queue = list(responses)
    calls = []

    def fake_call_llm(task=None, *args, **kwargs):
        calls.append(task)
        if queue:
            return queue.pop(0)
        return "no further action"

    monkeypatch.setattr(agent, "call_llm", fake_call_llm)
    return calls


def history(agent):
    """Full conversation text, for asserting what the model would next see."""
    return agent.short_memory.return_history_as_string()


def status_of(agent, step_id):
    for subtask in agent.autonomous_subtasks:
        if subtask["step_id"] == step_id:
            return subtask["status"]
    raise AssertionError(f"no subtask {step_id!r}")


# --------------------------------------------------------------------------
# #1966 — dependency resolution
# --------------------------------------------------------------------------


class TestDependencyResolution:
    """A dependency only counts as satisfied when it actually completed."""

    def test_failed_dependency_does_not_unblock_dependent(self):
        agent = build_agent()
        loop = AutonomousAgentLoop(agent)
        loop._create_plan_tool(
            "t",
            [
                {"step_id": "a"},
                {"step_id": "b", "dependencies": ["a"]},
            ],
        )

        agent.subtask_status["a"] = "failed"
        agent.autonomous_subtasks[0]["status"] = "failed"

        assert loop._get_next_executable_subtask() is None
        assert status_of(agent, "b") == "skipped"

    def test_completed_dependency_unblocks_dependent(self):
        agent = build_agent()
        loop = AutonomousAgentLoop(agent)
        loop._create_plan_tool(
            "t",
            [
                {"step_id": "a"},
                {"step_id": "b", "dependencies": ["a"]},
            ],
        )

        agent.subtask_status["a"] = "completed"
        agent.autonomous_subtasks[0]["status"] = "completed"

        nxt = loop._get_next_executable_subtask()
        assert nxt is not None and nxt["step_id"] == "b"

    def test_skip_cascades_through_a_dependency_chain(self):
        agent = build_agent()
        loop = AutonomousAgentLoop(agent)
        loop._create_plan_tool(
            "t",
            [
                {"step_id": "a"},
                {"step_id": "b", "dependencies": ["a"]},
                {"step_id": "c", "dependencies": ["b"]},
            ],
        )

        agent.subtask_status["a"] = "failed"
        agent.autonomous_subtasks[0]["status"] = "failed"

        # First pass skips b; second pass sees b skipped and skips c.
        assert loop._get_next_executable_subtask() is None
        assert loop._get_next_executable_subtask() is None
        assert status_of(agent, "b") == "skipped"
        assert status_of(agent, "c") == "skipped"

    def test_unknown_dependency_is_dropped_at_plan_time(self):
        agent = build_agent()
        loop = AutonomousAgentLoop(agent)
        loop._create_plan_tool(
            "t",
            [{"step_id": "a", "dependencies": ["ghost_step", "a"]}],
        )

        # The hallucinated id and the self-reference are both removed, so the
        # subtask is runnable rather than permanently blocked.
        assert agent.autonomous_subtasks[0]["dependencies"] == []
        nxt = loop._get_next_executable_subtask()
        assert nxt is not None and nxt["step_id"] == "a"

    def test_real_dependencies_survive_validation(self):
        agent = build_agent()
        loop = AutonomousAgentLoop(agent)
        loop._create_plan_tool(
            "t",
            [
                {"step_id": "a"},
                {"step_id": "b", "dependencies": ["a", "nope"]},
            ],
        )

        assert agent.autonomous_subtasks[1]["dependencies"] == ["a"]

    def test_skipped_counts_as_terminal(self):
        agent = build_agent()
        loop = AutonomousAgentLoop(agent)
        loop._create_plan_tool("t", [{"step_id": "a"}])

        agent.autonomous_subtasks[0]["status"] = "skipped"
        assert loop._all_subtasks_complete() is True


# --------------------------------------------------------------------------
# #1964 — tool calls batched after subtask_done
# --------------------------------------------------------------------------


class TestToolOutputIsCapped:
    """One oversized read is re-sent on every later call of the run."""

    def test_read_file_truncates_a_large_file(self, tmp_path):
        agent = build_agent(context_length=16000)
        budget = int(agent.context_length * TOOL_OUTPUT_CONTEXT_SHARE)
        big = tmp_path / "big.txt"
        big.write_text("word " * (budget * 2))

        output = read_file_tool(agent, str(big))

        assert "output truncated" in output
        assert (
            count_tokens(output, model=agent.model_name)
            <= budget + 50
        )

    def test_budget_follows_the_agent_context_window(self, tmp_path):
        big = tmp_path / "big.txt"
        big.write_text("word " * 20000)

        small = read_file_tool(
            build_agent(context_length=16000), str(big)
        )
        large = read_file_tool(
            build_agent(context_length=128000), str(big)
        )

        assert len(large) > len(small)


class TestBatchedToolCalls:
    """Every tool call in a response runs, whatever its position."""

    def test_call_after_subtask_done_still_executes(
        self, monkeypatch, tmp_path
    ):
        agent = build_agent()
        target = tmp_path / "written.txt"

        script_llm(
            agent,
            monkeypatch,
            [
                plan(("step1", [])),
                # subtask_done deliberately precedes the real work, which is
                # exactly what the execution prompt asks the model to batch.
                [
                    tool_call(
                        "subtask_done",
                        task_id="step1",
                        summary="done",
                        success=True,
                    ),
                    tool_call(
                        "create_file",
                        file_path=str(target),
                        content="payload",
                    ),
                ],
                [
                    tool_call(
                        "complete_task",
                        task_id="main",
                        summary="all done",
                        success=True,
                    )
                ],
            ],
        )

        agent.run("test task")

        assert (
            target.exists()
        ), "file write batched after subtask_done was dropped"
        assert target.read_text() == "payload"
        assert status_of(agent, "step1") == "completed"

    def test_call_after_complete_task_still_executes(
        self, monkeypatch, tmp_path
    ):
        agent = build_agent()
        target = tmp_path / "late.txt"

        script_llm(
            agent,
            monkeypatch,
            [
                plan(("step1", [])),
                [
                    tool_call(
                        "complete_task",
                        task_id="main",
                        summary="finished",
                        success=True,
                    ),
                    tool_call(
                        "create_file",
                        file_path=str(target),
                        content="payload",
                    ),
                ],
            ],
        )

        agent.run("test task")

        assert (
            target.exists()
        ), "file write batched after complete_task was dropped"


# --------------------------------------------------------------------------
# #1963 — tool errors reach the model
# --------------------------------------------------------------------------


class TestToolErrorFeedback:
    """A failing tool call is reported back, not silently discarded."""

    def test_handler_exception_is_added_to_the_conversation(
        self, monkeypatch
    ):
        agent = build_agent()

        def exploding_think(*args, **kwargs):
            raise RuntimeError("handler blew up")

        script_llm(
            agent,
            monkeypatch,
            [
                plan(("step1", [])),
                [
                    tool_call(
                        "think",
                        current_state="s",
                        analysis="a",
                        next_actions=["x"],
                        confidence=0.9,
                    )
                ],
                [
                    tool_call(
                        "subtask_done",
                        task_id="step1",
                        summary="done",
                        success=True,
                    )
                ],
                [
                    tool_call(
                        "complete_task",
                        task_id="main",
                        summary="done",
                        success=True,
                    )
                ],
            ],
        )
        monkeypatch.setattr(
            AutonomousAgentLoop, "_think_tool", exploding_think
        )

        agent.run("test task")

        text = history(agent)
        assert "handler blew up" in text
        assert "RuntimeError" in text

    def test_malformed_arguments_do_not_abort_the_iteration(
        self, monkeypatch, tmp_path
    ):
        agent = build_agent()
        target = tmp_path / "after_bad_json.txt"

        bad = {
            "type": "function",
            "function": {"name": "think", "arguments": "{not json"},
        }

        script_llm(
            agent,
            monkeypatch,
            [
                plan(("step1", [])),
                [
                    bad,
                    tool_call(
                        "create_file",
                        file_path=str(target),
                        content="payload",
                    ),
                    tool_call(
                        "subtask_done",
                        task_id="step1",
                        summary="done",
                        success=True,
                    ),
                ],
                [
                    tool_call(
                        "complete_task",
                        task_id="main",
                        summary="done",
                        success=True,
                    )
                ],
            ],
        )

        agent.run("test task")

        assert (
            target.exists()
        ), "a malformed call aborted the whole response"
        assert "ERROR: think failed" in history(agent)

    def test_failed_subtask_done_does_not_complete_the_subtask(
        self, monkeypatch
    ):
        agent = build_agent()

        def exploding_done(*args, **kwargs):
            raise RuntimeError("could not record completion")

        script_llm(
            agent,
            monkeypatch,
            [
                plan(("step1", [])),
                [
                    tool_call(
                        "subtask_done",
                        task_id="step1",
                        summary="done",
                        success=True,
                    )
                ],
            ],
        )
        monkeypatch.setattr(
            AutonomousAgentLoop, "_subtask_done_tool", exploding_done
        )

        agent.run("test task")

        # The handler never recorded anything, so the subtask must not be
        # reported as completed on the strength of the call alone.
        assert status_of(agent, "step1") != "completed"


# --------------------------------------------------------------------------
# #1977 - structured transcript
# --------------------------------------------------------------------------


class TestStructuredTranscript:
    """The loop sends a real conversation, not a flattened string."""

    def _run_simple(self, monkeypatch, tmp_path):
        agent = build_agent()
        target = tmp_path / "t.txt"
        script_llm(
            agent,
            monkeypatch,
            [
                plan(("step1", [])),
                [
                    tool_call(
                        "create_file",
                        file_path=str(target),
                        content="x",
                    )
                ],
                [
                    tool_call(
                        "subtask_done",
                        task_id="step1",
                        summary="d",
                        success=True,
                    )
                ],
                [
                    tool_call(
                        "complete_task",
                        task_id="main",
                        summary="d",
                        success=True,
                    )
                ],
            ],
        )
        agent.run("demo")
        return agent

    def test_transcript_uses_real_roles(self, monkeypatch, tmp_path):
        agent = self._run_simple(monkeypatch, tmp_path)
        roles = [
            m["role"]
            for m in agent.autonomous_loop._transcript.messages
        ]

        assert "assistant" in roles, "no assistant turn was recorded"
        assert (
            "tool" in roles
        ), "tool results were not recorded as tool messages"
        assert roles[0] == "user"

    def test_assistant_turns_carry_tool_calls(
        self, monkeypatch, tmp_path
    ):
        agent = self._run_simple(monkeypatch, tmp_path)
        assistant = [
            m
            for m in agent.autonomous_loop._transcript.messages
            if m["role"] == "assistant"
        ]
        assert assistant, "no assistant turns"
        assert all("tool_calls" in m for m in assistant)
        names = [
            c["function"]["name"]
            for m in assistant
            for c in m["tool_calls"]
        ]
        assert "create_file" in names

    def test_every_tool_call_has_a_matching_result(
        self, monkeypatch, tmp_path
    ):
        """
        The chat-completions contract: an assistant message with tool_calls
        must be followed by one tool message per tool_call_id. A gap makes the
        next request fail outright, so this is the invariant that matters most.
        """
        agent = self._run_simple(monkeypatch, tmp_path)
        transcript = agent.autonomous_loop._transcript.messages

        for index, message in enumerate(transcript):
            if message["role"] != "assistant" or not message.get(
                "tool_calls"
            ):
                continue
            expected = [c["id"] for c in message["tool_calls"]]
            following = [
                transcript[j]["tool_call_id"]
                for j in range(
                    index + 1,
                    min(index + 1 + len(expected), len(transcript)),
                )
                if transcript[j]["role"] == "tool"
            ]
            assert following == expected, (
                f"assistant turn {index} has unanswered tool calls: "
                f"expected {expected}, got {following}"
            )

    def test_llm_receives_messages_not_a_flattened_string(
        self, monkeypatch, tmp_path
    ):
        agent = build_agent()
        seen = {}

        def capture(task=None, *args, **kwargs):
            seen.setdefault("messages", kwargs.get("messages"))
            seen.setdefault("task", task)
            return [
                tool_call(
                    "complete_task",
                    task_id="m",
                    summary="d",
                    success=True,
                )
            ]

        monkeypatch.setattr(agent, "call_llm", capture)
        agent.run("demo")

        assert (
            seen["task"] is None
        ), "the loop still sends a task string"
        assert isinstance(seen["messages"], list)
        assert seen["messages"][0]["role"] == "user"


# --------------------------------------------------------------------------
# #1978 - the plan is mutable
# --------------------------------------------------------------------------


class TestMutablePlan:
    """Re-planning merges rather than wiping."""

    def test_revision_preserves_completed_work(self):
        agent = build_agent()
        loop = AutonomousAgentLoop(agent)
        loop._create_plan_tool(
            "t", [{"step_id": "a"}, {"step_id": "b"}]
        )

        agent.autonomous_subtasks[0]["status"] = "completed"
        agent.autonomous_subtasks[0]["summary"] = "did a"
        agent.subtask_status["a"] = "completed"

        loop._create_plan_tool(
            "t",
            [{"step_id": "a"}, {"step_id": "b"}, {"step_id": "c"}],
        )

        assert status_of(agent, "a") == "completed"
        assert agent.autonomous_subtasks[0]["summary"] == "did a"
        assert status_of(agent, "c") == "pending"

    def test_revision_can_add_discovered_work(self):
        agent = build_agent()
        loop = AutonomousAgentLoop(agent)
        loop._create_plan_tool("t", [{"step_id": "a"}])
        loop._create_plan_tool(
            "t", [{"step_id": "a"}, {"step_id": "new"}]
        )

        ids = [s["step_id"] for s in agent.autonomous_subtasks]
        assert ids == ["a", "new"]

    def test_revision_drops_pending_steps_left_out(self):
        agent = build_agent()
        loop = AutonomousAgentLoop(agent)
        loop._create_plan_tool(
            "t", [{"step_id": "a"}, {"step_id": "obsolete"}]
        )
        loop._create_plan_tool("t", [{"step_id": "a"}])

        ids = [s["step_id"] for s in agent.autonomous_subtasks]
        assert ids == ["a"]
        assert "obsolete" not in agent.subtask_status

    def test_finished_steps_left_out_are_kept_as_history(self):
        agent = build_agent()
        loop = AutonomousAgentLoop(agent)
        loop._create_plan_tool(
            "t", [{"step_id": "done"}, {"step_id": "b"}]
        )
        agent.autonomous_subtasks[0]["status"] = "completed"
        agent.subtask_status["done"] = "completed"

        loop._create_plan_tool("t", [{"step_id": "b"}])

        ids = [s["step_id"] for s in agent.autonomous_subtasks]
        assert "done" in ids, "finished work was erased by a revision"
        assert status_of(agent, "done") == "completed"

    def test_revision_can_update_a_pending_step(self):
        agent = build_agent()
        loop = AutonomousAgentLoop(agent)
        loop._create_plan_tool(
            "t", [{"step_id": "a", "description": "old"}]
        )
        loop._create_plan_tool(
            "t",
            [
                {
                    "step_id": "a",
                    "description": "new",
                    "priority": "low",
                }
            ],
        )

        subtask = agent.autonomous_subtasks[0]
        assert subtask["description"] == "new"
        assert subtask["priority"] == "low"

    def test_revision_reports_a_diff_not_the_whole_plan(self):
        agent = build_agent()
        loop = AutonomousAgentLoop(agent)
        loop._create_plan_tool("t", [{"step_id": "a"}])
        result = loop._create_plan_tool(
            "t", [{"step_id": "a"}, {"step_id": "b"}]
        )

        assert "added" in result and "'b'" in result
        assert "Plan updated" in result

    def test_first_call_still_reads_as_creation(self):
        agent = build_agent()
        loop = AutonomousAgentLoop(agent)
        result = loop._create_plan_tool("t", [{"step_id": "a"}])
        assert "created successfully" in result

    def test_revision_may_depend_on_finished_unmentioned_steps(self):
        agent = build_agent()
        loop = AutonomousAgentLoop(agent)
        loop._create_plan_tool("t", [{"step_id": "a"}])
        agent.autonomous_subtasks[0]["status"] = "completed"
        agent.subtask_status["a"] = "completed"

        loop._create_plan_tool(
            "t", [{"step_id": "b", "dependencies": ["a"]}]
        )
        b = [
            s
            for s in agent.autonomous_subtasks
            if s["step_id"] == "b"
        ][0]
        assert b["dependencies"] == [
            "a"
        ], "a finished dependency was dropped"


# --------------------------------------------------------------------------
# think_tool parameter
# --------------------------------------------------------------------------


class TestThinkToolParameter:
    """`think` is opt-in, and the prompt agrees with the tool list."""

    def _tools_offered(self, agent, monkeypatch):
        """Run the loop far enough to see the tool list it builds."""
        offered = {}

        def capture(task=None, *args, **kwargs):
            offered.setdefault(
                "names",
                [
                    t["function"]["name"]
                    for t in (agent.tools_list_dictionary or [])
                    if isinstance(t, dict) and "function" in t
                ],
            )
            return [
                tool_call(
                    "complete_task",
                    task_id="m",
                    summary="d",
                    success=True,
                )
            ]

        monkeypatch.setattr(agent, "call_llm", capture)
        agent.run("demo")
        return offered.get("names", [])

    def test_think_is_absent_by_default(self, monkeypatch):
        agent = build_agent()
        assert agent.think_tool is False
        assert "think" not in self._tools_offered(agent, monkeypatch)

    def test_think_is_offered_when_enabled(self, monkeypatch):
        agent = build_agent(think_tool=True)
        assert "think" in self._tools_offered(agent, monkeypatch)

    def test_prompt_matches_the_tool_list(self):
        """The model must not be told to call a tool it was not given."""
        without = build_agent()
        with_think = build_agent(think_tool=True)

        assert (
            "THE THINK TOOL IS NOT AVAILABLE" in without.system_prompt
        )
        assert (
            "THE THINK TOOL IS NOT AVAILABLE"
            not in with_think.system_prompt
        )

    def test_default_thinking_tokens_no_longer_decides(self):
        """
        Regression guard. The old filter tested `thinking_tokens is not None`,
        but `thinking_tokens` defaults to 1024, so that was always true and
        stripped `think` for every agent regardless of intent. The explicit
        parameter now decides.
        """
        agent = build_agent(think_tool=True)
        assert (
            agent.thinking_tokens is not None
        ), "precondition: thinking_tokens has a non-None default"
        # think_tool wins over the non-None default.
        assert agent.think_tool is True


# --------------------------------------------------------------------------
# #1965 - the consecutive-think guard, and containment of a stuck subtask
# --------------------------------------------------------------------------


class TestThinkLoopContainment:
    """A model stuck analysing must be interrupted, then contained."""

    def _thrashing_agent(self, monkeypatch):
        """An agent whose model only ever calls `think`, never acting."""
        agent = build_agent(think_tool=True)
        turn = {"n": 0}

        def scripted(task=None, *args, **kwargs):
            turn["n"] += 1
            if turn["n"] == 1:
                return plan(("s1", []))
            return [
                tool_call(
                    "think",
                    current_state=f"state {turn['n']}",
                    analysis="pondering",
                    next_actions=["keep thinking"],
                    confidence=0.5,
                )
            ]

        monkeypatch.setattr(agent, "call_llm", scripted)
        return agent, turn

    def test_the_guard_actually_fires(self, monkeypatch):
        """
        Regression for #1965. think_call_count used to reset at the top of
        every iteration, so one think per turn never reached the limit and the
        guard was dead code.
        """
        agent, _ = self._thrashing_agent(monkeypatch)
        agent.run("demo")

        nudges = [
            m
            for m in agent.autonomous_loop._transcript.messages
            if "times in a row" in str(m.get("content"))
        ]
        assert (
            nudges
        ), "the guard never fired despite endless think calls"

    def test_the_nudge_reaches_the_model(self, monkeypatch):
        """A nudge only in short_memory is invisible on the next request."""
        agent, _ = self._thrashing_agent(monkeypatch)
        agent.run("demo")

        transcript = agent.autonomous_loop._transcript.messages
        assert any(
            "Take concrete action now" in str(m.get("content"))
            for m in transcript
        ), "the intervention never entered the transcript sent to the model"

    def test_a_stuck_subtask_is_contained(self, monkeypatch):
        """
        A subtask that exhausts its budget must not stay `pending`: pending
        keeps it eligible, so the outer loop re-selects it and re-runs the same
        doomed budget up to MAX_SUBTASK_ITERATIONS times (100 x 20 = 2000
        LLM calls for one stuck subtask).
        """
        agent, turn = self._thrashing_agent(monkeypatch)
        agent.run("demo")

        assert status_of(agent, "s1") == "failed"
        assert turn["n"] < MAX_SUBTASK_LOOPS + 5, (
            f"a stuck subtask consumed {turn['n']} LLM turns; it should be "
            f"bounded by one budget of {MAX_SUBTASK_LOOPS}"
        )

    def test_a_real_action_resets_the_streak(self):
        """`consecutive` must mean consecutive."""
        agent = build_agent(think_tool=True)
        agent.think_call_count = 1
        loop = AutonomousAgentLoop(agent)
        loop._create_plan_tool("t", [{"step_id": "a"}])
        # _subtask_done_tool is one of the non-think handlers that clears it.
        loop._subtask_done_tool("a", "done", True)
        assert agent.think_call_count == 0


class TestHandoffPromptIsNotReappended:
    """#1968 — the handoff prompt was appended to ``system_prompt`` inside the
    per-run setup, so every ``run()`` stacked another copy and a long-lived
    agent's prompt grew without bound. Measured before the fix: +1,988
    characters per run, linearly.

    The loop is driven directly rather than through ``run()``: the append
    happens during setup, before the model is contacted, so stopping at the
    first provider call exercises the real code path offline.
    """

    class _Stop(Exception):
        """Halts the loop immediately after setup."""

    def _run_setup_only(self, agent, monkeypatch):
        """Run the loop far enough to apply the handoff prompt, then stop."""

        def boom(*args, **kwargs):
            raise TestHandoffPromptIsNotReappended._Stop()

        monkeypatch.setattr(type(agent), "llm_handling", boom)
        try:
            agent._run_autonomous_loop(task="anything")
        except Exception:
            pass

    def test_prompt_does_not_grow_across_runs(self, monkeypatch):
        worker = build_agent(agent_name="Worker", max_loops=1)
        agent = build_agent(agent_name="Boss", handoffs=[worker])

        lengths = []
        for _ in range(3):
            self._run_setup_only(agent, monkeypatch)
            lengths.append(len(agent.system_prompt))

        # The regression: these were 17347, 19335, 21323.
        assert lengths[0] == lengths[1] == lengths[2]

    def test_the_handoff_prompt_is_still_present(self, monkeypatch):
        """Not growing must not mean never applied — the delegation
        instructions have to survive, or handoffs stop working entirely.
        """
        worker = build_agent(agent_name="Worker", max_loops=1)
        agent = build_agent(agent_name="Boss", handoffs=[worker])

        self._run_setup_only(agent, monkeypatch)

        assert "Worker" in agent.system_prompt

    def test_a_changed_registry_refreshes_the_prompt(
        self, monkeypatch
    ):
        """A plain "already present" guard would pin the first registry's text
        forever, so a handoff target added later would never be described.
        """
        alpha = build_agent(agent_name="Alpha", max_loops=1)
        beta = build_agent(agent_name="Beta", max_loops=1)
        agent = build_agent(agent_name="Boss", handoffs=[alpha])

        self._run_setup_only(agent, monkeypatch)
        assert "Alpha" in agent.system_prompt
        assert "Beta" not in agent.system_prompt

        agent.handoffs = [alpha, beta]
        self._run_setup_only(agent, monkeypatch)
        assert "Alpha" in agent.system_prompt
        assert "Beta" in agent.system_prompt

    def test_the_refreshed_prompt_is_still_stable(self, monkeypatch):
        """After the registry changes, further runs must not resume growing."""
        alpha = build_agent(agent_name="Alpha", max_loops=1)
        beta = build_agent(agent_name="Beta", max_loops=1)
        agent = build_agent(agent_name="Boss", handoffs=[alpha])

        self._run_setup_only(agent, monkeypatch)
        agent.handoffs = [alpha, beta]
        self._run_setup_only(agent, monkeypatch)
        after_change = len(agent.system_prompt)

        self._run_setup_only(agent, monkeypatch)
        self._run_setup_only(agent, monkeypatch)

        assert len(agent.system_prompt) == after_change

    def test_an_agent_without_handoffs_is_untouched(
        self, monkeypatch
    ):
        agent = build_agent(agent_name="Solo")
        before = agent.system_prompt

        self._run_setup_only(agent, monkeypatch)

        assert agent.system_prompt == before


# --------------------------------------------------------------------------
# #1962 — context compression in the auto loop
# --------------------------------------------------------------------------


class TestContextCompression:
    """#1962 — ``maybe_compress`` was never called on the auto path.

    The compressor measures and compacts ``short_memory``, but the body
    sent to the model is the loop's ``Transcript`` — so the fix both
    triggers the compressor between subtask iterations and rebuilds the
    transcript from the summary it returns. Compressing only the mirror
    would leave the request payload growing unbounded.
    """

    class _StubCompressor:
        """Always compresses, returning a fixed summary."""

        def __init__(self, summary):
            self.summary = summary
            self.calls = 0

        def maybe_compress(self, agent):
            self.calls += 1
            return self.summary

    def test_no_compressor_is_a_no_op(self):
        agent = build_agent()  # context_compression=False
        loop = AutonomousAgentLoop(agent)
        loop._say_user("seed")

        assert loop._maybe_compress_context() is False
        assert len(loop._transcript) == 1

    def test_below_threshold_leaves_the_transcript_alone(self):
        agent = build_agent(
            context_compression=True, context_length=1_000_000
        )
        loop = AutonomousAgentLoop(agent)
        loop._say_user("seed")

        assert loop._maybe_compress_context() is False
        assert len(loop._transcript) == 1

    def test_compression_rebuilds_the_transcript(self):
        agent = build_agent()
        stub = self._StubCompressor("CANNED SUMMARY")
        agent._context_compressor = stub
        loop = AutonomousAgentLoop(agent)
        for i in range(6):
            loop._say_user(f"turn {i}")
        memory_before = history(agent)

        assert loop._maybe_compress_context() is True
        assert stub.calls == 1

        # The request body is now a single user turn holding the
        # summary, not the six accumulated turns.
        assert len(loop._transcript) == 1
        seeded = loop._transcript[0]
        assert seeded["role"] == "user"
        assert "[Compressed Memory Summary]" in seeded["content"]
        assert "CANNED SUMMARY" in seeded["content"]

        # The compressor owns the short_memory compaction; the
        # transcript re-seed must not mirror the summary there a
        # second time.
        assert history(agent) == memory_before

    def test_compression_fires_during_a_run(self, monkeypatch):
        """End to end: a run over a tiny context budget compresses
        between subtask iterations, and the next request body is the
        summary plus the re-issued subtask prompt — not the
        accumulated planning transcript.
        """
        agent = build_agent(
            context_compression=True, context_length=120
        )
        monkeypatch.setattr(
            agent._context_compressor,
            "_summarize",
            lambda agent, history: "CANNED SUMMARY",
        )

        bodies = []
        queue = [
            plan(("step1", [])),
            [
                tool_call(
                    "subtask_done",
                    task_id="step1",
                    summary="done",
                    success=True,
                )
            ],
            [
                tool_call(
                    "complete_task",
                    task_id="main",
                    summary="all done",
                    success=True,
                )
            ],
        ]

        def fake_call_llm(task=None, *args, **kwargs):
            bodies.append(list(kwargs.get("messages") or []))
            if queue:
                return queue.pop(0)
            return "no further action"

        monkeypatch.setattr(agent, "call_llm", fake_call_llm)

        agent.run("test task")

        # bodies[0] is the planning request; bodies[1] is the first
        # subtask iteration, which crossed the 90% threshold and was
        # compressed down to [summary, execution prompt].
        assert len(bodies) >= 2
        first_exec = bodies[1]
        assert len(first_exec) == 2
        assert (
            "[Compressed Memory Summary]" in first_exec[0]["content"]
        )
        assert "CANNED SUMMARY" in first_exec[0]["content"]
        assert "step1" in first_exec[1]["content"]

    def test_compression_failure_does_not_break_the_loop(
        self, monkeypatch
    ):
        """maybe_compress swallows summarizer errors and returns None;
        the loop must treat that as "no compression" and carry on.
        """
        agent = build_agent(
            context_compression=True, context_length=10
        )

        def boom(agent, history):
            raise RuntimeError("summarizer unavailable")

        monkeypatch.setattr(
            agent._context_compressor, "_summarize", boom
        )

        loop = AutonomousAgentLoop(agent)
        loop._say_user(
            "a turn long enough to cross a 10-token budget"
        )

        assert loop._maybe_compress_context() is False
        assert len(loop._transcript) == 1


# --------------------------------------------------------------------------
# #1983 — glob tool
# --------------------------------------------------------------------------


class TestGlobTool:
    """
    Without glob the model shells out to `find`, which the run_bash
    blocklist may refuse, or walks the tree one list_directory at a time.
    """

    def _tree(self, tmp_path):
        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "deep").mkdir()
        first = tmp_path / "pkg" / "old.py"
        second = tmp_path / "pkg" / "deep" / "new.py"
        other = tmp_path / "pkg" / "notes.md"
        for path in (first, second, other):
            path.write_text("x")
        os.utime(first, (1_000, 1_000))
        os.utime(second, (2_000, 2_000))
        os.utime(other, (3_000, 3_000))
        return first, second, other

    def test_matches_recursively_and_reports_relative_paths(
        self, tmp_path
    ):
        self._tree(tmp_path)

        output = glob_tool(build_agent(), "*.py", str(tmp_path))

        assert output.splitlines() == [
            "pkg/deep/new.py",
            "pkg/old.py",
        ]

    def test_newest_first(self, tmp_path):
        self._tree(tmp_path)

        output = glob_tool(build_agent(), "*", str(tmp_path))
        listed = [
            line
            for line in output.splitlines()
            if line.endswith(".md") or line.endswith(".py")
        ]

        assert listed[0] == "pkg/notes.md"

    def test_no_matches_says_so_rather_than_returning_empty(
        self, tmp_path
    ):
        self._tree(tmp_path)

        output = glob_tool(build_agent(), "*.rs", str(tmp_path))

        assert "no files" in output.lower()

    def test_a_missing_root_is_an_error(self, tmp_path):
        output = glob_tool(
            build_agent(), "*.py", str(tmp_path / "nope")
        )

        assert "Error" in output

    def test_output_is_capped_by_the_token_budget(self, tmp_path):
        agent = build_agent(context_length=16000)
        budget = int(agent.context_length * TOOL_OUTPUT_CONTEXT_SHARE)
        for i in range(budget):
            (
                tmp_path / f"file_{i:06d}_padding_padding.py"
            ).write_text("x")

        output = glob_tool(agent, "*.py", str(tmp_path))

        assert "output truncated" in output

    def test_the_schema_and_the_loop_both_know_about_glob(self):
        names = {
            tool["function"]["name"]
            for tool in get_autonomous_planning_tools()
        }
        assert "glob" in names

        glob_schema = next(
            tool["function"]
            for tool in get_autonomous_planning_tools()
            if tool["function"]["name"] == "glob"
        )
        assert glob_schema["parameters"]["required"] == ["pattern"]
        assert set(glob_schema["parameters"]["properties"]) == {
            "pattern",
            "path",
        }


# #1981 — read_file line numbers and windowing
# --------------------------------------------------------------------------


class TestReadFileWindowing:
    """
    A model that cannot ask for part of a file reads all of it, and a model
    that gets no line numbers cannot say where it looked.
    """

    def _file(self, tmp_path, lines):
        target = tmp_path / "sample.py"
        target.write_text("\n".join(lines) + "\n")
        return str(target)

    def test_lines_are_numbered_from_one(self, tmp_path):
        path = self._file(tmp_path, ["alpha", "beta", "gamma"])

        output = read_file_tool(build_agent(), path)

        assert output.splitlines() == [
            "     1\talpha",
            "     2\tbeta",
            "     3\tgamma",
        ]

    def test_offset_skips_lines_and_keeps_absolute_numbering(
        self, tmp_path
    ):
        path = self._file(
            tmp_path, [f"line {i}" for i in range(1, 11)]
        )

        output = read_file_tool(build_agent(), path, offset=6)

        assert output.splitlines()[0] == "     7\tline 7"
        assert output.splitlines()[-1] == "    10\tline 10"

    def test_limit_bounds_the_window(self, tmp_path):
        path = self._file(
            tmp_path, [f"line {i}" for i in range(1, 11)]
        )

        output = read_file_tool(
            build_agent(), path, offset=2, limit=3
        )

        assert [
            line.split("\t")[1] for line in output.splitlines()
        ] == [
            "line 3",
            "line 4",
            "line 5",
        ]

    def test_an_offset_past_the_end_says_how_long_the_file_is(
        self, tmp_path
    ):
        path = self._file(tmp_path, ["only", "two"])

        output = read_file_tool(build_agent(), path, offset=50)

        assert "2 lines" in output
        assert "Error" in output

    def test_a_negative_offset_is_rejected(self, tmp_path):
        path = self._file(tmp_path, ["a"])

        assert "Error" in read_file_tool(
            build_agent(), path, offset=-1
        )

    def test_a_non_positive_limit_is_rejected(self, tmp_path):
        path = self._file(tmp_path, ["a"])

        assert "Error" in read_file_tool(build_agent(), path, limit=0)

    def test_a_window_is_still_capped_by_the_token_budget(
        self, tmp_path
    ):
        agent = build_agent(context_length=16000)
        budget = int(agent.context_length * TOOL_OUTPUT_CONTEXT_SHARE)
        path = self._file(
            tmp_path, ["word " * 50 for _ in range(budget)]
        )

        output = read_file_tool(agent, path, offset=0, limit=budget)

        assert "output truncated" in output

    def test_a_form_feed_does_not_shift_the_numbering(self, tmp_path):
        """str.splitlines() breaks on \\f; grep -n does not (review on #2178)."""
        target = tmp_path / "pagebreak.py"
        target.write_text(
            "def a():\n    pass\n\x0c\ndef b():\n    return 1\n"
        )

        output = read_file_tool(build_agent(), str(target))
        numbered = {
            line.split("\t", 1)[1]: int(line.split("\t", 1)[0])
            for line in output.splitlines()
            if "\t" in line
        }

        grep = subprocess.run(
            ["grep", "-n", "def b", str(target)],
            capture_output=True,
            text=True,
        )
        grep_line = int(grep.stdout.split(":", 1)[0])

        assert numbered["def b():"] == grep_line == 4

    def test_an_empty_file_says_so(self, tmp_path):
        target = tmp_path / "empty.txt"
        target.write_text("")

        assert (
            read_file_tool(build_agent(), str(target))
            == "(empty file)"
        )

    def test_a_trailing_newline_is_not_an_extra_line(self, tmp_path):
        path = self._file(tmp_path, ["alpha", "beta"])

        output = read_file_tool(build_agent(), path)

        assert len(output.splitlines()) == 2

    def test_the_schema_advertises_the_window(self):
        read_file = next(
            tool["function"]
            for tool in get_autonomous_planning_tools()
            if tool["function"]["name"] == "read_file"
        )

        assert set(read_file["parameters"]["properties"]) == {
            "file_path",
            "offset",
            "limit",
        }
        assert read_file["parameters"]["required"] == ["file_path"]


if __name__ == "__main__":
    pytest.main([__file__, "-q", "-p", "no:randomly"])
